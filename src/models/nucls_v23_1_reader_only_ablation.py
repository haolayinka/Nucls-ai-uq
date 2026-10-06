#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nucls_v23_1_joint_bayes_logit as v231

EXPECTED_READERS = ["SP.3", "JP.1", "JP.2", "JP.5", "JP.6"]
EXPECTED_N_ROWS = 10833
EXPECTED_N_ANCHORS = 1144
EXPECTED_N_FOV = 52
EXPECTED_N_SLIDES = 5
EXPECTED_AI_POSITIVE = 593

SCALAR_VARS = [
    "alpha","beta_E","sd_reader","sd_reader_condition",
    "sd_fov","sd_reader_fov","sd_anchor","sd_reader_anchor",
]
ESTIMAND_COLS = [
    "theta_U_selected","theta_E_selected","delta_E_minus_U_selected"
]

def build_reader_only_model(data):
    import pymc as pm
    coords = {
        "reader": data.reader_names,
        "fov": np.arange(len(data.fov_values)),
        "anchor": np.arange(len(data.anchor_ids)),
        "rf_obs": np.arange(len(data.rf_keys)),
        "ra_obs": np.arange(len(data.ra_keys)),
        "obs": np.arange(len(data.y)),
    }
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", 0, 2.5)
        beta_E = pm.Normal("beta_E", 0, 1.0)

        sd_reader = pm.HalfNormal("sd_reader", 1.0)
        sd_reader_condition = pm.HalfNormal("sd_reader_condition", 1.0)
        sd_fov = pm.HalfNormal("sd_fov", 1.0)
        sd_reader_fov = pm.HalfNormal("sd_reader_fov", 1.0)
        sd_anchor = pm.HalfNormal("sd_anchor", 1.5)
        sd_reader_anchor = pm.HalfNormal("sd_reader_anchor", 1.5)

        z_reader = pm.Normal("z_reader", 0, 1, dims="reader")
        z_reader_condition = pm.Normal("z_reader_condition", 0, 1, dims="reader")
        z_fov = pm.Normal("z_fov", 0, 1, dims="fov")
        z_reader_fov = pm.Normal("z_reader_fov", 0, 1, dims="rf_obs")
        z_anchor = pm.Normal("z_anchor", 0, 1, dims="anchor")
        z_reader_anchor = pm.Normal("z_reader_anchor", 0, 1, dims="ra_obs")

        u = pm.Deterministic("u_reader", sd_reader*z_reader, dims="reader")
        b = pm.Deterministic("b_reader_condition", sd_reader_condition*z_reader_condition, dims="reader")
        g = pm.Deterministic("g_fov", sd_fov*z_fov, dims="fov")
        h = pm.Deterministic("h_reader_fov", sd_reader_fov*z_reader_fov, dims="rf_obs")
        aa = pm.Deterministic("a_anchor", sd_anchor*z_anchor, dims="anchor")
        q = pm.Deterministic("q_reader_anchor", sd_reader_anchor*z_reader_anchor, dims="ra_obs")

        d_shared = g[data.anchor_f] + aa
        eta = alpha + beta_E*data.c + d_shared[data.a] + u[data.r] + b[data.r]*data.c + h[data.rf] + q[data.ra]
        pm.Bernoulli("reader_detected", logit_p=eta, observed=data.y, dims="obs")
    return model

def summarize_real(post):
    rows = []
    for c in ESTIMAND_COLS:
        x = post[c].to_numpy(float)
        lo, hi = np.quantile(x, [0.025, 0.975])
        rows.append({
            "target": c,
            "estimate": float(x.mean()),
            "posterior_sd": float(x.std(ddof=1)),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "interval_width": float(hi-lo),
            "prob_gt_zero": float(np.mean(x > 0)) if c == "delta_E_minus_U_selected" else np.nan,
        })
    return pd.DataFrame(rows)

def atomic_netcdf(idata, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    idata.to_netcdf(tmp)
    os.replace(tmp, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--tune", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=.99)
    ap.add_argument("--seed", type=int, default=231003)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    v231.self_test()
    if v231.EXPECTED_N_ANCHORS != EXPECTED_N_ANCHORS:
        raise RuntimeError(f"Unexpected EXPECTED_N_ANCHORS={v231.EXPECTED_N_ANCHORS}")
    if list(v231.PRIMARY_READERS) != EXPECTED_READERS:
        raise RuntimeError(f"Unexpected PRIMARY_READERS={list(v231.PRIMARY_READERS)}")

    root = args.project_root
    real_path = root/"outputs"/"scoring_v16_1"/"restricted_recorded_reader_detection_agreement_v16.csv"
    ai_path = root/"outputs"/"scoring_v16_1"/"restricted_anchor_matches_v16.csv"

    raw = pd.read_csv(real_path)
    ai_tbl = pd.read_csv(ai_path)
    d = raw.loc[raw.reader.isin(EXPECTED_READERS)].copy()

    checks = {
        "rows": (len(d), EXPECTED_N_ROWS),
        "anchors": (d.anchor_id.nunique(), EXPECTED_N_ANCHORS),
        "fovs": (d.image_id.nunique(), EXPECTED_N_FOV),
        "slides": (d.slide.nunique(), EXPECTED_N_SLIDES),
    }
    for name, (got, exp) in checks.items():
        if got != exp:
            raise RuntimeError(f"Expected {exp} {name}, got {got}")
    if set(d.condition.unique()) != {"Unbiased","Evaluation"}:
        raise RuntimeError(f"Unexpected conditions: {sorted(d.condition.unique())}")

    fov_values = sorted(d.image_id.astype(str).unique())
    fov_map = {name:i for i,name in enumerate(fov_values)}
    rl = pd.DataFrame({
        "anchor_id": d.anchor_id.astype(str),
        "fov_id": d.image_id.astype(str).map(fov_map).astype(float),
        "reader": d.reader.astype(str),
        "condition": d.condition.map({"Unbiased":"U","Evaluation":"E"}),
        "observed": np.ones(len(d), dtype=np.int8),
        "reader_detected": d.reader_detected.astype(np.int8),
    })
    if rl.duplicated(["anchor_id","reader","condition"]).any():
        raise RuntimeError("Duplicate anchor-reader-condition rows")

    a = ai_tbl[["anchor_id","ai_detected"]].copy()
    a["anchor_id"] = a.anchor_id.astype(str)
    if a.anchor_id.duplicated().any():
        chk = a.groupby("anchor_id").ai_detected.nunique(dropna=False)
        if (chk > 1).any():
            raise RuntimeError("Conflicting AI labels")
        a = a.drop_duplicates("anchor_id")
    if len(a) != EXPECTED_N_ANCHORS:
        raise RuntimeError(f"Expected {EXPECTED_N_ANCHORS} AI anchors, got {len(a)}")

    ai = pd.Series(a.ai_detected.astype(np.int8).to_numpy(), index=a.anchor_id)
    if int(ai.sum()) != EXPECTED_AI_POSITIVE:
        raise RuntimeError(f"Expected {EXPECTED_AI_POSITIVE} AI positives, got {int(ai.sum())}")

    cross = d[["anchor_id","ai_detected"]].copy()
    cross["anchor_id"] = cross.anchor_id.astype(str)
    if (cross.groupby("anchor_id").ai_detected.nunique(dropna=False) > 1).any():
        raise RuntimeError("Reader table has inconsistent AI labels")
    cross = cross.drop_duplicates("anchor_id").set_index("anchor_id").ai_detected.astype(np.int8).reindex(ai.index)
    if cross.isna().any() or not np.array_equal(cross.to_numpy(), ai.to_numpy()):
        raise RuntimeError("AI vectors disagree across frozen sources")

    enc = v231.encode_observed(
        rl, ai, readers=v231.PRIMARY_READERS,
        expected_ai_matches=EXPECTED_AI_POSITIVE
    )

    audit = {
        "status":"V23_1_READER_ONLY_ABLATION_INPUT_OK",
        "analysis":"primary_1144_reader_only_ablation",
        "ablation":{
            "removed":["alpha_ai","beta_ai","ai_detected_obs likelihood"],
            "unchanged":[
                "reader likelihood","reader priors","noncentered parameterization",
                "d_i = g_f + a_i","future-reader quadrature",
                "posterior estimand calculation","fixed AI vector in agreement calculation"
            ],
        },
        "n_rows":len(d),
        "n_anchors":d.anchor_id.nunique(),
        "n_fovs":d.image_id.nunique(),
        "n_slides":d.slide.nunique(),
        "ai_positive":int(ai.sum()),
        "encoded_rf":len(enc.rf_keys),
        "encoded_ra":len(enc.ra_keys),
    }
    print("V23_1_READER_ONLY_ABLATION_INPUT_AUDIT "+json.dumps(audit,sort_keys=True), flush=True)

    import pymc as pm, arviz as az
    model = build_reader_only_model(enc)
    t0 = time.time()
    with model:
        idata = pm.sample(
            draws=args.draws, tune=args.tune, chains=args.chains,
            cores=min(args.chains,max(1,int(os.environ.get("SLURM_CPUS_PER_TASK","1")))),
            random_seed=args.seed, target_accept=args.target_accept,
            init="jitter+adapt_diag", progressbar=False, return_inferencedata=True
        )
    elapsed = time.time() - t0

    post = v231.posterior_estimands(idata, enc)
    summ = summarize_real(post)
    div = int(np.asarray(idata.sample_stats["diverging"]).sum())
    ds = az.summary(idata, var_names=SCALAR_VARS, round_to=None)

    diagnostics = {
        "elapsed_sec":elapsed,
        "divergences":div,
        "max_rhat_scalar":float(ds.r_hat.max()),
        "min_ess_bulk_scalar":float(ds.ess_bulk.min()),
        "min_ess_tail_scalar":float(ds.ess_tail.min()),
    }

    gate = {
        **audit, **diagnostics,
        "sampler":{
            "draws":args.draws,"tune":args.tune,"chains":args.chains,
            "target_accept":args.target_accept,"seed":args.seed,
            "parameterization":"same_v23_1_noncentered_reader_channel",
        },
        "posterior_summary":summ.to_dict(orient="records"),
        "status":"V23_1_READER_ONLY_ABLATION_COMPLETE",
    }

    outdir = args.outdir or root/"outputs"/"v23_1_reader_only_ablation"
    outdir.mkdir(parents=True,exist_ok=True)
    post.to_csv(outdir/"v23_1_reader_only_ablation_estimand_draws.csv",index=False)
    summ.to_csv(outdir/"v23_1_reader_only_ablation_summary.csv",index=False)
    ds.to_csv(outdir/"v23_1_reader_only_ablation_mcmc_scalar_diagnostics.csv")
    (outdir/"v23_1_reader_only_ablation_gate.json").write_text(json.dumps(gate,indent=2))
    atomic_netcdf(idata, outdir/"v23_1_reader_only_ablation_idata.nc")

    print("V23_1_READER_ONLY_ABLATION_POSTERIOR_SUMMARY", flush=True)
    print(summ.to_string(index=False), flush=True)
    print("V23_1_READER_ONLY_ABLATION_MCMC_DIAGNOSTICS "+json.dumps(diagnostics,sort_keys=True), flush=True)
    print("V23_1_READER_ONLY_ABLATION_COMPLETE", flush=True)

if __name__ == "__main__":
    main()
