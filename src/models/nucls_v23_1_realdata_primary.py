#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nucls_v23_1_joint_bayes_logit import (
    PRIMARY_READERS,
    encode_observed,
    build_model,
    posterior_estimands,
    self_test,
)

EXPECTED_READERS = ["SP.3", "JP.1", "JP.2", "JP.5", "JP.6"]
EXPECTED_N_ROWS = 10833
EXPECTED_N_ANCHORS = 1144
EXPECTED_N_FOV = 52
EXPECTED_N_SLIDES = 5
EXPECTED_AI_POSITIVE = 593

SCALAR_VARS = [
    "alpha","beta_E","alpha_ai","beta_ai","sd_reader",
    "sd_reader_condition","sd_fov","sd_reader_fov",
    "sd_anchor","sd_reader_anchor",
]
ESTIMAND_COLS = [
    "theta_U_selected",
    "theta_E_selected",
    "delta_E_minus_U_selected",
]

def summarize_real(post: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ESTIMAND_COLS if c not in post.columns]
    if missing:
        raise RuntimeError(f"posterior_estimands missing required columns: {missing}; got {list(post.columns)}")
    rows = []
    for c in ESTIMAND_COLS:
        x = post[c].to_numpy(float)
        if not np.isfinite(x).all():
            raise RuntimeError(f"Non-finite posterior estimand draws in {c}")
        rows.append({
            "target": c,
            "estimate": float(x.mean()),
            "posterior_sd": float(x.std(ddof=1)),
            "ci_low": float(np.quantile(x, .025)),
            "ci_high": float(np.quantile(x, .975)),
            "interval_width": float(np.quantile(x, .975)-np.quantile(x, .025)),
            "prob_gt_zero": float(np.mean(x > 0)) if c == "delta_E_minus_U_selected" else np.nan,
        })
    return pd.DataFrame(rows)

def atomic_netcdf(idata, path: Path):
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
    ap.add_argument("--seed", type=int, default=231001)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    self_test()
    root = args.project_root
    real_path = root/"outputs"/"scoring_v16_1"/"restricted_recorded_reader_detection_agreement_v16.csv"
    ai_path = root/"outputs"/"scoring_v16_1"/"restricted_anchor_matches_v16.csv"
    for p in [real_path, ai_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    raw = pd.read_csv(real_path)
    ai_tbl = pd.read_csv(ai_path)

    req = {"anchor_id","image_id","slide","patient_id","condition","reader",
           "ai_detected","reader_label","reader_detected","detection_agreement"}
    miss = req-set(raw.columns)
    if miss:
        raise RuntimeError(f"Real-data table missing columns: {sorted(miss)}")
    if not {"anchor_id","ai_detected"}.issubset(ai_tbl.columns):
        raise RuntimeError("AI table missing anchor_id and/or ai_detected")

    d = raw.loc[raw.reader.isin(EXPECTED_READERS)].copy()

    checks = {
        "rows": (len(d), EXPECTED_N_ROWS),
        "anchors": (d.anchor_id.nunique(), EXPECTED_N_ANCHORS),
        "fovs": (d.image_id.nunique(), EXPECTED_N_FOV),
        "slides": (d.slide.nunique(), EXPECTED_N_SLIDES),
    }
    for name,(got,exp) in checks.items():
        if got != exp:
            raise RuntimeError(f"Expected {exp} {name}, got {got}")
    if set(d.reader.unique()) != set(EXPECTED_READERS):
        raise RuntimeError(f"Unexpected reader set: {sorted(d.reader.unique())}")
    if set(d.condition.unique()) != {"Unbiased","Evaluation"}:
        raise RuntimeError(f"Unexpected conditions: {sorted(d.condition.unique())}")
    if list(PRIMARY_READERS) != EXPECTED_READERS:
        raise RuntimeError(f"PRIMARY_READERS changed: {list(PRIMARY_READERS)}")

    # FIX1: validated encoder requires numeric fov_id.
    # Use a deterministic lexicographic mapping from the 52 real image_id strings.
    fov_values = sorted(d.image_id.astype(str).unique())
    fov_map = {name:i for i,name in enumerate(fov_values)}
    if len(fov_map) != EXPECTED_N_FOV or set(fov_map.values()) != set(range(EXPECTED_N_FOV)):
        raise RuntimeError("FOV mapping construction failed")

    cond_map = {"Unbiased":"U","Evaluation":"E"}
    rl = pd.DataFrame({
        "anchor_id": d.anchor_id.astype(str),
        "fov_id": d.image_id.astype(str).map(fov_map).astype(float),
        "reader": d.reader.astype(str),
        "condition": d.condition.map(cond_map),
        "observed": np.ones(len(d), dtype=np.int8),
        "reader_detected": d.reader_detected.astype(np.int8),
    })

    ndups = int(rl.duplicated(["anchor_id","reader","condition"]).sum())
    if ndups:
        raise RuntimeError(f"Found {ndups} duplicate anchor-reader-condition rows")

    a = ai_tbl[["anchor_id","ai_detected"]].copy()
    a["anchor_id"] = a.anchor_id.astype(str)
    if a.anchor_id.duplicated().any():
        chk = a.groupby("anchor_id").ai_detected.nunique(dropna=False)
        if (chk > 1).any():
            raise RuntimeError("AI source has conflicting ai_detected values")
        a = a.drop_duplicates("anchor_id")
    if len(a) != EXPECTED_N_ANCHORS:
        raise RuntimeError(f"Expected {EXPECTED_N_ANCHORS} AI anchors, got {len(a)}")

    ai = pd.Series(a.ai_detected.astype(np.int8).to_numpy(), index=a.anchor_id)
    if int(ai.sum()) != EXPECTED_AI_POSITIVE:
        raise RuntimeError(f"Expected {EXPECTED_AI_POSITIVE} AI-positive anchors, got {int(ai.sum())}")

    cross = d[["anchor_id","ai_detected"]].copy()
    cross["anchor_id"] = cross.anchor_id.astype(str)
    if (cross.groupby("anchor_id").ai_detected.nunique(dropna=False) > 1).any():
        raise RuntimeError("Reader table has inconsistent AI labels within anchor")
    cross = cross.drop_duplicates("anchor_id").set_index("anchor_id").ai_detected.astype(np.int8).reindex(ai.index)
    if cross.isna().any() or not np.array_equal(cross.to_numpy(), ai.to_numpy()):
        raise RuntimeError("AI vectors disagree across frozen v16.1 sources")

    enc = encode_observed(
        rl, ai, readers=PRIMARY_READERS,
        expected_ai_matches=EXPECTED_AI_POSITIVE
    )

    if len(enc.y) != EXPECTED_N_ROWS:
        raise RuntimeError(f"Encoded observations expected {EXPECTED_N_ROWS}, got {len(enc.y)}")
    if len(enc.anchor_ids) != EXPECTED_N_ANCHORS:
        raise RuntimeError(f"Encoded anchors expected {EXPECTED_N_ANCHORS}, got {len(enc.anchor_ids)}")
    if len(enc.fov_values) != EXPECTED_N_FOV:
        raise RuntimeError(f"Encoded FOVs expected {EXPECTED_N_FOV}, got {len(enc.fov_values)}")

    audit = {
        "status":"V23_1_REALDATA_PRIMARY_INPUT_OK",
        "source_reader_table":str(real_path),
        "source_ai_table":str(ai_path),
        "readers":EXPECTED_READERS,
        "n_rows":len(d),
        "n_anchors":d.anchor_id.nunique(),
        "n_fovs":d.image_id.nunique(),
        "n_slides":d.slide.nunique(),
        "ai_positive":int(ai.sum()),
        "encoded_rf":len(enc.rf_keys),
        "encoded_ra":len(enc.ra_keys),
        "fov_mapping":fov_map,
        "by_condition":(
            d.groupby("condition")
             .agg(n=("reader_detected","size"),reader_positive=("reader_detected","sum"))
             .reset_index().to_dict(orient="records")
        ),
    }
    print("V23_1_REALDATA_INPUT_AUDIT "+json.dumps(audit,sort_keys=True),flush=True)

    import pymc as pm, arviz as az
    model=build_model(enc)
    t0=time.time()
    with model:
        idata=pm.sample(
            draws=args.draws,tune=args.tune,chains=args.chains,
            cores=min(args.chains,max(1,int(os.environ.get("SLURM_CPUS_PER_TASK","1")))),
            random_seed=args.seed,target_accept=args.target_accept,
            init="jitter+adapt_diag",progressbar=False,return_inferencedata=True
        )
    elapsed=time.time()-t0

    post=posterior_estimands(idata,enc)
    summ=summarize_real(post)
    div=int(np.asarray(idata.sample_stats["diverging"]).sum())
    ds=az.summary(idata,var_names=SCALAR_VARS,round_to=None)
    diagnostics={
        "elapsed_sec":elapsed,
        "divergences":div,
        "max_rhat_scalar":float(ds.r_hat.max()),
        "min_ess_bulk_scalar":float(ds.ess_bulk.min()),
        "min_ess_tail_scalar":float(ds.ess_tail.min()),
    }
    gate={
        **audit,**diagnostics,
        "sampler":{
            "draws":args.draws,"tune":args.tune,"chains":args.chains,
            "target_accept":args.target_accept,"seed":args.seed,
            "parameterization":"validated_v23_1_noncentered",
        },
        "posterior_summary":summ.to_dict(orient="records"),
        "status":"V23_1_REALDATA_PRIMARY_COMPLETE",
    }

    outdir=args.outdir or root/"outputs"/"v23_1_realdata_primary"
    outdir.mkdir(parents=True,exist_ok=True)
    post.to_csv(outdir/"v23_1_realdata_primary_estimand_draws.csv",index=False)
    summ.to_csv(outdir/"v23_1_realdata_primary_summary.csv",index=False)
    ds.to_csv(outdir/"v23_1_realdata_primary_mcmc_scalar_diagnostics.csv")
    (outdir/"v23_1_realdata_primary_gate.json").write_text(json.dumps(gate,indent=2))
    atomic_netcdf(idata,outdir/"v23_1_realdata_primary_idata.nc")

    print("V23_1_REALDATA_POSTERIOR_SUMMARY",flush=True)
    print(summ.to_string(index=False),flush=True)
    print("V23_1_REALDATA_MCMC_DIAGNOSTICS "+json.dumps(diagnostics,sort_keys=True),flush=True)
    print("V23_1_REALDATA_PRIMARY_COMPLETE",flush=True)

if __name__=="__main__":
    main()
