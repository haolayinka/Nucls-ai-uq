#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Import the validated module as a module object so the core-analysis-only
# anchor-count guard can be changed from 1144 to the prespecified 679.
import nucls_v23_1_joint_bayes_logit as v231

EXPECTED_READERS = ["SP.3", "JP.1", "JP.2", "JP.5", "JP.6"]
EXPECTED_N_ROWS = 6790
EXPECTED_N_ANCHORS = 679
EXPECTED_N_FOV = 32
EXPECTED_N_SLIDES = 4

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
        raise RuntimeError(f"posterior_estimands missing columns: {missing}")
    rows=[]
    for c in ESTIMAND_COLS:
        x=post[c].to_numpy(float)
        if not np.isfinite(x).all():
            raise RuntimeError(f"Non-finite posterior estimand draws in {c}")
        rows.append({
            "target":c,
            "estimate":float(x.mean()),
            "posterior_sd":float(x.std(ddof=1)),
            "ci_low":float(np.quantile(x,.025)),
            "ci_high":float(np.quantile(x,.975)),
            "interval_width":float(np.quantile(x,.975)-np.quantile(x,.025)),
            "prob_gt_zero":float(np.mean(x>0)) if c=="delta_E_minus_U_selected" else np.nan,
        })
    return pd.DataFrame(rows)

def atomic_netcdf(idata,path:Path):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.unlink(missing_ok=True)
    idata.to_netcdf(tmp)
    os.replace(tmp,path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument("--draws",type=int,default=4000)
    ap.add_argument("--tune",type=int,default=2000)
    ap.add_argument("--chains",type=int,default=4)
    ap.add_argument("--target-accept",type=float,default=.99)
    ap.add_argument("--seed",type=int,default=231002)
    ap.add_argument("--outdir",type=Path,default=None)
    args=ap.parse_args()

    # First verify the untouched validated module.
    v231.self_test()

    # CORE-SENSITIVITY INTERFACE PATCH ONLY.
    # The production module's encoder has an explicit 1144-anchor guard.
    # Confirm it is still the expected validated value before overriding it.
    original_expected = getattr(v231, "EXPECTED_N_ANCHORS", None)
    if original_expected != 1144:
        raise RuntimeError(
            f"Refusing unexpected module state: EXPECTED_N_ANCHORS={original_expected}, expected 1144"
        )
    v231.EXPECTED_N_ANCHORS = EXPECTED_N_ANCHORS
    print(
        f"V23_1_CORE_INTERFACE_PATCH_OK EXPECTED_N_ANCHORS "
        f"{original_expected}->{v231.EXPECTED_N_ANCHORS}",
        flush=True,
    )

    root=args.project_root
    real_path=root/"outputs"/"scoring_v16_1"/"restricted_recorded_reader_detection_agreement_v16.csv"
    ai_path=root/"outputs"/"scoring_v16_1"/"restricted_anchor_matches_v16.csv"
    for p in (real_path,ai_path):
        if not p.exists():
            raise FileNotFoundError(p)

    raw=pd.read_csv(real_path)
    ai_tbl=pd.read_csv(ai_path)

    d5=raw.loc[raw.reader.isin(EXPECTED_READERS)].copy()
    d5["condition_code"]=d5["condition"].map({"Unbiased":"U","Evaluation":"E"})
    if d5["condition_code"].isna().any():
        raise RuntimeError("Unexpected condition label")

    key=["anchor_id","reader","condition_code"]
    if d5.duplicated(key).any():
        raise RuntimeError("Duplicate anchor-reader-condition rows before core selection")

    # Exact 5-reader x 2-condition fully crossed anchors.
    cell_counts=d5.groupby("anchor_id").size()
    reader_counts=d5.groupby("anchor_id")["reader"].nunique()
    condition_counts=d5.groupby("anchor_id")["condition_code"].nunique()

    complete=cell_counts.index[
        (cell_counts==10) &
        (reader_counts.reindex(cell_counts.index)==5) &
        (condition_counts.reindex(cell_counts.index)==2)
    ]
    d=d5.loc[d5.anchor_id.isin(complete)].copy()

    checks={
        "rows":(len(d),EXPECTED_N_ROWS),
        "anchors":(d.anchor_id.nunique(),EXPECTED_N_ANCHORS),
        "fovs":(d.image_id.nunique(),EXPECTED_N_FOV),
        "slides":(d.slide.nunique(),EXPECTED_N_SLIDES),
    }
    for name,(got,exp) in checks.items():
        if got!=exp:
            raise RuntimeError(f"Expected {exp} {name}, got {got}")

    expected_slides={
        "TCGA-A1-A0SP-01Z-00-DX1",
        "TCGA-A7-A0DA-01Z-00-DX1",
        "TCGA-C8-A12V-01Z-00-DX1",
        "TCGA-E2-A158-01Z-00-DX1",
    }
    if set(d.slide.unique())!=expected_slides:
        raise RuntimeError(f"Unexpected fully crossed slide set: {sorted(d.slide.unique())}")
    if list(v231.PRIMARY_READERS)!=EXPECTED_READERS:
        raise RuntimeError(f"PRIMARY_READERS changed: {list(v231.PRIMARY_READERS)}")

    # Encoder requires numeric FOV IDs.
    fov_values=sorted(d.image_id.astype(str).unique())
    fov_map={name:i for i,name in enumerate(fov_values)}
    if len(fov_map)!=EXPECTED_N_FOV:
        raise RuntimeError("Incorrect FOV mapping size")

    rl=pd.DataFrame({
        "anchor_id":d.anchor_id.astype(str),
        "fov_id":d.image_id.astype(str).map(fov_map).astype(float),
        "reader":d.reader.astype(str),
        "condition":d.condition_code,
        "observed":np.ones(len(d),dtype=np.int8),
        "reader_detected":d.reader_detected.astype(np.int8),
    })

    a=ai_tbl[["anchor_id","ai_detected"]].copy()
    a["anchor_id"]=a.anchor_id.astype(str)
    if a.anchor_id.duplicated().any():
        chk=a.groupby("anchor_id").ai_detected.nunique(dropna=False)
        if (chk>1).any():
            raise RuntimeError("AI source has conflicting ai_detected values")
        a=a.drop_duplicates("anchor_id")

    core_ids=set(d.anchor_id.astype(str).unique())
    a=a.loc[a.anchor_id.isin(core_ids)].copy()
    if len(a)!=EXPECTED_N_ANCHORS:
        raise RuntimeError(f"Expected {EXPECTED_N_ANCHORS} core AI anchors, got {len(a)}")

    ai=pd.Series(a.ai_detected.astype(np.int8).to_numpy(),index=a.anchor_id)

    cross=d[["anchor_id","ai_detected"]].copy()
    cross["anchor_id"]=cross.anchor_id.astype(str)
    if (cross.groupby("anchor_id").ai_detected.nunique(dropna=False)>1).any():
        raise RuntimeError("Reader table has inconsistent AI labels within anchor")
    cross=(cross.drop_duplicates("anchor_id")
                .set_index("anchor_id").ai_detected.astype(np.int8)
                .reindex(ai.index))
    if cross.isna().any() or not np.array_equal(cross.to_numpy(),ai.to_numpy()):
        raise RuntimeError("AI vectors disagree across frozen v16.1 sources")

    enc=v231.encode_observed(
        rl,ai,readers=v231.PRIMARY_READERS,
        expected_ai_matches=int(ai.sum())
    )

    if len(enc.y)!=EXPECTED_N_ROWS:
        raise RuntimeError(f"Encoded observations expected {EXPECTED_N_ROWS}, got {len(enc.y)}")
    if len(enc.anchor_ids)!=EXPECTED_N_ANCHORS:
        raise RuntimeError(f"Encoded anchors expected {EXPECTED_N_ANCHORS}, got {len(enc.anchor_ids)}")
    if len(enc.fov_values)!=EXPECTED_N_FOV:
        raise RuntimeError(f"Encoded FOVs expected {EXPECTED_N_FOV}, got {len(enc.fov_values)}")

    audit={
        "status":"V23_1_REALDATA_CORE_INPUT_OK",
        "analysis":"five_reader_679_fully_crossed",
        "interface_patch":{
            "variable":"EXPECTED_N_ANCHORS",
            "validated_primary_value":1144,
            "core_sensitivity_value":679,
            "model_specification_changed":False,
        },
        "readers":EXPECTED_READERS,
        "n_rows":len(d),
        "n_anchors":d.anchor_id.nunique(),
        "n_fovs":d.image_id.nunique(),
        "n_slides":d.slide.nunique(),
        "slides":sorted(d.slide.unique()),
        "ai_positive":int(ai.sum()),
        "encoded_rf":len(enc.rf_keys),
        "encoded_ra":len(enc.ra_keys),
        "fov_mapping":fov_map,
        "by_condition":(
            d.groupby("condition")
             .agg(n=("reader_detected","size"),
                  reader_positive=("reader_detected","sum"),
                  agreement=("detection_agreement","sum"))
             .reset_index().to_dict(orient="records")
        ),
    }
    print("V23_1_REALDATA_CORE_INPUT_AUDIT "+json.dumps(audit,sort_keys=True),flush=True)

    import pymc as pm, arviz as az
    model=v231.build_model(enc)
    t0=time.time()
    with model:
        idata=pm.sample(
            draws=args.draws,tune=args.tune,chains=args.chains,
            cores=min(args.chains,max(1,int(os.environ.get("SLURM_CPUS_PER_TASK","1")))),
            random_seed=args.seed,target_accept=args.target_accept,
            init="jitter+adapt_diag",progressbar=False,return_inferencedata=True
        )
    elapsed=time.time()-t0

    post=v231.posterior_estimands(idata,enc)
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
        "status":"V23_1_REALDATA_CORE_COMPLETE",
    }

    outdir=args.outdir or root/"outputs"/"v23_1_realdata_core"
    outdir.mkdir(parents=True,exist_ok=True)
    post.to_csv(outdir/"v23_1_realdata_core_estimand_draws.csv",index=False)
    summ.to_csv(outdir/"v23_1_realdata_core_summary.csv",index=False)
    ds.to_csv(outdir/"v23_1_realdata_core_mcmc_scalar_diagnostics.csv")
    (outdir/"v23_1_realdata_core_gate.json").write_text(json.dumps(gate,indent=2))
    atomic_netcdf(idata,outdir/"v23_1_realdata_core_idata.nc")

    print("V23_1_REALDATA_CORE_POSTERIOR_SUMMARY",flush=True)
    print(summ.to_string(index=False),flush=True)
    print("V23_1_REALDATA_CORE_MCMC_DIAGNOSTICS "+json.dumps(diagnostics,sort_keys=True),flush=True)
    print("V23_1_REALDATA_CORE_COMPLETE",flush=True)

if __name__=="__main__":
    main()
