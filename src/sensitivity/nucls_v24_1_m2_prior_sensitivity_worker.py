#!/usr/bin/env python3
from __future__ import annotations
from repo_paths import repo_root
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DEFAULT=repo_root()
PRIMARY_READERS=("SP.3","JP.1","JP.2","JP.5","JP.6")

# Four orthogonal sensitivity fits. The already-completed primary fit is the baseline:
# beta_ai HalfNormal(1.5);
# reader/FOV-type SDs HalfNormal(1.0);
# anchor-type SDs HalfNormal(1.5).
CONFIGS={
    0:{
        "name":"ai_loading_more_regularizing",
        "beta_ai_scale":0.75,
        "sd_small_scale":1.0,
        "sd_anchor_scale":1.5,
    },
    1:{
        "name":"ai_loading_weaker",
        "beta_ai_scale":3.0,
        "sd_small_scale":1.0,
        "sd_anchor_scale":1.5,
    },
    2:{
        "name":"variance_more_regularizing",
        "beta_ai_scale":1.5,
        "sd_small_scale":0.50,
        "sd_anchor_scale":0.75,
    },
    3:{
        "name":"variance_weaker",
        "beta_ai_scale":1.5,
        "sd_small_scale":2.0,
        "sd_anchor_scale":3.0,
    },
}

def find_v231(project_root:Path):
    cand=project_root/"scripts"/"nucls_v23_1_joint_bayes_logit.py"
    if not cand.exists():
        raise FileNotFoundError(
            f"Frozen v23.1 module not found: {cand}. "
            "This sensitivity bundle deliberately does not overwrite or replace it."
        )
    sys.path.insert(0,str(cand.parent))
    import nucls_v23_1_joint_bayes_logit as v231
    return v231

def load_primary_data(project_root:Path,v231):
    src=project_root/"outputs"/"scoring_v16_1"/"restricted_recorded_reader_detection_agreement_v16.csv"
    if not src.exists():
        raise FileNotFoundError(src)
    x=pd.read_csv(src)

    req={"anchor_id","image_id","condition","reader","ai_detected","reader_detected"}
    miss=req-set(x.columns)
    if miss:
        raise ValueError(f"{src} missing required columns: {sorted(miss)}")

    x=x[x["reader"].isin(PRIMARY_READERS)].copy()
    if len(x)!=10833:
        raise ValueError(f"Expected 10,833 primary observed reader rows, got {len(x)}")
    if x["anchor_id"].astype(str).nunique()!=1144:
        raise ValueError("Expected 1,144 anchors")

    # This source contains recorded reader observations only, so every row is observed.
    # Normalize the real-data labels used by v16.1 to the U/E codes expected by
    # the frozen v23.1 encoder. Refuse any unknown label rather than silently
    # coercing it.
    cond_raw=set(x["condition"].astype(str).unique())
    cond_map={"Unbiased":"U","Evaluation":"E","U":"U","E":"E"}
    unknown=sorted(cond_raw-set(cond_map))
    if unknown:
        raise ValueError(f"Unexpected condition labels: {unknown}; observed={sorted(cond_raw)}")
    x["condition_code"]=x["condition"].astype(str).map(cond_map)
    if set(x["condition_code"].unique()) != {"U","E"}:
        raise ValueError(f"Condition normalization failed: {sorted(x['condition_code'].unique())}")

    fovs=sorted(x["image_id"].astype(str).unique())
    fmap={f:i for i,f in enumerate(fovs)}
    reader_long=pd.DataFrame({
        "anchor_id":x["anchor_id"].astype(str),
        "fov_id":x["image_id"].astype(str).map(fmap).astype(float),
        "reader":x["reader"].astype(str),
        "condition":x["condition_code"].astype(str),
        "observed":1,
        "reader_detected":x["reader_detected"].astype(int),
    })

    ai_check=x.groupby(x["anchor_id"].astype(str))["ai_detected"].nunique()
    if int(ai_check.max())!=1:
        raise ValueError("AI detection is not unique within anchor")
    ai=x.assign(anchor_id=x.anchor_id.astype(str)).groupby("anchor_id")["ai_detected"].first().astype(int)
    if len(ai)!=1144 or int(ai.sum())!=593:
        raise ValueError(f"AI audit failed: n={len(ai)}, positives={int(ai.sum())}")

    enc=v231.encode_observed(
        reader_long,ai,
        readers=v231.PRIMARY_READERS,
        expected_ai_matches=593
    )

    # Reproduce the frozen primary input audit.
    audit={
        "source":str(src),
        "n_rows":len(x),
        "n_anchors":len(enc.anchor_ids),
        "n_fovs":len(enc.fov_values),
        "n_readers":len(enc.reader_names),
        "ai_positive":int(enc.anchor_ai.sum()),
        "E_rows":int((enc.c==1).sum()),
        "U_rows":int((enc.c==0).sum()),
        "encoded_rf":len(enc.rf_keys),
        "encoded_ra":len(enc.ra_keys),
    }
    expected={
        "n_rows":10833,"n_anchors":1144,"n_fovs":52,"n_readers":5,
        "ai_positive":593,"E_rows":5204,"U_rows":5629,
        "encoded_rf":257,"encoded_ra":5645,
    }
    for k,v in expected.items():
        if audit[k]!=v:
            raise ValueError(f"Frozen audit mismatch for {k}: got {audit[k]}, expected {v}")
    return enc,audit

def build_model(data,cfg):
    import pymc as pm
    coords={
        "reader":data.reader_names,
        "fov":np.arange(len(data.fov_values)),
        "anchor":np.arange(len(data.anchor_ids)),
        "rf_obs":np.arange(len(data.rf_keys)),
        "ra_obs":np.arange(len(data.ra_keys)),
        "obs":np.arange(len(data.y)),
    }
    with pm.Model(coords=coords) as model:
        alpha=pm.Normal("alpha",0,2.5)
        beta_E=pm.Normal("beta_E",0,1.0)
        alpha_ai=pm.Normal("alpha_ai",0,2.5)
        beta_ai=pm.HalfNormal("beta_ai",cfg["beta_ai_scale"])

        ss=cfg["sd_small_scale"]
        sa=cfg["sd_anchor_scale"]
        sd_reader=pm.HalfNormal("sd_reader",ss)
        sd_reader_condition=pm.HalfNormal("sd_reader_condition",ss)
        sd_fov=pm.HalfNormal("sd_fov",ss)
        sd_reader_fov=pm.HalfNormal("sd_reader_fov",ss)
        sd_anchor=pm.HalfNormal("sd_anchor",sa)
        sd_reader_anchor=pm.HalfNormal("sd_reader_anchor",sa)

        z_reader=pm.Normal("z_reader",0,1,dims="reader")
        z_reader_condition=pm.Normal("z_reader_condition",0,1,dims="reader")
        z_fov=pm.Normal("z_fov",0,1,dims="fov")
        z_reader_fov=pm.Normal("z_reader_fov",0,1,dims="rf_obs")
        z_anchor=pm.Normal("z_anchor",0,1,dims="anchor")
        z_reader_anchor=pm.Normal("z_reader_anchor",0,1,dims="ra_obs")

        u=pm.Deterministic("u_reader",sd_reader*z_reader,dims="reader")
        b=pm.Deterministic("b_reader_condition",sd_reader_condition*z_reader_condition,dims="reader")
        g=pm.Deterministic("g_fov",sd_fov*z_fov,dims="fov")
        h=pm.Deterministic("h_reader_fov",sd_reader_fov*z_reader_fov,dims="rf_obs")
        aa=pm.Deterministic("a_anchor",sd_anchor*z_anchor,dims="anchor")
        q=pm.Deterministic("q_reader_anchor",sd_reader_anchor*z_reader_anchor,dims="ra_obs")

        d_shared=g[data.anchor_f]+aa
        pm.Bernoulli(
            "ai_detected_obs",
            logit_p=alpha_ai+beta_ai*d_shared,
            observed=data.anchor_ai,dims="anchor"
        )
        eta=(alpha+beta_E*data.c+d_shared[data.a]+u[data.r]+
             b[data.r]*data.c+h[data.rf]+q[data.ra])
        pm.Bernoulli("reader_detected",logit_p=eta,observed=data.y,dims="obs")
    return model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config-id",type=int,required=True,choices=sorted(CONFIGS))
    ap.add_argument("--project-root",type=Path,default=PROJECT_DEFAULT)
    ap.add_argument("--draws",type=int,default=4000)
    ap.add_argument("--tune",type=int,default=2000)
    ap.add_argument("--chains",type=int,default=4)
    ap.add_argument("--target-accept",type=float,default=.99)
    ap.add_argument("--outdir",type=Path,default=None)
    ap.add_argument("--preflight-only",action="store_true",
                    help="Validate frozen real-data input and exit before model fitting.")
    args=ap.parse_args()

    cfg=CONFIGS[args.config_id]
    v231=find_v231(args.project_root)
    v231.self_test()
    data,audit=load_primary_data(args.project_root,v231)

    print("M2_PRIOR_SENSITIVITY_INPUT "+
          json.dumps({"config_id":args.config_id,**cfg,**audit},sort_keys=True),
          flush=True)

    if args.preflight_only:
        print("M2_PRIOR_SENSITIVITY_PREFLIGHT_OK",flush=True)
        return

    import pymc as pm, arviz as az
    model=build_model(data,cfg)
    seed=244000+args.config_id
    t0=time.time()
    with model:
        idata=pm.sample(
            draws=args.draws,tune=args.tune,chains=args.chains,
            cores=min(args.chains,max(1,int(os.environ.get("SLURM_CPUS_PER_TASK","1")))),
            random_seed=seed,target_accept=args.target_accept,
            init="jitter+adapt_diag",progressbar=False,return_inferencedata=True
        )
    elapsed=time.time()-t0

    est=v231.posterior_estimands(idata,data)
    summ=v231.summarize(est)
    scalar_vars=[
        "alpha","beta_E","alpha_ai","beta_ai","sd_reader","sd_reader_condition",
        "sd_fov","sd_reader_fov","sd_anchor","sd_reader_anchor"
    ]
    ds=az.summary(idata,var_names=scalar_vars,round_to=None)
    div=int(np.asarray(idata.sample_stats["diverging"]).sum())

    outdir=args.outdir or args.project_root/"outputs"/"v24_1_m2_prior_sensitivity"
    outdir.mkdir(parents=True,exist_ok=True)
    stem=f"config{args.config_id}_{cfg['name']}"
    summ.to_csv(outdir/f"{stem}_estimands.csv",index=False)
    ds.to_csv(outdir/f"{stem}_scalar_diagnostics.csv")

    gate={
        "config_id":args.config_id,**cfg,**audit,
        "elapsed_sec":elapsed,"divergences":div,
        "max_rhat":float(ds.r_hat.max()),
        "min_ess_bulk":float(ds.ess_bulk.min()),
        "min_ess_tail":float(ds.ess_tail.min()),
        "estimands":summ.to_dict(orient="records"),
        "status":"M2_PRIOR_SENSITIVITY_COMPLETE"
    }
    (outdir/f"{stem}_gate.json").write_text(json.dumps(gate,indent=2))
    print("M2_PRIOR_SENSITIVITY_SUMMARY",flush=True)
    print(summ.to_string(index=False),flush=True)
    print("M2_PRIOR_SENSITIVITY_DIAGNOSTICS "+
          json.dumps({k:gate[k] for k in ["elapsed_sec","divergences","max_rhat","min_ess_bulk","min_ess_tail"]},
                     sort_keys=True),flush=True)
    print("M2_PRIOR_SENSITIVITY_COMPLETE",flush=True)

if __name__=="__main__":
    main()
