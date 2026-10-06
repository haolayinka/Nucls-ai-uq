#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np, pandas as pd
from nucls_v24_3_frozen_models import *

def fit_seed(model,scenario,rep):
    x=f"v24.3|{model}|{scenario}|{rep}".encode()
    return int.from_bytes(hashlib.sha256(x).digest()[:4],"little") % 2_000_000_000

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",choices=["M1","M2"],required=True)
    ap.add_argument("--index",type=int,required=True)
    ap.add_argument("--freeze-root",type=Path,required=True)
    ap.add_argument("--outdir",type=Path,required=True)
    ap.add_argument("--draws",type=int,default=4000)
    ap.add_argument("--tune",type=int,default=2000)
    ap.add_argument("--chains",type=int,default=4)
    ap.add_argument("--target-accept",type=float,default=.99)
    ap.add_argument("--validate-only",action="store_true")
    args=ap.parse_args()

    self_test()
    sid,rep=index_to_scenario_rep(args.index)
    dat=load_frozen(args.freeze_root,sid,rep,verify_hash=True)
    enc=encode_selected(dat)
    audit={"model":args.model,"scenario":sid,"rep":rep,"index":args.index,
           "source_file":str(dat["path"]),"n_selected":enc.n_anchors,
           "n_fovs":enc.n_fovs,"n_obs":len(enc.y),
           "n_obs_U":int((enc.c==0).sum()),"n_obs_E":int((enc.c==1).sum()),
           "ai_positive_selected":int(enc.anchor_ai.sum()),
           "n_rf":enc.n_rf,"n_ra":enc.n_ra}
    if args.validate_only:
        print("FROZEN_WORKER_VALIDATE_OK "+json.dumps(audit,sort_keys=True))
        return

    out=args.outdir/args.model/sid
    out.mkdir(parents=True,exist_ok=True)
    stem=f"rep_{rep:03d}"
    gate_path=out/f"{stem}_gate.json"
    if gate_path.exists():
        try:
            old=json.loads(gate_path.read_text())
            if old.get("status")=="COMPLETE" and old.get("freeze_hash")==EXPECTED_FREEZE_HASH:
                print(f"SKIP {args.model} {sid} {rep}: complete")
                return
        except Exception:
            pass

    import pymc as pm, arviz as az
    model=build_model(enc,args.model)
    seed=fit_seed(args.model,sid,rep)
    t0=time.time()
    with model:
        idata=pm.sample(draws=args.draws,tune=args.tune,chains=args.chains,
            cores=min(args.chains,max(1,int(os.environ.get("SLURM_CPUS_PER_TASK","1")))),
            random_seed=seed,target_accept=args.target_accept,
            init="jitter+adapt_diag",progressbar=False,return_inferencedata=True)
    elapsed=time.time()-t0

    pu,pe=baseline_future_probabilities(idata,enc,batch=100)
    est_draws=agreement_draws_from_prob(pu,pe,enc.anchor_ai)
    summ=summarize_draw_matrix(est_draws)
    summ.to_csv(out/f"{stem}_summary.csv",index=False)

    if args.model=="M1":
        m4=m4_summary_from_probabilities(pu,pe,enc.anchor_ai)
        m4.to_csv(out/f"{stem}_m4_sensitivity.csv",index=False)

    ds=az.summary(idata,var_names=scalar_var_names(args.model),round_to=None)
    div=int(np.asarray(idata.sample_stats["diverging"]).sum())
    diag={"divergences":div,"max_rhat_scalar":float(ds.r_hat.max()),
          "min_ess_bulk_scalar":float(ds.ess_bulk.min()),
          "min_ess_tail_scalar":float(ds.ess_tail.min()),
          "elapsed_sec":elapsed}
    ds.to_csv(out/f"{stem}_scalar_diagnostics.csv")
    gate={**audit,**diag,"freeze_hash":EXPECTED_FREEZE_HASH,
          "seed":seed,"sampler":{"draws":args.draws,"tune":args.tune,"chains":args.chains,
          "target_accept":args.target_accept},
          "computational_ok":bool(div==0 and diag["max_rhat_scalar"]<=1.01
                                  and diag["min_ess_bulk_scalar"]>=400
                                  and diag["min_ess_tail_scalar"]>=400),
          "status":"COMPLETE"}
    gate_path.write_text(json.dumps(gate,indent=2))
    print("FROZEN_BAYES_COMPLETE "+json.dumps(gate,sort_keys=True))

if __name__=="__main__":
    main()
