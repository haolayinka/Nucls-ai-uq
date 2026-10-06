#!/usr/bin/env python3
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd

SCENARIOS=[f"A{i}" for i in range(1,9)]
METHODS=["iMRMC_uStat11_conditionalD","iMRMC_uStat11_jointD"]
TARGETS=["theta_U_selected","theta_E_selected","delta_E_minus_U_selected"]

def metrics(z):
    e=z.estimate.to_numpy()-z.truth.to_numpy()
    n=len(z)
    sq=e**2
    cov=((z.ci_low<=z.truth)&(z.truth<=z.ci_high)).astype(float).to_numpy()
    wid=(z.ci_high-z.ci_low).to_numpy()
    rmse=float(np.sqrt(np.mean(sq)))
    return {
        "n":n,
        "bias":float(e.mean()),
        "mcse_bias":float(e.std(ddof=1)/np.sqrt(n)) if n>1 else np.nan,
        "rmse":rmse,
        "mcse_rmse":float(sq.std(ddof=1)/np.sqrt(n)/(2*rmse)) if n>1 and rmse>0 else np.nan,
        "coverage":float(cov.mean()),
        "mcse_coverage":float(np.sqrt(cov.mean()*(1-cov.mean())/n)),
        "mean_width":float(wid.mean()),
        "mcse_width":float(wid.std(ddof=1)/np.sqrt(n)) if n>1 else np.nan,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--freeze-root",type=Path,required=True)
    ap.add_argument("--results-root",type=Path,required=True)
    ap.add_argument("--outdir",type=Path,required=True)
    args=ap.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=True)

    files=sorted(args.results_root.glob("A*/rep_*_summary.csv"))
    if len(files)!=800:
        raise RuntimeError(f"iMRMC incomplete: found {len(files)}/800 summary files")
    allr=[]
    for f in files:
        d=pd.read_csv(f)
        if len(d)!=6:
            raise RuntimeError(f"Bad summary: {f}")
        allr.append(d)
    res=pd.concat(allr,ignore_index=True)
    if len(res)!=4800:
        raise RuntimeError(f"Expected 4800 rows, got {len(res)}")
    counts=res.groupby(["method","scenario","target"]).size()
    if not (counts==100).all():
        raise RuntimeError("Not exactly 100 replicates in every method/scenario/target cell")

    truth=pd.read_csv(args.freeze_root/"audit"/"truth_by_scenario_rep.csv")
    tmap=truth.set_index(["scenario","rep"])
    res["truth"]=[float(tmap.loc[(r.scenario,int(r.rep)),r.target]) for _,r in res.iterrows()]
    res.to_csv(args.outdir/"imrmc_all_replication_results.csv",index=False)

    rows=[]
    for (m,s,t),z in res.groupby(["method","scenario","target"]):
        rows.append({"method":m,"scenario":s,"target":t,**metrics(z)})
    pd.DataFrame(rows).to_csv(args.outdir/"imrmc_adversarial_metrics.csv",index=False)

    print("IMRMC_ADVERSARIAL_800_AGGREGATION_COMPLETE")

if __name__=="__main__":
    main()
