#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import t
from nucls_v24_3_frozen_models import *

TARGETS=["theta_U_selected","theta_E_selected","delta_E_minus_U_selected"]

def t_interval(vals, bounds):
    vals=np.asarray(vals,float)
    est=float(vals.mean())
    if len(vals)<2:
        return est,np.nan,np.nan,np.nan
    se=float(vals.std(ddof=1)/np.sqrt(len(vals)))
    q=float(t.ppf(.975,len(vals)-1))
    lo=max(bounds[0],est-q*se); hi=min(bounds[1],est+q*se)
    return est,se,lo,hi

def fit_one(root,scenario,rep):
    d=load_frozen(root,scenario,rep)
    sel=d["selected"]; ai=d["ai"]
    rates={"U":[],"E":[]}
    pooled={}
    for c,y,obs in [("U",d["reader_y_U"],d["obs_U"]),("E",d["reader_y_E"],d["obs_E"])]:
        all_ag=[]
        for r in range(5):
            m=obs[r]&sel
            if not m.any():
                raise RuntimeError(f"no observed rows {scenario} {rep} {c} reader{r}")
            ag=(y[r,m]==ai[m]).astype(float)
            rates[c].append(float(ag.mean()))
            all_ag.append(ag)
        pooled[c]=float(np.concatenate(all_ag).mean())
    u=np.asarray(rates["U"]); e=np.asarray(rates["E"]); de=e-u
    rows=[]
    for name,vals,bounds in [
        ("theta_U_selected",u,(0,1)),("theta_E_selected",e,(0,1)),
        ("delta_E_minus_U_selected",de,(-1,1))]:
        est,se,lo,hi=t_interval(vals,bounds)
        rows.append({"scenario":scenario,"rep":rep,"target":name,
                     "estimate":est,"se_reader_t":se,"ci_low":lo,"ci_high":hi,
                     "interval_width":hi-lo,"n_readers":5})
    meta={"scenario":scenario,"rep":rep,"n_selected":int(sel.sum()),
          "pooled_raw_U":pooled["U"],"pooled_raw_E":pooled["E"],
          "pooled_raw_delta":pooled["E"]-pooled["U"]}
    return pd.DataFrame(rows),meta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--freeze-root",type=Path,required=True)
    ap.add_argument("--outdir",type=Path,required=True)
    args=ap.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=True)
    allrows=[]; metas=[]
    for sid in SCENARIOS:
        for rep in range(100):
            s,m=fit_one(args.freeze_root,sid,rep)
            allrows.append(s); metas.append(m)
    pd.concat(allrows,ignore_index=True).to_csv(args.outdir/"m0_reader_equal_summary.csv",index=False)
    pd.DataFrame(metas).to_csv(args.outdir/"m0_pooled_raw_audit.csv",index=False)
    print("M0_FROZEN_800_COMPLETE")

if __name__=="__main__":
    main()
