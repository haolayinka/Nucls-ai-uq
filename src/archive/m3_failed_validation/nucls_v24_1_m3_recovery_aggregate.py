#!/usr/bin/env python3
from pathlib import Path
import argparse,json,math
import pandas as pd, numpy as np

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outdir",type=Path,default=Path("outputs/v24_1_m3_recovery"))
    ap.add_argument("--expected-reps",type=int,default=20)
    args=ap.parse_args()
    gates=sorted(args.outdir.glob("rep*_gate.json"))
    if len(gates)!=args.expected_reps:
        raise SystemExit(f"Expected {args.expected_reps} gates, found {len(gates)}")
    rows=[]; comp=[]
    for p in gates:
        g=json.loads(p.read_text())
        comp.append({"rep":g["rep"],"divergences":g["divergences"],"max_rhat":g["max_rhat_scalar"],
                     "min_bulk":g["min_ess_bulk_scalar"],"min_tail":g["min_ess_tail_scalar"],
                     "elapsed_sec":g["elapsed_sec"],
                     "realized_fov_corr":g["realized_fov_corr"],
                     "realized_anchor_corr":g["realized_anchor_corr"]})
        for r in g["summary"]:
            rows.append({"rep":g["rep"],**r})
    df=pd.DataFrame(rows); c=pd.DataFrame(comp)
    out=[]
    for tgt,g in df.groupby("target"):
        e=g.error.to_numpy(float); cov=g.covered.astype(float).to_numpy()
        p=float(cov.mean()); n=len(g)
        out.append({"target":tgt,"n":n,"bias":float(e.mean()),
                    "mcse_bias":float(e.std(ddof=1)/math.sqrt(n)),
                    "rmse":float(np.sqrt(np.mean(e**2))),
                    "coverage":p,
                    "mcse_coverage":math.sqrt(p*(1-p)/n),
                    "mean_width":float(g.interval_width.mean())})
    s=pd.DataFrame(out)
    s.to_csv(args.outdir/"m3_recovery_20rep_summary.csv",index=False)
    c.to_csv(args.outdir/"m3_recovery_20rep_computational.csv",index=False)
    print("M3 20-REP RECOVERY SUMMARY")
    print(s.to_string(index=False))
    print("\nCOMPUTATIONAL")
    print(c.agg({"divergences":"sum","max_rhat":"max","min_bulk":"min","min_tail":"min"}).to_string())
    print("\nMean realized correlations:")
    print(c[["realized_fov_corr","realized_anchor_corr"]].mean().to_string())
    print("M3_RECOVERY_AGGREGATION_COMPLETE")

if __name__=="__main__":
    main()
