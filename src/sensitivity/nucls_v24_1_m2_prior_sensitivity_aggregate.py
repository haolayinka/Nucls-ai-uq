#!/usr/bin/env python3
from pathlib import Path
import argparse,json
import pandas as pd

BASELINE={
    "name":"baseline_existing_M2",
    "theta_U_selected":0.578717,
    "theta_E_selected":0.578711,
    "delta_E_minus_U_selected":-0.000007,
}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outdir",type=Path,default=Path("outputs/v24_1_m2_prior_sensitivity"))
    args=ap.parse_args()
    gates=sorted(args.outdir.glob("config*_gate.json"))
    if len(gates)!=4:
        raise SystemExit(f"Expected 4 sensitivity gates, found {len(gates)}")
    rows=[BASELINE.copy()]
    comp=[]
    for p in gates:
        g=json.loads(p.read_text())
        rec={"name":g["name"]}
        for x in g["estimands"]:
            rec[x["target"]]=x["estimate"]
            rec[x["target"]+"_low"]=x["ci_low"]
            rec[x["target"]+"_high"]=x["ci_high"]
        rows.append(rec)
        comp.append({
            "name":g["name"],"divergences":g["divergences"],"max_rhat":g["max_rhat"],
            "min_ess_bulk":g["min_ess_bulk"],"min_ess_tail":g["min_ess_tail"],
            "elapsed_sec":g["elapsed_sec"]
        })
    df=pd.DataFrame(rows)
    for c in ["theta_U_selected","theta_E_selected","delta_E_minus_U_selected"]:
        df[c+"_minus_baseline"]=df[c]-df.loc[df.name=="baseline_existing_M2",c].iloc[0]
    df.to_csv(args.outdir/"m2_prior_sensitivity_estimand_comparison.csv",index=False)
    pd.DataFrame(comp).to_csv(args.outdir/"m2_prior_sensitivity_computational.csv",index=False)
    print(df.to_string(index=False))
    print("\nCOMPUTATIONAL")
    print(pd.DataFrame(comp).to_string(index=False))
    print("M2_PRIOR_SENSITIVITY_AGGREGATION_COMPLETE")

if __name__=="__main__":
    main()
