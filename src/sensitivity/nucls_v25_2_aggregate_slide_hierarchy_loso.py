#!/usr/bin/env python3
from __future__ import annotations
from repo_paths import repo_root
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",type=Path,default=repo_root())
    ap.add_argument("--root",type=Path,default=None)
    ap.add_argument("--outdir",type=Path,default=None)
    args=ap.parse_args()
    project=args.project_root
    root=args.root or project/"outputs"/"v25_2_slide_hierarchy_loso"
    out=args.outdir or project/"outputs"/"v25_2_slide_hierarchy_loso_aggregate"
    out.mkdir(parents=True,exist_ok=True)
    gates=[]
    for g in sorted(root.glob("*/gate.json")):
        x=json.loads(g.read_text()); x["_dir"]=str(g.parent); gates.append(x)
    if len(gates)!=6: raise RuntimeError(f"Expected 6 gate files, found {len(gates)}")
    flat=[{k:v for k,v in g.items() if not isinstance(v,(dict,list))} for g in gates]
    pd.DataFrame(flat).to_csv(out/"computational_gates.csv",index=False)
    bad=[g for g in gates if not g["computational_ok"]]
    if bad:
        print("V25_2_SLIDE_HIERARCHY_AGGREGATION_BLOCKED_BY_COMPUTATIONAL_FAILURE")
        for g in bad:
            print(g["task_index"],g["held_out_slide"],g["divergences"],g["max_rhat_scalar"],
                  g["min_ess_bulk_scalar"],g["min_ess_tail_scalar"])
        raise SystemExit(3)
    full=[g for g in gates if g["task_index"]==0][0]
    fulls=pd.read_csv(Path(full["_dir"])/"summary.csv")
    current=pd.read_csv(project/"outputs"/"v23_1_realdata_primary"/"v23_1_realdata_primary_summary.csv")
    comp=current[["target","estimate","ci_low","ci_high"]].merge(
        fulls[["target","estimate","ci_low","ci_high"]],on="target",
        suffixes=("_current_M2","_slide_M2"))
    comp["estimate_shift_slide_minus_current"]=comp.estimate_slide_M2-comp.estimate_current_M2
    comp.to_csv(out/"full_slide_hierarchy_vs_current_M2.csv",index=False)
    rows=[]
    for g in sorted(gates,key=lambda x:x["task_index"]):
        if g["task_index"]==0: continue
        s=pd.read_csv(Path(g["_dir"])/"summary.csv")
        for _,r in s.iterrows():
            rows.append({"held_out_slide":g["held_out_slide"],"n_anchors":g["n_anchors"],
                         "n_fovs":g["n_fovs"],"target":r["target"],"estimate":r["estimate"],
                         "ci_low":r["ci_low"],"ci_high":r["ci_high"],"posterior_sd":r["posterior_sd"]})
    loso=pd.DataFrame(rows); loso.to_csv(out/"loso_refit_summaries.csv",index=False)
    rg=(loso.groupby("target").agg(n_loso=("estimate","size"),estimate_min=("estimate","min"),
          estimate_max=("estimate","max"),
          estimate_range=("estimate",lambda x:float(x.max()-x.min())),
          median_estimate=("estimate","median")).reset_index())
    rg.to_csv(out/"loso_instability_ranges.csv",index=False)
    audit={"status":"V25_2_SLIDE_HIERARCHY_LOSO_AGGREGATION_COMPLETE",
           "all_six_computationally_valid":True,"full_slide_prior":"HalfNormal(0.5)",
           "loso_interpretation":"instability to slide composition; not new-patient external validation",
           "full_vs_current":comp.to_dict(orient="records"),"loso_ranges":rg.to_dict(orient="records")}
    (out/"audit.json").write_text(json.dumps(audit,indent=2))
    print("V25_2_SLIDE_HIERARCHY_LOSO_AGGREGATION_COMPLETE"); print(json.dumps(audit,indent=2))
if __name__=="__main__":
    main()
