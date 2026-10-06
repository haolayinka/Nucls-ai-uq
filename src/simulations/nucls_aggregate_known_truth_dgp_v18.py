#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

TARGETS={
    "n_selected":(800,1500),
    "selected_ai_detection_rate":(0.48,0.56),
    "paired_reader_condition_switch_rate":(0.17,0.23),
    "paired_ai_agreement_U":(0.50,0.58),
    "paired_ai_agreement_E":(0.50,0.58),
    "paired_ai_agreement_difference_E_minus_U":(-0.02,0.02),
    "paired_reader_detection_U":(0.77,0.89),
    "paired_reader_detection_E":(0.76,0.88),
}
SUMMARY_COLS=[
    "n_selected","selection_rate","selected_mean_detectability",
    "selected_ai_detection_rate","paired_reader_anchor_observations",
    "paired_reader_detection_U","paired_reader_detection_E",
    "paired_reader_condition_switch_rate","paired_ai_agreement_U",
    "paired_ai_agreement_E","paired_ai_agreement_difference_E_minus_U",
    "equal_reader_mean_agreement_difference_E_minus_U",
    "min_reader_paired_coverage","max_reader_paired_coverage",
    "oracle_theta_selected_U","oracle_theta_selected_E",
    "oracle_delta_selected_E_minus_U","oracle_theta_population_U",
    "oracle_theta_population_E","oracle_delta_population_E_minus_U",
    "selection_gap_U","selection_gap_E",
]

def summarize(g):
    out={"n_reps":len(g)}
    for c in SUMMARY_COLS:
        x=pd.to_numeric(g[c],errors="coerce").dropna()
        out[c+"_mean"]=float(x.mean())
        out[c+"_sd"]=float(x.std(ddof=1)) if len(x)>1 else 0.0
        out[c+"_q025"]=float(x.quantile(.025))
        out[c+"_q50"]=float(x.quantile(.5))
        out[c+"_q975"]=float(x.quantile(.975))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-dir",type=Path,required=True)
    ap.add_argument("--scenario-grid",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args()

    grid=pd.read_csv(args.scenario_grid).sort_values("scenario_id")
    frames=[]
    missing=[]
    for _,r in grid.iterrows():
        p=args.input_dir/f"scenario_{int(r.scenario_id):02d}.csv"
        if not p.exists():
            missing.append(str(p))
            continue
        x=pd.read_csv(p)
        if len(x)==0:
            missing.append(str(p)+" [EMPTY]")
            continue
        frames.append(x)
    if missing:
        print("V18_AGGREGATION_BLOCKED missing scenario outputs:")
        for m in missing: print(m)
        raise SystemExit(2)

    all_df=pd.concat(frames,ignore_index=True)
    rows=[]
    for (sid,name),g in all_df.groupby(["scenario_id","scenario"],sort=True):
        d={"scenario_id":int(sid),"scenario":name}
        d.update(summarize(g))
        rows.append(d)
    summary=pd.DataFrame(rows)

    baseline=summary[summary.scenario_id==0].iloc[0]
    checks={}
    for metric,(lo,hi) in TARGETS.items():
        val=float(baseline[metric+"_mean"])
        checks[metric]={"mean":val,"band":[lo,hi],"pass":bool(lo<=val<=hi)}

    # Informative-selection requirement: selected-vs-population truth should differ
    # by at least 0.02 in magnitude in at least one condition.
    gap_u=float(baseline["selection_gap_U_mean"])
    gap_e=float(baseline["selection_gap_E_mean"])
    gap_pass=max(abs(gap_u),abs(gap_e))>=0.02
    checks["informative_selection_gap"]={
        "selection_gap_U_mean":gap_u,
        "selection_gap_E_mean":gap_e,
        "criterion":"max absolute gap >= 0.02",
        "pass":bool(gap_pass),
    }

    passed=all(v["pass"] for v in checks.values())
    status=("PASS_V18_DGP_CALIBRATION_GATE" if passed
            else "REVIEW_V18_DGP_CALIBRATION_BEFORE_ESTIMATOR_STUDY")

    args.output_dir.mkdir(parents=True,exist_ok=True)
    all_df.to_csv(args.output_dir/"v18_all_replicates.csv",index=False)
    summary.to_csv(args.output_dir/"v18_scenario_summary.csv",index=False)
    report={
        "status":status,
        "baseline_calibration_checks":checks,
        "n_scenarios":int(summary.shape[0]),
        "total_replicates":int(all_df.shape[0]),
        "scientific_boundary":[
            "v18 validates the simulation behavior only; it does not validate a Bayesian estimator.",
            "P-truth is not used as biological truth.",
            "The primary future-reader truth is conditional on S=1, the source-selected frame.",
            "Population truth is generated only to quantify selection-induced target differences.",
            "No NuCLS real-data Bayesian result is produced in v18."
        ],
        "next_gate":(
            "If PASS, implement v19 estimator comparison: naive consensus, "
            "hierarchical multirater ignoring selection, and selection-conditioned hierarchical model."
        )
    }
    (args.output_dir/"v18_calibration_gate.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
