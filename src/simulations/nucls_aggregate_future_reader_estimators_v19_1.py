#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

METHODS=["M0_majority_consensus","M1_pooled_reader_anchor_beta","M2_reader_primary_paired_bayes_t"]
TARGETS=["theta_U_selected","theta_E_selected","delta_selected"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-dir",type=Path,required=True)
    ap.add_argument("--scenario-grid",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args()

    grid=pd.read_csv(args.scenario_grid).sort_values("scenario_id")
    frames=[]; problems=[]
    for sid in grid.scenario_id.astype(int):
        p=args.input_dir/f"scenario_{sid:02d}.csv"
        if not p.exists():
            problems.append(f"missing {p}"); continue
        x=pd.read_csv(p)
        if len(x)!=900:
            problems.append(f"{p}: rows={len(x)}, expected=900"); continue
        frames.append(x)
    if problems:
        print("V19_1_AGGREGATION_BLOCKED")
        print("\n".join(problems))
        raise SystemExit(2)

    df=pd.concat(frames,ignore_index=True)
    summary=[]
    for (sid,scenario,method,target),g in df.groupby(
        ["scenario_id","scenario","method","target"],sort=True
    ):
        finite=g[g.finite.astype(str).str.lower().isin(["true","1"])]
        err=pd.to_numeric(finite.error,errors="coerce")
        cov=finite.covered.astype(str).str.lower().isin(["true","1"])
        summary.append({
          "scenario_id":int(sid),"scenario":scenario,"method":method,"target":target,
          "n_reps":len(g),"finite_reps":len(finite),"finite_rate":len(finite)/len(g),
          "bias":float(err.mean()) if len(err) else np.nan,
          "rmse":float(np.sqrt(np.mean(err**2))) if len(err) else np.nan,
          "coverage_95":float(cov.mean()) if len(cov) else np.nan,
          "mean_interval_width":float(pd.to_numeric(finite.interval_width,errors="coerce").mean()) if len(finite) else np.nan,
          "mean_truth":float(pd.to_numeric(g.truth,errors="coerce").mean()),
          "mean_estimate":float(pd.to_numeric(finite.estimate,errors="coerce").mean()) if len(finite) else np.nan,
          "mean_paired_readers_used":float(pd.to_numeric(finite.paired_readers_used,errors="coerce").mean())
             if finite.paired_readers_used.notna().any() else np.nan,
        })
    s=pd.DataFrame(summary)

    checks={}
    base=s[(s.scenario_id==0)&(s.method=="M2_reader_primary_paired_bayes_t")]
    for target in TARGETS:
        z=base[base.target==target]
        if len(z)!=1:
            checks[target]={"pass":False,"reason":"missing baseline summary"}; continue
        r=z.iloc[0]
        checks[target]={
          "bias":float(r.bias),"abs_bias_limit":0.015,
          "coverage_95":float(r.coverage_95),"coverage_min":0.90,
          "mean_interval_width":float(r.mean_interval_width),"width_max":0.08,
          "finite_rate":float(r.finite_rate),"finite_rate_required":1.0,
          "pass":bool(abs(r.bias)<=0.015 and r.coverage_95>=0.90
                      and r.mean_interval_width<=0.08 and r.finite_rate==1.0)
        }
    passed=all(v.get("pass",False) for v in checks.values())

    args.output_dir.mkdir(parents=True,exist_ok=True)
    df.to_csv(args.output_dir/"v19_1_all_replicates.csv",index=False)
    s.to_csv(args.output_dir/"v19_1_method_performance_summary.csv",index=False)

    # compact scenario comparison
    comparisons=[]
    for sid in sorted(s.scenario_id.unique()):
        sc=str(s[s.scenario_id==sid].scenario.iloc[0])
        for target in TARGETS:
            d={"scenario_id":int(sid),"scenario":sc,"target":target,"methods":{}}
            for m in METHODS:
                z=s[(s.scenario_id==sid)&(s.target==target)&(s.method==m)]
                if len(z)==1:
                    r=z.iloc[0]
                    d["methods"][m]={
                      "bias":float(r.bias),"rmse":float(r.rmse),
                      "coverage_95":float(r.coverage_95),
                      "mean_interval_width":float(r.mean_interval_width),
                    }
            comparisons.append(d)

    oracle=(df[["scenario_id","scenario","rep","selection_gap_U","selection_gap_E",
                "truth_theta_U_population","truth_theta_E_population","truth_delta_population"]]
            .drop_duplicates(["scenario_id","rep"]))
    oracle_s=(oracle.groupby(["scenario_id","scenario"],sort=True)
              .agg(selection_gap_U_mean=("selection_gap_U","mean"),
                   selection_gap_E_mean=("selection_gap_E","mean"),
                   theta_U_population_mean=("truth_theta_U_population","mean"),
                   theta_E_population_mean=("truth_theta_E_population","mean"),
                   delta_population_mean=("truth_delta_population","mean"))
              .reset_index())
    oracle_s.to_csv(args.output_dir/"v19_1_oracle_population_gap_summary.csv",index=False)

    report={
      "status":"PASS_V19_1_READER_PRIMARY_CALIBRATION" if passed else "REVIEW_V19_1_BEFORE_REAL_DATA_FIT",
      "baseline_M2_checks":checks,
      "calibration_pass":passed,
      "n_scenarios":int(s.scenario_id.nunique()),
      "replicates_per_scenario":100,
      "methods":METHODS,
      "targets":TARGETS,
      "comparisons":comparisons,
      "scientific_boundary":[
        "M2 treats the paired readers, not individual reader-anchor rows, as the inferential sampling units.",
        "All methods see only S=1 anchors.",
        "No method receives unselected candidate labels, latent detectability, or biological truth.",
        "Whole-population truth is an oracle diagnostic only.",
        "Passing v19.1 is evidence of simulation calibration, not methodological novelty.",
        "No NuCLS real-data Bayesian fit is performed by v19.1."
      ],
      "next_gate":(
        "If M2 passes, inspect all stress scenarios and compare M0/M1/M2. "
        "Then audit reader-by-slide coverage before fitting the real-data reader-primary model, "
        "and perform a targeted literature/novelty review."
      )
    }
    (args.output_dir/"v19_1_estimator_gate.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
