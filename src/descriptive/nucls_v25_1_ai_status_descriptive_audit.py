#!/usr/bin/env python3
from __future__ import annotations
from repo_paths import repo_root
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

READERS = ["SP.3","JP.1","JP.2","JP.5","JP.6"]
CONDS = ["Unbiased","Evaluation"]
EXPECTED_ANCHORS = 1144
EXPECTED_AI_POS = 593
EXPECTED_OBS = 10833
EXPECTED_BY_COND = {"Unbiased":5629, "Evaluation":5204}
EXPECTED_AGREE = {"Unbiased":3077, "Evaluation":2848}

def classify_switch(u,e):
    if u==0 and e==1: return "U0_to_E1"
    if u==1 and e==0: return "U1_to_E0"
    if u==1 and e==1: return "both1"
    if u==0 and e==0: return "both0"
    raise RuntimeError((u,e))

def summarize_missing(panel, group):
    g=panel.groupby(group, dropna=False)
    out=g.agg(n_total=("observed","size"), n_observed=("observed","sum")).reset_index()
    out["n_missing"]=out["n_total"]-out["n_observed"]
    out["missing_rate"]=out["n_missing"]/out["n_total"]
    return out

def switch_table(pair, group):
    z=pair.groupby(group+["switch_type"], dropna=False).size().unstack(fill_value=0)
    for c in ["U0_to_E1","U1_to_E0","both1","both0"]:
        if c not in z.columns: z[c]=0
    z=z.reset_index()
    z["paired_n"]=z[["U0_to_E1","U1_to_E0","both1","both0"]].sum(axis=1)
    z["switches"]=z["U0_to_E1"]+z["U1_to_E0"]
    z["switch_rate"]=z["switches"]/z["paired_n"]
    return z[group+["paired_n","U0_to_E1","U1_to_E0","both1","both0","switches","switch_rate"]]

def agreement_table(d, group):
    return (d.groupby(group, dropna=False)
             .agg(n=("detection_agreement","size"),
                  agreements=("detection_agreement","sum"),
                  agreement_rate=("detection_agreement","mean"))
             .reset_index())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",type=Path,default=repo_root())
    ap.add_argument("--outdir",type=Path,default=None)
    args=ap.parse_args()
    root=args.project_root
    raw_path=root/"outputs"/"scoring_v16_1"/"restricted_recorded_reader_detection_agreement_v16.csv"
    ai_path=root/"outputs"/"scoring_v16_1"/"restricted_anchor_matches_v16.csv"
    for p in [raw_path,ai_path]:
        if not p.exists(): raise FileNotFoundError(p)
    raw=pd.read_csv(raw_path)
    ai_tbl=pd.read_csv(ai_path)

    req={"anchor_id","image_id","slide","patient_id","condition","reader",
         "reader_detected","detection_agreement","ai_detected"}
    miss=req-set(raw.columns)
    if miss: raise RuntimeError(f"reader table missing {sorted(miss)}")
    if not {"anchor_id","ai_detected"}.issubset(ai_tbl.columns):
        raise RuntimeError("AI table missing anchor_id/ai_detected")

    d=raw[raw.reader.isin(READERS)].copy()
    d["anchor_id"]=d.anchor_id.astype(str)
    if len(d)!=EXPECTED_OBS: raise RuntimeError(f"Expected {EXPECTED_OBS} observed rows, got {len(d)}")
    if d.anchor_id.nunique()!=EXPECTED_ANCHORS: raise RuntimeError("anchor count mismatch")
    if set(d.condition)!=set(CONDS): raise RuntimeError(f"condition mismatch: {sorted(d.condition.unique())}")
    if set(d.reader)!=set(READERS): raise RuntimeError("reader mismatch")
    if d.duplicated(["anchor_id","reader","condition"]).any():
        raise RuntimeError("duplicate anchor-reader-condition rows")

    # Frozen AI vector.
    a=ai_tbl[["anchor_id","ai_detected"]].copy()
    a["anchor_id"]=a.anchor_id.astype(str)
    if a.anchor_id.duplicated().any():
        if (a.groupby("anchor_id").ai_detected.nunique(dropna=False)>1).any():
            raise RuntimeError("conflicting AI labels")
        a=a.drop_duplicates("anchor_id")
    a["ai_detected"]=a.ai_detected.astype(int)
    if len(a)!=EXPECTED_ANCHORS or int(a.ai_detected.sum())!=EXPECTED_AI_POS:
        raise RuntimeError(f"AI vector validation failed: n={len(a)}, positives={int(a.ai_detected.sum())}")
    ai=a.set_index("anchor_id").ai_detected

    # Cross-source equality.
    x=d[["anchor_id","ai_detected"]].drop_duplicates()
    if x.anchor_id.duplicated().any(): raise RuntimeError("reader table AI inconsistent within anchor")
    xx=x.set_index("anchor_id").ai_detected.astype(int).reindex(ai.index)
    if xx.isna().any() or not np.array_equal(xx.to_numpy(),ai.to_numpy()):
        raise RuntimeError("AI vectors disagree across frozen sources")

    byc=d.groupby("condition").agg(n=("reader_detected","size"),agree=("detection_agreement","sum"))
    for c in CONDS:
        if int(byc.loc[c,"n"])!=EXPECTED_BY_COND[c] or int(byc.loc[c,"agree"])!=EXPECTED_AGREE[c]:
            raise RuntimeError(f"known raw benchmark mismatch for {c}")

    # Unique anchor -> FOV/slide/patient mapping.
    amap=(d.groupby("anchor_id")
           .agg(image_id=("image_id","first"),
                n_image=("image_id","nunique"),
                slide=("slide","first"),
                n_slide=("slide","nunique"),
                patient_id=("patient_id","first"),
                n_patient=("patient_id","nunique"))
          )
    if (amap[["n_image","n_slide","n_patient"]]!=1).any().any():
        raise RuntimeError("anchor hierarchy not unique")
    amap=amap[["image_id","slide","patient_id"]].reset_index()
    amap["ai_detected"]=amap.anchor_id.map(ai).astype(int)

    # Potential 1144 x 5 x 2 panel; observed is exact row presence.
    panel=(amap.assign(_k=1)
           .merge(pd.DataFrame({"reader":READERS,"_k":1}),on="_k")
           .merge(pd.DataFrame({"condition":CONDS,"_k":1}),on="_k")
           .drop(columns="_k"))
    obs=d[["anchor_id","reader","condition","reader_detected","detection_agreement"]].copy()
    obs["observed"]=1
    panel=panel.merge(obs,on=["anchor_id","reader","condition"],how="left")
    panel["observed"]=panel["observed"].fillna(0).astype(int)
    if int(panel["observed"].sum())!=EXPECTED_OBS or len(panel)!=EXPECTED_ANCHORS*len(READERS)*2:
        raise RuntimeError("potential-panel reconstruction failed")

    # Verify missingness is FOV-level in the source reconstruction.
    fv=(panel.groupby(["image_id","reader","condition"])["observed"].nunique().reset_index(name="nunique"))
    n_bad=int((fv["nunique"]>1).sum())
    if n_bad: raise RuntimeError(f"Found {n_bad} reader-FOV-condition cells with anchor-varying observation")

    out=args.outdir or root/"outputs"/"v25_1_phase4_descriptive_audit"
    out.mkdir(parents=True,exist_ok=True)

    summarize_missing(panel,["condition","ai_detected"]).to_csv(out/"missingness_by_condition_ai_status.csv",index=False)
    summarize_missing(panel,["reader","condition","ai_detected"]).to_csv(out/"missingness_by_reader_condition_ai_status.csv",index=False)
    summarize_missing(panel,["slide","condition","ai_detected"]).to_csv(out/"missingness_by_slide_condition_ai_status.csv",index=False)

    # Paired labels and directional switching.
    wide=d.pivot(index=["anchor_id","reader"],columns="condition",values="reader_detected").reset_index()
    wide=wide.dropna(subset=["Unbiased","Evaluation"]).copy()
    wide["Unbiased"]=wide["Unbiased"].astype(int)
    wide["Evaluation"]=wide["Evaluation"].astype(int)
    wide["ai_detected"]=wide.anchor_id.map(ai).astype(int)
    wide=wide.merge(amap[["anchor_id","slide"]],on="anchor_id",how="left")
    wide["switch_type"]=[classify_switch(u,e) for u,e in zip(wide.Unbiased,wide.Evaluation)]
    switch_table(wide,["ai_detected"]).to_csv(out/"directional_switching_by_ai_status.csv",index=False)
    switch_table(wide,["reader","ai_detected"]).to_csv(out/"directional_switching_by_reader_ai_status.csv",index=False)
    switch_table(wide,["slide","ai_detected"]).to_csv(out/"directional_switching_by_slide_ai_status.csv",index=False)

    # Raw agreement strata required for Results Q1 / Figure 2.
    agreement_table(d,["condition","ai_detected"]).to_csv(out/"raw_agreement_by_condition_ai_status.csv",index=False)
    agreement_table(d,["reader","condition","ai_detected"]).to_csv(out/"raw_agreement_by_reader_condition_ai_status.csv",index=False)
    agreement_table(d,["slide","condition","ai_detected"]).to_csv(out/"raw_agreement_by_slide_condition_ai_status.csv",index=False)

    audit={
        "status":"V25_1_PHASE4_AI_STATUS_AUDIT_COMPLETE",
        "n_anchors":EXPECTED_ANCHORS,
        "n_potential_reader_anchor_condition":len(panel),
        "n_observed":int(panel["observed"].sum()),
        "n_missing":int((1-panel["observed"]).sum()),
        "ai_positive":int(ai.sum()),
        "paired_n":int(len(wide)),
        "switches":int((wide.Unbiased!=wide.Evaluation).sum()),
        "fov_level_mask_violations":n_bad,
    }
    (out/"audit.json").write_text(json.dumps(audit,indent=2))
    print("V25_1_PHASE4_AI_STATUS_AUDIT_COMPLETE")
    print(json.dumps(audit,indent=2))

if __name__=="__main__":
    main()
