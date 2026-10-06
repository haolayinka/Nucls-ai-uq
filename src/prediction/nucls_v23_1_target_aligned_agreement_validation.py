#!/usr/bin/env python3
from __future__ import annotations
from repo_paths import repo_root
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

B=5000
SEED=232101

def metrics(y,p):
    y=np.asarray(y,int); p=np.asarray(p,float)
    pc=np.clip(p,1e-12,1-1e-12)
    return {
        "n":int(len(y)),
        "observed_mean":float(y.mean()),
        "predicted_mean":float(p.mean()),
        "calibration_mean_error":float(p.mean()-y.mean()),
        "brier":float(np.mean((p-y)**2)),
        "log_loss":float(-np.mean(y*np.log(pc)+(1-y)*np.log(1-pc))),
        "auroc":float(roc_auc_score(y,p)),
        "auprc":float(average_precision_score(y,p)),
    }

def boot_diff(d, metric, B=B, seed=SEED):
    # Joint - Reader-only; negative proper-score difference favors joint.
    anchors=d.anchor_id.unique()
    groups={a:np.flatnonzero(d.anchor_id.to_numpy()==a) for a in anchors}
    y=d.actual_agreement.to_numpy(int)
    pj=d.joint_agree_pred.to_numpy(float)
    pr=d.reader_agree_pred.to_numpy(float)
    rng=np.random.default_rng(seed)
    vals=np.empty(B)
    for b in range(B):
        samp=rng.choice(anchors,size=len(anchors),replace=True)
        idx=np.concatenate([groups[a] for a in samp])
        yy=y[idx]; aj=pj[idx]; ar=pr[idx]
        if metric=="brier":
            vals[b]=np.mean((aj-yy)**2)-np.mean((ar-yy)**2)
        elif metric=="log_loss":
            eps=1e-12
            aj=np.clip(aj,eps,1-eps); ar=np.clip(ar,eps,1-eps)
            lj=-(yy*np.log(aj)+(1-yy)*np.log(1-aj))
            lr=-(yy*np.log(ar)+(1-yy)*np.log(1-ar))
            vals[b]=lj.mean()-lr.mean()
        elif metric=="mean_pred":
            vals[b]=aj.mean()-ar.mean()
        else:
            raise ValueError(metric)
    return np.quantile(vals,[.025,.5,.975])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",type=Path,default=repo_root())
    ap.add_argument("--cv-dir",type=Path,default=None)
    ap.add_argument("--outdir",type=Path,default=None)
    args=ap.parse_args()
    root=args.project_root
    cv=args.cv_dir or root/"outputs"/"v23_1_shared_detectability_cv"
    out=args.outdir or cv
    out.mkdir(parents=True,exist_ok=True)

    js=[]; rs=[]
    for k in range(5):
        jp=cv/f"human_joint_fold{k}_predictions.csv"
        rp=cv/f"human_reader_only_fold{k}_predictions.csv"
        if not jp.exists() or not rp.exists():
            raise FileNotFoundError(f"Missing fold {k} prediction file(s)")
        j=pd.read_csv(jp); r=pd.read_csv(rp)
        j["anchor_id"]=j.anchor_id.astype(str); r["anchor_id"]=r.anchor_id.astype(str)
        keys=["anchor_id","image_id","slide","patient_id","condition","reader","reader_detected"]
        j=j[keys+["pred"]].rename(columns={"pred":"joint_human_pred"})
        r=r[keys+["pred"]].rename(columns={"pred":"reader_human_pred"})
        j["fold"]=k; r["fold"]=k
        js.append(j); rs.append(r)

    j=pd.concat(js,ignore_index=True)
    r=pd.concat(rs,ignore_index=True)

    keys=["anchor_id","image_id","slide","patient_id","condition","reader","reader_detected","fold"]
    d=j.merge(r,on=keys,validate="one_to_one")

    ai_tbl=pd.read_csv(root/"outputs"/"scoring_v16_1"/"restricted_anchor_matches_v16.csv")
    ai_tbl["anchor_id"]=ai_tbl.anchor_id.astype(str)
    if ai_tbl.anchor_id.duplicated().any():
        chk=ai_tbl.groupby("anchor_id").ai_detected.nunique(dropna=False)
        if (chk>1).any(): raise RuntimeError("Conflicting AI labels")
        ai_tbl=ai_tbl.drop_duplicates("anchor_id")
    ai=ai_tbl.set_index("anchor_id").ai_detected.astype(int)
    d["ai_detected"]=d.anchor_id.map(ai)
    if d.ai_detected.isna().any(): raise RuntimeError("Missing AI labels")
    d["ai_detected"]=d.ai_detected.astype(int)

    d["actual_agreement"]=(d.reader_detected.astype(int)==d.ai_detected).astype(int)
    A=d.ai_detected.to_numpy(float)
    d["joint_agree_pred"]=A*d.joint_human_pred+(1-A)*(1-d.joint_human_pred)
    d["reader_agree_pred"]=A*d.reader_human_pred+(1-A)*(1-d.reader_human_pred)

    overall={
        "joint":metrics(d.actual_agreement,d.joint_agree_pred),
        "reader_only":metrics(d.actual_agreement,d.reader_agree_pred),
    }
    diffs={
        "brier_joint_minus_reader":float(overall["joint"]["brier"]-overall["reader_only"]["brier"]),
        "logloss_joint_minus_reader":float(overall["joint"]["log_loss"]-overall["reader_only"]["log_loss"]),
        "auroc_joint_minus_reader":float(overall["joint"]["auroc"]-overall["reader_only"]["auroc"]),
        "auprc_joint_minus_reader":float(overall["joint"]["auprc"]-overall["reader_only"]["auprc"]),
        "mean_pred_joint_minus_reader":float(overall["joint"]["predicted_mean"]-overall["reader_only"]["predicted_mean"]),
    }
    boot={
        "brier_joint_minus_reader":boot_diff(d,"brier").tolist(),
        "logloss_joint_minus_reader":boot_diff(d,"log_loss").tolist(),
        "mean_pred_joint_minus_reader":boot_diff(d,"mean_pred").tolist(),
    }

    strata=[]
    for cols in [[],["condition"],["ai_detected"],["condition","ai_detected"]]:
        if not cols:
            groups=[("ALL",d)]
        else:
            groups=list(d.groupby(cols,dropna=False))
        for key,g in groups:
            if isinstance(key,tuple): label=" | ".join(f"{c}={v}" for c,v in zip(cols,key))
            elif cols: label=f"{cols[0]}={key}"
            else: label="ALL"
            mj=metrics(g.actual_agreement,g.joint_agree_pred)
            mr=metrics(g.actual_agreement,g.reader_agree_pred)
            strata.append({
                "stratum":label,
                "n":len(g),
                "observed_agreement":mj["observed_mean"],
                "joint_pred_agreement":mj["predicted_mean"],
                "reader_pred_agreement":mr["predicted_mean"],
                "joint_calibration_error":mj["calibration_mean_error"],
                "reader_calibration_error":mr["calibration_mean_error"],
                "joint_brier":mj["brier"],
                "reader_brier":mr["brier"],
            })

    pd.DataFrame(strata).to_csv(out/"target_aligned_agreement_strata.csv",index=False)
    d.to_csv(out/"target_aligned_agreement_predictions.csv",index=False)
    result={"overall":overall,"joint_minus_reader":diffs,"bootstrap_95":boot}
    (out/"target_aligned_agreement_comparison.json").write_text(json.dumps(result,indent=2))

    print("TARGET-ALIGNED HELD-OUT AI-READER AGREEMENT VALIDATION")
    print("\nOVERALL")
    print(pd.DataFrame([
        {"model":"joint",**overall["joint"]},
        {"model":"reader_only",**overall["reader_only"]},
    ]).to_string(index=False))
    print("\nJOINT - READER_ONLY")
    print(json.dumps(diffs,indent=2))
    print("\nANCHOR-CLUSTER BOOTSTRAP [2.5%, median, 97.5%]")
    print(json.dumps(boot,indent=2))
    print("\nCALIBRATION / STRATA")
    print(pd.DataFrame(strata).to_string(index=False))
    print("\nInterpretation: negative Brier/log-loss difference favors joint.")
    print("TARGET_ALIGNED_AGREEMENT_VALIDATION_COMPLETE")

if __name__=="__main__":
    main()
