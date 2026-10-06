#!/usr/bin/env python3
from pathlib import Path
import json, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def metrics(y,p):
    y=np.asarray(y,int); p=np.asarray(p,float)
    pc=np.clip(p,1e-12,1-1e-12)
    return {
        "n":len(y),
        "prevalence":y.mean(),
        "brier":np.mean((p-y)**2),
        "log_loss":-np.mean(y*np.log(pc)+(1-y)*np.log(1-pc)),
        "auroc":roc_auc_score(y,p),
        "auprc":average_precision_score(y,p),
    }

def boot_anchor_diff(joint, reader, metric, B=5000, seed=231999):
    m=joint.merge(
        reader[["anchor_id","condition","reader","pred"]],
        on=["anchor_id","condition","reader"],suffixes=("_j","_r"),validate="one_to_one"
    )
    y=m.reader_detected.astype(int).to_numpy()
    pj=m.pred_j.to_numpy(); pr=m.pred_r.to_numpy()
    anchors=m.anchor_id.unique()
    by={a:np.flatnonzero(m.anchor_id.to_numpy()==a) for a in anchors}
    rng=np.random.default_rng(seed)
    vals=np.empty(B)
    for b in range(B):
        samp=rng.choice(anchors,size=len(anchors),replace=True)
        idx=np.concatenate([by[a] for a in samp])
        if metric=="brier":
            vals[b]=np.mean((pj[idx]-y[idx])**2)-np.mean((pr[idx]-y[idx])**2)
        else:
            eps=1e-12
            lj=-(y[idx]*np.log(np.clip(pj[idx],eps,1-eps))+(1-y[idx])*np.log(np.clip(1-pj[idx],eps,1-eps)))
            lr=-(y[idx]*np.log(np.clip(pr[idx],eps,1-eps))+(1-y[idx])*np.log(np.clip(1-pr[idx],eps,1-eps)))
            vals[b]=lj.mean()-lr.mean()
    return np.quantile(vals,[.025,.5,.975])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outdir",type=Path,default=Path("outputs/v23_1_shared_detectability_cv"))
    args=ap.parse_args(); o=args.outdir

    expected=[o/f"{mode}_fold{k}_result.json" for mode in ["human_joint","human_reader_only","ai_holdout"] for k in range(5)]
    miss=[str(p) for p in expected if not p.exists()]
    if miss: raise SystemExit("Missing files:\n"+"\n".join(miss))

    j=pd.concat([pd.read_csv(o/f"human_joint_fold{k}_predictions.csv") for k in range(5)],ignore_index=True)
    r=pd.concat([pd.read_csv(o/f"human_reader_only_fold{k}_predictions.csv") for k in range(5)],ignore_index=True)
    a=pd.concat([pd.read_csv(o/f"ai_holdout_fold{k}_predictions.csv") for k in range(5)],ignore_index=True)

    mj=metrics(j.reader_detected,j.pred)
    mr=metrics(r.reader_detected,r.pred)
    ma=metrics(a.ai_detected,a.pred)

    bci=boot_anchor_diff(j,r,"brier")
    lci=boot_anchor_diff(j,r,"logloss")

    # AI prevalence-only pooled baseline using overall prevalence; per-fold baselines
    # are also retained in each result JSON.
    p0=float(a.ai_detected.mean())
    y=a.ai_detected.to_numpy(int)
    base_brier=float(np.mean((p0-y)**2))
    base_log=float(-np.mean(y*np.log(p0)+(1-y)*np.log(1-p0)))

    results=[]
    for name,m in [("joint_predicts_heldout_humans",mj),("reader_only_predicts_heldout_humans",mr),("humans_predict_heldout_ai",ma)]:
        results.append({"analysis":name,**{k:float(v) for k,v in m.items()}})
    summary=pd.DataFrame(results)
    summary.to_csv(o/"shared_detectability_cv_summary.csv",index=False)

    comparison={
        "human_prediction":{
            "joint":mj,"reader_only":mr,
            "joint_minus_reader_only":{
                "brier":mj["brier"]-mr["brier"],
                "log_loss":mj["log_loss"]-mr["log_loss"],
                "auroc":mj["auroc"]-mr["auroc"],
                "auprc":mj["auprc"]-mr["auprc"],
            },
            "anchor_cluster_bootstrap_95":{
                "brier_joint_minus_reader_only":bci.tolist(),
                "logloss_joint_minus_reader_only":lci.tolist(),
            },
        },
        "ai_prediction":{
            "human_latent_joint":ma,
            "prevalence_baseline":{"brier":base_brier,"log_loss":base_log},
            "improvement_vs_prevalence":{
                "brier":base_brier-ma["brier"],
                "log_loss":base_log-ma["log_loss"],
            },
        },
    }
    (o/"shared_detectability_cv_comparison.json").write_text(json.dumps(comparison,indent=2))

    print("\nPOOLED HELD-OUT HUMAN PREDICTION")
    print(summary.iloc[:2].to_string(index=False))
    print("\nJOINT - READER_ONLY (negative Brier/log-loss is better for joint)")
    print(json.dumps(comparison["human_prediction"]["joint_minus_reader_only"],indent=2))
    print("\nAnchor-cluster bootstrap [2.5%, median, 97.5%]")
    print("Brier:",bci)
    print("Log loss:",lci)
    print("\nHELD-OUT AI PREDICTION FROM HUMAN-INFORMED SHARED DETECTABILITY")
    print(summary.iloc[2:].to_string(index=False))
    print("Prevalence baseline:",comparison["ai_prediction"]["prevalence_baseline"])
    print("Improvement:",comparison["ai_prediction"]["improvement_vs_prevalence"])

    # Computational audit
    rows=[]
    for mode in ["human_joint","human_reader_only","ai_holdout"]:
        for k in range(5):
            q=json.loads((o/f"{mode}_fold{k}_result.json").read_text())
            rows.append({x:q[x] for x in ["mode","fold","divergences","max_rhat_scalar","min_ess_bulk_scalar","min_ess_tail_scalar","elapsed_sec"]})
    comp=pd.DataFrame(rows)
    comp.to_csv(o/"shared_detectability_cv_computational_audit.csv",index=False)
    print("\nCOMPUTATIONAL AUDIT")
    print(comp.to_string(index=False))
    print("\nTOTAL divergences:",int(comp.divergences.sum()))
    print("Max Rhat:",comp.max_rhat_scalar.max())
    print("Min bulk ESS:",comp.min_ess_bulk_scalar.min())
    print("Min tail ESS:",comp.min_ess_tail_scalar.min())

if __name__=="__main__":
    main()
