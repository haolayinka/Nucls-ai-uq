#!/usr/bin/env python3
from repo_paths import repo_root
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

root = repo_root()
out = root / "outputs" / "v25_5_loso_predictive"

ai_tbl = pd.read_csv(root / "outputs" / "scoring_v16_1" / "restricted_anchor_matches_v16.csv")
ai_tbl["anchor_id"] = ai_tbl["anchor_id"].astype(str)
chk = ai_tbl.groupby("anchor_id")["ai_detected"].nunique(dropna=False)
if (chk > 1).any():
    raise RuntimeError("Conflicting AI values")
ai = ai_tbl.drop_duplicates("anchor_id").set_index("anchor_id")["ai_detected"].astype(int)

rows = []
for k in range(5):
    for model, stem in [("joint", "human_joint"), ("reader_only", "human_reader_only")]:
        f = out / f"{stem}_slide{k}_predictions.csv"
        d = pd.read_csv(f)
        d["anchor_id"] = d["anchor_id"].astype(str)
        d["ai_detected"] = d["anchor_id"].map(ai)
        if d["ai_detected"].isna().any():
            raise RuntimeError(f"Missing AI label in {f}")
        d["actual_agreement"] = (d["reader_detected"].astype(int) == d["ai_detected"].astype(int)).astype(int)
        a = d["ai_detected"].to_numpy(int)
        ph = d["pred"].to_numpy(float)
        d["agreement_pred"] = a * ph + (1-a) * (1-ph)
        strata = [("ALL", d), ("U", d[d["condition"].eq("Unbiased")]), ("E", d[d["condition"].eq("Evaluation")])]
        for stratum, g in strata:
            y = g["actual_agreement"].to_numpy(int)
            p = g["agreement_pred"].to_numpy(float)
            pc = np.clip(p, 1e-12, 1-1e-12)
            rows.append({
                "fold": k,
                "heldout_slide": str(g["heldout_slide"].iloc[0]),
                "model": model,
                "stratum": stratum,
                "n": len(g),
                "observed_agreement": float(y.mean()),
                "predicted_agreement": float(p.mean()),
                "calibration_mean_error": float(p.mean()-y.mean()),
                "brier": float(np.mean((p-y)**2)),
                "log_loss": float(-np.mean(y*np.log(pc) + (1-y)*np.log(1-pc))),
                "auroc": float(roc_auc_score(y,p)),
                "auprc": float(average_precision_score(y,p)),
            })

res = pd.DataFrame(rows)
res.to_csv(out / "target_aligned_loso_summary.csv", index=False)

allr = res[res["stratum"].eq("ALL")].copy()
wide = allr.pivot(index=["fold","heldout_slide"], columns="model",
                  values=["brier","log_loss","auroc","auprc","predicted_agreement"])
diff = pd.DataFrame(index=wide.index).reset_index()
for m in ["brier","log_loss","auroc","auprc","predicted_agreement"]:
    diff[f"{m}_joint_minus_reader_only"] = (
        wide[(m,"joint")].to_numpy() - wide[(m,"reader_only")].to_numpy()
    )
diff.to_csv(out / "target_aligned_loso_joint_minus_reader.csv", index=False)

print("===== TARGET-ALIGNED LOSO — ALL ROWS =====")
print(allr.to_string(index=False))
print("\n===== JOINT - READER-ONLY BY HELDOUT SLIDE =====")
print(diff.to_string(index=False))
print("\n===== TARGET-ALIGNED LOSO RANGES =====")
print(allr.groupby("model")[["observed_agreement","predicted_agreement","brier","log_loss","auroc","auprc"]].agg(["min","max"]).to_string())
print("\nTARGET_ALIGNED_LOSO_POSTPROCESS_COMPLETE")
