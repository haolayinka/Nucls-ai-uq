#!/usr/bin/env python3
"""
Aggregate/validate all 52 blinded v15 GPU receipts.
No scoring/reference data are accessed.
"""
from repo_paths import repo_root
import csv, hashlib, json
from pathlib import Path

PROJECT=repo_root()
MANIFEST=PROJECT/"data/blinded/nucls_gpu_production_manifest_v15.csv"
OUTDIR=PROJECT/"outputs/blinded_inference_v15"
RECDIR=PROJECT/"outputs/receipts_v15"
REPDIR=PROJECT/"reports/production_v15"
EXPECTED_CKPT_SHA="5d1191d6bf72a077911aef12a75484ee3cab91ed81e7880c0361157094ff451d"
EXPECTED_RUNTIME="L40S_PYTORCH_2.10.0_CU126"
EXPECTED_STATUS="COMPLETE_BLINDED_GPU_PRODUCTION_V15"

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

manifest=list(csv.DictReader(MANIFEST.open()))
assert len(manifest)==52
assert len({int(r["fov_id"]) for r in manifest})==52

problems=[]
records=[]
all_predictions=[]
for r in manifest:
    fid=int(r["fov_id"])
    cp=OUTDIR/f"FOV{fid:02d}_instances.csv"
    rp=RECDIR/f"FOV{fid:02d}_receipt.json"
    if not cp.exists() or not rp.exists():
        problems.append(f"FOV{fid:02d}: missing CSV or receipt")
        continue
    try:
        rec=json.loads(rp.read_text())
        preds=list(csv.DictReader(cp.open()))
    except Exception as e:
        problems.append(f"FOV{fid:02d}: unreadable output: {e}")
        continue

    checks=[
        (rec.get("status")==EXPECTED_STATUS,"bad status"),
        (int(rec.get("fov_id",-1))==fid,"receipt fov mismatch"),
        (rec.get("image_sha256")==r["image_sha256"],"image hash mismatch"),
        (rec.get("checkpoint_sha256")==EXPECTED_CKPT_SHA,"checkpoint hash mismatch"),
        (rec.get("production_runtime")==EXPECTED_RUNTIME,"runtime mismatch"),
        ("L40S" in str(rec.get("gpu_name","")).upper(),"non-L40S GPU"),
        (not str(rec.get("node","")).split(".")[0].startswith("__LOCAL_FORBIDDEN_HOST__"),"forbidden local host"),
        (rec.get("reference_data_accessed") is False,"reference data accessed flag"),
        (rec.get("scoring_key_opened") is False,"scoring key flag"),
        (rec.get("reader_labels_opened") is False,"reader labels flag"),
        (rec.get("ai_accuracy_estimated") is False,"accuracy flag"),
        (rec.get("output_csv_sha256")==sha256(cp),"output CSV hash mismatch"),
        (int(rec.get("num_instances",-1))==len(preds),"instance count mismatch"),
    ]
    for ok,msg in checks:
        if not ok:
            problems.append(f"FOV{fid:02d}: {msg}")

    for p in preds:
        if p["image_sha256"]!=r["image_sha256"]:
            problems.append(f"FOV{fid:02d}: prediction image hash mismatch")
            break
        if p["checkpoint_sha256"]!=EXPECTED_CKPT_SHA:
            problems.append(f"FOV{fid:02d}: prediction checkpoint hash mismatch")
            break
        if p["detection_score"]!="NOT_AVAILABLE":
            problems.append(f"FOV{fid:02d}: unexpected detection score")
            break

    records.append(rec)
    all_predictions.extend(preds)

by_slide={}
for rec in records:
    s=rec["slide"]
    x=by_slide.setdefault(s,{"fovs":0,"instances":0})
    x["fovs"]+=1
    x["instances"]+=int(rec["num_instances"])

summary={
    "status":"PASS_ALL_52_BLINDED_L40S_PRODUCTION_RECEIPTS" if not problems and len(records)==52 else "FAIL_OR_INCOMPLETE",
    "n_expected_fovs":52,
    "n_valid_receipts":len(records),
    "n_problems":len(problems),
    "problems":problems,
    "total_instances":sum(int(r["num_instances"]) for r in records),
    "total_prediction_rows":len(all_predictions),
    "slides":by_slide,
    "nodes_used":sorted(set(r["node"] for r in records)),
    "gpu_names_used":sorted(set(r["gpu_name"] for r in records)),
    "torch_versions":sorted(set(r["torch_version"] for r in records)),
    "torch_cuda_versions":sorted(set(str(r["torch_cuda"]) for r in records)),
    "cudnn_versions":sorted(set(str(r["cudnn_version"]) for r in records)),
    "checkpoint_sha256":EXPECTED_CKPT_SHA,
    "production_runtime":EXPECTED_RUNTIME,
    "reference_data_accessed":False,
    "scoring_key_opened":False,
    "reader_labels_opened":False,
    "ai_accuracy_estimated":False,
    "next_gate":"Restricted selected-anchor scoring may be unlocked only if status PASS.",
}
REPDIR.mkdir(parents=True,exist_ok=True)
summary_path=REPDIR/"nucls_gpu_production_receipt_audit_v15.json"
summary_path.write_text(json.dumps(summary,indent=2)+"\n")

# Aggregate predictions only when all receipt checks pass.
if summary["status"]=="PASS_ALL_52_BLINDED_L40S_PRODUCTION_RECEIPTS":
    fields=[
        "image_id","fov_id","prediction_id","xmin_px","ymin_px","xmax_px","ymax_px",
        "detection_score","predicted_class","image_sha256","checkpoint_sha256",
        "production_runtime","prediction_state"
    ]
    agg_path=OUTDIR/"nucls_all52_blinded_L40S_predictions_v15.csv"
    with agg_path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(all_predictions)
    summary["aggregate_predictions_csv"]=str(agg_path)
    summary["aggregate_predictions_sha256"]=sha256(agg_path)
    summary_path.write_text(json.dumps(summary,indent=2)+"\n")

print(json.dumps(summary,indent=2))
if summary["status"]!="PASS_ALL_52_BLINDED_L40S_PRODUCTION_RECEIPTS":
    raise SystemExit(3)
