#!/usr/bin/env python3
"""
NuCLS restricted selected-anchor detection scoring — v16.

Primary analysis follows the detection-only HoVer-Net amendment:
- every native v15 instance is an eligible detection (no score threshold);
- class-agnostic one-to-one matching at bbox IoU >= 0.50;
- deterministic global greedy matching, because v11 explicitly used "greedy";
- no biological false-positive/sensitivity/specificity interpretation.

A prespecified sensitivity analysis also runs the older v8 maximum-cardinality,
maximum-total-IoU assignment and reports whether detection status or assignments differ.

This script may be run only after the v15 blinded L40S receipt audit passes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

EXPECTED_PRED_SHA = "cc6870b201592ec4715afa558ef781a0e59fa4ebc4fb4d974d3c8319ce379f2b"
EXPECTED_CKPT_SHA = "5d1191d6bf72a077911aef12a75484ee3cab91ed81e7880c0361157094ff451d"
EXPECTED_RUNTIME = "L40S_PYTORCH_2.10.0_CU126"
EXPECTED_AUDIT_STATUS = "PASS_ALL_52_BLINDED_L40S_PRODUCTION_RECEIPTS"
IOU_THRESHOLD = 0.50
# Geometry validation tolerance only; does not alter predictions or matching.
BOUNDARY_TOL_PX = 1e-9

READERS = ("SP.1","SP.2","SP.3","JP.1","JP.2","JP.3","JP.4","JP.5","JP.6")
MISSING_FOV = "DidNotAnnotateFOV"
UNDETECTED = "undetected"

PRED_BOX = ("xmin_px","ymin_px","xmax_px","ymax_px")
REF_BOX = ("contour_xmin_px","contour_ymin_px","contour_xmax_px","contour_ymax_px")

class ProtocolError(RuntimeError):
    pass

def require(ok, msg):
    if not ok:
        raise ProtocolError(msg)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def validate_gate(audit_path: Path, predictions_path: Path, checkpoint_path: Path):
    require(audit_path.exists(), f"Missing v15 audit: {audit_path}")
    audit = json.loads(audit_path.read_text())
    require(audit.get("status") == EXPECTED_AUDIT_STATUS,
            f"v15 audit did not pass: {audit.get('status')}")
    require(int(audit.get("n_expected_fovs",-1)) == 52, "Audit expected FOV count != 52")
    require(int(audit.get("n_valid_receipts",-1)) == 52, "Audit valid receipt count != 52")
    require(int(audit.get("n_problems",-1)) == 0 and audit.get("problems") == [],
            "Audit contains problems")
    require(int(audit.get("total_prediction_rows",-1)) == 1483,
            "Unexpected v15 aggregate prediction-row count")
    require(audit.get("checkpoint_sha256") == EXPECTED_CKPT_SHA,
            "Audit checkpoint SHA mismatch")
    require(audit.get("production_runtime") == EXPECTED_RUNTIME,
            "Audit production runtime mismatch")
    require(audit.get("gpu_names_used") == ["NVIDIA L40S"],
            "Audit includes a non-L40S GPU")
    require(audit.get("reference_data_accessed") is False, "Audit says reference data accessed")
    require(audit.get("scoring_key_opened") is False, "Audit says scoring key was opened")
    require(audit.get("reader_labels_opened") is False, "Audit says reader labels were opened")
    require(audit.get("ai_accuracy_estimated") is False, "Audit says accuracy was estimated")
    require(audit.get("aggregate_predictions_sha256") == EXPECTED_PRED_SHA,
            "Audit aggregate prediction SHA differs from frozen v15 SHA")
    require(predictions_path.exists(), "Aggregate v15 predictions file missing")
    actual_pred_sha = sha256(predictions_path)
    require(actual_pred_sha == EXPECTED_PRED_SHA,
            f"Prediction SHA mismatch: {actual_pred_sha}")
    require(checkpoint_path.exists(), "Checkpoint file missing")
    actual_ckpt_sha = sha256(checkpoint_path)
    require(actual_ckpt_sha == EXPECTED_CKPT_SHA,
            f"Checkpoint SHA mismatch: {actual_ckpt_sha}")
    return audit

def load_validate_inputs(predictions_path, manifest_path, key_path, e_path, u_path):
    pred = pd.read_csv(predictions_path, dtype={"image_id":str,"prediction_id":str})
    images = pd.read_csv(manifest_path)
    key = pd.read_csv(key_path)
    e_ref = pd.read_csv(e_path, low_memory=False)
    u_ref = pd.read_csv(u_path, low_memory=False)

    require(len(images) == 52 and images.image_id.is_unique,
            "Expected 52 distinct image-manifest rows")
    require(images.slide.nunique() == 5 and images.patient_id.nunique() == 5,
            "Expected five source slides/patients")
    require((images.image_geometry == "SOURCE_EXPORT_CONTOUR_TO_RGB_REGISTERED").all(),
            "Image geometry is not source-export registered")
    require(images.image_sha256.str.fullmatch(r"[0-9a-f]{64}").all(),
            "Invalid image SHA in manifest")

    require(len(key) == 1155 and key.anchor_id.is_unique,
            "Expected 1,155 distinct shared selected anchors")
    # Robust bool parsing in case pandas did not infer bool.
    if key.paired_anchor_rgb_available.dtype != bool:
        key["paired_anchor_rgb_available"] = (
            key.paired_anchor_rgb_available.astype(str).str.lower().map({"true":True,"false":False})
        )
    require(key.paired_anchor_rgb_available.notna().all(),
            "Could not parse paired_anchor_rgb_available")
    eligible = key[key.paired_anchor_rgb_available].copy()
    require(len(eligible) == 1144, "Expected 1,144 RGB-eligible selected anchors")
    require(eligible.anchor_id.is_unique, "Eligible anchor IDs duplicated")
    require(set(eligible.image_id) == set(images.image_id),
            "Eligible anchors do not cover exactly the 52 inference FOVs")
    require((eligible.coordinate_registration_status == "SOURCE_EXPORT_PIXEL_BOX_VERIFIED").all(),
            "Eligible anchor has unverified coordinate registration")

    merged = eligible.merge(
        images[["image_id","slide","patient_id","width_px","height_px","image_sha256"]],
        on="image_id", how="left", validate="many_to_one", suffixes=("","_image")
    )
    require((merged.slide == merged.slide_image).all(), "Anchor/image slide mismatch")
    require((merged.patient_id == merged.patient_id_image).all(), "Anchor/image patient mismatch")
    rb = merged.loc[:,REF_BOX].apply(pd.to_numeric,errors="coerce")
    require(np.isfinite(rb.to_numpy()).all(), "Nonfinite reference boxes")
    require((rb.contour_xmin_px >= -BOUNDARY_TOL_PX).all() and
            (rb.contour_ymin_px >= -BOUNDARY_TOL_PX).all(),
            "Negative reference coordinates beyond numerical tolerance")
    require((rb.contour_xmax_px > rb.contour_xmin_px).all() and
            (rb.contour_ymax_px > rb.contour_ymin_px).all(),
            "Invalid reference box area")
    require((rb.contour_xmax_px <= merged.width_px + BOUNDARY_TOL_PX).all() and
            (rb.contour_ymax_px <= merged.height_px + BOUNDARY_TOL_PX).all(),
            "Reference box outside exported RGB beyond numerical tolerance")

    required_pred = {
        "image_id","fov_id","prediction_id","xmin_px","ymin_px","xmax_px","ymax_px",
        "detection_score","predicted_class","image_sha256","checkpoint_sha256",
        "production_runtime","prediction_state"
    }
    require(required_pred.issubset(pred.columns), "v15 prediction schema incomplete")
    require(len(pred) == 1483, "Expected exactly 1,483 v15 prediction rows")
    require(pred.prediction_id.is_unique, "Prediction IDs are not globally unique")
    require(set(pred.image_id).issubset(set(images.image_id)), "Unknown prediction image_id")
    require(pred.checkpoint_sha256.eq(EXPECTED_CKPT_SHA).all(), "Prediction checkpoint SHA mismatch")
    require(pred.production_runtime.eq(EXPECTED_RUNTIME).all(), "Prediction runtime mismatch")
    require(pred.detection_score.astype(str).eq("NOT_AVAILABLE").all(),
            "Detection-only run contains an unexpected detection score")
    require(pred.predicted_class.astype(str).eq("NOT_SCORED_DETECTION_ONLY").all(),
            "Detection-only run contains a scored class")
    require(pred.prediction_state.eq("COMPLETE_BLINDED_GPU_PRODUCTION_V15").all(),
            "Unexpected v15 prediction state")

    p = pred.merge(
        images[["image_id","fov_id","width_px","height_px","image_sha256"]],
        on="image_id", how="left", validate="many_to_one",
        suffixes=("","_manifest")
    )
    require(pd.to_numeric(p.fov_id,errors="coerce").eq(
            pd.to_numeric(p.fov_id_manifest,errors="coerce")).all(),
            "Prediction FOV ID mismatch")
    require(p.image_sha256.eq(p.image_sha256_manifest).all(),
            "Prediction image SHA mismatch")
    pb = p.loc[:,PRED_BOX].apply(pd.to_numeric,errors="coerce")
    require(np.isfinite(pb.to_numpy()).all(), "Nonfinite prediction boxes")
    require((pb.xmin_px >= -BOUNDARY_TOL_PX).all() and
            (pb.ymin_px >= -BOUNDARY_TOL_PX).all(),
            "Negative prediction coordinates beyond numerical tolerance")
    require((pb.xmax_px > pb.xmin_px).all() and
            (pb.ymax_px > pb.ymin_px).all(),
            "Invalid prediction box area")
    require((pb.xmax_px <= p.width_px + BOUNDARY_TOL_PX).all() and
            (pb.ymax_px <= p.height_px + BOUNDARY_TOL_PX).all(),
            "Prediction box outside exported RGB beyond numerical tolerance")
    for c in PRED_BOX:
        pred[c] = pd.to_numeric(pred[c],errors="raise")

    for name, ref in (("Evaluation",e_ref),("Unbiased",u_ref)):
        require("anchor_id" in ref.columns, f"{name} source missing anchor_id")
        require(ref.anchor_id.is_unique, f"{name} source anchor IDs duplicated")
        require(set(eligible.anchor_id).issubset(set(ref.anchor_id)),
                f"{name} source lacks selected anchors")
        require(set(READERS).issubset(ref.columns), f"{name} source reader columns incomplete")
        sub = ref.loc[ref.anchor_id.isin(eligible.anchor_id), READERS]
        require(sub.notna().all().all(),
                f"{name} reader entries contain NA; DidNotAnnotateFOV must be explicit")

    return pred, images, eligible, e_ref, u_ref

def iou_matrix(ref, pred):
    if len(ref) == 0 or len(pred) == 0:
        return np.zeros((len(ref),len(pred)),dtype=float)
    a = ref.loc[:,REF_BOX].to_numpy(float)
    b = pred.loc[:,PRED_BOX].to_numpy(float)
    x0 = np.maximum(a[:,None,0], b[None,:,0])
    y0 = np.maximum(a[:,None,1], b[None,:,1])
    x1 = np.minimum(a[:,None,2], b[None,:,2])
    y1 = np.minimum(a[:,None,3], b[None,:,3])
    inter = np.maximum(0,x1-x0) * np.maximum(0,y1-y0)
    aa = (a[:,2]-a[:,0]) * (a[:,3]-a[:,1])
    bb = (b[:,2]-b[:,0]) * (b[:,3]-b[:,1])
    den = aa[:,None] + bb[None,:] - inter
    return np.divide(inter, den, out=np.zeros_like(inter), where=den>0)

def match_greedy_one_fov(ref, pred, threshold=IOU_THRESHOLD):
    """
    v11-primary deterministic global greedy rule:
      enumerate all valid pairs, sort by (-IoU, anchor_id, prediction_id),
      take a pair iff neither endpoint has been used.
    """
    a = ref.sort_values("anchor_id").reset_index(drop=True)
    p = pred.sort_values("prediction_id").reset_index(drop=True)
    out = a[["anchor_id","image_id","slide","patient_id"]].copy()
    out["prediction_id"] = ""
    out["matched_iou"] = np.nan
    out["ai_detected"] = False
    if len(a)==0 or len(p)==0:
        return out, set()

    M = iou_matrix(a,p)
    candidates = []
    for i,j in zip(*np.where(M >= threshold)):
        candidates.append((-float(M[i,j]), str(a.loc[i,"anchor_id"]),
                           str(p.loc[j,"prediction_id"]), int(i), int(j)))
    candidates.sort()

    used_a=set(); used_p=set()
    for neg_iou, _, _, i, j in candidates:
        if i in used_a or j in used_p:
            continue
        used_a.add(i); used_p.add(j)
        out.loc[i,"prediction_id"] = str(p.loc[j,"prediction_id"])
        out.loc[i,"matched_iou"] = -neg_iou
        out.loc[i,"ai_detected"] = True
    return out, used_p

def match_hungarian_one_fov(ref, pred, threshold=IOU_THRESHOLD):
    """v8 sensitivity: maximum-cardinality, then maximum-total-IoU."""
    a = ref.sort_values("anchor_id").reset_index(drop=True)
    p = pred.sort_values("prediction_id").reset_index(drop=True)
    out = a[["anchor_id","image_id","slide","patient_id"]].copy()
    out["prediction_id"] = ""
    out["matched_iou"] = np.nan
    out["ai_detected"] = False
    if len(a)==0 or len(p)==0:
        return out, set()

    M=iou_matrix(a,p)
    valid=M>=threshold
    # Any additional valid edge is worth more than the maximum possible total IoU change,
    # so cardinality is optimized first and total IoU second.
    reward = 1 + max(len(a),len(p))
    weights=np.where(valid,reward+M,0.0)
    rr,cc=linear_sum_assignment(-weights)
    used=set()
    for i,j in zip(rr,cc):
        if not valid[i,j]:
            continue
        out.loc[i,"prediction_id"]=str(p.loc[j,"prediction_id"])
        out.loc[i,"matched_iou"]=float(M[i,j])
        out.loc[i,"ai_detected"]=True
        used.add(int(j))
    return out,used

def match_all(eligible, pred, method):
    links=[]
    extras=[]
    for image_id, ref_fov in eligible.groupby("image_id",sort=True):
        pred_fov=pred[pred.image_id==image_id].copy()
        if method=="greedy":
            m,used=match_greedy_one_fov(ref_fov,pred_fov)
        elif method=="hungarian":
            m,used=match_hungarian_one_fov(ref_fov,pred_fov)
        else:
            raise ValueError(method)
        links.append(m)
        pred_sorted=pred_fov.sort_values("prediction_id").reset_index(drop=True)
        matched_pred_ids=set(pred_sorted.loc[list(used),"prediction_id"]) if used else set()
        for _,r in pred_sorted.iterrows():
            if r.prediction_id not in matched_pred_ids:
                extras.append({
                    "image_id":image_id,
                    "fov_id":int(r.fov_id),
                    "prediction_id":r.prediction_id,
                    "xmin_px":r.xmin_px,"ymin_px":r.ymin_px,
                    "xmax_px":r.xmax_px,"ymax_px":r.ymax_px,
                    "status":"unmatched_prediction_on_selected_anchor_frame",
                    "NOT_biological_false_positive":True,
                })
    return pd.concat(links,ignore_index=True), pd.DataFrame(extras)

def reader_events(links, ref, condition):
    x=links.merge(ref[["anchor_id",*READERS]],on="anchor_id",how="left",validate="one_to_one")
    parts=[]
    for reader in READERS:
        one=x[["anchor_id","image_id","slide","patient_id","ai_detected",reader]].copy()
        one.rename(columns={reader:"reader_label"},inplace=True)
        one=one[one.reader_label != MISSING_FOV].copy()
        one.insert(4,"condition",condition)
        one.insert(5,"reader",reader)
        one["reader_detected"] = one.reader_label != UNDETECTED
        one["detection_agreement"] = one.ai_detected == one.reader_detected
        parts.append(one)
    return pd.concat(parts,ignore_index=True)

def make_summaries(links, extras, events):
    by_slide_anchor=[]
    for slide,g in links.groupby("slide",sort=True):
        by_slide_anchor.append({
            "slide":slide,
            "patient_id":str(g.patient_id.iloc[0]),
            "eligible_selected_anchors":len(g),
            "ai_matched_selected_anchors":int(g.ai_detected.sum()),
            "ai_selected_anchor_match_rate":float(g.ai_detected.mean()),
        })
    by_slide_anchor=pd.DataFrame(by_slide_anchor)

    overall_anchor={
        "eligible_selected_anchors":int(len(links)),
        "ai_matched_selected_anchors":int(links.ai_detected.sum()),
        "anchor_weighted_selected_anchor_match_rate":float(links.ai_detected.mean()),
        "equal_slide_mean_selected_anchor_match_rate":float(
            by_slide_anchor.ai_selected_anchor_match_rate.mean()
        ),
        "total_predictions":int(len(links[links.ai_detected]) + len(extras)),
        "unmatched_predictions_on_selected_anchor_frame":int(len(extras)),
    }

    slide_condition=[]
    for (slide,condition),g in events.groupby(["slide","condition"],sort=True):
        slide_condition.append({
            "slide":slide,
            "condition":condition,
            "observed_reader_anchor_votes":len(g),
            "detection_agreements":int(g.detection_agreement.sum()),
            "detection_agreement_rate_CONDITIONAL_SELECTED":
                float(g.detection_agreement.mean()),
        })
    slide_condition=pd.DataFrame(slide_condition)

    condition_summary=[]
    for condition,g in events.groupby("condition",sort=True):
        ss=slide_condition[slide_condition.condition==condition]
        condition_summary.append({
            "condition":condition,
            "observed_reader_anchor_votes":len(g),
            "detection_agreements":int(g.detection_agreement.sum()),
            "reader_observation_weighted_detection_agreement":
                float(g.detection_agreement.mean()),
            "equal_slide_mean_detection_agreement":
                float(ss.detection_agreement_rate_CONDITIONAL_SELECTED.mean()),
        })
    condition_summary=pd.DataFrame(condition_summary)

    reader_condition=[]
    for (condition,reader),g in events.groupby(["condition","reader"],sort=True):
        slide_rates=(g.groupby("slide",sort=True).detection_agreement.mean())
        reader_condition.append({
            "condition":condition,
            "reader":reader,
            "observed_reader_anchor_votes":len(g),
            "reader_detected_votes":int(g.reader_detected.sum()),
            "detection_agreements":int(g.detection_agreement.sum()),
            "reader_observation_weighted_detection_agreement":
                float(g.detection_agreement.mean()),
            "n_slides_with_observations":int(len(slide_rates)),
            "equal_slide_mean_detection_agreement":
                float(slide_rates.mean()),
        })
    reader_condition=pd.DataFrame(reader_condition)
    return overall_anchor,by_slide_anchor,slide_condition,condition_summary,reader_condition

def compare_methods(primary,sensitivity):
    a=primary[["anchor_id","ai_detected","prediction_id","matched_iou"]].merge(
        sensitivity[["anchor_id","ai_detected","prediction_id","matched_iou"]],
        on="anchor_id",suffixes=("_greedy","_hungarian"),validate="one_to_one"
    )
    detection_changed=a.ai_detected_greedy != a.ai_detected_hungarian
    assignment_changed=(
        a.prediction_id_greedy.fillna("").astype(str) !=
        a.prediction_id_hungarian.fillna("").astype(str)
    )
    return {
        "n_eligible_anchors":int(len(a)),
        "greedy_matched":int(a.ai_detected_greedy.sum()),
        "hungarian_matched":int(a.ai_detected_hungarian.sum()),
        "n_anchor_detection_status_differences":int(detection_changed.sum()),
        "n_prediction_assignment_differences":int(assignment_changed.sum()),
        "detection_status_identical":bool((~detection_changed).all()),
        "assignments_identical":bool((~assignment_changed).all()),
    }, a

def self_test():
    ref=pd.DataFrame([
        dict(anchor_id="A",image_id="I",slide="S",patient_id="P",
             contour_xmin_px=0,contour_ymin_px=0,contour_xmax_px=10,contour_ymax_px=10),
        dict(anchor_id="B",image_id="I",slide="S",patient_id="P",
             contour_xmin_px=20,contour_ymin_px=20,contour_xmax_px=30,contour_ymax_px=30),
    ])
    pred=pd.DataFrame([
        dict(image_id="I",fov_id=0,prediction_id="P1",xmin_px=0,ymin_px=0,xmax_px=10,ymax_px=10),
        dict(image_id="I",fov_id=0,prediction_id="P2",xmin_px=20,ymin_px=20,xmax_px=30,ymax_px=30),
        dict(image_id="I",fov_id=0,prediction_id="P3",xmin_px=50,ymin_px=50,xmax_px=60,ymax_px=60),
    ])
    g,u=match_greedy_one_fov(ref,pred)
    h,v=match_hungarian_one_fov(ref,pred)
    require(g.ai_detected.sum()==2 and h.ai_detected.sum()==2, "Synthetic matching failed")
    require(set(g.prediction_id[g.ai_detected])=={"P1","P2"}, "Synthetic greedy assignment failed")
    require(len(u)==2 and len(v)==2, "Synthetic used-prediction accounting failed")
    print("SELF_TEST_PASS")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--predictions",type=Path)
    ap.add_argument("--v15-audit",type=Path)
    ap.add_argument("--checkpoint",type=Path)
    ap.add_argument("--manifest",type=Path)
    ap.add_argument("--key",type=Path)
    ap.add_argument("--evaluation-ref",type=Path)
    ap.add_argument("--unbiased-ref",type=Path)
    ap.add_argument("--output-dir",type=Path)
    args=ap.parse_args()

    if args.self_test:
        self_test()
        return

    for name in ("predictions","v15_audit","checkpoint","manifest","key",
                 "evaluation_ref","unbiased_ref","output_dir"):
        require(getattr(args,name) is not None, f"--{name.replace('_','-')} required")

    audit=validate_gate(args.v15_audit,args.predictions,args.checkpoint)
    pred,images,eligible,e_ref,u_ref=load_validate_inputs(
        args.predictions,args.manifest,args.key,args.evaluation_ref,args.unbiased_ref
    )

    primary,extras=match_all(eligible,pred,"greedy")
    sensitivity,extras_h=match_all(eligible,pred,"hungarian")
    sens_summary,sens_anchor=compare_methods(primary,sensitivity)

    e=reader_events(primary,e_ref,"Evaluation")
    u=reader_events(primary,u_ref,"Unbiased")
    events=pd.concat([e,u],ignore_index=True)

    overall,by_slide,slide_condition,condition_summary,reader_condition=make_summaries(
        primary,extras,events
    )

    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)

    primary.to_csv(out/"restricted_anchor_matches_v16.csv",index=False)
    extras.to_csv(out/"restricted_unmatched_predictions_on_selected_anchor_frame_v16.csv",index=False)
    events.to_csv(out/"restricted_recorded_reader_detection_agreement_v16.csv",index=False)
    by_slide.to_csv(out/"restricted_selected_anchor_detection_by_slide_v16.csv",index=False)
    slide_condition.to_csv(out/"restricted_slide_condition_detection_agreement_v16.csv",index=False)
    condition_summary.to_csv(out/"restricted_condition_detection_agreement_v16.csv",index=False)
    reader_condition.to_csv(out/"restricted_reader_condition_detection_agreement_v16.csv",index=False)
    sens_anchor.to_csv(out/"restricted_greedy_vs_hungarian_anchor_sensitivity_v16.csv",index=False)

    summary={
        "status":"SCORING_COMPLETE_CONDITIONAL_ON_SOURCE_SELECTED_ANCHORS",
        "primary_matching_rule":"deterministic_global_greedy_IoU_descending",
        "matching_iou_threshold":IOU_THRESHOLD,
        "geometry_validation_boundary_tolerance_source_px":BOUNDARY_TOL_PX,
        "detection_score_threshold":None,
        "all_native_instances_are_detections":True,
        "class_scoring_performed":False,
        "prediction_sha256":EXPECTED_PRED_SHA,
        "checkpoint_sha256":EXPECTED_CKPT_SHA,
        "v15_audit_status":audit["status"],
        "anchor_detection":overall,
        "matching_algorithm_sensitivity":sens_summary,
        "condition_detection_agreement":condition_summary.to_dict(orient="records"),
        "reader_condition_detection_agreement":reader_condition.to_dict(orient="records"),
        "limitations":[
            "Selected-anchor frame is source-reader constructed and informatively selected.",
            "Unmatched predictions are not biological false positives.",
            "Unmatched selected anchors are non-detections relative to the selected source frame, not verified biological false negatives.",
            "Reader outcomes are repeated reader-anchor observations, not independent nuclei.",
            "Only five TCGA slides/patients are represented.",
            "Evaluation versus Unbiased annotation conditions are not randomized causal treatments.",
            "No biological sensitivity, specificity, false-positive rate, future-reader estimand, or calibrated Bayesian uncertainty is established by this scoring step."
        ],
        "next_gate":"Use these descriptive results to assess empirical feasibility; Bayesian estimand/modeling still requires identifiability and known-truth simulation."
    }
    (out/"restricted_scoring_summary_v16.json").write_text(json.dumps(summary,indent=2)+"\n")

    print(json.dumps(summary,indent=2))
    print("SCORING_V16_COMPLETE")
    print("NO_BIOLOGICAL_TRUTH_OR_POPULATION_ACCURACY_CLAIM")

if __name__=="__main__":
    try:
        main()
    except ProtocolError as e:
        print("PROTOCOL_BLOCKED:",e)
        raise SystemExit(2)
