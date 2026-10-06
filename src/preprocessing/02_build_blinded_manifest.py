#!/usr/bin/env python3
"""
Reconstructed NuCLS v6 preprocessing step.

Builds the 52-image blinded inference manifest and the 1,155-anchor
restricted scoring key from the official NuCLS Evaluation/Unbiased P-truth
tables and RGB exports. No AI predictions are read or generated.

The reconstruction is validated against the preserved v7/v8 targets when
--target-v7-key and/or --target-v8-manifest are supplied.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

REGION_RE = re.compile(
    r"(?P<slide>TCGA-[A-Za-z0-9-]+?)_left-(?P<left>\d+)_top-(?P<top>\d+)"
    r"_bottom-(?P<bottom>\d+)_right-(?P<right>\d+)"
)
FOVID_RE = re.compile(r"^ANCHFOV-(\d+)_")


def region(s):
    m = REGION_RE.search(str(s))
    if not m:
        raise ValueError(f"Cannot parse source region: {s}")
    d = m.groupdict()
    l,t,b,r = map(int,(d["left"],d["top"],d["bottom"],d["right"]))
    return dict(slide=d["slide"], left=l, top=t, bottom=b, right=r,
                cx=(l+r)/2, cy=(t+b)/2)


def fovid(name):
    m = FOVID_RE.match(Path(name).name)
    if not m:
        raise ValueError(f"Cannot parse ANCHFOV id: {name}")
    return int(m.group(1))


def patient(slide):
    return "-".join(str(slide).split("-")[:3])


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def rgb_inventory(folder):
    rows=[]
    for p in sorted(Path(folder).glob("*.png")):
        z=region(p.name)
        with Image.open(p) as im:
            w,h=im.size
        rows.append(dict(fov_id=fovid(p.name), rgb_filename=p.name,
                         rgb_path=str(p.resolve()), width_px=w, height_px=h,
                         size_bytes=p.stat().st_size, image_sha256=sha256(p), **z))
    x=pd.DataFrame(rows)
    if x.empty: raise RuntimeError(f"No PNGs found in {folder}")
    if x["fov_id"].duplicated().any(): raise RuntimeError("Duplicate ANCHFOV id")
    return x


def match_E_fovs_to_E_rgb(e_fovs, e_rgb):
    # deterministic global nearest-center one-to-one match within slide
    pairs=[]
    for i,f in e_fovs.iterrows():
        for j,r in e_rgb[e_rgb.slide.eq(f.slide)].iterrows():
            d=float(np.hypot(f.cx-r.cx,f.cy-r.cy))
            pairs.append((d,i,j))
    pairs.sort(key=lambda q:(q[0],q[1],q[2]))
    used_i,used_j=set(),set()
    ans={}
    for d,i,j in pairs:
        if i not in used_i and j not in used_j:
            ans[i]=(j,d); used_i.add(i); used_j.add(j)
    out=e_fovs.copy()
    out["E_fov_id"]=np.nan
    out["E_rgb_center_distance"]=np.nan
    for i,(j,d) in ans.items():
        out.loc[i,"E_fov_id"]=e_rgb.loc[j,"fov_id"]
        out.loc[i,"E_rgb_center_distance"]=d
    return out


def overlap_fraction(a,b):
    if a["slide"]!=b["slide"]: return 0.0
    w=max(0,min(a["right"],b["right"])-max(a["left"],b["left"]))
    h=max(0,min(a["bottom"],b["bottom"])-max(a["top"],b["top"]))
    area=(a["right"]-a["left"])*(a["bottom"]-a["top"])
    return (w*h/area) if area>0 else 0.0


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--eval-csv",required=True)
    p.add_argument("--unbiased-csv",required=True)
    p.add_argument("--eval-rgb-dir",required=True)
    p.add_argument("--unbiased-rgb-dir",required=True)
    p.add_argument("--out-dir",required=True)
    p.add_argument("--target-v7-key")
    p.add_argument("--target-v8-manifest")
    a=p.parse_args()

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    E=pd.read_csv(a.eval_csv,low_memory=False)
    U=pd.read_csv(a.unbiased_csv,low_memory=False)
    if E.anchor_id.duplicated().any() or U.anchor_id.duplicated().any():
        raise RuntimeError("anchor_id must be unique in each P-truth table")

    shared=set(E.anchor_id)&set(U.anchor_id)
    Es=E[E.anchor_id.isin(shared)].copy()  # preserve source E row order
    if len(shared)!=1155: raise RuntimeError(f"Expected 1155 shared anchors, got {len(shared)}")

    # one row per source E FOV
    ef=[]
    for name in pd.unique(Es.fovname):
        z=region(name); ef.append(dict(fovname=name,**z))
    ef=pd.DataFrame(ef)

    ergb=rgb_inventory(a.eval_rgb_dir)
    urgb=rgb_inventory(a.unbiased_rgb_dir)
    ef=match_E_fovs_to_E_rgb(ef,ergb)

    # Paired image requires the E-assigned ANCHFOV id to exist in U.
    u_by_id=urgb.set_index("fov_id",drop=False)
    ef["paired_rgb"]=ef["E_fov_id"].map(
        lambda x: False if pd.isna(x) else int(x) in u_by_id.index
    )

    # Historical proposed_coverage: for E-mapped FOVs, best spatial overlap
    # with any U RGB from the same slide; if no E export exists, zero.
    cov=[]
    for _,f in ef.iterrows():
        if pd.isna(f.E_fov_id):
            cov.append(0.0); continue
        aa=region(f.fovname)
        vals=[overlap_fraction(aa,r) for _,r in urgb[urgb.slide.eq(f.slide)].iterrows()]
        cov.append(max(vals) if vals else 0.0)
    ef["proposed_coverage"]=cov

    meta=ef[["fovname","E_fov_id","paired_rgb","proposed_coverage"]]
    key=Es[["anchor_id","fovname","xmin","ymin","xmax","ymax"]].merge(
        meta,on="fovname",how="left",validate="many_to_one"
    )
    key["slide"]=key.fovname.map(lambda x:region(x)["slide"])
    key["patient_id"]=key.slide.map(patient)
    key["paired_anchor_rgb_available"]=key.paired_rgb.astype(bool)
    key["fold_held_out_slide"]=key.slide
    key["ai_prediction_status"]="not_generated"
    key["source_reference_type"]="condition_dependent_inferred_P_truth_not_independent"
    key["fov_id"]=key["E_fov_id"].where(key.paired_anchor_rgb_available)
    key["match_state"]=np.where(key.paired_anchor_rgb_available,"unique_spatial_match","no_match")
    key["image_id"]=key.fov_id.map(
        lambda x: np.nan if pd.isna(x) else f"nucls-U-FOV-{int(x):02d}"
    )
    key["evaluation_state"]=np.where(
        key.paired_anchor_rgb_available,
        "eligible_for_conditional_scoring_after_registration_and_AI_provenance",
        "NO_PAIRED_RGB_EXCLUDE"
    )
    key["original_reference_warning"]="selected_by_both_conditions_not_independent_biological_truth"
    key["tissue_ground_truth"]="NOT_INDEPENDENTLY_VALIDATED"
    key=key[[
        "anchor_id","fovname","slide","patient_id","xmin","ymin","xmax","ymax",
        "paired_anchor_rgb_available","fold_held_out_slide","ai_prediction_status",
        "source_reference_type","fov_id","proposed_coverage","match_state","image_id",
        "evaluation_state","original_reference_warning","tissue_ground_truth"
    ]]
    key.to_csv(out/"nucls_scoring_key_RESTRICTED_NO_PREDICTIONS_v6_RECONSTRUCTED.csv",index=False)

    paired_ids=sorted({int(x) for x in key.loc[key.paired_anchor_rgb_available,"fov_id"]})
    mani=urgb[urgb.fov_id.isin(paired_ids)].copy().sort_values("fov_id")
    mani["image_id"]=mani.fov_id.map(lambda x:f"nucls-U-FOV-{int(x):02d}")
    mani["patient_id"]=mani.slide.map(patient)
    mani["image_provenance"]="NuCLS_Unbiased_Control_Ptruths_rgbs"
    mani["image_geometry"]="LOCAL_OFFICIAL_SOURCE_EXPORT"
    mani["holdout_fold"]=mani.slide
    mani["prediction_state"]="NOT_GENERATED"
    mani=mani[[
        "image_id","fov_id","slide","patient_id","rgb_filename","rgb_path",
        "image_provenance","image_geometry","holdout_fold","prediction_state",
        "image_sha256","width_px","height_px","size_bytes"
    ]]
    mani.to_csv(out/"nucls_blinded_image_only_inference_v6_RECONSTRUCTED.csv",index=False)

    mani[["image_id","fov_id","slide","patient_id","holdout_fold"]].to_csv(
        out/"nucls_inference_slide_folds_v6_RECONSTRUCTED.csv",index=False
    )

    # Frozen structural invariants.
    assert len(key)==1155
    assert int(key.paired_anchor_rgb_available.sum())==1144
    assert len(mani)==52
    assert set(mani.fov_id)==(set(range(53))-{51})
    assert key.slide.nunique()==5

    validation={"gate":"PASS","shared_anchors":1155,"eligible_anchors":1144,
                "images":52,"slides":5}

    if a.target_v7_key:
        t=pd.read_csv(a.target_v7_key,low_memory=False)
        cols=["anchor_id","paired_anchor_rgb_available","slide","patient_id",
              "fov_id","proposed_coverage","match_state","image_id"]
        m=key[cols].merge(t[cols],on="anchor_id",suffixes=("_new","_target"),validate="one_to_one")
        for c in ["paired_anchor_rgb_available","slide","patient_id","match_state","image_id"]:
            x=m[f"{c}_new"].fillna("<NA>").astype(str)
            y=m[f"{c}_target"].fillna("<NA>").astype(str)
            if not x.equals(y): raise RuntimeError(f"v7 target mismatch: {c}")
        if not np.allclose(m.fov_id_new,m.fov_id_target,equal_nan=True):
            raise RuntimeError("v7 target mismatch: fov_id")
        if not np.allclose(m.proposed_coverage_new,m.proposed_coverage_target,atol=1e-12):
            raise RuntimeError("v7 target mismatch: proposed_coverage")
        validation["v7_semantic_mapping"]="PASS"

    if a.target_v8_manifest:
        t=pd.read_csv(a.target_v8_manifest,low_memory=False)
        cols=["image_id","fov_id","slide","patient_id","image_sha256","width_px","height_px","size_bytes"]
        m=mani[cols].merge(t[cols],on="image_id",suffixes=("_new","_target"),validate="one_to_one")
        for c in ["fov_id","slide","patient_id","image_sha256","width_px","height_px","size_bytes"]:
            if not m[f"{c}_new"].astype(str).equals(m[f"{c}_target"].astype(str)):
                raise RuntimeError(f"v8 target mismatch: {c}")
        validation["v8_local_file_identity"]="PASS"

    with open(out/"v6_reconstruction_validation.json","w") as f:
        json.dump(validation,f,indent=2,sort_keys=True)
    print(json.dumps(validation,indent=2))


if __name__=="__main__":
    main()
