#!/usr/bin/env python3
"""
Reconstructed NuCLS v7 source-export coordinate registration audit.

Uses exact anchor_id joins to the official Unbiased-Control contour exports,
then checks the contour pixel boxes against the affine transform implied by
the U-control RGB source-export filename and actual image dimensions.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

REGION_RE=re.compile(
    r"(?P<slide>TCGA-[A-Za-z0-9-]+?)_left-(?P<left>\d+)_top-(?P<top>\d+)"
    r"_bottom-(?P<bottom>\d+)_right-(?P<right>\d+)"
)
FOVID_RE=re.compile(r"^ANCHFOV-(\d+)_")


def fovid(name):
    m=FOVID_RE.match(Path(name).name)
    if not m: raise ValueError(name)
    return int(m.group(1))


def region(name):
    m=REGION_RE.search(str(name))
    if not m: raise ValueError(name)
    d=m.groupdict()
    return dict(slide=d["slide"],left=int(d["left"]),top=int(d["top"]),
                bottom=int(d["bottom"]),right=int(d["right"]))


def file_map(folder,suffix):
    ans={}
    for p in Path(folder).glob(f"*{suffix}"):
        ans[fovid(p.name)]=p
    return ans


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--v6-key",required=True)
    p.add_argument("--unbiased-rgb-dir",required=True)
    p.add_argument("--unbiased-contour-dir",required=True)
    p.add_argument("--out-dir",required=True)
    p.add_argument("--target-v7-key")
    a=p.parse_args()

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    key=pd.read_csv(a.v6_key,low_memory=False)
    rgbs=file_map(a.unbiased_rgb_dir,".png")
    contours=file_map(a.unbiased_contour_dir,".csv")

    contour_tables={}
    rgb_meta={}
    rows=[]

    for _,r in key.iterrows():
        z=r.to_dict()
        eligible=bool(r["paired_anchor_rgb_available"])
        if not eligible:
            z.update(
                coordinate_registration_status="NO_PAIRED_RGB_EXCLUDE",
                contour_xmin_px=np.nan,contour_ymin_px=np.nan,
                contour_xmax_px=np.nan,contour_ymax_px=np.nan,
                registration_basis="exact_anchor_id_U_control_contour_pixel_bbox_within_1px_of_image_filename_affine"
            )
            rows.append(z); continue

        fid=int(r["fov_id"])
        if fid not in rgbs or fid not in contours:
            raise RuntimeError(f"Missing RGB/contour for FOV {fid}")

        if fid not in contour_tables:
            c=pd.read_csv(contours[fid])
            if "anchor_id" not in c.columns: raise RuntimeError(f"No anchor_id in {contours[fid]}")
            if c.anchor_id.duplicated().any(): raise RuntimeError(f"Duplicate anchor_id in {contours[fid]}")
            contour_tables[fid]=c.set_index("anchor_id")

        ctab=contour_tables[fid]
        if r.anchor_id not in ctab.index:
            raise RuntimeError(f"Anchor absent from contour FOV {fid}: {r.anchor_id}")
        c=ctab.loc[r.anchor_id]

        if fid not in rgb_meta:
            zz=region(rgbs[fid].name)
            with Image.open(rgbs[fid]) as im: w,h=im.size
            rgb_meta[fid]=(zz,w,h)
        src,w,h=rgb_meta[fid]

        sx=w/(src["right"]-src["left"])
        sy=h/(src["bottom"]-src["top"])
        expected=np.array([
            (float(r.xmin)-src["left"])*sx,
            (float(r.ymin)-src["top"])*sy,
            (float(r.xmax)-src["left"])*sx,
            (float(r.ymax)-src["top"])*sy,
        ])
        observed=np.array([c.xmin,c.ymin,c.xmax,c.ymax],dtype=float)
        max_abs=float(np.max(np.abs(expected-observed)))
        if not max_abs < 1.0:
            raise RuntimeError(
                f"Registration >1px for {r.anchor_id}: max_abs={max_abs:.6f}"
            )

        z.update(
            coordinate_registration_status="SOURCE_EXPORT_PIXEL_BOX_VERIFIED",
            contour_xmin_px=float(c.xmin),contour_ymin_px=float(c.ymin),
            contour_xmax_px=float(c.xmax),contour_ymax_px=float(c.ymax),
            registration_basis="exact_anchor_id_U_control_contour_pixel_bbox_within_1px_of_image_filename_affine"
        )
        rows.append(z)

    res=pd.DataFrame(rows)
    # Match preserved v7 column order.
    cols=[
        "anchor_id","fovname","slide","patient_id","xmin","ymin","xmax","ymax",
        "paired_anchor_rgb_available","fold_held_out_slide","ai_prediction_status",
        "source_reference_type","fov_id","proposed_coverage","match_state","image_id",
        "evaluation_state","coordinate_registration_status","original_reference_warning",
        "contour_xmin_px","contour_ymin_px","contour_xmax_px","contour_ymax_px",
        "registration_basis","tissue_ground_truth"
    ]
    res=res[cols]
    path=out/"nucls_scoring_key_RESTRICTED_SOURCE_EXPORT_REGISTERED_v7_RECONSTRUCTED.csv"
    res.to_csv(path,index=False)

    elig=res[res.paired_anchor_rgb_available]
    assert len(res)==1155 and len(elig)==1144
    assert elig.coordinate_registration_status.eq("SOURCE_EXPORT_PIXEL_BOX_VERIFIED").all()
    assert elig.anchor_id.nunique()==1144
    assert elig.fov_id.nunique()==52

    validation={"gate":"PASS","anchors_total":1155,"registered_anchors":1144,
                "registered_fovs":52,"exact_anchor_id_join":"PASS",
                "all_bbox_affine_discrepancies_lt_1px":True}

    if a.target_v7_key:
        t=pd.read_csv(a.target_v7_key,low_memory=False)
        if list(res.anchor_id)!=list(t.anchor_id):
            raise RuntimeError("Target row order/anchor_id differs")
        textcols=["coordinate_registration_status","registration_basis"]
        numcols=["contour_xmin_px","contour_ymin_px","contour_xmax_px","contour_ymax_px"]
        for c in textcols:
            if not res[c].fillna("<NA>").astype(str).equals(t[c].fillna("<NA>").astype(str)):
                raise RuntimeError(f"Target mismatch: {c}")
        for c in numcols:
            if not np.allclose(res[c],t[c],equal_nan=True):
                raise RuntimeError(f"Target mismatch: {c}")
        validation["v7_registration_target"]="PASS"

    with open(out/"v7_registration_validation.json","w") as f:
        json.dump(validation,f,indent=2,sort_keys=True)
    print(json.dumps(validation,indent=2))


if __name__=="__main__":
    main()
