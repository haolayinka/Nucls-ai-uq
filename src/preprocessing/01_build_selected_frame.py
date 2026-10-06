#!/usr/bin/env python3
"""
Reconstructed NuCLS v5 preprocessing / selection audit.

Purpose
-------
Rebuild the source-selected E/U anchor frame from the official NuCLS
P-truth tables and the Evaluation / Unbiased P-truth RGB folders.

This is a reconstruction of the lost local v5 preprocessing step.  It is
written to reproduce the *scientific selection rules* preserved in the
project audit trail, then validate them against frozen invariants:

  E unique anchors        = 1,358
  U unique anchors        = 1,569
  shared anchor_ids       = 1,155
  E-only                  =   203
  U-only                  =   414
  union                   = 1,772
  paired-RGB shared frame = 1,144 anchors
  paired-RGB FOVs         =    52
  slides/patients         =     5

Important semantics
-------------------
- `DidNotAnnotateFOV` is missing/unobserved.
- `undetected` is an observed negative detection.
- P-truth labels are inferred reference labels, not biological truth.
- Selection is based on the source NuCLS frame only; no external-AI
  predictions are used here.

Usage
-----
python nucls_selection_split_audit_v5_RECONSTRUCTED.py \
  --eval-csv /path/v3.1_final_anchors_E_Ps_AreTruth.csv \
  --unbiased-csv /path/v3.1_final_anchors_U-control_Ps_AreTruth.csv \
  --eval-rgb-dir /path/PsAreTruth_E/rgbs \
  --unbiased-rgb-dir /path/PsAreTruth_U-control/rgbs \
  --out-dir outputs/preprocessing_v5
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PAIRED_READERS = ["SP.2", "SP.3", "JP.1", "JP.2", "JP.5", "JP.6"]

FROZEN = {
    "eval_unique_anchors": 1358,
    "unbiased_unique_anchors": 1569,
    "shared_anchors": 1155,
    "eval_only": 203,
    "unbiased_only": 414,
    "union": 1772,
    "paired_rgb_shared_anchors": 1144,
    "paired_rgb_fovs": 52,
    "slides": 5,
}

BOUNDS_RE = re.compile(
    r"(?P<slide>TCGA-[A-Za-z0-9-]+?)"
    r"_left-(?P<left>\d+)_top-(?P<top>\d+)"
    r"_bottom-(?P<bottom>\d+)_right-(?P<right>\d+)"
)


def parse_region(s: str):
    m = BOUNDS_RE.search(str(s))
    if not m:
        raise ValueError(f"Cannot parse slide/bounds from: {s}")
    d = m.groupdict()
    left, top = int(d["left"]), int(d["top"])
    bottom, right = int(d["bottom"]), int(d["right"])
    return {
        "slide": d["slide"],
        "left": left,
        "top": top,
        "bottom": bottom,
        "right": right,
        "cx": (left + right) / 2.0,
        "cy": (top + bottom) / 2.0,
    }


def rgb_inventory(rgb_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(rgb_dir.glob("*.png")):
        try:
            z = parse_region(p.name)
        except ValueError:
            continue
        rows.append({"rgb_name": p.name, "rgb_path": str(p.resolve()), **z})
    if not rows:
        raise RuntimeError(f"No parseable RGB PNGs found in {rgb_dir}")
    return pd.DataFrame(rows)


def fov_inventory(df: pd.DataFrame) -> pd.DataFrame:
    names = pd.Series(df["fovname"].dropna().astype(str).unique(), name="fovname")
    rows = []
    for name in names:
        z = parse_region(name)
        rows.append({"fovname": name, **z})
    return pd.DataFrame(rows)


def match_fovs_to_rgbs(fovs: pd.DataFrame, rgbs: pd.DataFrame, condition: str,
                       max_center_distance: float) -> pd.DataFrame:
    """
    Deterministic one-to-one nearest-center matching within slide.

    For each source P-truth FOV, candidate RGBs are restricted to the same
    TCGA slide. Candidate pairs are globally sorted by distance, then greedily
    assigned one-to-one. This avoids row-order dependence.
    """
    pairs = []
    for fi, f in fovs.iterrows():
        g = rgbs[rgbs["slide"] == f["slide"]]
        for ri, r in g.iterrows():
            dist = float(np.hypot(f["cx"] - r["cx"], f["cy"] - r["cy"]))
            pairs.append((dist, fi, ri))
    pairs.sort(key=lambda x: (x[0], x[1], x[2]))

    used_f, used_r = set(), set()
    assigned = {}
    for dist, fi, ri in pairs:
        if fi in used_f or ri in used_r:
            continue
        if dist <= max_center_distance:
            assigned[fi] = (ri, dist)
            used_f.add(fi)
            used_r.add(ri)

    out = fovs[["fovname", "slide", "cx", "cy"]].copy()
    out[f"{condition}_rgb_name"] = None
    out[f"{condition}_rgb_path"] = None
    out[f"{condition}_rgb_center_distance"] = np.nan

    for fi, (ri, dist) in assigned.items():
        out.loc[fi, f"{condition}_rgb_name"] = rgbs.loc[ri, "rgb_name"]
        out.loc[fi, f"{condition}_rgb_path"] = rgbs.loc[ri, "rgb_path"]
        out.loc[fi, f"{condition}_rgb_center_distance"] = dist

    return out


def state_is_observed(x) -> bool:
    return pd.notna(x) and str(x) != "DidNotAnnotateFOV"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-csv", required=True)
    ap.add_argument("--unbiased-csv", required=True)
    ap.add_argument("--eval-rgb-dir", required=True)
    ap.add_argument("--unbiased-rgb-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--max-center-distance",
        type=float,
        default=100.0,
        help="Maximum P-truth-FOV to RGB center distance in source pixels."
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    E = pd.read_csv(args.eval_csv, low_memory=False)
    U = pd.read_csv(args.unbiased_csv, low_memory=False)

    required = {"anchor_id", "fovname", "xmin", "ymin", "xmax", "ymax"}
    for label, df in [("Evaluation", E), ("Unbiased", U)]:
        miss = required - set(df.columns)
        if miss:
            raise RuntimeError(f"{label} table missing required columns: {sorted(miss)}")
        if df["anchor_id"].duplicated().any():
            raise RuntimeError(f"{label} contains duplicated anchor_id values.")

    e_ids, u_ids = set(E["anchor_id"]), set(U["anchor_id"])
    shared = e_ids & u_ids
    e_only, u_only = e_ids - u_ids, u_ids - e_ids
    union = e_ids | u_ids

    # Build union ledger.
    base_cols = ["anchor_id", "fovname", "xmin", "ymin", "xmax", "ymax"]
    e0 = E[base_cols].copy().rename(columns={c: f"E_{c}" for c in base_cols if c != "anchor_id"})
    u0 = U[base_cols].copy().rename(columns={c: f"U_{c}" for c in base_cols if c != "anchor_id"})
    ledger = e0.merge(u0, on="anchor_id", how="outer", validate="one_to_one")
    ledger["membership"] = np.select(
        [
            ledger["anchor_id"].isin(shared),
            ledger["anchor_id"].isin(e_only),
            ledger["anchor_id"].isin(u_only),
        ],
        ["shared", "E_only", "U_only"],
        default="ERROR",
    )
    ledger.to_csv(out / "nucls_condition_selected_anchor_union_NO_NEGATIVE_LABELS_v5.csv",
                  index=False)

    # Source P-truth FOV inventory. Shared anchor IDs use the exact common
    # source anchor identity, so E fovname is the canonical selected-frame key.
    E_shared = E[E["anchor_id"].isin(shared)].copy()
    U_shared = U[U["anchor_id"].isin(shared)].copy()

    fovs = fov_inventory(E_shared)
    e_rgb = rgb_inventory(Path(args.eval_rgb_dir))
    u_rgb = rgb_inventory(Path(args.unbiased_rgb_dir))

    me = match_fovs_to_rgbs(fovs, e_rgb, "E", args.max_center_distance)
    mu = match_fovs_to_rgbs(fovs, u_rgb, "U", args.max_center_distance)
    fm = me.merge(
        mu[["fovname", "U_rgb_name", "U_rgb_path", "U_rgb_center_distance"]],
        on="fovname", how="left", validate="one_to_one",
    )
    fm["paired_rgb"] = fm["E_rgb_name"].notna() & fm["U_rgb_name"].notna()
    fm.to_csv(out / "nucls_fov_rgb_linkage_v5.csv", index=False)

    eligible_fovs = set(fm.loc[fm["paired_rgb"], "fovname"])
    selected = E_shared[E_shared["fovname"].isin(eligible_fovs)].copy()

    # Confirm U has the same selected anchor IDs; source attributes are retained
    # from both conditions rather than silently overwritten.
    u_sel = U_shared[U_shared["anchor_id"].isin(selected["anchor_id"])].copy()
    if set(selected["anchor_id"]) != set(u_sel["anchor_id"]):
        raise RuntimeError("Selected E/U shared anchor IDs do not agree.")

    # AI evaluation manifest: deliberately contains no predictions.
    manifest = selected[
        ["anchor_id", "fovname", "xmin", "ymin", "xmax", "ymax"]
    ].copy()
    manifest["slide"] = manifest["fovname"].map(lambda x: parse_region(x)["slide"])
    manifest = manifest.merge(
        fm[["fovname", "E_rgb_name", "E_rgb_path", "U_rgb_name", "U_rgb_path"]],
        on="fovname", how="left", validate="many_to_one",
    )
    manifest["external_ai_prediction"] = pd.NA
    manifest = manifest.sort_values(["slide", "fovname", "anchor_id"]).reset_index(drop=True)
    manifest.to_csv(out / "nucls_ai_evaluation_manifest_NO_PREDICTIONS_v5.csv", index=False)

    # LOSO manifest.
    slides = sorted(manifest["slide"].unique())
    loso_rows = []
    for held in slides:
        for _, r in manifest.iterrows():
            loso_rows.append({
                "heldout_slide": held,
                "anchor_id": r["anchor_id"],
                "slide": r["slide"],
                "split": "test" if r["slide"] == held else "train",
            })
    pd.DataFrame(loso_rows).to_csv(out / "nucls_leave_one_slide_out_manifest_v5.csv",
                                   index=False)

    # Slide summary.
    summary = (
        manifest.groupby("slide", as_index=False)
        .agg(n_anchors=("anchor_id", "nunique"),
             n_fovs=("fovname", "nunique"))
    )
    summary.to_csv(out / "nucls_five_slide_selected_anchor_summary_v5.csv", index=False)

    # Paired-reader-by-slide support, preserving missingness semantics.
    reader_rows = []
    E_idx = E.set_index("anchor_id")
    U_idx = U.set_index("anchor_id")
    for reader in PAIRED_READERS:
        if reader not in E.columns or reader not in U.columns:
            continue
        for slide in slides:
            ids = manifest.loc[manifest["slide"] == slide, "anchor_id"]
            ev = E_idx.loc[ids, reader]
            uv = U_idx.loc[ids, reader]
            obs_e = ev.map(state_is_observed)
            obs_u = uv.map(state_is_observed)
            reader_rows.append({
                "reader": reader,
                "slide": slide,
                "n_selected_anchors": int(len(ids)),
                "n_observed_E": int(obs_e.sum()),
                "n_observed_U": int(obs_u.sum()),
                "n_paired_EU": int((obs_e & obs_u).sum()),
            })
    pd.DataFrame(reader_rows).to_csv(out / "nucls_paired_reader_by_slide_v5.csv",
                                     index=False)

    audit = {
        "reconstruction_status": "RECONSTRUCTED_FROM_PRESERVED_RULES",
        "external_ai_predictions_used": False,
        "did_not_annotate_semantics": "missing/unobserved",
        "undetected_semantics": "observed negative",
        "max_center_distance": args.max_center_distance,
        "counts": {
            "eval_unique_anchors": int(E["anchor_id"].nunique()),
            "unbiased_unique_anchors": int(U["anchor_id"].nunique()),
            "shared_anchors": int(len(shared)),
            "eval_only": int(len(e_only)),
            "unbiased_only": int(len(u_only)),
            "union": int(len(union)),
            "eval_rgb_files": int(len(e_rgb)),
            "unbiased_rgb_files": int(len(u_rgb)),
            "paired_rgb_fovs": int(fm["paired_rgb"].sum()),
            "paired_rgb_shared_anchors": int(manifest["anchor_id"].nunique()),
            "slides": int(manifest["slide"].nunique()),
        },
        "frozen_expected": FROZEN,
    }

    failures = []
    for key, expected in FROZEN.items():
        got = audit["counts"][key]
        if got != expected:
            failures.append({"metric": key, "expected": expected, "observed": got})
    audit["validation_failures"] = failures
    audit["gate"] = "PASS" if not failures else "FAIL"

    with open(out / "nucls_selection_and_split_audit_v5.json", "w") as f:
        json.dump(audit, f, indent=2, sort_keys=True)

    print(json.dumps(audit, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
