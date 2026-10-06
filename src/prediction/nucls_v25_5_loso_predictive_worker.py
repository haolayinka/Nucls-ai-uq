#!/usr/bin/env python3
from __future__ import annotations
from repo_paths import repo_root

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nucls_v23_1_shared_detectability_cv_worker as cv
import nucls_v23_1_joint_bayes_logit as v231

READERS = list(v231.PRIMARY_READERS)
N_SLIDES = 5
N_ANCHORS = 1144
AI_POS = 593


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument(
        "--project-root",
        type=Path,
        default=repo_root(),
    )
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--tune", type=int, default=2500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.995)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    if not (0 <= args.task < 15):
        raise ValueError("task must be 0..14")

    if args.task < 5:
        mode = "human_joint"
        fold = args.task
    elif args.task < 10:
        mode = "human_reader_only"
        fold = args.task - 5
    else:
        mode = "ai_holdout"
        fold = args.task - 10

    v231.self_test()

    root = args.project_root
    outdir = args.outdir or root / "outputs" / "v25_5_loso_predictive"
    outdir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(
        root
        / "outputs"
        / "scoring_v16_1"
        / "restricted_recorded_reader_detection_agreement_v16.csv"
    )
    ai_tbl = pd.read_csv(
        root
        / "outputs"
        / "scoring_v16_1"
        / "restricted_anchor_matches_v16.csv"
    )

    d = raw.loc[raw.reader.isin(READERS)].copy()
    d["anchor_id"] = d.anchor_id.astype(str)
    d["slide"] = d.slide.astype(str)

    if len(d) != 10833 or d.anchor_id.nunique() != N_ANCHORS or d.image_id.nunique() != 52:
        raise RuntimeError("Primary five-reader input counts changed")

    anchor_slide_n = d.groupby("anchor_id").slide.nunique()
    if (anchor_slide_n != 1).any():
        raise RuntimeError("At least one anchor maps to multiple slides")

    slides = sorted(d.slide.unique().tolist())
    if len(slides) != N_SLIDES:
        raise RuntimeError(f"Expected five slides, got {slides}")

    heldout_slide = slides[fold]
    heldout_ids = sorted(
        d.loc[d.slide == heldout_slide, "anchor_id"].unique().tolist()
    )
    train_ids = sorted(set(d.anchor_id.unique()) - set(heldout_ids))

    a = ai_tbl[["anchor_id", "ai_detected"]].copy()
    a["anchor_id"] = a.anchor_id.astype(str)
    if a.anchor_id.duplicated().any():
        chk = a.groupby("anchor_id").ai_detected.nunique(dropna=False)
        if (chk > 1).any():
            raise RuntimeError("Conflicting AI values")
        a = a.drop_duplicates("anchor_id")
    ai = pd.Series(a.ai_detected.astype(int).to_numpy(), index=a.anchor_id)

    if len(ai) != N_ANCHORS or int(ai.sum()) != AI_POS:
        raise RuntimeError("AI vector count changed")

    fov_values = sorted(d.image_id.astype(str).unique())
    fov_map = {x: i for i, x in enumerate(fov_values)}

    if mode.startswith("human_"):
        reader_long = cv.make_reader_long(d, fov_map, heldout_ids)
    else:
        reader_long = cv.make_reader_long(d, fov_map, None)

    enc = v231.encode_observed(
        reader_long,
        ai,
        readers=v231.PRIMARY_READERS,
        expected_ai_matches=AI_POS,
    )

    if mode == "human_joint":
        model = v231.build_model(enc)
        scalars = cv.SCALAR_JOINT
    elif mode == "human_reader_only":
        model = cv.build_reader_only(enc)
        scalars = cv.SCALAR_READER
    else:
        heldout_set = set(heldout_ids)
        ai_train_mask = np.array(
            [anchor not in heldout_set for anchor in enc.anchor_ids], dtype=bool
        )
        model = cv.build_ai_masked(enc, ai_train_mask)
        scalars = cv.SCALAR_JOINT

    import arviz as az
    import pymc as pm

    seed = 255500 + args.task
    t0 = time.time()
    with model:
        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=min(
                args.chains,
                max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
            ),
            random_seed=seed,
            target_accept=args.target_accept,
            init="jitter+adapt_diag",
            progressbar=False,
            return_inferencedata=True,
        )
    elapsed = time.time() - t0

    if mode.startswith("human_"):
        heldout_rows = d[d.anchor_id.isin(heldout_ids)].copy()
        pred = cv.predict_heldout_humans(idata, enc, heldout_rows, fov_map)
        pred["fold"] = fold
        pred["heldout_slide"] = heldout_slide
        pred["model"] = "joint" if mode == "human_joint" else "reader_only"
        met = cv.metrics(pred.reader_detected.astype(int), pred.pred)
        met["n_anchors"] = int(pred.anchor_id.nunique())
        met["n_fovs"] = int(pred.image_id.nunique())
        met["rf_unseen_rows"] = int((~pred.rf_seen_in_training).sum())
    else:
        predv = cv.predict_heldout_ai(idata, enc, heldout_ids)
        y = ai.reindex(heldout_ids).astype(int).to_numpy()

        meta = (
            d[d.anchor_id.isin(heldout_ids)]
            .groupby("anchor_id", as_index=False)
            .agg(
                image_id=("image_id", "first"),
                slide=("slide", "first"),
                patient_id=("patient_id", "first"),
            )
            .set_index("anchor_id")
            .reindex(heldout_ids)
            .reset_index()
        )
        pred = meta.copy()
        pred["ai_detected"] = y
        pred["pred"] = predv
        pred["fold"] = fold
        pred["heldout_slide"] = heldout_slide
        pred["model"] = "human_to_ai_joint"

        met = cv.metrics(y, predv)
        train_prev = float(ai.reindex(train_ids).mean())
        met["prevalence_baseline_brier"] = float(
            np.mean((train_prev - y) ** 2)
        )
        pc = float(np.clip(train_prev, 1e-12, 1 - 1e-12))
        met["prevalence_baseline_log_loss"] = float(
            -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))
        )

    pred.to_csv(
        outdir / f"{mode}_slide{fold}_predictions.csv",
        index=False,
    )

    ds = az.summary(idata, var_names=scalars, round_to=None)
    ds.to_csv(
        outdir / f"{mode}_slide{fold}_mcmc_scalar_diagnostics.csv"
    )

    diagnostics = {
        "mode": mode,
        "fold": fold,
        "heldout_slide": heldout_slide,
        "seed": seed,
        "n_heldout_anchors": len(heldout_ids),
        "elapsed_sec": elapsed,
        "divergences": int(np.asarray(idata.sample_stats["diverging"]).sum()),
        "max_rhat_scalar": float(ds.r_hat.max()),
        "min_ess_bulk_scalar": float(ds.ess_bulk.min()),
        "min_ess_tail_scalar": float(ds.ess_tail.min()),
        "metrics": met,
        "storage_policy": (
            "No posterior/idata object written; only predictions, "
            "scalar diagnostics, and JSON."
        ),
    }
    (
        outdir / f"{mode}_slide{fold}_result.json"
    ).write_text(json.dumps(diagnostics, indent=2))

    print(
        "V25_5_LOSO_RESULT " + json.dumps(diagnostics, sort_keys=True),
        flush=True,
    )
    print("V25_5_LOSO_TASK_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
