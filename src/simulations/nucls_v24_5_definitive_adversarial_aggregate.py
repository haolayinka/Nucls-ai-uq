#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCENARIOS = [f"A{i}" for i in range(1, 9)]
TARGETS = ["theta_U_selected", "theta_E_selected", "delta_E_minus_U_selected"]

EXPECTED_M1_PRIMARY_FAILURES = 33
EXPECTED_M1_FINAL_FAILURES = 0
EXPECTED_M2_PRIMARY_FAILURE_SET = {("A1", 67), ("A7", 56)}
EXPECTED_M2_RETRY_PASS_SET = {("A7", 56)}
EXPECTED_M2_PERSISTENT_FAILURE_SET = {("A1", 67)}

PRIMARY_IMRMC_METHOD = "iMRMC_uStat11_conditionalD"
SUPPLEMENTAL_IMRMC_METHOD = "iMRMC_uStat11_jointD"


def read_gate(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def mc_metrics(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        raise RuntimeError("Cannot compute Monte Carlo metrics on an empty dataframe.")
    err = df["estimate"].to_numpy(float) - df["truth"].to_numpy(float)
    n = len(df)
    sq = err ** 2
    rmse = float(np.sqrt(np.mean(sq)))
    cov = (
        (df["ci_low"].to_numpy(float) <= df["truth"].to_numpy(float))
        & (df["truth"].to_numpy(float) <= df["ci_high"].to_numpy(float))
    ).astype(float)
    width = df["ci_high"].to_numpy(float) - df["ci_low"].to_numpy(float)

    return {
        "n_valid": int(n),
        "bias": float(err.mean()),
        "mcse_bias": float(err.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
        "rmse": rmse,
        "mcse_rmse": (
            float(sq.std(ddof=1) / np.sqrt(n) / (2 * rmse))
            if n > 1 and rmse > 0
            else np.nan
        ),
        "coverage": float(cov.mean()),
        "mcse_coverage": float(np.sqrt(cov.mean() * (1 - cov.mean()) / n)),
        "mean_width": float(width.mean()),
        "mcse_width": float(width.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
    }


def require_columns(df: pd.DataFrame, cols, label: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")


def validate_three_target_summary(df: pd.DataFrame, label: str):
    require_columns(df, ["target", "estimate", "ci_low", "ci_high"], label)
    if len(df) != 3:
        raise RuntimeError(f"{label} expected exactly 3 target rows, found {len(df)}")
    if set(df["target"]) != set(TARGETS):
        raise RuntimeError(f"{label} target set mismatch: {set(df['target'])}")
    if not np.isfinite(df[["estimate", "ci_low", "ci_high"]].to_numpy(float)).all():
        raise RuntimeError(f"{label} contains non-finite point/interval results")


def choose_m1(primary_root: Path, retry_root: Path, sid: str, rep: int):
    pg = primary_root / "M1" / sid / f"rep_{rep:03d}_gate.json"
    if not pg.exists():
        raise RuntimeError(f"Missing primary M1 gate: {pg}")
    p = read_gate(pg)
    if p.get("computational_ok", False):
        return primary_root, "primary", p

    rg = retry_root / "M1" / sid / f"rep_{rep:03d}_gate.json"
    if not rg.exists():
        raise RuntimeError(f"Primary M1 failed and retry gate missing: {sid} rep {rep}")
    r = read_gate(rg)
    if not r.get("computational_ok", False):
        raise RuntimeError(f"M1 retry failed computational gate: {sid} rep {rep}")
    return retry_root, "retry", r


def adjudicate_m2(
    primary_root: Path, retry_root: Path, sid: str, rep: int
):
    """Return (source_root, fit_source, gate, final_ok, primary_failed, retry_attempted)."""
    pg = primary_root / "M2" / sid / f"rep_{rep:03d}_gate.json"
    if not pg.exists():
        raise RuntimeError(f"Missing primary M2 gate: {pg}")
    p = read_gate(pg)
    if p.get("computational_ok", False):
        return primary_root, "primary", p, True, False, False

    rg = retry_root / "M2" / sid / f"rep_{rep:03d}_gate.json"
    if not rg.exists():
        raise RuntimeError(f"Primary M2 failed and retry gate missing: {sid} rep {rep}")
    r = read_gate(rg)
    if r.get("computational_ok", False):
        return retry_root, "retry", r, True, True, True
    return retry_root, "retry_persistent_failure", r, False, True, True


def add_truth(df: pd.DataFrame, tmap: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    truth = []
    for _, r in out.iterrows():
        key = (str(r["scenario"]), int(r["rep"]))
        target = str(r["target"])
        try:
            truth.append(float(tmap.loc[key, target]))
        except Exception as e:
            raise RuntimeError(
                f"Could not retrieve truth for scenario={key[0]} rep={key[1]} target={target}"
            ) from e
    out["truth"] = truth
    return out


def compute_metric_table(est: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (comparator, sid, target), g in est.groupby(["comparator", "scenario", "target"]):
        valid = g[g["computational_ok"] == True].copy()
        rows.append(
            {
                "comparator": comparator,
                "scenario": sid,
                "target": target,
                "n_total": int(len(g)),
                "n_computational_fail": int((g["computational_ok"] == False).sum()),
                **mc_metrics(valid),
            }
        )
    return pd.DataFrame(rows)


def ratio_table(metrics: pd.DataFrame, comparators) -> pd.DataFrame:
    rows = []
    for comparator in comparators:
        for sid in SCENARIOS:
            z = metrics[
                (metrics["comparator"] == comparator)
                & (metrics["scenario"] == sid)
            ].set_index("target")
            if not set(TARGETS).issubset(z.index):
                raise RuntimeError(f"Missing ratio targets for {comparator}, {sid}")
            abs_rmse = 0.5 * (
                float(z.loc["theta_U_selected", "rmse"])
                + float(z.loc["theta_E_selected", "rmse"])
            )
            ratio = float(z.loc["delta_E_minus_U_selected", "rmse"]) / abs_rmse
            rows.append(
                {
                    "comparator": comparator,
                    "scenario": sid,
                    "rmse_delta": float(z.loc["delta_E_minus_U_selected", "rmse"]),
                    "mean_rmse_absolute": abs_rmse,
                    "ratio": ratio,
                    "n_delta": int(z.loc["delta_E_minus_U_selected", "n_valid"]),
                    "n_theta_U": int(z.loc["theta_U_selected", "n_valid"]),
                    "n_theta_E": int(z.loc["theta_E_selected", "n_valid"]),
                }
            )
    return pd.DataFrame(rows)


def frozen_gate_from_ratios(rdf: pd.DataFrame) -> dict:
    q = rdf[rdf["comparator"].isin(["M1_reader_only", "M2_shared_detectability"])].copy()
    if len(q) != 16:
        raise RuntimeError(f"Frozen comparative-stability rule requires 16 M1/M2 cells, found {len(q)}")
    median_ratio = float(q["ratio"].median())
    n_lt_1 = int((q["ratio"] < 1).sum())
    return {
        "definition": (
            "R = RMSE(Delta) / mean(RMSE(theta_U), RMSE(theta_E)); "
            "support comparative stability only if median R across 16 M1/M2 x A1-A8 cells <= 0.50 "
            "and at least 12/16 cells have R < 1."
        ),
        "median_ratio": median_ratio,
        "n_ratio_lt_1": n_lt_1,
        "n_cells": 16,
        "criterion_median_le_0_50": bool(median_ratio <= 0.50),
        "criterion_at_least_12_of_16_lt_1": bool(n_lt_1 >= 12),
        "overall_pass": bool(median_ratio <= 0.50 and n_lt_1 >= 12),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-root", type=Path, required=True)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--m1-retry-root", type=Path, required=True)
    ap.add_argument("--m2-retry-root", type=Path, required=True)
    ap.add_argument("--m0-csv", type=Path, required=True)
    ap.add_argument("--imrmc-csv", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    truth_path = args.freeze_root / "audit" / "truth_by_scenario_rep.csv"
    if not truth_path.exists():
        raise RuntimeError(f"Missing frozen truth audit: {truth_path}")
    truth = pd.read_csv(truth_path)
    require_columns(
        truth,
        ["scenario", "rep"] + TARGETS,
        "truth_by_scenario_rep.csv",
    )
    if len(truth) != 800:
        raise RuntimeError(f"Expected 800 frozen truth rows, found {len(truth)}")
    tmap = truth.set_index(["scenario", "rep"])

    point_parts = []
    computational = []
    m4_parts = []

    # ------------------------------------------------------------------
    # Raw M0 benchmark
    # ------------------------------------------------------------------
    m0 = pd.read_csv(args.m0_csv)
    require_columns(
        m0,
        ["scenario", "rep", "target", "estimate", "ci_low", "ci_high"],
        "M0 raw benchmark",
    )
    if len(m0) != 2400:
        raise RuntimeError(f"Expected 2400 M0 rows (800 x 3), found {len(m0)}")
    m0 = m0.copy()
    m0["comparator"] = "M0_raw_equal_reader"
    m0["fit_source"] = "frozen_local_M0"
    m0["computational_ok"] = True
    point_parts.append(m0)

    # ------------------------------------------------------------------
    # M1 adjudication: 767 primary + 33 successful retries expected.
    # M4 is paired to the adjudicated M1 posterior.
    # ------------------------------------------------------------------
    m1_retry_used = 0
    m1_primary_failed = 0

    for sid in SCENARIOS:
        for rep in range(100):
            primary_gate = read_gate(
                args.results_root / "M1" / sid / f"rep_{rep:03d}_gate.json"
            )
            if not primary_gate.get("computational_ok", False):
                m1_primary_failed += 1

            src, source_label, gate = choose_m1(
                args.results_root, args.m1_retry_root, sid, rep
            )
            if source_label == "retry":
                m1_retry_used += 1

            d = src / "M1" / sid
            sp = d / f"rep_{rep:03d}_summary.csv"
            mp = d / f"rep_{rep:03d}_m4_sensitivity.csv"
            if not sp.exists() or not mp.exists():
                raise RuntimeError(
                    f"Missing adjudicated M1/M4 output for {sid} rep {rep} from {source_label}"
                )

            s = pd.read_csv(sp)
            validate_three_target_summary(s, f"M1 {sid} rep {rep}")
            s["scenario"] = sid
            s["rep"] = rep
            s["comparator"] = "M1_reader_only"
            s["fit_source"] = source_label
            s["computational_ok"] = True
            point_parts.append(s)

            m4 = pd.read_csv(mp)
            require_columns(m4, ["target", "estimate", "ci_low", "ci_high"], f"M4 {sid} rep {rep}")
            if set(m4["target"]) != set(TARGETS):
                raise RuntimeError(f"M4 target mismatch for {sid} rep {rep}")
            m4["scenario"] = sid
            m4["rep"] = rep
            m4["m1_fit_source"] = source_label
            m4_parts.append(m4)

            computational.append(
                {
                    "comparator": "M1_reader_only",
                    "scenario": sid,
                    "rep": rep,
                    "fit_source": source_label,
                    "primary_failed": bool(not primary_gate.get("computational_ok", False)),
                    "retry_attempted": bool(source_label == "retry"),
                    "computational_ok": True,
                    "divergences": int(gate["divergences"]),
                    "max_rhat": float(gate["max_rhat_scalar"]),
                    "min_bulk": float(gate["min_ess_bulk_scalar"]),
                    "min_tail": float(gate["min_ess_tail_scalar"]),
                    "elapsed_sec": float(gate["elapsed_sec"]),
                }
            )

    if m1_primary_failed != EXPECTED_M1_PRIMARY_FAILURES:
        raise RuntimeError(
            f"Expected {EXPECTED_M1_PRIMARY_FAILURES} primary M1 failures, found {m1_primary_failed}"
        )
    if m1_retry_used != EXPECTED_M1_PRIMARY_FAILURES:
        raise RuntimeError(
            f"Expected {EXPECTED_M1_PRIMARY_FAILURES} successful M1 retries used, found {m1_retry_used}"
        )

    # ------------------------------------------------------------------
    # M2 adjudication:
    # 798 primary pass, A7/056 retry pass, A1/067 persistent failure.
    # Persistent failure is retained for audit but excluded from calibration.
    # ------------------------------------------------------------------
    m2_primary_fail_set = set()
    m2_retry_pass_set = set()
    m2_persistent_fail_set = set()

    for sid in SCENARIOS:
        for rep in range(100):
            primary_gate_path = (
                args.results_root / "M2" / sid / f"rep_{rep:03d}_gate.json"
            )
            primary_gate = read_gate(primary_gate_path)
            if not primary_gate.get("computational_ok", False):
                m2_primary_fail_set.add((sid, rep))

            src, source_label, gate, final_ok, primary_failed, retry_attempted = adjudicate_m2(
                args.results_root, args.m2_retry_root, sid, rep
            )
            if primary_failed and final_ok:
                m2_retry_pass_set.add((sid, rep))
            if primary_failed and not final_ok:
                m2_persistent_fail_set.add((sid, rep))

            d = src / "M2" / sid
            sp = d / f"rep_{rep:03d}_summary.csv"
            if not sp.exists():
                raise RuntimeError(f"Missing adjudicated M2 summary: {sp}")
            s = pd.read_csv(sp)
            validate_three_target_summary(s, f"M2 {sid} rep {rep}")
            s["scenario"] = sid
            s["rep"] = rep
            s["comparator"] = "M2_shared_detectability"
            s["fit_source"] = source_label
            s["computational_ok"] = bool(final_ok)
            point_parts.append(s)

            computational.append(
                {
                    "comparator": "M2_shared_detectability",
                    "scenario": sid,
                    "rep": rep,
                    "fit_source": source_label,
                    "primary_failed": bool(primary_failed),
                    "retry_attempted": bool(retry_attempted),
                    "computational_ok": bool(final_ok),
                    "divergences": int(gate["divergences"]),
                    "max_rhat": float(gate["max_rhat_scalar"]),
                    "min_bulk": float(gate["min_ess_bulk_scalar"]),
                    "min_tail": float(gate["min_ess_tail_scalar"]),
                    "elapsed_sec": float(gate["elapsed_sec"]),
                }
            )

    if m2_primary_fail_set != EXPECTED_M2_PRIMARY_FAILURE_SET:
        raise RuntimeError(
            f"M2 primary-failure set mismatch. Expected {EXPECTED_M2_PRIMARY_FAILURE_SET}, "
            f"found {m2_primary_fail_set}"
        )
    if m2_retry_pass_set != EXPECTED_M2_RETRY_PASS_SET:
        raise RuntimeError(
            f"M2 retry-pass set mismatch. Expected {EXPECTED_M2_RETRY_PASS_SET}, "
            f"found {m2_retry_pass_set}"
        )
    if m2_persistent_fail_set != EXPECTED_M2_PERSISTENT_FAILURE_SET:
        raise RuntimeError(
            f"M2 persistent-failure set mismatch. Expected {EXPECTED_M2_PERSISTENT_FAILURE_SET}, "
            f"found {m2_persistent_fail_set}"
        )

    # ------------------------------------------------------------------
    # Actual iMRMC comparator
    # ------------------------------------------------------------------
    im = pd.read_csv(args.imrmc_csv)
    require_columns(
        im,
        ["method", "scenario", "rep", "target", "estimate", "ci_low", "ci_high"],
        "iMRMC aggregate",
    )
    if len(im) != 4800:
        raise RuntimeError(f"Expected 4800 iMRMC rows (2 methods x 800 x 3), found {len(im)}")

    method_map = {
        PRIMARY_IMRMC_METHOD: "iMRMC_conditionalD",
        SUPPLEMENTAL_IMRMC_METHOD: "iMRMC_jointD",
    }
    if set(im["method"]) != set(method_map):
        raise RuntimeError(f"Unexpected iMRMC methods: {set(im['method'])}")
    im = im.copy()
    im["comparator"] = im["method"].map(method_map)
    im["fit_source"] = "actual_iMRMC_2.1.0"
    im["computational_ok"] = True
    point_parts.append(im)

    # ------------------------------------------------------------------
    # Unified long-form result file with frozen truth attached
    # ------------------------------------------------------------------
    est = pd.concat(point_parts, ignore_index=True, sort=False)
    est = add_truth(est, tmap)

    # Expect 5 point-estimator comparators:
    # raw + two iMRMC formulations + M1 + M2.
    expected_counts = {
        "M0_raw_equal_reader": 2400,
        "iMRMC_conditionalD": 2400,
        "iMRMC_jointD": 2400,
        "M1_reader_only": 2400,
        "M2_shared_detectability": 2400,
    }
    got_counts = est.groupby("comparator").size().to_dict()
    if got_counts != expected_counts:
        raise RuntimeError(f"Unified comparator row counts mismatch: {got_counts}")

    est.to_csv(
        args.outdir / "all_point_interval_results_definitive_adjudicated.csv",
        index=False,
    )

    compdf = pd.DataFrame(computational)
    compdf.to_csv(
        args.outdir / "bayesian_computational_diagnostics_definitive.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Model-specific valid-set calibration metrics
    # ------------------------------------------------------------------
    met = compute_metric_table(est)
    met.to_csv(
        args.outdir / "adversarial_metrics_all_point_comparators.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Paired-complete sensitivity:
    # intersection of computationally valid replicate IDs across ALL point
    # comparators, scenario by scenario. A1 should have 99 because M2 A1/067
    # is persistently invalid; A2-A8 should have 100.
    # ------------------------------------------------------------------
    paired_parts = []
    paired_counts = {}
    comparators = list(expected_counts)

    for sid in SCENARIOS:
        valid_sets = []
        for comparator in comparators:
            z = est[
                (est["comparator"] == comparator)
                & (est["scenario"] == sid)
                & (est["computational_ok"] == True)
            ]
            valid_sets.append(set(z["rep"].astype(int).unique()))
        common = set.intersection(*valid_sets)
        paired_counts[sid] = len(common)
        for comparator in comparators:
            z = est[
                (est["comparator"] == comparator)
                & (est["scenario"] == sid)
                & (est["rep"].astype(int).isin(common))
            ].copy()
            paired_parts.append(z)

    expected_paired_counts = {"A1": 99, **{f"A{i}": 100 for i in range(2, 9)}}
    if paired_counts != expected_paired_counts:
        raise RuntimeError(
            f"Unexpected paired-complete replicate counts: {paired_counts}; "
            f"expected {expected_paired_counts}"
        )

    paired = pd.concat(paired_parts, ignore_index=True)
    paired.to_csv(
        args.outdir / "paired_complete_point_results.csv",
        index=False,
    )
    paired_met = compute_metric_table(paired)
    paired_met.to_csv(
        args.outdir / "paired_complete_metrics_all_point_comparators.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Frozen comparative-stability rule: M1 + M2 only.
    # Main version uses each model's computationally valid set, exactly as
    # the prespecified calibration summaries do.
    # A paired-complete sensitivity is also reported.
    # ------------------------------------------------------------------
    ratios = ratio_table(
        met,
        ["M1_reader_only", "M2_shared_detectability"],
    )
    ratios.to_csv(
        args.outdir / "comparative_stability_ratios_prespecified_valid_sets.csv",
        index=False,
    )
    frozen_gate = frozen_gate_from_ratios(ratios)
    (args.outdir / "comparative_stability_gate_prespecified.json").write_text(
        json.dumps(frozen_gate, indent=2)
    )

    paired_ratios = ratio_table(
        paired_met,
        ["M1_reader_only", "M2_shared_detectability"],
    )
    paired_ratios.to_csv(
        args.outdir / "comparative_stability_ratios_paired_complete_sensitivity.csv",
        index=False,
    )
    paired_gate = frozen_gate_from_ratios(paired_ratios)
    paired_gate["note"] = (
        "Sensitivity analysis using the common computationally valid replicate set "
        "for all point comparators within each scenario; A1 excludes rep 067."
    )
    (args.outdir / "comparative_stability_gate_paired_complete_sensitivity.json").write_text(
        json.dumps(paired_gate, indent=2)
    )

    # ------------------------------------------------------------------
    # M4 fixed-grid sensitivity using adjudicated M1 posterior only.
    # No oracle gamma selection and no M4 point-estimator RMSE ranking.
    # ------------------------------------------------------------------
    m4df = pd.concat(m4_parts, ignore_index=True)
    m4df.to_csv(
        args.outdir / "m4_fixed_grid_all_replicates_adjudicated.csv",
        index=False,
    )

    m4_rep = []
    for (sid, rep, target), z in m4df.groupby(["scenario", "rep", "target"]):
        truthv = float(tmap.loc[(sid, int(rep)), target])
        m4_rep.append(
            {
                "scenario": sid,
                "rep": int(rep),
                "target": target,
                "truth": truthv,
                "estimate_min": float(z["estimate"].min()),
                "estimate_max": float(z["estimate"].max()),
                "estimate_span": float(z["estimate"].max() - z["estimate"].min()),
                "ci_envelope_low": float(z["ci_low"].min()),
                "ci_envelope_high": float(z["ci_high"].max()),
                "truth_in_any_grid_CI": bool(
                    ((z["ci_low"] <= truthv) & (truthv <= z["ci_high"])).any()
                ),
                "truth_in_full_CI_envelope": bool(
                    z["ci_low"].min() <= truthv <= z["ci_high"].max()
                ),
            }
        )

    m4rep = pd.DataFrame(m4_rep)
    if len(m4rep) != 2400:
        raise RuntimeError(f"Expected 2400 M4 replicate-target rows, found {len(m4rep)}")
    m4rep.to_csv(
        args.outdir / "m4_grid_sensitivity_replication_level_adjudicated.csv",
        index=False,
    )

    m4sum = (
        m4rep.groupby(["scenario", "target"])
        .agg(
            n=("rep", "size"),
            mean_span=("estimate_span", "mean"),
            median_span=("estimate_span", "median"),
            any_grid_CI_truth_containment=("truth_in_any_grid_CI", "mean"),
            full_envelope_truth_containment=("truth_in_full_CI_envelope", "mean"),
        )
        .reset_index()
    )
    m4sum.to_csv(
        args.outdir / "m4_grid_sensitivity_summary_adjudicated.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Per-scenario computational counts and final audit.
    # ------------------------------------------------------------------
    comp_summary = (
        compdf.groupby(["comparator", "scenario"])
        .agg(
            n_total=("rep", "size"),
            n_primary_failed=("primary_failed", "sum"),
            n_retry_attempted=("retry_attempted", "sum"),
            n_final_computational_fail=("computational_ok", lambda x: int((~x.astype(bool)).sum())),
        )
        .reset_index()
    )
    comp_summary.to_csv(
        args.outdir / "bayesian_computational_summary_by_scenario.csv",
        index=False,
    )

    audit = {
        "frozen_datasets": 800,
        "M0_raw": {"n_total": 800, "n_valid": 800},
        "iMRMC_conditionalD": {"n_total": 800, "n_valid": 800},
        "iMRMC_jointD": {"n_total": 800, "n_valid": 800, "role": "supplemental"},
        "M1": {
            "n_total": 800,
            "primary_failures": m1_primary_failed,
            "retry_used": m1_retry_used,
            "final_computational_failures": EXPECTED_M1_FINAL_FAILURES,
            "n_valid": 800,
        },
        "M2": {
            "n_total": 800,
            "primary_failure_set": sorted([f"{s}/rep_{r:03d}" for s, r in m2_primary_fail_set]),
            "retry_pass_set": sorted([f"{s}/rep_{r:03d}" for s, r in m2_retry_pass_set]),
            "persistent_failure_set": sorted([f"{s}/rep_{r:03d}" for s, r in m2_persistent_fail_set]),
            "final_computational_failures": 1,
            "n_valid": 799,
        },
        "M4": {
            "n_total": 800,
            "role": "fixed-grid sensitivity only; excluded from point-estimator RMSE ranking",
        },
        "paired_complete_counts_by_scenario": paired_counts,
        "comparative_stability_gate_prespecified": frozen_gate,
        "comparative_stability_gate_paired_complete_sensitivity": paired_gate,
    }
    (args.outdir / "DEFINITIVE_ADJUDICATION_AUDIT.json").write_text(
        json.dumps(audit, indent=2)
    )

    # ------------------------------------------------------------------
    # Compact human-readable status report. This reports numbers, not a
    # scientific interpretation beyond the prespecified rule.
    # ------------------------------------------------------------------
    lines = []
    lines.append("# NuCLS v24.5 definitive adversarial comparison — aggregation audit")
    lines.append("")
    lines.append("- Frozen adversarial datasets: 800/800.")
    lines.append(f"- M1: 800/800 computationally valid after {m1_retry_used} adjudicated retries.")
    lines.append(
        "- M2: 799/800 computationally valid; A7/rep_056 uses the successful retry; "
        "A1/rep_067 remains a persistent computational failure and is excluded from M2 calibration metrics."
    )
    lines.append("- Actual iMRMC conditionalD and jointD: 800/800 each.")
    lines.append("- M4: 800 fixed-grid sensitivity analyses paired to the adjudicated M1 posterior.")
    lines.append("")
    lines.append("## Prespecified comparative-stability rule")
    lines.append(f"- Median R across 16 M1/M2 scenario cells: {frozen_gate['median_ratio']:.6f}")
    lines.append(f"- Cells with R < 1: {frozen_gate['n_ratio_lt_1']}/16")
    lines.append(f"- Prespecified rule passed: {frozen_gate['overall_pass']}")
    lines.append("")
    lines.append("## Paired-complete sensitivity")
    lines.append(f"- Median R: {paired_gate['median_ratio']:.6f}")
    lines.append(f"- Cells with R < 1: {paired_gate['n_ratio_lt_1']}/16")
    lines.append(f"- Rule passed on common-valid-set sensitivity: {paired_gate['overall_pass']}")
    lines.append("")
    lines.append(
        "No M4 oracle-gamma selection was performed. The persistent M2 failure was classified using "
        "computational diagnostics only, without reference to simulation truth."
    )
    (args.outdir / "DEFINITIVE_AGGREGATION_STATUS.md").write_text("\n".join(lines) + "\n")

    print("V24_5_DEFINITIVE_ADVERSARIAL_AGGREGATION_COMPLETE")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
