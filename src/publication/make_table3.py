from pathlib import Path
import os
import json
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("NUCLS_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
SRC = ROOT / "outputs/v24_5_definitive_adversarial_comparison"
OUT = ROOT / "tables/simulation"
OUT.mkdir(parents=True, exist_ok=True)

metrics = pd.read_csv(SRC / "adversarial_metrics_all_point_comparators.csv")
ratios = pd.read_csv(SRC / "comparative_stability_ratios_prespecified_valid_sets.csv")
m4 = pd.read_csv(SRC / "m4_grid_sensitivity_summary_adjudicated.csv")
comp = pd.read_csv(SRC / "bayesian_computational_summary_by_scenario.csv")

scenarios = [f"A{i}" for i in range(1, 9)]
departures = {
    "A1": "Independent AI-specific item variation",
    "A2": "Correlated but nonidentical AI/human factors",
    "A3": "Nonlinear/threshold AI response",
    "A4": "Condition-specific AI-human dependence",
    "A5": "Omitted slide/patient effect",
    "A6": "Informative FOV missingness",
    "A7": "Selection on AI/human positivity",
    "A8": "Mixture/heavy-tailed heterogeneity",
}
models = [
    ("M1_reader_only", "M1 reader-only"),
    ("M2_shared_detectability", "M2 shared-detectability"),
]
targets = {
    "theta_U": "theta_U_selected",
    "theta_E": "theta_E_selected",
    "delta": "delta_E_minus_U_selected",
}

# Structural validation of source files.
assert set(scenarios) == set(metrics["scenario"].unique())
assert set(targets.values()) <= set(metrics["target"].unique())
assert set(x[0] for x in models) == set(ratios["comparator"].unique())
assert len(ratios) == 16
m4d = m4[m4["target"].eq(targets["delta"])].copy()
assert len(m4d) == 8

rows = []
for sc in scenarios:
    for model_code, model_label in models:
        sub = metrics[(metrics["scenario"] == sc) & (metrics["comparator"] == model_code)]
        assert len(sub) == 3
        u = sub[sub["target"].eq(targets["theta_U"])].iloc[0]
        e = sub[sub["target"].eq(targets["theta_E"])].iloc[0]
        d = sub[sub["target"].eq(targets["delta"])].iloc[0]
        rr = ratios[(ratios["scenario"] == sc) & (ratios["comparator"] == model_code)].iloc[0]
        mm = m4d[m4d["scenario"].eq(sc)].iloc[0]
        cc = comp[(comp["scenario"] == sc) & (comp["comparator"] == model_code)].iloc[0]
        rows.append({
            "scenario": sc,
            "departure": departures[sc],
            "model": model_label,
            "model_code": model_code,
            "n_valid": int(d["n_valid"]),
            "n_final_computational_fail": int(cc["n_final_computational_fail"]),
            "theta_U_bias_pp": 100 * float(u["bias"]),
            "theta_U_rmse_pp": 100 * float(u["rmse"]),
            "theta_U_coverage_pct": 100 * float(u["coverage"]),
            "theta_E_bias_pp": 100 * float(e["bias"]),
            "theta_E_rmse_pp": 100 * float(e["rmse"]),
            "theta_E_coverage_pct": 100 * float(e["coverage"]),
            "delta_bias_pp": 100 * float(d["bias"]),
            "delta_rmse_pp": 100 * float(d["rmse"]),
            "delta_coverage_pct": 100 * float(d["coverage"]),
            "R_delta_vs_absolute": float(rr["ratio"]),
            "m4_delta_mean_span_pp": 100 * float(mm["mean_span"]),
            "m4_delta_full_envelope_containment_pct": 100 * float(mm["full_envelope_truth_containment"]),
        })

tab = pd.DataFrame(rows)

# Prespecified comparative-stability gate.
median_R = float(ratios["ratio"].median())
n_lt1 = int((ratios["ratio"] < 1).sum())
ge1 = ratios.loc[ratios["ratio"] >= 1, ["comparator", "scenario", "ratio"]].copy()
assert abs(median_R - 0.2954470999) < 1e-8
assert n_lt1 == 14
assert set(ge1["scenario"]) == {"A4"} and len(ge1) == 2
assert set(ge1["comparator"]) == {"M1_reader_only", "M2_shared_detectability"}
assert np.allclose(
    tab.loc[tab["scenario"].eq("A4"), "delta_coverage_pct"].to_numpy(),
    [0.0, 0.0],
)
assert int(tab["n_final_computational_fail"].sum()) == 1
assert int(tab.loc[(tab["scenario"].eq("A1")) & (tab["model_code"].eq("M2_shared_detectability")), "n_valid"].iloc[0]) == 99
assert int(tab.loc[(tab["scenario"].eq("A1")) & (tab["model_code"].eq("M2_shared_detectability")), "n_final_computational_fail"].iloc[0]) == 1

main_csv = OUT / "Table3_adversarial_robustness_source.csv"
tex_file = OUT / "Table3_adversarial_robustness.tex"
gate_file = OUT / "Table3_adversarial_gate.json"
full_csv = OUT / "TableS_adversarial_full_metrics_source.csv"
m4_csv = OUT / "TableS_M4_adversarial_sensitivity_source.csv"

tab.to_csv(main_csv, index=False)
metrics.to_csv(full_csv, index=False)
m4.to_csv(m4_csv, index=False)

gate = {
    "definition": "R = RMSE(Delta) / mean{RMSE(theta_U), RMSE(theta_E)}",
    "prespecified_rule": {
        "median_R_le_0_50": True,
        "at_least_12_of_16_R_lt_1": True,
    },
    "median_R": median_R,
    "n_R_lt_1": n_lt1,
    "n_cells": 16,
    "cells_R_ge_1": ge1.to_dict(orient="records"),
    "paired_complete_sensitivity": {
        "median_R": 0.2964455811,
        "n_R_lt_1": 14,
        "n_cells": 16,
        "result": "PASS",
    },
    "interpretation": (
        "The condition contrast was comparatively more stable than the two "
        "absolute agreement levels in 7 of 8 adversarial scenarios for both "
        "M1 and M2. A4 condition-specific AI-human dependence was the common "
        "failure boundary. This is not a method-superiority claim."
    ),
}
gate_file.write_text(json.dumps(gate, indent=2) + "\n")

# Publication LaTeX. Metrics shown in percentage points; coverage/containment in percent.
bs = chr(92)
nl = chr(10)
L = [
    bs + "begin{table*}[!htbp]",
    bs + "centering",
    bs + "scriptsize",
    bs + "caption{Adversarial misspecification summary for the reader-only (M1) and shared-detectability (M2) models. RMSE and bias are reported in percentage points; bracketed values are empirical 95" + bs + "% interval coverage percentages. The stability ratio is $R={"
    + bs + "mathrm{RMSE}(" + bs + "Delta)}/[{"
    + bs + "mathrm{RMSE}(" + bs + "theta_U)+"
    + bs + "mathrm{RMSE}(" + bs + "theta_E)}/2]$. M4 is a prespecified residual-dependence sensitivity grid rather than an oracle-selected estimator; its column gives full-envelope truth containment for $"
    + bs + "Delta$.}",
    bs + "label{tab:adversarial_robustness}",
    bs + "resizebox{" + bs + "textwidth}{!}{%",
    bs + "begin{tabular}{lllrccccc}",
    bs + "toprule",
    "Scenario & Departure challenged & Model & $n$ & $"
    + bs + "theta_U$ RMSE [Cov.] & $"
    + bs + "theta_E$ RMSE [Cov.] & $"
    + bs + "Delta$ Bias / RMSE [Cov.] & $R$ & M4 $"
    + bs + "Delta$ envelope " + bs*2,
    bs + "midrule",
]

for _, z in tab.iterrows():
    sc = str(z["scenario"])
    dep = str(z["departure"]).replace("AI-human", "AI--human")
    sc_tex = bs + "textbf{" + sc + "}" if sc == "A4" else sc
    dep_tex = bs + "textbf{" + dep + "}" if sc == "A4" else dep
    model = str(z["model"]).replace("-", "--")
    ucell = f'{z["theta_U_rmse_pp"]:.3f} [{z["theta_U_coverage_pct"]:.0f}]'
    ecell = f'{z["theta_E_rmse_pp"]:.3f} [{z["theta_E_coverage_pct"]:.0f}]'
    dcell = f'{z["delta_bias_pp"]:+.3f} / {z["delta_rmse_pp"]:.3f} [{z["delta_coverage_pct"]:.0f}]'
    rcell = f'{z["R_delta_vs_absolute"]:.3f}'
    if sc == "A4":
        rcell = bs + "textbf{" + rcell + "}"
        dcell = bs + "textbf{" + dcell + "}"
    m4cell = f'{z["m4_delta_full_envelope_containment_pct"]:.0f}' + bs + "%"
    L.append(
        f'{sc_tex} & {dep_tex} & {model} & {int(z["n_valid"])} & '
        f'{ucell} & {ecell} & {dcell} & {rcell} & {m4cell} ' + bs*2
    )
    if str(z["model_code"]) == "M2_shared_detectability" and sc != "A8":
        L.append(bs + "addlinespace[1.5pt]")

L += [
    bs + "bottomrule",
    bs + "end{tabular}%",
    "}",
    bs + "vspace{2pt}",
    bs + "begin{minipage}{0.99" + bs + "textwidth}",
    bs + "footnotesize",
    bs + "textit{Note.} The prespecified comparative-stability gate required median $R"
    + bs + "leq0.50$ across the 16 M1/M2 scenario cells and at least 12/16 cells with $R<1$. "
    + f"The observed median was {median_R:.3f}, with {n_lt1}/16 cells below 1; the paired-complete sensitivity also passed (median $R=0.296$, 14/16 below 1). "
    + "A4 was the only scenario with $R" + bs + "geq1$ for both M1 and M2, and both Bayesian $"
    + bs + "Delta$ intervals had 0" + bs + "% empirical coverage there. "
    + "M2 A1 uses 99 valid replications because one prespecified computational failure remained after targeted retry. "
    + "The table is a robustness-boundary summary, not a ranking of methods. Full bias, RMSE, interval-width, Monte Carlo standard-error, raw/iMRMC, and M4 grid results are retained for the supplement.",
    bs + "end{minipage}",
    bs + "end{table*}",
]
tex_file.write_text(nl.join(L) + nl)

print("TABLE3_COMPLETE")
print(f"median_R = {median_R:.10f}")
print(f"R<1 = {n_lt1}/16")
print("R>=1 cells:")
print(ge1.to_string(index=False))
print("\n===== TABLE 3 SOURCE =====")
show_cols = [
    "scenario","departure","model","n_valid",
    "theta_U_rmse_pp","theta_U_coverage_pct",
    "theta_E_rmse_pp","theta_E_coverage_pct",
    "delta_bias_pp","delta_rmse_pp","delta_coverage_pct",
    "R_delta_vs_absolute","m4_delta_full_envelope_containment_pct",
    "n_final_computational_fail",
]
print(tab[show_cols].to_string(index=False))
print("\n===== FILES =====")
for f in [main_csv, tex_file, gate_file, full_csv, m4_csv]:
    print(f.name, f.stat().st_size)
print("\n===== LATEX TAIL =====")
print(repr(tex_file.read_text()[-220:]))
