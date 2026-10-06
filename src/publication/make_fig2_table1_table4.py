from __future__ import annotations

import argparse, json

from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt



def fmt3(x):

    return f"{float(x):.3f}"



def fmt_ci(point, lo, hi):

    return f"{float(point):.3f} [{float(lo):.3f}, {float(hi):.3f}]"



def make_figure2(root: Path):

    src = root / "outputs/v25_1_phase4_descriptive_audit"

    pair_src = root / "publication_sources/descriptive/Figure2_paired_switching_source.csv"

    out = root / "figures/descriptive"

    out.mkdir(parents=True, exist_ok=True)



    raw = pd.read_csv(src / "raw_agreement_by_condition_ai_status.csv")

    pair = pd.read_csv(pair_src)



    def rate(condition, ai):

        z = raw[(raw["condition"] == condition) & (raw["ai_detected"].astype(str).str.lower() == str(ai).lower())]

        if len(z) != 1:

            # handle bool/int parsing robustly

            val = 1 if ai else 0

            z = raw[(raw["condition"] == condition) & (raw["ai_detected"].astype(int) == val)]

        return float(z.iloc[0]["agreement_rate"])



    conds = ["Unbiased", "Evaluation"]

    neg = [rate("Unbiased", False), rate("Evaluation", False)]

    pos = [rate("Unbiased", True), rate("Evaluation", True)]



    state_map = {

        "agree_both": "Agree in both",

        "evaluation_only_agrees": "Evaluation only",

        "unbiased_only_agrees": "Unbiased only",

        "agree_neither": "Agree in neither",

    }

    order = ["agree_both", "evaluation_only_agrees", "unbiased_only_agrees", "agree_neither"]

    p = pair.set_index("pair_agreement_state").loc[order]

    counts = p["count"].astype(int).to_numpy()

    vals = p["proportion"].astype(float).to_numpy()



    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8))



    ax = axs[0]

    x = np.arange(2)

    w = .34

    b1 = ax.bar(x-w/2, neg, w, label="AI negative")

    b2 = ax.bar(x+w/2, pos, w, label="AI positive")

    ax.set_xticks(x)

    ax.set_xticklabels(conds)

    ax.set_ylim(0, 1)

    ax.set_ylabel("Observed AI-reader agreement proportion")

    ax.set_title("A. Raw agreement by condition and AI status")

    ax.legend(frameon=False, fontsize=8, loc="upper center")

    for bars in (b1, b2):

        for b in bars:

            ax.text(b.get_x()+b.get_width()/2, b.get_height()+.025,

                    f"{100*b.get_height():.1f}%",

                    ha="center", va="bottom", fontsize=8)



    ax = axs[1]

    labels = [state_map[x] for x in order]

    yp = np.arange(4)[::-1]

    ax.barh(yp, vals)

    ax.set_yticks(yp)

    ax.set_yticklabels(labels)

    ax.set_xlim(0, .52)

    ax.set_xlabel("Proportion of paired observations")

    ax.set_title("B. Paired agreement states across conditions")

    for y, v, n in zip(yp, vals, counts):

        ax.text(v+.008, y, f"{100*v:.1f}% (n={n})", va="center", fontsize=8)



    fig.tight_layout()

    for ext in ("pdf", "png"):

        fig.savefig(out / f"Figure2_switching_raw_agreement.{ext}",

                    dpi=400 if ext == "png" else None, bbox_inches="tight")

    plt.close(fig)



def make_table1(root: Path):

    src = root / "outputs/v25_1_phase4_descriptive_audit"

    score = root / "outputs/scoring_v16_1/restricted_recorded_reader_detection_agreement_v16.csv"

    pair_src = root / "publication_sources/descriptive/Figure2_paired_switching_source.csv"

    outdir = root / "tables/descriptive"

    outdir.mkdir(parents=True, exist_ok=True)



    audit = json.loads((src / "audit.json").read_text())

    miss = pd.read_csv(src / "missingness_by_condition_ai_status.csv")

    pair = pd.read_csv(pair_src).set_index("pair_agreement_state")

    d = pd.read_csv(score)



    n_anchors = int(audit["n_anchors"])

    ai_pos = int(audit["ai_positive"])

    ai_neg = n_anchors - ai_pos

    potential = int(audit["n_potential_reader_anchor_condition"])

    observed = int(audit["n_observed"])

    missing = int(audit["n_missing"])

    paired_n = int(audit["paired_n"])

    switches = int(audit["switches"])



    n_fov = int(d["image_id"].nunique()) if "image_id" in d.columns else 52

    n_slides = int(d["slide"].nunique()) if "slide" in d.columns else 5

    n_readers = int(potential // (n_anchors * 2))

    if "slide" in d.columns:

        patients = d["slide"].astype(str).str.split("-").str[:3].str.join("-").nunique()

    else:

        patients = n_slides



    def mrow(condition, ai):

        z = miss[(miss["condition"] == condition) & (miss["ai_detected"].astype(int) == int(ai))]

        if len(z) != 1:

            raise RuntimeError((condition, ai, len(z)))

        return z.iloc[0]



    rows = []

    def add(section, quantity, num, den=None):

        pct = None if den is None else 100.0 * float(num) / float(den)

        rows.append([section, quantity, num, den, pct])



    add("Empirical frame", "Selected anchors", n_anchors)

    add("Empirical frame", "AI-positive anchors", ai_pos, n_anchors)

    add("Empirical frame", "AI-negative anchors", ai_neg, n_anchors)

    add("Empirical frame", "Fields of view", n_fov)

    add("Empirical frame", "Slides", n_slides)

    add("Empirical frame", "Patients", int(patients))

    add("Empirical frame", "Primary readers", n_readers)

    add("Empirical frame", "Potential reader-anchor-condition observations", potential)

    add("Empirical frame", "Observed reader-anchor-condition observations", observed, potential)

    add("Empirical frame", "Unobserved because reader did not annotate FOV", missing, potential)



    for condition in ("Evaluation", "Unbiased"):

        z0, z1 = mrow(condition,0), mrow(condition,1)

        total_missing = int(z0["n_missing"] + z1["n_missing"])

        total_n = int(z0["n_total"] + z1["n_total"])

        add("Missingness", f"{condition}: unobserved", total_missing, total_n)

        add("Missingness", f"{condition}, AI negative: unobserved", int(z0["n_missing"]), int(z0["n_total"]))

        add("Missingness", f"{condition}, AI positive: unobserved", int(z1["n_missing"]), int(z1["n_total"]))



    agree_both = int(pair.loc["agree_both","count"])

    agree_neither = int(pair.loc["agree_neither","count"])

    eval_only = int(pair.loc["evaluation_only_agrees","count"])

    unb_only = int(pair.loc["unbiased_only_agrees","count"])



    add("Paired support", "Potential reader-anchor pairs", potential//2)

    add("Paired support", "Observed in both conditions", paired_n, potential//2)

    add("Paired support", "Agreement in both conditions", agree_both, paired_n)

    add("Paired support", "Agreement in neither condition", agree_neither, paired_n)

    add("Paired support", "Agreement in Evaluation only", eval_only, paired_n)

    add("Paired support", "Agreement in Unbiased only", unb_only, paired_n)

    add("Paired support", "Condition-switching observations", switches, paired_n)



    tab = pd.DataFrame(rows, columns=["Section","Quantity","Numerator","Denominator","Percent"])

    tab.to_csv(outdir / "Table1_empirical_frame_missingness_switching.csv", index=False)



def make_table4(root: Path):

    src = root / "outputs/v25_4_predictive_upgrade"

    outdir = root / "tables/predictive"

    outdir.mkdir(parents=True, exist_ok=True)



    pts = pd.read_csv(src / "predictive_model_summary.csv")

    ci = pd.read_csv(src / "bootstrap_metric_intervals.csv")

    ci = ci[ci["scheme"] == "slide_cluster"].copy()



    specs = [

        ("Held-out AI","heldout_ai","latent_bayes","Latent Bayesian"),

        ("Held-out AI","heldout_ai","nested_regularized_logistic","Regularized logistic"),

        ("Held-out AI","heldout_ai","hist_gradient_boosting","Gradient boosting"),

        ("Held-out human","heldout_human","joint","Joint"),

        ("Held-out human","heldout_human","reader_only","Reader-only"),

        ("Target-aligned agreement","target_aligned_agreement","joint","Joint"),

        ("Target-aligned agreement","target_aligned_agreement","reader_only","Reader-only"),

    ]



    metrics = [

        ("brier","Brier"),

        ("log_loss","Log loss"),

        ("auroc","AUROC"),

        ("auprc","AUPRC"),

        ("calibration_intercept","Calibration intercept"),

        ("calibration_slope","Calibration slope"),

    ]



    rows = []

    for target_label, target, model, model_label in specs:

        p = pts[(pts["target"] == target) & (pts["model"] == model)]

        if len(p) != 1:

            raise RuntimeError((target, model, len(p)))

        p = p.iloc[0]

        row = {

            "Target": target_label,

            "Model": model_label,

            "N": int(p["n"]),

            "Prevalence": fmt3(p["prevalence"]),

        }

        for metric, label in metrics:

            q = ci[(ci["target"] == target) & (ci["model"] == model) & (ci["metric"] == metric)]

            if len(q) != 1:

                raise RuntimeError((target, model, metric, len(q)))

            q = q.iloc[0]

            row[label] = fmt_ci(p[metric], q["q025"], q["q975"])

        rows.append(row)



    cols = ["Target","Model","N","Prevalence","Brier","Log loss","AUROC","AUPRC",

            "Calibration intercept","Calibration slope"]

    pd.DataFrame(rows)[cols].to_csv(outdir / "Table4_predictive_performance_calibration.csv", index=False)



def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=Path, required=True)

    args = ap.parse_args()

    make_figure2(args.root)

    make_table1(args.root)

    make_table4(args.root)

    print("PUBLICATION_OUTPUT_RECONSTRUCTION_COMPLETE")



if __name__ == "__main__":

    main()

