#!/usr/bin/env python3
"""
NuCLS paired same-reader/same-anchor condition audit — v17.

Purpose:
Compare a fixed external AI's detection agreement with the SAME reader on the SAME
selected anchor under Evaluation versus Unbiased annotation conditions.

This removes the differing-reader-composition problem in the unpaired v16 condition
totals. It remains descriptive and noncausal.

Important:
- uses only the 1,144 RGB-eligible selected anchors already scored in v16.1;
- keeps only reader-anchor pairs observed in BOTH conditions;
- excludes DidNotAnnotateFOV in either condition;
- AI detection status is fixed per anchor from the frozen v16.1 match file;
- does not reinterpret P-truth as biological truth;
- performs no independence-based p-values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PAIRED_READERS = ("SP.2","SP.3","JP.1","JP.2","JP.5","JP.6")
MISSING_FOV = "DidNotAnnotateFOV"
UNDETECTED = "undetected"
EXPECTED_ELIGIBLE_ANCHORS = 1144
EXPECTED_PAIRED_READER_ANCHOR_ROWS = 5334

class AuditError(RuntimeError):
    pass

def require(ok, msg):
    if not bool(ok):
        raise AuditError(msg)

def load_inputs(matches_path, e_path, u_path, scoring_summary_path):
    m = pd.read_csv(matches_path)
    e = pd.read_csv(e_path, low_memory=False)
    u = pd.read_csv(u_path, low_memory=False)
    summary = json.loads(Path(scoring_summary_path).read_text())

    require(summary.get("status") ==
            "SCORING_COMPLETE_CONDITIONAL_ON_SOURCE_SELECTED_ANCHORS",
            "v16.1 scoring summary did not complete")
    require(summary.get("v15_audit_status") ==
            "PASS_ALL_52_BLINDED_L40S_PRODUCTION_RECEIPTS",
            "v15 production gate not passed")
    require(summary.get("matching_algorithm_sensitivity",{}).get(
            "detection_status_identical") is True,
            "Greedy/Hungarian detection status differs")
    require(summary.get("matching_algorithm_sensitivity",{}).get(
            "assignments_identical") is True,
            "Greedy/Hungarian assignments differ")

    need_m = {"anchor_id","image_id","slide","patient_id","ai_detected"}
    require(need_m.issubset(m.columns), "Anchor-match file missing required columns")
    require(len(m) == EXPECTED_ELIGIBLE_ANCHORS,
            f"Expected {EXPECTED_ELIGIBLE_ANCHORS} scored anchors, found {len(m)}")
    require(m.anchor_id.is_unique, "Scored anchor IDs are not unique")

    if m.ai_detected.dtype != bool:
        mapped = m.ai_detected.astype(str).str.lower().map({"true":True,"false":False})
        require(mapped.notna().all(), "Could not parse ai_detected")
        m["ai_detected"] = mapped

    for name, d in (("Evaluation",e),("Unbiased",u)):
        require("anchor_id" in d.columns, f"{name} missing anchor_id")
        require(d.anchor_id.is_unique, f"{name} anchor IDs duplicated")
        require(set(PAIRED_READERS).issubset(d.columns),
                f"{name} does not contain all six paired readers")
        require(set(m.anchor_id).issubset(set(d.anchor_id)),
                f"{name} missing scored anchors")

    return m, e, u, summary

def build_pairs(m, e, u):
    e = e.set_index("anchor_id")
    u = u.set_index("anchor_id")

    rows = []
    coverage = []
    for reader in PAIRED_READERS:
        e_lab = e.loc[m.anchor_id, reader].reset_index(drop=True)
        u_lab = u.loc[m.anchor_id, reader].reset_index(drop=True)

        obs_e = e_lab.ne(MISSING_FOV)
        obs_u = u_lab.ne(MISSING_FOV)
        both = obs_e & obs_u

        coverage.append({
            "reader": reader,
            "eligible_selected_anchors": len(m),
            "observed_evaluation": int(obs_e.sum()),
            "observed_unbiased": int(obs_u.sum()),
            "observed_both_conditions": int(both.sum()),
        })

        idx = np.flatnonzero(both.to_numpy())
        for i in idx:
            r = m.iloc[i]
            le = str(e_lab.iloc[i])
            lu = str(u_lab.iloc[i])
            ai = bool(r.ai_detected)
            de = le != UNDETECTED
            du = lu != UNDETECTED
            ae = ai == de
            au = ai == du

            if ae and au:
                pair_state = "agree_both"
            elif ae and not au:
                pair_state = "evaluation_only_agrees"
            elif au and not ae:
                pair_state = "unbiased_only_agrees"
            else:
                pair_state = "agree_neither"

            if de and du:
                reader_transition = "detected_both"
            elif de and not du:
                reader_transition = "Evaluation_detected__Unbiased_undetected"
            elif not de and du:
                reader_transition = "Evaluation_undetected__Unbiased_detected"
            else:
                reader_transition = "undetected_both"

            rows.append({
                "anchor_id": r.anchor_id,
                "image_id": r.image_id,
                "slide": r.slide,
                "patient_id": r.patient_id,
                "reader": reader,
                "ai_detected": ai,
                "evaluation_label": le,
                "unbiased_label": lu,
                "evaluation_reader_detected": de,
                "unbiased_reader_detected": du,
                "evaluation_ai_agreement": ae,
                "unbiased_ai_agreement": au,
                "paired_agreement_difference_E_minus_U": int(ae) - int(au),
                "pair_agreement_state": pair_state,
                "reader_detection_transition": reader_transition,
            })

    pairs = pd.DataFrame(rows)
    coverage = pd.DataFrame(coverage)

    require(len(pairs) == EXPECTED_PAIRED_READER_ANCHOR_ROWS,
            f"Expected {EXPECTED_PAIRED_READER_ANCHOR_ROWS} paired observations, "
            f"found {len(pairs)}")
    require(pairs[["anchor_id","reader"]].duplicated().sum() == 0,
            "Duplicate reader-anchor pairs")
    require(pairs.slide.nunique() == 5, "Expected five slides")
    require(pairs.patient_id.nunique() == 5, "Expected five patients")

    return pairs, coverage

def summarize_group(g):
    e_ag = int(g.evaluation_ai_agreement.sum())
    u_ag = int(g.unbiased_ai_agreement.sum())
    n = len(g)
    e_only = int((g.pair_agreement_state == "evaluation_only_agrees").sum())
    u_only = int((g.pair_agreement_state == "unbiased_only_agrees").sum())
    changed = int((g.evaluation_reader_detected != g.unbiased_reader_detected).sum())

    return pd.Series({
        "paired_reader_anchor_observations": n,
        "evaluation_ai_agreements": e_ag,
        "unbiased_ai_agreements": u_ag,
        "evaluation_agreement_rate": e_ag/n,
        "unbiased_agreement_rate": u_ag/n,
        "paired_difference_E_minus_U": (e_ag-u_ag)/n,
        "evaluation_only_agrees": e_only,
        "unbiased_only_agrees": u_only,
        "discordant_agreement_pairs": e_only + u_only,
        "reader_detection_changed_between_conditions": changed,
        "reader_detection_change_rate": changed/n,
    })

def make_summaries(pairs):
    overall = summarize_group(pairs).to_dict()

    by_reader = (
        pairs.groupby("reader", sort=True)
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )

    by_slide = (
        pairs.groupby(["slide","patient_id"], sort=True)
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )

    by_reader_slide = (
        pairs.groupby(["reader","slide","patient_id"], sort=True)
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )

    # Equal weighting summaries avoid letting readers/slides with more observed pairs dominate.
    overall["equal_reader_mean_paired_difference_E_minus_U"] = float(
        by_reader.paired_difference_E_minus_U.mean()
    )
    overall["equal_slide_mean_paired_difference_E_minus_U"] = float(
        by_slide.paired_difference_E_minus_U.mean()
    )
    overall["min_slide_paired_difference_E_minus_U"] = float(
        by_slide.paired_difference_E_minus_U.min()
    )
    overall["max_slide_paired_difference_E_minus_U"] = float(
        by_slide.paired_difference_E_minus_U.max()
    )

    # Reader detection transition counts.
    transitions = (
        pairs.groupby("reader_detection_transition", sort=True)
        .size().rename("count").reset_index()
    )
    transitions["proportion"] = transitions["count"] / len(pairs)

    agreement_states = (
        pairs.groupby("pair_agreement_state", sort=True)
        .size().rename("count").reset_index()
    )
    agreement_states["proportion"] = agreement_states["count"] / len(pairs)

    return overall, by_reader, by_slide, by_reader_slide, transitions, agreement_states

def self_test():
    m = pd.DataFrame({
        "anchor_id":["a","b"],
        "image_id":["i","i"],
        "slide":["s","s"],
        "patient_id":["p","p"],
        "ai_detected":[True,False],
    })
    # Construct two-reader synthetic data by directly checking state logic.
    ai = True
    de, du = True, False
    ae, au = ai == de, ai == du
    require(ae is True and au is False, "Agreement-state self test failed")
    require(int(ae)-int(au) == 1, "Paired-difference self test failed")
    print("SELF_TEST_PASS")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--matches", type=Path)
    ap.add_argument("--evaluation-ref", type=Path)
    ap.add_argument("--unbiased-ref", type=Path)
    ap.add_argument("--scoring-summary", type=Path)
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    for x in ("matches","evaluation_ref","unbiased_ref","scoring_summary","output_dir"):
        require(getattr(args,x) is not None, f"--{x.replace('_','-')} is required")

    m,e,u,score_summary = load_inputs(
        args.matches,args.evaluation_ref,args.unbiased_ref,args.scoring_summary
    )
    pairs, coverage = build_pairs(m,e,u)
    overall, by_reader, by_slide, by_reader_slide, transitions, agreement_states = (
        make_summaries(pairs)
    )

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    pairs.to_csv(out/"paired_same_reader_same_anchor_events_v17.csv", index=False)
    coverage.to_csv(out/"paired_reader_coverage_v17.csv", index=False)
    by_reader.to_csv(out/"paired_condition_agreement_by_reader_v17.csv", index=False)
    by_slide.to_csv(out/"paired_condition_agreement_by_slide_v17.csv", index=False)
    by_reader_slide.to_csv(out/"paired_condition_agreement_by_reader_slide_v17.csv", index=False)
    transitions.to_csv(out/"paired_reader_detection_transitions_v17.csv", index=False)
    agreement_states.to_csv(out/"paired_ai_agreement_states_v17.csv", index=False)

    result = {
        "status":"PAIRED_SAME_READER_SAME_ANCHOR_AUDIT_COMPLETE",
        "analysis_frame":{
            "eligible_selected_anchors_with_rgb":EXPECTED_ELIGIBLE_ANCHORS,
            "paired_readers":list(PAIRED_READERS),
            "paired_reader_anchor_observations":len(pairs),
            "slides":int(pairs.slide.nunique()),
            "patients":int(pairs.patient_id.nunique()),
            "missing_rule":"Exclude reader-anchor pair if DidNotAnnotateFOV in either condition",
            "ai_detection_source":"Frozen v16.1 selected-anchor match status",
        },
        "overall_paired_agreement":overall,
        "reader_coverage":coverage.to_dict(orient="records"),
        "by_reader":by_reader.to_dict(orient="records"),
        "by_slide":by_slide.to_dict(orient="records"),
        "reader_detection_transitions":transitions.to_dict(orient="records"),
        "paired_ai_agreement_states":agreement_states.to_dict(orient="records"),
        "interpretation":{
            "estimand":
                "Descriptive paired difference in fixed-AI detection agreement with the same "
                "reader on the same source-selected anchor between Evaluation and Unbiased conditions.",
            "causal":False,
            "independence_based_p_value_reported":False,
            "why_no_p_value":
                "Reader-anchor observations are repeated/nested and only five slides/patients are represented.",
        },
        "limitations":[
            "The 1,144-anchor frame is source-reader constructed and informatively selected.",
            "Condition order/design is not a randomized causal treatment comparison.",
            "The same readers contributed to the source annotation process underlying anchor construction.",
            "Only six readers have observations usable in both conditions.",
            "Only five TCGA slides/patients are represented.",
            "This audit does not identify agreement with a genuinely future independent reader.",
            "This audit does not establish biological sensitivity, specificity, or calibrated uncertainty.",
        ],
        "next_gate":
            "Use the paired result to judge whether assistance dependence is empirically material; "
            "then formalize selection/reader observation model and known-truth simulation before Bayesian claims."
    }
    (out/"paired_same_reader_same_anchor_summary_v17.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    print(json.dumps(result, indent=2))
    print("PAIRED_AUDIT_V17_COMPLETE")
    print("DESCRIPTIVE_NONCAUSAL_ANALYSIS_ONLY")

if __name__ == "__main__":
    try:
        main()
    except AuditError as e:
        print("AUDIT_BLOCKED:", e)
        raise SystemExit(2)
