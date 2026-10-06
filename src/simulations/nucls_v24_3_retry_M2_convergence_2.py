#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, hashlib, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd

from nucls_v24_3_frozen_models import (
    self_test, load_frozen, encode_selected, build_model,
    baseline_future_probabilities, agreement_draws_from_prob,
    summarize_draw_matrix, scalar_var_names, EXPECTED_FREEZE_HASH
)

def retry_seed(scenario: str, rep: int) -> int:
    # Deliberately different from the primary M2 seed, but deterministic.
    x = f"v24.3|M2|CONVERGENCE_RETRY|{scenario}|{rep}".encode()
    return int.from_bytes(hashlib.sha256(x).digest()[:4], "little") % 2_000_000_000

def discover_failures(primary_root: Path):
    bad = []
    for p in glob.glob(str(primary_root / "M2" / "A*" / "rep_*_gate.json")):
        with open(p) as f:
            g = json.load(f)
        if not g.get("computational_ok", False):
            # This retry is only for non-divergent convergence/ESS failures.
            if int(g["divergences"]) != 0:
                raise RuntimeError(f"Unexpected divergent M2 failure: {p}")
            bad.append((
                int(g["index"]), g["scenario"], int(g["rep"]),
                float(g["max_rhat_scalar"]),
                float(g["min_ess_bulk_scalar"]),
                float(g["min_ess_tail_scalar"]),
                p
            ))
    bad.sort()
    if len(bad) != 2:
        raise RuntimeError(f"Expected exactly 2 non-divergent M2 failures, found {len(bad)}")
    return bad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retry-slot", type=int, required=True)
    ap.add_argument("--primary-root", type=Path, required=True)
    ap.add_argument("--retry-root", type=Path, required=True)
    ap.add_argument("--freeze-root", type=Path, required=True)
    args = ap.parse_args()

    self_test()
    bad = discover_failures(args.primary_root)
    if not 0 <= args.retry_slot < len(bad):
        raise RuntimeError("retry-slot must be 0 or 1")

    idx, sid, rep, old_rhat, old_bulk, old_tail, source_gate = bad[args.retry_slot]
    print(
        f"RETRY_SLOT={args.retry_slot} index={idx} scenario={sid} rep={rep} "
        f"original_rhat={old_rhat} original_bulk={old_bulk} original_tail={old_tail}",
        flush=True
    )

    dat = load_frozen(args.freeze_root, sid, rep, verify_hash=True)
    enc = encode_selected(dat)

    audit = {
        "model": "M2", "scenario": sid, "rep": rep, "index": idx,
        "source_file": str(dat["path"]), "n_selected": enc.n_anchors,
        "n_fovs": enc.n_fovs, "n_obs": len(enc.y),
        "n_obs_U": int((enc.c == 0).sum()),
        "n_obs_E": int((enc.c == 1).sum()),
        "ai_positive_selected": int(enc.anchor_ai.sum()),
        "n_rf": enc.n_rf, "n_ra": enc.n_ra,
        "retry_of_gate": source_gate,
        "retry_reason": "primary non-divergent Rhat/ESS computational-gate failure"
    }

    out = args.retry_root / "M2" / sid
    out.mkdir(parents=True, exist_ok=True)
    stem = f"rep_{rep:03d}"

    import pymc as pm
    import arviz as az

    model = build_model(enc, "M2")
    seed = retry_seed(sid, rep)

    # This is a computational remediation only:
    # same frozen data + same model, but new deterministic seed and longer sampling.
    draws = 10000
    tune = 5000
    chains = 4
    target_accept = 0.999

    t0 = time.time()
    with model:
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains,
            cores=min(chains, max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))),
            random_seed=seed,
            target_accept=target_accept,
            init="jitter+adapt_diag",
            progressbar=False,
            return_inferencedata=True,
        )
    elapsed = time.time() - t0

    pu, pe = baseline_future_probabilities(idata, enc, batch=100)
    est_draws = agreement_draws_from_prob(pu, pe, enc.anchor_ai)
    summarize_draw_matrix(est_draws).to_csv(out / f"{stem}_summary.csv", index=False)

    ds = az.summary(idata, var_names=scalar_var_names("M2"), round_to=None)
    div = int(np.asarray(idata.sample_stats["diverging"]).sum())
    diag = {
        "divergences": div,
        "max_rhat_scalar": float(ds.r_hat.max()),
        "min_ess_bulk_scalar": float(ds.ess_bulk.min()),
        "min_ess_tail_scalar": float(ds.ess_tail.min()),
        "elapsed_sec": elapsed,
    }
    ds.to_csv(out / f"{stem}_scalar_diagnostics.csv")

    gate = {
        **audit, **diag,
        "freeze_hash": EXPECTED_FREEZE_HASH,
        "seed": seed,
        "sampler": {
            "draws": draws, "tune": tune, "chains": chains,
            "target_accept": target_accept
        },
        "computational_ok": bool(
            div == 0
            and diag["max_rhat_scalar"] <= 1.01
            and diag["min_ess_bulk_scalar"] >= 400
            and diag["min_ess_tail_scalar"] >= 400
        ),
        "status": "COMPLETE",
    }
    (out / f"{stem}_gate.json").write_text(json.dumps(gate, indent=2))
    print("M2_CONVERGENCE_RETRY_COMPLETE " + json.dumps(gate, sort_keys=True))

if __name__ == "__main__":
    main()
