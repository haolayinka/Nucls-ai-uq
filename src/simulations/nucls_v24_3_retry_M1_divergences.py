#!/usr/bin/env python3
from __future__ import annotations
from repo_paths import repo_root
import argparse, glob, json, subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--retry-slot", type=int, required=True)
ap.add_argument("--primary-root", type=Path, required=True)
ap.add_argument("--retry-root", type=Path, required=True)
ap.add_argument("--freeze-root", type=Path, required=True)
args = ap.parse_args()

bad = []
for p in glob.glob(str(args.primary_root / "M1" / "A*" / "rep_*_gate.json")):
    with open(p) as f:
        g = json.load(f)
    if not g.get("computational_ok", False):
        if not (g["divergences"] > 0 and g["max_rhat_scalar"] <= 1.01
                and g["min_ess_bulk_scalar"] >= 400
                and g["min_ess_tail_scalar"] >= 400):
            raise RuntimeError(f"Non-divergence-only failure found: {p}")
        bad.append((int(g["index"]), g["scenario"], int(g["rep"]),
                    int(g["divergences"]), p))

bad.sort()
if len(bad) != 33:
    raise RuntimeError(f"Expected exactly 33 divergence-only failures, found {len(bad)}")
if not 0 <= args.retry_slot < len(bad):
    raise RuntimeError(f"retry-slot must be 0..{len(bad)-1}")

idx, scenario, rep, old_div, source_gate = bad[args.retry_slot]
print(f"RETRY_SLOT={args.retry_slot} index={idx} scenario={scenario} rep={rep} original_divergences={old_div}", flush=True)

root = repo_root()
py = "python"
worker = root / "scripts" / "nucls_v24_3_bayes_worker.py"

cmd = [
    py, str(worker),
    "--model", "M1",
    "--index", str(idx),
    "--freeze-root", str(args.freeze_root),
    "--outdir", str(args.retry_root),
    "--draws", "5000",
    "--tune", "3000",
    "--chains", "4",
    "--target-accept", "0.999",
]
print("COMMAND:", " ".join(cmd), flush=True)
raise SystemExit(subprocess.run(cmd).returncode)
