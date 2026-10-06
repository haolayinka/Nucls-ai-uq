#!/usr/bin/env python3
"""
Dispatch one of 95 batch workers across the frozen 800-replicate battery.

Worker w processes indices:
    w, w + 95, w + 2*95, ...
so the 800 fits are balanced across exactly 95 SLURM tasks (8 or 9 fits/task).

The underlying single-fit worker remains unchanged and restartable. Existing
valid outputs are skipped. A failed replicate is recorded and the batch
continues to the next assigned replicate.
"""
from __future__ import annotations
from repo_paths import repo_root
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["M1","M2"], required=True)
ap.add_argument("--worker-id", type=int, required=True)
ap.add_argument("--n-workers", type=int, default=95)
ap.add_argument("--freeze-root", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--draws", type=int, default=4000)
ap.add_argument("--tune", type=int, default=2000)
ap.add_argument("--chains", type=int, default=4)
ap.add_argument("--target-accept", type=float, default=0.99)
args = ap.parse_args()

if not (0 <= args.worker_id < args.n_workers):
    raise SystemExit(f"worker-id must be 0..{args.n_workers-1}")

root = repo_root()
py = "python"
single = root/"scripts"/"nucls_v24_3_bayes_worker.py"

indices = list(range(args.worker_id, 800, args.n_workers))
print(f"BATCH_START model={args.model} worker={args.worker_id}/{args.n_workers} "
      f"n_indices={len(indices)} indices={indices}", flush=True)

records = []
for idx in indices:
    t0 = time.time()
    cmd = [
        py, str(single),
        "--model", args.model,
        "--index", str(idx),
        "--freeze-root", args.freeze_root,
        "--outdir", args.outdir,
        "--draws", str(args.draws),
        "--tune", str(args.tune),
        "--chains", str(args.chains),
        "--target-accept", str(args.target_accept),
    ]
    print("RUN index", idx, flush=True)
    cp = subprocess.run(cmd)
    records.append({
        "index": idx,
        "returncode": int(cp.returncode),
        "elapsed_sec": time.time() - t0,
    })
    print(f"DONE index={idx} returncode={cp.returncode}", flush=True)

summary_dir = Path(args.outdir)/"batch_status"/args.model
summary_dir.mkdir(parents=True, exist_ok=True)
summary_path = summary_dir/f"worker_{args.worker_id:02d}.json"
summary_path.write_text(json.dumps({
    "model": args.model,
    "worker_id": args.worker_id,
    "n_workers": args.n_workers,
    "indices": indices,
    "records": records,
    "n_failed": sum(x["returncode"] != 0 for x in records),
}, indent=2))

n_failed = sum(x["returncode"] != 0 for x in records)
print(f"BATCH_COMPLETE model={args.model} worker={args.worker_id} failures={n_failed}", flush=True)
if n_failed:
    raise SystemExit(2)
