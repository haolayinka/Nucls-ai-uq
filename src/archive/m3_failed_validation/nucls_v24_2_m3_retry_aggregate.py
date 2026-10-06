#!/usr/bin/env python3
from pathlib import Path
import argparse, json, pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument("--outdir", type=Path, default=Path("outputs/v24_2_m3_retry_3000_7000_0999"))
args=ap.parse_args()

rows=[]
for rep in (3,4):
    p=args.outdir/f"rep{rep:03d}_gate.json"
    if not p.exists():
        raise FileNotFoundError(p)
    g=json.loads(p.read_text())
    rows.append({
        "rep":rep,
        "divergences":g["divergences"],
        "max_rhat":g["max_rhat_scalar"],
        "min_bulk":g["min_ess_bulk_scalar"],
        "min_tail":g["min_ess_tail_scalar"],
        "elapsed_sec":g["elapsed_sec"],
        "theta_U":next(x["estimate"] for x in g["summary"] if x["target"]=="theta_U_selected"),
        "theta_E":next(x["estimate"] for x in g["summary"] if x["target"]=="theta_E_selected"),
        "delta":next(x["estimate"] for x in g["summary"] if x["target"]=="delta_E_minus_U_selected"),
    })
df=pd.DataFrame(rows)
print("M3 TARGETED RETRY 3000/7000/0.999")
print(df.to_string(index=False))
print("STRICT_DIAGNOSTIC_PASS", bool((df.divergences.eq(0) & df.max_rhat.le(1.01) & df.min_bulk.ge(400) & df.min_tail.ge(400)).all()))
