#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

READERS = ["SP.3","JP.1","JP.2","JP.5","JP.6"]
SCENARIOS = [f"A{i}" for i in range(1,9)]
EXPECTED_PROTOCOL = "NuCLS-v24.3-adversarial-freeze-2026-09-27"
EXPECTED_FREEZE_HASH = "9c00329aeaf9d4344c3b94b9ce781a2c96fbc957a20135d791c7f3fb96e35d55"

def sha256_file(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()

def verify_root(root: Path):
    man=json.loads((root/"FREEZE_MANIFEST.json").read_text())
    if man.get("protocol_version") != EXPECTED_PROTOCOL:
        raise RuntimeError(f"Wrong frozen protocol: {man.get('protocol_version')}")
    if man.get("frozen_data_sha256") != EXPECTED_FREEZE_HASH:
        raise RuntimeError(f"Wrong frozen hash: {man.get('frozen_data_sha256')}")
    if int(man.get("n_frozen_datasets",-1)) != 800:
        raise RuntimeError("Expected 800 frozen datasets")

def expected_hash(root: Path, rel: str):
    for line in (root/"SHA256SUMS.txt").read_text().splitlines():
        if not line.strip():
            continue
        h,p=line.split(None,1)
        if p.strip()==rel:
            return h
    raise RuntimeError(f"Missing SHA entry for {rel}")

def index_to_sr(index: int):
    if not 0 <= index < 800:
        raise ValueError(index)
    return f"A{index//100+1}", index%100

def load_frozen(root: Path, scenario: str, rep: int):
    rel=f"data/{scenario}/rep_{rep:03d}.npz"
    p=root/rel
    if not p.exists():
        raise FileNotFoundError(p)
    got=sha256_file(p); exp=expected_hash(root,rel)
    if got != exp:
        raise RuntimeError(f"SHA mismatch: {rel}")
    d=np.load(p,allow_pickle=False)
    req={"ai","selected","reader_y_U","reader_y_E","obs_U","obs_E",
         "fov_index","slide_index","patient_index","discovery_read"}
    if set(d.files)!=req:
        raise RuntimeError(f"Unexpected keys in {rel}: {set(d.files)}")
    out={k:d[k] for k in d.files}
    ai=out["ai"].astype(np.int8)
    sel=out["selected"].astype(bool)
    yu=out["reader_y_U"].astype(np.int8)
    ye=out["reader_y_E"].astype(np.int8)
    ou=out["obs_U"].astype(bool)
    oe=out["obs_E"].astype(bool)
    if ai.shape!=(1144,) or sel.shape!=(1144,):
        raise RuntimeError("Unexpected anchor shape")
    if yu.shape!=(5,1144) or ye.shape!=(5,1144) or ou.shape!=(5,1144) or oe.shape!=(5,1144):
        raise RuntimeError("Unexpected reader array shape")
    if np.any(ou[:,~sel]) or np.any(oe[:,~sel]):
        raise RuntimeError("Unselected anchors must be unavailable")
    if not np.all((yu[ou]==0)|(yu[ou]==1)) or not np.all((ye[oe]==0)|(ye[oe]==1)):
        raise RuntimeError("Observed reader labels must be binary")
    anyobs=(ou|oe).any(axis=0)
    if not np.all(anyobs[sel]):
        raise RuntimeError("Selected anchor with no iMRMC-observable row")
    return dict(ai=ai,selected=sel,reader_y_U=yu,reader_y_E=ye,obs_U=ou,obs_E=oe)

def build_imrmc_input(dat):
    ai=dat["ai"]; sel=dat["selected"]
    rows=[]
    for rr,rname in enumerate(READERS):
        for modality, yy, oo in [
            ("U",dat["reader_y_U"],dat["obs_U"]),
            ("E",dat["reader_y_E"],dat["obs_E"])
        ]:
            idx=np.flatnonzero(sel & oo[rr])
            for src in idx:
                rows.append((rname,f"a{int(src):04d}",modality,int(yy[rr,src]==ai[src])))
    df=pd.DataFrame(rows,columns=["readerID","caseID","modalityID","score"])
    if df.empty:
        raise RuntimeError("Empty iMRMC input")
    if df.duplicated(["readerID","caseID","modalityID"]).any():
        raise RuntimeError("Duplicate reader-case-modality row")
    if set(df["modalityID"])!={"U","E"} or set(df["score"].unique())-set([0,1]):
        raise RuntimeError("Invalid iMRMC input")
    nsel=int(sel.sum())
    if df["caseID"].nunique()!=nsel:
        raise RuntimeError(f"Case count mismatch: {df['caseID'].nunique()} vs {nsel}")
    return df,nsel

def valid_existing(path: Path):
    if not path.exists():
        return False
    try:
        d=pd.read_csv(path)
        return (
            len(d)==6 and
            set(d["method"])=={"iMRMC_uStat11_conditionalD","iMRMC_uStat11_jointD"} and
            set(d["target"])=={"theta_U_selected","theta_E_selected","delta_E_minus_U_selected"} and
            np.isfinite(d["estimate"]).all() and np.isfinite(d["variance"]).all()
        )
    except Exception:
        return False

def run_one(project: Path, freeze_root: Path, outdir: Path, scenario: str, rep: int):
    sdir=outdir/scenario
    sdir.mkdir(parents=True,exist_ok=True)
    result=sdir/f"rep_{rep:03d}_summary.csv"
    status=sdir/f"rep_{rep:03d}_status.json"
    if valid_existing(result):
        return "skipped"

    dat=load_frozen(freeze_root,scenario,rep)
    inp,nsel=build_imrmc_input(dat)
    with tempfile.TemporaryDirectory(prefix=f"imrmc_{scenario}_{rep:03d}_") as td:
        td=Path(td)
        incsv=td/"input.csv"
        outcsv=td/"result.csv"
        inp.to_csv(incsv,index=False)
        cmd=[
            "Rscript",
            str(project/"scripts"/"nucls_v24_4_imrmc_one.R"),
            str(incsv),str(outcsv),scenario,str(rep),str(nsel)
        ]
        proc=subprocess.run(cmd,text=True,capture_output=True)
        if proc.returncode!=0:
            fail={
                "scenario":scenario,"rep":rep,"status":"failed",
                "returncode":proc.returncode,
                "stdout":proc.stdout[-12000:],
                "stderr":proc.stderr[-12000:]
            }
            status.write_text(json.dumps(fail,indent=2))
            raise RuntimeError(f"iMRMC failed {scenario} rep {rep}: {proc.stderr[-2000:]}")
        d=pd.read_csv(outcsv)
        if len(d)!=6:
            raise RuntimeError(f"Unexpected iMRMC output rows for {scenario} rep {rep}: {len(d)}")
        d["scenario"]=scenario
        d["rep"]=rep
        d["n_selected"]=nsel
        tmp=result.with_suffix(".csv.tmp")
        d.to_csv(tmp,index=False)
        os.replace(tmp,result)
        ok={
            "scenario":scenario,"rep":rep,"status":"ok",
            "n_selected":nsel,"n_input_rows":len(inp),
            "stdout":proc.stdout[-4000:],
            "stderr":proc.stderr[-4000:]
        }
        status.write_text(json.dumps(ok,indent=2))
    return "done"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",type=Path,required=True)
    ap.add_argument("--freeze-root",type=Path,required=True)
    ap.add_argument("--outdir",type=Path,required=True)
    ap.add_argument("--worker-id",type=int,required=True)
    ap.add_argument("--n-workers",type=int,default=95)
    args=ap.parse_args()
    if not 0 <= args.worker_id < args.n_workers:
        raise ValueError("worker-id out of range")
    verify_root(args.freeze_root)
    args.outdir.mkdir(parents=True,exist_ok=True)
    bdir=args.outdir/"batch_status"
    bdir.mkdir(parents=True,exist_ok=True)

    indices=list(range(args.worker_id,800,args.n_workers))
    if len(indices) not in (8,9):
        raise RuntimeError(f"Unexpected worker load {len(indices)}")
    completed=[]; skipped=[]
    for index in indices:
        scenario,rep=index_to_sr(index)
        state=run_one(args.project,args.freeze_root,args.outdir,scenario,rep)
        (skipped if state=="skipped" else completed).append(index)

    rec={
        "worker_id":args.worker_id,"n_workers":args.n_workers,
        "indices":indices,"completed":completed,"skipped":skipped,
        "status":"ok"
    }
    (bdir/f"worker_{args.worker_id:02d}.json").write_text(json.dumps(rec,indent=2))
    print(json.dumps(rec))

if __name__=="__main__":
    main()
