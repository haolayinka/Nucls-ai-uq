#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit

PROTOCOL = "NuCLS-v26.1-correct-spec-calibration-2026-10-12"
N_REPS = 200
SCENARIOS = [
    "M1_R5_S5", "M1_R5_S10", "M1_R5_S20",
    "M1_R10_S5", "M1_R10_S10", "M1_R10_S20",
    "M2_R5_S5", "M2_R5_S10", "M2_R5_S20",
    "M2_R10_S5", "M2_R10_S10", "M2_R10_S20",
]
GHX30, GHW30 = hermgauss(30)
GHW30 = GHW30 / np.sqrt(np.pi)

@dataclass
class EncodedData:
    y: np.ndarray
    c: np.ndarray
    r: np.ndarray
    f: np.ndarray
    a: np.ndarray
    rf: np.ndarray
    ra: np.ndarray
    anchor_ai: np.ndarray
    anchor_f: np.ndarray
    n_readers: int
    n_fovs: int
    n_anchors: int
    n_rf: int
    n_ra: int
    selected_source_index: np.ndarray

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def index_to_scenario_rep(index: int):
    n = len(SCENARIOS) * N_REPS
    if not 0 <= index < n:
        raise ValueError(f"index must be 0..{n-1}")
    return SCENARIOS[index // N_REPS], index % N_REPS

def scenario_spec(scenario: str):
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    gen, rpart, spart = scenario.split("_")
    return {
        "generator": gen,
        "n_readers": int(rpart[1:]),
        "n_slides": int(spart[1:]),
    }

def verify_freeze_root(root: Path, expected_hash: str):
    man = json.loads((root / "FREEZE_MANIFEST.json").read_text())
    if man.get("protocol_version") != PROTOCOL:
        raise RuntimeError(f"Wrong protocol: {man.get('protocol_version')}")
    if man.get("frozen_data_sha256") != expected_hash:
        raise RuntimeError(
            f"Wrong frozen hash: {man.get('frozen_data_sha256')} != {expected_hash}"
        )
    if int(man.get("n_frozen_datasets", -1)) != len(SCENARIOS) * N_REPS:
        raise RuntimeError("Wrong frozen dataset count")
    sha_path = root / "SHA256SUMS.txt"
    got_root = hashlib.sha256(sha_path.read_bytes()).hexdigest()
    if got_root != expected_hash:
        raise RuntimeError(f"SHA256SUMS root hash mismatch: {got_root}")
    return man

def _hash_lookup(root: Path):
    out = {}
    for line in (root / "SHA256SUMS.txt").read_text().splitlines():
        if not line.strip():
            continue
        h, p = line.split(None, 1)
        out[p.strip()] = h
    return out

def load_frozen(root: Path, scenario: str, rep: int, expected_hash: str, verify_hash=True):
    verify_freeze_root(root, expected_hash)
    if scenario not in SCENARIOS or not 0 <= rep < N_REPS:
        raise ValueError((scenario, rep))
    rel = f"data/{scenario}/rep_{rep:03d}.npz"
    p = root / rel
    if not p.exists():
        raise FileNotFoundError(p)
    if verify_hash:
        exp = _hash_lookup(root).get(rel)
        if exp is None:
            raise RuntimeError(f"No frozen hash entry for {rel}")
        got = sha256_file(p)
        if got != exp:
            raise RuntimeError(f"Frozen file hash mismatch: {rel}")
    d = np.load(p, allow_pickle=False)
    req = {
        "ai", "selected", "reader_y_U", "reader_y_E",
        "obs_U", "obs_E", "fov_index", "slide_index", "patient_index"
    }
    if set(d.files) != req:
        raise RuntimeError(f"Unexpected keys in {rel}: {set(d.files)}")
    ai = d["ai"].astype(np.int8)
    selected = d["selected"].astype(bool)
    yu, ye = d["reader_y_U"].astype(np.int8), d["reader_y_E"].astype(np.int8)
    ou, oe = d["obs_U"].astype(bool), d["obs_E"].astype(bool)
    fi = d["fov_index"].astype(int)
    si = d["slide_index"].astype(int)
    pi = d["patient_index"].astype(int)
    n = len(ai)
    if selected.shape != (n,) or fi.shape != (n,) or si.shape != (n,) or pi.shape != (n,):
        raise RuntimeError("Unexpected anchor-vector shape")
    if yu.shape != ye.shape or ou.shape != oe.shape or yu.shape != ou.shape:
        raise RuntimeError("Reader-array shape mismatch")
    if yu.ndim != 2 or yu.shape[1] != n:
        raise RuntimeError("Unexpected reader-array shape")
    if not np.isin(ai, [0, 1]).all() or not selected.all():
        raise RuntimeError("Invalid AI vector or selection mask")
    if not np.all((yu[ou] == 0) | (yu[ou] == 1)) or not np.all((ye[oe] == 0) | (ye[oe] == 1)):
        raise RuntimeError("Observed reader labels must be binary")
    if not np.all(yu[~ou] == -1) or not np.all(ye[~oe] == -1):
        raise RuntimeError("Unavailable reader labels must be -1")
    return {
        "ai": ai, "selected": selected,
        "reader_y_U": yu, "reader_y_E": ye,
        "obs_U": ou, "obs_E": oe,
        "fov_index": fi, "slide_index": si, "patient_index": pi,
        "path": p,
    }

def encode_selected(dat):
    sel = np.flatnonzero(dat["selected"])
    ai = dat["ai"][sel]
    src_f = dat["fov_index"][sel]
    fvals = np.unique(src_f)
    fmap = {int(f): i for i, f in enumerate(fvals)}
    anchor_f = np.array([fmap[int(f)] for f in src_f], dtype=np.int32)
    src_to_a = {int(src): j for j, src in enumerate(sel)}

    yu, ye, ou, oe = dat["reader_y_U"], dat["reader_y_E"], dat["obs_U"], dat["obs_E"]
    n_readers = yu.shape[0]
    rows = []
    for rr in range(n_readers):
        for cc, (yy, oo) in enumerate([(yu, ou), (ye, oe)]):
            idx = np.flatnonzero(oo[rr] & dat["selected"])
            for src in idx:
                rows.append(
                    (rr, cc, fmap[int(dat["fov_index"][src])],
                     src_to_a[int(src)], int(yy[rr, src]))
                )
    arr = np.asarray(rows, dtype=int)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise RuntimeError("No observed rows")
    r, c, f, a, y = arr.T

    rf_keys = sorted(set(zip(r.tolist(), f.tolist())))
    rf_map = {k: i for i, k in enumerate(rf_keys)}
    rf = np.array([rf_map[(int(rr), int(ff))] for rr, ff in zip(r, f)], dtype=np.int32)

    ra_keys = sorted(set(zip(r.tolist(), a.tolist())))
    ra_map = {k: i for i, k in enumerate(ra_keys)}
    ra = np.array([ra_map[(int(rr), int(aa))] for rr, aa in zip(r, a)], dtype=np.int32)

    enc = EncodedData(
        y=y.astype(np.int8), c=c.astype(np.int8), r=r.astype(np.int16),
        f=f.astype(np.int32), a=a.astype(np.int32), rf=rf, ra=ra,
        anchor_ai=ai.astype(np.int8), anchor_f=anchor_f,
        n_readers=n_readers, n_fovs=len(fvals), n_anchors=len(sel),
        n_rf=len(rf_keys), n_ra=len(ra_keys),
        selected_source_index=sel.astype(np.int32),
    )
    expected = int(dat["obs_U"][:, sel].sum() + dat["obs_E"][:, sel].sum())
    if len(enc.y) != expected:
        raise RuntimeError("Encoded row count mismatch")
    return enc

def build_model(data: EncodedData, model_name: str):
    import pymc as pm
    if model_name not in {"M1", "M2"}:
        raise ValueError(model_name)
    coords = {
        "reader": np.arange(data.n_readers),
        "fov": np.arange(data.n_fovs),
        "anchor": np.arange(data.n_anchors),
        "rf_obs": np.arange(data.n_rf),
        "ra_obs": np.arange(data.n_ra),
        "obs": np.arange(len(data.y)),
    }
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", 0, 2.5)
        beta_E = pm.Normal("beta_E", 0, 1.0)
        if model_name == "M2":
            alpha_ai = pm.Normal("alpha_ai", 0, 2.5)
            beta_ai = pm.HalfNormal("beta_ai", 1.5)

        sd_reader = pm.HalfNormal("sd_reader", 1.0)
        sd_reader_condition = pm.HalfNormal("sd_reader_condition", 1.0)
        sd_fov = pm.HalfNormal("sd_fov", 1.0)
        sd_reader_fov = pm.HalfNormal("sd_reader_fov", 1.0)
        sd_anchor = pm.HalfNormal("sd_anchor", 1.5)
        sd_reader_anchor = pm.HalfNormal("sd_reader_anchor", 1.5)

        z_reader = pm.Normal("z_reader", 0, 1, dims="reader")
        z_reader_condition = pm.Normal("z_reader_condition", 0, 1, dims="reader")
        z_fov = pm.Normal("z_fov", 0, 1, dims="fov")
        z_reader_fov = pm.Normal("z_reader_fov", 0, 1, dims="rf_obs")
        z_anchor = pm.Normal("z_anchor", 0, 1, dims="anchor")
        z_reader_anchor = pm.Normal("z_reader_anchor", 0, 1, dims="ra_obs")

        g = sd_fov * z_fov
        aa = sd_anchor * z_anchor
        d_shared = g[data.anchor_f] + aa

        if model_name == "M2":
            pm.Bernoulli(
                "ai_detected_obs",
                logit_p=alpha_ai + beta_ai * d_shared,
                observed=data.anchor_ai,
                dims="anchor",
            )

        eta = (
            alpha + beta_E * data.c + d_shared[data.a]
            + sd_reader * z_reader[data.r]
            + sd_reader_condition * z_reader_condition[data.r] * data.c
            + sd_reader_fov * z_reader_fov[data.rf]
            + sd_reader_anchor * z_reader_anchor[data.ra]
        )
        pm.Bernoulli("reader_detected", logit_p=eta, observed=data.y, dims="obs")
    return model

def _flat(post, name):
    x = np.asarray(post[name].values)
    return x.reshape((-1,) + x.shape[2:])

def logistic_normal_mean(mu, sd):
    mu = np.asarray(mu, float)
    sd = np.asarray(sd, float)
    z = mu[..., None] + np.sqrt(2.0) * sd[..., None] * GHX30
    return np.sum(expit(z) * GHW30, axis=-1)

def baseline_future_probabilities(idata, data: EncodedData, batch=100):
    p = idata.posterior
    alpha = _flat(p, "alpha")
    beta = _flat(p, "beta_E")
    sf = _flat(p, "sd_fov")
    za = _flat(p, "z_anchor")
    sa = _flat(p, "sd_anchor")
    zf = _flat(p, "z_fov")
    sr = _flat(p, "sd_reader")
    sb = _flat(p, "sd_reader_condition")
    srf = _flat(p, "sd_reader_fov")
    sra = _flat(p, "sd_reader_anchor")
    n = len(alpha)
    pu = np.empty((n, data.n_anchors), float)
    pe = np.empty_like(pu)
    for lo in range(0, n, batch):
        hi = min(n, lo + batch)
        g = sf[lo:hi, None] * zf[lo:hi]
        aa = sa[lo:hi, None] * za[lo:hi]
        base = alpha[lo:hi, None] + g[:, data.anchor_f] + aa
        sdu = np.sqrt(sr[lo:hi]**2 + srf[lo:hi]**2 + sra[lo:hi]**2)
        sde = np.sqrt(sr[lo:hi]**2 + sb[lo:hi]**2 + srf[lo:hi]**2 + sra[lo:hi]**2)
        pu[lo:hi] = logistic_normal_mean(base, sdu[:, None])
        pe[lo:hi] = logistic_normal_mean(base + beta[lo:hi, None], sde[:, None])
    return pu, pe

def agreement_draws_from_prob(pu, pe, ai):
    ai = np.asarray(ai, float)[None, :]
    tu = (ai * pu + (1 - ai) * (1 - pu)).mean(axis=1)
    te = (ai * pe + (1 - ai) * (1 - pe)).mean(axis=1)
    return np.column_stack([tu, te, te - tu])

def summarize_draw_matrix(draws):
    names = ["theta_U_selected", "theta_E_selected", "delta_E_minus_U_selected"]
    rows = []
    for j, name in enumerate(names):
        x = draws[:, j]
        lo, hi = np.quantile(x, [.025, .975])
        rows.append({
            "target": name, "estimate": float(x.mean()),
            "posterior_sd": float(x.std(ddof=1)),
            "ci_low": float(lo), "ci_high": float(hi),
            "interval_width": float(hi - lo),
        })
    return pd.DataFrame(rows)

def scalar_var_names(model_name):
    base = [
        "alpha", "beta_E", "sd_reader", "sd_reader_condition",
        "sd_fov", "sd_reader_fov", "sd_anchor", "sd_reader_anchor",
    ]
    if model_name == "M2":
        return ["alpha", "beta_E", "alpha_ai", "beta_ai"] + base[2:]
    return base
