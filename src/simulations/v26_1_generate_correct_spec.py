#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit
from v26_1_models import PROTOCOL, SCENARIOS, N_REPS, scenario_spec, sha256_file

PRIMARY_READERS = ["SP.3", "JP.1", "JP.2", "JP.5", "JP.6"]

DGP = {
    "M1": {
        "alpha": 2.606, "beta_E": -0.045,
        "sd_reader": 0.311, "sd_reader_condition": 0.856,
        "sd_fov": 0.600, "sd_reader_fov": 0.404,
        "sd_anchor": 1.246, "sd_reader_anchor": 1.307,
    },
    "M2": {
        "alpha": 2.548, "beta_E": -0.040,
        "alpha_ai": -0.173, "beta_ai": 3.012,
        "sd_reader": 0.297, "sd_reader_condition": 0.849,
        "sd_fov": 0.719, "sd_reader_fov": 0.963,
        "sd_anchor": 0.786, "sd_reader_anchor": 1.509,
    },
}

GHX80, GHW80 = hermgauss(80)
GHW80 = GHW80 / np.sqrt(np.pi)

def deterministic_seed(scenario: str, rep: int) -> int:
    x = f"{PROTOCOL}|{scenario}|{rep}".encode()
    return int.from_bytes(hashlib.sha256(x).digest()[:8], "little") % (2**32 - 1)

def logistic_normal_mean80(mu, sd):
    mu = np.asarray(mu, float)
    z = mu[..., None] + np.sqrt(2.0) * float(sd) * GHX80
    return np.sum(expit(z) * GHW80, axis=-1)

def source_template(path: Path):
    d = pd.read_csv(path)
    required = {
        "anchor_id", "image_id", "slide", "patient_id", "reader",
        "condition", "ai_detected", "reader_detected"
    }
    missing = required - set(d.columns)
    if missing:
        raise RuntimeError(f"Source CSV missing columns: {sorted(missing)}")
    d = d.loc[d["reader"].isin(PRIMARY_READERS)].copy()
    if d.duplicated(["anchor_id", "reader", "condition"]).any():
        raise RuntimeError("Duplicate primary-reader anchor/condition rows in source")
    if set(d["condition"].unique()) != {"Unbiased", "Evaluation"}:
        raise RuntimeError("Unexpected condition set")

    for c in ["image_id", "slide", "patient_id", "ai_detected"]:
        bad = d.groupby("anchor_id")[c].nunique(dropna=False)
        if (bad != 1).any():
            raise RuntimeError(f"Inconsistent {c} within anchor")

    meta = (
        d.groupby("anchor_id", as_index=False)
         .agg(image_id=("image_id", "first"),
              slide=("slide", "first"),
              patient_id=("patient_id", "first"),
              ai_detected=("ai_detected", "first"))
    )
    fov_order = (
        meta[["image_id"]].drop_duplicates()
        .assign(_num=lambda x: pd.to_numeric(
            x["image_id"].astype(str).str.extract(r"(\d+)$")[0], errors="coerce"))
        .sort_values(["_num", "image_id"], kind="mergesort")["image_id"].tolist()
    )
    if len(fov_order) != 52:
        raise RuntimeError(f"Expected 52 FOVs, got {len(fov_order)}")
    fov_map = {name: i for i, name in enumerate(fov_order)}

    slide_order = (
        meta.assign(_f=meta["image_id"].map(fov_map))
            .groupby("slide")["_f"].min()
            .sort_values().index.tolist()
    )
    if len(slide_order) != 5:
        raise RuntimeError(f"Expected 5 slides, got {len(slide_order)}")
    slide_map = {s: i for i, s in enumerate(slide_order)}

    meta["_fov"] = meta["image_id"].map(fov_map).astype(int)
    meta["_slide"] = meta["slide"].map(slide_map).astype(int)
    meta = meta.sort_values(["_fov", "anchor_id"], kind="mergesort").reset_index(drop=True)
    if len(meta) != 1144:
        raise RuntimeError(f"Expected 1144 anchors, got {len(meta)}")

    anchor_pos = {a: i for i, a in enumerate(meta["anchor_id"])}
    obs_u = np.zeros((5, len(meta)), dtype=bool)
    obs_e = np.zeros_like(obs_u)
    for rr, reader in enumerate(PRIMARY_READERS):
        for cond, target in [("Unbiased", obs_u), ("Evaluation", obs_e)]:
            ids = d.loc[(d.reader == reader) & (d.condition == cond), "anchor_id"]
            for a in ids:
                target[rr, anchor_pos[a]] = True

    if int(obs_u.sum() + obs_e.sum()) != 10833:
        raise RuntimeError(
            f"Primary observation total expected 10833, got {int(obs_u.sum()+obs_e.sum())}"
        )
    ai = meta["ai_detected"].astype(np.int8).to_numpy()
    if ai.sum() != 593:
        raise RuntimeError(f"Expected 593 AI positives, got {int(ai.sum())}")

    return {
        "ai": ai,
        "obs_U": obs_u,
        "obs_E": obs_e,
        "fov_index": meta["_fov"].to_numpy(np.int32),
        "slide_index": meta["_slide"].to_numpy(np.int16),
        "patient_index": meta["_slide"].to_numpy(np.int16),
        "slide_order": slide_order,
        "fov_order": fov_order,
    }

def expand_design(base, n_readers: int, n_slides: int):
    if n_readers not in {5, 10} or n_slides not in {5, 10, 20}:
        raise ValueError((n_readers, n_slides))
    blocks = n_slides // 5
    ai = np.tile(base["ai"], blocks)
    f = np.concatenate([base["fov_index"] + 52*b for b in range(blocks)]).astype(np.int32)
    s = np.concatenate([base["slide_index"] + 5*b for b in range(blocks)]).astype(np.int16)
    p = s.copy()
    ou = np.tile(base["obs_U"], (1, blocks))
    oe = np.tile(base["obs_E"], (1, blocks))
    if n_readers == 10:
        ou = np.tile(ou, (2, 1))
        oe = np.tile(oe, (2, 1))
    return ai.astype(np.int8), ou, oe, f, s, p

def true_estimands(params, d_shared, ai):
    sdu = np.sqrt(
        params["sd_reader"]**2 +
        params["sd_reader_fov"]**2 +
        params["sd_reader_anchor"]**2
    )
    sde = np.sqrt(
        params["sd_reader"]**2 +
        params["sd_reader_condition"]**2 +
        params["sd_reader_fov"]**2 +
        params["sd_reader_anchor"]**2
    )
    pu = logistic_normal_mean80(params["alpha"] + d_shared, sdu)
    pe = logistic_normal_mean80(params["alpha"] + params["beta_E"] + d_shared, sde)
    af = ai.astype(float)
    tu = float(np.mean(af*pu + (1-af)*(1-pu)))
    te = float(np.mean(af*pe + (1-af)*(1-pe)))
    return tu, te, te-tu

def generate_one(base, scenario, rep, freeze_root: Path):
    spec = scenario_spec(scenario)
    gen = spec["generator"]
    nr, ns = spec["n_readers"], spec["n_slides"]
    params = DGP[gen]
    seed = deterministic_seed(scenario, rep)
    rng = np.random.default_rng(seed)

    ai_template, ou, oe, f, s, p = expand_design(base, nr, ns)
    na, nf = len(f), int(f.max()) + 1

    g = rng.normal(0.0, params["sd_fov"], size=nf)
    aa = rng.normal(0.0, params["sd_anchor"], size=na)
    d_shared = g[f] + aa

    if gen == "M2":
        pai = expit(params["alpha_ai"] + params["beta_ai"] * d_shared)
        ai = rng.binomial(1, pai).astype(np.int8)
    else:
        ai = ai_template.copy()

    u = rng.normal(0.0, params["sd_reader"], size=nr)
    b = rng.normal(0.0, params["sd_reader_condition"], size=nr)
    h = rng.normal(0.0, params["sd_reader_fov"], size=(nr, nf))
    q = rng.normal(0.0, params["sd_reader_anchor"], size=(nr, na))

    eta_u = params["alpha"] + d_shared[None, :] + u[:, None] + h[:, f] + q
    eta_e = (
        params["alpha"] + params["beta_E"] + d_shared[None, :]
        + u[:, None] + b[:, None] + h[:, f] + q
    )
    yu_full = rng.binomial(1, expit(eta_u)).astype(np.int8)
    ye_full = rng.binomial(1, expit(eta_e)).astype(np.int8)
    yu = np.where(ou, yu_full, -1).astype(np.int8)
    ye = np.where(oe, ye_full, -1).astype(np.int8)
    selected = np.ones(na, dtype=bool)

    tu, te, delta = true_estimands(params, d_shared, ai)

    data_dir = freeze_root / "data" / scenario
    truth_dir = freeze_root / "truth" / scenario
    data_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / f"rep_{rep:03d}.npz"
    truth_path = truth_dir / f"rep_{rep:03d}.json"

    tmp = data_path.with_suffix(".npz.tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(
            fh,
            ai=ai, selected=selected,
            reader_y_U=yu, reader_y_E=ye,
            obs_U=ou, obs_E=oe,
            fov_index=f, slide_index=s, patient_index=p,
        )
    tmp.replace(data_path)

    truth = {
        "protocol_version": PROTOCOL,
        "scenario": scenario,
        "rep": rep,
        "seed": int(seed),
        **spec,
        "n_fovs": int(nf),
        "n_anchors": int(na),
        "n_obs_U": int(ou.sum()),
        "n_obs_E": int(oe.sum()),
        "ai_positive": int(ai.sum()),
        "theta_U_selected": tu,
        "theta_E_selected": te,
        "delta_E_minus_U_selected": delta,
        "truth_quadrature_nodes": 80,
    }
    truth_path.write_text(json.dumps(truth, indent=2) + "\n")
    return truth

def freeze_all(source_csv: Path, freeze_root: Path, expected_hash_out: Path):
    if freeze_root.exists() and any(freeze_root.iterdir()):
        raise RuntimeError(f"Freeze root is not empty: {freeze_root}")
    freeze_root.mkdir(parents=True, exist_ok=True)
    base = source_template(source_csv)

    (freeze_root / "DGP_PARAMETERS.json").write_text(json.dumps({
        "protocol_version": PROTOCOL,
        "parameter_source": "posterior means from validated v23.1 application fits",
        "M1": DGP["M1"],
        "M2": DGP["M2"],
        "primary_reader_mask_templates": PRIMARY_READERS,
        "M1_ai_rule": "fixed empirical 1144-anchor AI vector replicated by 5-slide block",
        "M2_ai_rule": "Bernoulli(expit(alpha_ai + beta_ai*d_shared))",
        "future_reader_truth": (
            "conditional on realized item/FOV effects and frozen AI vector; "
            "integrates new reader, reader-condition, reader-FOV, and reader-anchor Gaussian effects"
        ),
    }, indent=2) + "\n")
    (freeze_root / "SCENARIOS.json").write_text(
        json.dumps({s: scenario_spec(s) for s in SCENARIOS}, indent=2) + "\n"
    )

    rows = []
    for scenario in SCENARIOS:
        print(f"GENERATING {scenario}", flush=True)
        for rep in range(N_REPS):
            rows.append(generate_one(base, scenario, rep, freeze_root))

    pd.DataFrame([{
        "scenario": x["scenario"], "rep": x["rep"], "seed": x["seed"]
    } for x in rows]).to_csv(freeze_root / "SEEDS.csv", index=False)

    audit = (
        pd.DataFrame(rows)
        .groupby(["scenario", "generator", "n_readers", "n_slides"], as_index=False)
        .agg(
            n_reps=("rep", "size"),
            n_fovs=("n_fovs", "first"),
            n_anchors=("n_anchors", "first"),
            n_obs_U=("n_obs_U", "first"),
            n_obs_E=("n_obs_E", "first"),
            mean_ai_positive=("ai_positive", "mean"),
            mean_true_theta_U=("theta_U_selected", "mean"),
            mean_true_theta_E=("theta_E_selected", "mean"),
            mean_true_delta=("delta_E_minus_U_selected", "mean"),
        )
    )
    audit.to_csv(freeze_root / "DESIGN_AUDIT.csv", index=False)

    include = []
    for path in sorted(freeze_root.rglob("*")):
        if path.is_file() and path.name not in {"SHA256SUMS.txt", "FREEZE_MANIFEST.json"}:
            rel = path.relative_to(freeze_root).as_posix()
            include.append((sha256_file(path), rel))
    sha_text = "".join(f"{h}  {rel}\n" for h, rel in include)
    (freeze_root / "SHA256SUMS.txt").write_text(sha_text)
    freeze_hash = hashlib.sha256(sha_text.encode()).hexdigest()

    man = {
        "protocol_version": PROTOCOL,
        "n_frozen_datasets": len(SCENARIOS) * N_REPS,
        "n_scenarios": len(SCENARIOS),
        "n_reps_per_scenario": N_REPS,
        "frozen_data_sha256": freeze_hash,
        "source_csv": str(source_csv),
    }
    (freeze_root / "FREEZE_MANIFEST.json").write_text(json.dumps(man, indent=2) + "\n")
    expected_hash_out.parent.mkdir(parents=True, exist_ok=True)
    expected_hash_out.write_text(freeze_hash + "\n")

    print("\nV26_1_FREEZE_COMPLETE")
    print(json.dumps(man, indent=2))
    print("\nDESIGN_AUDIT")
    print(audit.to_string(index=False))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-csv", type=Path, required=True)
    ap.add_argument("--freeze-root", type=Path, required=True)
    ap.add_argument("--expected-hash-out", type=Path, required=True)
    args = ap.parse_args()
    freeze_all(args.source_csv, args.freeze_root, args.expected_hash_out)

if __name__ == "__main__":
    main()
