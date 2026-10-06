#!/usr/bin/env python3
"""
NuCLS v24.2 M4 residual-dependence sensitivity analysis.

Implements the reviewer-requested fallback when M3 is computationally unstable:
a PRESPECIFIED residual association sensitivity grid, not an estimated extra
latent factor.

Baseline = fitted M1 reader-only posterior. For each posterior draw and
condition c, let p0_ic be the future-reader detection probability for anchor i.

For fixed sensitivity gamma:
    logit(p_gamma_ic) = logit(p0_ic) + kappa_c(gamma) + gamma * A_i

where A_i is the observed fixed external-AI detection indicator and kappa is
solved separately for every posterior draw and condition so that:
    mean_i p_gamma_ic = mean_i p0_ic.

Thus gamma changes residual AI-reader association while preserving the
condition-specific marginal future-reader positivity in each posterior draw.
exp(gamma) is the residual odds-ratio multiplier comparing AI-positive with
AI-negative anchors, conditional on the M1 anchor/FOV predictions.

gamma is NOT estimated from the data. This is a structural sensitivity curve.
The default grid [-2,2] by 0.25 corresponds to residual OR 0.135 to 7.389 and
is intentionally broad; it is not labeled a scientifically plausible range.

No additional MCMC is performed. M4 propagates the already-fitted M1 posterior.
"""
from __future__ import annotations
from repo_paths import repo_root

import argparse, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit, logit

EXPECTED_READERS = ["SP.3", "JP.1", "JP.2", "JP.5", "JP.6"]
EXPECTED_N_ROWS = 10833
EXPECTED_N_ANCHORS = 1144
EXPECTED_N_FOV = 52
EXPECTED_N_SLIDES = 5
EXPECTED_AI_POSITIVE = 593

GHX, GHW = hermgauss(30)
GHW = GHW / np.sqrt(np.pi)

TARGETS = ["theta_U_selected", "theta_E_selected", "delta_E_minus_U_selected"]


def logistic_normal_mean(mu, sd):
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)
    z = mu[..., None] + np.sqrt(2.0) * sd[..., None] * GHX
    return np.sum(expit(z) * GHW, axis=-1)


def residual_recenter(p0, ai, gamma, n_iter=48):
    """Apply residual log-odds shift and preserve each row's mean probability."""
    p0 = np.asarray(p0, dtype=float)
    ai = np.asarray(ai, dtype=float)
    if p0.ndim != 2 or ai.ndim != 1 or p0.shape[1] != ai.size:
        raise ValueError("shape mismatch in residual_recenter")
    if abs(float(gamma)) < 1e-15:
        return p0.copy()

    eps = 1e-12
    lp = logit(np.clip(p0, eps, 1.0 - eps))
    target = p0.mean(axis=1)
    offset = float(gamma) * ai[None, :]

    lo = np.full(p0.shape[0], -30.0 - abs(float(gamma)))
    hi = np.full(p0.shape[0],  30.0 + abs(float(gamma)))
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        m = expit(lp + mid[:, None] + offset).mean(axis=1)
        go_right = m < target
        lo = np.where(go_right, mid, lo)
        hi = np.where(go_right, hi, mid)
    kappa = 0.5 * (lo + hi)
    out = expit(lp + kappa[:, None] + offset)

    err = np.max(np.abs(out.mean(axis=1) - target))
    if not np.isfinite(err) or err > 5e-11:
        raise RuntimeError(f"marginal-recentering check failed: max error={err}")
    return out


def self_test():
    rng = np.random.default_rng(2402)
    p = rng.uniform(0.05, 0.95, size=(7, 31))
    ai = np.array(([0, 1] * 16)[:31], dtype=float)
    q0 = residual_recenter(p, ai, 0.0)
    if not np.array_equal(p, q0):
        raise RuntimeError("gamma=0 does not return baseline exactly")
    for g in (-2.0, -0.5, 0.5, 2.0):
        q = residual_recenter(p, ai, g)
        if np.max(np.abs(q.mean(axis=1) - p.mean(axis=1))) > 5e-11:
            raise RuntimeError("self-test marginal preservation failed")
        if not np.all((q > 0) & (q < 1)):
            raise RuntimeError("self-test produced invalid probabilities")
    print("M4_SELF_TEST_OK", flush=True)


def load_frame(root):
    real_path = root/"outputs"/"scoring_v16_1"/"restricted_recorded_reader_detection_agreement_v16.csv"
    ai_path = root/"outputs"/"scoring_v16_1"/"restricted_anchor_matches_v16.csv"
    raw = pd.read_csv(real_path)
    ai_tbl = pd.read_csv(ai_path)
    d = raw.loc[raw.reader.isin(EXPECTED_READERS)].copy()

    checks = {
        "rows": (len(d), EXPECTED_N_ROWS),
        "anchors": (d.anchor_id.nunique(), EXPECTED_N_ANCHORS),
        "fovs": (d.image_id.nunique(), EXPECTED_N_FOV),
        "slides": (d.slide.nunique(), EXPECTED_N_SLIDES),
    }
    for name, (got, exp) in checks.items():
        if got != exp:
            raise RuntimeError(f"Expected {exp} {name}, got {got}")
    if set(d.condition.unique()) != {"Unbiased", "Evaluation"}:
        raise RuntimeError(f"Unexpected condition labels: {sorted(d.condition.unique())}")

    anchors = sorted(d.anchor_id.astype(str).unique())
    fovs = sorted(d.image_id.astype(str).unique())
    fmap = {x:i for i,x in enumerate(fovs)}
    tmp = d.assign(anchor_id=d.anchor_id.astype(str), image_id=d.image_id.astype(str))
    amap = tmp.groupby("anchor_id").image_id.agg(["first", "nunique"]).reindex(anchors)
    if (amap["nunique"] != 1).any():
        raise RuntimeError("anchor maps to multiple FOVs")
    anchor_f = amap["first"].map(fmap).astype(int).to_numpy()

    a = ai_tbl[["anchor_id", "ai_detected"]].copy()
    a["anchor_id"] = a.anchor_id.astype(str)
    if a.anchor_id.duplicated().any():
        nun = a.groupby("anchor_id").ai_detected.nunique(dropna=False)
        if (nun > 1).any():
            raise RuntimeError("conflicting AI labels")
        a = a.drop_duplicates("anchor_id")
    av = a.set_index("anchor_id").ai_detected.reindex(anchors)
    if av.isna().any():
        raise RuntimeError("AI vector missing selected anchors")
    ai = av.astype(np.int8).to_numpy()
    if int(ai.sum()) != EXPECTED_AI_POSITIVE:
        raise RuntimeError(f"Expected {EXPECTED_AI_POSITIVE} AI positives, got {int(ai.sum())}")

    return ai, anchor_f, {
        "n_rows": len(d),
        "n_anchors": len(anchors),
        "n_fovs": len(fovs),
        "n_slides": d.slide.nunique(),
        "ai_positive": int(ai.sum()),
    }


def flat(post, name):
    x = np.asarray(post[name].values)
    return x.reshape((-1,) + x.shape[2:])


def summarize(draw_mat, gammas):
    rows = []
    for j, g in enumerate(gammas):
        for k, target in enumerate(TARGETS):
            x = draw_mat[:, j, k]
            lo, hi = np.quantile(x, [0.025, 0.975])
            rows.append({
                "gamma": float(g),
                "residual_OR": float(np.exp(g)),
                "target": target,
                "estimate": float(x.mean()),
                "posterior_sd": float(x.std(ddof=1)),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "interval_width": float(hi-lo),
                "prob_gt_zero": float(np.mean(x > 0)) if target == "delta_E_minus_U_selected" else np.nan,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=repo_root())
    ap.add_argument("--m1-idata", type=Path, default=None)
    ap.add_argument("--m1-draws", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--gamma-min", type=float, default=-2.0)
    ap.add_argument("--gamma-max", type=float, default=2.0)
    ap.add_argument("--gamma-step", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=100)
    args = ap.parse_args()

    self_test()
    root = args.project_root
    idata_path = args.m1_idata or root/"outputs"/"v23_1_reader_only_ablation"/"v23_1_reader_only_ablation_idata.nc"
    m1_draws_path = args.m1_draws or root/"outputs"/"v23_1_reader_only_ablation"/"v23_1_reader_only_ablation_estimand_draws.csv"
    m1_gate_path = root/"outputs"/"v23_1_reader_only_ablation"/"v23_1_reader_only_ablation_gate.json"
    outdir = args.outdir or root/"outputs"/"v24_2_m4_residual_dependence"
    outdir.mkdir(parents=True, exist_ok=True)

    if not idata_path.exists():
        raise FileNotFoundError(f"M1 posterior not found: {idata_path}")

    ai, anchor_f, frame_audit = load_frame(root)

    import arviz as az
    t0 = time.time()
    idata = az.from_netcdf(idata_path)
    p = idata.posterior

    required = [
        "alpha", "beta_E", "g_fov", "a_anchor",
        "sd_reader", "sd_reader_condition", "sd_reader_fov", "sd_reader_anchor",
    ]
    missing = [x for x in required if x not in p]
    if missing:
        raise RuntimeError(f"M1 posterior is missing variables: {missing}")

    alpha = flat(p, "alpha")
    beta = flat(p, "beta_E")
    gf = flat(p, "g_fov")
    aa = flat(p, "a_anchor")
    sr = flat(p, "sd_reader")
    sb = flat(p, "sd_reader_condition")
    srf = flat(p, "sd_reader_fov")
    sra = flat(p, "sd_reader_anchor")

    n = len(alpha)
    if gf.shape[1] != EXPECTED_N_FOV or aa.shape[1] != EXPECTED_N_ANCHORS:
        raise RuntimeError(f"Unexpected M1 latent dimensions: gf={gf.shape}, aa={aa.shape}")

    gammas = np.arange(args.gamma_min, args.gamma_max + args.gamma_step/2, args.gamma_step)
    if not np.any(np.isclose(gammas, 0.0)):
        raise RuntimeError("gamma grid must include zero")
    zero_j = int(np.argmin(np.abs(gammas)))
    gammas[zero_j] = 0.0

    out = np.empty((n, len(gammas), 3), dtype=float)
    baseline = np.empty((n, 3), dtype=float)

    for lo in range(0, n, args.batch):
        hi = min(n, lo + args.batch)
        base = alpha[lo:hi, None] + gf[lo:hi][:, anchor_f] + aa[lo:hi]
        sdu = np.sqrt(sr[lo:hi]**2 + srf[lo:hi]**2 + sra[lo:hi]**2)
        sde = np.sqrt(sr[lo:hi]**2 + sb[lo:hi]**2 + srf[lo:hi]**2 + sra[lo:hi]**2)
        pu0 = logistic_normal_mean(base, sdu[:, None])
        pe0 = logistic_normal_mean(base + beta[lo:hi, None], sde[:, None])

        au0 = ai[None, :]*pu0 + (1-ai[None, :])*(1-pu0)
        ae0 = ai[None, :]*pe0 + (1-ai[None, :])*(1-pe0)
        tu0, te0 = au0.mean(axis=1), ae0.mean(axis=1)
        baseline[lo:hi] = np.column_stack([tu0, te0, te0-tu0])

        for j, g in enumerate(gammas):
            pu = residual_recenter(pu0, ai, g)
            pe = residual_recenter(pe0, ai, g)
            au = ai[None, :]*pu + (1-ai[None, :])*(1-pu)
            ae = ai[None, :]*pe + (1-ai[None, :])*(1-pe)
            tu, te = au.mean(axis=1), ae.mean(axis=1)
            out[lo:hi, j, :] = np.column_stack([tu, te, te-tu])

    gamma0_diff = float(np.max(np.abs(out[:, zero_j, :] - baseline)))
    if gamma0_diff > 1e-11:
        raise RuntimeError(f"gamma=0 does not reproduce internally computed M1: {gamma0_diff}")

    external_m1_check = None
    if m1_draws_path.exists():
        old = pd.read_csv(m1_draws_path)
        if len(old) != n or not set(TARGETS).issubset(old.columns):
            raise RuntimeError(f"Existing M1 draw file incompatible: {m1_draws_path}")
        ext = old[TARGETS].to_numpy(float)
        external_m1_check = float(np.max(np.abs(ext - baseline)))
        if external_m1_check > 5e-10:
            raise RuntimeError(f"M4 gamma=0 fails frozen M1 reproduction: max diff={external_m1_check}")

    summ = summarize(out, gammas)

    # Save posterior draws in a compact NPZ, plus human-readable summary.
    np.savez_compressed(
        outdir/"v24_2_m4_sensitivity_draws.npz",
        gammas=gammas,
        target_names=np.array(TARGETS, dtype="U64"),
        draws=out,
    )
    summ.to_csv(outdir/"v24_2_m4_sensitivity_summary.csv", index=False)

    # Wide display table for immediate terminal review.
    wide = []
    for g in gammas:
        z = summ[summ.gamma == g].set_index("target")
        wide.append({
            "gamma": float(g),
            "residual_OR": float(np.exp(g)),
            "theta_U": float(z.loc["theta_U_selected", "estimate"]),
            "theta_U_low": float(z.loc["theta_U_selected", "ci_low"]),
            "theta_U_high": float(z.loc["theta_U_selected", "ci_high"]),
            "theta_E": float(z.loc["theta_E_selected", "estimate"]),
            "theta_E_low": float(z.loc["theta_E_selected", "ci_low"]),
            "theta_E_high": float(z.loc["theta_E_selected", "ci_high"]),
            "delta": float(z.loc["delta_E_minus_U_selected", "estimate"]),
            "delta_low": float(z.loc["delta_E_minus_U_selected", "ci_low"]),
            "delta_high": float(z.loc["delta_E_minus_U_selected", "ci_high"]),
        })
    wide = pd.DataFrame(wide)
    wide.to_csv(outdir/"v24_2_m4_sensitivity_curve_wide.csv", index=False)

    env = {}
    for target in TARGETS:
        z = summ[summ.target == target]
        env[target] = {
            "estimate_min": float(z.estimate.min()),
            "estimate_max": float(z.estimate.max()),
            "estimate_span": float(z.estimate.max()-z.estimate.min()),
        }

    inherited = None
    if m1_gate_path.exists():
        g = json.loads(m1_gate_path.read_text())
        inherited = {
            "divergences": g.get("divergences"),
            "max_rhat_scalar": g.get("max_rhat_scalar"),
            "min_ess_bulk_scalar": g.get("min_ess_bulk_scalar"),
            "min_ess_tail_scalar": g.get("min_ess_tail_scalar"),
            "sampler": g.get("sampler"),
        }

    gate = {
        "status": "V24_2_M4_RESIDUAL_DEPENDENCE_SENSITIVITY_COMPLETE",
        "analysis": "M4 residual-dependence sensitivity curve anchored to frozen M1 posterior",
        "formula": "logit(p_gamma_ic)=logit(p0_ic)+kappa_c_draw(gamma)+gamma*A_i; kappa chosen so mean_i(p_gamma_ic)=mean_i(p0_ic)",
        "interpretation": "gamma is prespecified/not estimated; exp(gamma) is a residual odds-ratio multiplier for future-reader detection at AI-positive vs AI-negative anchors, conditional on M1 predictions",
        "gamma_grid": [float(x) for x in gammas],
        "residual_OR_range": [float(np.exp(gammas.min())), float(np.exp(gammas.max()))],
        "frame": frame_audit,
        "m1_idata": str(idata_path),
        "n_posterior_draws": int(n),
        "gamma0_internal_max_abs_diff": gamma0_diff,
        "gamma0_external_frozen_m1_max_abs_diff": external_m1_check,
        "inherited_M1_sampling_diagnostics": inherited,
        "sensitivity_envelope": env,
        "elapsed_sec": float(time.time()-t0),
    }
    (outdir/"v24_2_m4_gate.json").write_text(json.dumps(gate, indent=2))

    print("M4_INPUT_AUDIT "+json.dumps(frame_audit, sort_keys=True), flush=True)
    print(f"M4_M1_DRAWS n={n} gamma0_internal_max_abs_diff={gamma0_diff:.3g} "
          f"gamma0_external_max_abs_diff={external_m1_check}", flush=True)
    print("M4_SENSITIVITY_CURVE", flush=True)
    print(wide.to_string(index=False), flush=True)
    print("M4_ENVELOPE "+json.dumps(env, sort_keys=True), flush=True)
    print("M4_RESIDUAL_DEPENDENCE_SENSITIVITY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
