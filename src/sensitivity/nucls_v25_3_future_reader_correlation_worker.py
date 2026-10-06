
#!/usr/bin/env python3

from __future__ import annotations
from repo_paths import repo_root



import argparse, json, os, sys, time

from pathlib import Path



import numpy as np

import pandas as pd

from scipy.special import expit

from numpy.polynomial.hermite import hermgauss



ROOT_DEFAULT = repo_root()



EXPECTED_READERS = ["SP.3","JP.1","JP.2","JP.5","JP.6"]

EXPECTED_N_ROWS = 10833

EXPECTED_N_ANCHORS = 1144

EXPECTED_N_FOV = 52

EXPECTED_N_SLIDES = 5

EXPECTED_AI_POSITIVE = 593



GHX, GHW = hermgauss(30)

GHW = GHW / np.sqrt(np.pi)





def logistic_normal_mean(mu, sd):

    mu = np.asarray(mu, float)

    sd = np.asarray(sd, float)

    z = mu[..., None] + np.sqrt(2.0) * sd[..., None] * GHX

    return np.sum(expit(z) * GHW, axis=-1)





def rho_tag(rho):

    sign = "m" if rho < 0 else "p"

    return sign + f"{abs(rho):.2f}".replace(".", "p")





def load_encoded(root):

    sys.path.insert(0, str(root / "scripts"))

    import nucls_v23_1_joint_bayes_logit as v231



    real_path = (

        root / "outputs" / "scoring_v16_1" /

        "restricted_recorded_reader_detection_agreement_v16.csv"

    )

    ai_path = (

        root / "outputs" / "scoring_v16_1" /

        "restricted_anchor_matches_v16.csv"

    )



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



    if set(d.reader.unique()) != set(EXPECTED_READERS):

        raise RuntimeError(

            f"Unexpected readers: {sorted(d.reader.unique())}"

        )



    if set(d.condition.unique()) != {"Unbiased", "Evaluation"}:

        raise RuntimeError(

            f"Unexpected conditions: {sorted(d.condition.unique())}"

        )



    fov_values = sorted(d.image_id.astype(str).unique())

    fov_map = {name: i for i, name in enumerate(fov_values)}



    rl = pd.DataFrame({

        "anchor_id": d.anchor_id.astype(str),

        "fov_id": d.image_id.astype(str).map(fov_map).astype(float),

        "reader": d.reader.astype(str),

        "condition": d.condition.map({

            "Unbiased": "U",

            "Evaluation": "E"

        }),

        "observed": np.ones(len(d), dtype=np.int8),

        "reader_detected": d.reader_detected.astype(np.int8),

    })



    if rl.duplicated(["anchor_id", "reader", "condition"]).any():

        raise RuntimeError("Duplicate anchor-reader-condition rows")



    a = ai_tbl[["anchor_id", "ai_detected"]].copy()

    a["anchor_id"] = a.anchor_id.astype(str)



    if a.anchor_id.duplicated().any():

        chk = a.groupby("anchor_id").ai_detected.nunique(dropna=False)

        if (chk > 1).any():

            raise RuntimeError("Conflicting AI labels")

        a = a.drop_duplicates("anchor_id")



    if len(a) != EXPECTED_N_ANCHORS:

        raise RuntimeError(

            f"Expected {EXPECTED_N_ANCHORS} AI anchors, got {len(a)}"

        )



    ai = pd.Series(

        a.ai_detected.astype(np.int8).to_numpy(),

        index=a.anchor_id

    )



    if int(ai.sum()) != EXPECTED_AI_POSITIVE:

        raise RuntimeError(

            f"Expected {EXPECTED_AI_POSITIVE} AI positives, "

            f"got {int(ai.sum())}"

        )



    enc = v231.encode_observed(

        rl,

        ai,

        readers=v231.PRIMARY_READERS,

        expected_ai_matches=EXPECTED_AI_POSITIVE,

    )



    return v231, enc





def build_model(data, model_name, rho):

    import pymc as pm



    if model_name not in {"M1", "M2"}:

        raise ValueError(model_name)



    if not (-1.0 < rho < 1.0):

        raise ValueError("rho must be strictly between -1 and 1")



    coords = {

        "reader": data.reader_names,

        "fov": np.arange(len(data.fov_values)),

        "anchor": np.arange(len(data.anchor_ids)),

        "rf_obs": np.arange(len(data.rf_keys)),

        "ra_obs": np.arange(len(data.ra_keys)),

        "obs": np.arange(len(data.y)),

    }



    sqrt_one_minus_rho2 = float(np.sqrt(1.0 - rho*rho))



    with pm.Model(coords=coords) as model:



        alpha = pm.Normal("alpha", 0, 2.5)

        beta_E = pm.Normal("beta_E", 0, 1.0)



        if model_name == "M2":

            alpha_ai = pm.Normal("alpha_ai", 0, 2.5)

            beta_ai = pm.HalfNormal("beta_ai", 1.5)



        sd_reader = pm.HalfNormal("sd_reader", 1.0)

        sd_reader_condition = pm.HalfNormal(

            "sd_reader_condition", 1.0

        )

        sd_fov = pm.HalfNormal("sd_fov", 1.0)

        sd_reader_fov = pm.HalfNormal("sd_reader_fov", 1.0)

        sd_anchor = pm.HalfNormal("sd_anchor", 1.5)

        sd_reader_anchor = pm.HalfNormal(

            "sd_reader_anchor", 1.5

        )



        z_reader = pm.Normal(

            "z_reader", 0, 1, dims="reader"

        )

        z_reader_condition = pm.Normal(

            "z_reader_condition", 0, 1, dims="reader"

        )

        z_fov = pm.Normal(

            "z_fov", 0, 1, dims="fov"

        )

        z_reader_fov = pm.Normal(

            "z_reader_fov", 0, 1, dims="rf_obs"

        )

        z_anchor = pm.Normal(

            "z_anchor", 0, 1, dims="anchor"

        )

        z_reader_anchor = pm.Normal(

            "z_reader_anchor", 0, 1, dims="ra_obs"

        )



        # Fixed-rho bivariate reader construction:

        #

        # Corr(u_r, b_r) = rho

        #

        # Marginal scales remain sd_reader and sd_reader_condition.

        u = pm.Deterministic(

            "u_reader",

            sd_reader * z_reader,

            dims="reader",

        )



        b_standardized = (

            rho * z_reader

            + sqrt_one_minus_rho2 * z_reader_condition

        )



        b = pm.Deterministic(

            "b_reader_condition",

            sd_reader_condition * b_standardized,

            dims="reader",

        )



        g = pm.Deterministic(

            "g_fov",

            sd_fov * z_fov,

            dims="fov",

        )



        h = pm.Deterministic(

            "h_reader_fov",

            sd_reader_fov * z_reader_fov,

            dims="rf_obs",

        )



        aa = pm.Deterministic(

            "a_anchor",

            sd_anchor * z_anchor,

            dims="anchor",

        )



        q = pm.Deterministic(

            "q_reader_anchor",

            sd_reader_anchor * z_reader_anchor,

            dims="ra_obs",

        )



        d_shared = g[data.anchor_f] + aa



        if model_name == "M2":

            pm.Bernoulli(

                "ai_detected_obs",

                logit_p=alpha_ai + beta_ai*d_shared,

                observed=data.anchor_ai,

                dims="anchor",

            )



        eta = (

            alpha

            + beta_E * data.c

            + d_shared[data.a]

            + u[data.r]

            + b[data.r] * data.c

            + h[data.rf]

            + q[data.ra]

        )



        pm.Bernoulli(

            "reader_detected",

            logit_p=eta,

            observed=data.y,

            dims="obs",

        )



    return model





def estimands_rho(

    idata,

    data,

    rho,

    include_reader_anchor=True,

    batch=100,

):

    p = idata.posterior



    def flat(name):

        x = np.asarray(p[name].values)

        return x.reshape((-1,) + x.shape[2:])



    alpha = flat("alpha")

    beta = flat("beta_E")

    g = flat("g_fov")

    aa = flat("a_anchor")



    sr = flat("sd_reader")

    sb = flat("sd_reader_condition")

    srf = flat("sd_reader_fov")

    sra = flat("sd_reader_anchor")



    ai = data.anchor_ai.astype(float)



    n = len(alpha)

    out = np.empty((n, 3), float)



    if include_reader_anchor:

        qvar = sra**2

    else:

        qvar = np.zeros_like(sra)



    for lo in range(0, n, batch):

        hi = min(n, lo + batch)



        base = (

            alpha[lo:hi, None]

            + g[lo:hi][:, data.anchor_f]

            + aa[lo:hi]

        )



        var_u = (

            sr[lo:hi]**2

            + srf[lo:hi]**2

            + qvar[lo:hi]

        )



        var_e = (

            sr[lo:hi]**2

            + sb[lo:hi]**2

            + 2.0 * rho * sr[lo:hi] * sb[lo:hi]

            + srf[lo:hi]**2

            + qvar[lo:hi]

        )



        if (

            np.any(var_u < -1e-12)

            or np.any(var_e < -1e-12)

        ):

            raise RuntimeError(

                "Negative future-reader variance encountered"

            )



        sd_u = np.sqrt(np.maximum(var_u, 0.0))

        sd_e = np.sqrt(np.maximum(var_e, 0.0))



        p_u = logistic_normal_mean(

            base,

            sd_u[:, None],

        )



        p_e = logistic_normal_mean(

            base + beta[lo:hi, None],

            sd_e[:, None],

        )



        agree_u = (

            ai[None, :] * p_u

            + (1-ai[None, :]) * (1-p_u)

        )



        agree_e = (

            ai[None, :] * p_e

            + (1-ai[None, :]) * (1-p_e)

        )



        theta_u = agree_u.mean(axis=1)

        theta_e = agree_e.mean(axis=1)



        out[lo:hi] = np.column_stack([

            theta_u,

            theta_e,

            theta_e-theta_u,

        ])



    return pd.DataFrame(

        out,

        columns=[

            "theta_U_selected",

            "theta_E_selected",

            "delta_E_minus_U_selected",

        ],

    )





def summarize(draws, construction):

    rows = []



    for target in draws.columns:

        x = draws[target].to_numpy(float)

        lo, hi = np.quantile(x, [0.025, 0.975])



        rows.append({

            "future_reader_construction": construction,

            "target": target,

            "estimate": float(x.mean()),

            "posterior_sd": float(x.std(ddof=1)),

            "ci_low": float(lo),

            "ci_high": float(hi),

            "interval_width": float(hi-lo),

            "prob_gt_zero":

                float(np.mean(x > 0))

                if target == "delta_E_minus_U_selected"

                else np.nan,

        })



    return pd.DataFrame(rows)





def original_paths(root, model_name):

    if model_name == "M2":

        d = root / "outputs" / "v23_1_realdata_primary"

        return (

            d / "v23_1_realdata_primary_idata.nc",

            d / "v23_1_realdata_primary_estimand_draws.csv",

            d / "v23_1_realdata_primary_summary.csv",

        )



    d = root / "outputs" / "v23_1_reader_only_ablation"



    return (

        d / "v23_1_reader_only_ablation_idata.nc",

        d / "v23_1_reader_only_ablation_estimand_draws.csv",

        d / "v23_1_reader_only_ablation_summary.csv",

    )





def atomic_netcdf(idata, path):

    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.unlink(missing_ok=True)

    idata.to_netcdf(tmp)

    os.replace(tmp, path)





def main():

    ap = argparse.ArgumentParser()



    ap.add_argument(

        "--project-root",

        type=Path,

        default=ROOT_DEFAULT,

    )



    ap.add_argument(

        "--model",

        choices=["M1", "M2"],

        required=True,

    )



    ap.add_argument(

        "--rho",

        type=float,

        required=True,

    )



    ap.add_argument("--draws", type=int, default=4000)

    ap.add_argument("--tune", type=int, default=2000)

    ap.add_argument("--chains", type=int, default=4)

    ap.add_argument(

        "--target-accept",

        type=float,

        default=.99,

    )

    ap.add_argument("--seed", type=int, default=None)



    args = ap.parse_args()



    root = args.project_root

    rho = float(args.rho)



    if not (-1 < rho < 1):

        raise RuntimeError("rho must be in (-1,1)")



    v231, enc = load_encoded(root)

    v231.self_test()



    import arviz as az



    (

        original_idata_path,

        original_draws_path,

        original_summary_path,

    ) = original_paths(root, args.model)



    for pth in [

        original_idata_path,

        original_draws_path,

        original_summary_path,

    ]:

        if not pth.exists():

            raise FileNotFoundError(pth)



    # ----------------------------------------------------------

    # HARD IMPLEMENTATION CHECK

    #

    # The new estimator at rho=0, evaluated on the ORIGINAL

    # saved posterior, must reproduce every saved original

    # estimand draw to numerical precision.

    # ----------------------------------------------------------

    original_idata = az.from_netcdf(

        original_idata_path

    )



    original_saved = pd.read_csv(

        original_draws_path

    )[

        [

            "theta_U_selected",

            "theta_E_selected",

            "delta_E_minus_U_selected",

        ]

    ]



    original_recalculated = estimands_rho(

        original_idata,

        enc,

        0.0,

        True,

    )



    reproduction_max_abs = float(

        np.max(

            np.abs(

                original_recalculated.to_numpy()

                - original_saved.to_numpy()

            )

        )

    )



    if reproduction_max_abs > 1e-10:

        raise RuntimeError(

            "rho=0 postprocessing failed to reproduce "

            f"saved {args.model} estimand draws: "

            f"max abs diff={reproduction_max_abs}"

        )



    if args.seed is None:

        seed = 231003 if args.model == "M1" else 231001

    else:

        seed = args.seed



    import pymc as pm



    model = build_model(

        enc,

        args.model,

        rho,

    )



    t0 = time.time()



    with model:

        idata = pm.sample(

            draws=args.draws,

            tune=args.tune,

            chains=args.chains,

            cores=min(

                args.chains,

                max(

                    1,

                    int(

                        os.environ.get(

                            "SLURM_CPUS_PER_TASK",

                            "1",

                        )

                    ),

                ),

            ),

            random_seed=seed,

            target_accept=args.target_accept,

            init="jitter+adapt_diag",

            progressbar=False,

            return_inferencedata=True,

        )



    elapsed = time.time() - t0



    # Primary construction:

    # future reader gets new reader, reader×FOV, and reader×anchor effects.

    full = estimands_rho(

        idata,

        enc,

        rho,

        include_reader_anchor=True,

    )



    # Construction sensitivity:

    # condition on zero new reader×anchor deviation for the future reader.

    no_reader_anchor = estimands_rho(

        idata,

        enc,

        rho,

        include_reader_anchor=False,

    )



    summary = pd.concat(

        [

            summarize(

                full,

                "full_new_reader_effects",

            ),

            summarize(

                no_reader_anchor,

                "no_new_reader_anchor_effect",

            ),

        ],

        ignore_index=True,

    )



    scalar_vars = [

        "alpha",

        "beta_E",

        "sd_reader",

        "sd_reader_condition",

        "sd_fov",

        "sd_reader_fov",

        "sd_anchor",

        "sd_reader_anchor",

    ]



    if args.model == "M2":

        scalar_vars = [

            "alpha",

            "beta_E",

            "alpha_ai",

            "beta_ai",

            "sd_reader",

            "sd_reader_condition",

            "sd_fov",

            "sd_reader_fov",

            "sd_anchor",

            "sd_reader_anchor",

        ]



    scalar_diag = az.summary(

        idata,

        var_names=scalar_vars,

        round_to=None,

    )



    divergences = int(

        np.asarray(

            idata.sample_stats["diverging"]

        ).sum()

    )



    diagnostics = {

        "divergences": divergences,

        "max_rhat_scalar":

            float(scalar_diag.r_hat.max()),

        "min_ess_bulk_scalar":

            float(scalar_diag.ess_bulk.min()),

        "min_ess_tail_scalar":

            float(scalar_diag.ess_tail.min()),

    }



    computational_ok = bool(

        divergences == 0

        and diagnostics["max_rhat_scalar"] <= 1.01

        and diagnostics["min_ess_bulk_scalar"] >= 400

        and diagnostics["min_ess_tail_scalar"] >= 400

    )



    original_summary = pd.read_csv(

        original_summary_path

    )



    rho0_fit_shift = None



    if abs(rho) < 1e-15:

        primary_summary = summary[

            summary.future_reader_construction

            == "full_new_reader_effects"

        ]



        compare = primary_summary.merge(

            original_summary,

            on="target",

            suffixes=("_new", "_original"),

        )



        rho0_fit_shift = [

            {

                "target": row.target,

                "new_estimate":

                    float(row.estimate_new),

                "original_estimate":

                    float(row.estimate_original),

                "shift":

                    float(

                        row.estimate_new

                        - row.estimate_original

                    ),

            }

            for row in compare.itertuples(index=False)

        ]



    tag = rho_tag(rho)



    outdir = (

        root / "outputs" /

        "v25_3_future_reader_correlation" /

        args.model /

        tag

    )



    outdir.mkdir(

        parents=True,

        exist_ok=True,

    )



    draws_out = (

        full.add_suffix("__full")

        .join(

            no_reader_anchor.add_suffix(

                "__no_reader_anchor"

            )

        )

    )



    draws_out.insert(

        0,

        "rho_ub",

        rho,

    )



    draws_out.to_csv(

        outdir / "estimand_draws.csv",

        index=False,

    )



    summary.insert(

        0,

        "rho_ub",

        rho,

    )

    summary.insert(

        0,

        "model",

        args.model,

    )



    summary.to_csv(

        outdir / "summary.csv",

        index=False,

    )



    scalar_diag.to_csv(

        outdir / "scalar_diagnostics.csv"

    )



    atomic_netcdf(

        idata,

        outdir / "idata.nc",

    )



    gate = {

        "status":

            "V25_3_FUTURE_READER_CORRELATION_COMPLETE",



        "model":

            args.model,



        "rho_ub_fixed":

            rho,



        "sensitivity_interpretation":

            "Fixed-rho bounded sensitivity. "

            "rho_ub is not estimated as a precise "

            "population correlation from five readers.",



        "future_reader_full_variance_U":

            "sd_reader^2 + sd_reader_fov^2 "

            "+ sd_reader_anchor^2",



        "future_reader_full_variance_E":

            "sd_reader^2 + sd_reader_condition^2 "

            "+ 2*rho*sd_reader*sd_reader_condition "

            "+ sd_reader_fov^2 "

            "+ sd_reader_anchor^2",



        "alternate_construction":

            "no_new_reader_anchor_effect removes "

            "sd_reader_anchor^2 only from the "

            "future-reader integration; the fitted "

            "observed-reader likelihood is unchanged.",



        "rho0_saved_posterior_reproduction_max_abs":

            reproduction_max_abs,



        "elapsed_sec":

            elapsed,



        **diagnostics,



        "computational_ok":

            computational_ok,



        "sampler": {

            "draws": args.draws,

            "tune": args.tune,

            "chains": args.chains,

            "target_accept":

                args.target_accept,

            "seed": seed,

        },



        "rho0_newfit_vs_original_summary":

            rho0_fit_shift,



        "posterior_summary":

            summary.to_dict(

                orient="records"

            ),

    }



    (

        outdir / "gate.json"

    ).write_text(

        json.dumps(

            gate,

            indent=2,

        )

    )



    print(

        "V25_3_FUTURE_READER_CORRELATION_COMPLETE",

        flush=True,

    )

    print(

        json.dumps(

            gate,

            indent=2,

        ),

        flush=True,

    )





if __name__ == "__main__":

    main()

