#!/usr/bin/env python3
"""NuCLS M3: correlated dual-factor AI/human latent model.

Purpose
-------
M3 is an intermediate structural model between:
  M1 reader-only: AI does not inform the human latent process;
  M2 shared detectability: AI and human labels share exactly one FOV+anchor process.

M3 gives the human and AI channels distinct FOV and anchor latent components,
with estimated correlations at each level.

Human channel:
    d^H_i = g^H_f(i) + a^H_i

AI channel:
    d^A_i = g^A_f(i) + a^A_i
    A_i ~ Bernoulli(logit^{-1}(alpha_ai + d^A_i))

For each FOV f:
    (g^H_f, g^A_f)' ~ N(0, Sigma_f)

For each anchor i:
    (a^H_i, a^A_i)' ~ N(0, Sigma_a)

The AI and human loadings on their own latent scales are fixed to +1. This
avoids an otherwise redundant scale-loading parameterization. Correlations
rho_fov and rho_anchor describe the degree/direction of AI-human sharing.

The future-reader estimand still uses the HUMAN latent components and the
OBSERVED fixed AI vector. It is conditional on the selected 1,144-anchor frame.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import nucls_v23_1_joint_bayes_logit as v231

PRIMARY_READERS=v231.PRIMARY_READERS
EncodedData=v231.EncodedData
encode_observed=v231.encode_observed
logistic_normal_mean=v231.logistic_normal_mean

def build_model(data:EncodedData, *, lkj_eta:float=2.0,
                fov_sd_scale:float=1.0, anchor_sd_scale:float=1.5):
    import pymc as pm

    coords={
        "reader":data.reader_names,
        "fov":np.arange(len(data.fov_values)),
        "anchor":np.arange(len(data.anchor_ids)),
        "rf_obs":np.arange(len(data.rf_keys)),
        "ra_obs":np.arange(len(data.ra_keys)),
        "obs":np.arange(len(data.y)),
        "channel":["human","ai"],
    }

    with pm.Model(coords=coords) as model:
        # Fixed effects.
        alpha=pm.Normal("alpha",0,2.5)
        beta_E=pm.Normal("beta_E",0,1.0)
        alpha_ai=pm.Normal("alpha_ai",0,2.5)

        # Reader-side heterogeneity, matching M2.
        sd_reader=pm.HalfNormal("sd_reader",1.0)
        sd_reader_condition=pm.HalfNormal("sd_reader_condition",1.0)
        sd_reader_fov=pm.HalfNormal("sd_reader_fov",1.0)
        sd_reader_anchor=pm.HalfNormal("sd_reader_anchor",1.5)

        z_reader=pm.Normal("z_reader",0,1,dims="reader")
        z_reader_condition=pm.Normal("z_reader_condition",0,1,dims="reader")
        z_reader_fov=pm.Normal("z_reader_fov",0,1,dims="rf_obs")
        z_reader_anchor=pm.Normal("z_reader_anchor",0,1,dims="ra_obs")

        u=pm.Deterministic("u_reader",sd_reader*z_reader,dims="reader")
        b=pm.Deterministic("b_reader_condition",sd_reader_condition*z_reader_condition,dims="reader")
        h=pm.Deterministic("h_reader_fov",sd_reader_fov*z_reader_fov,dims="rf_obs")
        q=pm.Deterministic("q_reader_anchor",sd_reader_anchor*z_reader_anchor,dims="ra_obs")

        # Distinct but correlated human/AI FOV latent effects.
        chol_f,corr_f,std_f=pm.LKJCholeskyCov(
            "fov_cov",n=2,eta=lkj_eta,
            sd_dist=pm.HalfNormal.dist(fov_sd_scale,shape=2),
            compute_corr=True,
        )
        z_f=pm.Normal("z_fov_pair",0,1,dims=("fov","channel"))
        f_pair=pm.Deterministic("fov_pair",z_f @ chol_f.T,dims=("fov","channel"))
        g_h=pm.Deterministic("g_h_fov",f_pair[:,0],dims="fov")
        g_ai=pm.Deterministic("g_ai_fov",f_pair[:,1],dims="fov")
        pm.Deterministic("rho_fov",corr_f[0,1])

        # Distinct but correlated human/AI anchor latent effects.
        chol_a,corr_a,std_a=pm.LKJCholeskyCov(
            "anchor_cov",n=2,eta=lkj_eta,
            sd_dist=pm.HalfNormal.dist(anchor_sd_scale,shape=2),
            compute_corr=True,
        )
        z_a=pm.Normal("z_anchor_pair",0,1,dims=("anchor","channel"))
        a_pair=pm.Deterministic("anchor_pair",z_a @ chol_a.T,dims=("anchor","channel"))
        a_h=pm.Deterministic("a_h_anchor",a_pair[:,0],dims="anchor")
        a_ai=pm.Deterministic("a_ai_anchor",a_pair[:,1],dims="anchor")
        pm.Deterministic("rho_anchor",corr_a[0,1])

        d_h=g_h[data.anchor_f]+a_h
        d_ai=g_ai[data.anchor_f]+a_ai

        pm.Bernoulli(
            "ai_detected_obs",
            logit_p=alpha_ai+d_ai,
            observed=data.anchor_ai,
            dims="anchor",
        )

        eta=(alpha+beta_E*data.c+d_h[data.a]+u[data.r]+
             b[data.r]*data.c+h[data.rf]+q[data.ra])
        pm.Bernoulli("reader_detected",logit_p=eta,observed=data.y,dims="obs")

    return model

def posterior_estimands(idata,data:EncodedData,batch:int=100):
    """Selected-frame future-reader agreement using the HUMAN latent channel."""
    p=idata.posterior

    def flat(name):
        x=np.asarray(p[name].values)
        return x.reshape((-1,)+x.shape[2:])

    alpha=flat("alpha")
    beta=flat("beta_E")
    g_h=flat("g_h_fov")
    a_h=flat("a_h_anchor")
    sr=flat("sd_reader")
    sb=flat("sd_reader_condition")
    srf=flat("sd_reader_fov")
    sra=flat("sd_reader_anchor")

    n=len(alpha)
    out=np.empty((n,3),float)
    ai=data.anchor_ai.astype(float)

    for lo in range(0,n,batch):
        hi=min(n,lo+batch)
        base=alpha[lo:hi,None]+g_h[lo:hi][:,data.anchor_f]+a_h[lo:hi]
        sdu=np.sqrt(sr[lo:hi]**2+srf[lo:hi]**2+sra[lo:hi]**2)
        sde=np.sqrt(sr[lo:hi]**2+sb[lo:hi]**2+srf[lo:hi]**2+sra[lo:hi]**2)

        pu=logistic_normal_mean(base,sdu[:,None])
        pe=logistic_normal_mean(base+beta[lo:hi,None],sde[:,None])

        au=ai[None,:]*pu+(1-ai[None,:])*(1-pu)
        ae=ai[None,:]*pe+(1-ai[None,:])*(1-pe)
        tu=au.mean(axis=1)
        te=ae.mean(axis=1)
        out[lo:hi]=np.column_stack([tu,te,te-tu])

    return pd.DataFrame(
        out,
        columns=["theta_U_selected","theta_E_selected","delta_E_minus_U_selected"]
    )

def scalar_var_names():
    return [
        "alpha","beta_E","alpha_ai",
        "sd_reader","sd_reader_condition","sd_reader_fov","sd_reader_anchor",
        "fov_cov_stds","anchor_cov_stds","rho_fov","rho_anchor",
    ]

def self_test_numpy(seed:int=31415):
    """Tests the dual-factor construction without requiring PyMC."""
    rng=np.random.default_rng(seed)
    n=300_000
    for rho in [-.7,0,.55,.9]:
        z1=rng.normal(size=n); z2=rng.normal(size=n)
        h=z1
        a=rho*z1+np.sqrt(1-rho**2)*z2
        empirical=np.corrcoef(h,a)[0,1]
        assert abs(empirical-rho)<0.01,(rho,empirical)

    # Limiting interpretation:
    # rho≈0 -> little latent sharing; rho≈1 -> near-perfect sharing.
    print("M3_DUAL_FACTOR_NUMPY_SELF_TEST_OK")

if __name__=="__main__":
    self_test_numpy()
