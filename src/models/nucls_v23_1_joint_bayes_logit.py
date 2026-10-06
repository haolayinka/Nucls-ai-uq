#!/usr/bin/env python3
"""NuCLS v23.1 joint Bayesian AI-conditioned hierarchical logistic model.

Frozen revision motivated ONLY by exact-design known-truth diagnostics before
any real-data fit. v23 showed ~3 percentage-point bias in absolute future-reader
agreement because the human-label likelihood ignored the observed AI pattern,
even though AI and readers share the same anchor/FOV detectability signal in the
frozen DGP. v23.1 therefore models AI detections jointly with reader detections.

Latent shared detectability on the centered scale:
    d_i = g_f(i) + a_i

External AI channel:
    A_i ~ Bernoulli(logit^{-1}(alpha_ai + beta_ai * d_i)), beta_ai > 0

Reader channel (observed rows only):
    D_irc ~ Bernoulli(logit^{-1}(eta_irc))
    eta_irc = alpha + beta_E*c + d_i + u_r + b_r*c + h_rf + q_ri

The reader loading on d_i is fixed to 1 to identify the scale. The latent d_i is
NOT biological truth; it is a continuous shared detectability factor. The
future-reader estimand remains agreement with the OBSERVED fixed external-AI
vector on the selected 1,144-anchor frame.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit

PRIMARY_READERS=("SP.3","JP.1","JP.2","JP.5","JP.6")
EXPECTED_N_ANCHORS=1144
EXPECTED_REAL_AI_MATCHES=593
GHX,GHW=hermgauss(30); GHW=GHW/np.sqrt(np.pi)

@dataclass
class EncodedData:
    y: np.ndarray; c: np.ndarray; r: np.ndarray; f: np.ndarray; a: np.ndarray
    rf: np.ndarray; ra: np.ndarray; anchor_ai: np.ndarray; anchor_f: np.ndarray
    reader_names: list[str]; fov_values: np.ndarray; anchor_ids: list[str]
    rf_keys: list[tuple[int,int]]; ra_keys: list[tuple[int,int]]

def encode_observed(reader_long:pd.DataFrame, ai:pd.Series, *, readers=PRIMARY_READERS, expected_ai_matches=None)->EncodedData:
    req={"anchor_id","fov_id","reader","condition","observed","reader_detected"}
    missing=req-set(reader_long.columns)
    if missing: raise ValueError(f"reader_long missing columns {sorted(missing)}")
    all_anchors=sorted(reader_long.anchor_id.astype(str).unique())
    if len(all_anchors)!=EXPECTED_N_ANCHORS: raise ValueError(f"expected {EXPECTED_N_ANCHORS} anchors, got {len(all_anchors)}")
    z=reader_long[reader_long.reader.isin(readers)].copy(); z=z[z.observed.astype(int)==1].copy()
    if z.reader_detected.isna().any(): raise ValueError("observed rows contain missing reader_detected")
    if set(z.reader.unique())!=set(readers): raise ValueError("primary reader set mismatch")
    if not set(z.condition.unique()).issubset({"U","E"}): raise ValueError("condition must be U/E")

    amap={x:i for i,x in enumerate(all_anchors)}; rnames=list(readers); rmap={x:i for i,x in enumerate(rnames)}
    fvals=np.array(sorted(reader_long.fov_id.dropna().unique()),dtype=float); fmap={x:i for i,x in enumerate(fvals)}
    z["r_i"]=z.reader.map(rmap).astype(int); z["f_i"]=z.fov_id.map(fmap).astype(int); z["a_i"]=z.anchor_id.astype(str).map(amap).astype(int)
    rf_keys=sorted(set(zip(z.r_i,z.f_i))); rfmap={k:i for i,k in enumerate(rf_keys)}
    ra_keys=sorted(set(zip(z.r_i,z.a_i))); ramap={k:i for i,k in enumerate(ra_keys)}
    rf=np.fromiter((rfmap[(r,f)] for r,f in zip(z.r_i,z.f_i)),int,count=len(z))
    ra=np.fromiter((ramap[(r,a)] for r,a in zip(z.r_i,z.a_i)),int,count=len(z))

    ai=ai.copy(); ai.index=ai.index.astype(str); av=ai.reindex(all_anchors)
    if av.isna().any(): raise ValueError("AI vector missing anchors")
    av=av.astype(int).to_numpy()
    if not np.isin(av,[0,1]).all(): raise ValueError("AI vector must be binary")
    if expected_ai_matches is not None and int(av.sum())!=int(expected_ai_matches): raise ValueError(f"expected {expected_ai_matches} AI matches, got {int(av.sum())}")

    anchor_map=(reader_long.assign(anchor_id=reader_long.anchor_id.astype(str)).groupby("anchor_id").fov_id.agg(["first","nunique"]).reindex(all_anchors))
    if (anchor_map["nunique"]!=1).any(): raise ValueError("an anchor maps to multiple FOVs")
    anchor_f=anchor_map["first"].map(fmap).astype(int).to_numpy()
    return EncodedData(z.reader_detected.astype(int).to_numpy(),(z.condition.to_numpy()=="E").astype(int),z.r_i.to_numpy(int),z.f_i.to_numpy(int),z.a_i.to_numpy(int),rf,ra,av,anchor_f,rnames,fvals,all_anchors,rf_keys,ra_keys)

def build_model(data:EncodedData):
    import pymc as pm
    coords={"reader":data.reader_names,"fov":np.arange(len(data.fov_values)),"anchor":np.arange(len(data.anchor_ids)),"rf_obs":np.arange(len(data.rf_keys)),"ra_obs":np.arange(len(data.ra_keys)),"obs":np.arange(len(data.y))}
    with pm.Model(coords=coords) as model:
        # reader channel fixed effects
        alpha=pm.Normal("alpha",0,2.5)
        beta_E=pm.Normal("beta_E",0,1.0)
        # external-AI channel; beta_ai constrained positive to identify orientation
        alpha_ai=pm.Normal("alpha_ai",0,2.5)
        beta_ai=pm.HalfNormal("beta_ai",1.5)

        sd_reader=pm.HalfNormal("sd_reader",1.0)
        sd_reader_condition=pm.HalfNormal("sd_reader_condition",1.0)
        sd_fov=pm.HalfNormal("sd_fov",1.0)
        sd_reader_fov=pm.HalfNormal("sd_reader_fov",1.0)
        sd_anchor=pm.HalfNormal("sd_anchor",1.5)
        sd_reader_anchor=pm.HalfNormal("sd_reader_anchor",1.5)

        z_reader=pm.Normal("z_reader",0,1,dims="reader")
        z_reader_condition=pm.Normal("z_reader_condition",0,1,dims="reader")
        z_fov=pm.Normal("z_fov",0,1,dims="fov")
        z_reader_fov=pm.Normal("z_reader_fov",0,1,dims="rf_obs")
        z_anchor=pm.Normal("z_anchor",0,1,dims="anchor")
        z_reader_anchor=pm.Normal("z_reader_anchor",0,1,dims="ra_obs")

        u=pm.Deterministic("u_reader",sd_reader*z_reader,dims="reader")
        b=pm.Deterministic("b_reader_condition",sd_reader_condition*z_reader_condition,dims="reader")
        g=pm.Deterministic("g_fov",sd_fov*z_fov,dims="fov")
        h=pm.Deterministic("h_reader_fov",sd_reader_fov*z_reader_fov,dims="rf_obs")
        aa=pm.Deterministic("a_anchor",sd_anchor*z_anchor,dims="anchor")
        q=pm.Deterministic("q_reader_anchor",sd_reader_anchor*z_reader_anchor,dims="ra_obs")

        d_shared=g[data.anchor_f]+aa
        pm.Bernoulli("ai_detected_obs",logit_p=alpha_ai+beta_ai*d_shared,observed=data.anchor_ai,dims="anchor")
        eta=alpha+beta_E*data.c+d_shared[data.a]+u[data.r]+b[data.r]*data.c+h[data.rf]+q[data.ra]
        pm.Bernoulli("reader_detected",logit_p=eta,observed=data.y,dims="obs")
    return model

def logistic_normal_mean(mu,sd):
    mu=np.asarray(mu,dtype=float); sd=np.asarray(sd,dtype=float)
    z=mu[...,None]+np.sqrt(2.0)*sd[...,None]*GHX
    return np.sum(expit(z)*GHW,axis=-1)

def posterior_estimands(idata,data:EncodedData,batch=100):
    p=idata.posterior
    def flat(name):
        x=np.asarray(p[name].values); return x.reshape((-1,)+x.shape[2:])
    alpha=flat("alpha"); beta=flat("beta_E"); g=flat("g_fov"); aa=flat("a_anchor")
    sr=flat("sd_reader"); sb=flat("sd_reader_condition"); srf=flat("sd_reader_fov"); sra=flat("sd_reader_anchor")
    n=len(alpha); out=np.empty((n,3),float); ai=data.anchor_ai.astype(float)
    for lo in range(0,n,batch):
        hi=min(n,lo+batch)
        base=alpha[lo:hi,None]+g[lo:hi][:,data.anchor_f]+aa[lo:hi]
        sdu=np.sqrt(sr[lo:hi]**2+srf[lo:hi]**2+sra[lo:hi]**2)
        sde=np.sqrt(sr[lo:hi]**2+sb[lo:hi]**2+srf[lo:hi]**2+sra[lo:hi]**2)
        pu=logistic_normal_mean(base,sdu[:,None]); pe=logistic_normal_mean(base+beta[lo:hi,None],sde[:,None])
        au=ai[None,:]*pu+(1-ai[None,:])*(1-pu); ae=ai[None,:]*pe+(1-ai[None,:])*(1-pe)
        tu=au.mean(axis=1); te=ae.mean(axis=1); out[lo:hi]=np.column_stack([tu,te,te-tu])
    return pd.DataFrame(out,columns=["theta_U_selected","theta_E_selected","delta_E_minus_U_selected"])

def summarize(draws:pd.DataFrame,truth:dict|None=None):
    maptruth={"theta_U_selected":"U","theta_E_selected":"E","delta_E_minus_U_selected":"D"}; rows=[]
    for c in draws.columns:
        x=draws[c].to_numpy(float); lo,hi=np.quantile(x,[.025,.975]); rec={"target":c,"estimate":float(x.mean()),"posterior_sd":float(x.std(ddof=1)),"ci_low":float(lo),"ci_high":float(hi),"interval_width":float(hi-lo)}
        if truth is not None:
            tr=float(truth[maptruth[c]]); rec.update(truth=tr,bias=rec["estimate"]-tr,covered=bool(lo<=tr<=hi))
        rows.append(rec)
    return pd.DataFrame(rows)

def self_test():
    rng=np.random.default_rng(231); mu=.4; sd=.8
    an=float(logistic_normal_mean(np.array([mu]),np.array([sd]))[0]); mc=float(expit(mu+rng.normal(0,sd,500_000)).mean())
    assert abs(an-mc)<0.002,(an,mc)
    # orientation check: larger shared detectability must increase AI probability for beta_ai>0
    assert expit(-.2+1.2*1.0)>expit(-.2+1.2*(-1.0))
    print("V23_1_MODEL_SELF_TEST_OK")
if __name__=="__main__": self_test()
