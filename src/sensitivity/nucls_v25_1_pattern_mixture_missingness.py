#!/usr/bin/env python3
from __future__ import annotations
from repo_paths import repo_root
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit, logit

READERS=["SP.3","JP.1","JP.2","JP.5","JP.6"]
EXPECTED_ANCHORS=1144
EXPECTED_AI_POS=593
GHX,GHW=hermgauss(30); GHW=GHW/np.sqrt(np.pi)

def logistic_normal_mean(mu,sd):
    mu=np.asarray(mu,float); sd=np.asarray(sd,float)
    z=mu[...,None]+np.sqrt(2.0)*sd[...,None]*GHX
    return np.sum(expit(z)*GHW,axis=-1)

def flat(p,name):
    x=np.asarray(p[name].values)
    return x.reshape((-1,)+x.shape[2:])

def future_probs(idata, anchor_f, batch=100):
    p=idata.posterior
    alpha=flat(p,"alpha"); beta=flat(p,"beta_E")
    g=flat(p,"g_fov"); aa=flat(p,"a_anchor")
    sr=flat(p,"sd_reader"); sb=flat(p,"sd_reader_condition")
    srf=flat(p,"sd_reader_fov"); sra=flat(p,"sd_reader_anchor")
    n=len(alpha); A=len(anchor_f)
    pu=np.empty((n,A),float); pe=np.empty((n,A),float)
    for lo in range(0,n,batch):
        hi=min(n,lo+batch)
        base=alpha[lo:hi,None]+g[lo:hi][:,anchor_f]+aa[lo:hi]
        sdu=np.sqrt(sr[lo:hi]**2+srf[lo:hi]**2+sra[lo:hi]**2)
        sde=np.sqrt(sr[lo:hi]**2+sb[lo:hi]**2+srf[lo:hi]**2+sra[lo:hi]**2)
        pu[lo:hi]=logistic_normal_mean(base,sdu[:,None])
        pe[lo:hi]=logistic_normal_mean(base+beta[lo:hi,None],sde[:,None])
    return pu,pe

def shifted_mixture(p, missing_fraction, gamma):
    eps=np.finfo(float).eps
    p=np.clip(p,eps,1-eps)
    q=expit(logit(p)+gamma)
    m=np.asarray(missing_fraction,float)[None,:]
    return (1-m)*p+m*q

def summarize_draws(x):
    lo,hi=np.quantile(x,[.025,.975])
    return dict(estimate=float(x.mean()),posterior_sd=float(x.std(ddof=1)),
                ci_low=float(lo),ci_high=float(hi),interval_width=float(hi-lo))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",type=Path,default=repo_root())
    ap.add_argument("--outdir",type=Path,default=None)
    ap.add_argument("--gamma-min",type=float,default=-2.0)
    ap.add_argument("--gamma-max",type=float,default=2.0)
    ap.add_argument("--gamma-step",type=float,default=.25)
    args=ap.parse_args()
    root=args.project_root
    raw_path=root/"outputs"/"scoring_v16_1"/"restricted_recorded_reader_detection_agreement_v16.csv"
    ai_path=root/"outputs"/"scoring_v16_1"/"restricted_anchor_matches_v16.csv"
    paths={
        "M1_reader_only": root/"outputs"/"v23_1_reader_only_ablation"/"v23_1_reader_only_ablation_idata.nc",
        "M2_shared_detectability": root/"outputs"/"v23_1_realdata_primary"/"v23_1_realdata_primary_idata.nc",
    }
    summaries={
        "M1_reader_only": root/"outputs"/"v23_1_reader_only_ablation"/"v23_1_reader_only_ablation_summary.csv",
        "M2_shared_detectability": root/"outputs"/"v23_1_realdata_primary"/"v23_1_realdata_primary_summary.csv",
    }
    for p in [raw_path,ai_path,*paths.values()]:
        if not p.exists(): raise FileNotFoundError(p)

    raw=pd.read_csv(raw_path)
    d=raw[raw.reader.isin(READERS)].copy()
    d["anchor_id"]=d.anchor_id.astype(str)
    ai_tbl=pd.read_csv(ai_path)[["anchor_id","ai_detected"]].copy()
    ai_tbl["anchor_id"]=ai_tbl.anchor_id.astype(str)
    if ai_tbl.anchor_id.duplicated().any():
        if (ai_tbl.groupby("anchor_id").ai_detected.nunique(dropna=False)>1).any():
            raise RuntimeError("conflicting AI labels")
        ai_tbl=ai_tbl.drop_duplicates("anchor_id")
    ai=ai_tbl.set_index("anchor_id").ai_detected.astype(int)
    if len(ai)!=EXPECTED_ANCHORS or int(ai.sum())!=EXPECTED_AI_POS:
        raise RuntimeError("AI vector validation failed")

    anchors=sorted(d.anchor_id.unique())
    if len(anchors)!=EXPECTED_ANCHORS: raise RuntimeError("anchor count mismatch")
    # Match the exact v23.1 lexicographic mappings.
    anchor_map=(d.groupby("anchor_id").image_id.agg(["first","nunique"]).reindex(anchors))
    if (anchor_map["nunique"]!=1).any(): raise RuntimeError("anchor maps to multiple FOVs")
    fovs=sorted(d.image_id.astype(str).unique())
    fmap={x:i for i,x in enumerate(fovs)}
    anchor_f=anchor_map["first"].astype(str).map(fmap).astype(int).to_numpy()
    aivec=ai.reindex(anchors).to_numpy(int)
    if np.isnan(aivec.astype(float)).any(): raise RuntimeError("AI missing after anchor order")

    # Empirical fraction of primary readers in DidNotAnnotateFOV stratum,
    # by FOV and condition. Presence of any row identifies an observed FOV.
    rows=[]
    for f in fovs:
        for cond in ["Unbiased","Evaluation"]:
            z=d[(d.image_id.astype(str)==f)&(d.condition==cond)]
            observed_readers=set(z.reader.unique())
            m=1-len(observed_readers)/len(READERS)
            rows.append((f,cond,m,len(observed_readers)))
    mf=pd.DataFrame(rows,columns=["image_id","condition","missing_fraction","n_observed_readers"])
    mfU=anchor_map["first"].astype(str).map(mf[mf.condition=="Unbiased"].set_index("image_id").missing_fraction).to_numpy(float)
    mfE=anchor_map["first"].astype(str).map(mf[mf.condition=="Evaluation"].set_index("image_id").missing_fraction).to_numpy(float)
    if np.isnan(mfU).any() or np.isnan(mfE).any(): raise RuntimeError("missing fraction mapping failed")

    import arviz as az
    out=args.outdir or root/"outputs"/"v25_1_pattern_mixture_missingness"
    out.mkdir(parents=True,exist_ok=True)
    mf.to_csv(out/"fov_condition_missing_fraction.csv",index=False)

    gammas=np.round(np.arange(args.gamma_min,args.gamma_max+args.gamma_step/2,args.gamma_step),10)
    allrows=[]
    baseline_checks=[]

    for model,pth in paths.items():
        idata=az.from_netcdf(pth)
        pu,pe=future_probs(idata,anchor_f)
        # Baseline selected-frame agreement.
        au=aivec[None,:]*pu+(1-aivec[None,:])*(1-pu)
        ae=aivec[None,:]*pe+(1-aivec[None,:])*(1-pe)
        base=np.column_stack([au.mean(1),ae.mean(1),ae.mean(1)-au.mean(1)])
        base_means=base.mean(0)

        # Verify gamma=0 against frozen saved summary when available.
        if summaries[model].exists():
            ss=pd.read_csv(summaries[model]).set_index("target")
            saved=np.array([
                ss.loc["theta_U_selected","estimate"],
                ss.loc["theta_E_selected","estimate"],
                ss.loc["delta_E_minus_U_selected","estimate"],
            ],float)
            diff=np.abs(saved-base_means)
            baseline_checks.append({"model":model,"max_abs_gamma0_vs_saved_summary":float(diff.max())})
            if diff.max()>2e-5:
                raise RuntimeError(f"{model} baseline mismatch {diff}")

        for gamma in gammas:
            pU=shifted_mixture(pu,mfU,gamma)
            pE=shifted_mixture(pe,mfE,gamma)
            aU=aivec[None,:]*pU+(1-aivec[None,:])*(1-pU)
            aE=aivec[None,:]*pE+(1-aivec[None,:])*(1-pE)
            vals={
                "theta_U_selected":aU.mean(axis=1),
                "theta_E_selected":aE.mean(axis=1),
                "delta_E_minus_U_selected":aE.mean(axis=1)-aU.mean(axis=1),
            }
            for target,x in vals.items():
                allrows.append({
                    "model":model,
                    "gamma_mis":float(gamma),
                    "odds_ratio_missing_vs_observed":float(np.exp(gamma)),
                    "target":target,
                    **summarize_draws(x),
                })

    res=pd.DataFrame(allrows)
    res.to_csv(out/"pattern_mixture_gamma_mis_sensitivity.csv",index=False)

    # Compact range summaries: full grid and moderate |gamma|<=1.
    rr=[]
    for (model,target),z in res.groupby(["model","target"]):
        for label,zz in [("full_-2_to_2",z),("moderate_-1_to_1",z[z.gamma_mis.abs()<=1+1e-12])]:
            rr.append({
                "model":model,"target":target,"range":label,
                "estimate_min":zz.estimate.min(),"estimate_max":zz.estimate.max(),
                "estimate_span":zz.estimate.max()-zz.estimate.min(),
                "ci_envelope_low":zz.ci_low.min(),"ci_envelope_high":zz.ci_high.max(),
            })
    pd.DataFrame(rr).to_csv(out/"pattern_mixture_range_summary.csv",index=False)

    audit={
        "status":"V25_1_PATTERN_MIXTURE_MISSINGNESS_COMPLETE",
        "definition":"At each FOV/condition, future-reader detection probability is a mixture of an observed-like regime and a DidNotAnnotateFOV-like regime. The mixing weight is the empirical fraction of the five primary readers missing that FOV/condition; the missing-like log-odds are shifted by gamma_mis. The observed-data posterior is held fixed because gamma_mis is an unidentified sensitivity parameter.",
        "gamma_grid":[float(x) for x in gammas],
        "gamma0_reproduces_original_model":baseline_checks,
        "n_anchors":len(anchors),
        "n_fovs":len(fovs),
        "readers":READERS,
        "important_limitation":"This is a delta-adjusted posterior-predictive pattern-mixture sensitivity, not a missingness model identified from the observed data and not a causal model of why a reader skipped an FOV."
    }
    (out/"audit.json").write_text(json.dumps(audit,indent=2))
    print("V25_1_PATTERN_MIXTURE_MISSINGNESS_COMPLETE")
    print(json.dumps(audit,indent=2))

if __name__=="__main__":
    main()
