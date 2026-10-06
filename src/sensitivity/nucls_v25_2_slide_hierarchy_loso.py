#!/usr/bin/env python3
from __future__ import annotations
from repo_paths import repo_root
from dataclasses import dataclass
import argparse, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit

READERS=["SP.3","JP.1","JP.2","JP.5","JP.6"]
EXPECTED_FULL_ANCHORS=1144
EXPECTED_FULL_FOVS=52
EXPECTED_FULL_SLIDES=5
EXPECTED_FULL_OBS=10833
EXPECTED_AI_POS=593
GHX,GHW=hermgauss(30); GHW=GHW/np.sqrt(np.pi)
SCALAR_VARS=["alpha","beta_E","alpha_ai","beta_ai","sd_reader","sd_reader_condition",
             "sd_slide","sd_fov","sd_reader_fov","sd_anchor","sd_reader_anchor"]
TARGETS=["theta_U_selected","theta_E_selected","delta_E_minus_U_selected"]

@dataclass
class Enc:
    y:np.ndarray; c:np.ndarray; r:np.ndarray; s:np.ndarray; f:np.ndarray; a:np.ndarray
    rf:np.ndarray; ra:np.ndarray
    anchor_ai:np.ndarray; anchor_s:np.ndarray; anchor_f:np.ndarray
    reader_names:list; slide_values:list; fov_values:list; anchor_ids:list
    rf_keys:list; ra_keys:list

def logistic_normal_mean(mu,sd):
    mu=np.asarray(mu,float); sd=np.asarray(sd,float)
    z=mu[...,None]+np.sqrt(2.0)*sd[...,None]*GHX
    return np.sum(expit(z)*GHW,axis=-1)

def encode(d:pd.DataFrame, ai:pd.Series)->Enc:
    z=d.copy()
    z["anchor_id"]=z.anchor_id.astype(str)
    anchors=sorted(z.anchor_id.unique())
    rmap={r:i for i,r in enumerate(READERS)}
    slides=sorted(z.slide.astype(str).unique()); smap={s:i for i,s in enumerate(slides)}
    fovs=sorted(z.image_id.astype(str).unique()); fmap={f:i for i,f in enumerate(fovs)}
    amap={a:i for i,a in enumerate(anchors)}
    if set(z.reader.unique())!=set(READERS): raise RuntimeError("reader set mismatch in subset")
    if z.duplicated(["anchor_id","reader","condition"]).any(): raise RuntimeError("duplicate observed rows")

    z["r_i"]=z.reader.map(rmap).astype(int)
    z["s_i"]=z.slide.astype(str).map(smap).astype(int)
    z["f_i"]=z.image_id.astype(str).map(fmap).astype(int)
    z["a_i"]=z.anchor_id.map(amap).astype(int)
    z["c_i"]=(z.condition=="Evaluation").astype(int)

    rf_keys=sorted(set(zip(z.r_i,z.f_i))); rfmap={k:i for i,k in enumerate(rf_keys)}
    ra_keys=sorted(set(zip(z.r_i,z.a_i))); ramap={k:i for i,k in enumerate(ra_keys)}
    rf=np.fromiter((rfmap[(r,f)] for r,f in zip(z.r_i,z.f_i)),int,count=len(z))
    ra=np.fromiter((ramap[(r,a)] for r,a in zip(z.r_i,z.a_i)),int,count=len(z))

    h=(z.groupby("anchor_id")
       .agg(slide=("slide","first"),nslide=("slide","nunique"),
            image_id=("image_id","first"),nfov=("image_id","nunique"))
       .reindex(anchors))
    if (h[["nslide","nfov"]]!=1).any().any(): raise RuntimeError("anchor hierarchy not unique")
    anchor_s=h.slide.astype(str).map(smap).astype(int).to_numpy()
    anchor_f=h.image_id.astype(str).map(fmap).astype(int).to_numpy()

    av=ai.copy(); av.index=av.index.astype(str); av=av.reindex(anchors)
    if av.isna().any(): raise RuntimeError("AI missing for subset anchors")
    av=av.astype(int).to_numpy()
    return Enc(z.reader_detected.astype(int).to_numpy(),z.c_i.to_numpy(int),z.r_i.to_numpy(int),
               z.s_i.to_numpy(int),z.f_i.to_numpy(int),z.a_i.to_numpy(int),rf,ra,av,anchor_s,anchor_f,
               READERS,slides,fovs,anchors,rf_keys,ra_keys)

def build_model(data:Enc):
    import pymc as pm
    coords={"reader":data.reader_names,"slide":np.arange(len(data.slide_values)),
            "fov":np.arange(len(data.fov_values)),"anchor":np.arange(len(data.anchor_ids)),
            "rf_obs":np.arange(len(data.rf_keys)),"ra_obs":np.arange(len(data.ra_keys)),
            "obs":np.arange(len(data.y))}
    with pm.Model(coords=coords) as model:
        alpha=pm.Normal("alpha",0,2.5)
        beta_E=pm.Normal("beta_E",0,1.0)
        alpha_ai=pm.Normal("alpha_ai",0,2.5)
        beta_ai=pm.HalfNormal("beta_ai",1.5)
        sd_reader=pm.HalfNormal("sd_reader",1.0)
        sd_reader_condition=pm.HalfNormal("sd_reader_condition",1.0)
        sd_slide=pm.HalfNormal("sd_slide",0.5)
        sd_fov=pm.HalfNormal("sd_fov",1.0)
        sd_reader_fov=pm.HalfNormal("sd_reader_fov",1.0)
        sd_anchor=pm.HalfNormal("sd_anchor",1.5)
        sd_reader_anchor=pm.HalfNormal("sd_reader_anchor",1.5)
        z_reader=pm.Normal("z_reader",0,1,dims="reader")
        z_reader_condition=pm.Normal("z_reader_condition",0,1,dims="reader")
        z_slide=pm.Normal("z_slide",0,1,dims="slide")
        z_fov=pm.Normal("z_fov",0,1,dims="fov")
        z_reader_fov=pm.Normal("z_reader_fov",0,1,dims="rf_obs")
        z_anchor=pm.Normal("z_anchor",0,1,dims="anchor")
        z_reader_anchor=pm.Normal("z_reader_anchor",0,1,dims="ra_obs")
        u=pm.Deterministic("u_reader",sd_reader*z_reader,dims="reader")
        b=pm.Deterministic("b_reader_condition",sd_reader_condition*z_reader_condition,dims="reader")
        ss=pm.Deterministic("slide_effect",sd_slide*z_slide,dims="slide")
        g=pm.Deterministic("g_fov",sd_fov*z_fov,dims="fov")
        h=pm.Deterministic("h_reader_fov",sd_reader_fov*z_reader_fov,dims="rf_obs")
        aa=pm.Deterministic("a_anchor",sd_anchor*z_anchor,dims="anchor")
        q=pm.Deterministic("q_reader_anchor",sd_reader_anchor*z_reader_anchor,dims="ra_obs")
        shared=ss[data.anchor_s]+g[data.anchor_f]+aa
        pm.Bernoulli("ai_detected_obs",logit_p=alpha_ai+beta_ai*shared,
                     observed=data.anchor_ai,dims="anchor")
        eta=alpha+beta_E*data.c+shared[data.a]+u[data.r]+b[data.r]*data.c+h[data.rf]+q[data.ra]
        pm.Bernoulli("reader_detected",logit_p=eta,observed=data.y,dims="obs")
    return model

def posterior_estimands(idata,data:Enc,batch=100):
    p=idata.posterior
    def flat(name):
        x=np.asarray(p[name].values); return x.reshape((-1,)+x.shape[2:])
    alpha=flat("alpha"); beta=flat("beta_E"); ss=flat("slide_effect")
    g=flat("g_fov"); aa=flat("a_anchor")
    sr=flat("sd_reader"); sb=flat("sd_reader_condition")
    srf=flat("sd_reader_fov"); sra=flat("sd_reader_anchor")
    n=len(alpha); out=np.empty((n,3),float); ai=data.anchor_ai.astype(float)
    for lo in range(0,n,batch):
        hi=min(n,lo+batch)
        base=alpha[lo:hi,None]+ss[lo:hi][:,data.anchor_s]+g[lo:hi][:,data.anchor_f]+aa[lo:hi]
        sdu=np.sqrt(sr[lo:hi]**2+srf[lo:hi]**2+sra[lo:hi]**2)
        sde=np.sqrt(sr[lo:hi]**2+sb[lo:hi]**2+srf[lo:hi]**2+sra[lo:hi]**2)
        pu=logistic_normal_mean(base,sdu[:,None])
        pe=logistic_normal_mean(base+beta[lo:hi,None],sde[:,None])
        au=ai[None,:]*pu+(1-ai[None,:])*(1-pu)
        ae=ai[None,:]*pe+(1-ai[None,:])*(1-pe)
        tu=au.mean(1); te=ae.mean(1)
        out[lo:hi]=np.column_stack([tu,te,te-tu])
    return pd.DataFrame(out,columns=TARGETS)

def summarize(post):
    rows=[]
    for c in TARGETS:
        x=post[c].to_numpy(float); lo,hi=np.quantile(x,[.025,.975])
        rows.append({"target":c,"estimate":float(x.mean()),"posterior_sd":float(x.std(ddof=1)),
                     "ci_low":float(lo),"ci_high":float(hi),"interval_width":float(hi-lo),
                     "prob_gt_zero":float((x>0).mean()) if c.startswith("delta") else np.nan})
    return pd.DataFrame(rows)

def atomic_netcdf(idata,path):
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.unlink(missing_ok=True)
    idata.to_netcdf(tmp); os.replace(tmp,path)

def self_test():
    rng=np.random.default_rng(252)
    q=float(logistic_normal_mean(np.array([.4]),np.array([.8]))[0])
    mc=float(expit(.4+rng.normal(0,.8,300_000)).mean())
    assert abs(q-mc)<0.003,(q,mc)
    print("V25_2_SLIDE_HIERARCHY_SELF_TEST_OK")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",type=Path,default=repo_root())
    ap.add_argument("--task-index",type=int,required=True)
    ap.add_argument("--draws",type=int,default=6000)
    ap.add_argument("--tune",type=int,default=3000)
    ap.add_argument("--chains",type=int,default=4)
    ap.add_argument("--target-accept",type=float,default=.995)
    ap.add_argument("--seed-base",type=int,default=252000)
    ap.add_argument("--outroot",type=Path,default=None)
    args=ap.parse_args()
    self_test()

    root=args.project_root
    raw_path=root/"outputs"/"scoring_v16_1"/"restricted_recorded_reader_detection_agreement_v16.csv"
    ai_path=root/"outputs"/"scoring_v16_1"/"restricted_anchor_matches_v16.csv"
    for p in [raw_path,ai_path]:
        if not p.exists(): raise FileNotFoundError(p)
    raw=pd.read_csv(raw_path)
    d0=raw[raw.reader.isin(READERS)].copy(); d0["anchor_id"]=d0.anchor_id.astype(str)
    if len(d0)!=EXPECTED_FULL_OBS or d0.anchor_id.nunique()!=EXPECTED_FULL_ANCHORS:
        raise RuntimeError("full source table count mismatch")
    if d0.image_id.nunique()!=EXPECTED_FULL_FOVS or d0.slide.nunique()!=EXPECTED_FULL_SLIDES:
        raise RuntimeError("full hierarchy count mismatch")
    slides=sorted(d0.slide.astype(str).unique())
    if args.task_index<0 or args.task_index>5: raise RuntimeError("task-index must be 0..5")
    held_out=None if args.task_index==0 else slides[args.task_index-1]

    a=pd.read_csv(ai_path)[["anchor_id","ai_detected"]].copy()
    a["anchor_id"]=a.anchor_id.astype(str)
    if a.anchor_id.duplicated().any():
        if (a.groupby("anchor_id").ai_detected.nunique(dropna=False)>1).any():
            raise RuntimeError("conflicting AI")
        a=a.drop_duplicates("anchor_id")
    ai=a.set_index("anchor_id").ai_detected.astype(int)
    if len(ai)!=EXPECTED_FULL_ANCHORS or int(ai.sum())!=EXPECTED_AI_POS:
        raise RuntimeError("AI validation failed")

    d=d0 if held_out is None else d0[d0.slide.astype(str)!=held_out].copy()
    enc=encode(d,ai)
    audit={"status":"V25_2_SLIDE_HIERARCHY_INPUT_OK","task_index":args.task_index,
           "held_out_slide":held_out,
           "estimand_frame":"all 1144 frozen anchors" if held_out is None else "remaining selected-frame anchors after excluding one slide",
           "n_rows":len(d),"n_anchors":d.anchor_id.nunique(),"n_fovs":d.image_id.nunique(),
           "n_slides":d.slide.nunique(),"slides":sorted(d.slide.astype(str).unique()),
           "ai_positive":int(ai.reindex(enc.anchor_ids).sum()),"slide_prior":"HalfNormal(0.5)",
           "interpretation":"slide effect weakly identified with only five slides; LOSO is an instability diagnostic, not external validation."}
    print("V25_2_SLIDE_HIERARCHY_INPUT_AUDIT "+json.dumps(audit,sort_keys=True),flush=True)

    import pymc as pm, arviz as az
    model=build_model(enc); seed=args.seed_base+args.task_index
    t0=time.time()
    with model:
        idata=pm.sample(draws=args.draws,tune=args.tune,chains=args.chains,
                        cores=min(args.chains,max(1,int(os.environ.get("SLURM_CPUS_PER_TASK","1")))),
                        random_seed=seed,target_accept=args.target_accept,
                        init="jitter+adapt_diag",progressbar=False,return_inferencedata=True)
    elapsed=time.time()-t0
    post=posterior_estimands(idata,enc); summ=summarize(post)
    ds=az.summary(idata,var_names=SCALAR_VARS,round_to=None)
    div=int(np.asarray(idata.sample_stats["diverging"]).sum())
    diag={"elapsed_sec":elapsed,"divergences":div,"max_rhat_scalar":float(ds.r_hat.max()),
          "min_ess_bulk_scalar":float(ds.ess_bulk.min()),"min_ess_tail_scalar":float(ds.ess_tail.min())}
    gate_ok=bool(div==0 and diag["max_rhat_scalar"]<=1.01 and
                 diag["min_ess_bulk_scalar"]>=400 and diag["min_ess_tail_scalar"]>=400)

    tag="full" if held_out is None else f"drop_{args.task_index}_{held_out.replace('/','_')}"
    outroot=args.outroot or root/"outputs"/"v25_2_slide_hierarchy_loso"
    out=outroot/tag; out.mkdir(parents=True,exist_ok=True)
    post.to_csv(out/"estimand_draws.csv",index=False); summ.to_csv(out/"summary.csv",index=False)
    ds.to_csv(out/"scalar_diagnostics.csv")
    gate={**audit,**diag,"computational_ok":gate_ok,
          "sampler":{"draws":args.draws,"tune":args.tune,"chains":args.chains,
                     "target_accept":args.target_accept,"seed":seed},
          "posterior_summary":summ.to_dict(orient="records"),
          "status":"V25_2_SLIDE_HIERARCHY_COMPLETE"}
    (out/"gate.json").write_text(json.dumps(gate,indent=2)); atomic_netcdf(idata,out/"idata.nc")
    print("V25_2_SLIDE_HIERARCHY_SUMMARY"); print(summ.to_string(index=False))
    print("V25_2_SLIDE_HIERARCHY_GATE "+json.dumps({k:gate[k] for k in
          ["task_index","held_out_slide","n_rows","n_anchors","n_fovs","n_slides",
           "divergences","max_rhat_scalar","min_ess_bulk_scalar","min_ess_tail_scalar",
           "computational_ok"]},sort_keys=True))
    print("V25_2_SLIDE_HIERARCHY_COMPLETE")

if __name__=="__main__":
    main()
