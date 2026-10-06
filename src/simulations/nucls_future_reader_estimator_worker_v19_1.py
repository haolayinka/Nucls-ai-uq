#!/usr/bin/env python3
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import beta as beta_dist, t as t_dist
from numpy.polynomial.hermite import hermgauss

READERS=["SP.2","SP.3","JP.1","JP.2","JP.5","JP.6","SP.1","JP.3","JP.4"]
N_PAIRED=6
GH_X,GH_W=hermgauss(30)
GH_W=GH_W/np.sqrt(np.pi)

def coverage_probs(profile):
    if profile=="nucls_like":
        qu=np.array([0.85,0.93,0.99,0.98,0.99,0.99,0.90,0.90,0.90])
        qe=np.array([0.15,0.74,0.98,0.89,0.99,0.94,0.90,0.90,0.90])
    elif profile=="near_complete":
        qu=np.repeat(0.98,9); qe=np.repeat(0.98,9)
    elif profile=="severe":
        qu=np.array([0.45,0.55,0.70,0.65,0.75,0.70,0.55,0.55,0.55])
        qe=np.array([0.10,0.35,0.65,0.50,0.70,0.60,0.50,0.50,0.50])
    else:
        raise ValueError(profile)
    return qu,qe

def logistic_normal_mean(mu,sd):
    mu=np.asarray(mu,dtype=float)
    z=mu[...,None]+np.sqrt(2)*sd*GH_X
    return np.sum(expit(z)*GH_W,axis=-1)

def generate(cfg,seed):
    rng=np.random.default_rng(seed)
    n=int(cfg.n_candidates); nf=int(cfg.n_fovs); npat=int(cfg.n_patients); nr=len(READERS)
    fov_patient=np.arange(nf)%npat
    fov=rng.integers(0,nf,size=n)
    patient=fov_patient[fov]
    slide_re=rng.normal(0,float(cfg.sigma_slide),npat)
    d=slide_re[patient]+rng.normal(0,1,n)
    ai=rng.binomial(1,expit(float(cfg.alpha_ai)+float(cfg.beta_ai)*d))

    ur=rng.normal(0,float(cfg.sigma_reader),nr)
    br=rng.normal(float(cfg.gamma_eval),float(cfg.sigma_assist),nr)
    qu,qe=coverage_probs(str(cfg.coverage_profile))
    ou=np.array([rng.random(nf)<q for q in qu],dtype=bool)
    oe=np.array([rng.random(nf)<q for q in qe],dtype=bool)

    yu=np.zeros((nr,n),dtype=np.int8); ye=np.zeros((nr,n),dtype=np.int8)
    for r in range(nr):
        h=rng.normal(0,float(cfg.sigma_pair),n)
        eta=float(cfg.alpha_reader)+float(cfg.beta_reader)*d+ur[r]+h
        yu[r]=rng.binomial(1,expit(eta))
        ye[r]=rng.binomial(1,expit(eta+br[r]))

    cu=np.zeros(n,dtype=int); ce=np.zeros(n,dtype=int)
    for r in range(nr):
        cu += yu[r]*ou[r,fov]
        ce += ye[r]*oe[r,fov]

    pret=expit(float(cfg.tau0)+float(cfg.tau_d)*d)
    ru=rng.binomial(1,pret).astype(bool)
    re=rng.binomial(1,pret).astype(bool)
    k=int(cfg.selection_k)
    selected=(cu>=k)&ru&(ce>=k)&re
    idx=np.flatnonzero(selected)
    if len(idx)<50: raise RuntimeError("Too few selected anchors")

    # Exact future-reader truth under DGP.
    mu_u=float(cfg.alpha_reader)+float(cfg.beta_reader)*d
    sd_u=math.sqrt(float(cfg.sigma_reader)**2+float(cfg.sigma_pair)**2)
    qtu=logistic_normal_mean(mu_u,sd_u)
    mu_e=mu_u+float(cfg.gamma_eval)
    sd_e=math.sqrt(float(cfg.sigma_reader)**2+float(cfg.sigma_pair)**2+
                   float(cfg.sigma_assist)**2)
    qte=logistic_normal_mean(mu_e,sd_e)
    agu=ai*qtu+(1-ai)*(1-qtu)
    age=ai*qte+(1-ai)*(1-qte)

    truth={
      "theta_U_selected":float(agu[idx].mean()),
      "theta_E_selected":float(age[idx].mean()),
      "delta_selected":float(age[idx].mean()-agu[idx].mean()),
      "theta_U_population":float(agu.mean()),
      "theta_E_population":float(age.mean()),
      "delta_population":float(age.mean()-agu.mean()),
      "selection_gap_U":float(agu[idx].mean()-agu.mean()),
      "selection_gap_E":float(age[idx].mean()-age.mean()),
    }

    # Selected-frame observed agreement arrays.
    records=[]
    for r in range(nr):
        for cc,(yy,oo) in enumerate(((yu,ou),(ye,oe))):
            keep=oo[r,fov[idx]]
            ii=idx[keep]
            agree=(yy[r,ii]==ai[ii]).astype(np.int8)
            records.append({
                "reader":r,"condition":cc,"n":len(ii),
                "agreements":int(agree.sum()),
                "rate":float(agree.mean()) if len(ii) else np.nan,
            })

    # M0 also needs anchor-level observed labels by condition.
    selected_data={
      "idx":idx,"ai":ai[idx],"yu":yu[:,idx],"ye":ye[:,idx],
      "ou":ou[:,fov[idx]],"oe":oe[:,fov[idx]],
      "reader_cells":pd.DataFrame(records),
    }
    return selected_data,truth

def jeffreys_interval(s,n):
    a=s+0.5; b=n-s+0.5
    return (a/(a+b),float(beta_dist.ppf(.025,a,b)),float(beta_dist.ppf(.975,a,b)),a,b)

def method_m0(dat,rng):
    ai=dat["ai"]; m=len(ai); out={}; draws={}
    for cc,name in ((0,"U"),(1,"E")):
        yy=dat["yu"] if cc==0 else dat["ye"]
        oo=dat["ou"] if cc==0 else dat["oe"]
        nobs=oo.sum(axis=0)
        ndet=(yy*oo).sum(axis=0)
        valid=nobs>0
        consensus=(ndet>nobs/2).astype(np.int8)
        z=(consensus[valid]==ai[valid]).astype(np.int8)
        mean,lo,hi,a,b=jeffreys_interval(int(z.sum()),len(z))
        out[name]=(mean,lo,hi)
        draws[name]=rng.beta(a,b,6000)
    dd=draws["E"]-draws["U"]
    out["D"]=(float(dd.mean()),float(np.quantile(dd,.025)),float(np.quantile(dd,.975)))
    return out

def method_m1(dat,rng):
    cells=dat["reader_cells"]; out={}; draws={}
    for cc,name in ((0,"U"),(1,"E")):
        x=cells[cells.condition==cc]
        s=int(x.agreements.sum()); n=int(x.n.sum())
        mean,lo,hi,a,b=jeffreys_interval(s,n)
        out[name]=(mean,lo,hi)
        draws[name]=rng.beta(a,b,6000)
    dd=draws["E"]-draws["U"]
    out["D"]=(float(dd.mean()),float(np.quantile(dd,.025)),float(np.quantile(dd,.975)))
    return out

def method_m2(dat):
    cells=dat["reader_cells"]
    vals=[]
    for r in range(N_PAIRED):
        u=cells[(cells.reader==r)&(cells.condition==0)]
        e=cells[(cells.reader==r)&(cells.condition==1)]
        if len(u)!=1 or len(e)!=1 or int(u.iloc[0].n)==0 or int(e.iloc[0].n)==0:
            continue
        vals.append([float(u.iloc[0].rate),float(e.iloc[0].rate)])
    X=np.asarray(vals,dtype=float)
    if X.ndim!=2 or X.shape[1]!=2 or len(X)<=2:
        return None,len(X)
    R,p=X.shape
    xb=X.mean(axis=0)
    C=X-xb
    S=C.T@C
    df=R-p
    scale=S/(R*df)
    q=float(t_dist.ppf(.975,df))
    out={}
    for name,a in (
        ("U",np.array([1.0,0.0])),
        ("E",np.array([0.0,1.0])),
        ("D",np.array([-1.0,1.0])),
    ):
        mu=float(a@xb)
        se=math.sqrt(max(0.0,float(a@scale@a)))
        out[name]=(mu,mu-q*se,mu+q*se)
    return out,R

def rows_for(base,method,res,truth,paired_readers=np.nan):
    rows=[]
    for code,key in (("U","theta_U_selected"),("E","theta_E_selected"),("D","delta_selected")):
        tr=float(truth[key])
        if res is None:
            est=lo=hi=np.nan
        else:
            est,lo,hi=res[code]
        rows.append({
          **base,"method":method,"target":key,"truth":tr,
          "estimate":est,"ci_low":lo,"ci_high":hi,
          "covered":bool(lo<=tr<=hi) if np.isfinite(lo) else False,
          "interval_width":hi-lo if np.isfinite(lo) else np.nan,
          "error":est-tr if np.isfinite(est) else np.nan,
          "finite":bool(np.isfinite(est)),
          "paired_readers_used":paired_readers,
        })
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scenario-grid",type=Path,required=True)
    ap.add_argument("--scenario-id",type=int,required=True)
    ap.add_argument("--n-reps",type=int,default=100)
    ap.add_argument("--seed-base",type=int,default=180000)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()

    grid=pd.read_csv(args.scenario_grid)
    hit=grid[grid.scenario_id==args.scenario_id]
    if len(hit)!=1: raise SystemExit("Scenario lookup failed")
    cfg=hit.iloc[0]

    rows=[]
    for rep in range(args.n_reps):
        seed=args.seed_base+args.scenario_id*100000+rep
        dat,truth=generate(cfg,seed)
        base={
          "scenario_id":int(cfg.scenario_id),"scenario":str(cfg.scenario),
          "rep":rep,"seed":seed,"n_selected":len(dat["idx"]),
          "truth_theta_U_population":truth["theta_U_population"],
          "truth_theta_E_population":truth["theta_E_population"],
          "truth_delta_population":truth["delta_population"],
          "selection_gap_U":truth["selection_gap_U"],
          "selection_gap_E":truth["selection_gap_E"],
        }
        m0=method_m0(dat,np.random.default_rng(seed+91001))
        m1=method_m1(dat,np.random.default_rng(seed+91002))
        m2,nr=method_m2(dat)
        rows += rows_for(base,"M0_majority_consensus",m0,truth)
        rows += rows_for(base,"M1_pooled_reader_anchor_beta",m1,truth)
        rows += rows_for(base,"M2_reader_primary_paired_bayes_t",m2,truth,nr)

    out=pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=args.output.with_suffix(args.output.suffix+".tmp")
    out.to_csv(tmp,index=False); tmp.replace(args.output)
    print(f"V19_1_SCENARIO_COMPLETE scenario_id={args.scenario_id} rows={len(out)}")

if __name__=="__main__":
    main()
