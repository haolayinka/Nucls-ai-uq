#!/usr/bin/env python3
from __future__ import annotations
from repo_paths import repo_root
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import expit
from numpy.polynomial.hermite import hermgauss

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import nucls_v23_1_joint_bayes_logit as v231
import nucls_v24_m3_correlated_dual_factor as m3

READERS=list(v231.PRIMARY_READERS)
GHX,GHW=hermgauss(30); GHW=GHW/np.sqrt(np.pi)

# M3 recovery DGP. Human scales retain the validated baseline values.
ALPHA_READER=1.70
BETA_E=-0.15
SIGMA_READER=.40
SIGMA_READER_COND=.25
SIGMA_READER_FOV=.30
SIGMA_READER_ANCHOR=1.45

ALPHA_AI=-0.10
SD_FOV_H=.45
SD_FOV_AI=.55
RHO_FOV=.60
SD_ANCHOR_H=.80
SD_ANCHOR_AI=.90
RHO_ANCHOR=.60

def lnm(mu,sd):
    mu=np.asarray(mu,float); sd=np.asarray(sd,float)
    z=mu[...,None]+np.sqrt(2)*sd[...,None]*GHX
    return np.sum(expit(z)*GHW,axis=-1)

def correlated_pairs(rng,n,sd_h,sd_ai,rho):
    z1=rng.normal(size=n); z2=rng.normal(size=n)
    h=sd_h*z1
    a=sd_ai*(rho*z1+np.sqrt(1-rho**2)*z2)
    return h,a

def generate(design,seed):
    rng=np.random.default_rng(seed)
    fovs=pd.Index(sorted(design.image_id.astype(str).unique()))
    fmap={x:i for i,x in enumerate(fovs)}
    fi=design.image_id.astype(str).map(fmap).to_numpy(int)
    nf=len(fovs); n=len(design)

    gh,ga=correlated_pairs(rng,nf,SD_FOV_H,SD_FOV_AI,RHO_FOV)
    ah,aa=correlated_pairs(rng,n,SD_ANCHOR_H,SD_ANCHOR_AI,RHO_ANCHOR)
    dh=gh[fi]+ah
    dai=ga[fi]+aa
    ai=rng.binomial(1,expit(ALPHA_AI+dai))

    ur=rng.normal(0,SIGMA_READER,len(READERS))
    br=rng.normal(BETA_E,SIGMA_READER_COND,len(READERS))
    hrf=rng.normal(0,SIGMA_READER_FOV,(len(READERS),nf))
    yu=np.zeros((len(READERS),n),np.int8)
    ye=np.zeros_like(yu)
    for ri in range(len(READERS)):
        q=rng.normal(0,SIGMA_READER_ANCHOR,n)
        eta=ALPHA_READER+dh+ur[ri]+hrf[ri,fi]+q
        yu[ri]=rng.binomial(1,expit(eta))
        ye[ri]=rng.binomial(1,expit(eta+br[ri]))

    ou=np.vstack([design[f"obs_U_{r}"].astype(bool).to_numpy() for r in READERS])
    oe=np.vstack([design[f"obs_E_{r}"].astype(bool).to_numpy() for r in READERS])

    sdu=math.sqrt(SIGMA_READER**2+SIGMA_READER_FOV**2+SIGMA_READER_ANCHOR**2)
    sde=math.sqrt(sdu**2+SIGMA_READER_COND**2)
    pu=lnm(ALPHA_READER+dh,np.full(n,sdu))
    pe=lnm(ALPHA_READER+BETA_E+dh,np.full(n,sde))
    au=ai*pu+(1-ai)*(1-pu)
    ae=ai*pe+(1-ai)*(1-pe)
    truth={"U":float(au.mean()),"E":float(ae.mean()),"D":float(ae.mean()-au.mean())}

    rows=[]
    for ri,r in enumerate(READERS):
        for cond,yy,oo in [("U",yu,ou),("E",ye,oe)]:
            rows.append(pd.DataFrame({
                "anchor_id":design.anchor_id.astype(str),
                "fov_id":fi.astype(float),
                "reader":r,"condition":cond,
                "observed":oo[ri].astype(np.int8),
                "reader_detected":np.where(oo[ri],yy[ri],np.nan)
            }))
    rl=pd.concat(rows,ignore_index=True)
    ai_series=pd.Series(ai,index=design.anchor_id.astype(str))
    return rl,ai_series,truth,{
        "ai_positive":int(ai.sum()),
        "realized_fov_corr":float(np.corrcoef(gh,ga)[0,1]),
        "realized_anchor_corr":float(np.corrcoef(ah,aa)[0,1]),
    }

def summarize(draws,truth):
    mp={"theta_U_selected":"U","theta_E_selected":"E","delta_E_minus_U_selected":"D"}
    rows=[]
    for col,code in mp.items():
        x=draws[col].to_numpy(float); lo,hi=np.quantile(x,[.025,.975]); tr=truth[code]
        rows.append({"target":col,"truth":tr,"estimate":float(x.mean()),
                     "error":float(x.mean()-tr),"posterior_sd":float(x.std(ddof=1)),
                     "ci_low":float(lo),"ci_high":float(hi),
                     "interval_width":float(hi-lo),"covered":bool(lo<=tr<=hi)})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rep",type=int,required=True)
    ap.add_argument("--project-root",type=Path,default=repo_root())
    ap.add_argument("--draws",type=int,default=4000)
    ap.add_argument("--tune",type=int,default=2000)
    ap.add_argument("--chains",type=int,default=4)
    ap.add_argument("--target-accept",type=float,default=.99)
    ap.add_argument("--outdir",type=Path,default=None)
    args=ap.parse_args()

    m3.self_test_numpy()
    root=args.project_root
    dpath=root/"data"/"v24_1"/"nucls_exact_reader_fov_observation_design_v21.csv"
    if not dpath.exists():
        # backward-compatible project location
        dpath=root/"data"/"v23_1"/"nucls_exact_reader_fov_observation_design_v21.csv"
    design=pd.read_csv(dpath)
    seed=243000+args.rep
    rl,ai,truth,meta=generate(design,seed)
    enc=v231.encode_observed(rl,ai,readers=v231.PRIMARY_READERS,expected_ai_matches=None)

    audit={"rep":args.rep,"seed":seed,"n_obs":len(enc.y),"n_anchors":len(enc.anchor_ids),
           "n_fovs":len(enc.fov_values),"truth":truth,**meta}
    print("M3_RECOVERY_INPUT "+json.dumps(audit,sort_keys=True),flush=True)

    import pymc as pm, arviz as az
    model=m3.build_model(enc)
    t0=time.time()
    with model:
        idata=pm.sample(draws=args.draws,tune=args.tune,chains=args.chains,
            cores=min(args.chains,max(1,int(os.environ.get("SLURM_CPUS_PER_TASK","1")))),
            random_seed=seed+177,target_accept=args.target_accept,
            init="jitter+adapt_diag",progressbar=False,return_inferencedata=True)
    elapsed=time.time()-t0

    post=m3.posterior_estimands(idata,enc)
    summ=summarize(post,truth)
    ds=az.summary(idata,var_names=m3.scalar_var_names(),round_to=None)
    diag={"elapsed_sec":elapsed,"divergences":int(np.asarray(idata.sample_stats["diverging"]).sum()),
          "max_rhat_scalar":float(ds.r_hat.max()),"min_ess_bulk_scalar":float(ds.ess_bulk.min()),
          "min_ess_tail_scalar":float(ds.ess_tail.min())}

    outdir=args.outdir or root/"outputs"/"v24_1_m3_recovery"
    outdir.mkdir(parents=True,exist_ok=True)
    stem=f"rep{args.rep:03d}"
    summ.to_csv(outdir/f"{stem}_summary.csv",index=False)
    ds.to_csv(outdir/f"{stem}_diagnostics.csv")
    gate={**audit,**diag,"summary":summ.to_dict(orient="records"),"status":"M3_RECOVERY_COMPLETE"}
    (outdir/f"{stem}_gate.json").write_text(json.dumps(gate,indent=2))

    print("M3_RECOVERY_SUMMARY",flush=True)
    print(summ.to_string(index=False),flush=True)
    print("M3_RECOVERY_DIAGNOSTICS "+json.dumps(diag,sort_keys=True),flush=True)
    print("M3_RECOVERY_COMPLETE",flush=True)

if __name__=="__main__":
    main()
