#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nucls_v23_1_joint_bayes_logit as v231

READERS = list(v231.PRIMARY_READERS)
N_ANCHORS = 1144
N_FOLDS = 5
AI_POS = 593
SCALAR_JOINT = [
    "alpha","beta_E","alpha_ai","beta_ai","sd_reader",
    "sd_reader_condition","sd_fov","sd_reader_fov",
    "sd_anchor","sd_reader_anchor",
]
SCALAR_READER = [
    "alpha","beta_E","sd_reader","sd_reader_condition",
    "sd_fov","sd_reader_fov","sd_anchor","sd_reader_anchor",
]

def flatten(idata, name):
    x=np.asarray(idata.posterior[name].values)
    return x.reshape((-1,)+x.shape[2:])

def metrics(y,p):
    y=np.asarray(y,int); p=np.asarray(p,float)
    eps=1e-12
    pc=np.clip(p,eps,1-eps)
    return {
        "n":int(len(y)),
        "prevalence":float(y.mean()),
        "brier":float(np.mean((p-y)**2)),
        "log_loss":float(-np.mean(y*np.log(pc)+(1-y)*np.log(1-pc))),
        "auroc":float(roc_auc_score(y,p)),
        "auprc":float(average_precision_score(y,p)),
    }

def make_fold_map(raw, ai):
    # Anchor -> one FOV, then deterministic within-(FOV, AI-status) balancing.
    x=(raw.assign(anchor_id=raw.anchor_id.astype(str))
          .groupby("anchor_id")
          .agg(image_id=("image_id","first"), fov_n=("image_id","nunique"))
          .reset_index())
    if (x.fov_n!=1).any():
        raise RuntimeError("Anchor maps to multiple FOVs")
    x["ai_detected"]=x.anchor_id.map(ai.astype(int))
    if x.ai_detected.isna().any():
        raise RuntimeError("Missing AI value in fold map")
    rng=np.random.default_rng(231005)
    fold=np.empty(len(x),int)
    for _,idx in x.groupby(["image_id","ai_detected"],sort=True).groups.items():
        idx=np.array(list(idx),int)
        idx=idx[rng.permutation(len(idx))]
        fold[idx]=np.arange(len(idx))%N_FOLDS
    return pd.Series(fold,index=x.anchor_id,name="fold"),x

def make_reader_long(d, fov_map, heldout_anchors=None):
    heldout_anchors=set() if heldout_anchors is None else set(heldout_anchors)
    return pd.DataFrame({
        "anchor_id":d.anchor_id.astype(str),
        "fov_id":d.image_id.astype(str).map(fov_map).astype(float),
        "reader":d.reader.astype(str),
        "condition":d.condition.map({"Unbiased":"U","Evaluation":"E"}),
        "observed":(~d.anchor_id.astype(str).isin(heldout_anchors)).astype(np.int8),
        "reader_detected":d.reader_detected.astype(float),
    })

def build_reader_only(data):
    import pymc as pm
    coords={
        "reader":data.reader_names,
        "fov":np.arange(len(data.fov_values)),
        "anchor":np.arange(len(data.anchor_ids)),
        "rf_obs":np.arange(len(data.rf_keys)),
        "ra_obs":np.arange(len(data.ra_keys)),
        "obs":np.arange(len(data.y)),
    }
    with pm.Model(coords=coords) as model:
        alpha=pm.Normal("alpha",0,2.5)
        beta_E=pm.Normal("beta_E",0,1.0)
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
        eta=alpha+beta_E*data.c+d_shared[data.a]+u[data.r]+b[data.r]*data.c+h[data.rf]+q[data.ra]
        pm.Bernoulli("reader_detected",logit_p=eta,observed=data.y,dims="obs")
    return model

def build_ai_masked(data, ai_train_mask):
    import pymc as pm
    ai_idx=np.flatnonzero(np.asarray(ai_train_mask,bool))
    coords={
        "reader":data.reader_names,
        "fov":np.arange(len(data.fov_values)),
        "anchor":np.arange(len(data.anchor_ids)),
        "rf_obs":np.arange(len(data.rf_keys)),
        "ra_obs":np.arange(len(data.ra_keys)),
        "obs":np.arange(len(data.y)),
        "ai_train_obs":np.arange(len(ai_idx)),
    }
    with pm.Model(coords=coords) as model:
        alpha=pm.Normal("alpha",0,2.5)
        beta_E=pm.Normal("beta_E",0,1.0)
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
        pm.Bernoulli(
            "ai_detected_obs",
            logit_p=alpha_ai+beta_ai*d_shared[ai_idx],
            observed=data.anchor_ai[ai_idx],
            dims="ai_train_obs",
        )
        eta=alpha+beta_E*data.c+d_shared[data.a]+u[data.r]+b[data.r]*data.c+h[data.rf]+q[data.ra]
        pm.Bernoulli("reader_detected",logit_p=eta,observed=data.y,dims="obs")
    return model

def predict_heldout_humans(idata, enc_train, heldout_df, fov_map):
    # Entire anchors were withheld from the reader likelihood.
    # Therefore reader×anchor q is a NEW random effect and is integrated out.
    rmap={r:i for i,r in enumerate(enc_train.reader_names)}
    amap={a:i for i,a in enumerate(enc_train.anchor_ids)}
    rfmap={k:i for i,k in enumerate(enc_train.rf_keys)}

    hd=heldout_df.copy()
    hd["r_i"]=hd.reader.map(rmap).astype(int)
    hd["f_i"]=hd.image_id.astype(str).map(fov_map).astype(int)
    hd["a_i"]=hd.anchor_id.astype(str).map(amap).astype(int)
    hd["c_i"]=(hd.condition=="Evaluation").astype(int)
    rf_idx=np.array([rfmap.get((r,f),-1) for r,f in zip(hd.r_i,hd.f_i)],int)

    alpha=flatten(idata,"alpha")
    beta=flatten(idata,"beta_E")
    g=flatten(idata,"g_fov")
    aa=flatten(idata,"a_anchor")
    u=flatten(idata,"u_reader")
    b=flatten(idata,"b_reader_condition")
    h=flatten(idata,"h_reader_fov")
    sra=flatten(idata,"sd_reader_anchor")
    srf=flatten(idata,"sd_reader_fov")

    nobs=len(hd); acc=np.zeros(nobs,float)
    batch=100
    for lo in range(0,len(alpha),batch):
        hi=min(len(alpha),lo+batch)
        mu=(alpha[lo:hi,None]
            + beta[lo:hi,None]*hd.c_i.to_numpy()[None,:]
            + g[lo:hi][:,hd.f_i.to_numpy()]
            + aa[lo:hi][:,hd.a_i.to_numpy()]
            + u[lo:hi][:,hd.r_i.to_numpy()]
            + b[lo:hi][:,hd.r_i.to_numpy()]*hd.c_i.to_numpy()[None,:])
        # Use learned reader×FOV effect when present. If the whole reader×FOV
        # combination is absent from training, integrate it out as new.
        missing_rf=(rf_idx<0)
        if (~missing_rf).any():
            mu[:,~missing_rf]+=h[lo:hi][:,rf_idx[~missing_rf]]
        sd=np.broadcast_to(sra[lo:hi,None],mu.shape).copy()
        if missing_rf.any():
            sd[:,missing_rf]=np.sqrt(
                sra[lo:hi,None]**2+srf[lo:hi,None]**2
            )
        prob=v231.logistic_normal_mean(mu,sd)
        acc+=prob.sum(axis=0)
    pred=acc/len(alpha)
    out=hd[["anchor_id","image_id","slide","patient_id","condition","reader","reader_detected"]].copy()
    out["pred"]=pred
    out["rf_seen_in_training"]=(rf_idx>=0)
    return out

def predict_heldout_ai(idata, enc, heldout_anchor_ids):
    amap={a:i for i,a in enumerate(enc.anchor_ids)}
    idx=np.array([amap[a] for a in heldout_anchor_ids],int)
    alpha_ai=flatten(idata,"alpha_ai")
    beta_ai=flatten(idata,"beta_ai")
    g=flatten(idata,"g_fov")
    aa=flatten(idata,"a_anchor")
    acc=np.zeros(len(idx),float)
    batch=100
    for lo in range(0,len(alpha_ai),batch):
        hi=min(len(alpha_ai),lo+batch)
        d=g[lo:hi][:,enc.anchor_f[idx]]+aa[lo:hi][:,idx]
        acc+=expit(alpha_ai[lo:hi,None]+beta_ai[lo:hi,None]*d).sum(axis=0)
    return acc/len(alpha_ai)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task",type=int,required=True)
    ap.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument("--draws",type=int,default=4000)
    ap.add_argument("--tune",type=int,default=2000)
    ap.add_argument("--chains",type=int,default=4)
    ap.add_argument("--target-accept",type=float,default=.99)
    ap.add_argument("--outdir",type=Path,default=None)
    args=ap.parse_args()

    if not (0<=args.task<15):
        raise ValueError("task must be 0..14")
    if args.task<5:
        mode="human_joint"; fold=args.task
    elif args.task<10:
        mode="human_reader_only"; fold=args.task-5
    else:
        mode="ai_holdout"; fold=args.task-10

    v231.self_test()
    root=args.project_root
    outdir=args.outdir or root/"outputs"/"v23_1_shared_detectability_cv"
    outdir.mkdir(parents=True,exist_ok=True)

    raw=pd.read_csv(root/"outputs"/"scoring_v16_1"/"restricted_recorded_reader_detection_agreement_v16.csv")
    ai_tbl=pd.read_csv(root/"outputs"/"scoring_v16_1"/"restricted_anchor_matches_v16.csv")
    d=raw.loc[raw.reader.isin(READERS)].copy()
    if len(d)!=10833 or d.anchor_id.nunique()!=N_ANCHORS or d.image_id.nunique()!=52:
        raise RuntimeError("Primary five-reader input counts changed")

    a=ai_tbl[["anchor_id","ai_detected"]].copy()
    a["anchor_id"]=a.anchor_id.astype(str)
    if a.anchor_id.duplicated().any():
        chk=a.groupby("anchor_id").ai_detected.nunique(dropna=False)
        if (chk>1).any(): raise RuntimeError("Conflicting AI values")
        a=a.drop_duplicates("anchor_id")
    ai=pd.Series(a.ai_detected.astype(int).to_numpy(),index=a.anchor_id)
    if len(ai)!=N_ANCHORS or int(ai.sum())!=AI_POS:
        raise RuntimeError("AI vector count changed")

    fold_map,anchor_meta=make_fold_map(d,ai)
    heldout_ids=sorted(fold_map.index[fold_map==fold].tolist())
    train_ids=set(fold_map.index)-set(heldout_ids)

    fov_values=sorted(d.image_id.astype(str).unique())
    fov_map={x:i for i,x in enumerate(fov_values)}

    if mode.startswith("human_"):
        rl=make_reader_long(d,fov_map,heldout_ids)
    else:
        rl=make_reader_long(d,fov_map,None)

    enc=v231.encode_observed(
        rl,ai,readers=v231.PRIMARY_READERS,expected_ai_matches=AI_POS
    )

    if mode=="human_joint":
        model=v231.build_model(enc)
        scalars=SCALAR_JOINT
    elif mode=="human_reader_only":
        model=build_reader_only(enc)
        scalars=SCALAR_READER
    else:
        ai_train_mask=np.array([a not in set(heldout_ids) for a in enc.anchor_ids],bool)
        model=build_ai_masked(enc,ai_train_mask)
        scalars=SCALAR_JOINT

    seed=231500+args.task
    import pymc as pm, arviz as az
    t0=time.time()
    with model:
        idata=pm.sample(
            draws=args.draws,tune=args.tune,chains=args.chains,
            cores=min(args.chains,max(1,int(os.environ.get("SLURM_CPUS_PER_TASK","1")))),
            random_seed=seed,target_accept=args.target_accept,
            init="jitter+adapt_diag",progressbar=False,return_inferencedata=True
        )
    elapsed=time.time()-t0

    if mode.startswith("human_"):
        ho=d[d.anchor_id.astype(str).isin(heldout_ids)].copy()
        pred=predict_heldout_humans(idata,enc,ho,fov_map)
        met=metrics(pred.reader_detected.astype(int),pred.pred)
        met["n_anchors"]=int(pred.anchor_id.nunique())
        met["rf_unseen_rows"]=int((~pred.rf_seen_in_training).sum())
        pred["fold"]=fold
        pred["model"]="joint" if mode=="human_joint" else "reader_only"
        pred.to_csv(outdir/f"{mode}_fold{fold}_predictions.csv",index=False)
    else:
        predv=predict_heldout_ai(idata,enc,heldout_ids)
        y=ai.reindex(heldout_ids).astype(int).to_numpy()
        met=metrics(y,predv)
        train_prev=float(ai.reindex(sorted(train_ids)).mean())
        met["prevalence_baseline_brier"]=float(np.mean((train_prev-y)**2))
        pc=np.clip(train_prev,1e-12,1-1e-12)
        met["prevalence_baseline_log_loss"]=float(-np.mean(y*np.log(pc)+(1-y)*np.log(1-pc)))
        pred=pd.DataFrame({
            "anchor_id":heldout_ids,
            "ai_detected":y,
            "pred":predv,
            "fold":fold,
            "model":"human_to_ai_joint",
        })
        pred.to_csv(outdir/f"{mode}_fold{fold}_predictions.csv",index=False)

    ds=az.summary(idata,var_names=scalars,round_to=None)
    ds.to_csv(outdir/f"{mode}_fold{fold}_mcmc_scalar_diagnostics.csv")
    diagnostics={
        "mode":mode,"fold":fold,"seed":seed,
        "n_heldout_anchors":len(heldout_ids),
        "elapsed_sec":elapsed,
        "divergences":int(np.asarray(idata.sample_stats["diverging"]).sum()),
        "max_rhat_scalar":float(ds.r_hat.max()),
        "min_ess_bulk_scalar":float(ds.ess_bulk.min()),
        "min_ess_tail_scalar":float(ds.ess_tail.min()),
        "metrics":met,
    }
    (outdir/f"{mode}_fold{fold}_result.json").write_text(json.dumps(diagnostics,indent=2))
    print("V23_1_SHARED_CV_RESULT "+json.dumps(diagnostics,sort_keys=True),flush=True)
    print("V23_1_SHARED_CV_TASK_COMPLETE",flush=True)

if __name__=="__main__":
    main()
