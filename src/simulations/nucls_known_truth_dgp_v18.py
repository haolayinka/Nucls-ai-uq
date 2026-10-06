#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

READER_NAMES = ["SP.2","SP.3","JP.1","JP.2","JP.5","JP.6","SP.1","JP.3","JP.4"]
PAIRED_READERS = 6

def expit(x):
    x=np.asarray(x,dtype=float)
    out=np.empty_like(x)
    pos=x>=0
    out[pos]=1/(1+np.exp(-x[pos]))
    z=np.exp(x[~pos])
    out[~pos]=z/(1+z)
    return out

def coverage_probs(profile):
    if profile=="nucls_like":
        # Approximate the strong reader/condition coverage heterogeneity found in NuCLS.
        q_u=np.array([0.85,0.93,0.99,0.98,0.99,0.99,0.90,0.90,0.90])
        q_e=np.array([0.15,0.74,0.98,0.89,0.99,0.94,0.90,0.90,0.90])
    elif profile=="near_complete":
        q_u=np.repeat(0.98,9); q_e=np.repeat(0.98,9)
    elif profile=="severe":
        q_u=np.array([0.45,0.55,0.70,0.65,0.75,0.70,0.55,0.55,0.55])
        q_e=np.array([0.10,0.35,0.65,0.50,0.70,0.60,0.50,0.50,0.50])
    else:
        raise ValueError(f"Unknown coverage profile: {profile}")
    return q_u,q_e

def future_reader_truth(rng, d, ai, idx, cfg, n_new_readers):
    if len(idx)==0:
        return np.nan,np.nan
    agree_u=0
    agree_e=0
    total=0
    for _ in range(n_new_readers):
        u=rng.normal(0,float(cfg.sigma_reader))
        b=rng.normal(float(cfg.gamma_eval),float(cfg.sigma_assist))
        h=rng.normal(0,float(cfg.sigma_pair),len(idx))
        eta=(float(cfg.alpha_reader)+float(cfg.beta_reader)*d[idx]+u+h)
        y_u=rng.binomial(1,expit(eta))
        y_e=rng.binomial(1,expit(eta+b))
        agree_u += int(np.sum(y_u==ai[idx]))
        agree_e += int(np.sum(y_e==ai[idx]))
        total += len(idx)
    return agree_u/total, agree_e/total

def simulate_once(cfg, seed, n_new_readers=100):
    rng=np.random.default_rng(seed)
    n=int(cfg.n_candidates); nf=int(cfg.n_fovs); npat=int(cfg.n_patients)
    nr=len(READER_NAMES)

    # FOVs are assigned across patients/slides, then candidate anchors across FOVs.
    fov_patient=np.arange(nf)%npat
    fov=rng.integers(0,nf,size=n)
    patient=fov_patient[fov]

    slide_re=rng.normal(0,float(cfg.sigma_slide),npat)
    d=slide_re[patient] + rng.normal(0,1,n)

    p_ai=expit(float(cfg.alpha_ai)+float(cfg.beta_ai)*d)
    ai=rng.binomial(1,p_ai)

    reader_intercept=rng.normal(0,float(cfg.sigma_reader),nr)
    reader_assist=rng.normal(float(cfg.gamma_eval),float(cfg.sigma_assist),nr)

    q_u,q_e=coverage_probs(str(cfg.coverage_profile))
    obs_u=np.array([rng.random(nf)<q for q in q_u],dtype=bool)
    obs_e=np.array([rng.random(nf)<q for q in q_e],dtype=bool)

    y_u=np.zeros((nr,n),dtype=np.int8)
    y_e=np.zeros((nr,n),dtype=np.int8)
    for r in range(nr):
        pair_re=rng.normal(0,float(cfg.sigma_pair),n)
        eta=(float(cfg.alpha_reader)+float(cfg.beta_reader)*d+
             reader_intercept[r]+pair_re)
        y_u[r]=rng.binomial(1,expit(eta))
        y_e[r]=rng.binomial(1,expit(eta+reader_assist[r]))

    count_u=np.zeros(n,dtype=int)
    count_e=np.zeros(n,dtype=int)
    for r in range(nr):
        count_u += y_u[r]*obs_u[r,fov]
        count_e += y_e[r]*obs_e[r,fov]

    # Approximate the additional source-derived inferred-nucleus retention hurdle.
    p_ret=expit(float(cfg.tau0)+float(cfg.tau_d)*d)
    ret_u=rng.binomial(1,p_ret).astype(bool)
    ret_e=rng.binomial(1,p_ret).astype(bool)

    k=int(cfg.selection_k)
    sel_u=(count_u>=k)&ret_u
    sel_e=(count_e>=k)&ret_e
    selected=sel_u&sel_e
    sidx=np.flatnonzero(selected)

    paired_u=[]; paired_e=[]; paired_ai=[]
    coverage=[]
    per_reader_diff=[]
    for r in range(PAIRED_READERS):
        both=obs_u[r,fov[sidx]] & obs_e[r,fov[sidx]]
        ii=sidx[both]
        yu=y_u[r,ii]; ye=y_e[r,ii]; aa=ai[ii]
        coverage.append(len(ii))
        if len(ii):
            paired_u.append(yu); paired_e.append(ye); paired_ai.append(aa)
            per_reader_diff.append(float(np.mean(ye==aa)-np.mean(yu==aa)))
        else:
            per_reader_diff.append(np.nan)

    if paired_u:
        pu=np.concatenate(paired_u)
        pe=np.concatenate(paired_e)
        pa=np.concatenate(paired_ai)
        paired_n=len(pu)
        switch=float(np.mean(pu!=pe))
        agr_u=float(np.mean(pu==pa))
        agr_e=float(np.mean(pe==pa))
        det_u=float(np.mean(pu))
        det_e=float(np.mean(pe))
    else:
        paired_n=0
        switch=agr_u=agr_e=det_u=det_e=np.nan

    pop_idx=np.arange(n)
    th_s_u,th_s_e=future_reader_truth(rng,d,ai,sidx,cfg,n_new_readers)
    th_p_u,th_p_e=future_reader_truth(rng,d,ai,pop_idx,cfg,n_new_readers)

    out={
        "scenario_id":int(cfg.scenario_id),
        "scenario":str(cfg.scenario),
        "seed":int(seed),
        "n_candidates":n,
        "n_selected":int(selected.sum()),
        "selection_rate":float(np.mean(selected)),
        "selected_mean_detectability":float(np.mean(d[selected])) if selected.any() else np.nan,
        "population_mean_detectability":float(np.mean(d)),
        "selected_ai_detection_rate":float(np.mean(ai[selected])) if selected.any() else np.nan,
        "paired_reader_anchor_observations":int(paired_n),
        "paired_reader_detection_U":det_u,
        "paired_reader_detection_E":det_e,
        "paired_reader_condition_switch_rate":switch,
        "paired_ai_agreement_U":agr_u,
        "paired_ai_agreement_E":agr_e,
        "paired_ai_agreement_difference_E_minus_U":agr_e-agr_u if paired_n else np.nan,
        "equal_reader_mean_agreement_difference_E_minus_U":
            float(np.nanmean(per_reader_diff)),
        "min_reader_paired_coverage":int(min(coverage)) if coverage else 0,
        "max_reader_paired_coverage":int(max(coverage)) if coverage else 0,
        "oracle_theta_selected_U":float(th_s_u),
        "oracle_theta_selected_E":float(th_s_e),
        "oracle_delta_selected_E_minus_U":float(th_s_e-th_s_u),
        "oracle_theta_population_U":float(th_p_u),
        "oracle_theta_population_E":float(th_p_e),
        "oracle_delta_population_E_minus_U":float(th_p_e-th_p_u),
        "selection_gap_U":float(th_s_u-th_p_u),
        "selection_gap_E":float(th_s_e-th_p_e),
    }
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scenario-grid",type=Path,required=True)
    ap.add_argument("--scenario-id",type=int,required=True)
    ap.add_argument("--n-reps",type=int,default=100)
    ap.add_argument("--oracle-readers",type=int,default=100)
    ap.add_argument("--seed-base",type=int,default=180000)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()

    grid=pd.read_csv(args.scenario_grid)
    hit=grid[grid.scenario_id==args.scenario_id]
    if len(hit)!=1:
        raise SystemExit(f"Expected exactly one scenario_id={args.scenario_id}, found {len(hit)}")
    cfg=hit.iloc[0]

    rows=[]
    for rep in range(args.n_reps):
        seed=args.seed_base + args.scenario_id*100000 + rep
        rows.append(simulate_once(cfg,seed,args.oracle_readers))
    out=pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=args.output.with_suffix(args.output.suffix+".tmp")
    out.to_csv(tmp,index=False)
    tmp.replace(args.output)
    print(f"V18_SCENARIO_COMPLETE scenario_id={args.scenario_id} rows={len(out)} output={args.output}")

if __name__=="__main__":
    main()
