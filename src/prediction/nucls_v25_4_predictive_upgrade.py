#!/usr/bin/env python3
from __future__ import annotations
from repo_paths import repo_root
import argparse, json, math, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

READERS=["SP.3","JP.1","JP.2","JP.5","JP.6"]
FEATURES=["overall_prop","U_prop","E_prop","condition_diff","n_U","n_E","U_missing","E_missing"]
N_FOLDS=5
EPS=1e-8

def metrics(y,p):
    y=np.asarray(y,int); p=np.asarray(p,float); pc=np.clip(p,1e-12,1-1e-12)
    out={
        "n":int(len(y)),"prevalence":float(y.mean()),
        "brier":float(np.mean((p-y)**2)),
        "log_loss":float(-np.mean(y*np.log(pc)+(1-y)*np.log(1-pc))),
    }
    if np.unique(y).size==2:
        out["auroc"]=float(roc_auc_score(y,p)); out["auprc"]=float(average_precision_score(y,p))
    else:
        out["auroc"]=float("nan"); out["auprc"]=float("nan")
    return out

def calibration_ab(y,p,max_iter=50):
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),1e-6,1-1e-6)
    x=logit(p)
    a=0.0; b=1.0
    for _ in range(max_iter):
        eta=a+b*x; mu=expit(eta); w=np.clip(mu*(1-mu),1e-9,None)
        g0=np.sum(y-mu); g1=np.sum((y-mu)*x)
        h00=np.sum(w)+1e-8; h01=np.sum(w*x); h11=np.sum(w*x*x)+1e-8
        det=h00*h11-h01*h01
        if not np.isfinite(det) or abs(det)<1e-12: return (float("nan"),float("nan"))
        da=(g0*h11-g1*h01)/det; db=(g1*h00-g0*h01)/det
        a+=da; b+=db
        if max(abs(da),abs(db))<1e-8: break
        if not np.isfinite(a+b): return (float("nan"),float("nan"))
    return float(a),float(b)

def all_stats(y,p):
    out=metrics(y,p); a,b=calibration_ab(y,p); out["calibration_intercept"]=a; out["calibration_slope"]=b
    out["predicted_mean"]=float(np.mean(p)); out["calibration_mean_error"]=float(np.mean(p)-np.mean(y))
    return out

def make_features(raw):
    d=raw.loc[raw.reader.isin(READERS)].copy(); d["anchor_id"]=d.anchor_id.astype(str)
    overall=d.groupby("anchor_id")["reader_detected"].agg(overall_prop="mean",n_total="size")
    by=d.groupby(["anchor_id","condition"])["reader_detected"].agg(prop="mean",n="size").reset_index()
    pprop=by.pivot(index="anchor_id",columns="condition",values="prop").rename(columns={"Unbiased":"U_prop","Evaluation":"E_prop"})
    pn=by.pivot(index="anchor_id",columns="condition",values="n").rename(columns={"Unbiased":"n_U","Evaluation":"n_E"})
    x=overall.join(pprop,how="left").join(pn,how="left")
    x["U_missing"]=x["U_prop"].isna().astype(int); x["E_missing"]=x["E_prop"].isna().astype(int)
    x["U_prop"]=x["U_prop"].fillna(x["overall_prop"]); x["E_prop"]=x["E_prop"].fillna(x["overall_prop"])
    x["n_U"]=x["n_U"].fillna(0); x["n_E"]=x["n_E"].fillna(0); x["condition_diff"]=x["E_prop"]-x["U_prop"]
    meta=(d.groupby("anchor_id").agg(image_id=("image_id","first"),slide=("slide","first"),patient_id=("patient_id","first"),image_n=("image_id","nunique"),slide_n=("slide","nunique"),patient_n=("patient_id","nunique")))
    if (meta[["image_n","slide_n","patient_n"]]!=1).any().any(): raise RuntimeError("Anchor metadata mapping is not unique")
    return x.join(meta[["image_id","slide","patient_id"]],how="left")

def load_ai_oof(cvdir):
    parts=[]; flags=[]
    for k in range(N_FOLDS):
        p=cvdir/f"ai_holdout_fold{k}_predictions.csv"; q=cvdir/f"ai_holdout_fold{k}_result.json"
        z=pd.read_csv(p); z["anchor_id"]=z.anchor_id.astype(str); parts.append(z[["anchor_id","ai_detected","pred","fold"]])
        g=json.loads(q.read_text()); flags.append({"fold":k,"divergences":int(g["divergences"]),"max_rhat_scalar":float(g["max_rhat_scalar"]),"min_ess_bulk_scalar":float(g["min_ess_bulk_scalar"]),"min_ess_tail_scalar":float(g["min_ess_tail_scalar"])})
    d=pd.concat(parts,ignore_index=True).rename(columns={"pred":"latent_bayes_pred"})
    if len(d)!=1144 or d.anchor_id.nunique()!=1144 or not d.groupby("anchor_id").fold.nunique().eq(1).all(): raise RuntimeError("AI OOF coverage invalid")
    return d,pd.DataFrame(flags)

def fit_nested_baselines(dat):
    dat=dat.copy(); dat["nested_logit_pred"]=np.nan; dat["tree_pred"]=np.nan; dat["train_prevalence_pred"]=np.nan
    tuning=[]
    for k in range(N_FOLDS):
        tr=dat.fold!=k; te=dat.fold==k; Xtr=dat.loc[tr,FEATURES]; Xte=dat.loc[te,FEATURES]; ytr=dat.loc[tr,"ai_detected"].astype(int)
        inner=StratifiedKFold(n_splits=4,shuffle=True,random_state=254000+k)
        logpipe=make_pipeline(StandardScaler(),LogisticRegression(solver="lbfgs",max_iter=5000))
        lg=GridSearchCV(logpipe,{"logisticregression__C":[0.01,0.1,1.0,10.0,100.0]},scoring="neg_log_loss",cv=inner,n_jobs=1,refit=True)
        lg.fit(Xtr,ytr); dat.loc[te,"nested_logit_pred"]=lg.predict_proba(Xte)[:,1]
        tree=HistGradientBoostingClassifier(max_iter=200,early_stopping=False,random_state=254100+k)
        tg=GridSearchCV(tree,{"learning_rate":[0.05,0.10],"max_leaf_nodes":[7,15],"l2_regularization":[0.0,1.0]},scoring="neg_log_loss",cv=inner,n_jobs=1,refit=True)
        tg.fit(Xtr,ytr); dat.loc[te,"tree_pred"]=tg.predict_proba(Xte)[:,1]
        dat.loc[te,"train_prevalence_pred"]=float(ytr.mean())
        tuning.append({"fold":k,"model":"nested_regularized_logistic","best_score_neg_log_loss":float(lg.best_score_),"best_params":json.dumps(lg.best_params_,sort_keys=True)})
        tuning.append({"fold":k,"model":"hist_gradient_boosting","best_score_neg_log_loss":float(tg.best_score_),"best_params":json.dumps(tg.best_params_,sort_keys=True)})
    for c in ["nested_logit_pred","tree_pred","train_prevalence_pred"]:
        if dat[c].isna().any() or not np.isfinite(dat[c]).all(): raise RuntimeError(f"Missing/nonfinite OOF predictions in {c}")
    return dat,pd.DataFrame(tuning)

def bootstrap_indices(df,cluster_col,B,seed):
    vals=df[cluster_col].astype(str).to_numpy(); groups=np.unique(vals); by={g:np.flatnonzero(vals==g) for g in groups}; rng=np.random.default_rng(seed)
    for _ in range(B):
        samp=rng.choice(groups,size=len(groups),replace=True); yield np.concatenate([by[g] for g in samp])

def stable_seed(*parts):
    h=hashlib.sha256("|".join(map(str,parts)).encode()).digest()
    return int.from_bytes(h[:4],"little")

def summarize_models(df,ycol,model_map,target,B=1000):
    rows=[]; bootrows=[]; relrows=[]
    y=df[ycol].astype(int).to_numpy()
    for model,pcol in model_map.items():
        p=df[pcol].astype(float).to_numpy(); s=all_stats(y,p); rows.append({"target":target,"model":model,**s})
        for scheme,cluster,seed in [("anchor_cluster","anchor_id",255001),("slide_cluster","slide",255101)]:
            vals=[]
            for idx in bootstrap_indices(df,cluster,B,seed+stable_seed(target,model)%10000):
                vals.append(all_stats(y[idx],p[idx]))
            for metric in ["brier","log_loss","auroc","auprc","calibration_intercept","calibration_slope","predicted_mean"]:
                a=np.array([v[metric] for v in vals],float); a=a[np.isfinite(a)]
                q=np.quantile(a,[.025,.5,.975]) if len(a) else [np.nan]*3
                bootrows.append({"target":target,"model":model,"scheme":scheme,"metric":metric,"q025":q[0],"median":q[1],"q975":q[2],"n_boot_valid":len(a)})
        try:
            bins=pd.qcut(pd.Series(p),q=10,duplicates="drop",labels=False).to_numpy()
        except Exception:
            bins=pd.cut(pd.Series(p),bins=10,labels=False,include_lowest=True).to_numpy()
        nb=int(np.nanmax(bins))+1
        obs_boot={b:[] for b in range(nb)}; pred_boot={b:[] for b in range(nb)}
        for idx in bootstrap_indices(df,"slide",B,256001+stable_seed(target,model)%10000):
            for b in range(nb):
                m=(bins[idx]==b)
                if m.any(): obs_boot[b].append(float(y[idx][m].mean())); pred_boot[b].append(float(p[idx][m].mean()))
        for b in range(nb):
            m=bins==b; oq=np.quantile(obs_boot[b],[.025,.975]) if obs_boot[b] else [np.nan,np.nan]; pq=np.quantile(pred_boot[b],[.025,.975]) if pred_boot[b] else [np.nan,np.nan]
            relrows.append({"target":target,"model":model,"bin":b,"n":int(m.sum()),"pred_mean":float(p[m].mean()),"obs_mean":float(y[m].mean()),"slide_boot_pred_low":pq[0],"slide_boot_pred_high":pq[1],"slide_boot_obs_low":oq[0],"slide_boot_obs_high":oq[1]})
    return pd.DataFrame(rows),pd.DataFrame(bootrows),pd.DataFrame(relrows)

def pairwise_diff(df,ycol,pA,pB,nameA,nameB,target,B=1000):
    y=df[ycol].astype(int).to_numpy(); pa=df[pA].to_numpy(float); pb=df[pB].to_numpy(float); out=[]
    pointA=metrics(y,pa); pointB=metrics(y,pb)
    for metric in ["brier","log_loss","auroc","auprc"]:
        out.append({"target":target,"model_A":nameA,"model_B":nameB,"scheme":"point","metric":metric,"estimate_A_minus_B":pointA[metric]-pointB[metric],"q025":np.nan,"median":np.nan,"q975":np.nan})
    for scheme,cluster,seed in [("anchor_cluster","anchor_id",257001),("slide_cluster","slide",257101)]:
        vals={m:[] for m in ["brier","log_loss","auroc","auprc"]}
        for idx in bootstrap_indices(df,cluster,B,seed+stable_seed(target,nameA,nameB)%10000):
            ma=metrics(y[idx],pa[idx]); mb=metrics(y[idx],pb[idx])
            for m in vals: vals[m].append(ma[m]-mb[m])
        for m,a in vals.items():
            a=np.asarray(a,float); a=a[np.isfinite(a)]; q=np.quantile(a,[.025,.5,.975]) if len(a) else [np.nan]*3
            out.append({"target":target,"model_A":nameA,"model_B":nameB,"scheme":scheme,"metric":m,"estimate_A_minus_B":pointA[m]-pointB[m],"q025":q[0],"median":q[1],"q975":q[2]})
    return pd.DataFrame(out)

def load_human_pair(cvdir):
    js=[]; rs=[]
    keys=["anchor_id","image_id","slide","patient_id","condition","reader","reader_detected","fold"]
    for k in range(N_FOLDS):
        j=pd.read_csv(cvdir/f"human_joint_fold{k}_predictions.csv"); r=pd.read_csv(cvdir/f"human_reader_only_fold{k}_predictions.csv")
        j["anchor_id"]=j.anchor_id.astype(str); r["anchor_id"]=r.anchor_id.astype(str); j["fold"]=k; r["fold"]=k
        js.append(j[keys+['pred']].rename(columns={'pred':'joint_human_pred'})); rs.append(r[keys+['pred']].rename(columns={'pred':'reader_human_pred'}))
    return pd.concat(js).merge(pd.concat(rs),on=keys,validate='one_to_one')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',type=Path,default=repo_root()); ap.add_argument('--B',type=int,default=1000); args=ap.parse_args()
    root=args.project_root; cv=root/'outputs'/'v23_1_shared_detectability_cv'; out=root/'outputs'/'v25_4_predictive_upgrade'; out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(root/'outputs'/'scoring_v16_1'/'restricted_recorded_reader_detection_agreement_v16.csv')
    ai,ai_diag=load_ai_oof(cv); feat=make_features(raw); dat=ai.join(feat,on='anchor_id')
    if dat[FEATURES+['image_id','slide','patient_id']].isna().any().any(): raise RuntimeError('AI benchmark join contains missing values')
    dat,tune=fit_nested_baselines(dat); dat.to_csv(out/'ai_oof_predictions.csv',index=False); tune.to_csv(out/'nested_tuning.csv',index=False); ai_diag.to_csv(out/'ai_latent_fold_diagnostics.csv',index=False)
    ai_models={'latent_bayes':'latent_bayes_pred','nested_regularized_logistic':'nested_logit_pred','hist_gradient_boosting':'tree_pred','train_prevalence':'train_prevalence_pred'}
    sm_ai,bt_ai,rel_ai=summarize_models(dat,'ai_detected',ai_models,'heldout_ai',args.B)
    diffs=[]
    for name,pcol in [('nested_regularized_logistic','nested_logit_pred'),('hist_gradient_boosting','tree_pred'),('train_prevalence','train_prevalence_pred')]: diffs.append(pairwise_diff(dat,'ai_detected','latent_bayes_pred',pcol,'latent_bayes',name,'heldout_ai',args.B))
    human=load_human_pair(cv)
    human_summ=[]; human_boot=[]; human_rel=[]
    for label,sub in [('heldout_human',human),('heldout_human_U',human[human.condition=='Unbiased'].copy()),('heldout_human_E',human[human.condition=='Evaluation'].copy())]:
        sh,bh,rh=summarize_models(sub,'reader_detected',{'joint':'joint_human_pred','reader_only':'reader_human_pred'},label,args.B); human_summ.append(sh); human_boot.append(bh); human_rel.append(rh); diffs.append(pairwise_diff(sub,'reader_detected','joint_human_pred','reader_human_pred','joint','reader_only',label,args.B))
    agr=pd.read_csv(cv/'target_aligned_agreement_predictions.csv'); agr['anchor_id']=agr.anchor_id.astype(str)
    agr_summ=[]; agr_boot=[]; agr_rel=[]
    for label,sub in [('target_aligned_agreement',agr),('target_aligned_agreement_U',agr[agr.condition=='Unbiased'].copy()),('target_aligned_agreement_E',agr[agr.condition=='Evaluation'].copy())]:
        sa,ba,ra=summarize_models(sub,'actual_agreement',{'joint':'joint_agree_pred','reader_only':'reader_agree_pred'},label,args.B); agr_summ.append(sa); agr_boot.append(ba); agr_rel.append(ra); diffs.append(pairwise_diff(sub,'actual_agreement','joint_agree_pred','reader_agree_pred','joint','reader_only',label,args.B))
    sm_h=pd.concat(human_summ,ignore_index=True); bt_h=pd.concat(human_boot,ignore_index=True); rel_h=pd.concat(human_rel,ignore_index=True)
    sm_a=pd.concat(agr_summ,ignore_index=True); bt_a=pd.concat(agr_boot,ignore_index=True); rel_a=pd.concat(agr_rel,ignore_index=True)
    summary=pd.concat([sm_ai,sm_h,sm_a],ignore_index=True); boot=pd.concat([bt_ai,bt_h,bt_a],ignore_index=True); reliab=pd.concat([rel_ai,rel_h,rel_a],ignore_index=True); comp=pd.concat(diffs,ignore_index=True)
    summary.to_csv(out/'predictive_model_summary.csv',index=False); boot.to_csv(out/'bootstrap_metric_intervals.csv',index=False); reliab.to_csv(out/'reliability_with_slide_bootstrap.csv',index=False); comp.to_csv(out/'pairwise_metric_differences.csv',index=False)
    flag=(ai_diag.divergences.gt(0)|ai_diag.max_rhat_scalar.gt(1.01)|ai_diag.min_ess_bulk_scalar.lt(400)|ai_diag.min_ess_tail_scalar.lt(400))
    gate={
      'status':'V25_4_PREDICTIVE_UPGRADE_COMPLETE', 'n_ai_anchors':int(len(dat)), 'n_human_rows':int(len(human)), 'n_agreement_rows':int(len(agr)),
      'outer_fold_anchor_unique':bool(dat.groupby('anchor_id').fold.nunique().eq(1).all()), 'outer_fold_count':int(dat.fold.nunique()),
      'ai_latent_flagged_folds':ai_diag.loc[flag,'fold'].astype(int).tolist(), 'ai_latent_flags_are_nonblocking_diagnostic':True,
      'nested_regularized_logistic':True, 'nonlinear_tree_baseline':True, 'calibration_intercept_slope':True,
      'reliability_with_slide_bootstrap':True, 'anchor_cluster_bootstrap':True, 'slide_cluster_bootstrap':True,
      'loso_prediction_complete':False,
      'interpretation':['Prediction assesses shared information and calibration, not biological truth or unique validity of the latent factorization.','Outer folds retain observed FOV/slide contexts and do not establish new-patient generalization.','AUPRC is reported with event prevalence.']
    }
    (out/'gate.json').write_text(json.dumps(gate,indent=2))
    print('===== V25.4 PREDICTIVE UPGRADE ====='); print(summary.to_string(index=False)); print('\n===== AI LATENT FOLD DIAGNOSTICS ====='); print(ai_diag.to_string(index=False)); print('\n===== PAIRWISE DIFFERENCES ====='); print(comp.to_string(index=False)); print('\n===== GATE ====='); print(json.dumps(gate,indent=2)); print('WROTE',out)
if __name__=='__main__': main()
