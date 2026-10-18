#!/usr/bin/env python3
"""Combine four independently checkpointed NUTS chains into original v26.1 outputs.

This does not hide sampler failures or silently reuse incomplete chains.
Use --strict-all only after the full 2,400-fit campaign is expected complete.
"""
import argparse,csv,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
from v26_1_models import (summarize_draw_matrix,scalar_var_names)

TARGET=['theta_U_selected','theta_E_selected','delta_E_minus_U_selected']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1<<20),b''):h.update(data)
    return h.hexdigest()

def write_atomic(path,writer):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        writer(tmp)
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)

def one_fit(model,scenario,rep,root,outdir):
    base=root/f'fit_{model}'/scenario/f'rep_{rep:03d}'
    gates=[]; chain_samples=[]
    for chain in range(4):
        d=base/f'chain_{chain}'
        gate=d/'CHAIN_COMPLETE.json'; sample=d/'chain_samples.npz'
        if not gate.exists():return False
        g=json.loads(gate.read_text())
        p=g.get('provenance',{})
        if (g.get('status')!='COMPLETE' or p.get('model')!=model or
            p.get('scenario')!=scenario or p.get('rep')!=rep or p.get('chain')!=chain):
            raise RuntimeError('Incompatible chain gate: '+str(gate))
        if not sample.exists() or sha(sample)!=g.get('samples_sha256'):
            raise RuntimeError('Chain samples missing/corrupt: '+str(sample))
        with np.load(sample,allow_pickle=False) as z:
            sc=z['scalar_draws'].copy();ag=z['agreement_draws'].copy()
            div=z['divergences'].copy();names=z['scalar_names'].tolist()
        if names!=scalar_var_names(model) or ag.shape!=(int(p['sampler']['draws']),3) or sc.shape!=(int(p['sampler']['draws']),len(names)) or div.shape!=(int(p['sampler']['draws']),):
            raise RuntimeError('Bad sample arrays '+str(sample))
        if not np.isfinite(sc).all() or not np.isfinite(ag).all():
            raise RuntimeError('Nonfinite values '+str(sample))
        gates.append(g);chain_samples.append((sc,ag,div))
    stable_keys=['frozen_root_hash','dataset_sha256','models_sha256','worker_sha256','pymc_version','sampler','fit_seed']
    for key in stable_keys:
        if len({json.dumps(g['provenance'][key],sort_keys=True) for g in gates})!=1:
            raise RuntimeError('Cross-chain provenance mismatch: '+key)
    import arviz as az
    posterior={name:np.stack([v[0][:,j] for v in chain_samples]) for j,name in enumerate(names)}
    idata=az.from_dict(posterior=posterior)
    ds=az.summary(idata,var_names=names,round_to=None)
    div=int(sum(np.asarray(v[2],dtype=int).sum() for v in chain_samples))
    diag={'divergences':div,'max_rhat_scalar':float(ds.r_hat.max()),
          'min_ess_bulk_scalar':float(ds.ess_bulk.min()),
          'min_ess_tail_scalar':float(ds.ess_tail.min()),
          'elapsed_sec':float(sum(g.get('elapsed_sec',0) for g in gates))}
    computational_ok=(div==0 and diag['max_rhat_scalar']<=1.01 and
                      diag['min_ess_bulk_scalar']>=400 and diag['min_ess_tail_scalar']>=400)
    agreements=np.vstack([v[1] for v in chain_samples]); summ=summarize_draw_matrix(agreements)
    if summ['target'].tolist()!=TARGET:
        raise RuntimeError('Target schema mismatch')
    path=outdir/model/scenario
    stem=f'rep_{rep:03d}'
    summary_path=path/f'{stem}_summary.csv'; diag_path=path/f'{stem}_scalar_diagnostics.csv'
    gate_path=path/f'{stem}_gate.json'
    new_gate={'model':model,'scenario':scenario,'rep':rep,
       'freeze_hash':gates[0]['provenance']['frozen_root_hash'],
       'source_dataset_sha256':gates[0]['provenance']['dataset_sha256'],
       'seed':gates[0]['provenance']['fit_seed'],
       'sampler':gates[0]['provenance']['sampler'],
       'chain_seed_list':[g['provenance']['chain_seed'] for g in gates],
       **diag,'computational_ok':bool(computational_ok),'status':'COMPLETE',
       'chain_samples_sha256':[g['samples_sha256'] for g in gates]}
    if gate_path.exists():
        old=json.loads(gate_path.read_text())
        if (old.get('source_dataset_sha256')!=new_gate['source_dataset_sha256'] or
            old.get('chain_samples_sha256')!=new_gate['chain_samples_sha256']):
            raise RuntimeError('Existing fit gate has incompatible provenance: '+str(gate_path))
        if not summary_path.exists() or not diag_path.exists():
            raise RuntimeError('Existing fit gate missing summary/diagnostics')
        return True
    write_atomic(summary_path,lambda p:summ.to_csv(p,index=False))
    write_atomic(diag_path,lambda p:ds.to_csv(p))
    new_gate['summary_sha256']=sha(summary_path)
    new_gate['diagnostics_sha256']=sha(diag_path)
    write_atomic(gate_path,lambda p:p.write_text(json.dumps(new_gate,sort_keys=True,indent=2)+'\n'))
    print('FIT_COMPLETE',model,scenario,rep,'computational_ok',computational_ok,flush=True)
    return True

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--chain-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--strict-all',action='store_true')
    a=p.parse_args()
    with a.manifest.open(newline='') as f:rows=list(csv.DictReader(f))
    units={(r['fit_model'],r['scenario'],int(r['replication'])) for r in rows}
    if len(units)!=2400:raise RuntimeError('Expected 2400 unique fitted datasets')
    completed=0; missing=[]
    for model,scenario,rep in sorted(units):
        if one_fit(model,scenario,rep,a.chain_root,a.output_root): completed+=1
        else: missing.append((model,scenario,rep))
    print('FINALIZATION_AUDIT',json.dumps({'expected':2400,'complete':completed,'missing':len(missing)}))
    if missing:
        a.output_root.mkdir(parents=True,exist_ok=True)
        (a.output_root/'missing_fit_units.json').write_text(json.dumps(missing,indent=2)+'\n')
    if a.strict_all and missing:raise RuntimeError(f'{len(missing)} fitting results incomplete')

if __name__=='__main__':main()
