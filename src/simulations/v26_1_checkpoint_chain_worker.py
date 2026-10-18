#!/usr/bin/env python3
"""Checkpointable 1-chain PyMC NUTS worker for v26.1. EXPERIMENTAL; smoke-test on cluster.

A checkpoint contains the complete NUTS `sampling_state` (adaptation, step and
potential RNG), position, post-warmup scalar/estimand draws and provenance.
Two rotating atomic files retain the latest and penultimate transitions.
This does not alter `v26_1_models.build_model` or the selected-frame estimand.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, pickle, signal, sys, time
from pathlib import Path
import numpy as np
from scipy.special import expit
from v26_1_models import (PROTOCOL, index_to_scenario_rep, load_frozen,
                           encode_selected, build_model, GHX30, GHW30,
                           scalar_var_names, sha256_file)

SOFT_STOP=False
SCALAR_BASE=['alpha','beta_E','sd_reader','sd_reader_condition','sd_fov',
             'sd_reader_fov','sd_anchor','sd_reader_anchor']

def _stop(sig, frame):
    global SOFT_STOP
    SOFT_STOP=True

def sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()

def atomic_bytes(path:Path, data:bytes):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def atomic_json(path:Path, obj):
    atomic_bytes(path,(json.dumps(obj,sort_keys=True,indent=2)+'\n').encode())

def fit_seed(model, scenario, rep):
    x=f'{PROTOCOL}|fit|{model}|{scenario}|{rep}'.encode()
    return int.from_bytes(hashlib.sha256(x).digest()[:4],'little')%2_000_000_000

def chain_seed(model,scenario,rep,chain):
    # Same per-chain seed expansion used by PyMC's integer-seed, 4-chain mode.
    rng=np.random.default_rng(fit_seed(model,scenario,rep))
    while True:
        seeds=[int(x) for x in rng.integers(2**30,size=4,dtype=np.int64)]
        if len(set(seeds))==4:
            return seeds[chain]

def calc_draw(point, data):
    """Compute same selected-frame future-reader agreement as v26_1_models.

    HalfNormal variables are stored in PyMC's value-point log coordinates.
    The 30-node GH integral uses the original v26_1_models constants.
    """
    scalar={}
    for name in SCALAR_BASE:
        if name in ('alpha','beta_E'):
            scalar[name]=float(np.asarray(point[name]))
        else:
            scalar[name]=math.exp(float(np.asarray(point[f'{name}_log__'])))
    if 'alpha_ai' in point:
        scalar['alpha_ai']=float(np.asarray(point['alpha_ai']))
        scalar['beta_ai']=math.exp(float(np.asarray(point['beta_ai_log__'])))
    baseline=(scalar['alpha']+
       scalar['sd_fov']*np.asarray(point['z_fov'])[data.anchor_f]+
       scalar['sd_anchor']*np.asarray(point['z_anchor']))
    su=math.sqrt(scalar['sd_reader']**2+scalar['sd_reader_fov']**2+
                 scalar['sd_reader_anchor']**2)
    se=math.sqrt(su**2+scalar['sd_reader_condition']**2)
    pu=np.dot(expit(baseline[:,None]+math.sqrt(2)*su*GHX30[None,:]),GHW30)
    pe=np.dot(expit((baseline+scalar['beta_E'])[:,None]+math.sqrt(2)*se*GHX30[None,:]),GHW30)
    ai=np.asarray(data.anchor_ai,dtype=np.float64)
    theta_u=float(np.mean(ai*pu+(1-ai)*(1-pu)))
    theta_e=float(np.mean(ai*pe+(1-ai)*(1-pe)))
    return scalar,[theta_u,theta_e,theta_e-theta_u]

def version_check(pm):
    # PyMC sampling_state API was introduced in the modern PyMC 5 line;
    # do not assume availability on the cluster until a real smoke test.
    if not hasattr(pm.NUTS,'sampling_state'):
        raise RuntimeError('This PyMC NUTS does not expose sampling_state; cannot safely resume')

def make_provenance(a, source_file, data_file_sha, expected_hash, pm_version):
    cfg={'draws':a.draws,'tune':a.tune,'chains':4,'target_accept':a.target_accept,
         'checkpoint_every':a.checkpoint_every,'schema':'v26_1_nuts_chain_checkpoint_v1'}
    return {'protocol':PROTOCOL,'model':a.model,'scenario':a.scenario,'rep':a.rep,
            'original_index':a.index,'chain':a.chain,'fit_seed':fit_seed(a.model,a.scenario,a.rep),
            'chain_seed':chain_seed(a.model,a.scenario,a.rep,a.chain),
            'dataset_sha256':data_file_sha,'frozen_root_hash':expected_hash,
            'models_sha256':sha256_file(source_file),
            'worker_sha256':sha256_file(Path(__file__)),
            'pymc_version':pm_version,'sampler':cfg}

def checkpoint_bytes(payload):
    raw=pickle.dumps(payload,protocol=pickle.HIGHEST_PROTOCOL)
    return raw

def read_checkpoint(folder, prov):
    valid=[]; invalid=[]
    for p in [folder/'checkpoint_A.pkl',folder/'checkpoint_B.pkl']:
        if not p.exists():continue
        try:
            # Files are created by this trusted worker, not an external source.
            d=pickle.loads(p.read_bytes())
            if d.get('provenance')!=prov:
                raise RuntimeError('Incompatible provenance in checkpoint '+str(p))
            if not isinstance(d['iteration'],int) or d['iteration']<0:
                raise ValueError('invalid iteration')
            if len(d['scalar_draws'])!=len(d['agreement_draws']) or len(d['agreement_draws'])!=len(d['divergences']):
                raise ValueError('inconsistent posterior record lengths')
            n_expected=max(0,d['iteration']-prov['sampler']['tune'])
            if len(d['agreement_draws'])!=n_expected:
                raise ValueError(f'posterior count mismatch: {len(d["agreement_draws"])} != {n_expected}')
            valid.append((d['iteration'],p,d))
        except RuntimeError:
            raise  # provenance mismatch must fail closed
        except Exception as exc:
            invalid.append((p,str(exc)))
    if valid:
        valid.sort(key=lambda z:z[0],reverse=True)
        print(f'RESUME {folder} iter={valid[0][0]} file={valid[0][1].name}',flush=True)
        return valid[0][2]
    if invalid:
        raise RuntimeError('No valid checkpoint; corrupted files: '+repr(invalid))
    return None

def checkpoint(folder,payload):
    # Alternating snapshots guarantee an earlier usable state if a write aborts.
    which='A' if ((payload['iteration']//payload['provenance']['sampler']['checkpoint_every'])%2==0) else 'B'
    atomic_bytes(folder/f'checkpoint_{which}.pkl',checkpoint_bytes(payload))

def verified_done(folder,prov):
    gate=folder/'CHAIN_COMPLETE.json'
    if not gate.exists():return False
    d=json.loads(gate.read_text())
    if d.get('provenance')!=prov or d.get('status')!='COMPLETE':
        raise RuntimeError(f'Incompatible CHAIN_COMPLETE: {gate}')
    samples=folder/'chain_samples.npz'
    if not samples.is_file() or sha256_file(samples)!=d.get('samples_sha256'):
        raise RuntimeError(f'Corrupt chain_samples with completion marker: {samples}')
    with np.load(samples,allow_pickle=False) as arr:
        if arr['agreement_draws'].shape!=(prov['sampler']['draws'],3):
            raise RuntimeError('Wrong agreement dimensions')
        if arr['scalar_draws'].shape!=(prov['sampler']['draws'],len(scalar_var_names(prov['model']))):
            raise RuntimeError('Wrong scalar dimensions')
        if arr['divergences'].shape!=(prov['sampler']['draws'],):
            raise RuntimeError('Wrong divergence dimensions')
        if not np.isfinite(arr['scalar_draws']).all() or not np.isfinite(arr['agreement_draws']).all():
            raise RuntimeError('Nonfinite completed values')
    return True

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=['M1','M2'],required=True)
    ap.add_argument('--scenario',required=True)
    ap.add_argument('--rep',type=int,required=True)
    ap.add_argument('--index',type=int,required=True,help='ORIGINAL frozen-data index, 0..2399')
    ap.add_argument('--chain',type=int,choices=range(4),required=True)
    ap.add_argument('--freeze-root',type=Path,required=True)
    ap.add_argument('--expected-freeze-hash-file',type=Path,required=True)
    ap.add_argument('--out-root',type=Path,required=True)
    ap.add_argument('--draws',type=int,default=4000)
    ap.add_argument('--tune',type=int,default=2000)
    ap.add_argument('--target-accept',type=float,default=0.99)
    ap.add_argument('--checkpoint-every',type=int,default=50)
    ap.add_argument('--max-transitions',type=int,default=0,help='SMOKE ONLY: force exit 75 at iteration')
    ap.add_argument('--smoke',action='store_true',help='Required for nonproduction settings')
    ap.add_argument('--validate-only',action='store_true')
    a=ap.parse_args()
    if (a.draws,a.tune,a.target_accept)!=(4000,2000,0.99) and not a.smoke:
        raise RuntimeError('Production settings must be 4000/2000/0.99')
    if a.max_transitions and not a.smoke:
        raise RuntimeError('Artificial interruption is smoke-only')
    if a.draws<1 or a.tune<1 or a.checkpoint_every<1:
        raise ValueError('Invalid sampler settings')
    if (a.scenario,a.rep)!=index_to_scenario_rep(a.index):
        raise RuntimeError('Original frozen index is inconsistent with scenario/rep')
    if not 0<=a.rep<100:raise RuntimeError('Nondefinitive replication disallowed')
    expected_hash=a.expected_freeze_hash_file.read_text().strip()
    dat=load_frozen(a.freeze_root,a.scenario,a.rep,expected_hash,verify_hash=True)
    enc=encode_selected(dat)
    if a.validate_only:
        print('CHECKPOINT_WORKER_VALIDATE_OK',a.model,a.scenario,a.rep,a.chain,
              'nreaders',enc.n_readers,'anchors',enc.n_anchors,'fovs',enc.n_fovs,flush=True)
        return 0
    import pymc as pm
    version_check(pm)
    prov=make_provenance(a,Path(__file__).with_name('v26_1_models.py'),
                         sha256_file(dat['path']),expected_hash,pm.__version__)
    folder=(a.out_root/f'fit_{a.model}'/a.scenario/f'rep_{a.rep:03d}'/f'chain_{a.chain}')
    folder.mkdir(parents=True,exist_ok=True)
    if verified_done(folder,prov):
        print('SKIP_COMPLETED',folder,flush=True);return 0
    current=read_checkpoint(folder,prov)
    sig=chain_seed(a.model,a.scenario,a.rep,a.chain)
    model=build_model(enc,a.model)
    from pymc.sampling.mcmc import init_nuts
    # Compile exactly the original 'jitter+adapt_diag' NUTS initializer.
    initial_points,step=init_nuts(init='jitter+adapt_diag',chains=1,
        model=model,random_seed=sig,target_accept=a.target_accept,quiet=True)
    if current is None:
        point=initial_points[0]
        current={'provenance':prov,'iteration':0,'point':point,
                 'step_state':step.sampling_state,'scalar_draws':[],
                 'agreement_draws':[],'divergences':[],
                 'elapsed_sec':0.0}
    else:
        step.sampling_state=current['step_state']
        point=current['point']
    names=scalar_var_names(a.model)
    maxiter=a.draws+a.tune
    deadline=int(os.environ.get('SOFT_DEADLINE_EPOCH','9999999999999'))
    t0=time.monotonic()
    signal.signal(signal.SIGUSR1,_stop)
    signal.signal(signal.SIGTERM,_stop)
    while current['iteration']<maxiter:
        if SOFT_STOP or time.time()>=deadline or (a.max_transitions and current['iteration']>=a.max_transitions):
            current['point']=point
            current['step_state']=step.sampling_state
            current['elapsed_sec']+=time.monotonic()-t0
            checkpoint(folder,current)
            print(f'CHECKPOINT_STOP iter={current["iteration"]} unit={folder}',flush=True)
            return 75
        i=current['iteration']
        if i==a.tune:step.stop_tuning()
        point,stats=step.step(point)
        current['iteration']+=1
        if i>=a.tune:
            pars,theta=calc_draw(point,enc)
            current['scalar_draws'].append([pars[n] for n in names])
            current['agreement_draws'].append(theta)
            if not stats or 'diverging' not in stats[0]:
                raise RuntimeError('Unexpected NUTS statistics schema')
            current['divergences'].append(bool(stats[0]['diverging']))
        if current['iteration']%a.checkpoint_every==0:
            current['point']=point
            current['step_state']=step.sampling_state
            current['elapsed_sec']+=time.monotonic()-t0
            checkpoint(folder,current)
            t0=time.monotonic()
            if current['iteration']%500==0:
                print(f'CHECKPOINT iter={current["iteration"]}/{maxiter} unit={folder}',flush=True)
    current['point']=point
    current['step_state']=step.sampling_state
    current['elapsed_sec']+=time.monotonic()-t0
    checkpoint(folder,current)
    scalar=np.asarray(current['scalar_draws'],dtype=np.float64)
    agree=np.asarray(current['agreement_draws'],dtype=np.float64)
    divergence=np.asarray(current['divergences'],dtype=bool)
    if scalar.shape!=(a.draws,len(names)) or agree.shape!=(a.draws,3) or divergence.shape!=(a.draws,):
        raise RuntimeError('Incomplete chain despite reaching target iteration')
    if not np.isfinite(scalar).all() or not np.isfinite(agree).all():
        raise RuntimeError('Nonfinite posterior samples')
    import io
    buf=io.BytesIO()
    np.savez(buf,scalar_draws=scalar,agreement_draws=agree,divergences=divergence,
             scalar_names=np.asarray(names,dtype='U36'))
    atomic_bytes(folder/'chain_samples.npz',buf.getvalue())
    gate={'status':'COMPLETE','provenance':prov,'sampler_iterations':maxiter,
          'samples_sha256':sha256_file(folder/'chain_samples.npz'),
          'n_divergences':int(divergence.sum()),'elapsed_sec':current['elapsed_sec']}
    atomic_json(folder/'CHAIN_COMPLETE.json',gate)
    print('CHAIN_COMPLETE '+json.dumps({'unit':str(folder),'divergences':int(divergence.sum()),
          'elapsed_sec':current['elapsed_sec']}),flush=True)
    return 0

if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f'CHAIN_FATAL {type(e).__name__}: {e}',file=sys.stderr,flush=True)
        raise
