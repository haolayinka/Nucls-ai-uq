#!/usr/bin/env python3
"""Read-only NuCLS v26.1 Bayesian evaluation; locked 2026-10-15 rules.

Inputs are the ORIGINAL 2,400 finalized fits, the ORIGINAL 1,200 frozen
replication-specific truths, and the ORIGINAL frozen 2,400-fit manifest.
Never uses any higher-iteration sensitivity outputs.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

TARGETS = ('theta_U_selected', 'theta_E_selected', 'delta_E_minus_U_selected')
SAMPLES = {'chains':4, 'draws':4000, 'tune':2000, 'target_accept':0.99}
MODEL_PAIRS = ('M1', 'M2')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with open(path, encoding='utf8') as fh:
        return json.load(fh)


def read_csv(path):
    with open(path, newline='', encoding='utf8') as fh:
        return list(csv.DictReader(fh))


def assert_both(cond, message):
    if not cond:
        raise ValueError(message)


def finite_float(value, what):
    f = float(value)
    assert_both(math.isfinite(f), f'{what} must be finite, got {value!r}')
    return f


def read_freeze_index(freeze, expected):
    header = read_json(freeze/'FREEZE_MANIFEST.json')
    assert_both(header['frozen_data_sha256']==expected, 'Unexpected frozen manifest hash')
    assert_both(header['n_frozen_datasets']==2400, 'Unexpected frozen 2,400 universe')
    sums_file = freeze/'SHA256SUMS.txt'
    assert_both(sha256(sums_file)==expected, 'Frozen SHA256SUMS.txt checksum mismatch')
    lines = {}
    with open(sums_file, encoding='utf8') as f:
        for line in f:
            value, key = line.strip().split(None, 1)
            assert_both(len(value)==64 and key not in lines, 'Invalid frozen checksum index')
            lines[key.strip()] = value
    return lines


def expected_manifest(manifest):
    rows = read_csv(manifest)
    assert_both(len(rows)==2400, f'Fit manifest must have 2400 units, got {len(rows)}')
    indexed = {}
    pairs = collections.Counter()
    for row in rows:
        model = row['fit_model']; scen=row['scenario']; rep=int(row['replication'])
        generator, readers, slides = scen.split('_')
        assert_both(generator in MODEL_PAIRS and model in MODEL_PAIRS, 'Bad model')
        assert_both(readers in ('R5','R10') and slides in ('S5','S10','S20'), 'Bad support')
        assert_both(row['generator']==generator and int(row['readers'])==int(readers[1:]) and int(row['slides'])==int(slides[1:]), 'Manifest mismatch')
        assert_both(0<=rep<100 and row['unit_id']==f'fit_{model}/{scen}/rep_{rep:03d}', 'Invalid definitive replication/unit')
        key=(model,scen,rep)
        assert_both(key not in indexed, f'Duplicate manifest unit {key}')
        indexed[key]=row
        pairs[(model,scen)]+=1
    assert_both(len(pairs)==24 and all(n==100 for n in pairs.values()), '24 model×scenario cells each need 100 fits')
    return indexed


def truth_catalog(freeze, known_hashes, manifest):
    catalog={}
    for _,scen,rep in manifest:
        key=(scen,rep)
        if key in catalog: continue
        rel=f'truth/{scen}/rep_{rep:03d}.json'
        src=freeze/rel
        assert_both(rel in known_hashes and src.exists() and sha256(src)==known_hashes[rel], f'Frozen truth SHA mismatch: {rel}')
        t=read_json(src)
        assert_both(t['scenario']==scen and int(t['rep'])==rep and t['generator']==scen[:2], f'Truth identity mismatch: {rel}')
        values={target:finite_float(t[target],f'{rel}:{target}') for target in TARGETS}
        assert_both(all(0<=values[k]<=1 for k in TARGETS[:2]), 'Agreement truth outside [0,1]')
        assert_both(abs(values[TARGETS[1]]-values[TARGETS[0]]-values[TARGETS[2]])<1e-10, 'Truth delta inconsistent')
        catalog[key]=values
    assert_both(len(catalog)==1200, f'1200 distinct truth files expected, got {len(catalog)}')
    return catalog


def validate_one(row, fit_root, chain_root, freeze_root, freeze_hashes, expected_freeze, verify_chains):
    model=row['fit_model']; scenario=row['scenario']; rep=int(row['replication'])
    stem=f'rep_{rep:03d}'; folder=fit_root/model/scenario
    gate_path=folder/(stem+'_gate.json'); summary_path=folder/(stem+'_summary.csv'); diag_path=folder/(stem+'_scalar_diagnostics.csv')
    if not gate_path.exists():
        if summary_path.exists() or diag_path.exists():
            return 'integrity_fail','No gate but other fit output exists',None,None
        return 'incomplete','Fit gate absent',None,None
    try:
        gate=read_json(gate_path)
        for name,value in (('model',model),('scenario',scenario),('rep',rep),('freeze_hash',expected_freeze)):
            assert_both(gate[name]==value, f'Gate {name} mismatch')
        assert_both(gate['status']=='COMPLETE', 'Gate status is not COMPLETE')
        for key,val in SAMPLES.items():
            assert_both(gate['sampler'][key]==val, f'Original sampler setting {key} mismatch')
        rel=f'data/{scenario}/{stem}.npz'
        assert_both(rel in freeze_hashes, f'Missing frozen data checksum {rel}')
        assert_both(gate['source_dataset_sha256']==freeze_hashes[rel], 'Gate dataset checksum mismatch')
        file_data=freeze_root/rel
        assert_both(file_data.exists() and sha256(file_data)==freeze_hashes[rel], 'Frozen dataset altered/missing')
        assert_both(summary_path.is_file() and diag_path.is_file(), 'Summary or scalar diagnostics missing')
        assert_both(sha256(summary_path)==gate['summary_sha256'], 'Summary SHA mismatch')
        assert_both(sha256(diag_path)==gate['diagnostics_sha256'], 'Diagnostics SHA mismatch')
        if verify_chains:
            chain_hashes=gate['chain_samples_sha256']
            assert_both(len(chain_hashes)==4 and len(gate['chain_seed_list'])==4, 'Incomplete provenance for 4 chains')
            for j,digest in enumerate(chain_hashes):
                chain=chain_root/f'fit_{model}'/scenario/stem/f'chain_{j}'/'chain_samples.npz'
                assert_both(chain.exists() and sha256(chain)==digest, f'Chain {j} SHA mismatch')
        div=int(gate['divergences'])
        rhat=float(gate['max_rhat_scalar'])
        bulk=float(gate['min_ess_bulk_scalar'])
        tail=float(gate['min_ess_tail_scalar'])
        assert_both(div>=0,'Negative divergences')
        computed=(div==0 and all(map(math.isfinite,(rhat,bulk,tail))) and rhat<=1.01 and bulk>=400 and tail>=400)
        assert_both(computed==gate['computational_ok'], 'Computational gate boolean disagrees with locked rule')
        results=read_csv(summary_path)
        assert_both(len(results)==3 and {r['target'] for r in results}==set(TARGETS),'Wrong summary targets')
        numbers={}
        for r in results:
            target=r['target']
            vals={k:finite_float(r[k],f'{target}:{k}') for k in ('estimate','posterior_sd','ci_low','ci_high','interval_width')}
            assert_both(vals['ci_low']<=vals['ci_high'] and vals['posterior_sd']>=0, 'Invalid interval/SD')
            assert_both(abs(vals['interval_width']-(vals['ci_high']-vals['ci_low']))<1e-9, 'Interval width inconsistent')
            numbers[target]=vals
        return 'complete','',gate,numbers
    except (KeyError, ValueError, OSError, TypeError, IndexError, AssertionError) as exc:
        return 'integrity_fail',str(exc),None,None


def measure(valid, *, expected=100):
    n=len(valid)
    if n==0:
        return {k:None for k in ('bias','rmse','coverage','mean_interval_width','mcse_bias','mcse_coverage','mean_truth','mean_estimate','coverage_missing_lower','coverage_missing_upper')}
    err=[v['estimate']-v['truth'] for v in valid]
    cover=[int(v['ci_low']<=v['truth']<=v['ci_high']) for v in valid]
    p=sum(cover)/n
    return dict(bias=sum(err)/n,
                rmse=math.sqrt(sum(e*e for e in err)/n),
                coverage=p,mean_interval_width=sum(v['interval_width'] for v in valid)/n,
                mcse_bias=statistics.stdev(err)/math.sqrt(n) if n>=2 else None,
                mcse_coverage=math.sqrt(p*(1-p)/n) if n>=2 else None,
                mean_truth=sum(v['truth'] for v in valid)/n,
                mean_estimate=sum(v['estimate'] for v in valid)/n,
                coverage_missing_lower=sum(cover)/expected,
                coverage_missing_upper=(sum(cover)+expected-n)/expected)


def writetable(path, records, columns):
    with open(path,'w',newline='',encoding='utf8') as fh:
        w=csv.DictWriter(fh,fieldnames=columns,extrasaction='ignore')
        w.writeheader(); w.writerows(records)


def evaluation(manifest, catalog, outcome):
    audit=[]; replicates=[]
    for model,scenario,rep in sorted(manifest):
        generator=scenario[:2]
        status,why,gate,estimates=outcome[(model,scenario,rep)]
        audit.append(dict(analysis_group='primary_correct_specification' if generator==model else 'secondary_cross_model',
                          generator=generator,fit_model=model,scenario=scenario,replication=rep,status=status,
                          computational_ok=gate['computational_ok'] if gate is not None else '',
                          integrity_problem=why,
                          divergences=gate['divergences'] if gate else '',
                          max_scalar_rhat=gate['max_rhat_scalar'] if gate else '',
                          min_scalar_ess_bulk=gate['min_ess_bulk_scalar'] if gate else '',
                          min_scalar_ess_tail=gate['min_ess_tail_scalar'] if gate else ''))
        for target in TARGETS:
            truth=catalog[(scenario,rep)][target]
            a=dict(analysis_group='primary_correct_specification' if generator==model else 'secondary_cross_model',
                   generator=generator,fit_model=model,scenario=scenario,replication=rep,
                   n_readers=int(scenario.split('_')[1][1:]),n_slides=int(scenario.split('_')[2][1:]),
                   target=target,truth=truth,status=status,
                   computational_ok=gate['computational_ok'] if gate else '',integrity_problem=why)
            if estimates:
                a.update(estimates[target])
                a['error']=a['estimate']-truth
                a['covered_95']=int(a['ci_low']<=truth<=a['ci_high'])
            replicates.append(a)
    return audit,replicates


def aggregate(audit,rows,pass_only, rules):
    audit_groups=collections.defaultdict(list)
    groups=collections.defaultdict(list)
    for a in audit: audit_groups[(a['fit_model'],a['scenario'])].append(a)
    for r in rows: groups[(r['fit_model'],r['scenario'],r['target'])].append(r)
    output=[]
    for (model,scenario,target),rgroup in sorted(groups.items()):
        group=audit_groups[(model,scenario)]
        n_complete=sum(a['status']=='complete' for a in group)
        n_pass=sum(a['status']=='complete' and a['computational_ok'] is True for a in group)
        n_fail=n_complete-n_pass
        n_missing=sum(a['status']=='incomplete' for a in group)
        n_integrity=sum(a['status']=='integrity_fail' for a in group)
        assert_both(len(rgroup)==len(group)==100,'Denominator not 100')
        valid=[r for r in rgroup if r['status']=='complete' and (not pass_only or r['computational_ok'] is True)]
        measures=measure(valid)
        if pass_only:
            # Frozen missing-output bounds concern truly unavailable/corrupt outputs,
            # NOT conditional diagnostic-pass selection. Preserve the original bounds.
            all_integrity=[r for r in rgroup if r['status']=='complete']
            unconditional_bounds=measure(all_integrity)
            measures['coverage_missing_lower']=unconditional_bounds['coverage_missing_lower']
            measures['coverage_missing_upper']=unconditional_bounds['coverage_missing_upper']
        num=scenario.split('_')
        generator=num[0]
        diags=[a for a in group if a['status']=='complete']
        finite_max=[a['max_scalar_rhat'] for a in diags if math.isfinite(a['max_scalar_rhat'])]
        finite_bulk=[a['min_scalar_ess_bulk'] for a in diags if math.isfinite(a['min_scalar_ess_bulk'])]
        finite_tail=[a['min_scalar_ess_tail'] for a in diags if math.isfinite(a['min_scalar_ess_tail'])]
        row=dict(analysis_group='primary_correct_specification' if generator==model else 'secondary_cross_model',
                 generator=generator,fit_model=model,scenario=scenario,
                 n_readers=int(num[1][1:]),n_slides=int(num[2][1:]),target=target,
                 n_expected=100,n_complete_integrity_verified=n_complete,n_diagnostic_pass=n_pass,n_diagnostic_fail=n_fail,
                 n_incomplete=n_missing,n_integrity_fail=n_integrity,n_used_metrics=len(valid),
                 failure_rate=(n_missing+n_integrity)/100,diagnostic_pass_rate=n_pass/100,
                 divergences_total=sum(a['divergences'] for a in diags) if diags else None,
                 max_scalar_rhat=max(finite_max) if finite_max else None,
                 min_scalar_ess_bulk=min(finite_bulk) if finite_bulk else None,
                 min_scalar_ess_tail=min(finite_tail) if finite_tail else None)
        row.update(measures)
        threshold=rules['scientific_benchmarks']
        row['abs_bias_lt_0p005']=(abs(row['bias'])<threshold['abs_bias_lt']) if row['bias'] is not None else ''
        row['coverage_0p92_to_0p98']=(threshold['coverage_lower_inclusive']<=row['coverage']<=threshold['coverage_upper_inclusive']) if row['coverage'] is not None else ''
        row['failure_rate_lt_0p02']=(row['failure_rate']<threshold['computational_failure_rate_lt'])
        output.append(row)
    assert_both(len(output)==72, f'Expected 72 table cells, got {len(output)}')
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--rules',type=Path,default=None)
    p.add_argument('--skip-chain-rehash',action='store_true',help='Faster re-aggregation; provenance still checked using previously finalized hashes')
    a=p.parse_args()
    root=a.root.resolve()
    freeze=root/'outputs/frozen_data'
    fit_root=root/'outputs/v26_1_short_checkpoint_v1_fits'
    chain_root=root/'outputs/v26_1_short_checkpoint_v1'
    manifest=root/'data/manifests/v26_1_2400_fit_manifest.csv'
    rules_path=(a.rules or (root/'config/v26_1_pre_results_evaluation_rules_2026-10-15.json')).resolve()
    out=(a.out or root/'outputs/v26_1_primary_calibration_evaluation_v1').resolve()
    input_dirs=[root,freeze,fit_root,chain_root]
    assert_both(all(out!=d for d in input_dirs),'Output cannot overwrite an input root')
    assert_both(not any(out==p or out in p.parents for p in (freeze,fit_root,chain_root,manifest,rules_path)), 'Output collides with study input')
    assert_both(not out.exists() or not any(out.iterdir()),'Refusing to overwrite existing nonempty analysis output')
    rules=read_json(rules_path)
    assert_both(rules['protocol_id']=='NuCLS-v26.1-Bayesian-evaluation-pre-results-v1-2026-10-15', 'Wrong prespecified evaluation rules')
    expected=rules['original_frozen_dataset_root_sha256']
    manifest_rows=expected_manifest(manifest)
    print('FROZEN_FIT_MANIFEST_OK',len(manifest_rows),flush=True)
    frozen_hashes=read_freeze_index(freeze,expected)
    catalog=truth_catalog(freeze,frozen_hashes,manifest_rows)
    print('FROZEN_TRUTH_HASHES_OK',len(catalog),flush=True)
    outcome={}
    for n,(key,row) in enumerate(sorted(manifest_rows.items()),1):
        outcome[key]=validate_one(row,fit_root,chain_root,freeze,frozen_hashes,expected,not a.skip_chain_rehash)
        if n%300==0:
            print('FIT_INTEGRITY_PROGRESS',n,'/2400',flush=True)
    audit,replicates=evaluation(manifest_rows,catalog,outcome)
    full=aggregate(audit,replicates,False,rules)
    diagnostic=aggregate(audit,replicates,True,rules)
    nstates=dict(collections.Counter(a['status'] for a in audit))
    n_ok=sum(a['status']=='complete' and a['computational_ok'] is True for a in audit)
    assert_both(len(audit)==2400 and len(replicates)==7200,'Unexpected row counts')
    assert_both(len([x for x in full if x['analysis_group']=='primary_correct_specification'])==36,'Incorrect primary cell count')
    # Create a reproducible, nonoverwriting output only AFTER calculation succeeds.
    out.mkdir(parents=True,exist_ok=True)
    writetable(out/'per_fit_audit_2400.csv',audit,list(audit[0]))
    writetable(out/'per_target_replicates_7200.csv',replicates,
               ['analysis_group','generator','fit_model','scenario','replication','n_readers','n_slides','target','truth',
                'status','computational_ok','integrity_problem','estimate','posterior_sd','ci_low','ci_high','interval_width','error','covered_95'])
    col=list(full[0])
    writetable(out/'bayesian_72_cells_all_completed.csv',full,col)
    writetable(out/'bayesian_72_cells_diagnostic_pass_only.csv',diagnostic,col)
    writetable(out/'primary_36_cells_all_completed.csv',[r for r in full if r['analysis_group']=='primary_correct_specification'],col)
    writetable(out/'primary_36_cells_diagnostic_pass_only.csv',[r for r in diagnostic if r['analysis_group']=='primary_correct_specification'],col)
    selected=[r for r in full if r['analysis_group']=='primary_correct_specification' and (r['n_readers'],r['n_slides']) in ((5,5),(10,20))]
    writetable(out/'main_text_designs_12_cells.csv',selected,col)
    report=dict(protocol=rules['protocol_id'],rules_sha256=sha256(rules_path),fit_manifest_sha256=sha256(manifest),
                frozen_root_sha256=expected,verify_chain_sample_hashes=not a.skip_chain_rehash,
                n_fits=2400,n_primary=1200,n_secondary=1200,n_replication_target_rows=7200,n_result_cells=72,
                status_counts=nstates,n_diagnostic_pass=n_ok,n_diagnostic_fail=len(audit)-sum(1 for x in audit if x['status']!='complete')-n_ok,
                n_primary_diagnostic_pass=sum(x['computational_ok'] is True for x in audit if x['analysis_group']=='primary_correct_specification' and x['status']=='complete'),
                matched_primary_benchmark_cells=dict(n_cells=36,
                    n_abs_bias_pass=sum(x['abs_bias_lt_0p005'] is True for x in full if x['analysis_group']=='primary_correct_specification'),
                    n_coverage_band_pass=sum(x['coverage_0p92_to_0p98'] is True for x in full if x['analysis_group']=='primary_correct_specification')),
                notes=['Original 2026-10-15 analysis lock used; no exclusion of diagnostic failures from primary numerical table.',
                       'Diagnostic-pass-only results are separate descriptive sensitivity summaries.',
                       'Sensitivity reruns at target_accept=0.999 are excluded from every output.'])
    (out/'evaluation_audit.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('EVALUATION_COMPLETED',json.dumps(report,sort_keys=True),flush=True)
    print('PRIMARY_PREVIEW_BEGIN',flush=True)
    for x in selected:
        print('PRIMARY_CELL',x['scenario'],x['target'],'bias',round(x['bias'],6),'coverage',round(x['coverage'],4),
              'rmse',round(x['rmse'],6),'width',round(x['mean_interval_width'],6),'passed',x['n_diagnostic_pass'],'/100',flush=True)
    print('OUTPUT_ROOT',out,flush=True)

if __name__=='__main__':
    try: main()
    except Exception as e:
        print('EVALUATION_FATAL',type(e).__name__,str(e),file=sys.stderr,flush=True)
        sys.exit(1)
