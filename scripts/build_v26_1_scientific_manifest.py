#!/usr/bin/env python3
"""Build a portable scientific fit manifest (no scheduling/shard metadata).

Not byte-identical to the original execution manifest; scientific dataset
indices, generators and fitted-model assignments match the frozen design.
"""
import argparse
import csv
from pathlib import Path

SCENARIOS = [f"{gen}_R{r}_S{s}" for gen in ("M1", "M2") for r in (5, 10) for s in (5, 10, 20)]


def create(path: Path):
    columns = ['unit_id','fit_model','generator','scenario','readers','slides',
               'replication','definitive_index','original_index']
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError('Refusing to overwrite existing manifest: '+str(path))
    rows=[]
    for scenario_idx, scenario in enumerate(SCENARIOS):
        generator, r_part, s_part=scenario.split('_')
        for rep in range(100):
            for fit_model in ('M1','M2'):
                rows.append({'unit_id':f'fit_{fit_model}/{scenario}/rep_{rep:03d}',
                             'fit_model':fit_model,'generator':generator,
                             'scenario':scenario,'readers':int(r_part[1:]),
                             'slides':int(s_part[1:]),'replication':rep,
                             'definitive_index':scenario_idx*100+rep,
                             'original_index':scenario_idx*200+rep})
    with path.open('w',newline='',encoding='utf8') as handle:
        writer=csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader();writer.writerows(rows)
    assert len(rows)==2400 and len({r['unit_id'] for r in rows})==2400
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args()
    result=create(args.out)
    print('SCIENTIFIC_MANIFEST_OK',len(result))
