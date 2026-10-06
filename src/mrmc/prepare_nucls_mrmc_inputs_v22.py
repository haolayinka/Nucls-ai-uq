#!/usr/bin/env python3
from __future__ import annotations
from repo_paths import repo_root
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

READERS=["SP.3","JP.1","JP.2","JP.5","JP.6"]
ROOT=repo_root()
OUT=ROOT/'outputs'/'mrmc_benchmark_v22'
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'scripts'))
import nucls_v21_1_realdata_fit as core

rl=pd.read_csv(ROOT/'data/manifests/nucls_v21_1_real_reader_anchor_long.csv',low_memory=False)
ai,disc=core.discover_ai_vector(ROOT,rl)
val=core.paired_validation(rl,ai)
expected={'paired_n':5334,'paired_agree_U':2884,'paired_agree_E':2918}
for k,v in expected.items():
    if int(val.get(k,-1)) != v:
        raise RuntimeError(f'V17_VALIDATION_FAILED {k}: got {val.get(k)}, expected {v}')

z=rl.merge(ai.rename('ai_matched'),left_on='anchor_id',right_index=True,validate='many_to_one')
z=z[z.reader.isin(READERS)].copy()
z['agreement']=np.where(z.observed==1,(z.reader_detected.astype('Int64')==z.ai_matched).astype('Int64'),pd.NA)

# FDA/iMRMC uStat input: one row per observed reader x anchor x condition; the score itself is binary agreement.
full=z[z.observed==1][['reader','anchor_id','condition','agreement','slide','fov_id']].copy()
full=full.rename(columns={'reader':'readerID','anchor_id':'caseID','condition':'modalityID','agreement':'score'})
full['score']=full.score.astype(int)
full.to_csv(OUT/'nucls_mrmc_five_reader_1144_incomplete.csv',index=False)

# Fully crossed core: every retained reader observed each retained anchor under both conditions.
obs=z.pivot_table(index='anchor_id',columns=['reader','condition'],values='observed',aggfunc='first')
need=pd.MultiIndex.from_product([READERS,['U','E']])
obs=obs.reindex(columns=need)
core_ids=obs.index[(obs==1).all(axis=1)]
core_df=full[full.caseID.isin(core_ids)].copy()
core_df.to_csv(OUT/'nucls_mrmc_five_reader_679_fully_crossed.csv',index=False)

# Audits.
full_anchor_meta=z[['anchor_id','slide','fov_id']].drop_duplicates()
core_meta=full_anchor_meta[full_anchor_meta.anchor_id.isin(core_ids)]
summary={
 'status':'V22_MRMC_INPUT_PREP_COMPLETE',
 'readers':READERS,
 'v17_validation':val,
 'ai_discovery':disc,
 'full':{
   'n_rows':int(len(full)),
   'n_anchors':int(full.caseID.nunique()),
   'n_readers':int(full.readerID.nunique()),
   'n_slides':int(full.slide.nunique()),
   'by_condition':full.groupby('modalityID').agg(n=('score','size'),agreements=('score','sum'),mean=('score','mean')).reset_index().to_dict('records')
 },
 'core':{
   'n_rows':int(len(core_df)),
   'n_anchors':int(core_df.caseID.nunique()),
   'n_readers':int(core_df.readerID.nunique()),
   'n_slides':int(core_df.slide.nunique()),
   'slides':sorted(core_df.slide.unique().tolist()),
   'by_condition':core_df.groupby('modalityID').agg(n=('score','size'),agreements=('score','sum'),mean=('score','mean')).reset_index().to_dict('records')
 }
}
if summary['full']['n_anchors'] != 1144: raise RuntimeError(summary)
if summary['core']['n_anchors'] != 679: raise RuntimeError(summary)
if summary['core']['n_rows'] != 679*5*2: raise RuntimeError(summary)
with open(OUT/'nucls_mrmc_input_audit_v22.json','w') as f: json.dump(summary,f,indent=2)
print('V22_MRMC_INPUT_PREP_COMPLETE')
print(json.dumps(summary,indent=2))
