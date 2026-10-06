#!/usr/bin/env python3
from __future__ import annotations
from repo_paths import repo_root
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

VARS=["beta_ai","sd_fov","sd_anchor","sd_reader_fov","sd_reader_anchor"]
DISPLAY={"beta_ai":"lambda_A","sd_fov":"sigma_f","sd_anchor":"sigma_a",
         "sd_reader_fov":"sigma_rf","sd_reader_anchor":"sigma_ra"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",type=Path,default=repo_root())
    ap.add_argument("--outdir",type=Path,default=None)
    args=ap.parse_args()
    root=args.project_root
    nc=root/"outputs"/"v23_1_realdata_primary"/"v23_1_realdata_primary_idata.nc"
    if not nc.exists(): raise FileNotFoundError(nc)
    import arviz as az
    idata=az.from_netcdf(nc)
    p=idata.posterior
    vals={}
    for v in VARS:
        x=np.asarray(p[v].values).reshape(-1)
        vals[DISPLAY[v]]=x
    df=pd.DataFrame(vals)
    corr=df.corr()
    out=args.outdir or root/"outputs"/"v25_1_identification_diagnostics"
    out.mkdir(parents=True,exist_ok=True)
    corr.to_csv(out/"posterior_correlation_key_identification_parameters.csv")

    # Pairwise summaries emphasize ridge-like relationships without claiming identification.
    rows=[]
    for i,a in enumerate(df.columns):
        for b in df.columns[i+1:]:
            rows.append({"parameter_1":a,"parameter_2":b,
                         "posterior_correlation":float(corr.loc[a,b])})
    pd.DataFrame(rows).sort_values("posterior_correlation",key=lambda s:s.abs(),ascending=False).to_csv(
        out/"posterior_pairwise_correlations_ranked.csv",index=False)

    # Pair plot as supplement artifact.
    import matplotlib.pyplot as plt
    pd.plotting.scatter_matrix(df.sample(min(5000,len(df)),random_state=251),figsize=(11,11),
                               diagonal="hist",alpha=0.15)
    plt.suptitle("M2 posterior dependence among key identification parameters",y=0.995)
    plt.tight_layout()
    plt.savefig(out/"posterior_pairplot_key_identification_parameters.png",dpi=180,bbox_inches="tight")
    plt.close("all")

    audit={
        "status":"V25_1_IDENTIFICATION_DIAGNOSTICS_COMPLETE",
        "parameters":list(df.columns),
        "n_posterior_draws":len(df),
        "max_abs_offdiagonal_correlation":float(
            np.abs(corr.to_numpy()-np.eye(len(corr))).max()
        ),
        "interpretation_guardrail":"Posterior concentration or weak pairwise correlation does not prove latent-factor identification."
    }
    (out/"audit.json").write_text(json.dumps(audit,indent=2))
    print("V25_1_IDENTIFICATION_DIAGNOSTICS_COMPLETE")
    print(json.dumps(audit,indent=2))

if __name__=="__main__":
    main()
