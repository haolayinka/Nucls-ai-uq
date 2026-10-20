#!/usr/bin/env python3
"""Make the final v26.1 primary Bayesian calibration graphic from one public CSV.

Portable scientific visualization; no cluster/scheduler configuration is required.
This uses *only* the 36 precomputed final generator-matched calibration cells.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGETS = [
    ("theta_U_selected", r"$\theta_U^S$", "#3775a6", "o"),
    ("theta_E_selected", r"$\theta_E^S$", "#be5b2f", "s"),
    ("delta_E_minus_U_selected", r"$\Delta^S$", "#3a9272", "^"),
]
SCENARIOS = [f"{gen}_R{r}_S{sn}" for gen in ("M1", "M2")
             for r in (5, 10) for sn in (5, 10, 20)]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(__file__).resolve().parents[2] / "outputs" / "tables" / "TableS_v26_1_final_primary_calibration.csv")
    parser.add_argument("--output", type=Path, default=Path("Figure5_v26_1_final_primary_Bayesian_calibration.pdf"))
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    assert len(df) == 36 and df.scenario.nunique() == 12 and (df.n_diagnostic_pass == 100).all(), "Unexpected final primary input"
    rows = {(r.scenario, r.target): r for r in df.itertuples()}
    assert set(rows) == {(s,t[0]) for s in SCENARIOS for t in TARGETS}, "Missing/duplicate calibration cells"
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12.5,7.4), sharex=True,
         gridspec_kw={"height_ratios":[1,1.13],"hspace":0.045})
    x = np.arange(12)
    for target,label,color,mark in TARGETS:
        bias = np.array([rows[s,target].bias*100 for s in SCENARIOS])
        cover = np.array([rows[s,target].coverage*100 for s in SCENARIOS])
        mc = np.array([rows[s,target].mcse_coverage*100 for s in SCENARIOS])
        top.plot(x,bias,color=color,marker=mark,linewidth=1.4,markersize=5,label=label)
        bottom.errorbar(x,cover,yerr=mc,fmt='-'+mark,color=color,lw=1.1,
                        markersize=5,capsize=3,label=label)
    top.axhline(0,ls=':',lw=0.9,color='0.4')
    bottom.axhline(95,color='0.30',ls='--',lw=1)  # Nominal target, not a pass/fail boundary
    for ax in (top,bottom):
        ax.axvline(5.5,color='0.65',ls='--',lw=1)
        ax.grid(axis='y',alpha=.21)
        ax.set_xlim(-.6,11.6)
    top.set_ylabel('Bias (percentage points)')
    bottom.set_ylabel('95% interval coverage (%)')
    bottom.set_xlabel('Number of readers / nominal slides (five-slide template repeated)')
    bottom.set_ylim(88.5,102.7)
    top.set_ylim(-.16,.087)
    bottom.set_xticks(x,[s.split('_R')[1].replace('_S','/') for s in SCENARIOS])
    top.text(2.5,1.014,'M1 generator and M1 fitted',ha='center',va='bottom',transform=top.get_xaxis_transform(),fontsize=11)
    top.text(8.5,1.014,'M2 generator and M2 fitted',ha='center',va='bottom',transform=top.get_xaxis_transform(),fontsize=11)
    top.legend(loc='lower right',ncol=3,frameon=False)
    fig.suptitle('Correctly specified Bayesian inference: final 100 replications per scenario',y=.99,fontsize=13.2)
    fig.subplots_adjust(top=.91,bottom=.115,left=.095,right=.98)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,bbox_inches='tight')
    plt.close(fig)
    print('FIGURE_CREATED',args.output)

if __name__ == '__main__':
    main()
