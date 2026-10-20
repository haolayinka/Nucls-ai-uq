# NuCLS AI UQ

Reproducibility code and publication artifacts for Bayesian uncertainty quantification of breast-cancer pathology AI when reference labels depend on the reader and annotation condition.

## Study aim

The analysis focuses on agreement between a fixed external pathology AI system and a future qualified reader while accounting for reader variability and annotation-condition dependence. NuCLS inferred P-truth labels are not treated as independently verified biological truth.

## Repository contents

- `src/` — scientific analysis code
- `data/manifests/` — frozen simulation scenario definitions
- `outputs/canonical/` — deterministic figure source data
- `outputs/tables/` — manuscript and supplementary tables
- `outputs/figures/` — publication figures
- `config/` — portable configuration and SHA-256 manifest
- `docs/` — analysis and reproducibility documentation
- `provenance/` — preprocessing reconstruction provenance
- `scripts/verify.sh` — deterministic artifact verification
- `checkpoints/` — fixed HoVer-Net checkpoint information

## Main analysis components

The repository includes preprocessing and frame validation, fixed external HoVer-Net scoring, paired-reader descriptive analyses, iMRMC benchmarking, Bayesian M1 and M2 models, M4 residual-dependence sensitivity, prior and missingness sensitivity analyses, identification diagnostics, slide-hierarchy and future-reader sensitivity analyses, known-truth and adversarial simulations, predictive validation, and publication-output code.

The unsuccessful M3 development branch is retained separately under `src/archive/m3_failed_validation/` and is not part of the validated main analysis path.

## Interpretation limits

- Evaluation-minus-Unbiased contrasts are not interpreted as causal effects of AI assistance.
- Near-zero contrasts are not equivalence claims without a prespecified equivalence margin.
- AI-reader agreement is not biological or diagnostic accuracy.
- M2 is not treated as uniquely correct or biologically true.
- Leave-one-slide-out analyses are not interpreted as independent new-patient validation.
- The external HoVer-Net model used here is distinct from the AI assistance used during original NuCLS annotation.

## Data

The original NuCLS data are not redistributed in this repository. See `data/README.md` and `checkpoints/README.md` for requirements.

## v26.1 Bayesian calibration and design support

The manuscript is titled *Bayesian Uncertainty Quantification of Breast-Cancer Pathology AI Agreement with a Future Reader: Calibration and Reference-Process Sensitivity*.

The v26.1 extension evaluates Bayesian agreement with a future reader under
known M1/M2 generating mechanisms. The design fixes 12 scenarios (5 or 10
readers; 5, 10, or 20 nominal slides), 100 replications per scenario, and the
same 1,200 simulated datasets for both Bayesian fitted models. The primary
comparison uses 1,200 matched-model fits; 1,200 cross-model fits form the
secondary analysis. This extension complements rather than replaces A1–A8.

Posterior computation is assessed using zero divergences, maximum scalar R-hat <= 1.01, and minimum bulk/tail effective sample sizes of
400. Sampling effort may be increased for the **same simulation replicate and
unchanged model** until acceptable numerical diagnostics are obtained or a
persistent computational limitation is documented. Final analysis artifacts
must record sampler settings and convergence status for reproducibility.

These calibration results are conditional on the investigated reference-process
models; they do not establish biological truth, a causal effect of AI assistance,
or calibration under arbitrary misspecification. See
[`docs/v26_1_calibration_and_reproducibility.md`](docs/v26_1_calibration_and_reproducibility.md)
for code, design, and computational documentation.

The repository publishes compact scientific outputs, not individual MCMC chains
or institution-specific computing scripts.

## Verification

Run:

```bash
./scripts/verify.sh
```

The SHA-256 manifest covers deterministic CSV and text publication artifacts. Byte-identical reproduction is not claimed for MCMC serialization or rendered PDF/PNG files across different hardware and software environments.

## Documentation

Start with `docs/START_HERE.md`, `docs/analysis_pipeline.md`, and `docs/reproducibility_contract.md`.

## Final v26.1 primary calibration results

The [final 36-cell Bayesian calibration table](outputs/tables/TableS_v26_1_final_primary_calibration.csv)
reports all three estimands under 12 correctly specified scenarios. All 1,200
primary model fits meet the scalar convergence criteria, using the original
posterior fits for 1,181 datasets and longer converged sampling for the same
remaining 19 datasets (unchanged likelihood, priors, design, and estimands).

Absolute bias is below 0.005 in all 36 cells. Empirical 95% credible-interval
coverage ranges from **92% to 100%** across 100 replications per cell. In
33 cells coverage is between 92% and 98%, and in three it is 99%–100%.
The latter are **above**, not below, the descriptive comparison band; this
band is not a hard pass/fail test. Higher empirical coverage may indicate
conservatism and should be interpreted together with interval widths and
Monte Carlo variability. The supplementary cross-model analysis is distinct
and is not included in this final primary table.

To regenerate the manuscript's Bayesian calibration graphic from this
public table, run `python3 src/publication/make_v26_1_primary_calibration_figure.py
--output Figure5_v26_1_final_primary_Bayesian_calibration.pdf` (requires
NumPy, pandas, and Matplotlib). The generator plots the nominal 95% coverage
line without assigning a failure status to coverage above 98%.
