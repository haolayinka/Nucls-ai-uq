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

## Verification

Run:

```bash
./scripts/verify.sh
```

The SHA-256 manifest covers deterministic CSV and text publication artifacts. Byte-identical reproduction is not claimed for MCMC serialization or rendered PDF/PNG files across different hardware and software environments.

## Documentation

Start with `docs/START_HERE.md`, `docs/analysis_pipeline.md`, and `docs/reproducibility_contract.md`.
