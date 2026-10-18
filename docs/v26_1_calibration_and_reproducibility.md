# NuCLS v26.1 — Bayesian calibration and reference-process sensitivity

This document describes the v26.1 additional correct-specification calibration and design-support simulation. It supplements the existing A1–A8 adversarial misspecification study; it does **not** replace that study.

## Estimands and scientific scope

For a fixed external AI vector and the selected anchor frame, the agreement targets are
`theta_U_selected`, `theta_E_selected`, and `delta_E_minus_U_selected` (Evaluation minus Unbiased), integrated over a future qualified reader under the selected reference-process model. This is not biological truth, diagnostic accuracy, a causal effect of assistance, or new-patient validation.

## Frozen experimental design

- Generators: M1 reader-only and M2 shared-detectability.
- Factors: 5 or 10 readers and 5, 10, or 20 nominal slides, constructed by repeating the five-slide template 1, 2, or 4 times, giving 12 generating scenarios. Larger support is not evidence from newly sampled independent patients.
- The generator's original frozen universe comprises 200 replications per scenario (2,400 generated datasets). The October 12 *pre-results amendment* designated original replications `000`–`099` as definitive: 1,200 datasets. Replications `100`–`199` are ineligible for replacement.
- M1 and M2 were fitted to every definitive dataset (2,400 Bayesian fits). Primary analysis: 1,200 fits whose fitted model matches the generating model, 36 scenario–estimand cells. Secondary cross-model analysis: 1,200 fits, 36 separate cells.
- Initial sampling configuration: 4 independent chains, 2,000 tuning transitions plus 4,000 posterior draws per chain, target acceptance 0.99. Final convergence is assessed using zero divergences, max scalar R-hat <= 1.01, and min bulk/tail ESS >= 400; computational sampling length may be extended without changing the scientific design.
- Generating truths are **replication specific**, not one shared truth for a scenario. Intervals are equal-tailed 95% posterior intervals; posterior contrasts are computed draw by draw.

## Computational convergence and final-estimate reproducibility

Numerical convergence is evaluated against the **same diagnostic criteria**
for every fit: zero divergences, maximum scalar R-hat <= 1.01,
and minimum scalar bulk and tail ESS >= 400. Four independent chains are used.
The 2,000 tuning and 4,000 posterior draws per chain (`target_accept=0.99`)
in the dated October 15 configuration document the **initial sampling setup**,
not a scientific requirement to stop at that iteration count.

When necessary, additional tuning/posterior draws or sampler-control adjustments
may be used for the **same already-generated dataset, fitted model, and target**.
The simulation replication identifiers (000–099), likelihood, priors,
generating truths, and performance criteria remain fixed. Increasing sampling
effort may improve numerical accuracy but does not guarantee the convergence
gate will pass. Persistent failures must be retained in the computational
accounting rather than replaced with other simulated datasets.

For reproducibility, the final per-fit computational record should identify
the fit, random-seed/chain initialization policy, actual tuning and sampling
settings, diagnostic results, and which converged posterior output contributes
to published summaries. Previous numerical outputs should remain traceable.
This is distinct from changing a scientific estimand, model, simulated data,
or acceptance threshold.

The dated pre-results evaluation configuration is retained unchanged as
provenance. A final-run configuration and compact computational accounting
will document the settings actually used for publication.

## Portable source files

- `src/simulations/v26_1_models.py`: selected-frame data encoding, M1/M2 models and target functions.
- `src/simulations/v26_1_generate_correct_spec.py`: exact frozen generative mechanisms; requires an external, validated source-template CSV.
- `src/simulations/v26_1_checkpoint_chain_worker.py`: original scientific single-chain NUTS checkpointing and estimand calculation. Four independent chains complete one fit. Generic signals/checkpoints are not scheduler-specific.
- `src/simulations/v26_1_finalize_checkpoint_fits.py`: original four-chain result assembly and scalar diagnostics.
- `src/simulations/v26_1_evaluate_frozen_calibration.py`: frozen read-only operating-characteristic calculation. The public copy changes only *default location assumptions*, not the numerical evaluation functions.
- `scripts/build_v26_1_scientific_manifest.py`: reconstructs scientific fit units and original generator indices **without** any cluster scheduling or partition assignments.

External prerequisites: official NuCLS raw data (not redistributed), the validated selected-frame source CSV, sufficient memory, and a compatible Python/PyMC/ArviZ/scientific stack. The generator stores a 2,400-dataset frozen universe; the manifest selects only the original first 100 per scenario. The frozen SHA-256 root hash is recorded in `config/v26_1_expected_freeze_hash.txt`. The public reconstructed manifest has scientific-equivalent membership but need not have the identical byte hash of the original execution manifest.

This repository intentionally excludes institutional paths, scheduler scripts/logs, raw annotations and images, MCMC checkpoints, and bulk replication-level outputs. Source code publication does not imply that large-scale stochastic runs are bitwise reproducible on other hardware.

## Publication outputs

The planned public result addition is one compact 72-cell Bayesian
operating-characteristics table covering the 36 primary and 36 secondary
scenario–estimand cells, plus only the compact computational metadata required
to identify the final posterior runs. A publication figure can be included if
needed for the article and not already reproducibly generated from the table.
Full replication records, MCMC chains, and machine-specific processing files
are not distributed in Git.
