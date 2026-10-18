
# NuCLS AI-UQ canonical dependency graph



## Main reproducibility path



Official NuCLS source data

→ `src/preprocessing/01_build_selected_frame.py`

→ `src/preprocessing/02_build_blinded_manifest.py`

→ `src/preprocessing/03_validate_source_registration.py`

→ HoVer-Net v15 inference

→ restricted detection scoring v16.1

→ descriptive analyses

→ iMRMC benchmark

→ M1 reader-only model

→ M2 shared-detectability working model

→ M4 residual-dependence sensitivity

→ additional sensitivity analyses

→ adversarial simulations

→ predictive validation

→ publication tables and figures



## Reproducibility interpretation



- Deterministic preprocessing and frozen artifacts are verified with exact hashes where feasible.

- Stochastic Bayesian outputs are verified by numerical/scientific invariants unless exact runtime identity supports byte-level reproduction.

- NuCLS P-truth is an inferred multirater reference, not biological truth.

- Evaluation-minus-Unbiased comparisons are descriptive, not causal assistance effects.

- LOSO analyses are slide-composition sensitivity, not new-patient validation.

## v26.1 correct-specification calibration branch

A separate additional simulation branch uses 12 M1/M2 generating scenarios
(5 or 10 readers; 5, 10, or 20 nominal slides), with 100 preselected
replications each. For each of 1,200 datasets both M1 and M2 are fitted.
The 1,200 matched-generator fits define the frozen primary calibration
analysis; the 1,200 cross-model fits are secondary. The new branch proceeds
through the v26.1 generator, frozen dataset checks, Bayesian inference,
four-chain finalization, and replication-specific known-truth evaluation.
It supplements the existing A1–A8 misspecification program. See
`docs/v26_1_calibration_and_reproducibility.md` and
`src/simulations/v26_1_*.py`. No institution-specific scheduler is required
by the public documentation.

