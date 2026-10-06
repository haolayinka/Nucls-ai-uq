
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

