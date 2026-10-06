# Start here

This repository contains the public reproducibility materials for the NuCLS pathology-AI uncertainty-quantification study.

## Recommended reading order

1. Read `README.md` for the scientific scope and interpretation limits.
2. Read `docs/analysis_pipeline.md` for the analysis flow and dependencies.
3. Read `docs/reproducibility_contract.md` for what is and is not expected to reproduce byte-for-byte.
4. Read `data/README.md` and `checkpoints/README.md` before running analyses that require external NuCLS data or the fixed HoVer-Net checkpoint.

## Reproducibility principle

The public repository contains portable scientific code and compact canonical publication artifacts. Institution-specific scheduler configuration, local filesystem paths, and cluster-administration details are intentionally excluded.

Deterministic CSV, JSON, text, and manifest artifacts may be verified by SHA-256 when a corresponding public hash manifest is provided. MCMC serialization and rendered PDF/PNG files are not assumed to be byte-identical across hardware and software environments.
