# EIML Experiments

This directory contains **example workflows, scripts, and datasets**
used to demonstrate and validate EIML and SGPR functionality.

## Important Notes

- This directory is **not required** to use the EIML library
- Nothing here is considered a stable API
- Scripts may change between versions
- Datasets are illustrative and system-specific

## Purpose

The contents of this directory are intended for:
- reproducing example results
- testing new ideas
- benchmarking SOAP vs EIML
- internal and exploratory studies

For production use, users should rely on the public CLI:

```bash
eiml featurize
eiml train sgpr
eiml predict sgpr
