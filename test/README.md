# EIML Test & Validation Suite

This directory contains lightweight, reproducible tests for validating and
comparing **EIML (Experimentally Informed Machine Learning)** descriptors against
standard **SOAP**.

Nothing in this directory is imported by the main EIML library.  
These files are intended for regression testing, validation, and demonstration.

---

## Contents

### Input structure
- `water.xyz`  
  Minimal example structure used for descriptor comparison.

---

### Configuration files
- `soap.yaml`  
  Computes standard SOAP descriptors with a fixed cutoff and physical units.

- `eiml.yaml`  
  Computes EIML descriptors, supporting:
  - reduced coordinates
  - dynamic cutoff (`k_rcut × σ`)
  - reduced Gaussian width (`ω*`)
  - optional ε-based channel weighting

To toggle ε-weighting, set in `eiml.yaml`:
```yaml
enable_weighting: true
```
or
```yaml
enable_weighting: false
```

---

### Comparison script
- `compare.py`

This script loads SOAP and EIML feature vectors and reports:
- descriptor dimensionality
- vector norms
- cosine similarity
- elementwise difference statistics

It is used to verify that:
- EIML without ε preserves SOAP geometry (up to smooth rescaling)
- ε-weighting introduces controlled, interpretable chemical bias

---

## How to run

From the project root directory:

```bash
# Run SOAP
PYTHONPATH=src python -m eiml.cli --config test/soap.yaml

# Run EIML
PYTHONPATH=src python -m eiml.cli --config test/eiml.yaml

# Compare descriptors
python test/compare.py
```

Generated `.npy` feature files are not tracked by git.
