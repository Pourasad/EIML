# EIML — Experimentally Informed Machine Learning (v0.2)

**EIML** is a research toolkit for constructing **physically informed local atomic descriptors** and **learning materials properties with kernel-based machine learning**, with a focus on liquids and disordered systems.

EIML combines:
- SOAP-based geometric descriptors
- physically motivated normalization and weighting (EIML)
- Sparse Gaussian Process Regression (SGPR) for scalable property learning

The framework is designed to bridge **local atomic environments** and **macroscopic thermodynamic or mechanical properties**, while remaining **transparent, modular, and interpretable**.

---

## What EIML Is (v0.2)

✔ A **descriptor + property-learning toolkit**  
✔ Suitable for learning:
- forces
- pressures
- energies
- other local or global observables  

✔ Works with:
- classical MD data
- ab initio data (e.g. VASP, Quantum ESPRESSO, CP2K)
- arbitrary atomic systems (liquids, solids, clusters)

✔ Designed for:
- systematic comparison of descriptors
- transferability studies
- physics-aware machine learning analysis

---

## What EIML Is *Not*

- ❌ Not (yet) a production-ready ML interatomic potential (MLIP)
- ❌ No direct LAMMPS / ASE calculator interface in v0.2
- ❌ No enforced energy–force consistency

**EIML v0.2 focuses on property learning, not MD deployment.**  
A full MLIP interface is planned for a future version.

---

## Motivation

Standard local descriptors (e.g. SOAP) are defined in **absolute coordinates**, which limits transferability across:

- density changes
- temperature variations
- chemically similar systems with different length scales

EIML introduces **experimentally informed normalization principles**, inspired by statistical thermodynamics, to make descriptors:

- scale-consistent
- more transferable
- physically interpretable

without changing the underlying SOAP formalism.

---

## EIML Descriptor Extensions

EIML extends SOAP with **orthogonal, modular enhancements**.

---

### 1. Reduced Coordinates (Scale Invariance)

All interatomic distances are expressed in **reduced form**:

r_reduced = r / sigma

This enforces a **consistent number of coordination shells** across:
- different densities
- different molecular sizes
- different thermodynamic states

---

### 3. Adaptive Gaussian Width

The atomic density smearing width scales as:

omega = omega_ref * sigma

This ensures comparable smoothness of atomic density fields across systems with different length scales.

---

### 4. ε-Based Channel Weighting (Optional)

Optional per-species channel weighting allows encoding **relative chemical importance**:

- user provides raw importance values `epsilon_s`
- internal normalization enforces:
  - mean channel weight = 1
  - controlled influence via damping exponent `alpha` in (0, 1]

This mechanism is:
- off by default
- fully backward compatible with geometry-only SOAP / EIML
- independent of the physical origin of `epsilon_s` (SAFT, LJ, empirical, etc.)

---

## SGPR: Property Learning Engine (v0.2)

EIML v0.2 includes a **native Sparse Gaussian Process Regression (SGPR)** implementation:

- inducing-point approximation
- supports **multi-output regression** (e.g. vector forces)
- CPU-based NumPy implementation
- scalable to tens of thousands of local environments

Supported inducing strategies:
- random
- k-means
- farthest-point sampling (FPS)

SGPR is intended for **analysis, benchmarking, and property learning**, not yet for large-scale production MD.

---

## Typical Workflow

### 1. Featurization

```bash
eiml featurize --config path/to/config.yaml
```

Computes SOAP or EIML descriptors and writes .npy feature arrays.

### 2. Training (SGPR)

```bash
eiml train sgpr \
  --X X.npy \
  --y y.npy \
  --outdir model_dir \
  --inducing fps \
  --M 1024 \
  --standardize
```

Trains an SGPR model and saves:
- model parameters
- standardization scalers
- training and test metrics

### Prediction

```bash
eiml predict sgpr \
  --modeldir model_dir \
  --X X_new.npy \
  --outdir preds
```

Outputs predictions and approximate uncertainties.


---
## Repository Structure

```bash
src/eiml/
  descriptor.py      Descriptor implementation
  models/sgpr.py     SGPR model
  cli.py             Unified command-line interface
  config.py          YAML parsing
  save.py            Feature and model I/O

experiments/
  Example workflows, datasets, and scripts
```

The experiments/ directory contains examples only and is not required for library usage.

---
## Current Version

v0.2.0 — Descriptor + SGPR property-learning toolkit

---

## Roadmap

- v0.3: improved uncertainty estimates, force–energy consistency
- v0.4: ASE calculator interface
- v1.0: MLIP-ready EIML potential


