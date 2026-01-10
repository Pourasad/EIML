# EIML – Experimentally Informed Machine Learning (v1.1)

**EIML** is a research codebase for constructing **physically informed local atomic descriptors for liquids**, designed to bridge **microscopic machine-learning representations** and **macroscopic thermodynamic insight**.

This repository currently hosts **EIML-v1.1**, which extends SOAP with **scale-consistent reduced geometry** and **optional energy-aware channel weighting**, while remaining fully compatible with kernel methods such as **GPR / SGPR**.

---

## Motivation

Standard geometric descriptors (e.g. SOAP) are defined in **absolute coordinates**, which limits transferability across:

- density changes  
- temperature variations  
- chemically similar liquids with different molecular size  

EIML introduces **experimentally informed normalization principles**, inspired by statistical thermodynamics, to make descriptors:

- more transferable  
- more physically interpretable  
- better suited for liquid-state learning  

without changing the underlying SOAP formalism.

---

## EIML-v1.1: Implemented Features

EIML-v1.1 extends a SOAP-like framework with the following **orthogonal, modular enhancements**:

---

### 1. Reduced Coordinates (Scale Invariance)

All interatomic distances are expressed in reduced form:

\[
r^* = \frac{r}{\sigma}
\]

where \(\sigma\) is a characteristic molecular or segment size.

This removes explicit length-scale dependence from the descriptor geometry.

---

### 2. Dynamic Cutoff (Shell Consistency)

The neighbor cutoff is defined as:

\[
R_{\text{cut}} = k_{\text{rcut}} \cdot \sigma
\]

This enforces a **consistent number of solvation shells** across systems with different molecular sizes or densities.

---

### 3. Adaptive Gaussian Width

The atomic density smearing width is defined as:

\[
\omega = \omega^* \cdot \sigma
\]

This prevents over-localization and ensures comparable density smoothness across scales.

---

### 4. ε-Based Channel Weighting (Optional, v1.1)

EIML-v1.1 introduces **optional per-species channel weighting**, designed to encode relative chemical importance **without dominating the power spectrum**.

- User provides raw per-species importance values \(\epsilon_s\)
- Internal normalization ensures:
  - mean weight = 1  
  - controlled influence via damping exponent \(\alpha \in (0, 1]\)

This mechanism is:
- **off by default**
- fully backward compatible with geometry-only EIML
- independent of the source of \(\epsilon\) (SAFT, LJ, empirical, etc.)

---

## What EIML Is *Not*

- ❌ Not claimed to universally outperform SOAP  
- ❌ Not a force field or potential  
- ❌ Not tied to a specific thermodynamic model  

EIML is a **descriptor framework**, intended for:
- kernel methods (GPR, SGPR)
- systematic transferability studies
- physically interpretable ML for liquids

---

## Repository Structure

- src/eiml/        Core descriptor implementation
- test/            Minimal validation & comparison tests

The `test/` directory contains:
- SOAP vs EIML comparisons
- cosine similarity and norm diagnostics
- example YAML configurations

Nothing in `test/` is imported by the main library.

---

## Usage Example

```bash
PYTHONPATH=src python -m eiml.cli --config test/soap.yaml
PYTHONPATH=src python -m eiml.cli --config test/eiml.yaml
