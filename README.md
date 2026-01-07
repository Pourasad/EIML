# EIML – Experimentally Informed Machine Learning (v1)

EIML is a research codebase for constructing **physically informed local atomic descriptors for liquids**, designed to bridge **microscopic machine learning representations** and **macroscopic thermodynamic knowledge**.

This repository contains **EIML-v1**, the first stable implementation focusing on **geometric normalization and scale consistency**.

---

## Motivation

Standard geometric descriptors (e.g. SOAP) are defined in **absolute coordinates**, which can limit their transferability across:
- densities
- temperatures
- chemically similar liquids of different molecular size

EIML introduces **experimentally informed scaling rules** inspired by statistical thermodynamics to make descriptors:
- less data-hungry
- more physically interpretable
- better suited for liquid-state learning

---
## EIML-v1: Implemented Features

EIML-v1 extends a SOAP-like framework with:

### 1. Reduced coordinates
All interatomic distances are expressed in reduced form:

r* = r / sigma

where sigma represents a characteristic molecular size.

### 2. Dynamic cutoff
The neighbor cutoff is defined as:

R_cut = k_rcut * sigma

This ensures a consistent number of solvation shells across systems.

### 3. Adaptive Gaussian width
The Gaussian width is defined as:

omega = omega* * sigma

This prevents over-localization or delta-like atomic densities.

These changes preserve the mathematical structure of SOAP while enforcing scale invariance.
---

## What EIML-v1 Is *Not*

- It is **not** claimed to outperform SOAP in all benchmarks.
- It does **not yet include energetic (ε-weighted) density contributions**.
- It is **not a trained force field**.

EIML-v1 is a **descriptor foundation**, intended for integration with kernel methods (GPR, SGPR) and future extensions.

---

## Repository Structure

src/eiml/        Core descriptor implementation
test/            Minimal test cases and example configurations

---

## Usage (Example)

```bash
PYTHONPATH=src python -m eiml.cli --config test/config_eiml.yaml
