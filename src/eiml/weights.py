# src/eiml/weights.py
from __future__ import annotations

from typing import Dict, Sequence
import numpy as np


def _pair_weight(w: Dict[str, float], a: str, b: str) -> float:
    """
    If neighbor density of species s is scaled by w_s, then power spectrum block (a,b)
    scales by w_a * w_b.
    """
    return float(w.get(a, 1.0)) * float(w.get(b, 1.0))


def apply_species_pair_scaling(
    soap_obj,
    X: np.ndarray,
    species: Sequence[str],
    species_weights: Dict[str, float],
) -> np.ndarray:
    """
    Post-hoc scaling of SOAP power spectrum blocks by species-pair weights.

    Parameters
    ----------
    soap_obj : DScribe SOAP object
        Must provide get_location((a,b)) (or get_location(a,b) in older versions).
    X : np.ndarray
        SOAP output features. Shape can be (n_features,) or (n_centers, n_features).
    species : Sequence[str]
        Species list in the same convention used to create the DScribe SOAP object.
    species_weights : Dict[str, float]
        Per-species weights w_s.

    Returns
    -------
    np.ndarray
        Scaled feature array with same shape as X.
    """
    if X.ndim == 1:
        Y = X.reshape(1, -1)
        squeeze_back = True
    elif X.ndim == 2:
        Y = X
        squeeze_back = False
    else:
        raise ValueError(f"X must be 1D or 2D, got shape {X.shape}")

    Y2 = np.array(Y, copy=True)

    # DScribe provides blocks for (a,b); for safety, we scale all pairs present in `species`.
    for a in species:
        for b in species:
            scale = _pair_weight(species_weights, a, b)

            # Locate slice for this pair in DScribe output
            try:
                sl = soap_obj.get_location((a, b))
            except TypeError:
                # some DScribe versions expect two args
                sl = soap_obj.get_location(a, b)

            Y2[:, sl] *= scale

    if squeeze_back:
        return Y2.reshape(-1)
    return Y2


def normalize_species_epsilon(
    epsilon: Dict[str, float],
    species: Sequence[str],
    alpha: float = 1.0,
) -> Dict[str, float]:
    """
    Normalize user-provided per-species epsilon into relative channel weights.

    Steps:
      1) Geometric-mean normalization (scale invariant)
      2) Soft damping via exponent alpha in (0, 1]

    Returns
    -------
    Dict[str, float] : normalized weights for each species in `species`.
    """
    if epsilon is None:
        raise ValueError("epsilon must be provided when enable_weighting=True")

    eps = np.array([float(epsilon[s]) for s in species], dtype=float)

    if np.any(eps <= 0.0):
        raise ValueError("All epsilon values must be positive.")

    if not (0.0 < float(alpha) <= 1.0):
        raise ValueError("epsilon_alpha must be in (0, 1].")

    # Geometric mean normalization
    gmean = np.exp(np.mean(np.log(eps)))
    eps_norm = (eps / gmean) ** float(alpha)

    return {s: float(w) for s, w in zip(species, eps_norm)}
