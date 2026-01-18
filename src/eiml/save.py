# src/eiml/save.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple, Union
import numpy as np

Array = np.ndarray
Pair = Tuple[Array, Array]
XOrPair = Union[Array, Pair]


def _strip_known_suffixes(p: Path) -> Path:
    """Remove .npy or .npz suffix if user provided it."""
    if p.suffix.lower() in (".npy", ".npz"):
        return p.with_suffix("")
    return p


def _normalize_feature_array(
    X: np.ndarray,
    *,
    flatten_atoms: bool = True,
    pool: Optional[str] = None,
) -> np.ndarray:
    """
    Normalize descriptor output into a 2D feature matrix (N, D).

    Supported input shapes:
      - (N, D)                         -> unchanged
      - (n_frames, n_atoms, D)         -> either flatten or pool over atoms
      - (D,)                           -> treated as (1, D)

    Behavior for 3D:
      - if pool in {"mean","sum","max"}: reduce over axis=1 -> (n_frames, D)
      - else if flatten_atoms=True:      reshape -> (n_frames*n_atoms, D)
      - else: raise
    """
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if X.ndim == 1:
        X2 = X.reshape(1, -1)

    elif X.ndim == 2:
        X2 = X

    elif X.ndim == 3:
        # (frames, atoms, D)
        pool_kind = None if pool is None else str(pool).lower().strip()
        if pool_kind in ("mean", "avg"):
            X2 = X.mean(axis=1)
        elif pool_kind == "sum":
            X2 = X.sum(axis=1)
        elif pool_kind == "max":
            X2 = X.max(axis=1)
        else:
            if flatten_atoms:
                X2 = X.reshape(-1, X.shape[-1])
            else:
                raise ValueError(
                    f"Got 3D features {X.shape} but flatten_atoms=False and pool=None. "
                    "Set flatten_atoms=True or pool to one of {mean,sum,max}."
                )
    else:
        raise ValueError(f"Unsupported feature array shape {X.shape} (ndim={X.ndim}).")

    # ensure numeric, 2D, finite dtype
    X2 = np.asarray(X2, dtype=np.float64)
    if X2.ndim != 2:
        raise ValueError(f"Internal error: expected 2D array, got shape {X2.shape}")

    return X2


def _normalize_identity(theta: np.ndarray) -> np.ndarray:
    """Normalize identity vector shape."""
    if not isinstance(theta, np.ndarray):
        raise ValueError("Expected a numpy.ndarray for identity array.")

    if theta.ndim == 1:
        return theta

    if theta.ndim == 2 and theta.shape[0] == 1:
        return theta[0]

    raise ValueError(f"Unsupported identity array shape {theta.shape}.")

def save_features(
    *,
    mode: str,
    output_file: str,
    split_identity: bool,
    X_or_pair: Union[np.ndarray, Tuple[np.ndarray, Any]],
    flatten_atoms: bool = True,
    pool: Optional[str] = None,
) -> None:
    """
    Save descriptor features to disk.

    - If split_identity=False: saves a .npy array (N,D)
    - If split_identity=True: saves a .npz with keys:
          X = (N,D)
          theta = identity array/object (if provided)
    """
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    if split_identity:
        if not (isinstance(X_or_pair, (tuple, list)) and len(X_or_pair) == 2):
            raise ValueError("split_identity=True expects X_or_pair=(X_geom, theta_id).")
        X_geom, theta = X_or_pair
        X = _normalize_feature_array(X_geom, flatten_atoms=flatten_atoms, pool=pool)
        np.savez(out.with_suffix(".npz"), X=X, theta=theta, mode=str(mode))
    else:
        X = _normalize_feature_array(X_or_pair, flatten_atoms=flatten_atoms, pool=pool)
        np.save(out, X)
