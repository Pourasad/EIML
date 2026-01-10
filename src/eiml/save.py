from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union
import numpy as np

Array = np.ndarray
Pair = Tuple[Array, Array]
XOrPair = Union[Array, Pair]


def _strip_known_suffixes(p: Path) -> Path:
    """Remove .npy or .npz suffix if user provided it."""
    if p.suffix.lower() in (".npy", ".npz"):
        return p.with_suffix("")
    return p


def _normalize_feature_array(X: np.ndarray) -> np.ndarray:
    """
    Normalize descriptor array shapes into a consistent representation.

    Accept:
      - (nfeat,)
      - (1, nfeat)
      - (N, nfeat)

    Return:
      - (nfeat,) for single vectors
      - (N, nfeat) for multiple entries
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("Expected a numpy.ndarray for feature array.")

    if X.ndim == 1:
        return X

    if X.ndim == 2 and X.shape[0] == 1:
        return X[0]

    if X.ndim == 2:
        return X

    raise ValueError(f"Unsupported feature array shape {X.shape}.")


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
    X_or_pair: XOrPair,
) -> str:
    """
    Save computed descriptors to disk.

    Rules
    -----
    - split_identity == False  -> save .npy
    - split_identity == True   -> save .npz with keys:
          X_geom, theta_id
    """
    base = _strip_known_suffixes(Path(output_file)).expanduser().resolve()
    base.parent.mkdir(parents=True, exist_ok=True)

    if split_identity:
        if not isinstance(X_or_pair, tuple) or len(X_or_pair) != 2:
            raise ValueError("split_identity=True expects (X_geom, theta_id).")

        X_geom, theta_id = X_or_pair
        X_geom = _normalize_feature_array(X_geom)
        theta_id = _normalize_identity(theta_id)

        out_path = str(base.with_suffix(".npz"))
        np.savez(out_path, X_geom=X_geom, theta_id=theta_id, mode=str(mode))
        print(f"Saved features to {out_path}")
        return out_path

    # No identity splitting
    if not isinstance(X_or_pair, np.ndarray):
        raise ValueError("split_identity=False expects a numpy.ndarray.")

    X = _normalize_feature_array(X_or_pair)
    out_path = str(base.with_suffix(".npy"))
    np.save(out_path, X)
    print(f"Saved features to {out_path}")
    return out_path
