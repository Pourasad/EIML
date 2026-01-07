"""
eiml/save.py

Centralized, strict saving logic for SOAP / EIML descriptors.

Rules:
- split_identity == False  -> always save .npy
- split_identity == True   -> always save .npz with keys:
      X_geom, theta_id
- mode == "soap"           -> split_identity is ignored (warning handled upstream)
"""

from __future__ import annotations
from typing import Tuple, Union
import numpy as np


def _strip_known_suffixes(path: str) -> str:
    """
    Remove known numpy suffixes to avoid file.npy.npy or file.npz.npy.
    """
    for ext in (".npy", ".npz"):
        if path.endswith(ext):
            return path[: -len(ext)]
    return path


def _normalize_shape(X: np.ndarray) -> np.ndarray:
    """
    Normalize descriptor shape for saving.

    - If X is (1, nfeat), convert to (nfeat,)
    - Otherwise leave unchanged (e.g. (nfeat,) or (nframes, nfeat))
    """
    X = np.asarray(X)
    if X.ndim == 2 and X.shape[0] == 1:
        return X[0]
    return X


def save_features(
    mode: str,
    output_file: str,
    split_identity_requested: bool,
    X_or_pair: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
):
    """
    Save features according to EIML-v1 rules.

    Parameters
    ----------
    mode : str
        "soap" or "eiml"
    output_file : str
        Base output path (extension will be enforced)
    split_identity_requested : bool
        Whether user requested geometry + identity split
    X_or_pair : ndarray or (ndarray, ndarray)
        Descriptor output
    """
    base = _strip_known_suffixes(output_file)

    # ------------------------
    # Split geometry + identity
    # ------------------------
    if split_identity_requested:
        if not isinstance(X_or_pair, tuple) or len(X_or_pair) != 2:
            raise ValueError(
                "save_features: split_identity requested but X_or_pair is not (X_geom, theta_id)"
            )

        X_geom, theta_id = X_or_pair

        # normalize geometry shape (e.g., (1,nfeat) -> (nfeat,))
        X_geom = _normalize_shape(X_geom)

        out_path = base + ".npz"
        np.savez(out_path, X_geom=X_geom, theta_id=theta_id)

        print(f"Saved split features to {out_path}")
        print(f"  X_geom shape: {X_geom.shape}")
        print(f"  theta_id shape: {theta_id.shape if theta_id is not None else None}")
        return

    # ------------------------
    # Single-array save (.npy)
    # ------------------------
    X = X_or_pair
    if not isinstance(X, np.ndarray):
        raise ValueError("save_features: expected ndarray when split_identity is False")

    # normalize shape (e.g., (1,nfeat) -> (nfeat,))
    X = _normalize_shape(X)

    out_path = base + ".npy"
    np.save(out_path, X)

    print(f"Saved features to {out_path} with shape {X.shape}")
