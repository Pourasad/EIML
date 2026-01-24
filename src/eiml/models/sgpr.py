# src/eiml/models/sgpr.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np

try:
    from sklearn.cluster import MiniBatchKMeans
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def _rbf_kernel(X: np.ndarray, Z: np.ndarray, length_scale: float) -> np.ndarray:
    """
    RBF kernel k(x,z)=exp(-||x-z||^2/(2l^2))
    X: (N,D), Z: (M,D) -> (N,M)
    """
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    l2 = float(length_scale) ** 2
    # squared distance via (x^2 + z^2 - 2 x·z)
    X2 = np.sum(X * X, axis=1, keepdims=True)         # (N,1)
    Z2 = np.sum(Z * Z, axis=1, keepdims=True).T       # (1,M)
    d2 = X2 + Z2 - 2.0 * (X @ Z.T)                    # (N,M)
    d2 = np.maximum(d2, 0.0)
    return np.exp(-0.5 * d2 / l2)


def _chol_solve(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Solve (L L^T) X = B for X given Cholesky factor L (lower-triangular).
    """
    # forward solve L Y = B
    Y = np.linalg.solve(L, B)
    # backward solve L^T X = Y
    X = np.linalg.solve(L.T, Y)
    return X


def _fps_inducing(X: np.ndarray, M: int, seed: int = 0) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) in feature space using Euclidean distance.
    Returns Z with shape (M, D).
    Complexity: O(N*M*D) time, O(N) memory for distances.
    """
    rng = np.random.default_rng(seed)
    N, D = X.shape
    if M > N:
        raise ValueError(f"M={M} cannot exceed N={N}")

    # pick first point randomly
    idx = np.empty(M, dtype=np.int64)
    idx[0] = rng.integers(N)

    # distances to current selected set (start with dist to first center)
    diff = X - X[idx[0]]
    dist2 = np.einsum("ij,ij->i", diff, diff)  # squared norm, shape (N,)

    for i in range(1, M):
        idx[i] = int(np.argmax(dist2))
        diff = X - X[idx[i]]
        d2_new = np.einsum("ij,ij->i", diff, diff)
        dist2 = np.minimum(dist2, d2_new)

    return X[idx].copy()


def _choose_inducing(
    X: np.ndarray,
    M: int,
    method: str = "random",
    seed: int = 0,
    kmeans_batch_size: int = 4096,
    kmeans_max_iter: int = 200,
) -> np.ndarray:
    """
    Return inducing locations Z of shape (M,D).
    - method="random": choose M points from X
    - method="kmeans": MiniBatchKMeans centers (requires sklearn)
    - method="fps": farthest-point sampling in feature space (no sklearn)
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    if M > N:
        raise ValueError(f"M={M} cannot exceed N={N}")

    method = str(method).lower().strip()

    if method == "random":
        idx = rng.choice(N, size=M, replace=False)
        return X[idx].copy()

    if method == "kmeans":
        if not _HAS_SKLEARN:
            raise RuntimeError("method='kmeans' requires scikit-learn installed.")
        km = MiniBatchKMeans(
            n_clusters=M,
            random_state=seed,
            batch_size=kmeans_batch_size,
            max_iter=kmeans_max_iter,
            n_init="auto",
        )
        km.fit(X)
        return km.cluster_centers_.astype(np.float64, copy=True)

    if method == "fps":
        return _fps_inducing(X, M=M, seed=seed)

    raise ValueError("Unknown inducing method. Use 'random', 'kmeans', or 'fps'.")


def _heuristic_length_scale(X: np.ndarray, seed: int = 0, n_probe: int = 512) -> float:
    """
    Median pairwise distance heuristic on a subset.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    m = min(int(n_probe), N)
    idx = rng.choice(N, size=m, replace=False)
    Xs = X[idx].astype(np.float64, copy=False)

    # compute distances to a random anchor set for O(m^2) avoidance
    anchors = Xs[rng.choice(m, size=min(64, m), replace=False)]
    # dist^2 = ||x-a||^2
    X2 = np.sum(Xs * Xs, axis=1, keepdims=True)
    A2 = np.sum(anchors * anchors, axis=1, keepdims=True).T
    d2 = X2 + A2 - 2.0 * (Xs @ anchors.T)
    d = np.sqrt(np.maximum(d2, 0.0)).ravel()
    med = np.median(d[d > 0])
    if not np.isfinite(med) or med <= 0:
        return 1.0
    return float(med)


@dataclass
class SGPRModel:
    """
    Sparse GP regression (inducing-point / Nyström / DTC-style).

    Multi-output supported by fitting each output independently
    but sharing Z, length_scale, noise, jitter.
    """
    Z: np.ndarray                 # (M,D)
    length_scale: float
    noise: float
    jitter: float = 1e-6

    # learned cache (set after fit)
    alpha: Optional[np.ndarray] = None       # (M, P) where P=outputs
    Lmm: Optional[np.ndarray] = None         # Cholesky(Kmm + jitter I)
    LA: Optional[np.ndarray] = None          # Cholesky(A) where A = Kmm + (1/noise^2) Kmn Knm

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "SGPRModel":
        """
        Fit given:
          X: (N,D)
          Y: (N,P) or (N,)
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        N, D = X.shape
        N2, P = Y.shape
        if N2 != N:
            raise ValueError(f"X has N={N} but Y has N={N2}")

        M = self.Z.shape[0]
        if self.Z.shape[1] != D:
            raise ValueError(f"Z has D={self.Z.shape[1]} but X has D={D}")

        l = float(self.length_scale)
        s2 = float(self.noise) ** 2
        if s2 <= 0:
            raise ValueError("noise must be > 0")

        # Kernels
        Kmm = _rbf_kernel(self.Z, self.Z, l)  # (M,M)
        Kmm.flat[:: M + 1] += self.jitter

        Lmm = np.linalg.cholesky(Kmm)

        Knm = _rbf_kernel(X, self.Z, l)       # (N,M)
        Kmn = Knm.T                           # (M,N)

        # Build A = Kmm + (1/s2) * Kmn*Knm   (M,M)
        # Kmn*Knm = sum over N of outer products
        A = Kmm + (Kmn @ Knm) / s2
        # extra jitter for numerical safety
        A.flat[:: M + 1] += self.jitter
        LA = np.linalg.cholesky(A)

        # Compute alpha for each output:
        # alpha = Kmm^{-1} * (1/s2) * [A^{-1} * (Kmm * (Kmn Y))]  (M,P)
        # Steps:
        #   a = Kmn @ Y                 (M,P)
        #   b = Kmm @ a                 (M,P)
        #   c = A^{-1} b                (M,P) via chol solve
        #   mu = (1/s2) * c             (M,P)
        #   alpha = Kmm^{-1} mu         (M,P) via chol solve of Kmm
        a = Kmn @ Y                      # (M,P)
        c = _chol_solve(LA, a)           # (M,P)
        alpha = c / s2                   # (M,P)

        self.alpha = alpha
        self.Lmm = Lmm
        self.LA = LA
        return self

    def predict(self, Xs: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict:
          Xs: (Ns,D)
        Returns:
          mean: (Ns,P)
          std : (Ns,P) if return_std else None

        For mean predictions, we only need alpha (and Z, length_scale).
        For the cheap std proxy, we need Lmm. LA is NOT required.
        """
        if self.alpha is None:
            raise RuntimeError("Model not fit/loaded yet (alpha missing).")

        Xs = np.asarray(Xs, dtype=np.float64)
        Ks = _rbf_kernel(Xs, self.Z, self.length_scale)  # (Ns,M)
        mean = Ks @ self.alpha                            # (Ns,P)

        if not return_std:
            return mean, None

        # Need Lmm for q_diag
        if self.Lmm is None:
            # try rebuild prior cache from Z
            self.rebuild_cache()
        if self.Lmm is None:
            raise RuntimeError("Cannot compute std: Lmm cache missing and rebuild failed.")

        # Approx predictive variance (DTC-ish proxy):
        M = self.Z.shape[0]
        kxx = np.ones((Xs.shape[0], 1), dtype=np.float64)  # RBF: k(x,x)=1

        v = np.linalg.solve(self.Lmm, Ks.T)                # (M,Ns)
        q_diag = np.sum(v * v, axis=0, keepdims=True).T    # (Ns,1)

        var = np.maximum(kxx - q_diag, 0.0) + (self.noise ** 2)
        std = np.sqrt(var)

        P = self.alpha.shape[1]
        std = np.repeat(std, P, axis=1)
        return mean, std

    def to_dict(self) -> Dict[str, Any]:
        if self.alpha is None:
            raise RuntimeError("Cannot serialize an unfitted model.")
        d = {
            "Z": self.Z,
            "length_scale": float(self.length_scale),
            "noise": float(self.noise),
            "jitter": float(self.jitter),
            "alpha": self.alpha,
        }
        # Store Lmm if available (helps std without needing training data)
        if self.Lmm is not None:
            d["Lmm"] = self.Lmm
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SGPRModel":
        m = cls(
            Z=np.asarray(d["Z"], dtype=np.float64),
            length_scale=float(d["length_scale"]),
            noise=float(d["noise"]),
            jitter=float(d.get("jitter", 1e-6)),
        )
        m.alpha = np.asarray(d["alpha"], dtype=np.float64)
        m.Lmm = np.asarray(d["Lmm"], dtype=np.float64) if "Lmm" in d else None
        m.LA = None  # not needed for mean; we don't reconstruct it
        return m

    def rebuild_cache(self) -> None:
        """
        Recompute Cholesky caches from Z and hyperparams (needed if you load model and want std).
        """
        M = self.Z.shape[0]
        Kmm = _rbf_kernel(self.Z, self.Z, self.length_scale)
        Kmm.flat[:: M + 1] += self.jitter
        self.Lmm = np.linalg.cholesky(Kmm)
        # LA depends on training X; we can't rebuild without it, so keep None.
        self.LA = None


def train_sgpr(
    X: np.ndarray,
    Y: np.ndarray,
    M: int = 1024,
    inducing: str = "random",
    seed: int = 0,
    length_scale: Optional[float] = None,
    noise: Optional[float] = None,
    jitter: float = 1e-6,
) -> SGPRModel:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]

    Z = _choose_inducing(X, M=M, method=inducing, seed=seed)

    if length_scale is None:
        length_scale = _heuristic_length_scale(X, seed=seed)

    if noise is None:
        # heuristic: 5% of target std (per component), take median
        ys = np.std(Y, axis=0)
        nv = np.median(0.05 * ys)
        noise = float(max(nv, 1e-6))

    model = SGPRModel(Z=Z, length_scale=float(length_scale), noise=float(noise), jitter=float(jitter))
    model.fit(X, Y)
    return model

# =========================
# Serialization + utilities
# =========================

from pathlib import Path


@dataclass
class Standardizer:
    """Lightweight scaler (no sklearn)."""
    mean_: np.ndarray
    scale_: np.ndarray
    eps: float = 1e-12

    @classmethod
    def fit(cls, X: np.ndarray, eps: float = 1e-12) -> "Standardizer":
        X = np.asarray(X, dtype=np.float64)
        m = X.mean(axis=0)
        s = X.std(axis=0)
        s = np.where(s < eps, 1.0, s)
        return cls(mean_=m, scale_=s, eps=eps)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, Xs: np.ndarray) -> np.ndarray:
        Xs = np.asarray(Xs, dtype=np.float64)
        return Xs * self.scale_ + self.mean_

    def to_dict(self) -> Dict[str, Any]:
        return {"mean_": self.mean_, "scale_": self.scale_, "eps": float(self.eps)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Standardizer":
        return cls(
            mean_=np.asarray(d["mean_"], dtype=np.float64),
            scale_=np.asarray(d["scale_"], dtype=np.float64),
            eps=float(d.get("eps", 1e-12)),
        )


def save_sgpr(
    outdir: str | Path,
    model: SGPRModel,
    x_scaler: Optional[Standardizer] = None,
    y_scaler: Optional[Standardizer] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model + scalers into outdir:
      - model.npz  (Z, alpha, hyperparams, optional caches)
      - scalers.npz (optional)
      - meta.json   (optional)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if model.alpha is None:
        raise RuntimeError("Refusing to save an unfitted model (alpha is None).")

    # Save model arrays
    md = {
        "Z": np.asarray(model.Z, dtype=np.float64),
        "alpha": np.asarray(model.alpha, dtype=np.float64),
        "length_scale": np.asarray([float(model.length_scale)], dtype=np.float64),
        "noise": np.asarray([float(model.noise)], dtype=np.float64),
        "jitter": np.asarray([float(model.jitter)], dtype=np.float64),
    }

    # Optional caches: helpful for faster prediction/uncertainty
    if model.Lmm is not None:
        md["Lmm"] = np.asarray(model.Lmm, dtype=np.float64)
    if model.LA is not None:
        md["LA"] = np.asarray(model.LA, dtype=np.float64)

    np.savez(outdir / "model.npz", **md)

    # Save scalers (if any)
    sd: Dict[str, Any] = {}
    if x_scaler is not None:
        sd["x_mean"] = np.asarray(x_scaler.mean_, dtype=np.float64)
        sd["x_scale"] = np.asarray(x_scaler.scale_, dtype=np.float64)
        sd["x_eps"] = np.asarray([float(x_scaler.eps)], dtype=np.float64)
    if y_scaler is not None:
        sd["y_mean"] = np.asarray(y_scaler.mean_, dtype=np.float64)
        sd["y_scale"] = np.asarray(y_scaler.scale_, dtype=np.float64)
        sd["y_eps"] = np.asarray([float(y_scaler.eps)], dtype=np.float64)

    if sd:
        np.savez(outdir / "scalers.npz", **sd)

    # Save meta
    if meta is not None:
        import json
        with open(outdir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)


def load_sgpr(
    model_dir: str | Path,
    allow_missing_scalers: bool = True,
) -> Tuple[SGPRModel, Optional[Standardizer], Optional[Standardizer], Dict[str, Any]]:
    """
    Load model + scalers + meta from model_dir.
    Returns: (model, x_scaler, y_scaler, meta)
    """
    model_dir = Path(model_dir)

    # ---- model ----
    mfile = model_dir / "model.npz"
    if not mfile.exists():
        raise FileNotFoundError(f"Missing {mfile}")

    d = np.load(mfile, allow_pickle=False)
    Z = np.asarray(d["Z"], dtype=np.float64)
    alpha = np.asarray(d["alpha"], dtype=np.float64)
    length_scale = float(np.asarray(d["length_scale"]).ravel()[0])
    noise = float(np.asarray(d["noise"]).ravel()[0])
    jitter = float(np.asarray(d["jitter"]).ravel()[0])

    model = SGPRModel(Z=Z, length_scale=length_scale, noise=noise, jitter=jitter)
    model.alpha = alpha

    # caches optional
    model.Lmm = np.asarray(d["Lmm"], dtype=np.float64) if "Lmm" in d.files else None
    model.LA = np.asarray(d["LA"], dtype=np.float64) if "LA" in d.files else None

    # If Lmm missing, we can rebuild it from Z + hyperparams (always possible)
    if model.Lmm is None:
        model.rebuild_cache()

    # NOTE: LA depends on training X, so we can only have it if we saved it.
    # For mean prediction, LA is not required.

    # ---- scalers ----
    x_scaler = None
    y_scaler = None
    sfile = model_dir / "scalers.npz"
    if sfile.exists():
        s = np.load(sfile, allow_pickle=False)
        if "x_mean" in s.files and "x_scale" in s.files:
            x_eps = float(np.asarray(s.get("x_eps", [1e-12])).ravel()[0])
            x_scaler = Standardizer(np.asarray(s["x_mean"]), np.asarray(s["x_scale"]), eps=x_eps)
        if "y_mean" in s.files and "y_scale" in s.files:
            y_eps = float(np.asarray(s.get("y_eps", [1e-12])).ravel()[0])
            y_scaler = Standardizer(np.asarray(s["y_mean"]), np.asarray(s["y_scale"]), eps=y_eps)
    elif not allow_missing_scalers:
        raise FileNotFoundError(f"Missing {sfile}")

    # ---- meta ----
    meta: Dict[str, Any] = {}
    jfile = model_dir / "meta.json"
    if jfile.exists():
        import json
        meta = json.load(open(jfile))

    return model, x_scaler, y_scaler, meta


def predict_sgpr(
    model_dir: str | Path,
    X: np.ndarray,
    return_std: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convenience: load model+scalers and predict in original y units.
    """
    model, x_scaler, y_scaler, _meta = load_sgpr(model_dir)

    X = np.asarray(X, dtype=np.float64)
    Xs = x_scaler.transform(X) if x_scaler is not None else X

    y_mean_s, y_std_s = model.predict(Xs, return_std=return_std)

    # If y was standardized during training, invert it here
    if y_scaler is not None:
        y_mean = y_scaler.inverse_transform(y_mean_s)
        if y_std_s is not None:
            # std scales by y_scale (no mean shift)
            y_std = y_std_s * y_scaler.scale_
        else:
            y_std = None
    else:
        y_mean, y_std = y_mean_s, y_std_s

    # If single-output, return shape (N,) rather than (N,1)
    if y_mean.shape[1] == 1:
        y_mean = y_mean.ravel()
        if y_std is not None:
            y_std = y_std.ravel()

    return y_mean, y_std
