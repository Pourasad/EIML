#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional, Sequence, Dict, Any
from pathlib import Path
import argparse
import json
import pprint

import numpy as np

from .config import load_config_yaml, params_from_config
from .io_structures import read_structure_from_cfg
from .descriptor import compute_descriptor
from .save import save_features


# -------------------------
# small utilities
# -------------------------
def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.mean(np.abs(a - b)))


def _fit_standardizer(X: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Standardize per-feature: (X - mean) / scale
    scale = std, with zeros replaced by 1.
    """
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return {"mean": mean, "scale": scale}


def _transform_standardizer(X: np.ndarray, st: Dict[str, np.ndarray]) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return (X - st["mean"]) / st["scale"]


def _inverse_transform_standardizer(Xs: np.ndarray, st: Dict[str, np.ndarray]) -> np.ndarray:
    Xs = np.asarray(Xs, dtype=np.float64)
    return Xs * st["scale"] + st["mean"]


def _save_model_npz(model, outdir: Path) -> None:
    out = model.to_dict()
    # np.savez_compressed supports arrays + scalars
    np.savez_compressed(outdir / "model.npz", **out)


def _load_model_npz(modeldir: Path):
    from .models.sgpr import SGPRModel
    d = dict(np.load(modeldir / "model.npz", allow_pickle=True))
    # np.load gives numpy scalars sometimes; keep as-is, SGPRModel handles it.
    return SGPRModel.from_dict(d)


def _save_scalers_npz(outdir: Path, sx: Optional[Dict[str, np.ndarray]], sy: Optional[Dict[str, np.ndarray]]) -> None:
    if sx is None and sy is None:
        np.savez_compressed(outdir / "scalers.npz", has_scalers=False)
        return
    payload: Dict[str, Any] = {"has_scalers": True}
    if sx is not None:
        payload["sx_mean"] = sx["mean"]
        payload["sx_scale"] = sx["scale"]
    if sy is not None:
        payload["sy_mean"] = sy["mean"]
        payload["sy_scale"] = sy["scale"]
    np.savez_compressed(outdir / "scalers.npz", **payload)


def _load_scalers_npz(modeldir: Path):
    d = dict(np.load(modeldir / "scalers.npz", allow_pickle=True))
    if not bool(d.get("has_scalers", False)):
        return None, None
    sx = None
    sy = None
    if "sx_mean" in d and "sx_scale" in d:
        sx = {"mean": d["sx_mean"], "scale": d["sx_scale"]}
    if "sy_mean" in d and "sy_scale" in d:
        sy = {"mean": d["sy_mean"], "scale": d["sy_scale"]}
    return sx, sy


# -------------------------
# CLI builders
# -------------------------
def _add_featurize_cmd(subp: argparse._SubParsersAction) -> None:
    p = subp.add_parser(
        "featurize",
        help="Compute SOAP/EIML descriptors from a YAML config.",
        description="Compute SOAP/EIML descriptors from a YAML config.",
    )
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--debug", action="store_true", help="Print parsed configuration and parameters")
    p.set_defaults(_cmd="featurize")


def _add_train_cmd(subp: argparse._SubParsersAction) -> None:
    p = subp.add_parser(
        "train",
        help="Train a model (e.g., SGPR) from X/y arrays.",
        description="Train a model (e.g., SGPR) from X/y arrays.",
    )
    train_sub = p.add_subparsers(dest="train_model", required=True)

    ps = train_sub.add_parser("sgpr", help="Train Sparse Gaussian Process Regression (SGPR).")
    ps.add_argument("--X", required=True, help="Path to X.npy (N,D)")
    ps.add_argument("--y", required=True, help="Path to y.npy (N,) or (N,P)")
    ps.add_argument("--outdir", required=True, help="Output directory to save model bundle")
    ps.add_argument("--seed", type=int, default=0)

    ps.add_argument("--M", type=int, default=1024, help="Number of inducing points")
    ps.add_argument("--inducing", default="fps", choices=["random", "kmeans", "fps"])
    ps.add_argument("--standardize", action="store_true", help="Standardize X and y")

    ps.add_argument("--length_scale", type=float, default=None)
    ps.add_argument("--noise", type=float, default=None)
    ps.add_argument("--jitter", type=float, default=1e-6)

    ps.add_argument("--train_size", type=int, default=None, help="If set, subsample training points")
    ps.add_argument("--test_size", type=int, default=None, help="If set, hold out points for metrics")
    ps.set_defaults(_cmd="train_sgpr")


def _add_predict_cmd(subp: argparse._SubParsersAction) -> None:
    p = subp.add_parser(
        "predict",
        help="Predict using a trained model.",
        description="Predict using a trained model.",
    )
    pred_sub = p.add_subparsers(dest="pred_model", required=True)

    ps = pred_sub.add_parser("sgpr", help="Predict with an SGPR model bundle.")
    ps.add_argument("--modeldir", required=True, help="Directory produced by `eiml train sgpr`")
    ps.add_argument("--X", required=True, help="Path to X.npy (N,D)")
    ps.add_argument("--outdir", required=True, help="Directory to write predictions")
    ps.add_argument("--std", action="store_true", help="Also write y_std.npy (cheap proxy)")
    ps.set_defaults(_cmd="predict_sgpr")


def _parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="eiml", description="EIML: descriptors + SGPR utilities")
    subp = p.add_subparsers(dest="command", required=True)

    _add_featurize_cmd(subp)
    _add_train_cmd(subp)
    _add_predict_cmd(subp)

    return p.parse_args(argv)


# -------------------------
# commands
# -------------------------
def _run_featurize(args: argparse.Namespace) -> None:
    cfg = load_config_yaml(args.config)

    mode, structure_cfg, soap_params, saft_params, eiml_params, pool, output_file = params_from_config(cfg)
    atoms_or_traj = read_structure_from_cfg(structure_cfg)

    out_cfg = cfg.get("output", {}) or {}
    split_identity = bool(out_cfg.get("split_identity", False))

    if split_identity and mode == "soap":
        print("[WARN] output.split_identity is ignored when mode='soap'. Writing a normal .npy instead.")
        split_identity = False

    if args.debug:
        print("\n=== Raw config (resolved) ===")
        pprint.pprint(cfg)
        print("\n=== Parsed parameters ===")
        print("mode:", mode)
        print("structure_cfg:", structure_cfg)
        print("soap_params:", soap_params)
        print("saft_params:", saft_params)
        print("eiml_params:", eiml_params)
        print("pool:", pool)
        print("output_file:", output_file)
        print("split_identity:", split_identity)
        print()

    if split_identity:
        X_or_pair = compute_descriptor(
            atoms_or_traj,
            mode=mode,
            soap_params=soap_params,
            eiml_params=eiml_params,
            saft_params=saft_params,
            pool=pool,
            append_identity=False,
            return_identity=True,
        )
    else:
        X_or_pair = compute_descriptor(
            atoms_or_traj,
            mode=mode,
            soap_params=soap_params,
            eiml_params=eiml_params,
            saft_params=saft_params,
            pool=pool,
            append_identity=True,
            return_identity=False,
        )

    save_features(mode=mode, output_file=output_file, split_identity=split_identity, X_or_pair=X_or_pair)


def _run_train_sgpr(args: argparse.Namespace) -> None:
    from .models.sgpr import train_sgpr

    X = np.asarray(np.load(args.X), dtype=np.float64)
    y = np.asarray(np.load(args.y), dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]

    N = X.shape[0]
    if y.shape[0] != N:
        raise ValueError(f"X has N={N} but y has N={y.shape[0]}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)

    # choose train/test splits (optional)
    ntr = N if args.train_size is None else int(min(args.train_size, N))
    nte = 0 if args.test_size is None else int(min(args.test_size, max(N - ntr, 0)))

    tr_idx = perm[:ntr]
    te_idx = perm[ntr:ntr + nte] if nte > 0 else np.array([], dtype=np.int64)

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = (X[te_idx], y[te_idx]) if nte > 0 else (None, None)

    # standardize
    sx = None
    sy = None
    if args.standardize:
        sx = _fit_standardizer(Xtr)
        Xtr_s = _transform_standardizer(Xtr, sx)
        Xte_s = _transform_standardizer(Xte, sx) if Xte is not None else None

        sy = _fit_standardizer(ytr)
        ytr_s = _transform_standardizer(ytr, sy)
    else:
        Xtr_s = Xtr
        Xte_s = Xte
        ytr_s = ytr

    model = train_sgpr(
        Xtr_s,
        ytr_s,
        M=int(args.M),
        inducing=str(args.inducing),
        seed=int(args.seed),
        length_scale=args.length_scale,
        noise=args.noise,
        jitter=float(args.jitter),
    )

    # predictions for metrics
    ypred_tr_s, _ = model.predict(Xtr_s, return_std=False)
    ypred_tr = _inverse_transform_standardizer(ypred_tr_s, sy) if sy is not None else ypred_tr_s

    if Xte_s is not None and Xte_s.size > 0:
        ypred_te_s, _ = model.predict(Xte_s, return_std=False)
        ypred_te = _inverse_transform_standardizer(ypred_te_s, sy) if sy is not None else ypred_te_s
    else:
        ypred_te = None

    metrics: Dict[str, Any] = {
        "N_total": int(N),
        "N_train": int(ntr),
        "N_test": int(nte),
        "D": int(X.shape[1]),
        "P": int(y.shape[1]),
        "M": int(args.M),
        "inducing": str(args.inducing),
        "standardize": bool(args.standardize),
        "length_scale": float(model.length_scale),
        "noise": float(model.noise),
        "jitter": float(model.jitter),
        "rmse_train": _rmse(ytr, ypred_tr),
        "mae_train": _mae(ytr, ypred_tr),
    }
    if ypred_te is not None:
        metrics.update(
            rmse_test=_rmse(yte, ypred_te),
            mae_test=_mae(yte, ypred_te),
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save model bundle
    _save_model_npz(model, outdir)
    _save_scalers_npz(outdir, sx=sx, sy=sy)

    meta = {
        "model": "sgpr",
        "args": vars(args),
    }
    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print("Saved:", str(outdir))


def _run_predict_sgpr(args: argparse.Namespace) -> None:
    modeldir = Path(args.modeldir)
    model = _load_model_npz(modeldir)
    sx, sy = _load_scalers_npz(modeldir)

    X = np.asarray(np.load(args.X), dtype=np.float64)
    Xs = _transform_standardizer(X, sx) if sx is not None else X

    ypred_s, ystd_s = model.predict(Xs, return_std=bool(args.std))

    ypred = _inverse_transform_standardizer(ypred_s, sy) if sy is not None else ypred_s
    if args.std:
        # std rescales by sy.scale (not shift)
        if sy is not None:
            ystd = ystd_s * sy["scale"]
        else:
            ystd = ystd_s
    else:
        ystd = None

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "y_pred.npy", ypred)
    if ystd is not None:
        np.save(outdir / "y_std.npy", ystd)

    print("Saved:", str(outdir))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_cli(argv)

    if args._cmd == "featurize":
        _run_featurize(args)
        return

    if args._cmd == "train_sgpr":
        _run_train_sgpr(args)
        return

    if args._cmd == "predict_sgpr":
        _run_predict_sgpr(args)
        return

    raise RuntimeError(f"Unknown command: {args._cmd}")


if __name__ == "__main__":
    main()
