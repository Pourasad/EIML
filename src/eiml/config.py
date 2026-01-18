# src/eiml/config.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import yaml

from .params import SOAPParams, SAFTParams, EIMLParams


def load_config_yaml(path: str) -> Dict[str, Any]:
    """
    Load YAML config. Resolve relative paths inside the config relative to
    the config file directory (not the current working directory).
    """
    cfg_path = Path(path).expanduser().resolve()
    base_dir = cfg_path.parent

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("config: top-level YAML must be a mapping/dict")

    # Resolve structure.input relative to config file
    structure = cfg.get("structure", {})
    if isinstance(structure, dict) and "input" in structure and structure["input"] is not None:
        inp = Path(str(structure["input"]))
        if not inp.is_absolute():
            structure["input"] = str((base_dir / inp).resolve())
        cfg["structure"] = structure

    # Resolve output.file relative to config file (optional but convenient)
    out = cfg.get("output", {})
    if isinstance(out, dict) and "file" in out and out["file"] is not None:
        of = Path(str(out["file"]))
        if not of.is_absolute():
            out["file"] = str((base_dir / of).resolve())
        cfg["output"] = out

    return cfg


def params_from_config(cfg: Dict[str, Any]):
    """
    Parse config dictionary into parameter objects.

    Returns:
      mode, structure_cfg, soap_params, saft_params, eiml_params, pool, output_file
    """
    mode = str(cfg.get("mode", "soap")).strip().lower()
    if mode not in ("soap", "eiml", "saft"):
        raise ValueError("config: mode must be one of: soap, eiml, saft")

    # ---------- structure ----------
    structure_cfg = cfg.get("structure", {})
    if not isinstance(structure_cfg, dict):
        raise ValueError("config: structure must be a mapping/dict")
    if "input" not in structure_cfg:
        raise ValueError("config: structure.input is required")

    # ---------- pool ----------
    pool_cfg = cfg.get("pool", None)

    if pool_cfg is None:
        pool = None
    elif isinstance(pool_cfg, str):
        pool = pool_cfg.strip().lower()
    elif isinstance(pool_cfg, dict):
        pool = str(pool_cfg.get("kind", "")).strip().lower()
    else:
        raise ValueError("config: pool must be null, a string, or a mapping like {kind: mean}")

    if pool in ("", "none", "null"):
        pool = None
    if pool not in (None, "mean", "sum"):
        raise ValueError("config: pool.kind must be one of: mean, sum (or omit pool)")

    # ---------- soap ----------
    soap_cfg = cfg.get("soap", {})
    if not isinstance(soap_cfg, dict):
        raise ValueError("config: soap must be a mapping/dict")

    avg = soap_cfg.get("average", "off")

    # YAML 1.1 treats 'off' as False → normalize it
    if avg is False:
        avg = "off"
    elif isinstance(avg, bool):
        raise ValueError(
            "config: soap.average must be one of 'off', 'inner', 'outer'. "
            'If using YAML, quote strings: average: "off"'
        )
    else:
        avg = str(avg)

    soap_params = SOAPParams(
        species=list(soap_cfg.get("species", [])),
        rcut=soap_cfg.get("rcut", None),   # may be None in eiml mode (dynamic rcut)
        nmax=int(soap_cfg.get("nmax", 8)),
        lmax=int(soap_cfg.get("lmax", 6)),
        sigma=float(soap_cfg.get("sigma", 0.5)),
        periodic=bool(soap_cfg.get("periodic", False)),
        average=avg,
        sparse=bool(soap_cfg.get("sparse", False)),
    )

    # ---------- saft (optional identity block; keep for compatibility) ----------
    saft_params: Optional[SAFTParams] = None
    saft_cfg = cfg.get("saft", None)
    if isinstance(saft_cfg, dict):
        # Only build if user provided meaningful fields
        if any(k in saft_cfg for k in ("sigma_saft", "epsilon", "m", "kappa", "eps_assoc")):
            saft_params = SAFTParams(
                sigma_saft=float(saft_cfg["sigma_saft"]),
                epsilon=float(saft_cfg.get("epsilon", 0.0)),
                m=float(saft_cfg.get("m", 1.0)),
                kappa=float(saft_cfg.get("kappa", 0.0)),
                eps_assoc=float(saft_cfg.get("eps_assoc", 0.0)),
                extra=dict(saft_cfg.get("extra", {})),
            )

    # ---------- eiml ----------
    eiml_params: Optional[EIMLParams] = None
    eiml_cfg = cfg.get("eiml", None)
    if mode == "eiml":
        if not isinstance(eiml_cfg, dict):
            raise ValueError("config: mode='eiml' requires an 'eiml:' mapping in YAML")

        sigma = eiml_cfg.get("sigma", None)
        sigma_by_species = eiml_cfg.get("sigma_by_species", None)
        k_rcut = eiml_cfg.get("k_rcut", None)
        omega_rel = float(eiml_cfg.get("omega_rel", 0.1))

        if sigma_by_species is not None and not isinstance(sigma_by_species, dict):
            raise ValueError("config: eiml.sigma_by_species must be a dict if provided")

        if sigma_by_species is not None:
            sigma_by_species = {str(k): float(v) for k, v in sigma_by_species.items()}

        enable_weighting = bool(eiml_cfg.get("enable_weighting", False))
        epsilon = eiml_cfg.get("epsilon", None)
        epsilon_alpha = float(eiml_cfg.get("epsilon_alpha", 1.0))

        if epsilon is not None and not isinstance(epsilon, dict):
            raise ValueError("config: eiml.epsilon must be a dict if provided")

        # sanitize epsilon (keys -> str, values -> float)
        if epsilon is not None:
            epsilon = {str(k): float(v) for k, v in epsilon.items()}

        if enable_weighting and epsilon is None:
            raise ValueError(
                "config: enable_weighting=True requires eiml.epsilon to be provided"
            )

        if not (0.0 < epsilon_alpha <= 1.0):
            raise ValueError("config: eiml.epsilon_alpha must be in (0, 1]")

        eiml_params = EIMLParams(
            sigma=None if sigma in (None, "null") else float(sigma),
            sigma_by_species=sigma_by_species,
            k_rcut=None if k_rcut in (None, "null") else float(k_rcut),
            omega_rel=omega_rel,
            enable_weighting=enable_weighting,
            epsilon=epsilon,
            epsilon_alpha=epsilon_alpha,
        )

    # ---------- output ----------
    out_cfg = cfg.get("output", {}) or {}
    output_file = out_cfg.get("file", None)
    if output_file is None:
        raise ValueError("config: output.file is required")
    output_file = str(output_file)

    return mode, structure_cfg, soap_params, saft_params, eiml_params, pool, output_file
