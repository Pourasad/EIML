from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import yaml
import os
from pathlib import Path

from .params import SOAPParams, SAFTParams, EIMLParams


def load_config_yaml(path: str) -> Dict[str, Any]:
    """
    Load YAML config. Any relative file paths inside the config are resolved
    relative to the config file's directory (not the current working directory).
    """
    cfg_path = Path(path).expanduser().resolve()
    base_dir = cfg_path.parent

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Empty YAML config: {cfg_path}")
    if not isinstance(cfg, dict):
        raise ValueError("Top-level YAML must be a mapping/dict.")

    # ---- Resolve common paths relative to config location ----
    # structure.input
    structure = cfg.get("structure", {})
    if isinstance(structure, dict):
        inp = structure.get("input", None)
        if isinstance(inp, str) and inp.strip() != "":
            p = Path(inp).expanduser()
            if not p.is_absolute():
                structure["input"] = str((base_dir / p).resolve())

    # output.file (optional, but very useful)
    out = cfg.get("output", {})
    if isinstance(out, dict):
        ofile = out.get("file", None)
        if isinstance(ofile, str) and ofile.strip() != "":
            p = Path(ofile).expanduser()
            if not p.is_absolute():
                out["file"] = str((base_dir / p).resolve())

    return cfg

def _noneish(x: Any) -> bool:
    return x is None or x == "null" or x == "None"


def params_from_config(
    cfg: Dict[str, Any]
) -> Tuple[
    str,                 # mode
    Dict[str, Any],       # structure_cfg
    SOAPParams,           # soap_params
    Optional[SAFTParams], # saft_params
    Optional[EIMLParams], # eiml_params
    Optional[str],        # pool
    str,                 # output_file
]:
    """
    Returns:
      mode, structure_cfg, soap_params, saft_params, eiml_params, pool, output_file

    Notes:
      - mode can be: "soap", "eiml", "saft"
      - "saft" is treated as an alias of "eiml" in the descriptor layer (if you want)
        but we still parse saft_params for identity if present.
    """
    mode = str(cfg.get("mode", "soap")).strip().lower()
    if mode not in ("soap", "eiml", "saft"):
        raise ValueError("config: mode must be one of: soap, eiml, saft")

    # -------- structure --------
    structure_cfg = cfg.get("structure", {})
    if not isinstance(structure_cfg, dict):
        raise ValueError("config: structure must be a mapping/dict")
    if "input" not in structure_cfg:
        raise ValueError("config: structure.input is required")

    # -------- soap --------
    soap_cfg = cfg.get("soap", {})
    if not isinstance(soap_cfg, dict):
        raise ValueError("config: soap must be a mapping/dict")

    # NOTE: rcut can be optional if you compute it dynamically in descriptor.py (EIML)
    rcut = soap_cfg.get("rcut", None)

    soap_params = SOAPParams(
        species=list(soap_cfg["species"]),
        rcut=None if _noneish(rcut) else float(rcut),
        nmax=int(soap_cfg["nmax"]),
        lmax=int(soap_cfg["lmax"]),
        sigma=float(soap_cfg.get("sigma", 0.5)),
        periodic=bool(soap_cfg.get("periodic", False)),
        average=str(soap_cfg.get("average", "off")),
        sparse=bool(soap_cfg.get("sparse", False)),
    )

    # -------- eiml --------
    eiml_params: Optional[EIMLParams] = None
    eiml_cfg = cfg.get("eiml", None)

    if mode in ("eiml", "saft"):
        if eiml_cfg is None:
            # allow legacy configs that used "saft:" as the eiml block name
            eiml_cfg = cfg.get("saft", None)

        if not isinstance(eiml_cfg, dict):
            raise ValueError("config: eiml (or saft) block must be provided and must be a mapping/dict")

        sigma = eiml_cfg.get("sigma", None)
        sigma_by_species = eiml_cfg.get("sigma_by_species", None)
        k_rcut = eiml_cfg.get("k_rcut", None)
        omega_rel = eiml_cfg.get("omega_rel", 0.1)

        if sigma_by_species is not None and not isinstance(sigma_by_species, dict):
            raise ValueError("config: eiml.sigma_by_species must be a mapping/dict if provided")

        eiml_params = EIMLParams(
            sigma=None if _noneish(sigma) else float(sigma),
            sigma_by_species=None if sigma_by_species is None else {str(k): float(v) for k, v in sigma_by_species.items()},
            k_rcut=None if _noneish(k_rcut) else float(k_rcut),
            omega_rel=float(omega_rel),
        )

    # -------- saft identity (optional) --------
    saft_params: Optional[SAFTParams] = None
    saft_cfg = cfg.get("saft", None)

    # Only parse SAFT identity if explicitly provided as "saft:" block
    if isinstance(saft_cfg, dict) and "sigma_saft" in saft_cfg:
        saft_params = SAFTParams(
            sigma_saft=float(saft_cfg["sigma_saft"]),
            epsilon=float(saft_cfg["epsilon"]),
            m=float(saft_cfg["m"]),
            kappa=float(saft_cfg["kappa"]),
            eps_assoc=float(saft_cfg["eps_assoc"]),
            omega_rel=float(saft_cfg.get("omega_rel", 0.1)),
            extra=dict(saft_cfg.get("extra", {})),
        )

    # -------- output --------
    out_cfg = cfg.get("output", {})
    if not isinstance(out_cfg, dict):
        raise ValueError("config: output must be a mapping/dict")

    pool = out_cfg.get("pool", None)
    output_file = str(out_cfg.get("file", "features"))

    return mode, structure_cfg, soap_params, saft_params, eiml_params, pool, output_file
