from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------
# SOAP parameters (DScribe)
# ------------------------------------------------------------
@dataclass
class SOAPParams:
    """
    Parameters for DScribe SOAP.

    Notes
    -----
    - `rcut` is Optional so EIML can set it dynamically via:
        R_cut = k_rcut * sigma_ref
    - `sigma` here is the DScribe Gaussian width parameter in the coordinate
      system DScribe sees:
        * mode="soap": physical units (Å)
        * mode="eiml": reduced units (dimensionless), typically omega_rel
    """
    species: List[str]
    rcut: Optional[float]          # allow None -> computed later (EIML)
    nmax: int
    lmax: int
    sigma: float
    periodic: bool
    average: str = "off"
    sparse: bool = False
    weighting: Optional[Dict[str, Any]] = None  # DScribe-native weighting dict (optional)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "species": self.species,
            "rcut": self.rcut,
            "nmax": self.nmax,
            "lmax": self.lmax,
            "sigma": self.sigma,
            "periodic": self.periodic,
            "average": self.average,
            "sparse": self.sparse,
            "weighting": self.weighting,
        }


# ------------------------------------------------------------
# EIML parameters (EIML-v1)
# ------------------------------------------------------------
@dataclass
class EIMLParams:
    """
    Experimentally-Informed ML parameters (EIML-v1).

    Core scaling ideas
    ------------------
    1) Reduced coordinates:
        r* = r / sigma_ref

    2) Dynamic cutoff (physical):
        R_cut = k_rcut * sigma_ref
       In reduced coordinates, DScribe receives:
        r_cut* = R_cut / sigma_ref = k_rcut

    3) Adaptive Gaussian width (physical):
        omega = omega_rel * sigma_ref
       In reduced coordinates, DScribe receives:
        sigma* = omega_rel   (dimensionless)

    Parameters
    ----------
    sigma:
        Global characteristic size scale (one-component systems).
    sigma_by_species:
        Optional per-species sigma mapping, e.g. {"O": 3.0, "H": 2.5}.
        If provided, it can be used for center-based scaling choices.
    k_rcut:
        Dimensionless multiplier for dynamic cutoff: R_cut = k_rcut * sigma_ref.
        Used only if `soap.rcut` is None.
    omega_rel:
        Reduced Gaussian width ω* (dimensionless). Physical width is ω = ω* * sigma_ref.

    Epsilon-based channel weighting (EIML-v1.1)
    -------------------------------------------
    This is NOT SAFT. Epsilon here is a user-provided relative importance scale.

    enable_weighting:
        Turn on channel weighting.
    epsilon:
        Raw per-species importance values, e.g. {"H": 0.5, "O": 1.0}. Must be positive.
        These are normalized internally (geometric-mean normalization + damping).
    epsilon_alpha:
        Damping exponent alpha in (0, 1]. Smaller values reduce the influence of epsilon.
    """
    sigma: Optional[float] = None
    sigma_by_species: Optional[Dict[str, float]] = None
    k_rcut: Optional[float] = None
    omega_rel: float = 0.1

    # (EIML-v1.1) epsilon-based channel weighting
    enable_weighting: bool = False
    epsilon: Optional[Dict[str, float]] = None
    epsilon_alpha: float = 1.0

    def sigma_for_center(self, symbol: str) -> float:
        """
        Center-based sigma:
          - if sigma_by_species is provided and contains the symbol -> use it
          - else fall back to global sigma
        """
        if self.sigma_by_species is not None and symbol in self.sigma_by_species:
            return float(self.sigma_by_species[symbol])
        if self.sigma is None:
            raise ValueError("EIMLParams requires either 'sigma' or 'sigma_by_species' to be set.")
        return float(self.sigma)

    def sigma_ref_for_rcut(self) -> float:
        """
        Reference sigma used for:
          - reduced-coordinate scaling r* = r / sigma_ref
          - dynamic cutoff R_cut = k_rcut * sigma_ref
          - adaptive width omega = omega_rel * sigma_ref

        Default choice:
          - for mixtures: max(sigma_by_species.values()) (safe, shell-covering)
          - for one-component: global sigma
        """
        if self.sigma_by_species:
            return float(max(self.sigma_by_species.values()))
        if self.sigma is None:
            raise ValueError("EIMLParams requires 'sigma' (or sigma_by_species) to compute sigma_ref.")
        return float(self.sigma)

    def validate_weighting(self) -> None:
        """
        Optional helper to validate epsilon-weighting inputs.
        Call this from descriptor/config code if desired.
        """
        if not self.enable_weighting:
            return
        if self.epsilon is None:
            raise ValueError("enable_weighting=True requires eiml.epsilon to be provided.")
        if not (0.0 < float(self.epsilon_alpha) <= 1.0):
            raise ValueError("epsilon_alpha must be in (0, 1].")

        # ensure all epsilon values are positive
        for k, v in self.epsilon.items():
            if float(v) <= 0.0:
                raise ValueError(f"eiml.epsilon['{k}'] must be positive, got {v}.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sigma": self.sigma,
            "sigma_by_species": self.sigma_by_species or {},
            "k_rcut": self.k_rcut,
            "omega_rel": self.omega_rel,
            "enable_weighting": self.enable_weighting,
            "epsilon": self.epsilon or {},
            "epsilon_alpha": self.epsilon_alpha,
        }


# ------------------------------------------------------------
# Backward compatibility (optional): SAFTParams
# Keep if older YAMLs / identity vector code still depends on it.
# If you truly want to drop SAFT entirely, you can delete this class
# and remove SAFT usage from config/identity/descriptor.
# ------------------------------------------------------------
@dataclass
class SAFTParams:
    sigma_saft: float
    epsilon: float
    m: float
    kappa: float
    eps_assoc: float
    omega_rel: float = 0.1
    extra: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sigma_saft": self.sigma_saft,
            "epsilon": self.epsilon,
            "m": self.m,
            "kappa": self.kappa,
            "eps_assoc": self.eps_assoc,
            "omega_rel": self.omega_rel,
            "extra": self.extra or {},
        }
