from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SOAPParams:
    """
    Parameters for DScribe SOAP.
    NOTE: rcut is Optional so EIML can set it dynamically as k_rcut * sigma.
    """
    species: List[str]
    rcut: Optional[float]          # allow None -> computed later (EIML)
    nmax: int
    lmax: int
    sigma: float
    periodic: bool
    average: str = "off"
    sparse: bool = False
    # DScribe supports a "weighting" dict; we expose it to YAML
    weighting: Optional[Dict[str, Any]] = None

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


@dataclass
class EIMLParams:
    """
    Experimentally-Informed ML parameters (EIML-v1).

    sigma:
      - global size scale (one-component) OR reference sigma for mixtures.
    sigma_by_species:
      - optional mapping e.g. {"O": 3.0, "H": 2.5}
      - used for center-based reduced coordinates if provided.
    k_rcut:
      - if soap.rcut is None, set effective cutoff as rcut = k_rcut * sigma_ref
    """
    sigma: Optional[float] = None
    sigma_by_species: Optional[Dict[str, float]] = None
    k_rcut: Optional[float] = None
    omega_rel: float = 0.1   # reduced Gaussian width ω* (dimensionless)

    def sigma_for_center(self, symbol: str) -> float:
        """
        Center-based sigma:
          - if sigma_by_species is provided and contains the symbol -> use it
          - else fall back to global sigma
        """
        if self.sigma_by_species is not None and symbol in self.sigma_by_species:
            return float(self.sigma_by_species[symbol])
        if self.sigma is None:
            raise ValueError(
                "EIMLParams requires either 'sigma' or 'sigma_by_species' to be set."
            )
        return float(self.sigma)

    def sigma_ref_for_rcut(self) -> float:
        """
        Reference sigma for computing dynamic cutoff.
        For mixtures: we use max(sigma_by_species.values()) as a safe shell-covering default.
        For one-component: global sigma.
        """
        if self.sigma_by_species:
            return float(max(self.sigma_by_species.values()))
        if self.sigma is None:
            raise ValueError(
                "EIMLParams requires 'sigma' (or sigma_by_species) to compute rcut."
            )
        return float(self.sigma)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sigma": self.sigma,
            "sigma_by_species": self.sigma_by_species or {},
            "k_rcut": self.k_rcut,
            "omega_rel": self.omega_rel,
        }


# ---- Backward compatibility (optional) ----
# SAFTParams so old YAMLs still parse.
# In EIML-v1 we may not use all these fields, but we don't break users.
@dataclass
class SAFTParams:
    sigma_saft: float
    epsilon: float
    m: float
    kappa: float
    eps_assoc: float
    omega_rel: float = 0.1          # reduced Gaussian width ω* (dimensionless)
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
