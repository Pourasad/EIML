"""
soap_saft/descriptor.py

EIML-v1 goals:
- Standard SOAP (mode="soap")
- EIML SOAP (mode="eiml"): reduced coordinates r -> r / sigma_ref
  + Dynamic cutoff: if soap.rcut is None, use r_cut = k_rcut * sigma_ref
  + Energy-aware density proxy: pass DScribe weighting=... from YAML

Backward compatibility:
- mode="saft" is treated as alias of "eiml"
- SAFT identity vector support kept (epsilon, m, kappa, eps_assoc)
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
from .weights import apply_species_pair_scaling, normalize_species_epsilon
import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP as DScribeSOAP

from .params import SOAPParams, SAFTParams, EIMLParams
from .identity import build_identity_vector


def _to_dense(x) -> np.ndarray:
    """Convert possible sparse output from dscribe to dense numpy array."""
    if hasattr(x, "toarray"):
        return x.toarray().astype(float)
    return np.asarray(x, dtype=float)


def _scale_atoms(atoms: Atoms, scale: float) -> Atoms:
    """
    Return a copy of atoms with scaled coordinates:
      r -> r / scale
    If periodic cell exists, also scale the cell.
    """
    s = float(scale)
    if s <= 0.0:
        raise ValueError("scale must be > 0")

    atoms2 = atoms.copy()
    atoms2.positions = atoms2.positions / s

    # Scale cell only if it exists
    cell = atoms2.get_cell()
    if cell is not None and cell.rank > 0:
        atoms2.set_cell(cell / s, scale_atoms=False)

    return atoms2


class SOAPSAFT:
    """
    Descriptor engine (kept name for backward compatibility).
    Use .create(...) to compute feature vectors.

    Modes:
      - "soap": standard SOAP in physical units
      - "eiml": reduced coordinates + dynamic cutoff + optional weighting
      - "saft": alias of "eiml" (legacy)
    """

    def __init__(
        self,
        soap_params: SOAPParams,
        mode: str = "soap",
        eiml_params: Optional[EIMLParams] = None,
        saft_params: Optional[SAFTParams] = None,   # legacy identity support
    ):
        self.soap_params = soap_params

        self.mode = str(mode).strip().lower()
        if self.mode == "saft":
            self.mode = "eiml"

        if self.mode not in ("soap", "eiml"):
            raise ValueError("mode must be 'soap', 'eiml' (or legacy 'saft').")

        self.eiml_params = eiml_params
        self.saft_params = saft_params  # optional, for identity vector only
        # Debug guard: ensure epsilon weights printed once per run
        self._printed_epsilon_weights = False

        # ----------------------------
        # Determine effective length scale and cutoff
        # ----------------------------
        if self.mode == "soap":
            # Physical units (standard SOAP)
            sigma_ref = None

            if soap_params.rcut is None:
                raise ValueError(
                    "In mode='soap', soap.rcut must be provided (e.g. 5.0). "
                    "Dynamic cutoff is only available in mode='eiml'."
                )

            r_cut_eff = float(soap_params.rcut)
            sigma_eff = float(soap_params.sigma)

        else:
            # EIML mode requires sigma information (for reduced coordinates)
            if self.eiml_params is None:
                raise ValueError("eiml_params is required when mode='eiml' (or legacy 'saft').")

            sigma_ref = float(self.eiml_params.sigma_ref_for_rcut())

            # Dynamic cutoff: if soap.rcut not provided, compute from k_rcut * sigma_ref
            if soap_params.rcut is None:
                if self.eiml_params.k_rcut is None:
                    raise ValueError("In mode='eiml', either soap.rcut or eiml.k_rcut must be provided.")
                rcut_phys = float(self.eiml_params.k_rcut) * sigma_ref
            else:
                rcut_phys = float(soap_params.rcut)

            # In reduced coordinates: divide cutoff by sigma_ref
            r_cut_eff = rcut_phys / sigma_ref

            # --- EIML invariant check ---
            # Because rcut_phys = k_rcut * sigma_ref (when using dynamic cutoff),
            # the reduced cutoff must be exactly k_rcut.
            if soap_params.rcut is None:  # only when dynamic cutoff is active
                assert abs(r_cut_eff - float(self.eiml_params.k_rcut)) < 1e-8, \
                    "EIML invariant broken: reduced r_cut must equal k_rcut"

            # Reduced Gaussian width (dimensionless)
            # Preferred: omega_rel (dimensionless). If missing, fallback for backward compatibility.
            if getattr(self.eiml_params, "omega_rel", None) is not None:
                sigma_eff = float(self.eiml_params.omega_rel)
            else:
                sigma_eff = float(soap_params.sigma) / sigma_ref
        # ----------------------------
        # Build DScribe SOAP
        # DScribe signature:
        # SOAP(r_cut, n_max, l_max, sigma, ... , weighting=..., average=..., species=..., periodic=...)
        # ----------------------------
        soap_kwargs = dict(
            n_max=int(soap_params.nmax),
            l_max=int(soap_params.lmax),
            sigma=float(sigma_eff),
            species=soap_params.species,
            periodic=bool(soap_params.periodic),
            average=str(soap_params.average),
            sparse=bool(soap_params.sparse),
        )

        # Optional DScribe weighting (EIML-v1 energy-aware proxy)
        if soap_params.weighting is not None:
            soap_kwargs["weighting"] = dict(soap_params.weighting)

        # r_cut keyword is r_cut for the DScribe version; keep shim just in case
        # Decide effective cutoff
#        if self.mode == "soap":
#            if soap_params.rcut is None:
#                raise ValueError(
#                    "In mode='soap', you must set soap.rcut in the YAML (e.g., soap: {rcut: 5.0})."
#                )
#            r_cut_eff = float(soap_params.rcut)
#
#        elif self.mode == "eiml":
#            # dynamic cutoff: r_cut = k_rcut * sigma
#            r_cut_eff = float(self.eiml_params.k_rcut) * float(self.eiml_params.sigma)
#
#        else:
#            raise ValueError("mode must be 'soap' or 'eiml'")
        try:
            self._soap = DScribeSOAP(r_cut=float(r_cut_eff), **soap_kwargs)
        except TypeError:
            # very old DScribe fallback
            self._soap = DScribeSOAP(rcut=float(r_cut_eff), **soap_kwargs)

        # Identity vector exists only if legacy SAFT params are provided (optional)
        self._identity = build_identity_vector(self.saft_params) if self.saft_params is not None else None

        # Store the scaling used for reduced coords (None in SOAP mode)
        self._sigma_ref = sigma_ref

    def create(
        self,
        atoms: Atoms,
        centers=None,
        pool: Optional[str] = None,
        append_identity: bool = True,
        return_identity: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Compute descriptor.

        Args:
            atoms: ASE Atoms.
            centers: Passed through to DScribe SOAP.create (None => all atoms).
            pool:
                - None   => return per-center features (2D) when DScribe returns 2D
                - "mean" => mean over centers (global vector)
                - "sum"  => sum over centers (global vector)
            append_identity:
                If return_identity=False and identity vector exists:
                  - True  => concatenate identity to output (backward compatible)
                  - False => geometry only
            return_identity:
                - False: returns a single array (geometry or geometry+identity)
                - True : returns (X_geom, theta_id). Identity is NOT tiled per-atom.

        Returns:
            If return_identity=False:
                - pooled   => 1D array
                - unpooled => 2D array (n_centers, n_features) OR 1D depending on DScribe settings
            If return_identity=True:
                (X_geom, theta_id)
        """
        if pool is not None:
            pool = str(pool).strip().lower()
            if pool not in ("mean", "sum"):
                raise ValueError("pool must be None, 'mean', or 'sum'.")

        # Reduced coordinates affect geometry in EIML mode
        if self.mode == "soap":
            atoms_use = atoms
        else:
            atoms_use = _scale_atoms(atoms, self._sigma_ref)

        # Geometry-only feature(s)
        feat = _to_dense(self._soap.create(atoms_use, centers=centers))

        # EIML v1.1: epsilon-based channel weighting (normalized internally)
        if (
            self.mode == "eiml"
            and self.eiml_params is not None
            and getattr(self.eiml_params, "enable_weighting", False)
        ):
            species_weights = normalize_species_epsilon(
                epsilon=self.eiml_params.epsilon,
                species=self.soap_params.species,
                alpha=float(getattr(self.eiml_params, "epsilon_alpha", 1.0)),
            )

            # ---- DEBUG PRINT (once per run) ----
            if not self._printed_epsilon_weights:
                print("[EIML] Normalized epsilon-based species weights:")
                for sp, w in species_weights.items():
                    print(f"  {sp}: {w:.6f}")
                print(f"[EIML] epsilon_alpha = {self.eiml_params.epsilon_alpha}")
                self._printed_epsilon_weights = True
            # -----------------------------------

            feat = apply_species_pair_scaling(
                soap_obj=self._soap,
                X=feat,
                species=self.soap_params.species,
                species_weights=species_weights,
            )

        theta = self._identity

        # Pooling (only meaningful if we have per-center 2D output)
        if pool is not None and feat.ndim != 1:
            feat_pooled = feat.mean(axis=0) if pool == "mean" else feat.sum(axis=0)
        else:
            feat_pooled = feat

        # Return split outputs (recommended for kernels)
        if return_identity:
            return feat_pooled, theta

        # Backward-compatible concatenation (NOT recommended for kernel distances)
        if theta is not None and append_identity:
            if feat_pooled.ndim == 1:
                feat_pooled = np.concatenate([feat_pooled, theta], axis=0)
            else:
                # If pooled is 2D (only possible if pool is None), tile identity per row
                n = feat_pooled.shape[0]
                feat_pooled = np.concatenate([feat_pooled, np.tile(theta, (n, 1))], axis=1)

        return feat_pooled


def compute_descriptor(
    atoms: Union[Atoms, list],
    mode: str,
    soap_params: SOAPParams,
    eiml_params: Optional[EIMLParams] = None,
    saft_params: Optional[SAFTParams] = None,   # legacy identity support
    pool: Optional[str] = None,
    append_identity: bool = True,
    return_identity: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Convenience wrapper around SOAPSAFT.create(...).

    Supports:
      - single ASE Atoms -> returns (nfeat,) if pool is not None, else (natoms, nfeat)
      - list/tuple of Atoms (trajectory) -> returns (nframes, nfeat) if pool is not None
        (or (nframes, natoms, nfeat) if pool is None)
    """
    calc = SOAPSAFT(
        soap_params=soap_params,
        mode=mode,
        eiml_params=eiml_params,
        saft_params=saft_params,
    )

    # --- Trajectory case ---
    if isinstance(atoms, (list, tuple)):
        X_list = []
        theta = None

        for at in atoms:
            out = calc.create(
                atoms=at,
                pool=pool,
                append_identity=append_identity,
                return_identity=return_identity,
            )

            if return_identity:
                xg, th = out
                X_list.append(np.asarray(xg))
                theta = th
            else:
                X_list.append(np.asarray(out))

        X = np.stack(X_list, axis=0)
        if return_identity:
            return X, theta
        return X

    # --- Single frame case ---
    return calc.create(
        atoms=atoms,
        pool=pool,
        append_identity=append_identity,
        return_identity=return_identity,
    )
