from __future__ import annotations

from typing import Any, Dict, Optional, Union, List, Mapping, Sequence
import numpy as np
from ase import Atoms
from ase.io import read
from ase.data import atomic_numbers


def _normalize_type_map(type_map: Mapping[Any, Any]) -> Dict[int, str]:
    """
    Normalize a user-provided LAMMPS type_map to {int_type: "Symbol"}.

    Accepts keys as int/str, values as str/bytes.
    """
    tm: Dict[int, str] = {}
    for k, v in type_map.items():
        kt = int(k)
        sv = v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
        tm[kt] = sv
    return tm


def _find_type_array(atoms: Atoms) -> Optional[np.ndarray]:
    """
    Try to locate a per-atom LAMMPS type array in common ASE fields.
    Returns the array if found, else None.
    """
    for key in ("type", "types", "atom_types"):
        if key in atoms.arrays:
            arr = atoms.arrays[key]
            return np.asarray(arr)
    return None


def _looks_like_type_ids(nums: Sequence[int], tm: Dict[int, str], frac: float = 0.9) -> bool:
    """
    Heuristic: interpret atoms.numbers as *type ids* only if most entries are mapping keys.
    """
    if len(nums) == 0:
        return False
    in_keys = sum(1 for x in nums if x in tm)
    return in_keys >= max(1, int(frac * len(nums)))


def apply_type_map(atoms: Atoms, type_map: Optional[Mapping[Any, Any]]) -> Atoms:
    """
    Apply a LAMMPS type -> chemical symbol mapping to an ASE Atoms object.

    type_map example:
        {"1": "H", "2": "O"}  (keys can be str or int)

    We try, in order:
      1) atoms.arrays["type"] / ["types"] / ["atom_types"]
      2) atoms.get_atomic_numbers() as type ids *only if* they look like mapping keys

    If we can't find types, or atomic numbers look like real Z values, we return atoms unchanged.
    """
    if type_map is None:
        return atoms

    tm = _normalize_type_map(type_map)
    a = atoms.copy()

    types_arr = _find_type_array(a)
    if types_arr is not None:
        # Convert to python ints (handles numpy dtypes)
        types = [int(x) for x in types_arr.tolist()]
    else:
        # Fallback: treat atomic numbers as type ids only if they match mapping keys
        nums = [int(x) for x in a.get_atomic_numbers()]
        if _looks_like_type_ids(nums, tm, frac=0.9):
            types = nums
        else:
            return a  # atomic numbers likely already real Z values

    # Map type -> symbol
    try:
        symbols = [tm[t] for t in types]
    except KeyError as e:
        missing = int(e.args[0])
        raise ValueError(
            f"type_map is missing a key for LAMMPS type {missing}. "
            f"Available keys: {sorted(tm.keys())}"
        ) from None

    # Validate symbol -> atomic number and set on Atoms
    try:
        numbers = [atomic_numbers[s] for s in symbols]
    except KeyError as e:
        bad = str(e.args[0])
        raise ValueError(
            f"type_map maps to unknown chemical symbol '{bad}'. "
            "Check spelling/case (e.g. 'Si', 'O', 'H')."
        ) from None

    a.set_atomic_numbers(numbers)
    # Redundant but explicit:
    a.set_chemical_symbols(symbols)
    return a


def read_structure_from_cfg(structure_cfg: Dict[str, Any]) -> Union[Atoms, List[Atoms]]:
    """
    Read a structure/trajectory using ASE, with optional LAMMPS type_map support.

    structure_cfg expected keys:
      input:    path to file (required)
      format:   ASE format string (optional)
      index:    int | ":" | "start:stop:step"  (optional; ASE supports these)
      type_map: optional mapping {"1": "H", "2": "O", ...}
    """
    if "input" not in structure_cfg:
        raise KeyError("structure_cfg must contain 'input'")

    path = str(structure_cfg["input"])
    fmt = structure_cfg.get("format", None)
    index = structure_cfg.get("index", 0)
    type_map = structure_cfg.get("type_map", None)

    atoms = read(path, format=fmt, index=index)

    # Apply type_map to single frame or trajectory
    if isinstance(atoms, list):
        return [apply_type_map(a, type_map) for a in atoms]
    return apply_type_map(atoms, type_map)
