from __future__ import annotations

from typing import Any, Dict, Optional, Union, List
from ase import Atoms
from ase.io import read
from ase.data import atomic_numbers


def _apply_type_map_one(atoms: Atoms, type_map: Optional[Dict[Any, str]]) -> Atoms:
    """
    Apply LAMMPS type -> chemical symbol mapping.

    type_map example:
      {"1": "H", "2": "O"}  (keys can be str or int)

    We try to locate LAMMPS "type" information in common ASE fields:
      - atoms.arrays["type"]
      - atoms.arrays["types"]
      - atoms.arrays["atom_types"]
    If not present, we fall back to using atoms.numbers as type ids
    *only if* they look like they match the type_map keys.
    """
    if type_map is None:
        return atoms

    tm: Dict[int, str] = {int(k): str(v) for k, v in type_map.items()}

    a = atoms.copy()

    # 1) Prefer explicit per-atom type arrays if present
    types_arr = None
    for key in ("type", "types", "atom_types"):
        if key in a.arrays:
            types_arr = a.arrays[key]
            break

    if types_arr is not None:
        types = [int(x) for x in types_arr]
    else:
        # 2) Fallback: interpret atomic numbers as type ids if they match mapping keys
        nums = [int(x) for x in a.get_atomic_numbers()]
        # If most numbers are in tm keys, treat them as type ids
        in_keys = sum(1 for x in nums if x in tm)
        if in_keys >= max(1, int(0.9 * len(nums))):
            types = nums
        else:
            # Looks like atomic numbers are already real Z values; do nothing
            return a

    # Map type -> symbol
    try:
        symbols = [tm[t] for t in types]
    except KeyError as e:
        raise ValueError(
            f"type_map is missing a key for LAMMPS type {int(e.args[0])}. "
            f"Available keys: {sorted(tm.keys())}"
        )

    # Convert symbol -> atomic number and set on Atoms
    try:
        numbers = [atomic_numbers[s] for s in symbols]
    except KeyError as e:
        raise ValueError(
            f"type_map maps to unknown chemical symbol '{e.args[0]}'. "
            "Check spelling/case (e.g. 'Si', 'O', 'H')."
        )

    a.set_atomic_numbers(numbers)
    # Keep symbols consistent too (ASE will derive symbols from Z, but explicit is fine)
    a.set_chemical_symbols(symbols)

    return a


def read_structure_from_cfg(structure_cfg: Dict[str, Any]) -> Union[Atoms, List[Atoms]]:
    """
    Read structure/trajectory using ASE, with optional LAMMPS type_map support.

    structure_cfg keys (expected):
      input:  path
      format: ASE format string or None
      index:  int | ":" | "start:stop:step"  (ASE supports these)
      type_map: optional mapping {"1": "H", "2": "O", ...}
    """
    path = str(structure_cfg["input"])
    fmt = structure_cfg.get("format", None)
    index = structure_cfg.get("index", 0)
    type_map = structure_cfg.get("type_map", None)

    atoms = read(path, format=fmt, index=index)

    # Apply type_map to single frame or trajectory
    if isinstance(atoms, list):
        return [_apply_type_map_one(a, type_map) for a in atoms]
    return _apply_type_map_one(atoms, type_map)
