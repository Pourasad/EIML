"""
EIML-v1: Experimentally Informed Machine Learning descriptors.

This package provides:
- SOAP / EIML descriptor computation (DScribe-backed)
- Optional SAFT-based identity vector
- CLI + YAML-based configuration
"""

from __future__ import annotations

__version__ = "0.1.0"

# Import submodules so `eiml.config`, `eiml.descriptor`, etc. are available
from . import  config, descriptor, io_structures, params, save, identity  # noqa: F401

# Re-export commonly used API
from .descriptor import SOAPSAFT, compute_descriptor  # noqa: F401
from .config import load_config_yaml, params_from_config  # noqa: F401
from .io_structures import read_structure_from_cfg  # noqa: F401
from .save import save_features  # noqa: F401

__all__ = [
    "__version__",
    # submodules
    "cli",
    "config",
    "descriptor",
    "io_structures",
    "params",
    "save",
    "identity",
    # public API
    "SOAPSAFT",
    "compute_descriptor",
    "load_config_yaml",
    "params_from_config",
    "read_structure_from_cfg",
    "save_features",
]
