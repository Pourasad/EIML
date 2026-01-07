from __future__ import annotations
import numpy as np
from .params import SAFTParams


def build_identity_vector(saft: SAFTParams) -> np.ndarray:
    # Order matters: keep consistent across code + analysis
    # [epsilon, m, kappa, eps_assoc]
    return np.array([saft.epsilon, saft.m, saft.kappa, saft.eps_assoc], dtype=float)
