# candidate.py  —  MUTABLE. The seed is a literal copy of the reference (a known-correct floor).
import numpy as np


def solve(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = x[i] * x[i] + 1.0
    return out
