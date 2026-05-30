# target.py  —  FROZEN. Do not let the agent edit this.
import numpy as np

RNG_SEED = 0
INPUT_SIZE = 1_000_000


def reference(x: np.ndarray) -> np.ndarray:
    """The slow, obviously-correct Python version. Ground truth."""
    out = np.empty_like(x)
    for i in range(x.shape[0]):          # deliberately naive: a hot Python loop
        out[i] = x[i] * x[i] + 1.0
    return out


def make_inputs(n: int = INPUT_SIZE) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED)
    return rng.standard_normal(n)
