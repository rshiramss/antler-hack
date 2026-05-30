import random
import numpy as np

from .engine import Value
from .nn import MLP

# Fixed MLP with deterministic weights — do not change this seed.
random.seed(42)
_mlp = MLP(2, [4, 1])


def run_reference(inputs: np.ndarray) -> np.ndarray:
    """Forward + backward on a fixed 2->[4]->1 MLP.

    Returns np.array([x0.grad, x1.grad, output.data], dtype=float64).
    """
    _mlp.zero_grad()
    x = [Value(float(inputs[0])), Value(float(inputs[1]))]
    out = _mlp(x)
    out.backward()
    return np.array([x[0].grad, x[1].grad, out.data], dtype=np.float64)
