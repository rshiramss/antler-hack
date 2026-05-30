"""Subprocess helper: imports rustforge_port fresh, runs verify+bench, prints JSON.

Called by try_one_port after each maturin build to avoid C-extension reload limitations.
"""
import sys, os, json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from targets.micrograd.oracle import run_reference as oracle_fn
import rustforge_port

if hasattr(rustforge_port, "set_oracle"):
    rustforge_port.set_oracle(oracle_fn)


def rust_fn(inputs):
    return np.asarray(rustforge_port.run_rust(np.asarray(inputs, dtype=np.float64)))


from verify import verify
from bench import benchmark

v = verify(rust_fn, oracle_fn)
if not v["passed"]:
    print(json.dumps({"passed": False, "speedup": 0.0, "error": v["error"]}))
    sys.exit(0)

b = benchmark(rust_fn, oracle_fn)
print(json.dumps({"passed": True, **b, "error": None}))
