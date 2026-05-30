"""Subprocess helper: imports rustforge_port fresh, runs verify+bench, prints JSON.

Called by try_one_port after each maturin build to avoid C-extension reload limitations.

The target is described by a TargetSpec passed in via the RUSTFORGE_SPEC env var (a
path to a JSON file). The oracle is loaded from the spec (`spec.load_oracle()`) rather
than a hardcoded micrograd import. With no env var set, falls back to MICROGRAD_SPEC so
this stays runnable standalone.
"""
import sys, os, json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from target_spec import TargetSpec, MICROGRAD_SPEC

_spec_path = os.environ.get("RUSTFORGE_SPEC")
if _spec_path and os.path.exists(_spec_path):
    with open(_spec_path) as f:
        spec = TargetSpec.from_json(f.read())
else:
    spec = MICROGRAD_SPEC

oracle_fn = spec.load_oracle()

import importlib
rustforge_port = importlib.import_module(spec.module_name)

if hasattr(rustforge_port, "set_oracle"):
    rustforge_port.set_oracle(oracle_fn)


def rust_fn(inputs):
    return np.asarray(rustforge_port.run_rust(np.asarray(inputs, dtype=np.float64)))


from verify import verify
from bench import benchmark

v = verify(rust_fn, oracle_fn, spec)
if not v["passed"]:
    print(json.dumps({"passed": False, "speedup": 0.0, "error": v["error"]}))
    sys.exit(0)

b = benchmark(rust_fn, oracle_fn, spec)
print(json.dumps({"passed": True, **b, "error": None}))
