"""TargetSpec — the single source of truth for what the Rust pipeline ports.

This module removes the five hardcoded micrograd coupling points that used to be
scattered across mutate.py / verify.py / bench.py / _validate.py / run_stage2.py.
Everything the pipeline needs to port *one* function now lives on a TargetSpec:

  * what to port        — `source` (the Python function) + `description`
  * how to check it     — `oracle_path` / `oracle_source` (the reference impl)
  * the numeric profile — `input_shape`, `input_low/high`, `fixed_input`, tolerances
  * the Rust shell      — `module_name` (fixed) + `rust_signature` skeleton

Swarm A (analyze.py) hands back {name, source, file, ...}; `TargetSpec.from_swarm_a`
turns that into a spec the Rust ratchet can consume. The micrograd MLP is now just
one spec among many (`MICROGRAD_SPEC`) rather than baked into every file.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

# The #[pymodule] name is fixed for the whole project — it must match the
# [lib] name in rust_template/Cargo.toml. We do NOT generalize this.
DEFAULT_MODULE_NAME = "rustforge_port"

# Default Rust signature skeleton: a 1-D float64 array in, a 1-D float64 array out.
# This is the profile Swarm A's scoring already favors (pure, numeric, deterministic
# array functions), so it stays the demo default.
DEFAULT_RUST_SIGNATURE = (
    "fn run_rust<'py>(\n"
    "    py: Python<'py>,\n"
    "    inputs: PyReadonlyArray1<f64>,\n"
    ") -> PyResult<Bound<'py, PyArray1<f64>>>"
)

# Signature used by the Modal Rust swarm (optimization/rust_solve crate): the exported
# function is `solve`, returning a Bound array directly (matches the crate that the
# swarm image warm-builds).
SOLVE_RUST_SIGNATURE = (
    "fn solve<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) "
    "-> Bound<'py, PyArray1<f64>>"
)


@dataclass
class TargetSpec:
    """Everything the Rust ratchet needs to port and verify one function."""

    name: str
    description: str
    source: str  # Python source of the function being ported (fed to the LLM)

    # Reference oracle — supply exactly one of these:
    #   oracle_path:   "package.module:attr" — imported in the validation subprocess.
    #   oracle_source: a code string that, when exec'd, defines `oracle_entry`.
    oracle_path: Optional[str] = None
    oracle_source: Optional[str] = None
    oracle_entry: Optional[str] = None

    # Numeric I/O profile used by the correctness gate (verify) and benchmark.
    input_shape: tuple[int, ...] = (2,)
    input_low: float = -1.0
    input_high: float = 1.0
    fixed_input: Optional[list[float]] = None  # bench input; defaults from shape if None

    # Rust shell.
    module_name: str = DEFAULT_MODULE_NAME
    fn_name: str = "run_rust"            # exported Python/Rust function name
    rust_signature: str = DEFAULT_RUST_SIGNATURE
    seed_rust: Optional[str] = None      # explicit naive seed; generated if None

    # Differential-gate tolerances.
    rtol: float = 1e-5
    atol: float = 1e-6

    # Input generation. Two modes:
    #   array_mode=False → fixed-shape inputs (e.g. micrograd's (2,)); fuzz over
    #     input_shape with elements in [input_low, input_high]; bench on fixed_input.
    #   array_mode=True  → variable-length 1-D arrays (elementwise numeric ops, the
    #     Swarm-A / toy profile); fuzz over random lengths; bench on a large array.
    array_mode: bool = False
    fuzz_max_len: int = 5000             # max length of a fuzz array in array_mode
    bench_input_size: int = 1_000_000    # bench array length in array_mode

    # ── Oracle resolution ────────────────────────────────────────────────────
    def load_oracle(self) -> Callable:
        """Return the reference Python callable, importing or exec'ing as needed.

        Importable in a fresh subprocess (see _validate.py) — this is what makes the
        oracle a *parameter* of the pipeline instead of a hardcoded import.
        """
        if self.oracle_path:
            module_name, _, attr = self.oracle_path.partition(":")
            if not attr:
                raise ValueError(
                    f"oracle_path must be 'module:attr', got {self.oracle_path!r}"
                )
            module = importlib.import_module(module_name)
            return getattr(module, attr)

        if self.oracle_source:
            entry = self.oracle_entry or self.name
            import numpy as np

            ns: dict = {"np": np, "numpy": np}  # common deps for self-ported numeric fns
            exec(self.oracle_source, ns)  # trusted locally; Modal sandboxes it in the swarm
            if entry not in ns:
                raise ValueError(
                    f"oracle_source does not define {entry!r} "
                    f"(available: {sorted(k for k in ns if not k.startswith('__'))})"
                )
            return ns[entry]

        raise ValueError(
            f"TargetSpec {self.name!r} has no oracle — set oracle_path or oracle_source"
        )

    # ── Numeric helpers ──────────────────────────────────────────────────────
    def default_fixed_input(self) -> list[float]:
        """The fixed benchmark input — explicit `fixed_input`, or the shape midpoint."""
        if self.fixed_input is not None:
            return list(self.fixed_input)
        import numpy as np

        n = int(np.prod(self.input_shape))
        mid = (self.input_low + self.input_high) / 2.0
        return [float(mid)] * n

    def make_gate_inputs(self, rng, n: int = 20) -> list:
        """Random inputs for the differential correctness gate.

        These are generated at evaluation time and never shown to the LLM — the
        anti-cheat moat. In array_mode they are variable-length 1-D arrays; otherwise
        they match the fixed input_shape over [input_low, input_high].
        """
        if self.array_mode:
            return [
                rng.standard_normal(int(rng.integers(1, self.fuzz_max_len)))
                for _ in range(n)
            ]
        return [
            rng.uniform(self.input_low, self.input_high, size=self.input_shape).astype(
                "float64"
            )
            for _ in range(n)
        ]

    def make_bench_input(self):
        """The fixed benchmark input (identical for both Python and Rust sides)."""
        import numpy as np

        if self.array_mode:
            rng = np.random.default_rng(0)
            return rng.standard_normal(self.bench_input_size)
        return np.array(self.default_fixed_input(), dtype=np.float64).reshape(
            self.input_shape
        )

    def seed(self) -> str:
        """The naive Rust seed (explicit `seed_rust`, or generated for this spec)."""
        return self.seed_rust if self.seed_rust is not None else naive_seed(self)

    # ── Subprocess (JSON) boundary ─────────────────────────────────────────────
    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "TargetSpec":
        data = json.loads(text)
        if data.get("input_shape") is not None:
            data["input_shape"] = tuple(data["input_shape"])  # JSON lists → tuple
        return cls(**data)

    # ── Swarm A handoff ─────────────────────────────────────────────────────────
    @classmethod
    def from_swarm_a(
        cls,
        result: dict,
        *,
        oracle_path: Optional[str] = None,
        oracle_source: Optional[str] = None,
        oracle_entry: Optional[str] = None,
        description: Optional[str] = None,
        **overrides,
    ) -> "TargetSpec":
        """Build a TargetSpec from a Swarm A worker result.

        `result` is one entry from analyze.py: {name, score, features, source, file}.
        The oracle is provided by the caller (the driver decides how the reference is
        reached): pass `oracle_path` for a curated/importable reference, or
        `oracle_source` to build one from a self-contained function. If neither is
        given, the function's own source is used as the oracle (entry = its name),
        which works for the pure, self-contained numeric functions Swarm A favors.
        """
        name = result["name"]
        source = result["source"]

        if oracle_path is None and oracle_source is None:
            # Self-port: the picked function IS the reference. Valid only when the
            # source is self-contained — Swarm A's purity/determinism scoring filters
            # for exactly this. `oracle_entry` defaults to the bare function name
            # (methods come through as "Class.method", so allow an override).
            oracle_source = source
            oracle_entry = oracle_entry or name.split(".")[-1]

        if description is None:
            description = (
                f"Port the Python function `{name}` to Rust. "
                "It is pure, numeric, and deterministic; preserve its exact numerical "
                "behavior."
            )

        return cls(
            name=name,
            description=description,
            source=source,
            oracle_path=oracle_path,
            oracle_source=oracle_source,
            oracle_entry=oracle_entry,
            **overrides,
        )


def naive_seed(spec: TargetSpec) -> str:
    """Generate the known-correct naive Rust seed for `spec`.

    The seed delegates the exported function straight back to the Python oracle through
    the GIL. It always passes the correctness gate (it *is* the oracle) and benchmarks
    at ~1x, giving the ratchet a correct floor to improve on. `set_oracle()` must be
    called before the function runs. The #[pymodule] name and exported function name are
    templated from the spec (e.g. rustforge_port/run_rust locally, rust_solve/solve on
    the Modal swarm).
    """
    fn = spec.fn_name
    return f"""\
// Naive seed: {fn} delegates to the Python oracle via the GIL.
// Proves the pipeline compiles and the correctness gate passes.
// set_oracle() must be called before {fn}().
use numpy::{{IntoPyArray, PyArray1, PyReadonlyArray1}};
use pyo3::prelude::*;
use std::sync::Mutex;

static ORACLE_FN: Mutex<Option<Py<PyAny>>> = Mutex::new(None);

#[pyfunction]
fn set_oracle(oracle: Py<PyAny>) {{
    *ORACLE_FN.lock().unwrap() = Some(oracle);
}}

#[pyfunction]
fn {fn}<'py>(
    py: Python<'py>,
    inputs: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {{
    let oracle = {{
        let guard = ORACLE_FN.lock().unwrap();
        guard
            .as_ref()
            .ok_or_else(|| {{
                pyo3::exceptions::PyRuntimeError::new_err("oracle not set — call set_oracle() first")
            }})?
            .clone_ref(py)
    }};
    let input_arr = inputs.as_array().to_owned().into_pyarray(py);
    let result = oracle.bind(py).call1((input_arr,))?;
    result
        .cast_into::<PyArray1<f64>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("oracle must return a 1-D f64 array"))
}}

#[pymodule]
fn {spec.module_name}(m: &Bound<'_, PyModule>) -> PyResult<()> {{
    m.add_function(wrap_pyfunction!(set_oracle, m)?)?;
    m.add_function(wrap_pyfunction!({fn}, m)?)?;
    Ok(())
}}
"""


# ── The micrograd MLP as one concrete spec (the former hardcoded default) ────────
# Was the implicit, baked-in target across the whole pipeline. Now it is just the
# default spec run_stage2.py falls back to when no Swarm A candidate is supplied.
_MICROGRAD_DESCRIPTION = (
    "Port the micrograd Value forward+backward MLP workload to Rust using PyO3. "
    "The Rust function must be named run_rust, accept a 1-D numpy array of 2 float64 "
    "inputs (PyReadonlyArray1<f64>), and return a 1-D numpy array of 3 float64 values: "
    "[x0.grad, x1.grad, output.data] (Bound<'py, PyArray1<f64>>). "
    "The MLP is 2->[4 ReLU neurons]->[1 linear neuron]. "
    "The weights must match exactly what Python's random.seed(42) + MLP(2,[4,1]) produces. "
    "Implement it directly with scalars — do NOT build a Value class.\n\n"
    "Forward pass (hardcoded weights):\n"
    "  z0 =  0.27885360*x0 + (-0.94997849)*x1\n"
    "  z1 = -0.44994136*x0 + (-0.55357852)*x1\n"
    "  z2 =  0.47294243*x0 +   0.35339897*x1\n"
    "  z3 =  0.78435914*x0 + (-0.82612233)*x1\n"
    "  h{i} = relu(z{i})  // relu(v) = if v > 0.0 { v } else { 0.0 }\n"
    "  out = (-0.15615636)*h0 + (-0.94040556)*h1 + (-0.56272405)*h2 + 0.01071058*h3\n\n"
    "Backward pass (d_out = 1.0):\n"
    "  d_h{i} = w_out{i} * 1.0\n"
    "  d_z{i} = d_h{i} * (if z{i} > 0.0 { 1.0 } else { 0.0 })\n"
    "  x0_grad =  0.27885360*d_z0 + (-0.44994136)*d_z1 + 0.47294243*d_z2 + 0.78435914*d_z3\n"
    "  x1_grad = -0.94997849*d_z0 + (-0.55357852)*d_z1 + 0.35339897*d_z2 + (-0.82612233)*d_z3\n\n"
    "Return numpy array [x0_grad, x1_grad, out] as float64."
)

MICROGRAD_SPEC = TargetSpec(
    name="micrograd_mlp",
    description=_MICROGRAD_DESCRIPTION,
    source="",  # the description fully specifies the computation for this curated target
    oracle_path="targets.micrograd.oracle:run_reference",
    input_shape=(2,),
    input_low=-1.0,
    input_high=1.0,
    fixed_input=[0.5, -0.3],
)
