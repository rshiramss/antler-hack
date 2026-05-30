"""Stage 2 runner: thin end-to-end slice.

Flow:
  1. Check OPENAI_API_KEY
  2. Build the naive Rust seed (call-through-GIL, always correct)
  3. Smoke-test the naive seed
  4. Call try_one_port (LLM mutate → build → verify → benchmark)
  5. Print result and STAGE 2 GATE: PASS / FAIL
"""
import os
import sys
import subprocess
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

NAIVE_SEED = """\
// Naive seed: run_rust delegates to the Python oracle via the GIL.
// Proves the pipeline compiles and the correctness gate passes.
// set_oracle() must be called before run_rust().
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::sync::Mutex;

static ORACLE_FN: Mutex<Option<Py<PyAny>>> = Mutex::new(None);

#[pyfunction]
fn set_oracle(oracle: Py<PyAny>) {
    *ORACLE_FN.lock().unwrap() = Some(oracle);
}

#[pyfunction]
fn run_rust<'py>(
    py: Python<'py>,
    inputs: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let oracle = {
        let guard = ORACLE_FN.lock().unwrap();
        guard
            .as_ref()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("oracle not set — call set_oracle() first")
            })?
            .clone_ref(py)
    };
    let input_arr = inputs.as_array().to_owned().into_pyarray(py);
    let result = oracle.bind(py).call1((input_arr,))?;
    result
        .cast_into::<PyArray1<f64>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("oracle must return a 1-D f64 array"))
}

#[pymodule]
fn rustforge_port(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_oracle, m)?)?;
    m.add_function(wrap_pyfunction!(run_rust, m)?)?;
    Ok(())
}
"""


def _maturin_build(lib_rs: str) -> tuple[bool, str]:
    lib_path = os.path.join(PROJECT_ROOT, "rust_template", "src", "lib.rs")
    with open(lib_path, "w") as f:
        f.write(lib_rs)
    venv_bin = os.path.join(PROJECT_ROOT, ".venv", "bin")
    env = os.environ.copy()
    env["PATH"] = f"{venv_bin}:{os.path.expanduser('~/.cargo/bin')}:{env.get('PATH', '')}"
    env["VIRTUAL_ENV"] = os.path.join(PROJECT_ROOT, ".venv")
    proc = subprocess.run(
        ["maturin", "develop", "--release"],
        cwd=os.path.join(PROJECT_ROOT, "rust_template"),
        capture_output=True, text=True, env=env,
    )
    return proc.returncode == 0, proc.stderr


def main():
    # ── Preflight ────────────────────────────────────────────────────────────
    from mutate import _load_openai_key
    try:
        _load_openai_key()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # ── Build naive seed ─────────────────────────────────────────────────────
    print("=== Building naive seed ===")
    ok, stderr = _maturin_build(NAIVE_SEED)
    if not ok:
        print(f"NAIVE SEED BUILD FAILED:\n{stderr}")
        sys.exit(1)
    print("Naive seed built OK.")

    # ── Smoke-test naive seed in a subprocess (fresh .so load) ───────────────
    print("\n=== Smoke-testing naive seed ===")
    venv_python = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
    venv_bin = os.path.join(PROJECT_ROOT, ".venv", "bin")
    env = os.environ.copy()
    env["PATH"] = f"{venv_bin}:{os.path.expanduser('~/.cargo/bin')}:{env.get('PATH', '')}"
    env["VIRTUAL_ENV"] = os.path.join(PROJECT_ROOT, ".venv")

    smoke = subprocess.run(
        [venv_python, "-c", f"""
import sys, numpy as np
sys.path.insert(0, {repr(PROJECT_ROOT)})
from targets.micrograd.oracle import run_reference as oracle_fn
import rustforge_port
rustforge_port.set_oracle(oracle_fn)
inp = np.array([0.5, -0.3], dtype=np.float64)
r = np.asarray(rustforge_port.run_rust(inp))
o = oracle_fn(inp)
assert np.allclose(r, o, rtol=1e-5, atol=1e-6), f"smoke FAIL: {{r}} vs {{o}}"
print("Smoke test PASS — rust:", r, "oracle:", o)
"""],
        cwd=PROJECT_ROOT, capture_output=True, text=True, env=env,
    )
    if smoke.returncode != 0:
        print(f"SMOKE TEST FAILED:\n{smoke.stderr}")
        sys.exit(1)
    print(smoke.stdout.strip())

    # ── try_one_port with retry-on-compiler-error ────────────────────────────
    print("\n=== Running try_one_port (LLM mutate → build → verify → benchmark) ===")
    from targets.micrograd.oracle import run_reference as oracle_fn
    from try_one_port import try_one_port

    MAX_ATTEMPTS = 3
    result = None
    variant: dict = {}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n--- Attempt {attempt}/{MAX_ATTEMPTS} ---")
        result = try_one_port(
            parent_rust=NAIVE_SEED,
            target_fn=oracle_fn,
            oracle_fn=oracle_fn,
            variant=variant,
        )
        if result["passed"]:
            break
        # Feed the compiler/verify error back for the next attempt
        variant = {"compiler_error": result.get("error", "")[:2000]}
        print(f"  attempt {attempt} failed — retrying with error context")

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n=== Final Result ===")
    print(f"  passed:     {result['passed']}")
    print(f"  speedup:    {result.get('speedup', 0):.3f}x")
    if result.get("python_ms") is not None:
        print(f"  python_ms:  {result['python_ms']:.3f}")
        print(f"  rust_ms:    {result['rust_ms']:.3f}")
    if result.get("error"):
        print(f"  error:\n{result['error'][:800]}")

    if result["passed"] and result.get("speedup", 0) > 0:
        print("\nSTAGE 2 GATE: PASS")
    else:
        print("\nSTAGE 2 GATE: FAIL")
        if not result.get("error"):
            print("(speedup was 0 or negative)")


if __name__ == "__main__":
    main()
