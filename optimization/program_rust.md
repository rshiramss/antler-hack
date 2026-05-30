# program_rust.md — instructions to the optimizing LLM (Rust / PyO3 port)

You are an autonomous performance researcher. Rewrite the function as a **Rust PyO3
extension** that computes the SAME result as the frozen Python reference, but faster.
Output a COMPLETE `src/lib.rs`.

## Hard requirements (the build depends on these — do not change them)
- Crates available in Cargo.toml are ONLY `pyo3 = "0.28"` (feature `extension-module`)
  and `numpy = "0.28"` (rust-numpy, which re-exports `ndarray`). **Do not use any other
  crate** — adding one will fail to compile.
- The module MUST be `#[pymodule] fn rust_solve(...)` exposing `#[pyfunction] fn solve`.
- `solve` takes a 1-D f64 NumPy array and returns a 1-D f64 NumPy array, elementwise
  equal to the reference within rtol=1e-9, atol=1e-9.
- The reference is: `out[i] = x[i] * x[i] + 1.0`. Match it exactly.

## Required skeleton (keep the signatures + names; change the body / add helpers)
```rust
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
fn solve<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> {
    let view = x.as_array();
    // your fast implementation here — must equal v*v + 1.0 elementwise
}

#[pymodule]
fn rust_solve(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
```

## Rules
- Return ONLY a single ```rust code block — the complete `lib.rs`. No prose.
- Do not special-case inputs; the correctness fuzz inputs are hidden from you.
- It must compile with the given deps and pass the differential gate, or it scores 0.

## Metric
`speedup = reference_time / candidate_time`, higher is better, gated to 0 on any
compile error or correctness failure.
