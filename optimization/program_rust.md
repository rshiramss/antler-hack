# program_rust.md — instructions to the optimizing LLM (Rust / PyO3 port)

This file is a TEMPLATE. The loop fills the `<<...>>` placeholders from the active
TargetSpec before sending it to the model (see optimization/mutate.py `propose_rust`).

You are an autonomous performance researcher. Rewrite the function as a **Rust PyO3
extension** that computes the SAME result as the frozen Python reference, but faster.
Output a COMPLETE `src/lib.rs`.

## What to port
<<REFERENCE_DESC>>

Reference Python implementation (match its output exactly):
```python
<<REFERENCE_SOURCE>>
```

## Hard requirements (the build depends on these — do not change them)
- Crates available in Cargo.toml are ONLY `pyo3 = "0.28"` (feature `extension-module`)
  and `numpy = "0.28"` (rust-numpy, which re-exports `ndarray`). **Do not use any other
  crate** — adding one will fail to compile.
- The module MUST be `#[pymodule] fn <<MODULE>>(...)` exposing `#[pyfunction] fn <<FN>>`.
- `<<FN>>` takes a 1-D f64 NumPy array and returns a 1-D f64 NumPy array, elementwise
  equal to the reference within rtol=<<RTOL>>, atol=<<ATOL>>.

## Required skeleton (keep the signatures + names; change the body / add helpers)
```rust
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
<<SIGNATURE>> {
    let view = x.as_array();
    // your fast implementation here — must equal the reference elementwise
}

#[pymodule]
fn <<MODULE>>(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(<<FN>>, m)?)?;
    Ok(())
}
```

## Rules
- Return ONLY a single ```rust code block — the complete `lib.rs`. No prose.
- Do not call back into Python; implement the computation directly in Rust.
- Do not special-case inputs; the correctness fuzz inputs are hidden from you.
- It must compile with the given deps and pass the differential gate, or it scores 0.

## Metric
`speedup = reference_time / candidate_time`, higher is better, gated to 0 on any
compile error or correctness failure.
