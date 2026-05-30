use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
fn solve<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> {
    let view = x.as_array();
    let n = view.len();

    let mut out: Vec<f64> = Vec::with_capacity(n);
    unsafe {
        let dst = out.as_mut_ptr();

        if let Some(slice) = view.as_slice() {
            let mut i = 0usize;
            while i + 4 <= n {
                let v0 = *slice.get_unchecked(i);
                let v1 = *slice.get_unchecked(i + 1);
                let v2 = *slice.get_unchecked(i + 2);
                let v3 = *slice.get_unchecked(i + 3);

                dst.add(i).write(v0 * v0 + 1.0);
                dst.add(i + 1).write(v1 * v1 + 1.0);
                dst.add(i + 2).write(v2 * v2 + 1.0);
                dst.add(i + 3).write(v3 * v3 + 1.0);

                i += 4;
            }
            while i < n {
                let v = *slice.get_unchecked(i);
                dst.add(i).write(v * v + 1.0);
                i += 1;
            }
        } else {
            for (i, &v) in view.iter().enumerate() {
                dst.add(i).write(v * v + 1.0);
            }
        }

        out.set_len(n);
    }

    out.into_pyarray(py)
}

#[pymodule]
fn rust_solve(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}