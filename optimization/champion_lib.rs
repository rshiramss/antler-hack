use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
fn solve<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> {
    let view = x.as_array();
    let n = view.len();

    let mut out: Vec<f64> = Vec::with_capacity(n);

    unsafe {
        out.set_len(n);

        let src = view.as_ptr();
        let dst = out.as_mut_ptr();
        let stride = view.strides()[0];

        if stride == 1 {
            let mut i = 0usize;
            let chunks = (n / 8) * 8;

            while i < chunks {
                let v0 = *src.add(i);
                let v1 = *src.add(i + 1);
                let v2 = *src.add(i + 2);
                let v3 = *src.add(i + 3);
                let v4 = *src.add(i + 4);
                let v5 = *src.add(i + 5);
                let v6 = *src.add(i + 6);
                let v7 = *src.add(i + 7);

                *dst.add(i) = v0 * v0 + 1.0;
                *dst.add(i + 1) = v1 * v1 + 1.0;
                *dst.add(i + 2) = v2 * v2 + 1.0;
                *dst.add(i + 3) = v3 * v3 + 1.0;
                *dst.add(i + 4) = v4 * v4 + 1.0;
                *dst.add(i + 5) = v5 * v5 + 1.0;
                *dst.add(i + 6) = v6 * v6 + 1.0;
                *dst.add(i + 7) = v7 * v7 + 1.0;

                i += 8;
            }

            while i < n {
                let v = *src.add(i);
                *dst.add(i) = v * v + 1.0;
                i += 1;
            }
        } else {
            let mut i = 0usize;
            while i < n {
                let v = *src.offset((i as isize) * stride);
                *dst.add(i) = v * v + 1.0;
                i += 1;
            }
        }
    }

    out.into_pyarray(py)
}

#[pymodule]
fn rust_solve(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}