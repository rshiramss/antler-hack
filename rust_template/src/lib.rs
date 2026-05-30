use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
fn run_rust<'py>(
    py: Python<'py>,
    inputs: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let inp = inputs.as_array();
    let x0: f64 = inp[0];
    let x1: f64 = inp[1];

    // Forward pass
    let z0 = 0.27885360 * x0 + (-0.94997849) * x1;
    let z1 = -0.44994136 * x0 + (-0.55357852) * x1;
    let z2 = 0.47294243 * x0 + 0.35339897 * x1;
    let z3 = 0.78435914 * x0 + (-0.82612233) * x1;
    
    let h0 = if z0 > 0.0 { z0 } else { 0.0 };
    let h1 = if z1 > 0.0 { z1 } else { 0.0 };
    let h2 = if z2 > 0.0 { z2 } else { 0.0 };
    let h3 = if z3 > 0.0 { z3 } else { 0.0 };
    
    let out = (-0.15615636) * h0 + (-0.94040556) * h1 + (-0.56272405) * h2 + 0.01071058 * h3;

    // Backward pass (d_out = 1.0)
    let d_h0 = -0.15615636 * 1.0;
    let d_h1 = -0.94040556 * 1.0;
    let d_h2 = -0.56272405 * 1.0;
    let d_h3 = 0.01071058 * 1.0;
    
    let d_z0 = d_h0 * if z0 > 0.0 { 1.0 } else { 0.0 };
    let d_z1 = d_h1 * if z1 > 0.0 { 1.0 } else { 0.0 };
    let d_z2 = d_h2 * if z2 > 0.0 { 1.0 } else { 0.0 };
    let d_z3 = d_h3 * if z3 > 0.0 { 1.0 } else { 0.0 };

    let x0_grad = 0.27885360 * d_z0 + (-0.44994136) * d_z1 + 0.47294243 * d_z2 + 0.78435914 * d_z3;
    let x1_grad = -0.94997849 * d_z0 + (-0.55357852) * d_z1 + 0.35339897 * d_z2 + (-0.82612233) * d_z3;

    // Return result as numpy array
    let result = numpy::ndarray::Array1::from_vec(vec![x0_grad, x1_grad, out]);
    Ok(result.into_pyarray(py))
}

#[pymodule]
fn rustforge_port(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_rust, m)?)?;
    Ok(())
}