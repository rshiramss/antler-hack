import os

TARGET_DESCRIPTION = (
    "Port the micrograd Value forward+backward MLP workload to Rust using PyO3. "
    "The Rust function must be named run_rust, accept a 1-D numpy array of 2 float64 inputs "
    "(PyReadonlyArray1<f64>), and return a 1-D numpy array of 3 float64 values: "
    "[x0.grad, x1.grad, output.data] (Bound<'py, PyArray1<f64>>). "
    "The MLP is 2->[4 ReLU neurons]->[1 linear neuron]. "
    "The weights must match exactly what Python's random.seed(42) + MLP(2,[4,1]) produces. "
    "The #[pymodule] name must be rustforge_port and match the [lib] name in Cargo.toml."
)

_SYSTEM_PROMPT = (
    "You are a Rust expert. Your job is to port a Python function to Rust using PyO3. "
    "Return ONLY the complete contents of src/lib.rs, no explanation, no markdown fences, no preamble."
)


def _load_openai_key() -> str:
    """Return the OpenAI key from OPENAI_KEY env var or .env file."""
    key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
    if key and key.startswith("sk-"):
        return key
    # Fall back to .env file in the project root
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENAI_KEY=") or line.startswith("OPENAI_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val.startswith("sk-"):
                        return val
    raise RuntimeError("No valid OpenAI key found (OPENAI_KEY in env or .env)")


def mutate(parent_rust: str, target_description: str, compiler_error: str | None) -> str:
    """Call gpt-4o-mini to mutate parent_rust toward target_description.

    Returns a raw Rust string (complete src/lib.rs contents).
    """
    import litellm

    api_key = _load_openai_key()

    project_root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(project_root, "targets", "micrograd", "oracle.py")) as f:
        oracle_source = f.read()
    with open(os.path.join(project_root, "targets", "micrograd", "engine.py")) as f:
        engine_source = f.read()

    error_section = (
        f"\n\nCOMPILER ERROR from previous attempt — the code below failed to compile. Fix it:\n{compiler_error}"
        if compiler_error
        else ""
    )

    user_content = f"""Write a Rust PyO3 extension (src/lib.rs) that implements run_rust.

EXACT COMPUTATION (do NOT build a Value class — implement this directly with scalars):

Step 1 — forward pass with these hardcoded weights:
  z0 =  0.27885360*x0 + (-0.94997849)*x1
  z1 = -0.44994136*x0 + (-0.55357852)*x1
  z2 =  0.47294243*x0 +   0.35339897*x1
  z3 =  0.78435914*x0 + (-0.82612233)*x1
  h0 = relu(z0)   // relu(v) = if v > 0.0 {{ v }} else {{ 0.0 }}
  h1 = relu(z1)
  h2 = relu(z2)
  h3 = relu(z3)
  out = (-0.15615636)*h0 + (-0.94040556)*h1 + (-0.56272405)*h2 + 0.01071058*h3

Step 2 — backward pass (d_out = 1.0):
  d_h0 = -0.15615636 * 1.0
  d_h1 = -0.94040556 * 1.0
  d_h2 = -0.56272405 * 1.0
  d_h3 =  0.01071058 * 1.0
  d_z0 = d_h0 * (if z0 > 0.0 {{ 1.0 }} else {{ 0.0 }})
  d_z1 = d_h1 * (if z1 > 0.0 {{ 1.0 }} else {{ 0.0 }})
  d_z2 = d_h2 * (if z2 > 0.0 {{ 1.0 }} else {{ 0.0 }})
  d_z3 = d_h3 * (if z3 > 0.0 {{ 1.0 }} else {{ 0.0 }})
  x0_grad =  0.27885360*d_z0 + (-0.44994136)*d_z1 + 0.47294243*d_z2 + 0.78435914*d_z3
  x1_grad = -0.94997849*d_z0 + (-0.55357852)*d_z1 + 0.35339897*d_z2 + (-0.82612233)*d_z3

Step 3 — return numpy array [x0_grad, x1_grad, out] as float64.

REQUIRED src/lib.rs SKELETON (fill in the body of run_rust):

use numpy::{{IntoPyArray, PyArray1, PyReadonlyArray1}};
use pyo3::prelude::*;

#[pyfunction]
fn run_rust<'py>(
    py: Python<'py>,
    inputs: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {{
    let inp = inputs.as_array();
    let x0: f64 = inp[0];
    let x1: f64 = inp[1];
    // ... your forward+backward code here ...
    let result = numpy::ndarray::Array1::from_vec(vec![x0_grad, x1_grad, out]);
    Ok(result.into_pyarray(py))
}}

#[pymodule]
fn rustforge_port(m: &Bound<'_, PyModule>) -> PyResult<()> {{
    m.add_function(wrap_pyfunction!(run_rust, m)?)?;
    Ok(())
}}
{error_section}

Return ONLY the complete src/lib.rs with the body of run_rust filled in. No markdown fences, no text outside the Rust code."""

    response = litellm.completion(
        model="gpt-4o-mini",
        api_key=api_key,
        api_base="https://api.openai.com/v1",  # explicit: bypass any OPENAI_BASE_URL override
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw = response.choices[0].message.content.strip()
    # Strip any markdown fences the model may have added despite instructions
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return raw
