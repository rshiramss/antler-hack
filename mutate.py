"""The mutation operator: an LLM that ports a target Python function to Rust.

The prompt is built dynamically from a TargetSpec (its source + description + the
Rust signature skeleton) instead of being hardwired to the micrograd MLP. Point the
pipeline at any TargetSpec and this writes the matching src/lib.rs.
"""
import os

from target_spec import TargetSpec

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


def build_prompt(spec: TargetSpec, parent_rust: str, compiler_error: str | None) -> str:
    """Assemble the user prompt for porting `spec` to Rust.

    Pulls everything target-specific from the spec: the reference Python source, the
    human description, the required Rust signature, and the #[pymodule] name. Nothing
    micrograd-specific is hardcoded here anymore.
    """
    reference_section = (
        f"\n\nREFERENCE PYTHON (port this exact behavior):\n```python\n{spec.source.strip()}\n```"
        if spec.source.strip()
        else ""
    )

    parent_section = (
        f"\n\nCURRENT CHAMPION src/lib.rs (improve on this — make it faster while keeping it correct):\n"
        f"```rust\n{parent_rust.strip()}\n```"
        if parent_rust and parent_rust.strip()
        else ""
    )

    error_section = (
        f"\n\nCOMPILER/VERIFY ERROR from the previous attempt — the code below failed. Fix it:\n{compiler_error}"
        if compiler_error
        else ""
    )

    return f"""Write a Rust PyO3 extension (src/lib.rs) that implements `run_rust`.

WHAT TO PORT:
{spec.description}
{reference_section}

REQUIREMENTS:
- The function must have exactly this signature:

  {spec.rust_signature}

- The #[pymodule] must be named `{spec.module_name}` and match the [lib] name in Cargo.toml.
- Implement the computation directly in Rust. Do NOT call back into Python.
- Be numerically faithful to the reference (the port is checked with np.allclose,
  rtol={spec.rtol}, atol={spec.atol}, against hundreds of random fuzzed inputs).

A minimal module shell looks like:

use numpy::{{IntoPyArray, PyArray1, PyReadonlyArray1}};
use pyo3::prelude::*;

#[pyfunction]
{spec.rust_signature} {{
    let inp = inputs.as_array();
    // ... your computation here ...
}}

#[pymodule]
fn {spec.module_name}(m: &Bound<'_, PyModule>) -> PyResult<()> {{
    m.add_function(wrap_pyfunction!(run_rust, m)?)?;
    Ok(())
}}
{parent_section}{error_section}

Return ONLY the complete src/lib.rs. No markdown fences, no text outside the Rust code."""


def mutate(parent_rust: str, spec: TargetSpec, compiler_error: str | None) -> str:
    """Call gpt-4o-mini to produce a Rust port of `spec`.

    Returns a raw Rust string (complete src/lib.rs contents).
    """
    import litellm

    api_key = _load_openai_key()
    user_content = build_prompt(spec, parent_rust, compiler_error)

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
