# mutate.py  —  the mutation operator IS an LLM that rewrites solve(x).
import os
import re
from dotenv import load_dotenv
from litellm import completion

load_dotenv(override=True)                       # .env is authoritative (beats stale shell vars); no-op in swarm containers

# Newest models, May 2026 (confirm ids against your provider):
#   gpt-5.5         — newest frontier, Chat Completions (default)
#   gpt-5.3-codex   — most capable agentic coding model (best for code edits)
#   gpt-5.4-mini    — cheap/fast; good for breadth across a wide swarm
#   claude-...      — any Anthropic model, if you set ANTHROPIC_API_KEY instead
MODEL = os.environ.get("OPT_MODEL", "gpt-5.5")   # any litellm-supported model id

# The seed candidate (string form), used to start the ratchet.
SEED = '''import numpy as np
def solve(x):
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = x[i]*x[i] + 1.0
    return out
'''

_PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "program.md")


def _program() -> str:
    with open(_PROGRAM_PATH) as f:
        return f.read()


def propose(champion_src: str, history: str, temperature: float = 0.7) -> str:
    """Ask the LLM to rewrite solve() faster while staying correct. Returns new source."""
    messages = [
        {"role": "system", "content": _program()},
        {"role": "user", "content": (
            f"Current champion code:\n```python\n{champion_src}\n```\n\n"
            f"Journal of past attempts (most recent last):\n{history or '(none yet)'}\n\n"
            "Propose ONE change that makes it faster while keeping it numerically "
            "identical. Return the COMPLETE new module as a single ```python block."
        )},
    ]
    return _extract_code(_complete(messages, temperature))


def _supports_temperature(model: str) -> bool:
    """Reasoning-family models (gpt-5.x, o-series) only allow the default temperature."""
    m = model.lower()
    return not (m.startswith("gpt-5") or re.match(r"^o\d", m))


def _complete(messages, temperature: float) -> str:
    """Call the model, sending temperature only to models that support it.
    Keeps a fallback in case an unanticipated model also locks temperature."""
    kwargs = {"model": MODEL, "messages": messages}
    if _supports_temperature(MODEL):
        kwargs["temperature"] = temperature
    try:
        resp = completion(**kwargs)
    except Exception as e:
        if "temperature" in str(e).lower():          # safety net for unanticipated locked models
            kwargs.pop("temperature", None)
            resp = completion(**kwargs)
        else:
            raise
    return resp.choices[0].message.content


def _extract_code(text: str) -> str:
    """Pull the python source out of the model's ```python ...``` block."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def load_solve(src: str):
    """Turn a candidate source string into a callable solve(x)."""
    ns: dict = {}
    exec(src, ns)            # candidate is trusted-ish locally; Modal sandboxes it in the swarm
    return ns["solve"]


# ---- Rust port (§8): the LLM rewrites lib.rs instead of Python ----------------
_PROGRAM_RUST_PATH = os.path.join(os.path.dirname(__file__), "program_rust.md")

# Seed = the hand-written, FFI-proven lib.rs (a known-correct floor for the ratchet).
SEED_RUST = '''use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
fn solve<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> {
    let view = x.as_array();
    let out: Vec<f64> = view.iter().map(|&v| v * v + 1.0).collect();
    out.into_pyarray(py)
}

#[pymodule]
fn rust_solve(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
'''


def _fill_program_rust(spec) -> str:
    """Fill the program_rust.md template from a TargetSpec.

    Replaces the <<...>> sentinels (the file is full of Rust braces, so a sentinel
    replace is used rather than str.format). Removes the old hardcoded x*x+1 / rust_solve
    coupling — the reference, module/fn names, signature, and tolerances all come from
    the spec.
    """
    with open(_PROGRAM_RUST_PATH) as f:
        template = f.read()
    ref_source = (spec.source or "").strip() or "(no source provided; follow the description)"
    return (
        template
        .replace("<<REFERENCE_DESC>>", spec.description.strip())
        .replace("<<REFERENCE_SOURCE>>", ref_source)
        .replace("<<MODULE>>", spec.module_name)
        .replace("<<FN>>", spec.fn_name)
        .replace("<<SIGNATURE>>", spec.rust_signature)
        .replace("<<RTOL>>", repr(spec.rtol))
        .replace("<<ATOL>>", repr(spec.atol))
    )


def propose_rust(champion_src: str, history: str, temperature: float = 0.7, spec=None) -> str:
    """Ask the LLM to rewrite lib.rs faster while staying correct. Returns Rust source.

    The system prompt is built from `spec` (a TargetSpec) so the swarm can port any
    target, not just the hardcoded x*x+1 toy. `spec` is required by the spec-driven
    app_rust.py entrypoint.
    """
    if spec is None:
        raise ValueError("propose_rust requires a TargetSpec (build one from the target / Swarm A)")
    program = _fill_program_rust(spec)
    messages = [
        {"role": "system", "content": program},
        {"role": "user", "content": (
            f"Current champion lib.rs:\n```rust\n{champion_src}\n```\n\n"
            f"Journal of past attempts (most recent last):\n{history or '(none yet)'}\n\n"
            "Propose ONE change that makes it faster while keeping it numerically "
            "identical. Return the COMPLETE lib.rs as a single ```rust block."
        )},
    ]
    return _extract_rust(_complete(messages, temperature))


def _extract_rust(text: str) -> str:
    """Pull the rust source out of the model's ```rust ...``` block."""
    m = re.search(r"```(?:rust)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()
