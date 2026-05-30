import os
import json
import subprocess

from mutate import mutate
from target_spec import TargetSpec

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_SPEC_FILE = os.path.join(PROJECT_ROOT, ".rustforge_spec.json")


def _build(lib_rs: str) -> tuple[bool, str]:
    """Write lib.rs and run maturin develop --release. Returns (success, stderr)."""
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
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.returncode == 0, proc.stderr


def _validate(spec: TargetSpec) -> dict:
    """Run _validate.py in a fresh subprocess to pick up the newly built .so.

    The spec is handed to the subprocess via a JSON file referenced by RUSTFORGE_SPEC,
    so the validator loads the right oracle and I/O profile.
    """
    with open(_SPEC_FILE, "w") as f:
        f.write(spec.to_json())

    venv_python = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
    venv_bin = os.path.join(PROJECT_ROOT, ".venv", "bin")
    env = os.environ.copy()
    env["PATH"] = f"{venv_bin}:{os.path.expanduser('~/.cargo/bin')}:{env.get('PATH', '')}"
    env["VIRTUAL_ENV"] = os.path.join(PROJECT_ROOT, ".venv")
    env["RUSTFORGE_SPEC"] = _SPEC_FILE

    proc = subprocess.run(
        [venv_python, "_validate.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return {
            "passed": False,
            "speedup": 0.0,
            "error": proc.stderr or proc.stdout or "validation subprocess failed",
        }
    try:
        return json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        return {
            "passed": False,
            "speedup": 0.0,
            "error": f"JSON parse error from _validate.py: {proc.stdout!r}",
        }


def try_one_port(parent_rust: str, spec: TargetSpec, variant: dict) -> dict:
    """One cycle: mutate → build → verify → benchmark, all driven by `spec`.

    Returns a result dict with keys: passed, speedup, error, code, variant,
    and (on success) python_ms, rust_ms.
    """
    # 1. Mutate
    print("  [mutate] calling LLM...")
    candidate = mutate(parent_rust, spec, variant.get("compiler_error"))
    print(f"  [mutate] received {len(candidate)} chars of Rust")

    # 2. Build
    print("  [build] running maturin develop --release...")
    ok, stderr = _build(candidate)
    if not ok:
        print("  [build] FAILED")
        return {
            "passed": False,
            "speedup": 0.0,
            "error": stderr,
            "code": candidate,
        }
    print("  [build] OK")

    # 3. Validate (verify + benchmark) in a fresh subprocess
    print("  [validate] running verify + bench in subprocess...")
    result = _validate(spec)
    result["code"] = candidate
    result["variant"] = variant
    return result
