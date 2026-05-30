import os
import sys
import json
import subprocess
import numpy as np

from mutate import mutate, TARGET_DESCRIPTION

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


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


def _validate() -> dict:
    """Run _validate.py in a fresh subprocess to pick up the newly built .so."""
    venv_python = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
    venv_bin = os.path.join(PROJECT_ROOT, ".venv", "bin")
    env = os.environ.copy()
    env["PATH"] = f"{venv_bin}:{os.path.expanduser('~/.cargo/bin')}:{env.get('PATH', '')}"
    env["VIRTUAL_ENV"] = os.path.join(PROJECT_ROOT, ".venv")

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


def try_one_port(parent_rust: str, target_fn, oracle_fn, variant: dict) -> dict:
    """One cycle: mutate → build → verify → benchmark.

    Returns a result dict with keys: passed, speedup, error, code, variant,
    and (on success) python_ms, rust_ms.
    """
    # 1. Mutate
    print("  [mutate] calling LLM...")
    candidate = mutate(parent_rust, TARGET_DESCRIPTION, variant.get("compiler_error"))
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
    result = _validate()
    result["code"] = candidate
    result["variant"] = variant
    return result
