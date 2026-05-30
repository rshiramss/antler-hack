"""Stage 2 runner: thin end-to-end slice, driven by a TargetSpec.

Flow:
  1. Check OPENAI_API_KEY
  2. Build the naive Rust seed for the spec (call-through-GIL, always correct)
  3. Smoke-test the naive seed against the spec's oracle
  4. Call try_one_port (LLM mutate → build → verify → benchmark)
  5. Print result and STAGE 2 GATE: PASS / FAIL

Target selection:
  * default                 → MICROGRAD_SPEC (backward compatible)
  * --swarm-a <result.json> → build a TargetSpec from a Swarm A top candidate
                              (analyze.py output: {name, source, file, ...})
  * --oracle module:attr    → oracle for the Swarm A candidate (else self-port)
"""
import argparse
import json
import os
import sys
import subprocess
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from target_spec import TargetSpec, MICROGRAD_SPEC, naive_seed


def _maturin_build(lib_rs: str) -> tuple[bool, str]:
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
        capture_output=True, text=True, env=env,
    )
    return proc.returncode == 0, proc.stderr


def _resolve_spec(args) -> TargetSpec:
    """Pick the TargetSpec to run from CLI args (Swarm A candidate, or micrograd)."""
    if not args.swarm_a:
        return MICROGRAD_SPEC
    with open(args.swarm_a) as f:
        result = json.load(f)
    # analyze.py can emit either the single top candidate or the full ranked list.
    if isinstance(result, list):
        result = result[0]
    return TargetSpec.from_swarm_a(result, oracle_path=args.oracle)


def main():
    parser = argparse.ArgumentParser(description="RustForge Stage 2 runner")
    parser.add_argument(
        "--swarm-a", metavar="JSON",
        help="path to a Swarm A result JSON (top candidate or ranked list)",
    )
    parser.add_argument(
        "--oracle", metavar="MODULE:ATTR",
        help="dotted path to the reference oracle for the Swarm A candidate "
             "(omit to self-port the candidate's own source)",
    )
    args = parser.parse_args()

    spec = _resolve_spec(args)
    print(f"=== Target: {spec.name} ===")

    # ── Preflight ────────────────────────────────────────────────────────────
    from mutate import _load_openai_key
    try:
        _load_openai_key()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # ── Build naive seed ─────────────────────────────────────────────────────
    print("=== Building naive seed ===")
    seed = naive_seed(spec)
    ok, stderr = _maturin_build(seed)
    if not ok:
        print(f"NAIVE SEED BUILD FAILED:\n{stderr}")
        sys.exit(1)
    print("Naive seed built OK.")

    # ── Smoke-test naive seed in a subprocess (fresh .so load) ───────────────
    print("\n=== Smoke-testing naive seed ===")
    venv_python = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
    venv_bin = os.path.join(PROJECT_ROOT, ".venv", "bin")
    env = os.environ.copy()
    env["PATH"] = f"{venv_bin}:{os.path.expanduser('~/.cargo/bin')}:{env.get('PATH', '')}"
    env["VIRTUAL_ENV"] = os.path.join(PROJECT_ROOT, ".venv")

    spec_file = os.path.join(PROJECT_ROOT, ".rustforge_spec.json")
    with open(spec_file, "w") as f:
        f.write(spec.to_json())
    env["RUSTFORGE_SPEC"] = spec_file

    smoke = subprocess.run(
        [venv_python, "-c", f"""
import sys, os, json, numpy as np
sys.path.insert(0, {repr(PROJECT_ROOT)})
from target_spec import TargetSpec
spec = TargetSpec.from_json(open(os.environ["RUSTFORGE_SPEC"]).read())
oracle_fn = spec.load_oracle()
import importlib
mod = importlib.import_module(spec.module_name)
mod.set_oracle(oracle_fn)
inp = np.array(spec.default_fixed_input(), dtype=np.float64).reshape(spec.input_shape)
r = np.asarray(mod.run_rust(inp))
o = np.asarray(oracle_fn(inp))
assert np.allclose(r, o, rtol=spec.rtol, atol=spec.atol), f"smoke FAIL: {{r}} vs {{o}}"
print("Smoke test PASS — rust:", r, "oracle:", o)
"""],
        cwd=PROJECT_ROOT, capture_output=True, text=True, env=env,
    )
    if smoke.returncode != 0:
        print(f"SMOKE TEST FAILED:\n{smoke.stderr}")
        sys.exit(1)
    print(smoke.stdout.strip())

    # ── try_one_port with retry-on-compiler-error ────────────────────────────
    print("\n=== Running try_one_port (LLM mutate → build → verify → benchmark) ===")
    from try_one_port import try_one_port

    MAX_ATTEMPTS = 3
    result = None
    variant: dict = {}
    parent = naive_seed(spec)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n--- Attempt {attempt}/{MAX_ATTEMPTS} ---")
        result = try_one_port(
            parent_rust=parent,
            spec=spec,
            variant=variant,
        )
        if result["passed"]:
            break
        # Feed the compiler/verify error back for the next attempt
        variant = {"compiler_error": result.get("error", "")[:2000]}
        print(f"  attempt {attempt} failed — retrying with error context")

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n=== Final Result ===")
    print(f"  passed:     {result['passed']}")
    print(f"  speedup:    {result.get('speedup', 0):.3f}x")
    if result.get("python_ms") is not None:
        print(f"  python_ms:  {result['python_ms']:.3f}")
        print(f"  rust_ms:    {result['rust_ms']:.3f}")
    if result.get("error"):
        print(f"  error:\n{result['error'][:800]}")

    if result["passed"] and result.get("speedup", 0) > 0:
        print("\nSTAGE 2 GATE: PASS")
    else:
        print("\nSTAGE 2 GATE: FAIL")
        if not result.get("error"):
            print("(speedup was 0 or negative)")


if __name__ == "__main__":
    main()
