# app_rust.py — the Rust swarm: each worker has the LLM write lib.rs, compiles it with
# maturin in-container, and scores it (differential gate + benchmark) vs the Python reference.
#
# The target is described by a TargetSpec, so the same swarm ports any function:
#   uv run modal run optimization/app_rust.py::evolve_rust          # default toy target
#   RUSTFORGE_SWARM_A=candidates.json uv run modal run optimization/app_rust.py::evolve_rust
#       # port the top function Swarm A (analyze.py) selected
#   N_PER_ROUND=1 MAX_ROUNDS=1 uv run modal run optimization/app_rust.py::evolve_rust   # 1-container de-risk
import os
import sys

import modal

# target_spec.py lives at the repo root; append it (not prepend) so the local `mutate`
# import below still resolves to optimization/mutate.py.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from target_spec import TargetSpec, SOLVE_RUST_SIGNATURE, select_portable_candidate
from mutate import SEED_RUST

app = modal.App("rustforge-swarm")

# Image: Python + Rust toolchain + maturin, frozen files baked in, and a WARM build of the
# seed so pyo3/numpy are already compiled into the image (workers then build incrementally).
rust_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --default-toolchain stable --profile minimal"
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin", "CARGO_HOME": "/root/.cargo"})
    .run_commands("python -m ensurepip --upgrade")          # ensure `python -m pip` exists for wheel install
    .uv_pip_install("numpy", "litellm", "python-dotenv", "maturin")
    .add_local_file("target_spec.py", "/root/target_spec.py", copy=True)   # the shared target abstraction
    .add_local_file("optimization/mutate.py", "/root/mutate.py", copy=True)
    .add_local_file("optimization/program_rust.md", "/root/program_rust.md", copy=True)
    .add_local_dir("optimization/rust_solve", "/root/proj", copy=True)   # pre-baked cargo project
    # warm the cargo cache: compile pyo3+numpy once into the image (workers reuse it)
    .run_commands("cd /root/proj && maturin build --release --out /root/proj/dist --interpreter python3")
    # the shared, artifact-agnostic metric (held-out gate + multi-size regression + mem + stability).
    # Added last so the expensive layers above stay cached when only this file changes.
    .add_local_file("optimization/evaluate.py", "/root/evaluate.py", copy=True)
)

secret = modal.Secret.from_dotenv()


# ── The default target: the x*x+1 toy (reproduces the original hardcoded behavior) ──
# Default demo target: a PURE-PYTHON numeric hot loop (degree-5 polynomial via Horner,
# evaluated per element). The oracle is genuinely slow interpreted Python — NOT a numpy
# one-liner — so the Rust port's win is real and large, and because both sides are O(n)
# the speedup is flat/growing with size (it does NOT collapse at scale like a C/numpy-
# backed function would). Pure arithmetic, no transcendentals: the speedup is the loop.
_DEMO_SOURCE = """\
def poly_eval(x):
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        xi = x[i]
        # fixed degree-5 polynomial, written out explicitly (pure multiply/add):
        out[i] = ((((0.5 * xi - 1.2) * xi + 0.3) * xi + 2.0) * xi - 0.7) * xi + 1.1
    return out
"""

# A correct, direct Rust Horner port — used only as the starting champion the LLM mutates
# (the swarm never compiles the seed; the prompt carries the reference source).
_DEMO_SEED_RUST = """\
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
fn solve<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> {
    let view = x.as_array();
    let out: Vec<f64> = view.iter().map(|&xi| {
        ((((0.5 * xi - 1.2) * xi + 0.3) * xi + 2.0) * xi - 0.7) * xi + 1.1
    }).collect();
    out.into_pyarray(py)
}

#[pymodule]
fn rust_solve(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
"""


def _demo_spec() -> TargetSpec:
    return TargetSpec(
        name="poly_eval_hotloop",
        description=("Evaluate this fixed degree-5 polynomial at each element of a 1-D f64 "
                     "array, EXACTLY as written (do not reorder terms): "
                     "out[i] = ((((0.5*x - 1.2)*x + 0.3)*x + 2.0)*x - 0.7)*x + 1.1. "
                     "The reference is a pure-Python loop; match it to rtol/atol 1e-9."),
        source=_DEMO_SOURCE,
        oracle_source=_DEMO_SOURCE,
        oracle_entry="poly_eval",
        module_name="rust_solve",          # must match the baked crate's [lib] name
        fn_name="solve",
        rust_signature=SOLVE_RUST_SIGNATURE,
        seed_rust=_DEMO_SEED_RUST,
        array_mode=True,
        rtol=1e-9,
        atol=1e-9,
        bench_input_size=1_000_000,
        fuzz_max_len=5000,
    )


def _load_spec() -> TargetSpec:
    """Pick the target: a Swarm A candidate (RUSTFORGE_SWARM_A=candidates.json) or the toy.

    Swarm A ranks by optimization potential, not interface shape, so we pick the
    highest-scored candidate that actually fits the swarm's 1-D-array contract and
    print why any higher-scored ones were skipped.
    """
    path = os.environ.get("RUSTFORGE_SWARM_A")
    if path and os.path.exists(path):
        import json

        with open(path) as f:
            data = json.load(f)
        results = data if isinstance(data, list) else [data]
        spec, report = select_portable_candidate(
            results,
            module_name="rust_solve",
            fn_name="solve",
            rust_signature=SOLVE_RUST_SIGNATURE,
            array_mode=True,
            rtol=1e-9,
            atol=1e-9,
        )
        print("Swarm A → swarm candidate selection (by score):")
        for name, score, ok, reason in report:
            picked = spec is not None and ok and name == spec.name
            mark = "PORT" if picked else (" fit" if ok else "skip")
            print(f"  [{mark}] {name} (score {score:.2f}) — {reason}")
            if picked:
                break
        if spec is None:
            raise SystemExit(
                "No Swarm A candidate fits the 1-D f64 array -> 1-D array contract.\n"
                "Point analyze.py at functions with that shape, or pass an explicit oracle."
            )
        return spec
    return _demo_spec()


# Runs in a FRESH subprocess per attempt (mirrors the local _validate.py design): loads the
# just-built .so by path, then scores it with the SHARED evaluate() metric — the held-out
# differential gate (gate 1) plus the multi-size regression sweep, memory, stability, and the
# gated geomean. A new interpreter avoids two failure modes seen on Modal: (a) pip-installing
# the wheel was silently ineffective; (b) re-importing a C-extension in a warm container
# returns stale code.
_EVAL_SCRIPT = r'''
import os, sys, json, glob, importlib.util
import numpy as np
sys.path.insert(0, "/root")
from target_spec import TargetSpec
import evaluate as E
spec = TargetSpec.from_json(os.environ["RF_SPEC"])
ref = spec.load_oracle()
libs = (glob.glob("/root/proj/target/release/lib%s.so" % spec.module_name)
        + glob.glob("/root/proj/target/release/lib%s.dylib" % spec.module_name))
if not libs:
    print(json.dumps({"passed": False, "speedup": 0.0,
        "error": "no built lib: " + str(os.listdir("/root/proj/target/release"))[:300],
        "verdict": "build produced no library"})); sys.exit(0)
loader = importlib.util.spec_from_file_location(spec.module_name, libs[0])
mod = importlib.util.module_from_spec(loader); loader.loader.exec_module(mod)
fn = getattr(mod, spec.fn_name)
# Reuse the frozen, artifact-agnostic metric: oracle = the spec's reference; inputs are
# harness-generated inside evaluate() with a seed unseen by the mutation model.
res = E.evaluate(fn, reference=ref, sample_input=np.ones(8, dtype=np.float64))
print(json.dumps(res, default=float))
'''


@app.function(image=rust_image, cpu=4, secrets=[secret], timeout=900)
def try_one_rust(spec_json: str, champion_src: str, history: str, temperature: float) -> dict:
    """Swarm worker: LLM writes lib.rs -> cargo build -> (fresh subprocess) gate + benchmark."""
    import json
    import subprocess

    sys.path.insert(0, "/root")
    from mutate import propose_rust
    from target_spec import TargetSpec as _TargetSpec

    spec = _TargetSpec.from_json(spec_json)
    meta = {"model": os.environ.get("OPT_MODEL", "gpt-5.5"), "temperature": temperature}

    # 1) LLM rewrites the Rust, in-container (prompt is built from the spec)
    try:
        src = propose_rust(champion_src, history, temperature, spec)
    except Exception as e:
        return {"passed": False, "speedup": 0.0, "error": "llm: " + repr(e), "src": "", **meta}
    open("/root/proj/src/lib.rs", "w").write(src)

    # 2) compile (incremental — pyo3/numpy already warm-built into the image's cargo cache)
    build = subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", "/root/proj/Cargo.toml"],
        capture_output=True, text=True, cwd="/root/proj",
    )
    if build.returncode != 0:
        tail = (build.stderr.strip().splitlines() or ["compile failed"])[-1]
        return {"passed": False, "speedup": 0.0, "error": "compile: " + tail[:300], "src": src, **meta}

    # 3) load the freshly built .so + run gate + benchmark in a fresh interpreter
    proc = subprocess.run(
        [sys.executable, "-c", _EVAL_SCRIPT],
        env={**os.environ, "RF_SPEC": spec_json}, capture_output=True, text=True,
    )
    line = (proc.stdout.strip().splitlines() or [""])[-1]
    try:
        res = json.loads(line)
    except Exception:
        return {"passed": False, "speedup": 0.0,
                "error": ("eval failed: " + (proc.stderr or proc.stdout or "no output"))[:300],
                "src": src, **meta}
    res["src"] = src
    res.update(meta)
    return res


N_PER_ROUND = int(os.environ.get("N_PER_ROUND", "10"))
MAX_ROUNDS  = int(os.environ.get("MAX_ROUNDS", "5"))


@app.local_entrypoint()
def evolve_rust():
    import debuglog

    spec = _load_spec()
    spec_json = spec.to_json()
    print(f"=== RustForge swarm target: {spec.name} ===")

    champion, best, history = spec.seed(), 1.0, ""
    debuglog.reset()
    for round_n in range(1, MAX_ROUNDS + 1):
        champion_before = champion
        args = [(spec_json, champion, history, 0.4 + 0.5 * (i / N_PER_ROUND)) for i in range(N_PER_ROUND)]
        results = [r for r in try_one_rust.starmap(args, return_exceptions=True) if isinstance(r, dict)]

        for idx, r in enumerate(results):
            debuglog.log_attempt(round_n, idx, champion_before, r,
                                 model=r.get("model", "?"), temperature=r.get("temperature", 0.0))

        survivors = [r for r in results if r["passed"]]
        kept = False
        if survivors:
            top = max(survivors, key=lambda r: r["speedup"])
            if top["speedup"] > best:
                champion, best, kept = top["src"], top["speedup"], True
                print(f"  round {round_n} winner ({best:.0f}x) — Rust change applied:")
                debuglog.show_change(champion_before, champion)
        history += f"round {round_n}: best={best:.2f}x, {len(survivors)}/{len(results)} correct\n"
        debuglog.log_round(round_n, results, best, kept)

    open(os.path.join(os.path.dirname(__file__), "champion_lib.rs"), "w").write(champion)
