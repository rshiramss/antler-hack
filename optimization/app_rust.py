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

from target_spec import TargetSpec, SOLVE_RUST_SIGNATURE
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
)

secret = modal.Secret.from_dotenv()


# ── The default target: the x*x+1 toy (reproduces the original hardcoded behavior) ──
_TOY_SOURCE = """\
def reference(x):
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = x[i] * x[i] + 1.0
    return out
"""


def _toy_spec() -> TargetSpec:
    return TargetSpec(
        name="toy_square_plus_one",
        description="Elementwise: out[i] = x[i] * x[i] + 1.0 over a 1-D f64 array.",
        source=_TOY_SOURCE,
        oracle_source=_TOY_SOURCE,
        oracle_entry="reference",
        module_name="rust_solve",          # must match the baked crate's [lib] name
        fn_name="solve",
        rust_signature=SOLVE_RUST_SIGNATURE,
        seed_rust=SEED_RUST,
        array_mode=True,
        rtol=1e-9,
        atol=1e-9,
        bench_input_size=1_000_000,
        fuzz_max_len=5000,
    )


def _load_spec() -> TargetSpec:
    """Pick the target: a Swarm A candidate (RUSTFORGE_SWARM_A=candidates.json) or the toy."""
    path = os.environ.get("RUSTFORGE_SWARM_A")
    if path and os.path.exists(path):
        import json

        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):          # analyze.py can emit the full ranked list
            data = data[0]
        return TargetSpec.from_swarm_a(
            data,
            module_name="rust_solve",
            fn_name="solve",
            rust_signature=SOLVE_RUST_SIGNATURE,
            array_mode=True,
            rtol=1e-9,
            atol=1e-9,
        )
    return _toy_spec()


@app.function(image=rust_image, cpu=4, secrets=[secret], timeout=900)
def try_one_rust(spec_json: str, champion_src: str, history: str, temperature: float) -> dict:
    """Swarm worker: LLM writes lib.rs -> maturin build -> import -> gate + benchmark."""
    import importlib
    import subprocess
    import time

    import numpy as np

    sys.path.insert(0, "/root")
    from mutate import propose_rust
    from target_spec import TargetSpec as _TargetSpec

    spec = _TargetSpec.from_json(spec_json)
    reference = spec.load_oracle()
    meta = {"model": os.environ.get("OPT_MODEL", "gpt-5.5"), "temperature": temperature}

    # 1) LLM rewrites the Rust, in-container (prompt is built from the spec)
    try:
        src = propose_rust(champion_src, history, temperature, spec)
    except Exception as e:
        return {"passed": False, "speedup": 0.0, "error": "llm: " + repr(e), "src": "", **meta}
    open("/root/proj/src/lib.rs", "w").write(src)

    # 2) compile (incremental — deps already warm-built into the image)
    build = subprocess.run(
        ["maturin", "build", "--release", "--out", "/root/proj/dist",
         "--manifest-path", "/root/proj/Cargo.toml", "--interpreter", sys.executable],
        capture_output=True, text=True, cwd="/root/proj",
    )
    if build.returncode != 0:
        tail = (build.stderr.strip().splitlines() or ["compile failed"])[-1]
        return {"passed": False, "speedup": 0.0, "error": "compile: " + tail[:200], "src": src, **meta}

    # 3) install the freshly built wheel into this interpreter and import it
    wheels = sorted(f for f in os.listdir("/root/proj/dist") if f.endswith(".whl"))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps",
         os.path.join("/root/proj/dist", wheels[-1])],
        capture_output=True, text=True,
    )
    importlib.invalidate_caches()
    mod = importlib.import_module(spec.module_name)
    fn = getattr(mod, spec.fn_name)

    # 4) differential gate on unseen fuzz inputs (the anti-cheat moat)
    rng = np.random.default_rng(12345)
    for xi in spec.make_gate_inputs(rng, n=20):
        try:
            got = np.asarray(fn(xi))
        except Exception as e:
            return {"passed": False, "speedup": 0.0, "error": repr(e), "src": src, **meta}
        if not np.allclose(got, np.asarray(reference(xi)), rtol=spec.rtol, atol=spec.atol):
            return {"passed": False, "speedup": 0.0, "error": "differential mismatch", "src": src, **meta}

    # 5) benchmark vs the reference on the same input
    x = spec.make_bench_input()
    def med(f):
        f(x)
        ts = []
        for _ in range(5):
            t = time.perf_counter(); f(x); ts.append(time.perf_counter() - t)
        return sorted(ts)[2]
    return {"passed": True, "speedup": med(reference) / med(fn), "error": "", "src": src, **meta}


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
