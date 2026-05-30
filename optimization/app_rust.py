# app_rust.py — the Rust swarm: each worker has the LLM write lib.rs, compiles it with
# maturin in-container, and scores it (differential gate + benchmark) vs the Python reference.
#
#   uv run modal run optimization/app_rust.py::evolve_rust          # full run
#   N_PER_ROUND=1 MAX_ROUNDS=1 uv run modal run optimization/app_rust.py::evolve_rust   # 1-container de-risk
import os

import modal

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
    .add_local_file("optimization/target.py", "/root/target.py", copy=True)
    .add_local_file("optimization/mutate.py", "/root/mutate.py", copy=True)
    .add_local_file("optimization/program_rust.md", "/root/program_rust.md", copy=True)
    .add_local_dir("optimization/rust_solve", "/root/proj", copy=True)   # pre-baked cargo project
    # warm the cargo cache: compile pyo3+numpy once into the image (workers reuse it)
    .run_commands("cd /root/proj && maturin build --release --out /root/proj/dist --interpreter python3")
)

secret = modal.Secret.from_dotenv()


@app.function(image=rust_image, cpu=4, secrets=[secret], timeout=900)
def try_one_rust(champion_src: str, history: str, temperature: float) -> dict:
    """Swarm worker: LLM writes lib.rs -> maturin build -> import -> gate + benchmark."""
    import importlib
    import subprocess
    import sys
    import time

    import numpy as np

    sys.path.insert(0, "/root")
    from mutate import propose_rust
    from target import reference, make_inputs
    meta = {"model": os.environ.get("OPT_MODEL", "gpt-5.5"), "temperature": temperature}

    # 1) LLM rewrites the Rust, in-container
    try:
        src = propose_rust(champion_src, history, temperature)
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
    import rust_solve
    fn = rust_solve.solve

    # 4) differential gate on unseen fuzz inputs (the anti-cheat moat)
    rng = np.random.default_rng(12345)
    for _ in range(20):
        xi = rng.standard_normal(rng.integers(1, 5000))
        try:
            got = fn(xi)
        except Exception as e:
            return {"passed": False, "speedup": 0.0, "error": repr(e), "src": src, **meta}
        if not np.allclose(got, reference(xi), rtol=1e-9, atol=1e-9):
            return {"passed": False, "speedup": 0.0, "error": "differential mismatch", "src": src, **meta}

    # 5) benchmark vs the reference
    x = make_inputs()
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
    champion, best, history = SEED_RUST, 1.0, ""
    debuglog.reset()
    for round_n in range(1, MAX_ROUNDS + 1):
        champion_before = champion
        args = [(champion, history, 0.4 + 0.5 * (i / N_PER_ROUND)) for i in range(N_PER_ROUND)]
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
