# app.py
import os

import modal

from mutate import SEED

app = modal.App("optimization-swarm")

# Image: NumPy + litellm + python-dotenv, plus the frozen files and prompt at /root (on sys.path).
# Source paths are relative to where `modal run` is invoked (the project root).
image = (
    modal.Image.debian_slim()
    .uv_pip_install("numpy", "litellm", "python-dotenv")
    .add_local_file("optimization/evaluate.py", "/root/evaluate.py")
    .add_local_file("optimization/target.py", "/root/target.py")
    .add_local_file("optimization/mutate.py", "/root/mutate.py")
    .add_local_file("optimization/program.md", "/root/program.md")
)
secret = modal.Secret.from_dotenv()                # reads your local .env (key + OPT_MODEL) into each container


@app.function(image=image, cpu=2, secrets=[secret], timeout=600)
def try_one(champion_src: str, history: str, temperature: float) -> dict:
    """Swarm worker: one LLM rewrite + score, fully in-container."""
    from mutate import propose, load_solve, MODEL
    from evaluate import evaluate
    meta = {"model": MODEL, "temperature": temperature}     # echoed back so the entrypoint can log accurately
    try:
        src = propose(champion_src, history, temperature)   # the LLM edits the code, in the cloud
        print(f"[worker temp={temperature:.2f}] {MODEL}: proposed {len(src)} chars", flush=True)
        r = evaluate(load_solve(src))
        print(f"[worker temp={temperature:.2f}] passed={r['passed']} speedup={r['speedup']:.1f}x "
              f"err={r['error']!r}", flush=True)
        return {**r, "src": src, **meta}
    except Exception as e:
        print(f"[worker temp={temperature:.2f}] EXCEPTION {e!r}", flush=True)
        return {"passed": False, "speedup": 0.0, "error": repr(e), "src": "", **meta}


N_PER_ROUND = int(os.environ.get("N_PER_ROUND", "20"))   # swarm width; ≤1000 concurrent per call (scale.md)
MAX_ROUNDS  = int(os.environ.get("MAX_ROUNDS", "5"))     # override both for a small test, e.g. N_PER_ROUND=3 MAX_ROUNDS=1


@app.local_entrypoint()
def evolve():
    import debuglog                                        # local-only; logs every worker's output + diffs
    champion, best, history = SEED, 1.0, ""
    debuglog.reset()                                       # fresh debug.jsonl for this run
    for round_n in range(1, MAX_ROUNDS + 1):
        champion_before = champion
        # diversity: spread temperature across the swarm so workers don't return identical code
        args = [(champion, history, 0.4 + 0.5 * (i / N_PER_ROUND)) for i in range(N_PER_ROUND)]

        # THE SWARM: N containers, each doing an LLM rewrite + score, in parallel.
        # return_exceptions=True so one bad container doesn't kill the round (scale.md).
        results = [r for r in try_one.starmap(args, return_exceptions=True) if isinstance(r, dict)]

        # log what every worker produced (one line each; full source + diff go to debug.jsonl)
        for idx, r in enumerate(results):
            debuglog.log_attempt(round_n, idx, champion_before, r,
                                 model=r.get("model", "?"), temperature=r.get("temperature", 0.0))

        survivors = [r for r in results if r["passed"]]
        kept = False
        if survivors:
            top = max(survivors, key=lambda r: r["speedup"])
            if top["speedup"] > best:                       # KEEP
                champion, best, kept = top["src"], top["speedup"], True
                print(f"  winner of round {round_n} (speedup={best:.1f}x) — change applied to the code file:")
                debuglog.show_change(champion_before, champion)
        history += f"round {round_n}: best={best:.2f}x, {len(survivors)}/{len(results)} correct\n"
        debuglog.log_round(round_n, results, best, kept)
