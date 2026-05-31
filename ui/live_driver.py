"""Adapter B (capture): run the REAL swarm once, write ui/events.jsonl for replay.

Backend is untouched. This imports the unmodified swarm (optimization/app_rust.py) and
re-runs the same per-round loop as evolve_rust, translating each REAL per-attempt result
into the dashboard's event schema. The ONLY synthesized thing is per-cell grid
pacing/animation (Modal .starmap() is blocking, so true per-cell streaming doesn't exist);
every number, verdict, and champion code written here is the real captured value.

    python ui/live_driver.py            # default polynomial hot-loop target
    N_PER_ROUND=10 MAX_ROUNDS=3 python ui/live_driver.py

Mapping (as agreed): description=verdict, code=src, status from passed/error/speedup,
reward_hack=True when the verdict shows a correctness/differential mismatch, journal
keep/discard/crash derived from passed and speedup>best.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Make app_rust's own imports resolve exactly as they do under `modal run`: optimization/
# must be FIRST so `import mutate` finds optimization/mutate.py (not the root mutate.py).
# app_rust appends the repo root itself for `target_spec`.
sys.path.insert(0, os.path.join(ROOT, "optimization"))

import app_rust  # the backend swarm, UNMODIFIED

OUT = os.path.join(HERE, "events.jsonl")
N = app_rust.N_PER_ROUND
ROUNDS = app_rust.MAX_ROUNDS


def _classify(r: dict):
    """(cell_status, reward_hack) from a real result dict. No values invented."""
    err = (r.get("error") or "")
    verdict = (r.get("verdict") or "")
    if not r.get("passed"):
        if "compile" in err.lower() or "compile" in verdict.lower():
            return "crashed", False
        # correctness / differential mismatch == reward-hack-shaped rejection
        hack = ("correctness FAIL" in verdict) or ("differential mismatch" in (err + verdict))
        return "failed", hack
    return "passed", False


def main():
    events = []
    emit = events.append

    with app_rust.app.run():
        spec = app_rust._load_spec()
        spec_json = spec.to_json()
        champion, best = spec.seed(), 1.0
        jid = 0

        # Baseline + seed champion (real seed Rust source).
        jid += 1
        emit({"type": "journal", "id": jid, "speedup": 1.0, "status": "keep",
              "description": "seed naive port (baseline)"})
        emit({"type": "champion", "round": 0, "speedup": 1.0, "cell": None,
              "description": "seed naive port", "why": "known-correct floor for the ratchet",
              "code": champion})

        for rnd in range(1, ROUNDS + 1):
            emit({"type": "round_start", "round": rnd, "target": spec.name, "n_cells": N})

            # ---- the REAL swarm round (identical to evolve_rust) ----
            args = [(spec_json, champion, "", 0.4 + 0.5 * (i / N)) for i in range(N)]
            results = list(app_rust.try_one_rust.starmap(args, return_exceptions=True))
            results = [r if isinstance(r, dict)
                       else {"passed": False, "speedup": 0.0, "error": repr(r),
                             "verdict": "worker exception", "src": ""}
                       for r in results]

            # ---- translate to events (pacing/animation synthesized; data real) ----
            for i in range(len(results)):
                emit({"type": "cell_update", "round": rnd, "cell": i, "status": "spawning"})

            round_best, champ_cell, champ_src, champ_desc = best, -1, None, ""
            for i, r in enumerate(results):
                status, hack = _classify(r)
                sp = float(r.get("speedup", 0.0))
                desc = r.get("verdict") or r.get("error") or ""
                model = r.get("model", "")

                emit({"type": "cell_update", "round": rnd, "cell": i, "status": "compiling", "model": model})
                if status != "crashed":
                    emit({"type": "cell_update", "round": rnd, "cell": i, "status": "testing"})
                    if status == "passed":
                        emit({"type": "cell_update", "round": rnd, "cell": i, "status": "benchmarking"})

                emit({"type": "cell_result", "round": rnd, "cell": i, "passed": bool(r.get("passed")),
                      "speedup": sp, "status": status, "description": desc, "reward_hack": hack})

                jid += 1
                if status == "crashed":
                    emit({"type": "journal", "id": jid, "speedup": 0.0, "status": "crash", "description": desc})
                elif status == "failed":
                    emit({"type": "journal", "id": jid, "speedup": 0.0, "status": "discard",
                          "description": desc, "reward_hack": hack})
                else:  # passed
                    improved = sp > round_best
                    emit({"type": "journal", "id": jid, "speedup": sp,
                          "status": "keep" if improved else "discard", "description": desc})
                    if improved:
                        round_best, champ_cell, champ_src, champ_desc = sp, i, r.get("src"), desc

            kept = round_best > best
            if kept:
                best, champion = round_best, champ_src
                emit({"type": "champion", "round": rnd, "speedup": best, "cell": champ_cell,
                      "description": champ_desc, "why": champ_desc, "code": champ_src})
            emit({"type": "round_end", "round": rnd, "best_speedup": best, "kept": kept})

    with open(OUT, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    champs = [e for e in events if e["type"] == "champion"]
    print(f"wrote {len(events)} events -> {OUT}")
    print(f"final champion speedup (real): {best:.2f}x over {ROUNDS} rounds, "
          f"{len(champs)} champion events")


if __name__ == "__main__":
    main()
