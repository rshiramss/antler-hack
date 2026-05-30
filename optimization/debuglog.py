# debuglog.py — verbose, structured logging so you can see exactly what the LLM does each attempt.
#
# Two outputs per attempt:
#   1. A line + diff on the console (Modal streams worker/entrypoint stdout back live).
#   2. A JSON record appended to debug.jsonl — the durable, greppable trail (full source + diff).
import difflib
import json
import os
from datetime import datetime

DEBUG_PATH = os.path.join(os.path.dirname(__file__), "debug.jsonl")


def reset(path: str = DEBUG_PATH) -> None:
    """Truncate the debug log at the start of a run."""
    open(path, "w").close()


def diff(old_src: str, new_src: str) -> str:
    """Unified diff champion -> candidate: exactly what the LLM changed in the code file."""
    return "".join(difflib.unified_diff(
        old_src.splitlines(keepends=True),
        new_src.splitlines(keepends=True),
        fromfile="champion", tofile="candidate",
    ))


def _print_diff(d: str) -> None:
    if d:
        print("    --- diff champion -> candidate ---")
        for line in d.splitlines():
            print("      " + line)
    else:
        print("    (identical to champion — no change)")


def show_change(champion_src: str, new_src: str) -> None:
    """Print the diff for a chosen candidate (e.g. the round winner)."""
    _print_diff(diff(champion_src, new_src))


def log_attempt(round_n: int, idx: int, champion_src: str, result: dict, *,
                model: str, temperature: float, show_diff: bool = False,
                path: str = DEBUG_PATH) -> dict:
    """Record one LLM attempt: what it produced, whether it passed, how fast, and the diff."""
    d = diff(champion_src, result.get("src", ""))
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "round": round_n,
        "idx": idx,
        "model": model,
        "temperature": temperature,
        "passed": result.get("passed"),
        "speedup": round(result.get("speedup", 0.0), 4),
        "error": result.get("error", ""),
        "diff": d,
        "src": result.get("src", ""),
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

    verdict = ("PASS %.1fx" % record["speedup"]) if record["passed"] \
        else f"FAIL ({record['error'] or 'no output'})"
    print(f"  [r{round_n} #{idx}] {model} temp={temperature:.2f} -> {verdict}")
    if show_diff:
        _print_diff(d)
    return record


def log_round(round_n: int, results: list, best: float, kept: bool,
              path: str = DEBUG_PATH) -> dict:
    """Record the round decision: how many were correct and whether the champion advanced."""
    n_ok = sum(1 for r in results if r.get("passed"))
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "event": "round_summary",
        "round": round_n,
        "n_correct": n_ok,
        "n_total": len(results),
        "champion_speedup": round(best, 4),
        "kept_new_champion": kept,
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    tag = "NEW CHAMPION" if kept else "no improvement"
    print(f"== round {round_n}: {n_ok}/{len(results)} correct | champion={best:.2f}x | {tag} ==")
    return record
