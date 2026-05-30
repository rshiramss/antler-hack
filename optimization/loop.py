# loop.py  (sequential, local; app.py swaps the single call for the Modal swarm)
import os

import debuglog
from evaluate import evaluate
from mutate import propose, load_solve, SEED, MODEL

MAX_ROUNDS = 10

_HERE = os.path.dirname(__file__)
_JOURNAL = os.path.join(_HERE, "results.tsv")     # outputs land next to the loop, like autoresearch
_CHAMPION = os.path.join(_HERE, "champion.py")


def attempt(src: str) -> dict:
    """Score one candidate. (This becomes the Modal worker in app.py.)"""
    try:
        return {**evaluate(load_solve(src)), "src": src}
    except Exception as e:
        return {"passed": False, "speedup": 0.0, "error": repr(e), "src": src}


def main():
    champion, best, history = SEED, 1.0, ""
    with open(_JOURNAL, "w") as f:
        f.write("round\tspeedup\tstatus\tdescription\n")
    debuglog.reset()                                   # fresh debug.jsonl for this run

    for round_n in range(1, MAX_ROUNDS + 1):
        champion_before = champion
        cand = propose(champion, history)              # <-- the LLM edits the code

        r = attempt(cand)
        # show exactly what the LLM produced this round (diff vs champion) + verdict
        debuglog.log_attempt(round_n, 0, champion_before, r,
                             model=MODEL, temperature=0.7, show_diff=True)

        # status is computed against the OLD best, before we update the champion
        status = "keep" if (r["passed"] and r["speedup"] > best) else \
                 ("discard" if r["passed"] else "crash")
        with open(_JOURNAL, "a") as f:
            desc = r["error"] or f"speedup={r['speedup']:.2f}x"
            f.write(f"{round_n}\t{r['speedup']:.4f}\t{status}\t{desc}\n")
        history += f"round {round_n}: {status}, speedup={r['speedup']:.2f}x {r['error']}\n"

        kept = status == "keep"
        if kept:                                       # KEEP: advance the champion
            champion, best = r["src"], r["speedup"]
        debuglog.log_round(round_n, [r], best, kept)

    with open(_CHAMPION, "w") as f:                    # save the winner
        f.write(champion)


if __name__ == "__main__":
    main()
