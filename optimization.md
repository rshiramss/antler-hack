# optimization.md — An LLM-Driven Autoresearch Loop + Modal Swarm

> **Goal of this doc.** Lay out, step by step, the smallest thing that still
> captures Karpathy's autoresearch idea **with a real LLM in the loop**: a
> language model **edits the Python code** each round, a frozen evaluator scores
> it on one number, and we keep-if-better — then fan that out into a **Modal
> swarm** so many LLM rewrites run in parallel per round.
>
> This is the on-ramp. The full Rust-port system is in `build.md` and
> `RustForge_Project_Breakdown.md`. Everything here is grounded only in the
> local docs in this repo (`auto_research/`, `modal_docs/`, `litellm_docs/`).

---

## 0. The idea in one breath

Karpathy's `auto_research/` repo (`README.md`, `program.md`) is three files that
matter: a **frozen** setup/evaluator (`prepare.py`), one **mutable** file *an
agent edits* (`train.py`), and **instructions** the human edits (`program.md`).
The loop is: **the LLM changes the mutable file → run → read one number → keep if
better, else revert → repeat.** Results go in a `results.tsv` journal.

We copy that loop exactly. The mutation operator is a **real LLM call** — given
the current champion code, the journal of what's been tried, and the goal from
`program.md`, the model returns a rewritten `solve(x)`. No hand-written strategy
list; the LLM does the editing, like a researcher hacking on `train.py`.

The one divergence from Karpathy — the part that needs Modal — is that **each
round fans out into a parallel swarm of LLM rewrites** instead of one at a time:

```
Round 1:  [ N LLM rewrites of the seed — PARALLEL on Modal ] → keep best correct → Champion A
                                                                                       |
Round 2:  [ N LLM rewrites of Champion A — PARALLEL ] → keep best → Champion B          |
                                                                                       v
Round 3:  ...
```

### Autoresearch → this loop (one-to-one)

| Autoresearch (`auto_research/`) | Here | Notes |
|---|---|---|
| `train.py` — the one mutable file an agent edits | **`candidate.py`** / the `solve(x)` source the **LLM rewrites** | Tiny scope, reviewable diffs |
| `program.md` — human-edited instructions to the agent | **`program.md`** — the prompt/goal handed to the LLM | The "research org code" you iterate on |
| The agent (Claude/Codex) editing the file | **`mutate.propose()`** — an LLM call via `litellm` | The mutation operator |
| `prepare.py` — frozen setup + `evaluate_bpb` metric | **`evaluate.py`** — frozen correctness gate + benchmark | The LLM can't touch it, so it can't game it |
| `val_bpb` (lower = better) | **`speedup`** (higher = better), **gated to 0 on any correctness failure** | One comparable scalar per attempt |
| git commit vs `git reset`; `results.tsv` | **champion replacement + `results.tsv`** | Same keep/discard ratchet + journal |
| sequential, ~12 exp/hour, one GPU | **N-wide Modal swarm per round** | `.map()` / `.starmap()` fan-out (`modal_docs/guide/scale.md`) |

---

## 1. File structure (mirrors Karpathy's "three files that matter")

```
optimization/
  target.py        # FROZEN. The Python function we want faster (the oracle) + constants.   (≈ prepare.py)
  evaluate.py      # FROZEN. Correctness gate (differential test) + benchmark → speedup.     (≈ prepare.py eval)
  program.md       # The goal/instructions handed to the LLM each round. Human-edited.       (≈ program.md)
  candidate.py     # MUTABLE. The seed solve(x); the LLM rewrites this each attempt.          (≈ train.py)
  mutate.py        # The mutation operator: an LLM call (litellm) that rewrites solve(x).
  loop.py          # The keep/discard ratchet + journal. Sequential, local (pure autoresearch).
  app.py           # Modal app: the swarm worker (LLM rewrite + score) + the fan-out.
  results.tsv      # Journal (untracked, like autoresearch). Header + one row per attempt.
  champion.py      # Output: the best solve(x) found.
```

Keep it this small. No UI, no config system, no second target until the loop +
swarm are green.

---

## 2. Prerequisites

```bash
# Deps (uv, like autoresearch — or plain venv + pip)
uv add modal numpy litellm python-dotenv     # or: pip install ...

# Authenticate Modal once (opens a browser; stores a token locally)
modal setup                                   # or: modal token new
```

**Config via `.env` (not `export`).** litellm reads the key from the process
environment but does **not** auto-load a `.env` (litellm_docs only sets
`os.environ[...]`), so we load it ourselves with `python-dotenv`. Create a `.env`:

```bash
# .env  — gitignored; never commit. Commit a keyless .env.example instead.
OPENAI_API_KEY=your-openai-api-key
OPT_MODEL=gpt-5.5            # newest frontier model (May 2026); see §5 for options
```

- **Local loop:** `mutate.py` calls `load_dotenv(override=True)` once at import,
  so `.env` is picked up automatically and beats stale shell vars.
- **Swarm:** `modal.Secret.from_dotenv()` reads the same `.env` and injects it into
  every container (`modal_docs/guide/secrets.md`) — **no `modal secret create` step
  needed**. (Needs `python-dotenv` installed.)
- `modal setup` is the only account step (`modal_docs/guide/modal-user-account-setup.md`).

> **Why `.env` over `export`:** keys stay out of shell history, live in one
> gitignored file, and the same file feeds both the local loop and the swarm.

---

## 3. Step 1 — The frozen target + evaluator (the metric)

This is the `prepare.py` analog: it never changes, and it defines fitness. The
metric is **speedup vs the reference, gated to 0 if the candidate is wrong**.

`target.py` — the reference function (the oracle) and a fixed input generator:

```python
# target.py  —  FROZEN. Do not let the LLM edit this.
import numpy as np

RNG_SEED = 0
INPUT_SIZE = 1_000_000

def reference(x: np.ndarray) -> np.ndarray:
    """The slow, obviously-correct Python version. Ground truth."""
    out = np.empty_like(x)
    for i in range(x.shape[0]):          # deliberately naive: a hot Python loop
        out[i] = x[i] * x[i] + 1.0
    return out

def make_inputs(n: int = INPUT_SIZE) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED)
    return rng.standard_normal(n)
```

`evaluate.py` — correctness gate + benchmark. Returns one scalar. **Frozen.**

```python
# evaluate.py  —  FROZEN. The metric. The LLM cannot touch this, so it can't game it.
import time
import numpy as np
from target import reference, make_inputs

RTOL, ATOL = 1e-9, 1e-9      # differential-correctness tolerance
N_FUZZ      = 20             # random input batches the LLM never saw at generation time
N_BENCH     = 5             # benchmark repeats; take the median

def evaluate(candidate_fn) -> dict:
    """Score one candidate. speedup is gated to 0.0 on any correctness failure."""
    rng = np.random.default_rng(12345)   # NOT the target seed — unseen inputs

    # 1) Correctness gate: differential test on fresh random inputs.
    for _ in range(N_FUZZ):
        x = rng.standard_normal(rng.integers(1, 5000))
        try:
            got = candidate_fn(x)
        except Exception as e:
            return {"passed": False, "speedup": 0.0, "error": repr(e)}
        if not np.allclose(got, reference(x), rtol=RTOL, atol=ATOL):
            return {"passed": False, "speedup": 0.0, "error": "differential mismatch"}

    # 2) Benchmark: median wall-time of candidate vs reference on the same input.
    x = make_inputs()
    ref_t = _median_time(reference, x)
    cand_t = _median_time(candidate_fn, x)
    return {"passed": True, "speedup": ref_t / cand_t, "error": ""}

def _median_time(fn, x) -> float:
    fn(x)                                 # warm up
    times = []
    for _ in range(N_BENCH):
        t0 = time.perf_counter(); fn(x); times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]
```

> **Why this is the anti-cheat moat.** The fuzz inputs use a *different seed* and
> random shapes, generated inside the frozen evaluator. The LLM never sees them at
> generation time, so hardcoding outputs is structurally impossible. A wrong
> candidate scores `0.0` no matter how fast — correctness is a hard gate, not a
> weighted term. (Full version uses Hypothesis; see `hypothesis_docs/`.)

---

## 4. Step 2 — The mutable candidate + `program.md` (what the LLM edits, and the goal)

`candidate.py` is the `train.py` analog — the one thing the LLM rewrites. The seed
is a literal copy of the reference (a known-correct floor for the ratchet):

```python
# candidate.py  —  MUTABLE. Seed = the slow reference. The LLM rewrites solve(x) to go faster.
import numpy as np

def solve(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = x[i] * x[i] + 1.0
    return out
```

`program.md` is the `program.md` analog — the instructions/goal handed to the LLM
every round. **You (the human) iterate on this file**, exactly like autoresearch:

```markdown
# program.md — instructions to the optimizing LLM

You are an autonomous performance researcher. Your job: rewrite the Python
function `solve(x)` so it returns the SAME result as the frozen reference, but
runs faster.

## Rules
- `solve(x)` takes a 1-D NumPy float array and must return an array equal to the
  reference within rtol=1e-9, atol=1e-9. Any mismatch scores 0 — correctness is a
  hard gate, not a tradeoff.
- You may use NumPy and the Python standard library only. No other packages.
- The fuzz inputs are hidden from you, so never special-case or hardcode outputs.
- Return the COMPLETE module as a single ```python code block defining `solve(x)`.
  No prose, no explanation.

## Metric
`speedup = reference_time / candidate_time`, higher is better, gated to 0 on any
correctness failure.

## Style (autoresearch simplicity criterion)
All else equal, simpler is better. A small speedup that adds ugly complexity is
not worth it; an equal result with simpler code is a win.
```

---

## 5. Step 3 — The mutation operator: an LLM that edits the code (`mutate.py`)

This replaces autoresearch's "a coding agent edits `train.py`" with one `litellm`
call. Given the champion source + the journal so far + `program.md`, the model
returns a rewritten module. (`litellm` usage from `litellm_docs/`: `completion(...)`,
text at `resp.choices[0].message.content`.)

```python
# mutate.py  —  the mutation operator IS an LLM that rewrites solve(x).
import os
import re
from dotenv import load_dotenv
from litellm import completion

load_dotenv(override=True)                       # .env is authoritative (beats stale shell vars); no-op in swarm containers

# Newest models, May 2026 (confirm ids against your provider):
#   gpt-5.5         — newest frontier, Chat Completions (default)
#   gpt-5.3-codex   — most capable agentic coding model (best for code edits)
#   gpt-5.4-mini    — cheap/fast; good for breadth across a wide swarm
#   claude-...      — any Anthropic model, if you set ANTHROPIC_API_KEY instead
MODEL = os.environ.get("OPT_MODEL", "gpt-5.5")   # any litellm-supported model id

# The seed candidate (string form), used to start the ratchet.
SEED = '''import numpy as np
def solve(x):
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = x[i]*x[i] + 1.0
    return out
'''

_PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "program.md")   # resolves regardless of cwd

def _program() -> str:
    with open(_PROGRAM_PATH) as f:
        return f.read()

def propose(champion_src: str, history: str, temperature: float = 0.7) -> str:
    """Ask the LLM to rewrite solve() faster while staying correct. Returns new source."""
    messages = [
        {"role": "system", "content": _program()},
        {"role": "user", "content": (
            f"Current champion code:\n```python\n{champion_src}\n```\n\n"
            f"Journal of past attempts (most recent last):\n{history or '(none yet)'}\n\n"
            "Propose ONE change that makes it faster while keeping it numerically "
            "identical. Return the COMPLETE new module as a single ```python block."
        )},
    ]
    return _extract_code(_complete(messages, temperature))

def _supports_temperature(model: str) -> bool:
    """Reasoning-family models (gpt-5.x, o-series) only allow the default temperature."""
    m = model.lower()
    return not (m.startswith("gpt-5") or re.match(r"^o\d", m))

def _complete(messages, temperature: float) -> str:
    """Call the model, sending temperature only to models that support it.
    Keeps a fallback in case an unanticipated model also locks temperature."""
    kwargs = {"model": MODEL, "messages": messages}
    if _supports_temperature(MODEL):
        kwargs["temperature"] = temperature
    try:
        resp = completion(**kwargs)
    except Exception as e:
        if "temperature" in str(e).lower():          # safety net for unanticipated locked models
            kwargs.pop("temperature", None)
            resp = completion(**kwargs)
        else:
            raise
    return resp.choices[0].message.content

def _extract_code(text: str) -> str:
    """Pull the python source out of the model's ```python ...``` block."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()

def load_solve(src: str):
    """Turn a candidate source string into a callable solve(x)."""
    ns: dict = {}
    exec(src, ns)            # candidate is trusted-ish locally; Modal sandboxes it in the swarm
    return ns["solve"]
```

> `temperature` is the diversity knob. The local loop (§6) uses one call per round;
> the swarm (§7) spreads temperature across N parallel calls so workers don't all
> return identical code.
>
> **Caveat — gpt-5.x ignores the temperature spread.** Reasoning-family models
> (gpt-5.5, gpt-5.4, …) only accept the default `temperature=1`, so `_complete()`
> drops it for them and the swarm's spread has no effect. Diversity then comes from
> sampling variance at temp=1 (plus, if you need more, per-worker prompt nudges).
> To actually use the temperature spread, pick a temperature-supporting model
> (e.g. `gpt-4.1`) or a Claude model. Verified live: `gpt-5.5` rewrote the seed
> loop to `x*x + 1.0` → **516.7×**, passing the frozen gate.
>
> **Reward-hacking shows up naturally now.** Sometimes the LLM returns code that's
> fast but subtly wrong (drops the `+1`, mishandles dtype). The differential gate
> scores it `0.0` and the journal logs it as a `crash`/wrong — which is exactly the
> behavior you demo.

---

## 6. Step 4 — The local keep/discard loop (pure autoresearch, NO Modal yet)

Run this first, entirely local: one LLM rewrite per round, keep-if-better, journal
it. This is `auto_research/program.md`'s "make one change → run → keep/discard →
repeat", with the LLM making the change.

```python
# loop.py  (sequential, local; §7 swaps the single call for the Modal swarm)
import os
from evaluate import evaluate
from mutate import propose, load_solve, SEED

MAX_ROUNDS = 10

_HERE = os.path.dirname(__file__)
_JOURNAL = os.path.join(_HERE, "results.tsv")     # outputs land next to the loop, like autoresearch
_CHAMPION = os.path.join(_HERE, "champion.py")

def attempt(src: str) -> dict:
    """Score one candidate. (This becomes the Modal worker in §7.)"""
    try:
        return {**evaluate(load_solve(src)), "src": src}
    except Exception as e:
        return {"passed": False, "speedup": 0.0, "error": repr(e), "src": src}

def main():
    champion, best, history = SEED, 1.0, ""
    with open(_JOURNAL, "w") as f:
        f.write("round\tspeedup\tstatus\tdescription\n")
    for round_n in range(1, MAX_ROUNDS + 1):
        cand = propose(champion, history)              # <-- the LLM edits the code
        r = attempt(cand)

        # status is computed against the OLD best, before we update the champion
        status = "keep" if (r["passed"] and r["speedup"] > best) else \
                 ("discard" if r["passed"] else "crash")
        with open(_JOURNAL, "a") as f:
            desc = r["error"] or f"speedup={r['speedup']:.2f}x"
            f.write(f"{round_n}\t{r['speedup']:.4f}\t{status}\t{desc}\n")
        history += f"round {round_n}: {status}, speedup={r['speedup']:.2f}x {r['error']}\n"

        if status == "keep":                           # KEEP: advance the champion
            champion, best = r["src"], r["speedup"]
        print(f"round {round_n}: {status}  champion={best:.2f}x")

    with open(_CHAMPION, "w") as f:                    # save the winner
        f.write(champion)

if __name__ == "__main__":
    main()
```

```bash
# key comes from .env via load_dotenv() — no export needed
uv run python optimization/loop.py    # the LLM rewrites solve() each round; results.tsv fills up
```

**Checkpoint:** the journal shows `keep` rows where the LLM found a real speedup
(e.g. vectorizing the loop → hundreds of ×), `discard` where it didn't beat the
champion, and `crash` where it produced wrong/broken code. The ratchet works. Now
make it a swarm.

---

## 7. Step 5 — The Modal swarm (the part you want to "see work")

### 7a. Smoke test first: prove fan-out works at all

Confirm auth + parallel containers with the canonical Modal hello-swarm
(`modal_docs/examples/hello_world.md`):

```python
# swarm_hello.py
import modal
app = modal.App("swarm-hello")

@app.function()
def square(i: int) -> int:
    return i * i

@app.local_entrypoint()
def main():
    print(sum(square.map(range(200))))   # 200 inputs, run roughly in parallel in the cloud
```

```bash
uv run modal run optimization/swarm_hello.py     # prints 2646700 if your swarm plumbing works
```

`.map()` is a method on the *function object* — you don't call `square(...)`
yourself (`scale.md` "Gotchas").

### 7b. Fan out N LLM rewrites per round

Each worker does the **whole attempt in the cloud**: its own LLM rewrite + score.
That parallelizes the LLM calls too — the "megastructure." The provider key
reaches each container via a Modal Secret; `program.md` / `mutate.py` / the frozen
files are baked into the image.

```python
# app.py
import os
import modal
from mutate import SEED

app = modal.App("optimization-swarm")

# Image: NumPy + litellm + python-dotenv, plus the frozen files and prompt at /root (on sys.path).
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
    from mutate import propose, load_solve
    from evaluate import evaluate
    try:
        src = propose(champion_src, history, temperature)   # the LLM edits the code, in the cloud
        return {**evaluate(load_solve(src)), "src": src}
    except Exception as e:
        return {"passed": False, "speedup": 0.0, "error": repr(e), "src": ""}

N_PER_ROUND = int(os.environ.get("N_PER_ROUND", "20"))   # swarm width; ≤1000 concurrent per call (scale.md)
MAX_ROUNDS  = int(os.environ.get("MAX_ROUNDS", "5"))     # override both for a small test, e.g. N_PER_ROUND=3 MAX_ROUNDS=1

@app.local_entrypoint()
def evolve():
    champion, best, history = SEED, 1.0, ""
    for round_n in range(1, MAX_ROUNDS + 1):
        # diversity: spread temperature across the swarm so workers don't return identical code
        args = [(champion, history, 0.4 + 0.5 * (i / N_PER_ROUND)) for i in range(N_PER_ROUND)]

        # THE SWARM: N containers, each doing an LLM rewrite + score, in parallel.
        # return_exceptions=True so one bad container doesn't kill the round (scale.md).
        results = [r for r in try_one.starmap(args, return_exceptions=True) if isinstance(r, dict)]

        survivors = [r for r in results if r["passed"]]
        if survivors:
            top = max(survivors, key=lambda r: r["speedup"])
            if top["speedup"] > best:                       # KEEP
                champion, best = top["src"], top["speedup"]
        history += f"round {round_n}: best={best:.2f}x, {len(survivors)}/{len(results)} correct\n"
        print(f"round {round_n}: {len(survivors)}/{len(results)} correct, champion={best:.2f}x")
```

```bash
uv run modal run optimization/app.py::evolve     # N LLM rewrites per round, scored in parallel on Modal
```

`.starmap()` *is* the megastructure: each tuple lands in its own container, Modal
bursts out concurrently and streams results back; the outer loop keeps the fastest
correct survivor and feeds it into the next round's prompt.

### Useful swarm knobs (all from `modal_docs/guide/scale.md`)

- **Stream as they finish:** `order_outputs=False` returns results in completion
  order (good for a live UI later).
- **Limits:** ≤1000 inputs concurrent per call; 2,000 pending / 25,000 total per
  function. Stay well under for a smoke test.
- **Warm pool / cost:** `min_containers` keeps containers warm; default scales to
  zero when idle.

### Watching the agent work — debug logs (`debuglog.py`)

To see *exactly* what the LLM does each attempt, both `loop.py` and `app.py` call
`debuglog`, which writes two things per attempt:

- **Console (live):** one verdict line per attempt — `[r1 #3] gpt-5.5 temp=0.50 ->
  PASS 712.3x` or `-> FAIL (differential mismatch)` — and, for the local loop and
  each round's winner, the **unified diff champion → candidate** (the actual code
  change). Modal streams worker stdout back, so you also get per-container lines
  like `[worker temp=0.50] gpt-5.5: proposed 64 chars`.
- **`debug.jsonl` (durable, gitignored):** one JSON record per attempt with
  `round, idx, model, temperature, passed, speedup, error, diff, src` — the full
  proposed source and diff for every candidate, plus a `round_summary` record.

```python
# the wiring (already in loop.py / app.py):
debuglog.reset()                                   # fresh debug.jsonl at run start
debuglog.log_attempt(round_n, idx, champion_before, result,
                     model=MODEL, temperature=t, show_diff=...)   # per attempt
debuglog.show_change(champion_before, champion)    # the diff that became the new champion
debuglog.log_round(round_n, results, best, kept)   # round verdict
```

Read it back later, e.g. every kept change:

```bash
# what each round's champion changed, in order
python -c "import json; [print(r['round'], r['diff']) for r in map(json.loads, open('optimization/debug.jsonl')) if r.get('passed')]"
# or just the verdicts
grep -o '\"round\": [0-9]*, .*\"speedup\": [0-9.]*' optimization/debug.jsonl
```

If you'd rather it be quieter, set `show_diff=False` (console keeps the one-line
verdicts; full diffs still land in `debug.jsonl`).

---

## 8. Step 6 — Swap in the *real* target (Python → Rust port)

The loop, the LLM mutation operator, the journal, the fan-out, and the gated-speedup
metric all stay identical. Only two things change:

- **Mutable artifact:** the LLM rewrites `src/lib.rs` (Rust via PyO3) instead of
  `candidate.py`. `mutate.propose()` already returns code — point its prompt at Rust.
- **Worker `try_one`:** write the candidate to `src/lib.rs` → `maturin build
  --release` → import the compiled module → run the differential gate → benchmark.
  Add the Rust toolchain to the image (`.apt_install("curl", "build-essential")`,
  `.run_commands(<rustup install>)`, `.pip_install("maturin")`) and mount a
  `modal.Volume` as a shared cargo `target/` cache (RustForge §7–8;
  `modal_docs/guide/volumes.md`).
- **Target:** the Whisper mel-filterbank projection (`RustForge_Project_Breakdown.md`
  §6), with Mandelbrot as the watchable fallback.

Full details, pinned versions, and the hour-by-hour plan are in **`build.md`** and
**`RustForge_Project_Breakdown.md`** — `optimization.md` is the minimal skeleton
that proves the two hard mechanics (the LLM keep/discard ratchet and the swarm)
before taking on the PyO3/maturin FFI.

---

## 9. Build order (checklist)

1. [ ] `target.py` + `evaluate.py` — frozen metric; `evaluate(reference)` ≈ 1.0x, `passed=True`.
2. [ ] `program.md` + `candidate.py` — the goal + the seed `solve(x)`.
3. [ ] `mutate.py` — `propose()` returns a parseable module; one LLM call works locally.
4. [ ] `loop.py` — local, one LLM rewrite/round; `results.tsv` ratchets up; `champion.py` saved. **Demoable even with no cloud.**
5. [ ] `.env` (gitignored) with `OPENAI_API_KEY` + `OPT_MODEL`; `load_dotenv()` picks it up. Commit a keyless `.env.example`.
6. [ ] `swarm_hello.py` — `modal run` prints `2646700`. Auth + fan-out confirmed.
7. [ ] `app.py` — `Secret.from_dotenv()` feeds the key to workers; `try_one` does LLM rewrite + score on Modal; `evolve` fans out N/round; champion speedup climbs.
8. [ ] (later) Swap to the Rust-port target per §8 → `build.md`.

> Stop after step 7 for "see how it works." That's a real autoresearch loop — an
> LLM editing the code, kept-if-better — running as a parallel swarm on Modal.
