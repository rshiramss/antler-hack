# RustForge — Knowledgebase Dump

*Everything built and learned so far: the premise, the working minimal loop, how we
built it, proof it works, and what's left. Self-contained snapshot for the RustForge
knowledgebase. Date: 2026-05-30.*

---

## 1. What RustForge is (the premise)

A swarm-based, autoresearch-style platform that takes a Python codebase, finds the
functions most worth optimizing, and **evolves verified-faster Rust ports of them**.
It is Andrej Karpathy's `autoresearch` keep/discard loop — one mutable file, a frozen
evaluator, a single scalar metric, keep-if-better — with each sequential round fanned
out into a massively parallel **Modal** swarm, and every candidate gated on
differential-fuzz correctness so the speedup is always real.

Built for the **Autoresearch Systems Hackathon (Modal · OpenAI · Raindrop · Antler), May 30 2026.**

- **Primary prize (Modal):** a visibly parallel "megastructure" of agents doing real
  work — hundreds of concurrent containers compiling/benchmarking — around a legible
  problem ("rewrite-it-in-Rust to make it fast").
- **Side (Raindrop):** A/B the swarm's LLM mutation operators live (model+prompt → faster correct code).
- **Antler:** "rewrite-it-in-Rust as a service" — automated native acceleration for any Python codebase.

**The one-sentence pitch:** Karpathy's keep/discard loop applied to Python→Rust
porting, each round fanned out into a Modal swarm, every candidate gated on
differential-fuzz correctness so the speedup is provably real.

---

## 2. The autoresearch mapping (the conceptual spine)

| Autoresearch (`auto_research/`) | RustForge / our loop | Notes |
|---|---|---|
| `train.py` — the one mutable file an agent edits | the `solve(x)` source the **LLM rewrites** (later `src/lib.rs`) | tiny scope, reviewable diffs |
| `program.md` — human-edited agent instructions | `program.md` — the prompt/goal handed to the LLM | the "research org code" you iterate on |
| the agent (Claude/Codex) editing the file | `mutate.propose()` — a `litellm` call | the mutation operator |
| `prepare.py` — frozen setup + `evaluate_bpb` metric | `evaluate.py` — frozen correctness gate + benchmark | LLM can't touch it → can't game it |
| `val_bpb` (lower=better) | `speedup` (higher=better), **gated to 0 on any correctness failure** | one comparable scalar per attempt |
| git commit vs `git reset`; `results.tsv` | champion replacement + `results.tsv` | same keep/discard ratchet + journal |
| sequential, ~12 exp/hr, 1 GPU | **N-wide Modal swarm per round** | `.starmap()` fan-out |

**The one deliberate divergence:** autoresearch is strictly sequential; RustForge keeps
the exact outer ratchet but fans each round into a parallel swarm. Parallel *within* a
round, sequential *across* rounds.

---

## 3. What we actually built (the minimal working loop)

A Python→Python skeleton that proves the two hard mechanics (the LLM keep/discard
ratchet + the Modal swarm) before taking on the PyO3/maturin Rust FFI. Target function:
a deliberately-naive `reference(x)` that computes `x*x + 1.0` via a Python loop over 1M
floats — slow on purpose, so vectorizing it gives a real, large speedup.

### File structure
```
optimization/
  target.py        # FROZEN. reference(x) (the oracle, naive loop) + make_inputs + constants.
  evaluate.py      # FROZEN. correctness gate (differential fuzz) + benchmark → speedup.
  program.md       # the goal/instructions handed to the LLM each round. human-edited.
  candidate.py     # the seed solve(x) (mutable artifact, literal copy of reference).
  mutate.py        # the mutation operator: an LLM call (litellm) that rewrites solve(x).
  loop.py          # sequential local keep/discard ratchet + journal (pure autoresearch, NO Modal).
  app.py           # Modal app: swarm worker (LLM rewrite + score) + .starmap fan-out + ratchet.
  swarm_hello.py   # Modal smoke test (square.map over 200 inputs → 2646700).
  debuglog.py      # verbose per-attempt logging: console verdict+diff, durable debug.jsonl.
  verify_evidence.py # independent proof script (timeit re-measure + negative controls).
  results.tsv      # journal (untracked). champion.py — best solve found (untracked).
  debug.jsonl      # full per-attempt trail: round, model, temp, passed, speedup, error, diff, src.
# project root: pyproject.toml, uv.lock, .env (gitignored), optimization.md (the build doc)
```

### The metric (`evaluate.py`, FROZEN — the anti-cheat moat)
- **Correctness gate:** `N_FUZZ=20` random inputs, seed `12345` (NOT the target seed),
  random shapes 1–5000. Compare candidate vs `reference` with
  `np.allclose(rtol=1e-9, atol=1e-9)`. Any mismatch/exception → `{passed: False, speedup: 0.0}`.
- **Benchmark (only if correct):** `speedup = ref_t / cand_t`, each the **median of
  `N_BENCH=5`** timed runs on a 1M input (one warmup).
- **Why it can't be gamed:** the fuzz inputs are generated *inside the frozen evaluator*
  with an unseen seed and random shapes, so hardcoding outputs is structurally
  impossible. A wrong candidate scores `0.0` no matter how fast. The number is produced
  by the harness's own timer measuring real code — the LLM only emits code, it never
  reports the speedup.

### The mutation operator (`mutate.py`)
- `propose(champion_src, history, temperature)` → `litellm.completion(...)` with
  system=`program.md`, user=champion code + journal, asking for ONE faster-but-identical
  rewrite. Extracts the ```python block.
- `MODEL = os.environ.get("OPT_MODEL", "gpt-5.5")`. Config via `.env` (`load_dotenv(override=True)`).
- **Temperature handling:** `_supports_temperature()` detects reasoning models
  (`gpt-5.x`, `o`-series) which only allow the default `temperature=1` — for those it
  omits the arg; a try/except fallback retries without temperature as a safety net.

### The keep-best logic (selection rule)
1. A candidate must be **correct** (pass all 20 fuzz checks) to be eligible; wrong → speedup 0, excluded.
2. Among correct candidates in a round, take the **max speedup**.
3. Replace the champion **only if strictly greater** than the current `best` (starts at
   `1.0` = reference baseline). Ties/slower/wrong → champion unchanged.
4. Champion persists across rounds and seeds the next round's prompt → `best` is
   monotonically non-decreasing (the ratchet).

Known caveat: `speedup` is a timing (noisy); `best` is the value measured when the
champion won, and `>` is strict, so jitter can cause harmless churn. Optional hardening:
require `> best * 1.02`, or re-benchmark champion+challenger back-to-back before swapping.

### The swarm (`app.py`, Modal)
- Image: `debian_slim().uv_pip_install("numpy","litellm","python-dotenv")` + the frozen
  files & `program.md` baked to `/root`. Key via `modal.Secret.from_dotenv()`.
- Worker `try_one(champion_src, history, temperature)` does the **whole attempt in the
  container**: LLM rewrite + score. Returns `{passed, speedup, error, src, model, temperature}`.
- `evolve()` (local entrypoint): per round, build N temperature-spread args →
  `try_one.starmap(args, return_exceptions=True)` → keep best correct survivor → feed
  champion into next round. `N_PER_ROUND`/`MAX_ROUNDS` env-overridable.
- Diversity caveat: gpt-5.x ignores the temperature spread (locked to 1) → diversity
  comes from sampling variance; use gpt-4.1/Claude for true temperature spread, or
  per-worker prompt nudges.

### Debug logging (`debuglog.py`)
- Per attempt → console verdict line (`[r1 #3] gpt-5.5 temp=0.50 -> PASS 712.3x`) + the
  unified **diff champion→candidate** (exactly what the LLM changed), and a JSON record
  to `debug.jsonl` (`round, idx, model, temperature, passed, speedup, error, diff, src`).
- Swarm workers also stream per-container lines via Modal stdout.

---

## 4. How we built it (sequence + key decisions)

1. **Prereqs** — `uv` project; `uv add modal numpy` (later `litellm python-dotenv`);
   confirmed Modal auth (profile `rshiramss1`) already set.
2. **Frozen metric** — `target.py` + `evaluate.py`. Verified: vectorized → ~500x passes;
   `broken` (drops +1) → gated to 0.0 (`differential mismatch`).
3. **Mutation + loop** — first built fixed-strategy placeholder, then (per direction)
   **rewrote to a real LLM-in-the-loop** like Karpathy's repo: `mutate.propose()` edits
   the code each round. `program.md` holds the instructions.
4. **Config** — switched `export` → gitignored `.env` + `python-dotenv`
   (`load_dotenv(override=True)` so `.env` beats stale shell vars);
   `Secret.from_dotenv()` for the swarm. Default model set to **gpt-5.5** (newest
   frontier, May 2026; `gpt-5.3-codex` is the coding-specialized option).
5. **Restructure** — moved everything into an `optimization/` subdir; made paths
   `__file__`-relative so cwd doesn't matter.
6. **Temperature fix** — gpt-5.5 rejects `temperature != 1`; added detection + fallback
   so the swarm runs clean (no 400 error banners, no wasted calls).
7. **Debug logging** — `debuglog.py` wired into `loop.py` and `app.py`.
8. **Evidence** — `verify_evidence.py` re-measures independently with negative controls.

---

## 5. Proof it works (live results)

- **Local loop (`loop.py`, gpt-5.5, 10 rounds):** champion ratcheted **446 → 530.70×**.
  Journal showed `keep` (real improvements), `discard` (didn't beat champion), and a
  `crash` (round 10: LLM produced fast-but-wrong code → `differential mismatch` → 0.0).
  Winning `champion.py`:
  ```python
  import numpy as np
  def solve(x):
      out = np.empty_like(x)
      np.square(x, out=out)
      out += 1.0
      return out
  ```
- **Modal swarm (`app.py`, 3-worker smoke):** `round 1: 3/3 correct, champion=706.71x`,
  clean output after the temperature fix. Auth (`.env` → `from_dotenv` → containers) works.
- **Why ~500–700× is real, not hallucinated:** the reference is a naive Python loop over
  1e6 floats (~hundreds of ms); vectorized numpy is sub-ms. The ratio is measured by the
  frozen harness's own timer. Negative controls confirm: reference-vs-itself ≈ 1×
  (harness doesn't inflate), and any wrong/hardcoded candidate is gated to 0×.

---

## 6. What's left to achieve

- **§8 — the real target: Python → Rust port.** Swap the mutable artifact to `src/lib.rs`
  (PyO3), point `mutate.propose()`'s prompt at Rust, and make `try_one` write `lib.rs` →
  `maturin build --release` → import → differential gate → benchmark. Add the Rust
  toolchain to the image + a `modal.Volume` cargo cache. Target: Whisper mel-filterbank
  projection (`filters @ magnitudes` + clamp/log/normalize); Mandelbrot as watchable fallback.
- **Anti-cheat upgrade:** swap the inline `np.allclose` fuzz for full **Hypothesis** property tests.
- **Scale the swarm:** N_PER_ROUND→50, MAX_ROUNDS up; stream with `order_outputs=False`.
- **Raindrop:** tag each attempt `variant_id=(model, prompt_strategy)`; log
  `compile_failure`/`correctness_failure`/`speedup_value`; use Experiments to A/B operators live.
- **UI:** FastAPI + websockets 4-pane dashboard (candidate grid, speedup curve, champion
  diff, autoresearch journal) for the live demo.
- **OpenEvolve:** optionally adopt its MAP-Elites archive / island model / artifacts
  side-channel (feeds compiler errors back into the next prompt).

Full plan + hour-by-hour in `build.md` and `RustForge_Project_Breakdown.md`; the buildable
skeleton + run commands in `optimization.md`.

---

## 7. Key gotchas / learnings

- **gpt-5.x only allows `temperature=1`** (reasoning family). Don't send a custom
  temperature; it 400s. The swarm's temperature-spread diversity is a no-op there.
- **`.env` vs shell:** `load_dotenv(override=True)` so a file `.env` wins over a stale
  placeholder shell var. `Secret.from_dotenv()` ships the same `.env` to containers.
- **`loop.py` is LOCAL** (no Modal containers, by design). Containers only appear with
  `app.py::evolve`, and are ephemeral (scale to zero) — check the printed run URL.
- **Run from the worktree root** with `uv run` (not bare `python3` — anaconda lacks the deps).
- **Speedup is measured, not claimed** — the LLM emits code; the frozen evaluator times
  it. That separation is the anti-hallucination guarantee.
- **Secret hygiene:** never paste API keys in chat; put them straight in the gitignored `.env`.
