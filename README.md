# RustForge

**A swarm of AI agents that automatically rewrites slow Python into provably-correct, faster Rust — and can never cheat to get there.**

RustForge applies Andrej Karpathy's autoresearch *keep/discard loop* to Python→Rust porting. You point it at a repository; it finds the functions worth optimizing, rewrites them in Rust with a swarm of agents running in parallel on [Modal](https://modal.com), and gates every single rewrite behind a frozen evaluator the agents are not allowed to touch. A port is only kept if it is **correct** and **regresses on nothing**. Quality only ever ratchets up — never down.

---

## Why it can't cheat

The obvious failure mode for an "AI that optimizes code" is an agent that wins by hardcoding the expected outputs. RustForge makes that **structurally impossible**:

- The **correctness oracle is always the original Python function**, executed at evaluation time to produce the ground-truth outputs.
- A **frozen evaluator** (which the agent cannot read or edit) fuzzes each Rust port against that oracle on **hundreds of new, random inputs the agent never sees**, asserting `np.allclose`.
- Fitness is a **hard gate**, not a weighted term: any correctness failure forces the score to `0`, so a wrong port can never be crowned.

Because the inputs are generated *after* the code is written and are never shown to the model, there is nothing to memorize.

---

## How it works

```
Repo URL ──► Swarm A (rank) ──► Swarm B (rewrite + evolve) ──► verified faster Rust
```

### 1. Swarm A — find what's worth porting (`analyze.py`)
Clones the target repo, AST-extracts every top-level function, and fans out **one Modal worker per function**. Each worker scores its function on optimization potential using **objective AST features only — no LLM calls**, so the ranking is fast, free, and bias-free:

```
score = purity·2 + numeric_density + loop_score + determinism·2 − dependency_penalty
```

Results stream back in completion order and are ranked. The top candidate seeds Swarm B.

### 2. Swarm B — evolve a Rust port (`run_stage2.py`, `try_one_port.py`)
For the chosen function, each round fans out a wide swarm of Modal containers. Every worker runs the full cycle:

1. **Mutate** — an LLM (via `litellm`) rewrites the current champion's Rust.
2. **Build** — `maturin develop --release` compiles a PyO3 extension.
3. **Verify** — pytest + Hypothesis differential gate against the Python oracle.
4. **Benchmark** — median-of-many timing vs. the Python baseline.

The outer loop keeps the fastest *correct* survivor as the new champion and discards the rest — Karpathy's ratchet, parallel within a round and sequential across rounds.

### 3. Adaptive regression gate (`evaluate.py`, `bench.py`)
The champion is **"the fastest port that regresses on nothing,"** not the biggest number at one input size. Every run is benchmarked across **multiple held-out input sizes**, so a win at one size can't hide a slowdown at another. This is what makes the speedups trustworthy rather than cherry-picked.

### Target-agnostic by design (`target_spec.py`)
A `TargetSpec` bundles a function's source, oracle, and input generator behind a uniform interface, so the same pipeline runs on any function from any repo — not a hardwired demo.

---

## Quickstart

```bash
# 1. Install deps (Python ≥ 3.12)
uv sync

# 2. Configure credentials
cp .env.example .env        # set OPENAI_API_KEY and OPT_MODEL

# 3. Rank the functions in a repo (Swarm A)
python analyze.py https://github.com/karpathy/micrograd
#   or run it on Modal:
modal run analyze.py --filepath https://github.com/karpathy/micrograd

# 4. Evolve a Rust port of a top candidate (Swarm B)
python run_stage2.py --swarm-a result.json
```

---

## Stack

| Layer | Tech |
|---|---|
| Orchestration / compute | **Modal** (one container per function / per rewrite) |
| Mutation LLM | **litellm** (`OPT_MODEL`, default `gpt-5.5`) |
| Rust ↔ Python bridge | **PyO3 0.28.3** + **rust-numpy 0.28.0**, built with **maturin** |
| Correctness gate | **pytest** + **Hypothesis** differential property testing |
| Observability | **Raindrop** — live A/B of mutation operators |
| Toolchain | Rust ≥ 1.83, Python ≥ 3.12 |

---

## Repository layout

| Path | Responsibility |
|---|---|
| `analyze.py` | Swarm A — repo parse + per-function AST scoring |
| `run_stage2.py` | Stage-2 end-to-end driver (TargetSpec → port → gate) |
| `try_one_port.py` | One mutate → build → verify → benchmark cycle (the swarm worker) |
| `verify.py` | pytest + Hypothesis differential correctness gate |
| `bench.py` | Timing harness (median-of-N, stability) |
| `target_spec.py` | Target-agnostic function/oracle/input bundling |
| `rust_template/` | Seed PyO3/maturin Cargo project |
| `targets/` | Pre-loaded optimization targets |
| `ui/` | Live dashboard |
| `stress/` | Stress-test suites for the gates |
| `KNOWLEDGEBASE.md`, `docs/` | Architecture and build notes |

---

*Built at the Autoresearch Systems Hackathon.*
