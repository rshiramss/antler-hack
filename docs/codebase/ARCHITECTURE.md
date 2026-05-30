# Architecture

## Core Sections (Required)

### 1) Architectural Style

- Primary style: intended agentic/evolutionary pipeline with cloud fan-out workers and a sequential keep/discard ratchet.
- Why this classification: the KB maps RustForge to Karpathy's autoresearch loop: mutate one file, run a frozen evaluation harness, keep the best correct candidate, journal results, and repeat.
- Primary constraints: correctness is a hard gate; FFI build compatibility must be proven first; swarm parallelism is within a round while champion evolution stays sequential across rounds.

### 2) System Flow

```text
user target -> function analysis -> candidate mutation -> Rust build -> correctness gate -> benchmark -> champion/journal -> live UI/output
```

1. User supplies a GitHub URL or picks a preloaded target; demo scope now uses micrograd as the hero target, Mandelbrot as fallback, and not Whisper mel.
2. Swarm A scores candidate Python functions for optimization potential.
3. User selects a function; the system extracts the Python oracle, tests, and signature, then seeds a Rust candidate.
4. Swarm B runs many candidate attempts per round: mutate Rust, build with maturin/Cargo, verify with tests plus differential fuzzing, and benchmark.
5. The evolution loop keeps the fastest correct candidate as champion and records keep/discard/crash results.
6. The dashboard streams worker state, tests, journal events, and speedup progression.

### 3) Layer/Module Responsibilities

| Layer or module | Owns | Must not own | Evidence |
|-----------------|------|--------------|----------|
| Repo/function analysis | Ranking functions worth porting. | Rust build, correctness, or champion policy. | cntxt RustForge KB |
| Mutation generation | Producing candidate Rust source. | Declaring a candidate correct or fast. | cntxt RustForge KB |
| FFI build template | PyO3/maturin/Cargo shape for compiled module. | Swarm orchestration. | cntxt RustForge KB |
| Verification | Original tests and unseen Hypothesis differential inputs. | Speed-only acceptance. | cntxt RustForge KB |
| Benchmarking | Stable Python-vs-Rust timing. | Correctness bypasses or prompt generation. | cntxt RustForge KB |
| Evolution/champion state | Keep/discard ratchet and journal. | Provider-specific integrations. | cntxt RustForge KB |
| UI/observability | Human-visible state and demo narrative. | Source of truth for correctness. | cntxt RustForge KB |

### 4) Reused Patterns

| Pattern | Where found | Why it exists |
|---------|-------------|---------------|
| Frozen evaluation harness | Intended `verify.py` + `bench.py` | Prevents the LLM from modifying the scoring rules. |
| Correctness-gated metric | Intended evolution loop | Speedup is set to zero unless correctness passes. |
| Champion ratchet | Intended `evolve.py` + `journal.py` | Preserves only the fastest correct candidate between rounds. |
| Parallel fan-out, sequential rounds | Intended Modal swarm | Makes the demo visibly parallel while keeping the autoresearch ratchet legible. |
| Preloaded hero/fallback targets | Intended `targets/mandelbrot/` | Keeps the demo focused on pure-Python hot loops where Rust speedups are visible; Mandelbrot protects demo viability if micrograd integration slips. |

### 5) Known Architectural Risks

- The actual workspace has no implementation yet, so architecture is currently intent from cntxt, not executable code.
- PyO3/maturin/numpy compatibility is the first hard risk; if this fails, the swarm/UI work should not proceed.
- Arbitrary repo ingestion is explicitly not part of the demo-critical path; treating it as required would endanger the deadline.
- Worker-to-websocket live updates are unresolved; `.map()`/`.starmap()` result collection may not provide enough live streaming without an out-of-band channel.

### 6) Evidence

- cntxt knowledge base: `RustForge — Hackathon Build (Pivot)`
- Phase 1 scan output: no actual source files, entry points, manifests, tests, or CI exist yet.
- Workspace path scanned: `/Users/abrahambhatti/Desktop/RustForge`
