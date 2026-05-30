# Codebase Structure

## Core Sections (Required)

### 1) Top-Level Map

| Path | Purpose | Evidence |
|------|---------|----------|
| `docs/codebase/` | Generated project-understanding docs for this onboarding pass. | Phase 1 scan output |
| `app.py` | [INTENDED] Modal app, image, volume, worker functions. Not present yet. | cntxt RustForge KB |
| `evolve.py` | [INTENDED] Outer keep/discard loop and champion management. Not present yet. | cntxt RustForge KB |
| `mutate.py` | [INTENDED] LLM mutation calls and prompt construction. Not present yet. | cntxt RustForge KB |
| `verify.py` | [INTENDED] pytest plus Hypothesis differential correctness gate. Not present yet. | cntxt RustForge KB |
| `bench.py` | [INTENDED] Timing harness with median-of-N and stability checks. Not present yet. | cntxt RustForge KB |
| `analyze.py` | [INTENDED] Swarm A function-scoring worker. Not present yet. | cntxt RustForge KB |
| `journal.py` | [INTENDED] Autoresearch-style journal and event emission. Not present yet. | cntxt RustForge KB |
| `raindrop_hooks.py` | [INTENDED] Variant tagging and signal logging. Not present yet. | cntxt RustForge KB |
| `ui/server.py` | [INTENDED] FastAPI websocket server. Not present yet. | cntxt RustForge KB |
| `ui/index.html` | [INTENDED] Vanilla JS live dashboard. Not present yet. | cntxt RustForge KB |
| `targets/micrograd/` | [INTENDED] primary hero target based on `karpathy/micrograd`; tests are expected from `test/test_engine.py` and check gradients against PyTorch. Not present yet. | cntxt RustForge KB |
| `targets/mandelbrot/` | [INTENDED] fallback target with visual/high-speedup demo. Not present yet. | cntxt RustForge KB |
| `rust_template/` | [INTENDED] Seed Cargo/maturin project that the LLM edits. Not present yet. | cntxt RustForge KB |

### 2) Entry Points

- Main runtime entry: [TODO] no actual entry file exists; intended main entry is `app.py`.
- Secondary entry points: intended `ui/server.py`, Modal workers in `app.py`, and local scripts such as `evolve.py`.
- How entry is selected: [TODO] no manifest, CLI, or run script exists yet.

### 3) Module Boundaries

| Boundary | What belongs here | What must not be here |
|----------|-------------------|------------------------|
| Modal orchestration (`app.py`) | Images, volumes, worker definitions, cloud execution wiring. | Verification semantics or benchmark scoring rules. |
| Evolution loop (`evolve.py`) | Round orchestration, champion replacement, keep/discard policy. | LLM provider-specific prompt/API glue. |
| Mutation (`mutate.py`) | Prompting and candidate Rust generation. | Correctness decisions or benchmark timing. |
| Verification (`verify.py`) | Original tests plus differential fuzz gate. | Performance ranking that bypasses correctness. |
| Benchmarking (`bench.py`) | Warmup, median-of-N timing, stability checks. | Generating or mutating code. |
| Targets (`targets/*`) | Python oracle functions, tests, fixtures. | Infrastructure, LLM prompts, dashboard code. |
| Rust template (`rust_template/`) | Minimal PyO3/maturin/Cargo project and mutable `src/lib.rs`. | Python orchestration logic. |
| UI (`ui/*`) | Live dashboard and websocket presentation. | Core correctness or champion state as source of truth. |

### 4) Naming and Organization Rules

- File naming pattern: intended Python files use snake_case (`raindrop_hooks.py`, `test_mel.py`); actual source files are [TODO] because none exist.
- Directory organization pattern: intended layout is layer/component oriented around orchestration, target code, UI, and Rust template.
- Import aliasing or path conventions: [TODO] no actual imports or config exist yet.

### 5) Evidence

- cntxt knowledge base: `RustForge — Hackathon Build (Pivot)`
- Phase 1 scan output: actual tree only contained `docs/codebase/.codebase-scan.txt` at scan time.
- Workspace path scanned: `/Users/abrahambhatti/Desktop/RustForge`
