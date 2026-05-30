# Testing Patterns

## Core Sections (Required)

### 1) Test Stack and Commands

- Primary test framework: intended `pytest`; actual version [TODO].
- Assertion/mocking tools: intended plain pytest assertions plus NumPy `allclose`; exact libraries [TODO].
- Commands:

```bash
# [TODO] no checked-in commands exist yet
pytest
maturin develop
```

### 2) Test Layout

- Test file placement pattern: intended tests live beside preloaded target oracle files. New KB context says micrograd tests are in `test/test_engine.py` upstream and should verify gradients against PyTorch; Mandelbrot remains fallback.
- Naming convention: intended `test_*.py` for Python target tests.
- Setup files and where they run: [TODO] no fixtures or setup files exist yet.

### 3) Test Scope Matrix

| Scope | Covered? | Typical target | Notes |
|-------|----------|----------------|-------|
| Unit | No actual coverage yet. | [TODO] | Workspace has no tests. |
| Integration | No actual coverage yet. | PyO3/maturin import and NumPy round-trip. | This is the first go/no-go gate from cntxt. |
| E2E | No actual coverage yet. | Full mutate/build/verify/bench loop. | Intended Stage 2/3 milestone. |
| Differential/property | No actual coverage yet. | Python oracle vs Rust port on generated inputs. | Intended anti-cheat moat and hard correctness gate. |

### 4) Mocking and Isolation Strategy

- Main mocking approach: [TODO] no implementation exists.
- Isolation guarantees: intended verification must keep oracle/tests/harness frozen relative to candidate mutation.
- Common failure mode in tests: expected FFI import/build failures from PyO3/maturin/numpy version mismatch until the version triple is proven.

### 5) Coverage and Quality Signals

- Coverage tool + threshold: [TODO] no coverage config exists.
- Current reported coverage: [TODO] no tests exist.
- Known gaps/flaky areas: all testing is currently unimplemented; priority should be PyO3 import smoke test, micrograd target tests, and Hypothesis differential gate.

### 6) Evidence

- cntxt knowledge base: `RustForge — Hackathon Build (Pivot)`
- Phase 1 scan output: no test files, test config, CI, or performance configs detected.
- Workspace path scanned: `/Users/abrahambhatti/Desktop/RustForge`
