# Codebase Concerns

## Core Sections (Required)

### 1) Top Risks (Prioritized)

| Severity | Concern | Evidence | Impact | Suggested action |
|----------|---------|----------|--------|------------------|
| High | The actual workspace has no implementation, manifests, tests, or entry points yet. | Phase 1 scan output | Nothing can run until skeleton and FFI proof are created. | Create intended skeleton and prove `maturin develop` + NumPy round-trip first. |
| High | PyO3/maturin/numpy version triple is the highest integration risk. | cntxt RustForge KB | FFI failure blocks the entire product demo. | Pin `pyo3=0.28.3`, `numpy=0.28.0`, Rust >=1.83, and keep `#[pymodule]` name aligned with `Cargo.toml` `lib.name`. |
| High | Correctness must be impossible to game. | cntxt RustForge KB | Speedup claims lose credibility if candidates can hardcode tests. | Implement original tests plus unseen Hypothesis differential fuzz gate before optimizing swarm polish. |
| Medium | Live worker-to-UI streaming transport is unresolved. | cntxt RustForge KB | Dashboard may lag or fail to show the visible swarm. | Decide between ordered `.map()` collection and out-of-band event channel before UI work. |
| Medium | Raindrop SDK/API details are unresolved. | cntxt RustForge KB | Observability/A-B stage may consume deadline time. | Keep Raindrop non-blocking and wrap integration behind a minimal adapter. |
| Medium | Arbitrary repo ingestion is not demo-critical but can distract. | cntxt RustForge KB | Deadline risk if product scope expands too early. | Ship preloaded micrograd/Mandelbrot path first; frame URL ingestion as product direction unless ahead of schedule. |

### 2) Technical Debt

| Debt item | Why it exists | Where | Risk if ignored | Suggested fix |
|-----------|---------------|-------|-----------------|---------------|
| No project skeleton | Pivot was just decided in cntxt; workspace is empty. | Workspace root | No executable path. | Create `app.py`, `evolve.py`, `verify.py`, `bench.py`, targets, and `rust_template/`. |
| No manifest or env template | No code has been written yet. | Workspace root | Dependency and credential ambiguity. | Add `pyproject.toml`, `rust_template/Cargo.toml`, `rust_template/pyproject.toml`, and `.env.example`. |
| No test harness | Correctness design exists only in KB. | Workspace root | Cannot prove anti-cheat or numerical equivalence. | Implement minimal pytest + Hypothesis differential test for Mandelbrot or NumPy smoke target. |

### 3) Security Concerns

| Risk | OWASP category (if applicable) | Evidence | Current mitigation | Gap |
|------|--------------------------------|----------|--------------------|-----|
| Unspecified API credential handling | N/A | Phase 1 scan output | None yet. | Add `.env.example` and avoid logging Modal/OpenAI/Raindrop secrets. |
| Arbitrary repository ingestion could execute or build untrusted code | N/A | cntxt RustForge KB | Demo scope can use preloaded targets. | If implemented, sandbox builds and avoid running untrusted tests outside isolated workers. |
| Websocket dashboard auth unspecified | A01 if exposed beyond localhost | cntxt RustForge KB | None yet. | Keep local-only for hackathon demo or add basic access control. |

### 4) Performance and Scaling Concerns

| Concern | Evidence | Current symptom | Scaling risk | Suggested improvement |
|---------|----------|-----------------|-------------|-----------------------|
| Rust compile time may dominate loop budget. | cntxt RustForge KB | No implementation yet. | Swarm spends time compiling instead of exploring. | Use shared `target/` cache/Volume and debug builds for correctness before release benchmarking. |
| Speedup can be measurement noise. | cntxt RustForge KB | No benchmark harness yet. | Champion may regress in real use. | Use warmup, median-of-N, and require stable improvement across repeated runs. |
| Modal cold starts/concurrency can hide swarm spectacle. | cntxt RustForge KB | No Modal app yet. | Demo shows waiting instead of parallelism. | Use warm pools/min containers if available; fallback to local multiprocessing if needed. |

### 5) Fragile/High-Churn Areas

| Area | Why fragile | Churn signal | Safe change strategy |
|------|-------------|-------------|----------------------|
| `rust_template/` | Version/name mismatches can break import. | No git churn; intended high-risk area from KB. | Make the first milestone a tiny hand-written NumPy round-trip and keep module names aligned. |
| `verify.py` | Credibility depends on frozen correctness checks. | No git churn; intended core correctness gate. | Treat correctness as a hard gate before benchmarking. |
| `app.py` / Modal workers | Cloud execution, shared caches, and live results interact. | No git churn; intended integration hotspot. | Keep worker return schema simple and add out-of-band events only after local loop works. |

### 6) `[ASK USER]` Questions

1. [ASK USER] Should I build the skeleton exactly from the cntxt layout now, starting with the FFI proof, or only keep documenting/understanding for the moment?
2. [ASK USER] For the first runnable target, should we implement micrograd first as the hero target, or stand up Mandelbrot first as the safer fallback?
3. [ASK USER] Which live-update path do you want for Modal workers: simple `.map()` result collection first, or an out-of-band websocket/event sink from the start?
4. [ASK USER] What are the intended environment variable names for Modal, OpenAI/LiteLLM, and Raindrop credentials?

### 7) Evidence

- cntxt knowledge base: `RustForge — Hackathon Build (Pivot)`
- Phase 1 scan output: no manifests, source files, tests, CI, containers, security config, or git commits detected.
- Workspace path scanned: `/Users/abrahambhatti/Desktop/RustForge`
