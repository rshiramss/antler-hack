# External Integrations

## Core Sections (Required)

### 1) Integration Inventory

| System | Type (API/DB/Queue/etc) | Purpose | Auth model | Criticality | Evidence |
|--------|---------------------------|---------|------------|-------------|----------|
| Modal | Cloud compute/workers | Run parallel candidate build/verify/benchmark attempts. | [TODO] likely environment-based Modal auth. | High | cntxt RustForge KB |
| OpenAI / LiteLLM | LLM API wrapper | Generate mutated Rust candidate implementations. | [TODO] likely API key env vars. | High | cntxt RustForge KB |
| Raindrop | Experiment/observability API | A/B mutation operators and log signals. | [TODO] SDK/API auth not yet specified. | Medium | cntxt RustForge KB |
| FastAPI websocket clients | Local/browser UI transport | Stream live dashboard events. | [TODO] no auth specified. | Medium | cntxt RustForge KB |
| GitHub/repo source | Repository input | Intended user-facing repo ingestion flow; demo may use preloaded targets. | [TODO] public clone or token-based auth. | Low for MVP | cntxt RustForge KB |

### 2) Data Stores

| Store | Role | Access layer | Key risk | Evidence |
|-------|------|--------------|----------|----------|
| Modal Volume | Intended shared build/cache workspace. | `app.py`/workers, not implemented yet. | Cache invalidation and compile artifact consistency. | cntxt RustForge KB |
| Local journal file/store | Intended experiment log and UI event source. | `journal.py`, not implemented yet. | Losing source of truth if only UI tracks state. | cntxt RustForge KB |
| [TODO] no database | No persistent DB appears in the current plan or workspace. | [TODO] | Future product may need state persistence. | Phase 1 scan output |

### 3) Secrets and Credentials Handling

- Credential sources: [TODO] no `.env.example`, env template, or config exists.
- Hardcoding checks: current workspace has no source files to inspect; no hardcoded secrets found by scan because there is no implementation.
- Rotation or lifecycle notes: [TODO] not documented.

### 4) Reliability and Failure Behavior

- Retry/backoff behavior: [TODO] no implementation exists.
- Timeout policy: [TODO] no implementation exists.
- Circuit-breaker or fallback behavior: intended demo fallback is Mandelbrot if micrograd integration slips; Whisper mel has been dropped because BLAS-backed code is not speedup-friendly. Operational fallback is local multiprocessing if Modal concurrency/cold starts block the demo.

### 5) Observability for Integrations

- Logging around external calls: intended `journal.py` and `raindrop_hooks.py`, not implemented.
- Metrics/tracing coverage: intended Raindrop experiment telemetry and UI speedup curve, not implemented.
- Missing visibility gaps: exact worker-to-websocket transport and Raindrop SDK/API surface are unresolved.

### 6) Evidence

- cntxt knowledge base: `RustForge — Hackathon Build (Pivot)`
- Phase 1 scan output: no env templates, source integrations, monitoring config, or containers detected.
- Workspace path scanned: `/Users/abrahambhatti/Desktop/RustForge`
