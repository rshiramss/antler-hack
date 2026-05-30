# Technology Stack

## Core Sections (Required)

### 1) Runtime Summary

| Area | Value | Evidence |
|------|-------|----------|
| Primary language | Intended: Python + Rust. Actual workspace: no source files yet. | cntxt RustForge KB; Phase 1 scan output |
| Runtime + version | Intended: Python >=3.10 and Rust >=1.83. Actual workspace: no runtime config present. | cntxt RustForge KB; Phase 1 scan output |
| Package manager | Intended: pip/maturin/Cargo. Actual workspace: no manifest present. | cntxt RustForge KB; Phase 1 scan output |
| Module/build system | Intended: PyO3 + rust-numpy + maturin native-extension build. Actual workspace: not created yet. | cntxt RustForge KB; Phase 1 scan output |

### 2) Production Frameworks and Dependencies

| Dependency | Version | Role in system | Evidence |
|------------|---------|----------------|----------|
| `pyo3` | 0.28.3 | Python/Rust binding layer for generated native module. | cntxt RustForge KB |
| `numpy` Rust crate (`rust-numpy`) | 0.28.0 | NumPy array interop for PyO3 functions. | cntxt RustForge KB |
| Rust toolchain | >=1.83 | Required by the intended PyO3 version. | cntxt RustForge KB |
| `maturin` | >=1.0,<2.0 | Build/develop workflow for Python extension modules. | cntxt RustForge KB |
| Modal | [TODO] actual version | Intended cloud worker platform for parallel rewrite/compile/verify/benchmark attempts. | cntxt RustForge KB |
| OpenAI / LLM provider via LiteLLM | [TODO] actual version | Intended mutation generation layer for Rust candidate code. | cntxt RustForge KB |
| FastAPI + websockets | [TODO] actual versions | Intended live dashboard server. | cntxt RustForge KB |
| Hypothesis + pytest | [TODO] actual versions | Intended correctness gate: original tests plus differential fuzz tests. | cntxt RustForge KB |
| Raindrop | [TODO] actual version/API | Intended experiment tagging and live A/B of mutation operators. | cntxt RustForge KB |

### 3) Development Toolchain

| Tool | Purpose | Evidence |
|------|---------|----------|
| `maturin` | Build and install the Rust extension into Python. | cntxt RustForge KB |
| `cargo` | Compile the Rust candidate module. | cntxt RustForge KB |
| `pytest` | Run original target tests and verification harness. | cntxt RustForge KB |
| `hypothesis` | Generate unseen differential inputs at evaluation time. | cntxt RustForge KB |
| [TODO] formatter/linter | No formatter or linter config exists yet. | Phase 1 scan output |

### 4) Key Commands

```bash
# [TODO] no checked-in install/build/test commands exist yet
maturin develop
pytest
cargo --version
maturin --version
```

### 5) Environment and Config

- Config sources: [TODO] no `.env.example`, manifest, or config file exists in the workspace.
- Required env vars: likely Modal/OpenAI/Raindrop credentials, but exact names are [TODO] until code or env templates exist.
- Deployment/runtime constraints: first integration risk is the PyO3/maturin/numpy version triple; target choice now favors pure-Python hot loops via micrograd because BLAS-backed Whisper mel would not show enough Rust speedup.

### 6) Evidence

- cntxt knowledge base: `RustForge — Hackathon Build (Pivot)`
- Phase 1 scan output: no manifest files, no entry points, no lint/test/CI/container/security config.
- Workspace path scanned: `/Users/abrahambhatti/Desktop/RustForge`
