# Coding Conventions

## Core Sections (Required)

### 1) Naming Rules

| Item | Rule | Example | Evidence |
|------|------|---------|----------|
| Files | Intended Python files use snake_case. Actual source files are not present. | `raindrop_hooks.py`, `test_mandelbrot.py` | cntxt RustForge KB |
| Functions/methods | [TODO] no actual source exists to verify naming. | [TODO] | Phase 1 scan output |
| Types/interfaces | [TODO] no actual source exists to verify naming. | [TODO] | Phase 1 scan output |
| Constants/env vars | [TODO] no actual source or env template exists. | [TODO] | Phase 1 scan output |

### 2) Formatting and Linting

- Formatter: [TODO] no formatter config exists.
- Linter: [TODO] no linter config exists.
- Most relevant enforced rules: [TODO] none are configured in the workspace.
- Run commands: [TODO] no manifest or Makefile exists.

### 3) Import and Module Conventions

- Import grouping/order: [TODO] no source files exist.
- Alias vs relative import policy: [TODO] no source files or config exist.
- Public exports/barrel policy: [TODO] not applicable until package structure exists.

### 4) Error and Logging Conventions

- Error strategy by layer: intended candidate failures should return structured status such as compile/test/benchmark failure into the journal, but no actual schema exists yet.
- Logging style and required context fields: intended journal shape is `id`, `speedup`, `status`, and `description`; exact event schema is [TODO].
- Sensitive-data redaction rules: [TODO] no implementation or config exists for OpenAI/Modal/Raindrop credentials.

### 5) Testing Conventions

- Test file naming/location rule: intended target tests are named `test_*.py` under each target directory.
- Mocking strategy norm: [TODO] no source or tests exist.
- Coverage expectation: [TODO] no coverage config exists.

### 6) Evidence

- cntxt knowledge base: `RustForge — Hackathon Build (Pivot)`
- Phase 1 scan output: no source, lint, format, test, or env config detected.
- Workspace path scanned: `/Users/abrahambhatti/Desktop/RustForge`
