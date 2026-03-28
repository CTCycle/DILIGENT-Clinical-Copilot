# README Writing Guidelines

Use this structure for repository README updates.

Goal: keep README user-facing, accurate, and consistent with `assets/docs/*`.

## 1. Scope and tone

- Write for users/operators, not internal code contributors.
- Explain capabilities and workflows, not class/function internals.
- Keep claims factual and verifiable from current behavior.

## 2. Required section order

Use this section sequence (skip only when truly not applicable):
1. Project Overview
2. Runtime Model (local/cloud/desktop if applicable)
3. Installation
4. How to Use
5. Configuration
6. Setup and Maintenance
7. Resources (if relevant)
8. License

If you skip a section, renumber remaining sections.

## 3. Content requirements by section

### 3.1 Project Overview
- Purpose and problem solved.
- High-level system shape (backend/frontend/desktop interactions).

### 3.2 Runtime Model
- Identify active config file (`DILIGENT/settings/.env`).
- Explain mode switching using `.env.*.example` profiles.
- Link `assets/docs/PACKAGING_AND_RUNTIME_MODES.md` for details.

### 3.3 Installation
- Provide concise, reproducible steps.
- Include Windows launcher flow when available.
- For manual setup, separate backend and frontend steps.
- Reference virtual environment path as `runtimes/.venv`.

### 3.4 How to Use
- Describe operational user workflow.
- Include URLs/entrypoints for UI and API.
- Include screenshots from `assets/figures` with short captions.

### 3.5 Configuration
- List config location and loading model.
- Include a variable table.
- Do not expose secrets.

### 3.6 Setup and Maintenance
- List available maintenance scripts/actions and outcomes.
- Describe what actions do, not script internals.

### 3.7 Resources
- Summarize resource directories and purpose (for example models, sources, logs).

### 3.8 License
- State license clearly and point to `LICENSE`.

## 4. Consistency checks before finalizing

- Runtime/version values match repository truth (`pyproject.toml`, package manifests, Dockerfiles).
- Endpoint names and commands match current behavior.
- Section links and screenshot paths resolve.
- No contradictions with `assets/docs/GENERAL_RULES.md` or `assets/docs/ARCHITECTURE.md`.
