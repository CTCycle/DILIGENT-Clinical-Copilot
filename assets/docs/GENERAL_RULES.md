# General Rules

This file is the entrypoint for all work in this repository.

## 1. Required doc review

Always read:
- `assets/docs/GENERAL_RULES.md`
- `assets/docs/ERROR_HANDLING.md`

Read as needed by task:
- `assets/docs/ARCHITECTURE.md`: backend/frontend structure, API surface, data flow.
- `assets/docs/BACKGROUND_JOBS.md`: job lifecycle, polling, cancellation.
- `assets/docs/GUIDELINES_PYTHON.md`: Python standards and backend conventions.
- `assets/docs/GUIDELINES_TYPESCRIPT.md`: frontend standards for `DILIGENT/client`.
- `assets/docs/GUIDELINES_TESTS.md`: test layout and execution.
- `assets/docs/PACKAGING_AND_RUNTIME_MODES.md`: local/cloud/desktop runtime behavior.
- `assets/docs/README_WRITING.md`: README format and required sections.
- `assets/docs/UI_STANDARDS.md`: frontend design system rules.
- `assets/docs/UI_AUDIT_REPORT.md`: current UI debt and known gaps.

## 2. Documentation responsibilities

- If you change behavior, architecture, runtime contracts, or test strategy, update the corresponding docs in `assets/docs`.
- Keep cross-references coherent: endpoint names, path names, and runtime/version statements must not conflict across files.
- Prefer concise, stable documentation over exhaustive file inventories that quickly become stale.

## 3. Cross-language engineering principles

- Favor clear naming, low coupling, and small focused units.
- Use explicit contracts at boundaries (API schema, typed payloads, structured errors).
- Prioritize deterministic behavior and testability.
- Keep security defaults strict: input validation, least privilege, and no secret leakage.

## 4. Execution rules

- Use PowerShell by default in this repository.
- Use `cmd /c` only for `.bat` scripts or CMD-specific syntax.
- In docs, always reference the project virtual environment as `runtimes/.venv`.
- Bundled frontend runtime is `runtimes/nodejs`.
- For changes under `DILIGENT/client`, run a validation build before finalizing:
  - `npm run build`
  - If `npm` is unavailable in PATH, run:
    - `runtimes/nodejs/node.exe runtimes/nodejs/node_modules/npm/bin/npm-cli.js run build`

## 5. Skills and external references

- Use relevant skills when the task matches a reusable workflow.
- Use web search only when it improves factual accuracy (for example evolving external tooling or standards).

