# Coding Rules

Last updated: 2026-04-24

## 1. Shared Rules (all languages)

- Keep changes tightly scoped to the task.
- Prefer clear contracts at boundaries over implicit behavior.
- Keep side effects explicit and localized.
- Avoid large stylistic rewrites unrelated to the change.
- Keep imports at the top of files.
- Remove dead code and obsolete paths when touched.
- Use project-level error-handling rules from `ERROR_HANDLING.md`.

## 2. Python Rules (mandatory baseline)

### Runtime and tooling

- Target Python version: `>=3.14` (`pyproject.toml`).
- Use `runtimes/.venv` when present.
- Keep dependency resolution aligned with `uv` and `runtimes/uv.lock`.
- Preferred quality tools:
  - Ruff for lint/format
  - Pylance for type checking
  - pytest for tests (`tests/unit` and relevant `tests/e2e`)

### Typing

- Type annotations are required for:
  - public APIs
  - non-trivial internal logic
- Use built-in generics (`list[str]`, `dict[str, Any]`).
- Prefer `|` for unions.
- Use `collections.abc` for abstract types.
- Treat typing as a required quality standard, not optional documentation.

### Validation and API design

- Validate request/response data with Pydantic/domain models.
- Avoid manual ad-hoc validation when schema models can express constraints.
- Use explicit HTTP status codes.
- Keep response models consistent and stable.
- Ensure safe error handling and request/job traceability.

### Async and job execution

- Use async only with non-blocking dependencies.
- Do not run CPU-heavy work directly in async handlers.
- Use existing job system (`server/services/jobs.py`) for long-running work.
- For long-running operations, provide:
  - start endpoint
  - poll/status endpoint
  - cancel endpoint

### Code structure

- Keep functions small and focused.
- Prefer composable logic over deeply nested branching.
- Avoid nested functions unless strictly necessary.
- Use classes to group cohesive behavior where appropriate.
- Keep modules under approximately 1000 LOC when practical.
- Comment only where needed for clarity/safety.

## 3. TypeScript Rules (Angular client)

### Type safety and contracts

- Keep strict typing; avoid `any` for untrusted inputs.
- Centralize shared API contracts in `client/src/app/core/models/types.ts`.
- Normalize backend payloads before rendering.

### Frontend architecture

- Keep HTTP transport/error normalization in `core/services/http-api.ts`.
- Keep domain API calls in `core/services/*-api.ts`.
- Keep page orchestration in `pages/*`.
- Keep reusable UI controls in `components/*`.
- Keep shared app state in `core/state/app-state.service.ts`.

### Interaction and UX behavior

- Preserve deterministic job state transitions:
  - `pending`, `running`, `completed`, `failed`, `cancelled`
- Disable conflicting actions during active operations.
- Preserve keyboard accessibility and ARIA semantics.

### Tooling and verification

- Use project scripts in `DILIGENT/client/package.json`.
- Run build validation when frontend code changes:
  - `npm run build` (from `DILIGENT/client`)

## 4. Testing Standards

- Unit tests for logic changes in backend/services/repositories.
- E2E coverage for user-visible/API workflow changes.
- Use deterministic assertions and explicit skip conditions for unavailable external dependencies.
