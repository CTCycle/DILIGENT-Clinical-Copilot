# Python Guidelines (DILIGENT Backend)

Scope: `DILIGENT/server`, backend scripts, and Python tests.

Also mandatory: apply `assets/docs/ERROR_HANDLING.md`.

## 1. Runtime and environment

- Target Python version is `>=3.14` (see `pyproject.toml`).
- Use the project virtual environment at `runtimes/.venv` when available.
- Keep dependencies aligned with lockfile/runtime workflows (`uv`, `runtimes/uv.lock`).

## 2. Typing and contracts

- Type annotate public functions, API boundary models, and non-trivial internal helpers.
- Prefer built-in generics (`list[str]`, `dict[str, Any]`) and `|` unions.
- Use `collections.abc` for abstract container/callable types.
- Treat static typing as mandatory quality control, not a test substitute.

## 3. Module and architecture conventions

- API layer: `DILIGENT/server/api/*` for HTTP handlers and boundary mapping.
- Domain schemas: `DILIGENT/server/domain/*` for request/response and typed payload contracts.
- Services: `DILIGENT/server/services/*` for business logic and orchestration.
- Repositories: `DILIGENT/server/repositories/*` for persistence and storage access.
- Keep business logic out of route handlers whenever possible.

## 4. FastAPI conventions

- Define endpoints in router modules and register them in `DILIGENT/server/app.py`.
- Validate inputs with Pydantic/domain models; avoid ad-hoc manual validation.
- Use explicit HTTP status codes and stable response models.
- Preserve safe error boundaries and request/job correlation behavior.

## 5. Async, jobs, and long-running work

- Use async only with non-blocking dependencies.
- Do not block async handlers with CPU-heavy tasks.
- For long operations, use the job system in `services/jobs.py` and expose start/poll/cancel APIs.
- Ensure cooperative cancellation checks in job runners.

## 6. Style and maintainability

- Keep functions focused and side effects explicit.
- Prefer simple, composable logic over deep abstraction.
- Write comments only for non-obvious constraints or safety behavior.
- Follow existing code style in nearby modules; avoid broad stylistic rewrites.

## 7. Validation checklist

- Lint/format: Ruff (or project-standard formatter/linter commands).
- Type checks: mypy where configured.
- Tests: pytest (`tests/unit` and relevant `tests/e2e` flows when impacted).
