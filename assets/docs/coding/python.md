# Python Rules
Last updated: 2026-08-02

## Runtime And Tooling
- Target Python version: `>=3.14` from `pyproject.toml`.
- Use `app/server/.venv` when available, otherwise `runtimes/.venv`.
- Keep dependency resolution aligned with `uv` and `app/server/uv.lock`.
- Preferred quality tools:
  - Ruff for lint and format
  - Pylance-compatible typing
  - pytest for `tests/unit` and relevant `tests/e2e`

## Typing
- Type annotations are required for public APIs and non-trivial internal logic.
- Use built-in generics such as `list[str]` and `dict[str, Any]`.
- Prefer `|` for unions.
- Use `collections.abc` for abstract types.
- Treat typing as a required quality standard.

## Validation And API Design
- Validate request and response data with Pydantic or domain models.
- Avoid ad-hoc manual validation when schema models can express the constraints.
- Use explicit HTTP status codes.
- Keep response models stable and consistent.
- Preserve safe error handling and request or job traceability.

## Async And Job Execution
- Use async only with non-blocking dependencies.
- Do not run CPU-heavy work directly in async handlers.
- Use the existing job system in `app/server/services/runtime/jobs.py` for long-running work.
- Long-running operations should expose:
  - start endpoint
  - poll or status endpoint
  - cancel endpoint

## Code Structure
- Keep functions small and focused.
- Prefer composable logic over deeply nested branching.
- Do not define nested functions or nested async functions.
- Do not place imports inside functions or classes.
- Do not use conditional imports for application modules.
- Do not retain module-level mutable service instances.
- Keep provider catalog persistence behind the repository serializer; services should receive the cache capability explicitly rather than opening ad-hoc database sessions.
- Keep Python files at or below 1000 physical lines.
- Use classes to group cohesive behavior where appropriate.
- Add comments only when they materially improve clarity or safety.
