# Testing And Quality
Last updated: 2026-08-20

## Testing Standards
- Add unit tests for logic changes in backend services and repositories.
- Add E2E coverage for user-visible or API workflow changes.
- Use deterministic assertions and explicit skip conditions for unavailable external dependencies.

## Quality Gates
- Lint and format with Ruff, or the project-standard equivalent if that changes later.
- Type-check expectations are Pylance-compatible typing on the backend and strict TypeScript on the frontend.
- Keep architecture layering intact: API to service to repository.
- Do not bypass domain validation models.
- Do not duplicate business logic across backend and frontend without necessity.
- Add or adjust tests whenever behavior, contracts, or data schemas change.
- For model-configuration changes, include persistence/cache coverage and API contract coverage; provider contact should be asserted only for explicit load, refresh, or connectivity operations.
- Database schema changes require a reviewed Alembic revision. Persistence
  contract fixtures use Alembic to build and reset schemas; isolated unit
  fixtures may use metadata creation only when migration behavior is outside
  the test's scope. CI runs Alembic drift checking.
- Documentation changes should be checked for stale paths, routes, version claims, and commands with repository-local searches before handoff.
