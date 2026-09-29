# Testing And Quality
Last updated: 2026-09-29

## Testing standards

- Add unit tests for backend service and repository logic changes.
- Add API and rendered browser coverage for user-visible or workflow changes.
- Use deterministic assertions and isolate external integrations in explicitly named lanes.
- Store durable validation evidence under `assets/QA/`; keep generated caches and logs disposable.

## Mandatory automated-regression matrix

`test.automated-regression` is the deterministic product-regression component. Its mandatory CI matrix is:

- backend compilation;
- Alembic upgrade, current/head, and metadata-drift checks;
- Ruff;
- Pyright;
- the complete backend unit suite;
- the SQLite and PostgreSQL persistence contracts;
- Angular/Vitest;
- the Angular production build;
- deterministic Windows browser E2E;
- the synthetic access-key lifecycle (unit, API, and rendered UI coverage);
- dependency and security audits.

The Windows browser job invokes the single harness at
`app/tests/ci/run_browser_e2e.ps1 -Suite Full`. The harness seeds its isolated
SQLite database through repositories, starts the controlled Ollama protocol
fixture, starts backend/frontend services, runs the selected E2E files, checks
the JUnit result, emits diagnostics, and cleans up its process trees. The Full
suite excludes `test_live_provider_flow.py` and the heavyweight
`test_multilingual_embedding_runtime.py`; those are not unexplained mandatory
skips. A non-zero JUnit `skipped` count fails the Full suite.

The deterministic seed owns the persisted clinical-session, detail, timeline,
and timetable prerequisites. The fake Ollama service supplies one installed
model and the `/api/tags`, pull-job, show, status, and minimal embedding
protocol needed by the mandatory model API path. Real Ollama compatibility and
real embedding execution remain separate provider/RAG integration coverage.

## Persistence and external-lane boundaries

`app/tests/persistence/conftest.py` may skip PostgreSQL parameterizations in a
local SQLite-only run when `TEST_DATABASE_URL` is absent. That is a convenience
boundary, not missing mandatory coverage: the hosted `persistence-contract`
job supplies PostgreSQL and executes both engines. Do not require every
developer workstation to run PostgreSQL just to produce a zero-skip headline.

The `LiveProvider` harness lane runs only
`app/tests/e2e/test_live_provider_flow.py` and requires the hosted OpenCode Go
credential. Provider-side credential validity, provider-specific key formats,
real provider connectivity, and local Ollama installation are outside
`auth.access-key-management` and the deterministic Full suite. The access-key
gate uses synthetic format-valid values and validates DILIGENT's encrypted
storage, metadata-only responses, activation/rotation, restart persistence,
provider scope, deletion, and redaction behavior.

Embedding-dependent browser behavior may remain in its explicit optional RAG
lane. Core CI proves controlled ready/unavailable states without downloading
an uncontrolled embedding artifact; heavyweight real embedding coverage belongs
to `rag.ingestion-retrieval`.

## Quality gates

- Keep API-to-service-to-repository layering intact and do not bypass domain validation models.
- For model-configuration changes, include persistence/cache and API contract coverage; provider contact is asserted only for explicit load, refresh, or connectivity operations.
- Database schema changes require a reviewed Alembic revision. CI runs upgrade/head and drift checks.
- Documentation changes must be checked for stale paths, routes, version claims, and commands.
