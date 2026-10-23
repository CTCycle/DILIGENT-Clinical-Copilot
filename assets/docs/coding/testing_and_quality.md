# Testing and Quality
Last updated: 2026-10-22

## Testing standards

- Add unit tests for backend service and repository changes.
- Add API and rendered-browser coverage for user-visible behavior and workflows.
- Use deterministic assertions and isolate external integrations in explicitly named lanes.
- Group related backend and frontend behaviors into subsystem suites; keep provider-backed validation in explicit integration lanes.
- Keep evidence claims scoped to the code, environment, and workflows actually exercised.
- Put durable behavior and limitations in their existing product-concept documents. Local logs, caches, and screenshots belong in ignored scratch space; do not add a tracked report for every run.

## Mandatory regression matrix

The Windows CI workflow is the required source-regression gate. It covers:

- backend compilation, Alembic upgrade/head and metadata-drift checks, Ruff, Pyright, and backend unit tests;
- SQLite and PostgreSQL persistence contracts;
- the consolidated Angular/Vitest suite, including runtime network configuration contracts, and the Angular production build;
- deterministic Windows browser E2E and the synthetic access-key lifecycle;
- Python, frontend, and desktop dependency/security audits.

The canonical browser command, from the repository root, is:

~~~powershell
.\app\tests\ci\run_browser_e2e.ps1 -Suite Full
~~~

The Full harness seeds an isolated SQLite database through repositories, starts a controlled Ollama protocol fixture and the application services, runs the browser suite, validates JUnit results, emits diagnostics, and cleans up its process trees. It fails when a mandatory test is skipped. It excludes the live-provider test and heavyweight real-embedding integration; those are explicit lanes.

Run backend unit tests from `app/server` with `.\.venv\Scripts\python.exe -m pytest ..\tests\unit -q`. Run frontend unit tests from the repository root with `npm.cmd --prefix app/client run test -- --no-watch`. Runtime network configuration tests are part of the Vitest suite; there is no separate Node test command.

The deterministic fixture supplies the model API responses needed by mandatory browser coverage. It does not prove compatibility with an installed Ollama model or real embedding execution.

## External integration lanes

The hosted PostgreSQL job supplies PostgreSQL for the persistence contract. A local SQLite-only persistence run may skip PostgreSQL parameterizations when TEST_DATABASE_URL is absent; that local convenience does not replace hosted coverage.

To exercise the live provider route, use the explicit LiveProvider lane:

~~~powershell
.\app\tests\ci\run_browser_e2e.ps1 -Suite LiveProvider
~~~

It runs the live-provider flow and requires DILIGENT_LIVE_PROVIDER_E2E=1 plus OPENCODE_GO_API_KEY in the hosted secret store. Never copy provider credentials into source, logs, or evidence. Synthetic access-key lifecycle tests prove DILIGENT's storage and redaction behavior, not provider-side credential validity.

Real local Ollama compatibility and heavyweight embedding behavior remain provider/RAG integration coverage; a deterministic fixture or a successful API response is not a substitute for those lanes.

## Local persistence slice

For SQLite-only work, from the repository root:

~~~powershell
Set-Location .\app\server
.\.venv\Scripts\python.exe -m pytest ..\tests\persistence -q
~~~

Without TEST_DATABASE_URL, PostgreSQL parameterizations are intentionally skipped. The hosted persistence contract remains the acceptance gate for both engines.

## Quality and documentation review

- Keep API-to-service-to-repository layering intact and validate requests through domain models.
- For model-configuration changes, cover persistence/cache and API contracts; assert provider contact only for explicit load, refresh, or connectivity operations.
- Database schema changes require a reviewed Alembic revision and upgrade/head/drift coverage.
- For UI changes, inspect rendered behavior and keyboard/accessibility behavior at the supported desktop widths.
- Check documentation links, routes, version claims, and commands after documentation changes.
- Use [desktop release](../runtime/desktop_release.md) for portable/MSI build, lifecycle, publication, and downloaded-byte gates. Source regression does not establish package parity, signing, clean-machine installation, or release publication.
