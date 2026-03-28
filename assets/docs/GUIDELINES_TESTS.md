# Testing Guidelines

This document defines the current testing approach for DILIGENT Clinical Copilot.

## 1. Test layers

- Unit tests: `tests/unit`
  - Validate backend logic, parsing, repositories, configuration behavior, and safety checks.
- E2E tests: `tests/e2e`
  - Validate API contracts and core frontend flows through pytest + Playwright.

Primary tools:
- `pytest`
- `pytest-playwright`

## 2. Current suite inventory (summary)

`tests/e2e` currently covers:
- root/docs/openapi routes
- clinical API flow
- models API flow
- access keys API flow
- research API flow
- app UI flow
- rxnav concurrency diagnostic

`tests/unit` currently covers:
- clinical extraction/matching/pipeline logic
- repository and DB mode behavior
- error-handling enforcement and timeout config
- inspection/research/rxnav/rag behavior
- security hardening checks

Use `tests\run_tests.bat` or directory listing if you need the exact file list.

## 3. Preferred execution

Recommended full run on Windows:
```cmd
tests\run_tests.bat
```

What it does:
1. Validates portable runtimes under `runtimes/`.
2. Starts backend and frontend.
3. Ensures Playwright browser availability.
4. Runs unit tests then E2E tests.
5. Cleans up ports/processes.

## 4. Manual execution

Use the project runtime/venv (`runtimes/.venv`) and Python 3.14+.

Examples:
```cmd
uv run pytest -q tests/unit
uv run pytest -q tests/e2e
uv run pytest -q tests/e2e/test_clinical_api.py
uv run pytest tests/e2e/test_app_flow.py --headed --slowmo 300
```

## 5. Fixture and env behavior

`tests/conftest.py` resolves test URLs in this order:
1. `APP_TEST_FRONTEND_URL`, `APP_TEST_BACKEND_URL`
2. alias vars such as `UI_BASE_URL`, `API_BASE_URL`
3. host/port vars such as `UI_HOST/UI_PORT`, `FASTAPI_HOST/FASTAPI_PORT`

Core fixtures include:
- `base_url`
- `api_base_url`
- `api_context`
- `page`

## 6. Authoring rules

- Place tests in `tests/unit` or `tests/e2e`.
- Use `test_` naming for files and functions.
- Follow Arrange-Act-Assert structure.
- Keep assertions deterministic and avoid brittle timing assumptions.
- For external dependencies (for example Ollama), skip explicitly with a clear reason when prerequisites are missing.

## 7. Mandatory error-path coverage

Tests must enforce `assets/docs/ERROR_HANDLING.md` by covering:
- invalid inputs
- malformed payloads/responses
- dependency failures
- timeout behavior
- cancellation/retry behavior where applicable
- safe user-facing error messages (no internals leaked)

## 8. Common troubleshooting

- Services not reachable:
  - Ensure `FASTAPI_PORT` (default `8000`) and `UI_PORT` (default `7861`) are free.
  - Run `DILIGENT/start_on_windows.bat` for first-time runtime provisioning.
- Playwright browser issues:
  - `uv run python -m playwright install`
- Access key failures:
  - Ensure `ACCESS_KEY_ENCRYPTION_KEY` is configured in `DILIGENT/settings/.env`.
- Ollama-dependent failures:
  - Ensure Ollama is running and reachable from configured host/port.
