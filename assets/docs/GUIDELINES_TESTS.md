# How To Test

This document describes the current testing strategy for DILIGENT Clinical Copilot.

## Overview

The repository uses two complementary layers:
- Unit tests (`tests/unit`): fast validation of backend logic and regression-prone helpers.
- E2E tests (`tests/e2e`): Playwright + pytest scenarios that validate API and key UI flows.

Primary stack:
- `pytest`
- `pytest-playwright`
- Playwright browser runtime (`chromium` by default in `tests/run_tests.bat`)

## Current Test Layout

```text
tests/
├── conftest.py
├── run_tests.bat
├── e2e/
│   ├── test_access_keys_api.py
│   ├── test_app_flow.py
│   ├── test_clinical_api.py
│   ├── test_models_api.py
│   └── test_root_api.py
└── unit/
    ├── test_access_keys.py
    ├── test_anamnesis_disease_extraction.py
    ├── test_anamnesis_drug_extraction.py
    ├── test_database_mode_env_override.py
    ├── test_drugs_parser.py
    ├── test_external_data_timeouts.py
    ├── test_hepatox_assessment.py
    ├── test_livertox_matching_pipeline.py
    ├── test_pandas_migration.py
    ├── test_polling_interval_centralization.py
    └── test_seed_scripts_idempotency.py
```

## Recommended Execution

### Windows full-stack run (recommended)

```cmd
tests\run_tests.bat
```

This script:
1. Validates portable runtimes from `runtimes/`.
2. Ensures Playwright browsers are available.
3. Starts backend and frontend.
4. Runs pytest suite.
5. Cleans up processes on used ports.

## Manual Execution

Prerequisites:
- Python 3.14+
- Project dependencies installed (`pip install -e .[test]` or `uv sync --all-extras`)
- Playwright browsers installed (`python -m playwright install`)

Typical commands:

```bash
uv run pytest -q tests/unit
uv run pytest -q tests/e2e
```

Targeted examples:

```bash
uv run pytest -q tests/e2e/test_root_api.py
uv run pytest -q tests/e2e/test_clinical_api.py
```

For visible browser runs:

```bash
uv run pytest tests/e2e/test_app_flow.py --headed --slowmo 300
```

## Environment and Fixtures

`tests/conftest.py` resolves URLs in this order:
- `APP_TEST_FRONTEND_URL`, `APP_TEST_BACKEND_URL`
- fallback aliases (`UI_BASE_URL`, `API_BASE_URL`, etc.)
- fallback host/port env pairs (`UI_HOST/UI_PORT`, `FASTAPI_HOST/FASTAPI_PORT`)

Key fixtures:
- `base_url`: frontend origin.
- `api_base_url`: backend origin.
- `api_context`: Playwright API request context.
- `page`: Playwright browser page.

## Endpoint Coverage (Current)

- Root/OpenAPI:
  - `GET /`
  - `GET /docs`
  - `GET /openapi.json`
- Clinical:
  - `POST /clinical`
- Models/Ollama:
  - `GET /models/list`
  - `GET /models/pull`
- Access keys:
  - `GET /access-keys`
  - `POST /access-keys`
  - `PUT /access-keys/{id}/activate`
  - `DELETE /access-keys/{id}`

## Writing New Tests

- Keep test files under `tests/unit` or `tests/e2e`.
- Name files/functions with `test_` prefix.
- Follow Arrange-Act-Assert style.
- Prefer deterministic assertions over timing-sensitive checks.
- For external dependencies (Ollama, encryption keys), skip with explicit reason when preconditions are not met.

## Troubleshooting

- Backend/frontend unavailable:
  - Verify ports `8000` and `7861` are free.
  - Run `DILIGENT/start_on_windows.bat` first on a fresh environment.
- Playwright browser errors:
  - Run `python -m playwright install`.
- Access key tests fail with 5xx:
  - Ensure `ACCESS_KEY_ENCRYPTION_KEY` is set in `DILIGENT/settings/.env`.
- Model endpoint tests fail with 502/504:
  - Ensure Ollama is reachable and running.

- Clinical endpoint tests return a report without LLM synthesis:
  - This is expected when no active cloud key is configured; the response includes baseline pattern/drug sections and a `clinical_llm_unavailable` warning in pipeline issues.
