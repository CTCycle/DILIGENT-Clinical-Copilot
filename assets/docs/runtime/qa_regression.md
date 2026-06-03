# QA Regression
Last updated: 2026-06-03

## Scope
This file captures the repeatable regression slice for model configuration and app-flow validation.

## Recommended Runner

```powershell
.\app\tests\run_model_config_regression.ps1
```

This runner performs startup, health checks, focused unit and E2E commands, and cleanup.

## Full Regression Variant

```powershell
.\app\tests\run_model_config_full_regression.ps1
```

Use this when validating the full `test_app_flow.py` suite plus `test_model_config_api.py`.

## `run_tests.bat` Shortcuts

```cmd
app\tests\run_tests.bat modelconfig
app\tests\run_tests.bat modelconfigfull
```

These shortcuts invoke the PowerShell runners, set `UV_CACHE_DIR` to `%PROJECT_ROOT%\.uv-cache`, and propagate non-zero exit codes on failure.

## SQLite Writeability Hardening
Regression scripts set a per-run temporary database path through:

- `DILIGENT_SQLITE_PATH=<temp file>`

This avoids accidental writes to a shared `app/resources/database.db` and prevents readonly-state failures during concurrent or constrained runs.

## Local-first Test Execution
- If `pytest` and `pytest-playwright` are installed in `app/server/.venv`, the scripts run `python -m pytest` directly.
- Otherwise they fall back to `uv run --with ...`.
- The focused E2E step uses `uv --with pytest-playwright`.
- If package metadata is not cached locally, first-run success may require outbound package access.

## Manual Validation Sequence
### 1. Start Backend And Frontend

```powershell
Start-Process -FilePath '.\app\server\.venv\Scripts\python.exe' -ArgumentList '-m','uvicorn','app:app','--host','127.0.0.1','--port','7690' -WorkingDirectory '.\app\server' -WindowStyle Hidden
Start-Process -FilePath 'npm.cmd' -ArgumentList 'run','start' -WorkingDirectory '.\app\client' -WindowStyle Hidden
```

### 2. Confirm Backend Health

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:7690/api/health
```

### 3. Run Model-config Unit Tests

```powershell
.\runtimes\uv\uv.exe run --directory app/server --with pytest pytest ..\tests\unit\test_model_config_persistence.py -q
```

### 4. Run Focused E2E Slice

```powershell
$env:APP_TEST_FRONTEND_URL='http://127.0.0.1:9847'
$env:APP_TEST_BACKEND_URL='http://127.0.0.1:7690'
.\runtimes\uv\uv.exe run --directory app/server --with pytest --with pytest-playwright pytest ..\tests\e2e\test_model_config_api.py ..\tests\e2e\test_app_flow.py -k "runtime_toggle_enables_save_and_submits_put or model_config or dili_run_burst_click_submits_single_job or dili_run_conflict_surfaces_clear_error_message" -q
```

### 5. Optional Full App-flow Pass

```powershell
$env:APP_TEST_FRONTEND_URL='http://127.0.0.1:9847'
$env:APP_TEST_BACKEND_URL='http://127.0.0.1:7690'
.\runtimes\uv\uv.exe run --directory app/server --with pytest --with pytest-playwright pytest ..\tests\e2e\test_model_config_api.py ..\tests\e2e\test_app_flow.py -q
```

## Expected Pass Signatures
- Model-config unit pass:
  - `3 passed`
- Focused E2E slice:
  - `7 passed, 3 deselected`
- Full model-config plus app-flow pass:
  - `21 passed`

If results differ:
- Re-check backend and frontend health and port listeners.
- Confirm `PLAYWRIGHT_NODEJS_PATH` is set by `app/tests/conftest.py` and points to `runtimes/nodejs/node.exe`.
- If toggle or save tests fail, remove stale persisted runtime state and rerun once after cleanup.

## Cleanup

```powershell
$ports=7690,9847
$conns=Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -in $ports }
$ids=@($conns | Select-Object -ExpandProperty OwningProcess -Unique)
foreach($id in $ids){ Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
```
