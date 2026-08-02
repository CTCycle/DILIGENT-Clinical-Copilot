# QA Regression
Last updated: 2026-08-02

## Scope
This file captures the repeatable regression slice for model configuration and app-flow validation.

## Packaged desktop smoke test

After a Windows desktop release build, validate the built portable artifact separately from source-mode tests:

1. Verify `release\DILIGENT-v<version>-windows-x64-portable.exe` and the matching `.sha256` entry.
2. Open the portable EXE and confirm a responding window titled `DILIGENT Clinical Copilot`.
3. Confirm `%LOCALAPPDATA%\DILIGENT\runtime\<version>\<payload-sha256>\extraction.complete` exists.
4. Read `%LOCALAPPDATA%\DILIGENT\data\state\desktop-backend-ready.json` and request `/api/health` on its recorded port.
5. Confirm `%LOCALAPPDATA%\DILIGENT\data\resources\logs\desktop-backend.log` contains successful startup and static-asset requests.
6. Open Model Configurations twice and confirm the second read uses the persisted provider catalog without a provider request. Click **Refresh** once and confirm only that action replaces the catalog. Repeat with a freshly initialized database to verify the cold-load path.

This smoke test does not replace MSI install/upgrade/uninstall, offline WebView2, code-signing, or clean-machine distribution testing. Packaged desktop uses a random backend port and should not be tested through the source-mode `7690`/`9847` URLs.

## Recommended Runner

```cmd
app\tests\run_tests.bat modelconfig
```

This runner performs startup, health checks, focused unit and E2E commands, and cleanup.

## Full Regression Variant

```cmd
app\tests\run_tests.bat modelconfigfull
```

Use this when validating the full `test_app_flow.py` suite plus `test_model_config_api.py`.

## `run_tests.bat` Shortcuts

Both variants are available through `run_tests.bat`:

```cmd
app\tests\run_tests.bat modelconfig
app\tests\run_tests.bat modelconfigfull
```

These set `DILIGENT_SQLITE_PATH` to a temporary database, override ports 7690/9847, and propagate non-zero exit codes on failure.

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

## Expected pass criteria

- The selected unit, API, and app-flow tests complete with exit code `0`.
- Model-configuration tests cover persisted state, provider catalog cache reuse, explicit refresh behavior, and save validation.
- The UI remains usable after the provider catalog is unavailable; a cached valid catalog remains visible and an empty Ollama catalog is treated as a valid result.

Test counts are intentionally not fixed here because the repository adds and removes focused cases as contracts evolve. If a current run fails:
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
