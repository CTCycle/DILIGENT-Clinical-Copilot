# Launch Troubleshooting

Last updated: 2026-05-25

## Scope

This note captures recurring startup issues found while launching local backend and frontend for DILIGENT Clinical Copilot.

## Expected local ports

- Backend: `127.0.0.1:7690`
- Frontend: `127.0.0.1:9847`

## Issue 1: Frontend exits immediately with backend-unreachable error

### Symptom

Frontend preview prints:

`Configured backend is unreachable at http://127.0.0.1:7690`

and does not keep running.

### Cause

`app/client/scripts/preview-server.mjs` validates backend availability before serving UI.

### Fix

1. Start backend first.
2. Verify backend health before launching frontend:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:7690/docs
```

3. Only then run frontend preview:

```powershell
Set-Location app/client
npm run preview -- --host 127.0.0.1 --port 9847 --strictPort
```

## Issue 2: Backend fails to bind with WinError 10048

### Symptom

`[Errno 10048] ... only one usage of each socket address ... ('127.0.0.1', 7690)`

### Cause

Port `7690` already occupied by an existing Python process (often from a previous session).

### Fix

1. Check current listener:

```powershell
Get-NetTCPConnection -LocalPort 7690 | Select-Object LocalAddress,LocalPort,State,OwningProcess
```

2. Inspect process:

```powershell
Get-Process -Id <PID>
```

3. Stop stale process if needed:

```powershell
Stop-Process -Id <PID> -Force
```

4. Restart backend.

## Issue 3: Backend command works only with correct module path context

### Symptom

Backend startup appears inconsistent when run from mismatched working directory or without app path context.

### Reliable command

From repository root:

```powershell
app/server/.venv/Scripts/python.exe -m uvicorn app:app --app-dir app --host 127.0.0.1 --port 7690 --log-level info
```

## Quick startup checklist

1. Confirm `7690` is free or intentionally used by current backend process.
2. Start backend with `--app-dir app`.
3. Verify `http://127.0.0.1:7690/docs` responds.
4. Start frontend preview on `9847`.
5. Open `http://127.0.0.1:9847`.

## Repeatable QA Regression Slice (Model Config + App Flow)

Use this sequence from repository root to run the validated regression slice that covers:
- model-config unit behavior,
- model-config API contracts,
- UI app-flow navigation/interactions including runtime toggle/save and explicit clinical-run conflict feedback.

### One-command runner (preferred)

```powershell
.\app\tests\run_model_config_regression.ps1
```

This script performs startup, health checks, unit + e2e regression commands, and cleanup.

### One-command full pass

```powershell
.\app\tests\run_model_config_full_regression.ps1
```

This variant runs the full `test_app_flow.py` plus `test_model_config_api.py` after unit checks.

### run_tests.bat shortcuts

From repository root:

```cmd
app\tests\run_tests.bat modelconfig
app\tests\run_tests.bat modelconfigfull
```

These invoke the same PowerShell runners, set `UV_CACHE_DIR` to `%PROJECT_ROOT%\.uv-cache` for the run, and propagate non-zero exit codes on failure.

### SQLite writeability hardening used by regression scripts

The regression scripts now set a per-run temporary database path via:

- `DILIGENT_SQLITE_PATH=<temp file>`

This avoids accidental writes against a shared `app/resources/database.db` file and prevents `PUT /api/model-config` failures caused by readonly DB state during concurrent or constrained runs.

Note:
- Model-config runners now use local-first test execution:
  - If `pytest` / `pytest-playwright` are installed in `app/server/.venv`, scripts run `python -m pytest` directly.
  - Otherwise they fall back to `uv run --with ...` behavior.
- The focused e2e step uses `uv --with pytest-playwright`. If package metadata/artifacts are not already cached locally, outbound package access is required for the first successful run.
- In restricted environments, dependency fetch may fail with socket permission/network errors (for example `os error 10013` while contacting `pypi.org`). In that case, rerun with allowed outbound access or pre-provision the uv cache.
- In restricted environments, the direct PowerShell runner is the authoritative path:
  - `.\app\tests\run_model_config_regression.ps1`

### 1) Start backend and frontend

```powershell
Start-Process -FilePath '.\app\server\.venv\Scripts\python.exe' -ArgumentList '-m','uvicorn','app:app','--host','127.0.0.1','--port','7690' -WorkingDirectory '.\app\server' -WindowStyle Hidden
Start-Process -FilePath 'npm.cmd' -ArgumentList 'run','start' -WorkingDirectory '.\app\client' -WindowStyle Hidden
```

### 2) Confirm backend health

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:7690/api/health
```

### 3) Run model-config unit tests

```powershell
.\runtimes\uv\uv.exe run --directory app/server --with pytest pytest ..\tests\unit\test_model_config_persistence.py -q
```

### 4) Run model-config + app-flow e2e slice

```powershell
$env:APP_TEST_FRONTEND_URL='http://127.0.0.1:9847'
$env:APP_TEST_BACKEND_URL='http://127.0.0.1:7690'
.\runtimes\uv\uv.exe run --directory app/server --with pytest --with pytest-playwright pytest ..\tests\e2e\test_model_config_api.py ..\tests\e2e\test_app_flow.py -k "runtime_toggle_enables_save_and_submits_put or model_config or dili_run_burst_click_submits_single_job or dili_run_conflict_surfaces_clear_error_message" -q
```

### 4b) Optional full app-flow + model-config pass

Use this when validating broader UI flow stability, not just the focused regression slice:

```powershell
$env:APP_TEST_FRONTEND_URL='http://127.0.0.1:9847'
$env:APP_TEST_BACKEND_URL='http://127.0.0.1:7690'
.\runtimes\uv\uv.exe run --directory app/server --with pytest --with pytest-playwright pytest ..\tests\e2e\test_model_config_api.py ..\tests\e2e\test_app_flow.py -q
```

### Expected pass signatures

- Model-config unit pass:
  - `3 passed` from `test_model_config_persistence.py`
- Focused e2e slice pass:
  - `7 passed, 3 deselected` for the filtered app-flow/model-config run
- Full app-flow + model-config pass:
  - `21 passed` total (`test_model_config_api.py` + full `test_app_flow.py`, including timetable + navigation + console/network + keyboard + form-focus guards)

If signatures differ:
- Re-check backend/frontend health and port listeners (`7690`, `9847`).
- Confirm `PLAYWRIGHT_NODEJS_PATH` is set by `app/tests/conftest.py` and points to `runtimes/nodejs/node.exe`.
- If model-config toggle/save tests fail, ensure no stale persisted runtime state is biasing defaults; rerun once after cleanup.

### 5) Cleanup (stop listeners on 7690 and 9847)

```powershell
$ports=7690,9847
$conns=Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -in $ports }
$ids=@($conns | Select-Object -ExpandProperty OwningProcess -Unique)
foreach($id in $ids){ Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
```
