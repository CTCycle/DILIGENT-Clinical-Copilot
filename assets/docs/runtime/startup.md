# Startup
Last updated: 2026-07-11

## Recommended Local Startup
On Windows, use:

```powershell
.\start_on_windows.ps1
```

The launcher:
- ensures portable Python, `uv`, and Node runtimes under `runtimes/`
- runs `uv sync`
- installs frontend dependencies
- builds frontend dist
- starts the backend with the synchronized virtual-environment Python and `uvicorn`
- starts the frontend preview server
- recreates a stale backend virtual environment when the repository has moved
- provides database, test, log, cache, and uninstall maintenance options

Use this launcher as the default startup path for local development, Codex sessions, and browser-driven UI work. Do not start backend and frontend manually first unless the task specifically requires isolating one side or the launcher has already failed and the failure has been diagnosed.

## Manual Backend Startup
From repository root:

```powershell
Set-Location app/server
./.venv/Scripts/python.exe -m uvicorn app:app --host 127.0.0.1 --port 7690 --log-level info
```

Alternative runtime-managed path:

```powershell
runtimes\uv\uv.exe run --python runtimes\python\python.exe python -m uvicorn DILIGENT.app:app --host 127.0.0.1 --port 7690
```

## Manual Frontend Startup

```powershell
Set-Location app/client
npm run preview -- --host 127.0.0.1 --port 9847 --strictPort
```

## Quick Startup Checklist
1. Confirm port `7690` is free or intentionally used by the current backend.
2. Start the backend.
3. Verify `http://127.0.0.1:7690/docs` responds.
4. Start the frontend on `9847`.
5. Open `http://127.0.0.1:9847`.
