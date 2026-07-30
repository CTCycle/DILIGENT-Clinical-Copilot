# Startup
Last updated: 2026-07-29

## Recommended Local Startup
On Windows, use:

```powershell
.\start_on_windows.ps1
```

The launcher:
- creates `settings/.env` from `settings/.env.example` on first application start when the local environment file is missing
- ensures portable Python, `uv`, and Node runtimes under `runtimes/`
- runs `uv sync`
- installs frontend dependencies
- builds frontend dist at application startup when `ALWAYS_REBUILD=true`; skips that build when `false`
- starts the backend with the synchronized virtual-environment Python and `uvicorn`
- starts the frontend preview server
- recreates a stale backend virtual environment when the repository has moved
- provides database, test, log, cache, and uninstall maintenance options

Use this launcher as the default startup path for local development, Codex sessions, and browser-driven UI work. Do not start backend and frontend manually first unless the task specifically requires isolating one side or the launcher has already failed and the failure has been diagnosed.

## Packaged desktop startup

The Windows portable executable and MSI use the Tauri shell. Open the published portable EXE directly, or launch the application installed by the MSI; do not run the source launcher for packaged operation. On first launch the shell verifies the embedded runtime archive, extracts it to a versioned hash directory under `%LOCALAPPDATA%\DILIGENT\runtime`, creates persistent data directories under `%LOCALAPPDATA%\DILIGENT\data`, starts the packaged backend on a random localhost port, waits for its atomic ready file and `/api/health`, and then shows the desktop window. The backend is owned by a Windows Job Object and is terminated when the shell exits.

The packaged desktop does not use the development ports `7690` and `9847`. If the window does not appear, inspect `%LOCALAPPDATA%\DILIGENT\data\resources\logs\desktop-backend.log`, confirm that `state\desktop-backend-ready.json` exists, and request `/api/health` on the recorded port. A successful launch leaves `runtime\<version>\<payload-sha256>\extraction.complete` in place.

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
### Source/development mode
1. Confirm port `7690` is free or intentionally used by the current backend.
2. Start the backend.
3. Verify `http://127.0.0.1:7690/docs` responds.
4. Start the frontend on `9847`.
5. Open `http://127.0.0.1:9847`.

### Packaged desktop mode
1. Open the portable EXE or launch the installed MSI application.
2. Confirm the desktop window appears with the title `DILIGENT Clinical Copilot`.
3. If startup fails, inspect the packaged log and ready file under `%LOCALAPPDATA%\DILIGENT\data`.
