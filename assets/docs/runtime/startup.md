# Startup
Last updated: 2026-08-21

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
- rebuilds the frontend when the main-menu install option 4 or frontend rebuild option 5 is executed, or when option 1 detects missing or unusable dependencies or frontend output during recovery
- validates that the frontend build is available before starting the preview server
- starts the backend with the synchronized virtual-environment Python and `uvicorn`
- starts the frontend preview server
- recreates a stale backend virtual environment when the repository has moved
- provides grouped source-control, database, test, log, cache, user-data, uninstall, and desktop-release options

Runtime and development tool caches are split between `runtimes/cache/` and
`app/tests/cache/`. The launcher routes uv, pip, npm, Playwright, Python
bytecode, and Cargo build caches to `runtimes/cache/`, and routes pytest, Ruff,
Mypy, Angular, and coverage state to `app/tests/cache/`, while leaving
functional frontend and desktop release outputs in their required runtime
locations.

Database startup behavior is migration-driven:

- In SQLite mode, application startup creates the configured `.db` when
  missing, applies Alembic migrations to head, and seeds catalogs once. An
  existing file is checked and upgraded without reseeding.
- In PostgreSQL mode, application startup connects to the configured database,
  creates it when permitted, applies Alembic migrations to head, and seeds only
  a newly created database. Existing databases do not require `CREATEDB`.
  When the target is absent, provision it in advance or grant the configured
  role `CREATEDB`; authentication, network, and permission failures abort startup.
- The explicit `InitializeDatabase` action remains the repeatable operator path
  for either backend. It applies pending migrations and seeds idempotently;
  `--drop-existing` is the explicit destructive reset path.
- Install option 4 runs the same database synchronization after backend
  and frontend dependencies are ready. Launch performs the check again so
  startup remains safe when installation was skipped.

Use this launcher as the default startup path for local development, Codex sessions, and browser-driven UI work. On a fresh checkout, execute option 4 first to install dependencies, synchronize the database, and build the frontend, then execute option 1 to launch the application. Use option 5, or `.\start_on_windows.ps1 -Action RebuildFrontend`, to rebuild only the frontend after frontend changes or when its production output needs refreshing; this does not synchronize Python dependencies or the database. Option 2 checks `origin/main` with `git ls-remote` and does not download or apply changes. Option 3 runs `git pull origin main` against the current checkout. Option 10 removes local user data, including the SQLite database and generated/user-created resource files, while preserving tracked application files; use `.\start_on_windows.ps1 -Action RemoveAllData -Force` for a non-interactive invocation. Option 1 also recovers missing or unusable environments and frontend output. Do not start backend and frontend manually first unless the task specifically requires isolating one side or the launcher has already failed and the failure has been diagnosed.

## Packaged desktop startup

The Windows portable executable and MSI use the Tauri shell. Open the verified portable EXE directly, or launch the application installed by the MSI; do not run the source launcher for packaged operation. On first launch the shell verifies the embedded runtime archive, extracts it to a versioned hash directory under `%LOCALAPPDATA%\DILIGENT\runtime`, creates persistent data directories under `%LOCALAPPDATA%\DILIGENT\data`, starts the packaged backend on a random localhost port, waits for its atomic ready file and `/api/health`, and then shows the desktop window. The backend is owned by a Windows Job Object and is terminated when the shell exits.

The packaged desktop does not use the development ports `7690` and `9847`. If the window does not appear, inspect `%LOCALAPPDATA%\DILIGENT\data\resources\logs\desktop-backend.log`, confirm that `state\desktop-backend-ready.json` exists, and request `/api/health` on the recorded port. A successful launch leaves `runtime\<version>\<payload-sha256>\extraction.complete` in place.

## Manual Backend Startup
From repository root:

```powershell
Set-Location app/server
./.venv/Scripts/python.exe -m uvicorn app:app --host 127.0.0.1 --port 7690 --log-level info
```

Alternative runtime-managed path:

```powershell
runtimes\uv\uv.exe run --directory app/server python -m uvicorn app:app --host 127.0.0.1 --port 7690
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
