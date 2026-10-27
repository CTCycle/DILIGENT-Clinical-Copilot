# Startup
Last updated: 2026-10-27

## Recommended Local Startup
On Windows, use:

```powershell
.\start_on_windows.ps1
```

The launcher:
- creates `settings/.env` from `settings/.env.example` on first application start when the local environment file is missing
- ensures the backend Python runtime for a warm launch; `uv` is prepared when backend/runtime repair or an explicit install requires it
- runs `uv sync --locked` against the tracked `app/server/uv.lock` during backend/runtime repair and explicit installation
- prepares Node and installs frontend dependencies only when an explicit frontend install/rebuild or a stale/missing frontend build requires them
- rebuilds the frontend when the main-menu install option 2 or frontend rebuild option 3 is executed, or when option 1 detects missing or stale source-mode build output
- validates the deterministic frontend build-state marker before reusing an existing production build
- starts one FastAPI process with the synchronized virtual-environment Python; FastAPI serves the built Angular SPA and `/api/*` from the same origin
- waits for `/api/health`, the SPA entry page, and a built JavaScript asset before opening the backend origin in the browser
- reports each major launch phase and total time to application readiness in milliseconds
- keeps the backend process hidden and writes its output to `runtimes/cache/logs/source-backend.log` and `source-backend-error.log`; the visible launcher owns the process
- keeps the interactive menu open while the app runs; `EXIT` and Ctrl+C request authenticated graceful shutdown, wait, then stop only the launcher-owned process tree if needed
- makes `-Action Launch` supervise the server until it stops or Ctrl+C is pressed
- recreates a stale backend virtual environment when the repository has moved
- provides grouped `APPLICATION`, `SETUP & VALIDATION`, `SOURCE CONTROL`, `BUILD & DISTRIBUTION`, and `DATA & MAINTENANCE` options, followed by a final sequential `EXIT` option; the desktop-release submenu uses the same aligned numeric rows
- checks the configured API port immediately before starting the production process; `UI_PORT` is reserved for frontend development
- prompts once before directly stopping the listed unique port holders; redirected or declined launches fail safely without terminating anything

All source-mode disposable runtime, application, test, and tool state is rooted
at `runtimes/cache/`. The launcher routes uv, pip, npm, Playwright, Python
bytecode, Cargo, pytest, Ruff, Mypy, Angular, logs, embeddings, test
databases, Hugging Face metadata, and reports below that root. The hierarchy is disposable and may be
deleted and recreated without affecting settings, databases, source documents,
vectors, or user exports. Packaged desktop mode uses the equivalent writable
`%LOCALAPPDATA%\DILIGENT\data\cache` root.

All launcher recursive cleanup uses the local `Remove-LauncherPath` contract:
targets are normalized, recursively inventoried, filtered for tracked files and
sentinels, removed one entry at a time deepest-first, and reported with planned,
removed, preserved, skipped, and enumeration-error results. Strict staging and
setup cleanup raises after reporting incomplete removal; user-facing cleanup is
best effort.

Database connection settings are loaded from `<data-root>/database_config.json`
before the backend initializes its repository. This file contains only the
selected backend and connection parameters; no database configuration is
required in `.env` or process environment variables. Application settings are
loaded from the selected database after it is ready.

Database startup behavior is migration-driven:

- In SQLite mode, application startup creates the configured `.db` when
  missing, applies Alembic migrations to head, and seeds catalogs when the app
  schema is new or the reference catalog is empty.
- In PostgreSQL mode, application startup connects to the configured database,
  creates it when permitted, applies Alembic migrations to head, and seeds when
  the app schema is new or the reference catalog is empty. Existing databases
  do not require `CREATEDB`.
  When the target is absent, provision it in advance or grant the configured
  role `CREATEDB`; authentication, network, and permission failures abort startup.
- The Settings → Database page can switch SQLite/PostgreSQL without restarting.
  It pauses new API work, waits up to 60 seconds for current API requests, and
  rejects the switch while background jobs are pending or running. It verifies
  the target schema before publishing the new connection and reloads settings
  from that database. Failed activation restores the previous configuration
  and runtime services.
- Restart is required after changing the data root, packaged runtime location,
  access-key-material path, or other process-level startup settings. If an
  activation failure reports that automatic rollback could not restore
  services, restart to load the saved working backend.
- The explicit `InitializeDatabase` action remains the repeatable operator path
  for either backend. It applies pending migrations and seeds idempotently;
  `--drop-existing` is the explicit destructive reset path.
- Install option 2 runs the same database synchronization after backend
  and frontend dependencies are ready. Launch performs the check again so
  startup remains safe when installation was skipped.

For a deterministic disposable-state reset, run `.\start_on_windows.ps1 -Action ClearCache`; it requires the same interactive `[y/N]` confirmation, preserves only `runtimes/cache/.gitkeep`, and reports locked entries without aborting the cleanup.

If a previous launch left backend, frontend development, or launcher wrapper processes running,
run `.\start_on_windows.ps1 -Action KillApplicationProcesses` and confirm the
cleanup when prompted.

## Source-mode frontend build freshness

Source-mode production output is accompanied by
`app/client/dist/.diligent-build-state.json`. The marker contains a schema
version, source/build fingerprint, dependency fingerprint, pinned Node version,
the canonical `npm run build` command, and the inventory of files emitted under
`dist/browser`. The launcher reuses the output only when the marker and
`dist/browser/index.html` are present, all fingerprints still match, and every
recorded output file still exists.

The build fingerprint includes sorted, normalized paths and content digests for
production files under `app/client/src/` (excluding `*.spec.ts`),
`app/client/public/**`, the frontend manifests/configuration files,
`app/client/scripts/build.mjs`, `app/client/scripts/runtime-config.mjs`, the
pinned Node version, the build command, and the effective
`VITE_API_BASE_URL`. It excludes preview/dev-server scripts, other
`settings/.env` values, backend Python, QA files, and documentation. A missing,
corrupt, old-schema, or stale output marker, or a missing recorded output file,
triggers one source-mode rebuild. Dependency-manifest or pinned-Node changes
run `npm ci` from the tracked lockfile before that build; source or build-script
changes reuse installed dependencies. The marker is written only after a
successful build whose inputs are unchanged, using an atomic replacement.
Desktop-release staging output does not require this marker.

## Source-mode port conflicts

Immediately before production process creation, the launcher scans the configured
API port. It aggregates listeners by unique PID, resolves a process name when
permitted, and shows each owning PID. An
interactive confirmation applies to the displayed set once, and only holders
identified as DILIGENT by repository-qualified command line are eligible:
an unknown owner is reported and left running, and launch aborts. A decline, a
noninteractive invocation, or a failed termination leaves processes untouched or
causes launch to abort; a final listener scan is authoritative if a process exits
concurrently. A replacement holder or any remaining listener is reported with
its current PID and port. The explicit `KillApplicationProcesses` action keeps
its narrower repository-qualified process-tree cleanup contract.

## Packaged desktop startup

The Windows portable executable and MSI use the Tauri shell. Open the verified portable EXE directly, or launch the application installed by the MSI; do not run the source launcher for packaged operation. The Angular build and startup splash are embedded as Tauri assets. On first launch the shell verifies the embedded backend runtime archive, extracts it to a versioned hash directory under `%LOCALAPPDATA%\DILIGENT\runtime`, creates persistent data directories under `%LOCALAPPDATA%\DILIGENT\data`, starts the packaged backend on a random localhost port, waits for its atomic ready file and `/api/health`, then loads the bundled SPA. The SPA connects to the loopback API with the one-time session token in a request header; the backend allows CORS only from the packaged Tauri origin. The backend is owned by a Windows Job Object and is asked to stop gracefully when the shell exits, with a bounded process termination fallback.

The packaged desktop does not use the source or development ports `7690` and `9847`. If the window does not appear, inspect `%LOCALAPPDATA%\DILIGENT\data\cache\logs\desktop-backend.log`, confirm that `state\desktop-backend-ready.json` exists, and request `/api/health` on the recorded port. A successful launch leaves `runtime\<version>\<payload-sha256>\extraction.complete` in place.

## Manual Development Startup

Run the backend and Angular development server in separate terminals. This is the development workflow; normal source production startup is handled by the launcher.

### Backend development server

```powershell
Set-Location app/server
./.venv/Scripts/python.exe -m uvicorn app:app --host 127.0.0.1 --port 7690 --reload --log-level info
```

### Angular development server

```powershell
Set-Location app/client
npm run dev
```

The development UI uses `UI_HOST` and `UI_PORT` (default `127.0.0.1:9847`). Its API proxy forwards same-origin `/api` requests to FastAPI.

## Quick Startup Checklist
### Source production mode
1. Run `.\start_on_windows.ps1` from an interactive PowerShell console.
2. Select **Launch application**. The launcher checks `FASTAPI_PORT` (default `7690`), starts FastAPI, and opens `http://127.0.0.1:7690` after health and built-asset checks pass.
3. Select **EXIT** or press Ctrl+C in the launcher to stop the app. `.\start_on_windows.ps1 -Action Launch` keeps the launcher supervising until Ctrl+C.

### Source development mode
1. Start the backend and Angular development server in separate terminals as described above.
2. Open `http://127.0.0.1:9847`; the Angular dev server proxies API requests to FastAPI.

Production launch checks only the API port. Development startup uses both the
API and `UI_PORT` (default `9847`). A redirected production launch with an
occupied API port cannot confirm termination and exits without stopping it.

### Packaged desktop mode
1. Open the portable EXE or launch the installed MSI application.
2. Confirm the desktop window appears with the title `DILIGENT Clinical Copilot`.
3. If startup fails, inspect the packaged log and ready file under `%LOCALAPPDATA%\DILIGENT\data`.
