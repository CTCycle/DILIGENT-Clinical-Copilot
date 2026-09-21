# Startup
Last updated: 2026-09-22

## Recommended Local Startup
On Windows, use:

```powershell
.\start_on_windows.ps1
```

The launcher:
- creates `settings/.env` from `settings/.env.example` on first application start when the local environment file is missing
- ensures portable Python and Node runtimes for a warm launch; `uv` is prepared when backend/runtime repair or an explicit install requires it
- runs `uv sync --locked` against the tracked `app/server/uv.lock` during backend/runtime repair and explicit installation
- installs frontend dependencies only when an explicit install/rebuild or a stale/missing frontend build requires them
- rebuilds the frontend when the main-menu install option 2 or frontend rebuild option 3 is executed, or when option 1 detects missing or stale source-mode build output
- validates the deterministic frontend build-state marker before reusing an existing production build
- starts the backend with the synchronized virtual-environment Python and `uvicorn`
- opens a visible dedicated backend terminal for every source-mode launch
- starts the frontend preview server
- recreates a stale backend virtual environment when the repository has moved
- provides grouped `APPLICATION`, `SETUP & VALIDATION`, `SOURCE CONTROL`, `BUILD & DISTRIBUTION`, and `DATA & MAINTENANCE` options, followed by a final sequential `EXIT` option; the desktop-release submenu uses the same aligned numeric rows
- checks both configured source-mode ports immediately before starting application processes; every owning PID is listed, including foreign listeners
- prompts once before directly stopping the listed unique port holders; redirected or declined launches fail safely without terminating anything

All source-mode disposable runtime, application, test, and tool state is rooted
at `runtimes/cache/`. The launcher routes uv, pip, npm, Playwright, Python
bytecode, Cargo, pytest, Ruff, Mypy, Angular, coverage, logs, embeddings, test
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
- Install option 2 runs the same database synchronization after backend
  and frontend dependencies are ready. Launch performs the check again so
  startup remains safe when installation was skipped.

For a deterministic disposable-state reset, run `.\start_on_windows.ps1 -Action ClearCache`; it requires the same interactive `[y/N]` confirmation, preserves only `runtimes/cache/.gitkeep`, and reports locked entries without aborting the cleanup.

If a previous launch left backend, frontend, or launcher wrapper processes running,
run `.\start_on_windows.ps1 -Action KillApplicationProcesses` and confirm the
cleanup when prompted.

## Source-mode frontend build freshness

Source-mode production output is accompanied by
`app/client/dist/.diligent-build-state.json`. The marker contains a schema
version, source/build fingerprint, dependency fingerprint, pinned Node version,
and the canonical `npm run build` command. The launcher reuses the output only
when the marker and `dist/browser/index.html` are present and all fingerprints
still match.

The build fingerprint includes sorted, normalized paths and content digests for
production files under `app/client/src/` (excluding `*.spec.ts`),
`app/client/public/**`, the frontend manifests/configuration files, the pinned
Node version, and the build command. It does not include test configuration,
preview/dev-server scripts, `settings/.env`, backend Python, QA files, or
documentation. A missing, corrupt, stale, or absent output marker triggers one
source-mode rebuild. Dependency-manifest or pinned-Node changes additionally
run `npm ci` from the tracked lockfile before that build. The marker is written
only after a successful build whose inputs are unchanged, using an atomic
replacement; desktop-release staging output does not require this marker.

## Source-mode port conflicts

Immediately before application process creation, the launcher performs one listener
scan for both configured ports. It aggregates listeners by unique PID, resolves a process
name when permitted, and shows one line per process with all ports it owns. An
interactive confirmation applies to the displayed set once. A decline, a
noninteractive invocation, or a failed termination leaves processes untouched or
causes launch to abort; a final listener scan is authoritative if a process exits
concurrently. A replacement holder or any remaining listener is reported with
its current PID and port. The explicit `KillApplicationProcesses` action keeps
its narrower repository-qualified process-tree cleanup contract.

## Packaged desktop startup

The Windows portable executable and MSI use the Tauri shell. Open the verified portable EXE directly, or launch the application installed by the MSI; do not run the source launcher for packaged operation. On first launch the shell verifies the embedded runtime archive, extracts it to a versioned hash directory under `%LOCALAPPDATA%\DILIGENT\runtime`, creates persistent data directories under `%LOCALAPPDATA%\DILIGENT\data`, starts the packaged backend on a random localhost port, waits for its atomic ready file and `/api/health`, and then shows the desktop window. The backend is owned by a Windows Job Object and is terminated when the shell exits.

The packaged desktop does not use the development ports `7690` and `9847`. If the window does not appear, inspect `%LOCALAPPDATA%\DILIGENT\data\cache\logs\desktop-backend.log`, confirm that `state\desktop-backend-ready.json` exists, and request `/api/health` on the recorded port. A successful launch leaves `runtime\<version>\<payload-sha256>\extraction.complete` in place.

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
1. Confirm ports `7690` and `9847` are free, or start the launcher from an interactive PowerShell console so it can display and request termination of any configured-port holders.
2. Start the backend.
3. Verify `http://127.0.0.1:7690/docs` responds.
4. Start the frontend on `9847`.
5. Open `http://127.0.0.1:9847`.

If any configured port is occupied during an interactive launch, the launcher
lists the unique owning PIDs and asks once before stopping those PIDs directly.
A launch from a redirected or noninteractive console cannot confirm that action
and exits without terminating anything. If the prompt is declined, use the
explicit `KillApplicationProcesses` action for stale repository-owned process
trees or stop the listed service yourself, then retry.

### Packaged desktop mode
1. Open the portable EXE or launch the installed MSI application.
2. Confirm the desktop window appears with the title `DILIGENT Clinical Copilot`.
3. If startup fails, inspect the packaged log and ready file under `%LOCALAPPDATA%\DILIGENT\data`.
