# Runtime Troubleshooting
Last updated: 2026-07-29

## Scope
This file covers recurring local startup and launch failures.

## Portable Desktop Window Does Not Appear
### Symptom

Opening `releases\DILIGENT-v<version>-windows-x64-portable.exe` produces no visible window and no backend process.

### Checks

1. Check **Event Viewer → Windows Logs → Application** for an `Application Error` entry naming the portable executable.
2. Inspect `%LOCALAPPDATA%\DILIGENT\runtime\<version>` for a hash directory containing `extraction.complete`.
3. Inspect `%LOCALAPPDATA%\DILIGENT\data\resources\logs\desktop-backend.log` and `state\desktop-backend-ready.json`.

The packaged shell extracts the runtime before starting the backend. A stale `.extract-*` directory means extraction or validation was interrupted; close any running DILIGENT process and remove only those temporary directories under the matching version before retrying. Do not remove `%LOCALAPPDATA%\DILIGENT\data` unless a full user-data reset is intentional.

If Event Viewer reports exception `0xc00000fd` (stack overflow) from the portable executable, use a newly built artifact from the current release pipeline. This indicates an obsolete desktop binary, not a model provider or development-port failure.

## Packaged Backend Health Fails

The packaged backend uses a random localhost port rather than `7690`. Read the port from:

```powershell
$ready = Get-Content "$env:LOCALAPPDATA\DILIGENT\data\state\desktop-backend-ready.json" | ConvertFrom-Json
Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$($ready.port)/api/health"
```

If the ready file is absent, inspect the packaged backend log. If it is present but health fails, close the desktop application and retry after confirming that no stale `DILIGENTBackend.exe` remains.

## WebView2 Is Unavailable

The portable executable uses the system WebView2 runtime. The standard MSI uses the WebView2 bootstrapper and may require network access during installation. Maintainers can build an MSI with `-OfflineWebView2` when an offline installer is available. This setting applies only to MSI packaging, not the portable EXE.

## RAG Embedding Cache Is Missing or Invalid

The canonical runtime uses only the pinned Granite 97M multilingual ONNX artifact at `app/resources/models/embeddings/<revision>/onnx/model_quint8_avx2.onnx`. A missing cache requires network access followed by a RAG rebuild. A dependency error requires reinstalling the synchronized Python environment. For an invalid digest or incomplete snapshot, remove only that revision directory and rebuild. The runtime never substitutes another artifact or backend; disable RAG for the current assessment while repairing the cache.

## Expected Local Ports
- Backend: `127.0.0.1:7690`
- Frontend: `127.0.0.1:9847`

## Frontend Exits With Backend-unreachable Error
### Symptom
Frontend preview reports:

```text
Configured backend is unreachable at http://127.0.0.1:7690
```

### Cause
`app/client/scripts/preview-server.mjs` validates backend availability before serving the UI.

### Fix
1. Start the backend first.
2. Verify backend health:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:7690/docs
```

3. Launch frontend preview only after the backend responds:

```powershell
Set-Location app/client
npm run preview -- --host 127.0.0.1 --port 9847 --strictPort
```

## Backend Fails To Bind With WinError 10048
### Symptom

```text
[Errno 10048] ... only one usage of each socket address ... ('127.0.0.1', 7690)
```

### Cause
Port `7690` is already occupied, often by a stale Python process.

### Fix
1. Check the current listener:

```powershell
Get-NetTCPConnection -LocalPort 7690 | Select-Object LocalAddress,LocalPort,State,OwningProcess
```

2. Inspect the owning process:

```powershell
Get-Process -Id <PID>
```

3. If `http://127.0.0.1:7690/api/health` returns `200` and the owning process is the repository backend, reuse the existing server instead of starting a duplicate:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:7690/api/health
Get-Process -Id <PID> | Select-Object Id,ProcessName,Path
```

4. Stop the stale or unrelated process only when the health probe fails or the process is not the intended backend:

```powershell
Stop-Process -Id <PID> -Force
```

5. Restart the backend.

## Backend Startup Is Inconsistent
### Symptom
Backend startup works only from some directories or only with specific commands.

### Reliable Command
From repository root:

```powershell
app/server/.venv/Scripts/python.exe -m uvicorn app:app --app-dir app --host 127.0.0.1 --port 7690 --log-level info
```

## SQLite Fails With `disk I/O error`
### Symptom

```text
sqlite3.OperationalError: disk I/O error
```

This can occur on startup while loading `reference_catalog_entries`, or during direct `sqlite3` reads against `app/resources/database.db`.

### Cause
A stale `app/resources/database.db-journal` can remain after an interrupted write or failed process shutdown. If direct reads fail on `database.db` but a copied database file opens successfully, the live file and journal state are the likely cause.

### Fix
1. Confirm no backend process is using the embedded database:

```powershell
Get-Process -Name python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,Path
Get-NetTCPConnection -LocalPort 7690 -ErrorAction SilentlyContinue
```

2. Validate whether the database copy is readable before replacing anything:

```powershell
Copy-Item -LiteralPath app/resources/database.db -Destination assets/QAdatabase-recovery-check.db -Force
app/server/.venv/Scripts/python.exe -c "import sqlite3; c=sqlite3.connect('assets/QA/database-recovery-check.db'); print(c.execute('select count(*) from sqlite_master').fetchone()[0]); c.close()"
```

3. If a reset is acceptable, remove `app/resources/database.db` and `app/resources/database.db-journal`, then let startup recreate the embedded SQLite database or restore a known-readable copy.

4. Restart the backend and verify health:

```powershell
app/server/.venv/Scripts/python.exe -m uvicorn app:app --app-dir app --host 127.0.0.1 --port 7690 --log-level info
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:7690/api/health
```

## Angular Build Fails With `spawn EPERM`
### Symptom

```text
ng build
Building...
[FAILED: spawn EPERM]
```

### Cause
In this repository, Angular CLI child-process spawning can fail inside the Codex sandbox even when the project and dependencies are otherwise healthy.

### Fix
1. First confirm the code still type-checks from `app/client`:

```powershell
..\..\runtimes\nodejs\node.exe .\node_modules\typescript\bin\tsc -p .\tsconfig.app.json --noEmit
```

2. If the TypeScript check passes but `ng build` still fails with `spawn EPERM`, rerun the build outside the sandbox:

```powershell
..\..\runtimes\nodejs\npm.cmd run build
```

3. Treat this as an environment-execution issue, not an automatic signal that the Angular code is broken.

## Angular Dev Server Fails With `spawn EPERM`
### Symptom

```text
npm run dev
> node ./scripts/ng-serve.mjs
Building...
× Building... [FAILED: spawn EPERM]
An unhandled exception occurred: spawn EPERM
```

### Cause
`ng serve` can hit the same Codex sandbox child-process restriction as `ng build`.

### Fix
1. Keep the backend running and confirm it responds first:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:7690/api/health
```

2. Relaunch the frontend dev server outside the sandbox:

```powershell
Set-Location app/client
..\..\runtimes\nodejs\npm.cmd run dev
```

3. Verify the dev server is listening on the configured UI port:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:9847
```

4. Confirm browser-side API communication through the frontend origin:

```javascript
await fetch('/api/health').then(async (response) => ({
  ok: response.ok,
  status: response.status,
  body: await response.text(),
}));
```
