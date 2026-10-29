# User Troubleshooting
Last updated: 2026-10-29

## Windows desktop app does not open

- Confirm that you are opening the current `DILIGENT-v<version>-windows-x64-portable.exe` or launching the application installed by the matching MSI.
- Check **Event Viewer → Windows Logs → Application** for an error naming the portable executable.
- Inspect `%LOCALAPPDATA%\DILIGENT\data\cache\logs\desktop-backend.log` and `state\desktop-backend-ready.json`.
- If the runtime contains stale `.extract-*` directories, close DILIGENT and remove only those temporary directories under `%LOCALAPPDATA%\DILIGENT\runtime\<version>`, then retry.
- The portable app uses the system WebView2 runtime. An MSI built with the standard bootstrapper may need network access for WebView2; use an offline-WebView2 MSI when network access is unavailable.

Packaged desktop startup uses a random localhost backend port, so source and development ports are not a packaged health check. Read the port from `desktop-backend-ready.json` and request `/api/health` on that port. Source production serves the SPA and API at the FastAPI origin (default `http://127.0.0.1:7690`). `UI_PORT` (default `9847`) is used only by `npm run dev`.

## Browser Page Does Not Load
- Check whether the frontend is available at:

```text
http://127.0.0.1:7690
```

- If not, restart with:

```text
start_on_windows.ps1
```

## Backend Health Check Fails
- Open:

```text
http://127.0.0.1:7690/api/health
```

- If unreachable, restart the application and inspect the backend output in the launcher console.
- If an earlier launch left stale application processes,
  run `start_on_windows.ps1 -Action KillApplicationProcesses`, confirm the
  cleanup, and launch again.

## Model Call Fails
Check that:
- provider is selected
- model is selected
- required access key is saved and active
- local Ollama is running for local models
- network access is available for cloud providers
- provider quota or billing is available where applicable

For a structured-output HTTP 400, read the complete warning. It identifies the
provider, model, operation, status, and a redacted upstream detail. For example:

```text
Check the provider connection, credentials, rate limits, or transient service status. Detail: opencode-go rejected deepseek-v4-flash during structured_output (HTTP 400): Model is unavailable.
```

This indicates a provider/model contract or availability response, not an
automatic model substitution. Confirm the provider catalog, credentials, quota,
and service status before retrying.

## Catalog Refresh Or Timeline Fallback

- Opening **Settings → Models** reuses the last saved provider catalog. Use
  **Refresh** when you intentionally want to contact the provider for a new
  listing.
- If refresh fails, review the reported failure class and confirm the active
  key, network access, provider availability, and endpoint before retrying.
- If timeline generation falls back, treat the resulting chronology as a
  navigation aid, review its warning, and retry after correcting the reported
  provider or model condition.

## No Saved Sessions Appear
Check that:
- database initialization has been run
- the previous assessment completed successfully
- the application was not interrupted during save
- you are reviewing the correct local repository and database

## Knowledge Is Empty
Check that:
- local data exists under the expected data directories
- database initialization completed successfully
- embedding or catalog update jobs completed successfully
- the backend was restarted after maintenance

## Drugs Are Unresolved Or Ambiguous
- **Every drug is Unresolved**: open **Knowledge → Terminology**. If RxNorm has
  never completed an import, run **Update all**.
- **DrugCentral import failed**: the public DrugCentral server may be
  unavailable or port 5433 may be blocked. Skip DrugCentral in the update
  dialog or point **Settings → Integrations** to another DrugCentral server.
- **A name shows Probable — review**: it was matched by spelling correction.
  Confirm it in **Knowledge → Terminology** to make it verified.
- **A name shows Ambiguous**: several drugs are close to it (for example
  hydroxyzine and hydralazine). Confirm the correct one; nothing is attached
  until you do.
- **A brand resolves to the wrong drug**: use **Reject mapping**, confirm the
  correct drug, and check **Open conflicts** for the source involved.
- **No LiverTox evidence for an identified drug**: LiverTox may cover it only
  in a class chapter, or not at all; the report lists it as missing a LiverTox
  link rather than guessing.

## Database Connection or Switch Failed

Open **Settings → Database** and use **Test connection** to check the selected
target. Confirm the SQLite file path or PostgreSQL host, port, database name,
username, and SSL requirements. The test action requires the PostgreSQL
database to exist. **Save and switch** can create a missing PostgreSQL database
only when the configured role has `CREATEDB` permission.

If a switch fails, the app keeps the previous database and its saved connection
settings. Reopen the Database page to confirm the active backend. Restart the
application only if the error says automatic recovery could not restore the
runtime services.

## Ports Are Already In Use
Current defaults:

```text
Production FastAPI and SPA: 7690
Angular development UI: 9847
```

The production launcher checks the API port only. Close a conflicting process
or update `FASTAPI_PORT` in `settings/.env`, then restart. `UI_PORT` matters
only when the Angular development server is running.
