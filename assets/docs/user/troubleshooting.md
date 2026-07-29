# User Troubleshooting
Last updated: 2026-07-29

## Windows desktop app does not open

- Confirm that you are opening the current `DILIGENT-v<version>-windows-x64-portable.exe` or launching the application installed by the matching MSI.
- Check **Event Viewer → Windows Logs → Application** for an error naming the portable executable.
- Inspect `%LOCALAPPDATA%\DILIGENT\data\resources\logs\desktop-backend.log` and `state\desktop-backend-ready.json`.
- If the runtime contains stale `.extract-*` directories, close DILIGENT and remove only those temporary directories under `%LOCALAPPDATA%\DILIGENT\runtime\<version>`, then retry.
- The portable app uses the system WebView2 runtime. An MSI built with the standard bootstrapper may need network access for WebView2; use an offline-WebView2 MSI when network access is unavailable.

Packaged desktop startup uses a random localhost backend port, so the development ports below are not a packaged health check. Read the port from `desktop-backend-ready.json` and request `/api/health` on that port.

## Browser Page Does Not Load
- Check whether the frontend is available at:

```text
http://127.0.0.1:9847
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

- If unreachable, restart the application and inspect backend console output.

## Model Call Fails
Check that:
- provider is selected
- model is selected
- required access key is saved and active
- local Ollama is running for local models
- network access is available for cloud providers
- provider quota or billing is available where applicable

## No Saved Sessions Appear
Check that:
- database initialization has been run
- the previous assessment completed successfully
- the application was not interrupted during save
- you are reviewing the correct local repository and database

## Data Inspection Is Empty
Check that:
- local resources exist under the expected resource directories
- database initialization completed successfully
- embedding or catalog update jobs completed successfully
- the backend was restarted after maintenance

## Ports Are Already In Use
Current defaults:

```text
Backend: 7690
Frontend: 9847
```

Close conflicting processes or update `settings/.env`, then restart the application.
