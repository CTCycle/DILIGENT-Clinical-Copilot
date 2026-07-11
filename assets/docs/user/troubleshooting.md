# User Troubleshooting
Last updated: 2026-07-11

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
