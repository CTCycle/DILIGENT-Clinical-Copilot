# Runtime Troubleshooting
Last updated: 2026-06-03

## Scope
This file covers recurring local startup and launch failures.

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

3. Stop the stale process if needed:

```powershell
Stop-Process -Id <PID> -Force
```

4. Restart the backend.

## Backend Startup Is Inconsistent
### Symptom
Backend startup works only from some directories or only with specific commands.

### Reliable Command
From repository root:

```powershell
app/server/.venv/Scripts/python.exe -m uvicorn app:app --app-dir app --host 127.0.0.1 --port 7690 --log-level info
```
