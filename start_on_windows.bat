@echo off
where pwsh.exe >nul 2>nul
if errorlevel 1 (
  echo PowerShell 7 (pwsh.exe) is required to run DILIGENT.
  exit /b 1
)
pwsh.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_and_maintenance.ps1" -Launch
set "LAUNCHER_EXIT_CODE=%ERRORLEVEL%"
exit /b %LAUNCHER_EXIT_CODE%
