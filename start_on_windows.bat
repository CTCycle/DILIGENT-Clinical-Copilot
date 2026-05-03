@echo off
setlocal enabledelayedexpansion

set "root_folder=%~dp0"
set "app_dir=%root_folder%app"
set "server_dir=%app_dir%\server"
set "client_dir=%app_dir%\client"
set "settings_dir=%root_folder%settings"
set "runtimes_dir=%root_folder%runtimes"

set "python_exe=%runtimes_dir%\python\python.exe"
set "uv_exe=%runtimes_dir%\uv\uv.exe"
set "npm_cmd=%runtimes_dir%\nodejs\npm.cmd"
set "pyproject=%server_dir%\pyproject.toml"
set "uvicorn_module=app:app"
set "dotenv=%settings_dir%\.env"
set "uv_cache_dir=%runtimes_dir%\.uv-cache"
set "frontend_dist=%client_dir%\dist"

set "FASTAPI_HOST=127.0.0.1"
set "FASTAPI_PORT=8000"
set "UI_HOST=127.0.0.1"
set "UI_PORT=7861"
set "RELOAD=false"
set "OPTIONAL_DEPENDENCIES=false"

if exist "%dotenv%" (
  for /f "usebackq tokens=* delims=" %%L in ("%dotenv%") do (
    set "line=%%L"
    if not "!line!"=="" if "!line:~0,1!" NEQ "#" if "!line:~0,1!" NEQ ";" (
      for /f "tokens=1,* delims==" %%A in ("!line!") do set "%%A=%%B"
    )
  )
)

if not exist "%python_exe%" ( echo [FATAL] Missing python runtime: %python_exe% & exit /b 1 )
if not exist "%uv_exe%" ( echo [FATAL] Missing uv runtime: %uv_exe% & exit /b 1 )
if not exist "%npm_cmd%" ( echo [FATAL] Missing npm runtime: %npm_cmd% & exit /b 1 )
if not exist "%pyproject%" ( echo [FATAL] Missing pyproject: %pyproject% & exit /b 1 )

echo [STEP 1/5] Installing dependencies with uv from pyproject.toml
pushd "%server_dir%" >nul
set "uv_extras="
if /i "%OPTIONAL_DEPENDENCIES%"=="true" set "uv_extras=--all-extras"
"%uv_exe%" sync --python "%python_exe%" %uv_extras%
if errorlevel 1 (
  echo [WARN] uv sync with embedded Python failed. Falling back to uv-managed Python.
  "%uv_exe%" sync %uv_extras%
  if errorlevel 1 ( popd >nul & echo [FATAL] uv sync failed. & exit /b 1 )
)
popd >nul

echo [STEP 2/5] Installing frontend dependencies
if not exist "%client_dir%\node_modules" (
  pushd "%client_dir%" >nul
  if exist "package-lock.json" ( call "%npm_cmd%" ci ) else ( call "%npm_cmd%" install )
  if errorlevel 1 ( popd >nul & echo [FATAL] Frontend dependency install failed. & exit /b 1 )
  popd >nul
)

echo [STEP 3/5] Building frontend
if not exist "%frontend_dist%" (
  pushd "%client_dir%" >nul
  call "%npm_cmd%" run build
  if errorlevel 1 ( popd >nul & echo [FATAL] Frontend build failed. & exit /b 1 )
  popd >nul
)

echo [STEP 4/5] Pruning uv cache
if exist "%uv_cache_dir%" rd /s /q "%uv_cache_dir%" >nul 2>&1

echo [STEP 5/5] Launching backend and frontend
call :kill_port %FASTAPI_PORT%
pushd "%server_dir%" >nul
start "" /b "%uv_exe%" run --project "%server_dir%" --no-sync --python "%python_exe%" python -m uvicorn %uvicorn_module% --host %FASTAPI_HOST% --port %FASTAPI_PORT% --log-level info
popd >nul

set "BACKEND_BASE_URL=http://%FASTAPI_HOST%:%FASTAPI_PORT%"
echo [WAIT] Waiting for backend readiness at %BACKEND_BASE_URL%...
for /L %%i in (1,1,60) do (
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$base='%BACKEND_BASE_URL%'; $paths=@('/api/health','/health','/api/model-config','/docs','/'); foreach ($p in $paths) { try { $r = Invoke-WebRequest -UseBasicParsing -Uri ($base + $p) -TimeoutSec 2; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) { exit 0 } } catch {} }; exit 1" >nul 2>&1
  if !errorlevel! equ 0 goto :backend_ready
  timeout /t 1 /nobreak >nul 2>&1
)
echo [WARN] Backend did not pass readiness checks in time. Continuing with frontend launch.

:backend_ready
pushd "%client_dir%" >nul
call :kill_port %UI_PORT%
start "" /b "%npm_cmd%" run preview -- --host %UI_HOST% --port %UI_PORT% --strictPort
popd >nul
start "" "http://%UI_HOST%:%UI_PORT%"

echo [SUCCESS] Backend and frontend launch command dispatched.
endlocal & exit /b 0

:kill_port
set "target_port=%~1"
if not defined target_port goto :eof
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R ":%target_port%"') do taskkill /PID %%P /F >nul 2>&1
goto :eof
