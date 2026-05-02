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

if not exist "%python_exe%" exit /b 1
if not exist "%uv_exe%" exit /b 1
if not exist "%npm_cmd%" exit /b 1
if not exist "%pyproject%" exit /b 1

echo [STEP 1/5] Installing dependencies with uv from pyproject.toml
pushd "%server_dir%" >nul
set "uv_extras="
if /i "%OPTIONAL_DEPENDENCIES%"=="true" set "uv_extras=--all-extras"
"%uv_exe%" sync --python "%python_exe%" %uv_extras%
if errorlevel 1 (
  popd >nul
  exit /b 1
)
popd >nul

if not exist "%client_dir%\node_modules" (
  echo [STEP 2/5] Installing frontend dependencies
  call "%npm_cmd%" --prefix "%client_dir%" install || exit /b 1
)

echo [STEP 3/5] Building frontend
call "%npm_cmd%" --prefix "%client_dir%" run build || exit /b 1

echo [STEP 4/5] Pruning uv cache
if exist "%runtimes_dir%\.uv-cache" rd /s /q "%runtimes_dir%\.uv-cache" >nul 2>&1

echo [STEP 5/5] Launching backend and frontend
start "" /b /D "%server_dir%" "%uv_exe%" run --project "%server_dir%" --no-sync --python "%python_exe%" python -m uvicorn %uvicorn_module% --host %FASTAPI_HOST% --port %FASTAPI_PORT% --log-level info
start "" /b /D "%client_dir%" "%npm_cmd%" run preview -- --host %UI_HOST% --port %UI_PORT% --strictPort
start "" "http://%UI_HOST%:%UI_PORT%"

endlocal & exit /b 0






