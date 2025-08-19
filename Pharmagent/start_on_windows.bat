@echo off
setlocal enableextensions enabledelayedexpansion

REM ============================================================================
REM == Activate the Conda environment
REM ============================================================================
call conda activate openkeras
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment "openkeras".
    exit /b 1
)

REM ============================================================================
REM == Paths
REM ============================================================================
set "SCRIPT_DIR=%~dp0"
set "DOTENV=%SCRIPT_DIR%setup\.env"

REM ============================================================================
REM == Load .env (key=value), ignore comments (# or ;) and blank lines
REM ============================================================================
if exist "%DOTENV%" (
    for /f "usebackq tokens=* delims=" %%L in ("%DOTENV%") do (
        set "line=%%L"
        if not "!line!"=="" if "!line:~0,1!" NEQ "#" if "!line:~0,1!" NEQ ";" (
            for /f "tokens=1* delims==" %%K in ("!line!") do (
                set "k=%%K"
                set "v=%%L"
                REM Trim surrounding single/double quotes if present
                if defined v (
                    if "!v:~0,1!"=="\"" set "v=!v:~1,-1!"
                    if "!v:~0,1!"=="'"  set "v=!v:~1,-1!"
                )
                set "!k!=!v!"
            )
        )
    )
) else (
    echo [WARN] .env not found at "%DOTENV%". Using defaults.
)

REM ============================================================================
REM == Defaults if not provided in .env
REM ============================================================================
if not defined FASTAPI_HOST  set "FASTAPI_HOST=127.0.0.1"
if not defined FASTAPI_PORT  set "FASTAPI_PORT=8000"
if not defined RELOAD set "RELOAD=true"

REM uvicorn target (aligns with your imports)
set "UVICORN_MODULE=Pharmagent.app.app:app"

REM Map RELOAD env to flag
set "RELOAD_FLAG="
if /i "%RELOAD%"=="true" set "RELOAD_FLAG=--reload"

set "DOCS_URL=http://%HOST%:%PORT%/docs"

echo [INFO] FASTAPI_HOST=%HOST%  FASTAPI_PORT=%PORT%  RELOAD=%RELOAD%
echo [INFO] Starting FastAPI app from %UVICORN_MODULE% ...
start "" "%DOCS_URL%"
uvicorn %UVICORN_MODULE% --host "%FASTAPI_HOST%" --port %FASTAPI_PORT% %RELOAD_FLAG% --log-level debug