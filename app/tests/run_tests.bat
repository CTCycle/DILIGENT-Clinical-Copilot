@echo off
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "PROJECT_ROOT=%%~fI"
set "APP_DIR=%PROJECT_ROOT%\app"
set "SERVER_DIR=%APP_DIR%\server"
set "CLIENT_DIR=%APP_DIR%\client"
set "TESTS_DIR=%APP_DIR%\tests"
set "SETTINGS_ENV=%PROJECT_ROOT%\settings\.env"
set "VENV_PYTHON=%SERVER_DIR%\.venv\Scripts\python.exe"
set "RUNTIME_NPM=%PROJECT_ROOT%\runtimes\nodejs\npm.cmd"

set "FASTAPI_HOST=127.0.0.1"
set "FASTAPI_PORT=8000"
set "UI_HOST=127.0.0.1"
set "UI_PORT=7861"
set "TEST_RESULT=0"
set "LIVE_SERVER_PHASE=SKIPPED"
set "PYTEST_PHASE=SKIPPED"
set "FRONTEND_BOOTSTRAP_PHASE=SKIPPED"
set "FRONTEND_UNIT_PHASE=SKIPPED"
set "FRONTEND_E2E_PHASE=SKIPPED"
set "STARTED_BACKEND=0"
set "STARTED_FRONTEND=0"

if exist "%SETTINGS_ENV%" (
  for /f "usebackq tokens=* delims=" %%L in ("%SETTINGS_ENV%") do (
    set "line=%%L"
    if not "!line!"=="" if "!line:~0,1!" NEQ "#" if "!line:~0,1!" NEQ ";" (
      for /f "tokens=1,* delims==" %%A in ("!line!") do (
        if /i "%%A"=="FASTAPI_HOST" set "FASTAPI_HOST=%%B"
        if /i "%%A"=="FASTAPI_PORT" set "FASTAPI_PORT=%%B"
        if /i "%%A"=="UI_HOST" set "UI_HOST=%%B"
        if /i "%%A"=="UI_PORT" set "UI_PORT=%%B"
      )
    )
  )
)

set "TEST_FASTAPI_HOST=%FASTAPI_HOST%"
set "TEST_UI_HOST=%UI_HOST%"
if /i "%TEST_FASTAPI_HOST%"=="0.0.0.0" set "TEST_FASTAPI_HOST=127.0.0.1"
if /i "%TEST_FASTAPI_HOST%"=="::" set "TEST_FASTAPI_HOST=127.0.0.1"
if /i "%TEST_UI_HOST%"=="0.0.0.0" set "TEST_UI_HOST=127.0.0.1"
if /i "%TEST_UI_HOST%"=="::" set "TEST_UI_HOST=127.0.0.1"

set "APP_TEST_BACKEND_URL=http://%TEST_FASTAPI_HOST%:%FASTAPI_PORT%"
set "APP_TEST_FRONTEND_URL=http://%TEST_UI_HOST%:%UI_PORT%"
set "API_BASE_URL=%APP_TEST_BACKEND_URL%"
set "UI_BASE_URL=%APP_TEST_FRONTEND_URL%"

if exist "%VENV_PYTHON%" (
  set "PYTHON_CMD=%VENV_PYTHON%"
) else (
  echo [ERROR] Missing backend venv: "%VENV_PYTHON%"
  exit /b 1
)

if exist "%RUNTIME_NPM%" (
  set "NPM_CMD=%RUNTIME_NPM%"
) else (
  set "NPM_CMD=npm"
)

set "UVICORN_APP=app:app"
set "BACKEND_WORKDIR=%SERVER_DIR%"
set "PYTHONPATH=%SERVER_DIR%;%APP_DIR%"
"%PYTHON_CMD%" -c "import importlib; importlib.import_module('app')" >nul 2>&1
if errorlevel 1 (
  set "UVICORN_APP=server.app:app"
  set "BACKEND_WORKDIR=%SERVER_DIR%"
  set "PYTHONPATH=%APP_DIR%"
)

set "PYTEST_TARGET=%TESTS_DIR%\unit"
if not "%STANDARD_TEST_PYTEST_TARGET%"=="" set "PYTEST_TARGET=%STANDARD_TEST_PYTEST_TARGET%"

set "NEED_FRONTEND=0"
echo %PYTEST_TARGET% | findstr /I "\\e2e" >nul 2>&1
if not errorlevel 1 set "NEED_FRONTEND=1"


echo.
echo ============================================================
echo  Standard Test Runner
echo ============================================================
echo [INFO] Project root: %PROJECT_ROOT%
echo [INFO] Backend URL : %APP_TEST_BACKEND_URL%
echo [INFO] Frontend URL: %APP_TEST_FRONTEND_URL%
echo [INFO] Target      : %PYTEST_TARGET%
echo.

set "LIVE_SERVER_PHASE=PASS"
curl -s --max-time 2 "%APP_TEST_BACKEND_URL%/docs" >nul 2>&1
if errorlevel 1 (
  echo [INFO] Starting backend server...
  start "" /B /D "%BACKEND_WORKDIR%" "%PYTHON_CMD%" -m uvicorn %UVICORN_APP% --host %FASTAPI_HOST% --port %FASTAPI_PORT% --log-level warning
  set "STARTED_BACKEND=1"
)

if "%NEED_FRONTEND%"=="1" (
  if not exist "%CLIENT_DIR%\node_modules" (
    echo [INFO] Installing frontend dependencies...
    call "%NPM_CMD%" --prefix "%CLIENT_DIR%" install
    if errorlevel 1 set "TEST_RESULT=1" & goto cleanup
  )
  echo [INFO] Starting frontend preview server...
  start "" /B /D "%CLIENT_DIR%" "%NPM_CMD%" run preview -- --host %UI_HOST% --port %UI_PORT%
  set "STARTED_FRONTEND=1"
)

set "ATTEMPTS=0"
:wait_loop
if %ATTEMPTS% geq 90 (
  set "LIVE_SERVER_PHASE=FAIL"
  set "TEST_RESULT=1"
  goto cleanup
)
curl -s --max-time 2 "%APP_TEST_BACKEND_URL%/docs" >nul 2>&1
if errorlevel 1 (
  set /a ATTEMPTS+=1
  timeout /t 1 /nobreak >nul
  goto wait_loop
)
if "%NEED_FRONTEND%"=="1" (
  curl -s --max-time 2 "%APP_TEST_FRONTEND_URL%" >nul 2>&1
  if errorlevel 1 (
    set /a ATTEMPTS+=1
    timeout /t 1 /nobreak >nul
    goto wait_loop
  )
)

echo [STEP] Running Python tests...
"%PYTHON_CMD%" -m pytest "%PYTEST_TARGET%" -v --tb=short %*
if errorlevel 1 (
  set "PYTEST_PHASE=FAIL"
  set "TEST_RESULT=1"
) else (
  set "PYTEST_PHASE=PASS"
)

:cleanup
if "%STARTED_BACKEND%"=="1" (
  for /f "tokens=5" %%P in ('netstat -ano ^| findstr LISTENING ^| findstr ":%FASTAPI_PORT%"') do taskkill /PID %%P /F >nul 2>&1
)
if "%STARTED_FRONTEND%"=="1" (
  for /f "tokens=5" %%P in ('netstat -ano ^| findstr LISTENING ^| findstr ":%UI_PORT%"') do taskkill /PID %%P /F >nul 2>&1
)

echo.
echo ============================================================
echo  Test Summary
echo ============================================================
echo  Live server phase   : %LIVE_SERVER_PHASE%
echo  Python tests        : %PYTEST_PHASE%
echo  Frontend bootstrap  : %FRONTEND_BOOTSTRAP_PHASE%
echo  Frontend unit tests : %FRONTEND_UNIT_PHASE%
echo  Frontend E2E tests  : %FRONTEND_E2E_PHASE%
echo ============================================================
echo.

exit /b %TEST_RESULT%
