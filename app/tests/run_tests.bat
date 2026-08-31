@echo off
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "PROJECT_ROOT=%%~fI"
set "APP_DIR=%PROJECT_ROOT%\app"
set "SERVER_DIR=%APP_DIR%\server"
set "CLIENT_DIR=%APP_DIR%\client"
set "TESTS_DIR=%APP_DIR%\tests"
set "RUNTIME_CACHE_DIR=%PROJECT_ROOT%\runtimes\cache"
set "CACHE_DIR=%TESTS_DIR%\cache"
if defined DILIGENT_TEST_CACHE_ROOT set "CACHE_DIR=%DILIGENT_TEST_CACHE_ROOT%"
set "PYTEST_CACHE_DIR=%CACHE_DIR%\pytest"
set "PYTEST_BASETEMP=%PYTEST_CACHE_DIR%\basetemp"
set "UV_CACHE_DIR=%RUNTIME_CACHE_DIR%\uv"
set "RUFF_CACHE_DIR=%CACHE_DIR%\ruff"
set "NPM_CONFIG_CACHE=%RUNTIME_CACHE_DIR%\npm"
set "PLAYWRIGHT_BROWSERS_PATH=%RUNTIME_CACHE_DIR%\playwright"
set "MYPY_CACHE_DIR=%CACHE_DIR%\mypy"
set "COVERAGE_FILE=%CACHE_DIR%\coverage\.coverage"
set "PIP_CACHE_DIR=%RUNTIME_CACHE_DIR%\pip"
set "PYTHONPYCACHEPREFIX=%RUNTIME_CACHE_DIR%\python"
set "SETTINGS_ENV=%PROJECT_ROOT%\settings\.env"
set "VENV_PYTHON=%SERVER_DIR%\.venv\Scripts\python.exe"
set "RUNTIME_NPM=%PROJECT_ROOT%\runtimes\nodejs\npm.cmd"

set "FASTAPI_HOST=127.0.0.1"
set "FASTAPI_PORT=8000"
set "UI_HOST=127.0.0.1"
set "UI_PORT=7861"
set "TEST_RESULT=0"
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

if not exist "%VENV_PYTHON%" (
  echo [ERROR] Missing backend venv: "%VENV_PYTHON%"
  exit /b 1
)
set "PYTHON_CMD=%VENV_PYTHON%"

if not exist "%RUNTIME_CACHE_DIR%" mkdir "%RUNTIME_CACHE_DIR%" >nul 2>&1
if not exist "%CACHE_DIR%" mkdir "%CACHE_DIR%" >nul 2>&1
set "PYTEST_TEMP_ARGS=--basetemp "%PYTEST_BASETEMP%" -o cache_dir="%PYTEST_CACHE_DIR%""

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
  echo [ERROR] Backend import check failed for app:app.
  exit /b 1
)

set "SUITE=%~1"
if "%SUITE%"=="" goto :choose_suite
shift
goto :suite_selected

:choose_suite
echo.
echo ============================================================
echo  Test Runner
echo ============================================================
echo 1. Unit tests
echo 2. Integration tests
echo 3. End-to-end tests
echo 4. Regression tests
echo 5. Stress tests
echo 6. All tests
echo 7. Model-config regression slice
echo 8. Model-config full pass
echo.
set /p SUITE="Select an option (1-8): "

if "%SUITE%"=="1" set "SUITE=unit"
if "%SUITE%"=="2" set "SUITE=integration"
if "%SUITE%"=="3" set "SUITE=e2e"
if "%SUITE%"=="4" set "SUITE=regression"
if "%SUITE%"=="5" set "SUITE=stress"
if "%SUITE%"=="6" set "SUITE=all"
if "%SUITE%"=="7" set "SUITE=modelconfig"
if "%SUITE%"=="8" set "SUITE=modelconfigfull"

:suite_selected
set "PYTEST_TARGET=%TESTS_DIR%\unit"
set "NEED_BACKEND=0"
set "NEED_FRONTEND=0"
set "EXTRA_PYTEST_ARGS="

if /i "%SUITE%"=="unit" goto :suite_done
if /i "%SUITE%"=="integration" (
  set "PYTEST_TARGET=%TESTS_DIR%\e2e"
  set "NEED_BACKEND=1"
  set "NEED_FRONTEND=1"
  goto :suite_done
)
if /i "%SUITE%"=="e2e" (
  set "PYTEST_TARGET=%TESTS_DIR%\e2e\test_app_flow.py"
  set "NEED_BACKEND=1"
  set "NEED_FRONTEND=1"
  goto :suite_done
)
if /i "%SUITE%"=="regression" (
  set "PYTEST_TARGET=%TESTS_DIR%\unit"
  set "EXTRA_PYTEST_ARGS=-k cloud_llm_native or native_embeddings or ollama_native"
  goto :suite_done
)
if /i "%SUITE%"=="stress" (
  set "PYTEST_TARGET=%TESTS_DIR%\e2e\test_rxnav_concurrency_diagnostic.py"
  set "NEED_BACKEND=1"
  goto :suite_done
)
if /i "%SUITE%"=="all" (
  set "PYTEST_TARGET=%TESTS_DIR%"
  set "NEED_BACKEND=1"
  set "NEED_FRONTEND=1"
  goto :suite_done
)
if /i "%SUITE%"=="modelconfig" (
  set "FASTAPI_PORT=7690"
  set "UI_PORT=9847"
  set "NEED_BACKEND=1"
  set "NEED_FRONTEND=1"
  set "MODELCONFIG_KIND=slice"
  goto :suite_done
)
if /i "%SUITE%"=="modelconfigfull" (
  set "FASTAPI_PORT=7690"
  set "UI_PORT=9847"
  set "NEED_BACKEND=1"
  set "NEED_FRONTEND=1"
  set "MODELCONFIG_KIND=full"
  goto :suite_done
)

echo [ERROR] Unknown suite: %SUITE%
exit /b 1

:suite_done
echo.
echo ============================================================
echo  Selected suite: %SUITE%
echo ============================================================
echo [INFO] Target      : %PYTEST_TARGET%
echo [INFO] Backend URL : %APP_TEST_BACKEND_URL%
echo [INFO] Frontend URL: %APP_TEST_FRONTEND_URL%
echo.

if defined MODELCONFIG_KIND (
  set "TIMESTAMP=%DATE:/=-%_%TIME::=-%"
  set "TIMESTAMP=!TIMESTAMP: =0!"

  set "RUN_DB=%PYTEST_CACHE_DIR%\modelconfig-runner.!TIMESTAMP!.db"
  set "PYTEST_CACHE=%PYTEST_CACHE_DIR%\modelconfig-cache.!TIMESTAMP!"
  set "PYTEST_BASETEMP=%PYTEST_CACHE_DIR%\modelconfig-basetemp.!TIMESTAMP!"
  set "DILIGENT_SQLITE_PATH=!RUN_DB!"
  set "PYTEST_TEMP_ARGS=--basetemp "!PYTEST_BASETEMP!" -o cache_dir="!PYTEST_CACHE!""
  if exist "!RUN_DB!" del /f "!RUN_DB!" >nul 2>&1
  if exist "!PYTEST_BASETEMP!" rmdir /s /q "!PYTEST_BASETEMP!" >nul 2>&1
)

if "%NEED_BACKEND%"=="1" (
  curl -s --max-time 2 "%APP_TEST_BACKEND_URL%/api/health" >nul 2>&1
  if errorlevel 1 (
    echo [INFO] Starting backend server...
    start "" /B /D "%BACKEND_WORKDIR%" "%PYTHON_CMD%" -m uvicorn %UVICORN_APP% --host %FASTAPI_HOST% --port %FASTAPI_PORT% --log-level warning
    set "STARTED_BACKEND=1"
  )
)

if "%NEED_BACKEND%"=="1" (
  set "BACKEND_READY=0"
  for /l %%I in (1,1,90) do (
    if "!BACKEND_READY!"=="0" (
      curl -s --max-time 2 "%APP_TEST_BACKEND_URL%/api/health" >nul 2>&1
      if not errorlevel 1 (
        set "BACKEND_READY=1"
      ) else (
        ping -n 2 127.0.0.1 >nul
      )
    )
  )
  if not "!BACKEND_READY!"=="1" set "TEST_RESULT=1" & goto cleanup
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

if "%NEED_FRONTEND%"=="1" (
  set "FRONTEND_READY=0"
  for /l %%I in (1,1,90) do (
    if "!FRONTEND_READY!"=="0" (
      curl -s --max-time 2 "%APP_TEST_FRONTEND_URL%" >nul 2>&1
      if not errorlevel 1 (
        set "FRONTEND_READY=1"
      ) else (
        ping -n 2 127.0.0.1 >nul
      )
    )
  )
  if not "!FRONTEND_READY!"=="1" set "TEST_RESULT=1" & goto cleanup
)

if defined MODELCONFIG_KIND (
  set "MC_UNIT=%TESTS_DIR%\unit\test_model_config_persistence.py"
  set "MC_API=%TESTS_DIR%\e2e\test_model_config_api.py"
  set "MC_FLOW=%TESTS_DIR%\e2e\test_app_flow.py"

  echo [STEP] Running model-config unit tests...
  "%PYTHON_CMD%" -m pytest "!MC_UNIT!" -q !PYTEST_TEMP_ARGS!
  if errorlevel 1 set "TEST_RESULT=1" & goto cleanup

  set "APP_TEST_BACKEND_URL=http://127.0.0.1:7690"
  set "APP_TEST_FRONTEND_URL=http://127.0.0.1:9847"

  if /i "%MODELCONFIG_KIND%"=="slice" (
    set "MC_K=runtime_toggle_enables_save_and_submits_put or model_config or dili_run_burst_click_submits_single_job or dili_run_conflict_surfaces_clear_error_message"
    echo [STEP] Running model-config API + app-flow e2e slice...
    "%PYTHON_CMD%" -m pytest "!MC_API!" "!MC_FLOW!" -k "!MC_K!" -q !PYTEST_TEMP_ARGS!
  ) else (
    echo [STEP] Running full model-config API + app-flow e2e suite...
    "%PYTHON_CMD%" -m pytest "!MC_API!" "!MC_FLOW!" -q !PYTEST_TEMP_ARGS!
  )
  if errorlevel 1 set "TEST_RESULT=1"
  goto cleanup
)

echo [STEP] Running pytest...
"%PYTHON_CMD%" -m pytest "%PYTEST_TARGET%" -v --tb=short %EXTRA_PYTEST_ARGS% !PYTEST_TEMP_ARGS!
if errorlevel 1 set "TEST_RESULT=1"

:cleanup
if "%STARTED_BACKEND%"=="1" (
  for /f "tokens=5" %%P in ('netstat -ano ^| findstr LISTENING ^| findstr ":%FASTAPI_PORT%"') do taskkill /PID %%P /F >nul 2>&1
)
if "%STARTED_FRONTEND%"=="1" (
  for /f "tokens=5" %%P in ('netstat -ano ^| findstr LISTENING ^| findstr ":%UI_PORT%"') do taskkill /PID %%P /F >nul 2>&1
)
if defined PYTEST_BASETEMP if exist "!PYTEST_BASETEMP!" rmdir /s /q "!PYTEST_BASETEMP!" >nul 2>&1
if defined PYTEST_CACHE if exist "!PYTEST_CACHE!" rmdir /s /q "!PYTEST_CACHE!" >nul 2>&1
if defined RUN_DB if exist "!RUN_DB!" del /f "!RUN_DB!" >nul 2>&1

exit /b %TEST_RESULT%
