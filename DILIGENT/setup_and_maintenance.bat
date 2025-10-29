@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM == Configuration: define project and Python paths
REM ============================================================================
set "project_folder=%~dp0"
set "root_folder=%project_folder%..\"
set "setup_dir=%project_folder%setup"
set "python_dir=%setup_dir%\python"
set "python_exe=%python_dir%\python.exe"
set "python_pth_file=%python_dir%\python312._pth"
set "env_marker=%python_dir%\.is_installed"

set "uv_dir=%setup_dir%\uv"
set "uv_exe=%uv_dir%\uv.exe"
set "uv_zip_path=%uv_dir%\uv.zip"
set "UV_CACHE_DIR=%setup_dir%\uv_cache"


:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Show setup menu
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:setup_menu
cls
echo ==========================================================================
echo                         Setup  and Maintenance                          
echo ==========================================================================
echo 1. Enable root path imports
echo 2. Update project
echo 3. Remove logs
echo 4. Exit
echo.
set /p sub_choice="Select an option (1-4): "

if "%sub_choice%"=="1" goto :eggs
if "%sub_choice%"=="2" goto :update
if "%sub_choice%"=="3" goto :logs
if "%sub_choice%"=="4" goto :exit
echo Invalid option, try again.
goto :setup_menu

:eggs
pushd "%root_folder%"
"%pip_exe%" install -e . --use-pep517 || popd
popd
goto :setup_menu

:update
echo Updating project... 
"%python_exe%" "%project_folder%setup\scripts\update_project.py"
pause
goto :setup_menu

:logs
pushd %log_path%
if not exist *.log (
    echo No log files found.
    popd
    pause
    goto :setup_menu
)

del *.log /q || popd
echo Log files deleted 
popd 
pause
goto :setup_menu


:exit
endlocal