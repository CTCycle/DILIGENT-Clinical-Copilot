@echo off
setlocal
cd /d "%~dp0"

set "SCRIPT=add_aesthetic_separators.py"

if not exist "%SCRIPT%" (
    echo ERROR: %SCRIPT% was not found in this folder.
    echo Place this .bat file in the same repo root folder as %SCRIPT%.
    pause
    exit /b 1
)

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set "PYTHON_CMD=python"
    ) else (
        echo ERROR: Python was not found.
        echo Install Python, or add it to PATH.
        pause
        exit /b 1
    )
)

echo.
echo Aesthetic separator tool
echo Current folder: %cd%
echo.
echo [D] Preview changes with diff
echo [W] Apply changes in place
echo [Q] Quit
echo.

set /p "CHOICE=Choose D, W, or Q: "

if /i "%CHOICE%"=="D" (
    %PYTHON_CMD% "%SCRIPT%" --diff
    echo.
    pause
    exit /b %errorlevel%
)

if /i "%CHOICE%"=="W" (
    %PYTHON_CMD% "%SCRIPT%" --write
    echo.
    pause
    exit /b %errorlevel%
)

if /i "%CHOICE%"=="Q" (
    exit /b 0
)

echo Invalid choice.
pause
exit /b 1
