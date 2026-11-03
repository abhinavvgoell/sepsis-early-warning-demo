@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo   Running Sepsis Early Warning Unit and Regression Tests...
echo ============================================================
echo.

set "PYTHON_CMD="
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
) else (
    set "PYTHON_CMD=python"
)

!PYTHON_CMD! -m unittest discover tests -v
set "TEST_CODE=%ERRORLEVEL%"

echo.
if !TEST_CODE! EQU 0 (
    echo [SUCCESS] All unit and regression tests passed successfully!
) else (
    echo [FAILURE] Tests finished with error code !TEST_CODE!.
)
echo.
pause
exit /b !TEST_CODE!
