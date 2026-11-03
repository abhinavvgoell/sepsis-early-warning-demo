@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo      Starting Sepsis Early Warning Web Dashboard...
echo ============================================================
echo.

:: Detect Python executable
set "PYTHON_CMD="
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
) else if exist ".venv\python.exe" (
    set "PYTHON_CMD=.venv\python.exe"
) else (
    where py >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set "PYTHON_CMD=py"
    ) else (
        set "PYTHON_CMD=python"
    )
)

echo Using Python: !PYTHON_CMD!
echo Dashboard URL: http://127.0.0.1:8766
echo.
echo Launching web browser in 2 seconds...
echo (Keep this window open while using the dashboard. Press Ctrl+C to stop)
echo.

:: Launch browser in background after 2 seconds
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8766"

:: Run the server
!PYTHON_CMD! -m src.app %*

if !ERRORLEVEL! NEQ 0 (
    echo.
    echo [ERROR] Dashboard exited with error code !ERRORLEVEL!.
)
echo.
pause
