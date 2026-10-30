@echo off
setlocal
cd /d "%~dp0"
echo Starting Sepsis Early Warning Web Dashboard...
echo Open http://127.0.0.1:8766 in your browser.
.venv\Scripts\python.exe -m src.app
pause
