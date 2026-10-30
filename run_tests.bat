@echo off
setlocal
cd /d "%~dp0"
echo Running Sepsis Early Warning Unit and Regression Tests...
.venv\Scripts\python.exe -m unittest discover tests -v
pause
