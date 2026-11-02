@echo off
setlocal
set "PY=%~dp0.venv-host\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "%~dp0python\tools\benchmark_kev.py" %*
exit /b %errorlevel%
