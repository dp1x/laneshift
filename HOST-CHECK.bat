@echo off
setlocal
set "PY=%~dp0.venv-host\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "%~dp0python\tools\host_check.py" %*
exit /b %errorlevel%
