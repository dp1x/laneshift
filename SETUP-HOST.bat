@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_host.ps1" %*
exit /b %errorlevel%
