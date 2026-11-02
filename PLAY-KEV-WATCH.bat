@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\play_kev.ps1" -Watch %*
exit /b %errorlevel%
