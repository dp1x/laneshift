@echo off
setlocal
cd /d "%~dp0"
godot --path . -- --record-on-start --no-controller
exit /b %errorlevel%
