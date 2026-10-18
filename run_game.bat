@echo off
setlocal
cd /d "%~dp0"
where godot >nul 2>nul
if errorlevel 1 (
  echo Godot 4.7.2 was not found on PATH.
  exit /b 1
)
godot --path "%~dp0." %*
