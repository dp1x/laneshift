@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\play_kev.ps1" -Evaluation -Episodes 1 -EpisodeDuration 20 -Seeds 5 -Windowed %*
exit /b %errorlevel%
