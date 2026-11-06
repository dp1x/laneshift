@echo off
setlocal
cd /d "%~dp0"

rem Prefer the Godot 4.7.2 runtime bundled inside the workspace.
set "GODOT=%~dp0..\tools\godot-4.7.2\Godot_v4.7.2-stable_windows_arm64.exe"
if exist "%GODOT%" goto :launch

rem Fall back to a Godot install on PATH.
where godot >nul 2>nul
if errorlevel 1 goto :missing
set "GODOT=godot"

:launch
echo LANESHIFT human recording
echo.
echo   A / LEFT    move left
echo   D / RIGHT   move right
echo   SPACE/DOWN  brake
echo   R           restart (same recording session, new seed)
echo   F9          stop and save the recording
echo   ESC         menu
echo.
echo Recording is already active. Close the window when finished.
echo.

"%GODOT%" --path "%~dp0." --windowed -- --record-on-start --no-controller
set "CODE=%errorlevel%"
if not "%CODE%"=="0" pause
exit /b %CODE%

:missing
echo Godot 4.7.2 was not found.
echo Expected the bundled runtime at:
echo   %~dp0..\tools\godot-4.7.2\Godot_v4.7.2-stable_windows_arm64.exe
echo Put Godot on PATH or restore that file, then run this again.
pause
exit /b 1
