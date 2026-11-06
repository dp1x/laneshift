@echo off
rem Double-click launcher for LANESHIFT with the original Kev-0.8B ONNX
rem checkpoint. Everything it uses lives inside C:\Users\Dhane\dev\cris@ds.
setlocal
set "ROOT=%~dp0"
set "PY=%~dp0..\.venv-host\Scripts\python.exe"
set "GODOT=%~dp0..\tools\godot-4.7.2\Godot_v4.7.2-stable_windows_arm64_console.exe"
set "MODELDIR=%~dp0models\kev-original"
set "MODEL=%MODELDIR%\kev07b_fp16_L256.onnx"

rem Keep Godot user data, caches and temporary files inside the workspace.
set "RUNTIME=%~dp0..\runtime-data"
set "APPDATA=%RUNTIME%\appdata\Roaming"
set "LOCALAPPDATA=%RUNTIME%\appdata\Local"
set "TEMP=%RUNTIME%\temp"
set "TMP=%RUNTIME%\temp"
if not exist "%APPDATA%" mkdir "%APPDATA%"
if not exist "%LOCALAPPDATA%" mkdir "%LOCALAPPDATA%"
if not exist "%TEMP%" mkdir "%TEMP%"

if not exist "%PY%" (
    echo Python environment not found: "%PY%"
    echo Run SETUP-HOST.bat or restore LANESHIFT\.venv-host.
    pause
    exit /b 1
)
if not exist "%GODOT%" (
    echo Godot 4.7.2 not found: "%GODOT%"
    pause
    exit /b 1
)
if not exist "%MODEL%" (
    echo Kev ONNX model not found: "%MODEL%"
    pause
    exit /b 1
)

echo Starting LANESHIFT with the original Kev-0.8B model...
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\play_kev.ps1" -Windowed -Python "%PY%" -Godot "%GODOT%" -Model "%MODEL%" -Tokenizer "%MODELDIR%\tokenizer.json" -Data "%MODELDIR%\kev07b_fp16.data" %*
set "CODE=%errorlevel%"
if not "%CODE%"=="0" pause
exit /b %CODE%
