@echo off
setlocal
cd /d "%~dp0python"
python -m unittest discover -s tests -v
if errorlevel 1 exit /b 1
cd /d "%~dp0"
godot --headless --path "%~dp0." --editor --quit
if errorlevel 1 exit /b 1
godot --headless --path "%~dp0." --script res://tests/simulation_test.gd
if errorlevel 1 exit /b 1
godot --headless --path "%~dp0." --script res://tests/game_flow_test.gd
if errorlevel 1 exit /b 1
godot --headless --path "%~dp0." --script res://tests/network_test.gd
if errorlevel 1 exit /b 1
godot --headless --path "%~dp0." --script res://tests/telemetry_test.gd
if errorlevel 1 exit /b 1
godot --headless --path "%~dp0." --script res://tests/options_test.gd
exit /b %errorlevel%
