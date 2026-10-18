@echo off
setlocal
cd /d "%~dp0"
python python\experiment_runner.py --project "%~dp0." %*
