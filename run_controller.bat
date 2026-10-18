@echo off
setlocal
cd /d "%~dp0python"
python -c "import websockets" >nul 2>nul
if errorlevel 1 (
  echo Install Python dependencies with: python -m pip install -r requirements.txt
  exit /b 1
)
python server.py %*
