@echo off
setlocal
cd /d "%~dp0"
if not exist "logs" mkdir "logs"
if not exist "recordings" mkdir "recordings"
if not exist "config" mkdir "config"
if not exist "transfer" mkdir "transfer"
"%~dp0LANESHIFT.exe" --rendering-method mobile --log-file "%~dp0logs\LANESHIFT-GPU.log" %*
if errorlevel 1 (
  echo.
  echo Mobile renderer failed; trying the compatibility renderer...
  "%~dp0LANESHIFT.exe" --rendering-method gl_compatibility --log-file "%~dp0logs\LANESHIFT-Compatibility.log" %*
)
exit /b %errorlevel%

