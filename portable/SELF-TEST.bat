@echo off
setlocal
cd /d "%~dp0"
echo LANESHIFT 2D self-test
echo.
set FAIL=0
if not exist "LANESHIFT.exe" (
  echo FAIL: LANESHIFT.exe missing
  set FAIL=1
) else (
  echo [1/2] executable OK
)
if not exist "recordings" mkdir "recordings"
"%~dp0LANESHIFT.exe" --headless -- --auto-run --seeds=1 --episode-duration=0.5 --max-episodes=1 --ws-port=8799
if errorlevel 1 set FAIL=1
if exist "recordings\session_*" (
  echo [2/2] recording session created
) else (
  echo FAIL: no recording session created
  set FAIL=1
)
echo.
if "%FAIL%"=="0" (echo SELF-TEST PASS) else (echo SELF-TEST FAIL)
echo.
pause
exit /b %FAIL%
