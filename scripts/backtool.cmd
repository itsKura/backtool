@echo off
rem Start Backtool and open it in a browser. Target of the desktop shortcut.
rem
rem %~dp0 is this file's own folder, so the shortcut keeps working if the
rem repository is moved or cloned somewhere else -- nothing here is tied to
rem one machine's paths.

cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo Could not find .venv\Scripts\python.exe in:
  echo   %CD%
  echo.
  echo Create the environment first:
  echo   python -m venv .venv
  echo   .venv\Scripts\pip install -e ".[dev]"
  echo.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" "scripts\launch.py" %*

rem Held open on a crash so the traceback is readable rather than a window
rem that blinks and vanishes.
if errorlevel 1 (
  echo.
  echo Backtool exited with an error.
  pause
)
