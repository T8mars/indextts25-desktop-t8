@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title T8star-Aix IndexTTS 2.5 - API Service

set "T8_RESOURCE_ROOT=%~dp0resources"
set "T8_PYTHON="
for /d %%D in ("%T8_RESOURCE_ROOT%\cpython-*") do (
  if not defined T8_PYTHON if exist "%%D\python.exe" set "T8_PYTHON=%%D\python.exe"
)

if not defined T8_PYTHON (
  echo [ERROR] Bundled Python was not found under resources\cpython-*.
  echo Keep this script beside the portable EXE and the resources folder.
  pause
  exit /b 1
)

set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "PYTHONPATH=%T8_RESOURCE_ROOT%;%T8_RESOURCE_ROOT%\site-packages"
"%T8_PYTHON%" -u "%T8_RESOURCE_ROOT%\desktop_api_launcher.py" serve
set "T8_EXIT_CODE=%ERRORLEVEL%"

if not "%T8_EXIT_CODE%"=="0" echo [ERROR] API service exited with code %T8_EXIT_CODE%.
echo.
echo Press any key to close this window.
pause >nul
exit /b %T8_EXIT_CODE%
