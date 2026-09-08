@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title T8star-Aix IndexTTS 2.5 - Stop API

set "T8_RESOURCE_ROOT=%~dp0resources"
set "T8_PYTHON="
for /d %%D in ("%T8_RESOURCE_ROOT%\cpython-*") do (
  if not defined T8_PYTHON if exist "%%D\python.exe" set "T8_PYTHON=%%D\python.exe"
)
if not defined T8_PYTHON (
  echo [ERROR] Bundled Python was not found.
  pause
  exit /b 1
)
set "PYTHONUTF8=1"
set "PYTHONPATH=%T8_RESOURCE_ROOT%;%T8_RESOURCE_ROOT%\site-packages"
"%T8_PYTHON%" -u "%T8_RESOURCE_ROOT%\desktop_api_launcher.py" stop
echo.
pause
