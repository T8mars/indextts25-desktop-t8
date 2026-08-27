@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "HF_HOME=%CD%\checkpoints\hf_cache"

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: The local Python environment is missing.
  echo Wait for dependency installation to finish, then run this launcher again.
  pause
  exit /b 1
)

echo Starting IndexTTS 2.5 WebUI...
echo The browser will open when the server is ready.
echo Press Ctrl+C in this window to stop the server.

set "PORT=7860"
powershell.exe -NoProfile -Command "try { $r=Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:7860/gradio_api/info' -TimeoutSec 2; if ($r.StatusCode -eq 200 -and $r.Content -match 'lang_choice') { exit 0 }; exit 2 } catch { exit 1 }"
if "%ERRORLEVEL%"=="0" (
  echo IndexTTS 2.5 is already running. Opening the existing WebUI.
  start "" "http://127.0.0.1:7860"
  exit /b 0
)
if "%ERRORLEVEL%"=="2" (
  set "PORT=7861"
  echo Port 7860 is occupied by another program. Using port 7861.
)

start "IndexTTS browser helper" /b powershell.exe -NoProfile -WindowStyle Hidden -Command "$deadline=(Get-Date).AddMinutes(10); while ((Get-Date) -lt $deadline) { try { $r=Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:!PORT!' -TimeoutSec 2; if ($r.StatusCode -eq 200) { Start-Process 'http://127.0.0.1:!PORT!'; break } } catch {}; Start-Sleep -Seconds 2 }"

".venv\Scripts\python.exe" webui.py --version 2.5 --model_dir ".\checkpoints" --host 127.0.0.1 --port !PORT! --fp16
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
  echo.
  echo IndexTTS 2.5 stopped with exit code %EXITCODE%.
  pause
)
exit /b %EXITCODE%
