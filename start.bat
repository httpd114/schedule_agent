@echo off
chcp 65001 >nul
title Schedule Agent - One Click Start
cd /d "%~dp0"

set "PY=C:\Users\456\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo Starting Schedule Agent...
echo Browser will open automatically.
echo Close this window to stop the server.
echo.
"%PY%" server.py
echo.
echo Server stopped.
pause
