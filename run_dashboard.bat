@echo off
title Health Pipeline Dashboard Launcher
cd /d "%~dp0"
echo ===================================================
echo   Starting Secure Health Pipeline API & Dashboard
echo ===================================================
echo.
echo URL: http://localhost:8000/ or http://127.0.0.1:8000/
echo.
start http://localhost:8000/
python scripts/run_api.py
pause
