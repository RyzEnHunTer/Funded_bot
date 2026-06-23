@echo off
title Antigravity Autopilot Engine
color 0B

echo ========================================================
echo   HUNTER PROP FIRM ENGINE - AUTOPILOT MODE
echo ========================================================
echo.
echo Press Ctrl+C at any time to stop the bot.
echo.

:loop
python scripts\run_headless.py

echo.
echo [!] Bot process ended or crashed.
echo [!] Restarting automatically in 10 seconds to maintain uptime...
timeout /t 10
goto loop
