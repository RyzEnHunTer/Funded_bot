@echo off
title Antigravity Bot Installer
color 0A

echo ========================================================
echo   HUNTER PROP FIRM BOT - ONE-CLICK INSTALLER
echo ========================================================
echo.
echo Installing all required Python dependencies...
echo.

python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ========================================================
echo   INSTALLATION COMPLETE!
echo ========================================================
echo.
echo You can now double-click "start_bot.bat" to launch the bot!
echo.
pause
