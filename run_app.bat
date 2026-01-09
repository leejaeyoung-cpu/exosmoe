@echo off
chcp 65001 > nul
echo ========================================================
echo        🧬 Mela-Exosome AI Platform Launching...
echo ========================================================
echo.
echo Starting the dashboard...
echo Please wait while the browser opens.
echo.

cd /d "%~dp0"
streamlit run app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] An error occurred while running the app.
    echo Please check if 'streamlit' is installed.
    echo.
    pause
)
