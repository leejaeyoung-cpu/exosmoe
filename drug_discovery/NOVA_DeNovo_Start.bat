@echo off
REM NOVA De Novo Drug Design System Launcher
REM Version 1.0

echo ========================================
echo  NOVA De Novo Drug Design System
echo  AI-Powered Molecule Generation
echo ========================================
echo.

REM Change to drug_discovery directory
cd /d "%~dp0"

echo [1/3] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)
echo Python found!
echo.

echo [2/3] Checking required packages...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install streamlit pandas plotly numpy rdkit stmol py3Dmol matplotlib
    if errorlevel 1 (
        echo ERROR: Failed to install packages
        pause
        exit /b 1
    )
)
echo All packages ready!
echo.

echo [3/3] Starting NOVA De Novo Designer...
echo.
echo ========================================
echo  Server will start on:
echo  http://localhost:8502
echo ========================================
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start Streamlit
streamlit run denovo_ui.py --server.port 8502

pause
