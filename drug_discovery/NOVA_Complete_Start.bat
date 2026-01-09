@echo off
REM NOVA Complete System Launcher
REM Launches both De Novo and In Silico Validation
REM Version 1.0

echo ========================================
echo  NOVA Complete Drug Discovery System
echo ========================================
echo.
echo  This will launch TWO apps:
echo  1. De Novo Designer (Port 8502)
echo  2. In Silico Validation (Port 8503)
echo.
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
    pip install streamlit pandas plotly numpy rdkit torch scikit-learn stmol py3Dmol matplotlib seaborn ipython_genutils
    if errorlevel 1 (
        echo ERROR: Failed to install packages
        pause
        exit /b 1
    )
)
echo All packages ready!
echo.

echo [3/3] Starting NOVA Complete System...
echo.
echo ========================================
echo  Servers will start on:
echo  - De Novo: http://localhost:8502
echo  - In Silico: http://localhost:8503
echo ========================================
echo.
echo Opening browsers...
echo.

REM Start De Novo in background
start /B streamlit run denovo_ui.py --server.port 8502

REM Wait 3 seconds
timeout /t 3 /nobreak >nul

REM Start In Silico Validation in background
start /B streamlit run nova_insilico_validation_ui.py --server.port 8503

REM Wait 3 seconds
timeout /t 3 /nobreak >nul

REM Open browsers
start http://localhost:8502
start http://localhost:8503

echo.
echo ========================================
echo  Both systems are now running!
echo  Press any key to STOP all servers
echo ========================================
echo.
pause >nul

REM Kill all Streamlit processes
echo Stopping servers...
taskkill /F /IM streamlit.exe >nul 2>&1
echo Servers stopped.
pause
