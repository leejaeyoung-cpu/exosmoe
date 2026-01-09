@echo off
REM NOVA De Novo Designer v2.0 Launcher
REM Learning-based Generation with IND Gate Constraints

echo ========================================
echo  NOVA De Novo Designer v2.0
echo  Learning from Candidate 1
echo ========================================
echo.

REM Change to drug_discovery directory
cd /d "%~dp0"

echo [1/2] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed
    pause
    exit /b 1
)
echo Python found!
echo.

echo [2/2] Starting NOVA De Novo v2.0...
echo.
echo ========================================
echo  Server will start on:
echo  http://localhost:8504
echo.
echo  Features:
echo  - Candidate 1 learning
echo  - EGFR selectivity constraints
echo  - IND Gate filtering
echo  - Scaffold diversity
echo ========================================
echo.
echo Press Ctrl+C to stop
echo.

REM Start Streamlit
streamlit run denovo_v2_ui.py --server.port 8504

pause
