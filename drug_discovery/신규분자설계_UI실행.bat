@echo off
REM miRNA 신규 분자 설계 UI 실행

echo ============================================================
echo miRNA 기반 신규 분자 설계 - UI 실행
echo ============================================================
echo.

REM Streamlit 설치 확인
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo Streamlit이 설치되어 있지 않습니다.
    echo 설치 중...
    python -m pip install streamlit plotly
)

echo.
echo 웹 브라우저가 자동으로 열립니다...
echo 종료하려면 Ctrl+C를 누르세요.
echo.

REM Streamlit 실행
streamlit run denovo_ui.py --server.port 8502

pause
