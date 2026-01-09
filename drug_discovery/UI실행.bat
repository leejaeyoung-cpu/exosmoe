@echo off
REM Streamlit UI 실행 배치 파일

echo ============================================================
echo AI 신약 발견 파이프라인 - UI 실행
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
streamlit run app_ui.py

pause
