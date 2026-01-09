@echo off
echo ===================================================
echo MI-EXO Lite 실행 중...
echo 잠시만 기다려주세요. 브라우저가 자동으로 열립니다.
echo ===================================================
cd /d "%~dp0"
streamlit run simple_app.py
pause
