@echo off
REM CKD-CVD AI 신약 발견 파이프라인 실행파일 빌더
REM 이 배치 파일을 더블클릭하면 자동으로 .exe 파일이 생성됩니다

echo ============================================================
echo AI 신약 발견 파이프라인 - 실행파일 빌드
echo ============================================================
echo.

REM Python 경로 확인
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python이 설치되어 있지 않거나 PATH에 없습니다.
    echo Python을 먼저 설치해주세요: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/4] Python 확인 완료
echo.

REM 필수 패키지 설치
echo [2/4] 필수 패키지 설치 중...
python -m pip install --upgrade pip
python -m pip install pyinstaller torch pandas numpy matplotlib seaborn requests
echo.

echo [3/4] 실행파일 빌드 중... (수 분 소요)
pyinstaller --onefile ^
    --name CKD_CVD_DrugDiscovery ^
    --add-data "data;data" ^
    --hidden-import torch ^
    --hidden-import pandas ^
    --hidden-import numpy ^
    --hidden-import matplotlib ^
    --hidden-import seaborn ^
    --hidden-import requests ^
    run_pipeline.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 빌드 실패!
    pause
    exit /b 1
)

echo.
echo [4/4] 빌드 완료!
echo.
echo ============================================================
echo 성공! 실행파일이 생성되었습니다
echo ============================================================
echo.
echo 실행파일 위치: dist\CKD_CVD_DrugDiscovery.exe
echo.
echo 사용 방법:
echo   1. dist 폴더로 이동
echo   2. CKD_CVD_DrugDiscovery.exe 더블클릭
echo.
echo 참고: 처음 실행시 Windows Defender가 차단할 수 있습니다.
echo      "추가 정보" - "실행" 클릭하여 허용하세요.
echo.

pause
