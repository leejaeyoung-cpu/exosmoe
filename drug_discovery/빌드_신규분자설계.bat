@echo off
REM De Novo 분자 설계 실행파일 빌드

echo ============================================================
echo miRNA 신규 분자 설계 - 실행파일 빌드
echo ============================================================
echo.

REM Python 확인
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python이 설치되어 있지 않습니다.
    pause
    exit /b 1
)

echo [1/3] 필수 패키지 설치 중...
python -m pip install --upgrade pip
python -m pip install pyinstaller pandas numpy plotly streamlit
echo.

echo [2/3] 실행파일 빌드 중... (수 분 소요)
pyinstaller --onefile ^
    --name DeNovo_MoleculeDesigner ^
    --add-data "denovo_molecule_generator.py;." ^
    --hidden-import pandas ^
    --hidden-import numpy ^
    --hidden-import plotly ^
    denovo_ui.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 빌드 실패!
    pause
    exit /b 1
)

echo.
echo [3/3] 빌드 완료!
echo.
echo ============================================================
echo 성공! 실행파일이 생성되었습니다
echo ============================================================
echo.
echo 실행파일 위치: dist\DeNovo_MoleculeDesigner.exe
echo.
echo ⚠️ 참고: Streamlit UI는 실행파일로 빌드하는 것보다
echo          배치 파일(신규분자설계_UI실행.bat)로 실행하는 것을 권장합니다.
echo.

pause
