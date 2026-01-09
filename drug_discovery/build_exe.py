"""
AI 신약 발견 파이프라인 실행파일 빌드 스크립트

PyInstaller를 사용하여 .exe 파일 생성
"""

import subprocess
import sys
from pathlib import Path

def install_pyinstaller():
    """PyInstaller 설치"""
    print("🔧 PyInstaller 설치 중...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✅ PyInstaller 설치 완료")
        return True
    except Exception as e:
        print(f"❌ 설치 실패: {e}")
        return False

def build_executable():
    """실행파일 빌드"""
    print("\n🏗️ 실행파일 빌드 시작...")
    
    # PyInstaller 옵션
    options = [
        'pyinstaller',
        '--onefile',  # 단일 실행파일
        '--windowed',  # 콘솔 창 숨기기 (원하면 제거)
        '--name=CKD_CVD_DrugDiscovery',  # 실행파일 이름
        '--icon=NONE',  # 아이콘 (있으면 경로 지정)
        '--add-data=data;data',  # 데이터 폴더 포함
        '--hidden-import=torch',
        '--hidden-import=pandas',
        '--hidden-import=numpy',
        '--hidden-import=matplotlib',
        '--hidden-import=seaborn',
        '--hidden-import=requests',
        'run_pipeline.py'  # 메인 스크립트
    ]
    
    try:
        subprocess.check_call(options)
        print("\n✅ 빌드 완료!")
        print("📁 실행파일 위치: dist/CKD_CVD_DrugDiscovery.exe")
        return True
    except Exception as e:
        print(f"❌ 빌드 실패: {e}")
        return False

def main():
    print("="*70)
    print("AI 신약 발견 파이프라인 실행파일 빌더")
    print("="*70)
    
    # PyInstaller 설치
    if not install_pyinstaller():
        return
    
    # 빌드
    if build_executable():
        print("\n" + "="*70)
        print("🎉 실행파일 생성 완료!")
        print("="*70)
        print("\n실행 방법:")
        print("  1. dist 폴더로 이동")
        print("  2. CKD_CVD_DrugDiscovery.exe 더블클릭")
        print("\n주의: 첫 실행은 시간이 걸릴 수 있습니다.")

if __name__ == "__main__":
    main()
