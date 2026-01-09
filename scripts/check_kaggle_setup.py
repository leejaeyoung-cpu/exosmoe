import os
import json
from pathlib import Path

def check_kaggle_setup():
    """Kaggle API 설정 체크"""
    
    print("\n" + "="*80)
    print("🔍 Kaggle API 설정 확인")
    print("="*80 + "\n")
    
    # 1. Kaggle 패키지 확인
    try:
        import kaggle
        print("✅ Kaggle 패키지 설치됨")
    except ImportError:
        print("❌ Kaggle 패키지 미설치")
        print("   설치: pip install kaggle")
        return False
    
    # 2. .kaggle 디렉토리 확인
    kaggle_dir = Path.home() / '.kaggle'
    
    if kaggle_dir.exists():
        print(f"✅ .kaggle 디렉토리 존재: {kaggle_dir}")
    else:
        print(f"❌ .kaggle 디렉토리 없음: {kaggle_dir}")
        print("   생성 중...")
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 디렉토리 생성 완료: {kaggle_dir}")
    
    # 3. kaggle.json 파일 확인
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if kaggle_json.exists():
        print(f"✅ kaggle.json 파일 존재: {kaggle_json}")
        
        # 파일 내용 확인 (민감정보이므로 키는 숨김)
        try:
            with open(kaggle_json, 'r') as f:
                config = json.load(f)
                username = config.get('username', '???')
                has_key = 'key' in config
                
                print(f"   Username: {username}")
                print(f"   API Key: {'✅ 설정됨' if has_key else '❌ 없음'}")
        except Exception as e:
            print(f"   ⚠️  파일 읽기 오류: {e}")
    else:
        print(f"❌ kaggle.json 파일 없음: {kaggle_json}")
        print("\n📝 다음 단계:")
        print("   1. https://www.kaggle.com/settings/account 접속")
        print("   2. 'Create New API Token' 클릭")
        print(f"   3. 다운로드한 kaggle.json을 {kaggle_dir}로 이동")
        return False
    
    # 4. API 인증 테스트
    print("\n🔐 API 인증 테스트...")
    try:
        kaggle.api.authenticate()
        print("✅ API 인증 성공!")
    except Exception as e:
        print(f"❌ API 인증 실패: {e}")
        return False
    
    # 5. 데이터셋 조회 테스트
    print("\n📦 데이터셋 조회 테스트...")
    try:
        datasets = list(kaggle.api.dataset_list(search='cell'))[:3]
        print(f"✅ 검색 성공! (예시 3개)")
        for ds in datasets:
            print(f"   - {ds.ref}")
    except Exception as e:
        print(f"❌ 데이터셋 조회 실패: {e}")
        return False
    
    print("\n" + "="*80)
    print("✅ Kaggle API 설정 완료!")
    print("="*80)
    
    return True


if __name__ == "__main__":
    success = check_kaggle_setup()
    
    if success:
        print("\n🎯 다음 단계:")
        print("   python scripts/collect_large_dataset.py")
    else:
        print("\n⚠️  설정을 완료한 후 다시 실행하세요.")
