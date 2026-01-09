"""
Public Dataset Collection for 5 Functions
공개 데이터셋에서 5개 기능별 세포 이미지 수집
"""

import subprocess
import json
from pathlib import Path
import shutil


class MultiClassDataCollector:
    """5개 기능 전체를 위한 데이터 수집"""
    
    def __init__(self, output_dir="data/multiclass_training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 카테고리
        self.categories = {
            'antioxidant': self.output_dir / 'antioxidant',
            'anti_fibrotic': self.output_dir / 'anti_fibrotic',
            'anti_inflammatory': self.output_dir / 'anti_inflammatory',
            'angiogenic': self.output_dir / 'angiogenic',
            'proliferation': self.output_dir / 'proliferation'
        }
        
        for cat_dir in self.categories.values():
            cat_dir.mkdir(exist_ok=True)
    
    def setup_kaggle(self):
        """Kaggle API 설정 안내"""
        
        print("\n" + "="*80)
        print("📦 Kaggle API 설정")
        print("="*80 + "\n")
        
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        
        if kaggle_json.exists():
            print("✅ Kaggle API 설정 완료!")
            return True
        else:
            print("❌ Kaggle API 설정 필요\n")
            print("📝 설정 방법:")
            print("   1. https://www.kaggle.com/settings 접속")
            print("   2. 'API' 섹션에서 'Create New API Token' 클릭")
            print("   3. 다운로드된 kaggle.json 파일을:")
            print(f"      → {Path.home() / '.kaggle'} 폴더에 저장")
            print("\n   Windows 명령:")
            print(f"      mkdir {Path.home() / '.kaggle'}")
            print(f"      move kaggle.json {Path.home() / '.kaggle'}")
            
            return False
    
    def download_kaggle_datasets(self):
        """Kaggle 데이터셋 다운로드"""
        
        print("\n" + "="*80)
        print("📥 Kaggle 데이터셋 다운로드")
        print("="*80 + "\n")
        
        # 추천 데이터셋
        datasets = [
            {
                'name': 'shariful07/cell-image-classification',
                'description': '세포 이미지 분류',
                'category': 'general'
            },
            {
                'name': 'paultimothymooney/blood-cells',
                'description': '혈구 세포 이미지',
                'category': 'proliferation'
            }
        ]
        
        for ds in datasets:
            print(f"\n다운로드: {ds['name']}")
            print(f"  설명: {ds['description']}")
            
            try:
                cmd = f"kaggle datasets download -d {ds['name']} -p data/kaggle_downloads --unzip"
                subprocess.run(cmd, shell=True, check=True)
                print(f"  ✅ 다운로드 완료!")
            except subprocess.CalledProcessError:
                print(f"  ❌ 다운로드 실패 - Kaggle API 설정 확인 필요")
            except FileNotFoundError:
                print(f"  ❌ Kaggle CLI 설치 필요: pip install kaggle")
    
    def download_alternative_sources(self):
        """대안 데이터 소스"""
        
        print("\n" + "="*80)
        print("🔗 대안 데이터 소스")
        print("="*80 + "\n")
        
        sources = {
            'Cell Image Library': {
                'url': 'http://cellimagelibrary.org',
                'method': '수동 다운로드 및 분류',
                'keywords': ['antioxidant', 'fibrosis', 'inflammation', 'angiogenesis', 'proliferation']
            },
            'BioImage Archive': {
                'url': 'https://www.ebi.ac.uk/bioimage-archive',
                'method': '검색 후 수동 다운로드',
                'keywords': ['cell treatment', 'drug response', 'cellular function']
            },
            'Broad Bioimage Benchmark': {
                'url': 'https://bbbc.broadinstitute.org',
                'method': '벤치마크 데이터셋 다운로드',
                'data': 'BBBC013, BBBC021 등'
            }
        }
        
        for name, info in sources.items():
            print(f"\n📌 {name}")
            print(f"   URL: {info['url']}")
            print(f"   방법: {info['method']}")
            if 'keywords' in info:
                print(f"   검색어: {', '.join(info['keywords'])}")
    
    def create_download_instructions(self):
        """다운로드 가이드 생성"""
        
        guide_file = self.output_dir / "DOWNLOAD_GUIDE.md"
        
        guide_content = """# 공개 데이터셋 다운로드 가이드

## 🎯 목표
5개 기능별 세포 이미지 수집 (각 100+ 이미지)

## 📦 Kaggle 데이터셋 (추천)

### 1. Kaggle API 설정
```bash
# 1. https://www.kaggle.com/settings 에서 API Token 생성
# 2. kaggle.json 다운로드
# 3. ~/.kaggle/ 에 저장
pip install kaggle
```

### 2. 추천 데이터셋
```bash
# 세포 이미지 분류
kaggle datasets download -d shariful07/cell-image-classification --unzip

# 혈구 세포
kaggle datasets download -d paultimothymooney/blood-cells --unzip

# 세포 형태
kaggle datasets download -d kmader/bioimage-classification --unzip
```

## 🔬 전문 데이터 소스

### Cell Image Library
- URL: http://cellimagelibrary.org
- 검색: "antioxidant", "fibrosis", "inflammation" 등
- 수동 다운로드 후 각 폴더에 분류

### BioImage Archive  
- URL: https://www.ebi.ac.uk/bioimage-archive
- 엑소좀 관련 논문 데이터
- 메타데이터 포함

### BBBC (Broad Bioimage Benchmark)
- URL: https://bbbc.broadinstitute.org
- BBBC013: Human U2OS cells
- BBBC021: MCF-7 breast cancer cells

## 📂 데이터 구조

다운로드 후 다음 구조로 정리:
```
data/multiclass_training/
├── antioxidant/       ← 항산화 관련 이미지
├── anti_fibrotic/     ← 항섬유화 관련 이미지  
├── anti_inflammatory/ ← 항염증 관련 이미지 (HUVEC)
├── angiogenic/        ← 혈관형성 관련 이미지
└── proliferation/     ← 세포증식 관련 이미지
```

## 🚀 다음 단계

데이터 수집 후:
```bash
# 증강
python scripts/augment_multiclass_dataset.py

# 학습
python scripts/train_multiclass_model.py
```

## 💡 팁

1. **각 카테고리 최소 50개** 원본 이미지 필요
2. 증강으로 100배 확장 가능
3. 품질 > 양: 선명하고 라벨링이 정확한 이미지 선택
4. 메타데이터 확인: 처리 조건, 약물 정보 등
"""
        
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        print(f"\n📋 다운로드 가이드 생성: {guide_file}")
        
        return guide_file


def main():
    """실행"""
    
    print("\n" + "="*80)
    print("🌐 Multi-Class Data Collection Setup")
    print("5개 기능 전체 학습을 위한 데이터 수집")
    print("="*80 + "\n")
    
    collector = MultiClassDataCollector()
    
    # 1. Kaggle 설정 확인
    if collector.setup_kaggle():
        # 2. Kaggle 데이터 다운로드 시도
        collector.download_kaggle_datasets()
    
    # 3. 대안 소스 안내
    collector.download_alternative_sources()
    
    # 4. 가이드 생성
    guide_file = collector.create_download_instructions()
    
    print("\n" + "="*80)
    print("📋 다음 단계")
    print("="*80 + "\n")
    
    print("1️⃣  Kaggle API 설정 (아직 안 했다면)")
    print("2️⃣  추천 데이터셋 다운로드:")
    print("     kaggle datasets download -d shariful07/cell-image-classification --unzip")
    print("3️⃣  수동 분류:")
    print("     - 다운로드한 이미지를 5개 카테고리로 분류")
    print("     - 각 폴더에 최소 50개 이미지")
    print("4️⃣  증강 & 학습:")
    print("     python scripts/train_multiclass_model.py")
    
    print(f"\n📖 자세한 가이드: {guide_file}")
    
    return collector


if __name__ == "__main__":
    collector = main()
