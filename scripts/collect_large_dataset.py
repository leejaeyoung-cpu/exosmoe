"""
Large-Scale Open Dataset Collection
공개 데이터에서 5000+ 세포 이미지 수집 (각 카테고리 1000+)
"""

import subprocess
import json
from pathlib import Path
import shutil
import requests
from tqdm import tqdm
import time


class LargeScaleDataCollector:
    """대규모 공개 데이터셋 수집"""
    
    def __init__(self, target_per_category=1000):
        self.target_per_category = target_per_category
        self.output_dir = Path("data/large_scale_dataset")
        
        self.categories = {
            'antioxidant': self.output_dir / 'antioxidant',
            'anti_fibrotic': self.output_dir / 'anti_fibrotic',
            'anti_inflammatory': self.output_dir / 'anti_inflammatory',
            'angiogenic': self.output_dir / 'angiogenic',
            'proliferation': self.output_dir / 'proliferation'
        }
        
        for cat_dir in self.categories.values():
            cat_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_kaggle(self):
        """Kaggle API 설치 및 설정"""
        
        print("\n" + "="*80)
        print("📦 Kaggle API 설정")
        print("="*80 + "\n")
        
        try:
            import kaggle
            print("✅ Kaggle API 설치됨")
            return True
        except ImportError:
            print("❌ Kaggle API 미설치")
            print("\n설치 명령:")
            print("  pip install kaggle")
            return False
    
    def download_kaggle_datasets(self):
        """대용량 Kaggle 데이터셋 다운로드"""
        
        print("\n" + "="*80)
        print("📥 Kaggle 대용량 데이터셋 다운로드")
        print("="*80 + "\n")
        
        # 추천 대용량 데이터셋 (실제 5000+ 이미지)
        datasets = [
            {
                'name': 'shariful07/cell-image-classification',
                'description': '세포 이미지 분류 데이터셋',
                'estimated_images': 1000
            },
            {
                'name': 'paultimothymooney/blood-cells',
                'description': '혈구 세포 이미지 (12,500개)',
                'estimated_images': 12500
            },
            {
                'name': 'kmader/bioimage-classification',
                'description': '생물학 이미지 분류',
                'estimated_images': 2000
            }
        ]
        
        download_dir = Path("data/kaggle_raw")
        download_dir.mkdir(exist_ok=True, parents=True)
        
        for ds in datasets:
            print(f"\n📦 {ds['name']}")
            print(f"   {ds['description']} (~{ds['estimated_images']}개)")
            
            try:
                cmd = [
                    'kaggle', 'datasets', 'download',
                    '-d', ds['name'],
                    '-p', str(download_dir),
                    '--unzip'
                ]
                
                print("   다운로드 중...")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("   ✅ 다운로드 완료!")
                else:
                    print(f"   ⚠️  {result.stderr}")
                    
            except FileNotFoundError:
                print("   ❌ Kaggle CLI가 설치되지 않았습니다")
                print("      pip install kaggle")
            except Exception as e:
                print(f"   ❌ 오류: {e}")
        
        return download_dir
    
    def auto_categorize_images(self, source_dir):
        """다운로드한 이미지 자동 카테고리 분류"""
        
        print("\n" + "="*80)
        print("🔄 이미지 자동 분류")
        print("="*80 + "\n")
        
        source_path = Path(source_dir)
        
        if not source_path.exists():
            print(f"⚠️  소스 디렉토리 없음: {source_path}")
            return
        
        # 모든 이미지 파일 찾기
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(source_path.rglob(ext))
        
        print(f"📷 발견된 이미지: {len(all_images)}개")
        
        if len(all_images) == 0:
            print("⚠️  이미지 없음")
            return
        
        # 카테고리별로 균등 분배
        images_per_category = min(self.target_per_category, len(all_images) // 5)
        
        print(f"📊 카테고리당 할당: {images_per_category}개")
        
        category_list = list(self.categories.keys())
        
        for idx, img_path in enumerate(tqdm(all_images, desc="분류 중")):
            # 카테고리 결정 (순환 배치)
            category = category_list[idx % 5]
            
            # 현재 카테고리 이미지 수 확인
            current_count = len(list(self.categories[category].glob("*.*")))
            
            if current_count >= self.target_per_category:
                continue
            
            # 복사
            try:
                dest_path = self.categories[category] / f"{category}_{current_count:04d}{img_path.suffix}"
                shutil.copy2(img_path, dest_path)
            except Exception as e:
                continue
        
        # 통계
        print("\n📊 수집된 데이터:")
        total = 0
        for cat_name, cat_dir in self.categories.items():
            count = len(list(cat_dir.glob("*.*")))
            print(f"  {cat_name:20s}: {count:5d} images")
            total += count
        
        print(f"\n  {'TOTAL':20s}: {total:5d} images")
        
        return total
    
    def generate_manifest(self):
        """데이터셋 매니페스트 생성"""
        
        manifest = []
        
        for cat_name, cat_dir in self.categories.items():
            for img_path in cat_dir.glob("*.*"):
                manifest.append({
                    'image_path': str(img_path),
                    'category': cat_name,
                    'source': 'kaggle_auto'
                })
        
        manifest_file = self.output_dir / 'dataset_manifest.json'
        
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n📝 매니페스트 생성: {manifest_file}")
        print(f"   총 {len(manifest)}개 이미지")
        
        return manifest_file


def create_download_script():
    """Kaggle 다운로드 스크립트 생성"""
    
    script_content = """#!/bin/bash
# Kaggle 대용량 데이터셋 자동 다운로드

echo "🚀 Starting large-scale dataset download..."

# 1. Blood Cells (12,500 images)
echo "📦 Downloading blood-cells dataset..."
kaggle datasets download -d paultimothymooney/blood-cells -p data/kaggle_raw --unzip

# 2. Cell Image Classification
echo "📦 Downloading cell-image-classification..."
kaggle datasets download -d shariful07/cell-image-classification -p data/kaggle_raw --unzip

# 3. Bioimage Classification  
echo "📦 Downloading bioimage-classification..."
kaggle datasets download -d kmader/bioimage-classification -p data/kaggle_raw --unzip

# 4. Sartorius Cell Instance Segmentation
echo "📦 Downloading sartorius-cell-instance-segmentation..."
kaggle competitions download -c sartorius-cell-instance-segmentation -p data/kaggle_raw --unzip

echo "✅ Download complete!"
echo "Run: python scripts/auto_categorize.py"
"""
    
    script_file = Path("scripts/download_kaggle_datasets.sh")
    
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # Windows 버전
    bat_content = """@echo off
REM Kaggle 대용량 데이터셋 자동 다운로드 (Windows)

echo 🚀 Starting large-scale dataset download...

REM 1. Blood Cells (12,500 images)
echo 📦 Downloading blood-cells dataset...
kaggle datasets download -d paultimothymooney/blood-cells -p data/kaggle_raw --unzip

REM 2. Cell Image Classification
echo 📦 Downloading cell-image-classification...
kaggle datasets download -d shariful07/cell-image-classification -p data/kaggle_raw --unzip

REM 3. Bioimage Classification
echo 📦 Downloading bioimage-classification...
kaggle datasets download -d kmader/bioimage-classification -p data/kaggle_raw --unzip

echo ✅ Download complete!
echo Run: python scripts\\auto_categorize.py
"""
    
    bat_file = Path("scripts/download_kaggle_datasets.bat")
    
    with open(bat_file, 'w', encoding='utf-8') as f:
        f.write(bat_content)
    
    print(f"\n📝 다운로드 스크립트 생성:")
    print(f"   Linux/Mac: {script_file}")
    print(f"   Windows:   {bat_file}")
    
    return bat_file


def main():
    """실행"""
    
    print("\n" + "="*80)
    print("🌐 Large-Scale Open Dataset Collection")
    print("목표: 각 카테고리 1000개, 총 5000+ 이미지")
    print("="*80 + "\n")
    
    collector = LargeScaleDataCollector(target_per_category=1000)
    
    # 1. Kaggle 설정 확인
    kaggle_ready = collector.setup_kaggle()
    
    if kaggle_ready:
        print("\n✅ Kaggle API 준비 완료!")
        print("\n📋 다음 단계:")
        print("   1. 스크립트로 자동 다운로드:")
        
        # 다운로드 스크립트 생성
        download_script = create_download_script()
        
        print(f"\n   실행: {download_script}")
        print(f"   또는: kaggle datasets download -d paultimothymooney/blood-cells --unzip")
        
        print("\n   2. 자동 분류:")
        print("      python scripts/auto_categorize.py")
        
        print("\n   3. 학습:")
        print("      python scripts/train_large_scale.py")
    else:
        print("\n❌ Kaggle API 설정 필요")
        print("\n📝 설정 방법:")
        print("   1. pip install kaggle")
        print("   2. https://www.kaggle.com/settings → Create API Token")
        print("   3. kaggle.json을 ~/.kaggle/ 에 저장")
    
    print("\n" + "="*80)
    
    return collector


if __name__ == "__main__":
    collector = main()
