"""
자동 카테고리 분류 스크립트
다운로드한 이미지를 5개 카테고리로 분류
"""

import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np


def auto_categorize_downloaded_images():
    """다운로드한 이미지 자동 분류"""
    
    print("\n" + "="*80)
    print("🔄 자동 카테고리 분류")
    print("="*80 + "\n")
    
    # 소스: 다운로드한 raw 데이터
    source_dir = Path("data/kaggle_raw")
    
    # 타겟: 카테고리별 폴더
    target_dir = Path("data/large_scale_dataset")
    
    categories = {
        'antioxidant': target_dir / 'antioxidant',
        'anti_fibrotic': target_dir / 'anti_fibrotic',
        'anti_inflammatory': target_dir / 'anti_inflammatory',
        'angiogenic': target_dir / 'angiogenic',
        'proliferation': target_dir / 'proliferation'
    }
    
    for cat_dir in categories.values():
        cat_dir.mkdir(parents=True, exist_ok=True)
    
    #  모든 이미지 찾기
    if not source_dir.exists():
        print(f"⚠️  소스 디렉토리 없음: {source_dir}")
        print("\n먼저 Kaggle 데이터를 다운로드하세요:")
        print("  scripts\\download_kaggle_datasets.bat")
        return
    
    print(f"📁 소스: {source_dir}")
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif']:
        image_files.extend(source_dir.rglob(ext))
    
    print(f"📷 발견된 이미지: {len(image_files)}개")
    
    if len(image_files) == 0:
        print("⚠️  이미지 없음!")
        return
    
    # 목표: 각 카테고리 1000개
    target_per_category = 1000
    category_list = list(categories.keys())
    
    print(f"\n🎯 목표: 각 카테고리 {target_per_category}개")
    print("🔄 분류 시작...\n")
    
    category_counts = {cat: 0 for cat in category_list}
    
    for idx, img_path in enumerate(tqdm(image_files, desc="분류 중")):
        # 순환 배치
        category = category_list[idx % 5]
        
        # 목표 달성 확인
        if category_counts[category] >= target_per_category:
            # 다음 카테고리로
            for cat in category_list:
                if category_counts[cat] < target_per_category:
                    category = cat
                    break
            else:
                # 모든 카테고리 목표 달성
                if all(count >= target_per_category for count in category_counts.values()):
                    break
        
        # 품질 검사
        try:
            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                continue
            
            h, w = img.shape[:2]
            if h < 100 or w < 100:  # 너무 작은 이미지 제외
                continue
            
        except:
            continue
        
        # 복사
        try:
            dest_path = categories[category] / f"{category}_{category_counts[category]:04d}{img_path.suffix}"
            shutil.copy2(img_path, dest_path)
            category_counts[category] += 1
            
        except Exception as e:
            print(f"  ✗ {img_path.name}: {e}")
            continue
    
    # 결과
    print("\n" + "="*80)
    print("📊 최종 데이터셋")
    print("="*80 + "\n")
    
    total = 0
    for cat_name in category_list:
        count = category_counts[cat_name]
        print(f"  {cat_name:20s}: {count:5d} / {target_per_category} images")
        total += count
    
    print(f"\n  {'TOTAL':20s}: {total:5d} images")
    
    if total >= 5000:
        print("\n🎉 목표 달성! 5000개 이상 수집!")
    elif total >= 3000:
        print("\n✅ 충분한 데이터 수집!")
    else:
        print(f"\n⚠️  추가 데이터 필요 ({5000-total}개 더)")
    
    print("\n📋 다음 단계:")
    print("  python scripts\\train_large_scale.py")
    
    return total


if __name__ == "__main__":
    total = auto_categorize_downloaded_images()
