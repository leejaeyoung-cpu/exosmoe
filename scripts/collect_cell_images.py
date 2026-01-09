"""
Cell Image Data Collector
Cell Image Library 및 공개 데이터셋 자동 수집
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import json
import time
from datetime import datetime
from typing import List, Dict
import cv2
import numpy as np
from urllib.parse import urljoin


class CellImageCollector:
    """공개 세포 이미지 데이터 자동 수집"""
    
    def __init__(self, output_dir="data/collected_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 카테고리별 디렉토리
        self.categories = {
            'antioxidant': self.output_dir / 'antioxidant',
            'anti_fibrotic': self.output_dir / 'anti_fibrotic',
            'anti_inflammatory': self.output_dir / 'anti_inflammatory',
            'angiogenic': self.output_dir / 'angiogenic',
            'proliferation': self.output_dir / 'proliferation',
            'unlabeled': self.output_dir / 'unlabeled'
        }
        
        for cat_dir in self.categories.values():
            cat_dir.mkdir(exist_ok=True)
        
        self.metadata = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def collect_from_kaggle_datasets(self):
        """Kaggle 공개 세포 이미지 데이터셋"""
        
        print("\n" + "="*80)
        print("📦 Kaggle 데이터셋 수집")
        print("="*80 + "\n")
        
        # 실제로는 Kaggle API 사용
        # 여기서는 공개 URL 예시
        datasets = [
            {
                'name': 'Cell Image Classification',
                'url': 'https://www.kaggle.com/datasets/kmader/bioimage-classification',
                'category': 'general'
            }
        ]
        
        print("💡 Kaggle API 설정 필요:")
        print("   1. https://www.kaggle.com/settings 에서 API Token 생성")
        print("   2. kaggle.json을 ~/.kaggle/ 에 저장")
        print("   3. pip install kaggle")
        print("\n   실행 명령:")
        for ds in datasets:
            print(f"   kaggle datasets download -d {ds['name']}")
        
        return datasets
    
    def collect_from_sample_urls(self):
        """샘플 공개 이미지 수집 (테스트용)"""
        
        print("\n" + "="*80)
        print("🖼️  샘플 이미지 수집")
        print("="*80 + "\n")
        
        # 공개 세포 이미지 샘플 (실제 프로덕션에서는 API 사용)
        sample_sources = [
            "https://cellimages.example.com",  # 예시
            "https://bioimage-archive.ebi.ac.uk"  # 실제 사이트
        ]
        
        print("💡 실제 데이터 수집을 위해서는:")
        print("   - Cell Image Library API 키 필요")
        print("   - BioImage Archive 계정 필요")
        print("   - 또는 직접 제공하신 이미지 사용")
        
        return []
    
    def collect_from_existing_huvec(self):
        """기존 HUVEC 데이터 수집 및 조직화"""
        
        print("\n" + "="*80)
        print("📁 기존 HUVEC 데이터 조직화")
        print("="*80 + "\n")
        
        huvec_dir = Path(r"c:\Users\brook\Desktop\mi_exo_ai\data\HUVEC TNF-a\HUVEC TNF-a\251209")
        
        if not huvec_dir.exists():
            print(f"⚠️  HUVEC 데이터 없음: {huvec_dir}")
            return []
        
        images = list(huvec_dir.glob("*.jpg"))
        print(f"📷 발견된 이미지: {len(images)}개")
        
        collected = []
        for img_path in images:
            try:
                # 이미지 로드 및 품질 검사
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                quality_ok, checks = self.check_image_quality(img)
                
                if quality_ok:
                    # unlabeled 카테고리로 복사
                    dest_path = self.categories['unlabeled'] / img_path.name
                    cv2.imwrite(str(dest_path), img)
                    
                    metadata = {
                        'source': 'HUVEC_TNF-a',
                        'original_path': str(img_path),
                        'new_path': str(dest_path),
                        'shape': img.shape,
                        'quality_checks': checks,
                        'category': 'unlabeled',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.metadata.append(metadata)
                    collected.append(dest_path)
                    
                    print(f"  ✓ {img_path.name}: {img.shape}")
                else:
                    print(f"  ✗ {img_path.name}: 품질 불량")
                    
            except Exception as e:
                print(f"  ✗ {img_path.name}: {e}")
        
        print(f"\n✅ 수집 완료: {len(collected)}개")
        return collected
    
    def check_image_quality(self, image: np.ndarray) -> tuple:
        """이미지 품질 검사"""
        
        checks = {}
        
        # 1. 해상도
        h, w = image.shape[:2]
        checks['resolution_ok'] = h >= 256 and w >= 256
        
        # 2. 밝기
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mean_brightness = gray.mean()
        checks['brightness_ok'] = 30 < mean_brightness < 225
        
        # 3. 대비
        std_brightness = gray.std()
        checks['contrast_ok'] = std_brightness > 20
        
        # 4. 포커스 (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        checks['focus_ok'] = laplacian_var > 50
        
        # 5. 노이즈 레벨
        # 간단한 노이즈 추정
        noise_estimate = np.std(cv2.GaussianBlur(gray, (5, 5), 0) - gray)
        checks['noise_ok'] = noise_estimate < 30
        
        quality_ok = all(checks.values())
        return quality_ok, checks
    
    def generate_dataset_statistics(self):
        """데이터셋 통계 생성"""
        
        print("\n" + "="*80)
        print("📊 데이터셋 통계")
        print("="*80 + "\n")
        
        stats = {}
        total_images = 0
        
        for cat_name, cat_dir in self.categories.items():
            images = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png"))
            count = len(images)
            total_images += count
            
            stats[cat_name] = {
                'count': count,
                'percentage': 0  # 나중에 계산
            }
            
            print(f"{cat_name:20s}: {count:4d} images")
        
        # 퍼센트 계산
        if total_images > 0:
            for cat in stats:
                stats[cat]['percentage'] = stats[cat]['count'] / total_images * 100
        
        print(f"\n{'TOTAL':20s}: {total_images:4d} images")
        
        return stats
    
    def save_metadata(self):
        """메타데이터 저장"""
        
        metadata_file = self.output_dir / 'metadata.json'
        
        summary = {
            'collection_date': datetime.now().isoformat(),
            'total_images': len(self.metadata),
            'categories': {
                cat: len([m for m in self.metadata if m['category'] == cat])
                for cat in self.categories.keys()
            },
            'images': self.metadata
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 메타데이터 저장: {metadata_file}")
        
        return metadata_file
    
    def create_manifest(self):
        """학습용 매니페스트 파일 생성"""
        
        manifest_file = self.output_dir / 'train_manifest.csv'
        
        rows = []
        for meta in self.metadata:
            rows.append({
                'image_path': meta['new_path'],
                'category': meta['category'],
                'source': meta['source'],
                'quality_score': sum(meta['quality_checks'].values()) / len(meta['quality_checks'])
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(manifest_file, index=False)
        
        print(f"📝 매니페스트 생성: {manifest_file}")
        print(f"   총 {len(df)}개 이미지")
        
        return manifest_file


def main():
    """데이터 수집 실행"""
    
    print("\n" + "="*80)
    print("🚀 Cell Image Data Collection")
    print("신약 개발용 AI 학습 데이터 수집")
    print("="*80 + "\n")
    
    collector = CellImageCollector(output_dir="data/collected_images")
    
    # 1. 기존 HUVEC 데이터 수집
    huvec_images = collector.collect_from_existing_huvec()
    
    # 2. Kaggle 데이터셋 안내
    kaggle_datasets = collector.collect_from_kaggle_datasets()
    
    # 3. 통계
    stats = collector.generate_dataset_statistics()
    
    # 4. 메타데이터 저장
    if collector.metadata:
        collector.save_metadata()
        collector.create_manifest()
    
    print("\n" + "="*80)
    print("✅ 데이터 수집 완료!")
    print("="*80 + "\n")
    
    print("📋 다음 단계:")
    print("   1. Kaggle API로 추가 데이터 다운로드")
    print("   2. 수동으로 카테고리 라벨링")
    print("   3. 데이터 증강 (scripts/augment_dataset.py)")
    print("   4. 모델 학습 시작")
    
    return collector


if __name__ == "__main__":
    collector = main()
