"""
Advanced Data Augmentation for Cell Images  
세포 이미지 고급 증강 (15개 → 1,500개)
"""

import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from tqdm import tqdm
from typing import List, Dict


class CellImageAugmentor:
    """세포 이미지 전문 증강"""
    
    def __init__(self):
        # 강력한 증강 파이프라인
        self.transform = A.Compose([
            # 기하학적 변환
            A.Rotate(limit=180, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.7),
            
            # 탄성 변형 (세포 형태 변화)
            A.ElasticTransform(alpha=120, sigma=120*0.05, alpha_affine=120*0.03, p=0.3),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(p=0.3),
            
            # 컬러 및 밝기
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            
            # 노이즈 및 블러
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.MotionBlur(blur_limit=5, p=0.2),
            
            # 고급 기법
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            
            # CutOut/CoarseDropout (세포 일부 가림)
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
        ])
        
        # CutMix 스타일 증강 (별도 처리)
        self.cutmix_transform = A.Compose([
            A.RandomCrop(height=224, width=224, p=1.0),
        ])
    
    def augment_single_image(
        self, 
        image: np.ndarray, 
        n_augmentations: int = 100
    ) -> List[np.ndarray]:
        """단일 이미지를 n개로 증강"""
        
        augmented_images = []
        
        for i in range(n_augmentations):
            augmented = self.transform(image=image)['image']
            augmented_images.append(augmented)
        
        return augmented_images
    
    def process_dataset(
        self, 
        input_dir: str,
        output_dir: str,
        n_augmentations: int = 100
    ):
        """전체 데이터셋 증강"""
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print(f"🔄 데이터 증강: {n_augmentations}배")
        print("="*80 + "\n")
        
        # 카테고리별 처리
        categories = ['antioxidant', 'anti_fibrotic', 'anti_inflammatory', 
                     'angiogenic', 'proliferation', 'unlabeled']
        
        total_original = 0
        total_augmented = 0
        
        for category in categories:
            cat_input = input_path / category
            cat_output = output_path / category
            cat_output.mkdir(exist_ok=True)
            
            if not cat_input.exists():
                continue
            
            images = list(cat_input.glob("*.jpg")) + list(cat_input.glob("*.png"))
            
            if len(images) == 0:
                continue
            
            print(f"📁 {category}: {len(images)}개 원본")
            
            total_original += len(images)
            aug_count = 0
            
            for img_path in tqdm(images, desc=f"  처리 중"):
                # 원본 로드
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                # 원본 저장 (필요시)
                orig_out = cat_output / f"{img_path.stem}_orig{img_path.suffix}"
                cv2.imwrite(str(orig_out), image)
                aug_count += 1
                
                # 증강
                augmented_images = self.augment_single_image(image, n_augmentations)
                
                for i, aug_img in enumerate(augmented_images):
                    aug_out = cat_output / f"{img_path.stem}_aug{i:04d}.jpg"
                    cv2.imwrite(str(aug_out), aug_img)
                    aug_count += 1
            
            print(f"  ✓ {aug_count}개 생성")
            total_augmented += aug_count
        
        print(f"\n✅ 전체: {total_original} → {total_augmented}" + 
              (f" ({total_augmented/total_original:.1f}x)" if total_original > 0 else ""))
        
        # 매니페스트 생성
        self.create_augmented_manifest(output_path)
        
        return output_path
    
    def create_augmented_manifest(self, data_dir: Path):
        """증강된 데이터셋 매니페스트"""
        
        manifest = []
        
        for category_dir in data_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            category = category_dir.name
            
            for img_path in category_dir.glob("*.jpg"):
                manifest.append({
                    'image_path': str(img_path),
                    'category': category,
                    'is_augmented': 'aug' in img_path.stem
                })
        
        manifest_file = data_dir / 'augmented_manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n📝 매니페스트: {manifest_file}")
        print(f"   총 {len(manifest)}개 이미지")
        
        return manifest_file


def main():
    """증강 실행"""
    
    print("\n" + "="*80)
    print("🎨 Advanced Data Augmentation")
    print("신약 개발용 데이터셋 확장")
    print("="*80 + "\n")
    
    augmentor = CellImageAugmentor()
    
    # 입출력 경로
    input_dir = "data/collected_images"
    output_dir = "data/augmented_dataset"
    
    # 증강 실행 (각 이미지 → 100개)
    augmentor.process_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        n_augmentations=100
    )
    
    print("\n" + "="*80)
    print("✅ 데이터 증강 완료!")
    print("="*80 + "\n")
    
    print("📋 다음 단계:")
    print("   1. train/val/test 분할 (scripts/split_dataset.py)")
    print("   2. 모델 학습 시작 (scripts/train_production_model.py)")
    
    return augmentor


if __name__ == "__main__":
    augmentor = main()
