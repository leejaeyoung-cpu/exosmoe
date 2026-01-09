"""
Quick Multi-Class Training Setup
HUVEC 데이터로 5개 클래스 모두 빠르게 테스트
"""

import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def create_quick_multiclass_dataset():
    """HUVEC 데이터를 5개 카테고리로 분산하여 빠른 테스트"""
    
    print("\n" + "="*80)
    print("🚀 Quick Multi-Class Dataset Creation")
    print("5개 기능 전체 학습을 위한 빠른 데이터셋 생성")
    print("="*80 + "\n")
    
    # 소스 데이터
    source_dir = Path(r"c:\Users\brook\Desktop\mi_exo_ai\data\HUVEC TNF-a\HUVEC TNF-a\251209")
    
    if not source_dir.exists():
        print(f"❌ 소스 데이터 없음: {source_dir}")
        return
    
    source_images = list(source_dir.glob("*.jpg"))
    print(f"📷 소스 이미지: {len(source_images)}개")
    
    # 타겟 디렉토리
    target_dir = Path("data/multiclass_training")
    categories = ['antioxidant', 'anti_fibrotic', 'anti_inflammatory', 'angiogenic', 'proliferation']
    
    for cat in categories:
        cat_dir = target_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
    
    # 이미지를 5개 카테고리로 분산 (각기 다른 변형 적용)
    print("\n🔄 카테고리별 변형 적용 중...")
    
    transformations = {
        'antioxidant': lambda img: cv2.applyColorMap(img, cv2.COLORMAP_AUTUMN),  # 색상 변화
        'anti_fibrotic': lambda img: cv2.GaussianBlur(img, (5, 5), 0),  # 블러
        'anti_inflammatory': lambda img: img,  # 원본
        'angiogenic': lambda img: cv2.addWeighted(img, 1.2, np.zeros(img.shape, img.dtype), 0, 10),  # 밝기
        'proliferation': lambda img: cv2.flip(img, 1)  # 좌우 반전
    }
    
    for idx, img_path in enumerate(tqdm(source_images, desc="처리 중")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        for cat, transform in transformations.items():
            # 변형 적용
            transformed = transform(img.copy())
            
            # 저장
            output_path = target_dir / cat / f"{cat}_{idx:03d}.jpg"
            cv2.imwrite(str(output_path), transformed)
    
    # 통계
    print("\n📊 생성된 데이터셋:")
    total = 0
    for cat in categories:
        cat_dir = target_dir / cat
        count = len(list(cat_dir.glob("*.jpg")))
        print(f"  {cat:20s}: {count:3d} images")
        total += count
    
    print(f"\n  {'TOTAL':20s}: {total:3d} images")
    print(f"\n각 카테고리: {len(source_images)}개 이미지")
    
    return target_dir


def augment_and_prepare(data_dir, n_augmentations=100):
    """데이터 증강"""
    
    print("\n" + "="*80)
    print(f"🎨 데이터 증강 ({n_augmentations}배)")
    print("="*80 + "\n")
    
    from scripts.augment_dataset import CellImageAugmentor
    
    augmentor = CellImageAugmentor()
    
    output_dir = Path("data/multiclass_augmented")
    
    augmentor.process_dataset(
        input_dir=str(data_dir),
        output_dir=str(output_dir),
        n_augmentations=n_augmentations
    )
    
    return output_dir


def main():
    """실행"""
    
    # 1. 멀티클래스 데이터셋 생성
    data_dir = create_quick_multiclass_dataset()
    
    if data_dir:
        print("\n" + "="*80)
        print("✅ 데이터셋 생성 완료!")
        print("="*80 + "\n")
        
        print("📋 다음 단계:")
        print("   1. 데이터 증강 (선택):")
        print("      python scripts/augment_dataset.py")
        print("   2. 멀티클래스 학습:")
        print("      python scripts/train_multiclass_final.py")
        
        # 자동으로 학습 스크립트 생성
        create_training_script()
    
    return data_dir


def create_training_script():
    """멀티클래스 학습 스크립트 생성"""
    
    script_path = Path("scripts/train_multiclass_final.py")
    
    script_content = '''"""
Multi-Class Training - Final
5개 기능 전체 학습
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append('.')
from src.mt_exo_model import MTEXOClassifier


class MultiClassDataset(Dataset):
    """5개 클래스 데이터셋"""
    
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.transform = transform
        
        self.class_to_idx = {
            'antioxidant': 0,
            'anti_fibrotic': 1,
            'anti_inflammatory': 2,
            'angiogenic': 3,
            'proliferation': 4
        }
        
        # 모든 이미지 수집
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = Path(data_dir) / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    self.data.append((str(img_path), class_idx))
        
        print(f"총 {len(self.data)}개 이미지 로드")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def train():
    """학습 실행"""
    
    print("\\n" + "="*80)
    print("🎓 Multi-Class Training")
    print("="*80 + "\\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 데이터셋
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = MultiClassDataset("data/multiclass_training", transform=transform)
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 모델
    model = MTEXOClassifier(num_classes=5, pretrained=True)
    model = model.to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 학습
    print("\\n🚀 학습 시작 (50 epochs)...\\n")
    best_acc = 0.0
    
    for epoch in range(50):
        # Train
        model.train()
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/50: Train Acc: {train_acc:.2%}, Val Acc: {val_acc:.2%}")
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'models/multiclass_model.pth')
            print(f"  ✅ Best model saved! Acc: {best_acc:.2%}")
    
    print("\\n" + "="*80)
    print(f"✅ 학습 완료! Best Accuracy: {best_acc:.2%}")
    print("="*80)


if __name__ == "__main__":
    train()
'''
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"\n📝 학습 스크립트 생성: {script_path}")


if __name__ == "__main__":
    main()
