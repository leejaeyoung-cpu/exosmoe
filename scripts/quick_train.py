"""
Quick Training with HUVEC Data
HUVEC 데이터로 빠른 학습 시작
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2

import sys
sys.path.append('.')

from src.mt_exo_model import MTEXOClassifier


class HUVECDataset(Dataset):
    """HUVEC 데이터셋 (간단 버전)"""
    
    def __init__(self, image_dir, transform=None):
        self.image_paths = list(Path(image_dir).glob("*.jpg"))
        self.transform = transform
        
        # 임시 라벨: 모두 '항염증'으로 (TNF-α 처리이므로)
        self.label = 2  # anti_inflammatory
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.label


class SimpleAugmentation:
    """간단한 증강"""
    
    @staticmethod
    def augment_image(image_path, n=100):
        """이미지 1개를 n개로 증강"""
        
        img = cv2.imread(str(image_path))
        augmented = [img]  # 원본 포함
        
        for i in range(n-1):
            # 랜덤 변환
            aug = img.copy()
            
            # 회전
            angle = np.random.uniform(-180, 180)
            h, w = aug.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            aug = cv2.warpAffine(aug, M, (w, h))
            
            # 플립
            if np.random.rand() > 0.5:
                aug = cv2.flip(aug, 1)
            if np.random.rand() > 0.5:
                aug = cv2.flip(aug, 0)
            
            # 밝기
            factor = np.random.uniform(0.7, 1.3)
            aug = np.clip(aug * factor, 0, 255).astype(np.uint8)
            
            augmented.append(aug)
        
        return augmented


def quick_train():
    """빠른 학습 시작"""
    
    print("\n" + "="*80)
    print("🚀 Quick Training with HUVEC Data")
    print("="*80 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # HUVEC 데이터 경로
    huvec_dir = Path(r"c:\Users\brook\Desktop\mi_exo_ai\data\HUVEC TNF-a\HUVEC TNF-a\251209")
    
    if not huvec_dir.exists():
        print(f"❌ HUVEC 데이터 없음: {huvec_dir}")
        return
    
    images = list(huvec_dir.glob("*.jpg"))
    print(f"📷 원본 이미지: {len(images)}개")
    
    # 증강
    print("\n🔄 데이터 증강 중...")
    augmented_dir = Path("data/quick_train")
    augmented_dir.mkdir(exist_ok=True, parents=True)
    
    augmentor = SimpleAugmentation()
    total_augmented = 0
    
    for img_path in tqdm(images, desc="증강"):
        augmented = augmentor.augment_image(img_path, n=100)
        
        for i, aug_img in enumerate(augmented):
            save_path = augmented_dir / f"{img_path.stem}_aug{i:04d}.jpg"
            cv2.imwrite(str(save_path), aug_img)
            total_augmented += 1
    
    print(f"✅ 총 {total_augmented}개 이미지 생성")
    
    # 데이터셋
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = HUVECDataset(augmented_dir, transform=transform)
    
    # Train/Val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"\n📊 데이터: Train {train_size}, Val {val_size}")
    
    # 모델
    print("\n🤖 모델 로딩...")
    model = MTEXOClassifier(num_classes=5, pretrained=True)
    model = model.to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 학습
    print("\n🎓 학습 시작 (50 epochs)...")
    best_acc = 0.0
    
    for epoch in range(50):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
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
            torch.save(model.state_dict(), 'models/quick_trained_model.pth')
            print(f"  ✅ Best model saved! Acc: {best_acc:.2%}")
    
    print("\n" + "="*80)
    print(f"✅ 학습 완료! Best Accuracy: {best_acc:.2%}")
    print("="*80)
    
    return model


if __name__ == "__main__":
    model = quick_train()
