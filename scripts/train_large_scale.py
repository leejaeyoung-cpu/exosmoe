"""
대규모 데이터셋 (5,000 images) 학습 스크립트
5개 카테고리 분류: antioxidant, anti_fibrotic, anti_inflammatory, angiogenic, proliferation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import time
from sklearn.model_selection import train_test_split


class LargeScaleDataset(Dataset):
    """5,000개 이미지 데이터셋"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # 레이블 인코딩
        self.label_to_idx = {
            'antioxidant': 0,
            'anti_fibrotic': 1,
            'anti_inflammatory': 2,
            'angiogenic': 3,
            'proliferation': 4
        }
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 이미지 로드
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 오류 시 빈 이미지 반환
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        label_idx = self.label_to_idx[label]
        
        return image, label_idx


def prepare_datasets(data_dir='data/large_scale_dataset', test_size=0.2, val_size=0.1):
    """데이터셋 준비 및 분할"""
    
    print("\n" + "="*80)
    print("📂 데이터셋 준비")
    print("="*80 + "\n")
    
    data_path = Path(data_dir)
    
    # 모든 이미지 수집
    all_images = []
    all_labels = []
    
    categories = ['antioxidant', 'anti_fibrotic', 'anti_inflammatory', 'angiogenic', 'proliferation']
    
    for category in categories:
        cat_path = data_path / category
        
        if not cat_path.exists():
            print(f"⚠️  {category} 폴더 없음")
            continue
        
        images = list(cat_path.glob('*.*'))
        
        all_images.extend(images)
        all_labels.extend([category] * len(images))
        
        print(f"  {category:20s}: {len(images):5d} images")
    
    print(f"\n  {'TOTAL':20s}: {len(all_images):5d} images")
    
    # Train/Val/Test 분할
    # 먼저 Train+Val / Test 분할
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        all_images, all_labels, 
        test_size=test_size, 
        stratify=all_labels,
        random_state=42
    )
    
    # Train / Val 분할
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio,
        stratify=y_trainval,
        random_state=42
    )
    
    print(f"\n📊 데이터 분할:")
    print(f"  Train: {len(X_train):5d} ({len(X_train)/len(all_images)*100:.1f}%)")
    print(f"  Val:   {len(X_val):5d} ({len(X_val)/len(all_images)*100:.1f}%)")
    print(f"  Test:  {len(X_test):5d} ({len(X_test)/len(all_images)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_transforms():
    """데이터 증강 및 전처리"""
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


class ImprovedResNet(nn.Module):
    """ResNet50 기반 개선 모델"""
    
    def __init__(self, num_classes=5):
        super(ImprovedResNet, self).__init__()
        
        # ResNet50 백본 (사전학습)
        self.backbone = models.resnet50(pretrained=True)
        
        # 피처 차원
        num_features = self.backbone.fc.in_features
        
        # 분류 헤드 개선
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def train_large_scale_model(
    data_dir='data/large_scale_dataset',
    num_epochs=50,
    batch_size=32,
    lr=1e-4,
    device=None
):
    """대규모 데이터셋 학습"""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*80)
    print("🚀 대규모 학습 시작")
    print("="*80 + "\n")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {lr}")
    
    # 1. 데이터 준비
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_datasets(data_dir)
    
    # 2. Transforms
    train_transform, val_transform = get_transforms()
    
    # 3. Datasets
    train_dataset = LargeScaleDataset(X_train, y_train, transform=train_transform)
    val_dataset = LargeScaleDataset(X_val, y_val, transform=val_transform)
    test_dataset = LargeScaleDataset(X_test, y_test, transform=val_transform)
    
    # 4. DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 5. 모델 초기화
    model = ImprovedResNet(num_classes=5).to(device)
    
    # 6. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 7. 학습 기록
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = 'best_large_scale_model.pth'
    
    print("\n" + "="*80)
    print("🔥 학습 시작")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*80}\n")
        
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Training', leave=False)
        for images, labels in train_pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 실시간 진행률
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100. * train_correct / train_total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Validation', leave=False)
            for images, labels in val_pbar:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100. * val_correct / val_total
        
        # 기록
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # 출력
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"  Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.2f}%")
        
        # Best 모델 저장
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_loss': epoch_val_loss
            }, best_model_path)
            print(f"  ✅ New best model saved! Val Acc: {best_val_acc:.2f}%")
        
        # Scheduler 업데이트
        scheduler.step()
        
        # 학습 로그 저장
        pd.DataFrame(history).to_csv('large_scale_training_log.csv', index=False)
    
    # 학습 완료
    elapsed_time = time.time() - start_time
    print("\n" + "="*80)
    print("✅ 학습 완료!")
    print("="*80 + "\n")
    print(f"⏱️  소요 시간: {elapsed_time//60:.0f}분 {elapsed_time%60:.0f}초")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Test 평가
    print("\n" + "="*80)
    print("🧪 Test Set 평가")
    print("="*80 + "\n")
    
    # Best 모델 로드
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = 100. * test_correct / test_total
    print(f"\n📊 Test Accuracy: {test_acc:.2f}%")
    
    # 결과 저장
    results = {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'training_time_seconds': elapsed_time,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'total_images': len(X_train) + len(X_val) + len(X_test)
    }
    
    with open('large_scale_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n💾 결과 저장:")
    print(f"  - 모델: {best_model_path}")
    print(f"  - 로그: large_scale_training_log.csv")
    print(f"  - 결과: large_scale_training_results.json")
    
    return model, history, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/large_scale_dataset')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    args = parser.parse_args()
    
    model, history, results = train_large_scale_model(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    print("\n🎉 모든 작업 완료!")
