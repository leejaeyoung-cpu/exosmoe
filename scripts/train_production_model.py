"""
Production Model Training Pipeline
신약 개발용 고성능 모델 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm
import pandas as pd
from datetime import datetime

import sys
sys.path.append('.')

from src.mt_exo_model import MTEXOClassifier


class CellImageDataset(Dataset):
    """세포 이미지 데이터셋"""
    
    def __init__(self, manifest_path, transform=None):
        with open(manifest_path, 'r') as f:
            self.data = json.load(f)
        
        self.transform = transform
        
        # 카테고리 매핑
        self.category_to_idx = {
            'antioxidant': 0,
            'anti_fibrotic': 1,
            'anti_inflammatory': 2,
            'angiogenic': 3,
            'proliferation': 4
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 이미지 로드
        image = Image.open(item['image_path']).convert('RGB')
        
        # 변환
        if self.transform:
            image = self.transform(image)
        
        # 라벨 (unlabeled는 스킵)
        category = item['category']
        if category not in self.category_to_idx:
            label = -1  # Unknown
        else:
            label = self.category_to_idx[category]
        
        return image, label


class FocalLoss(nn.Module):
    """Focal Loss (클래스 불균형 처리)"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class EarlyStopping:
    """Early Stopping"""
    
    def __init__(self, patience=15, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class ProductionTrainer:
    """프로덕션 학습 파이프라인"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n🔧 Device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        # 모델
        self.model = MTEXOClassifier(num_classes=5, pretrained=True)
        self.model = self.model.to(self.device)
        
        # Loss
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        # Optimizer
        self.optimizer = torch.opt im.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        # Early Stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 15),
            delta=0.001
        )
        
        self.history = []
    
    def train_epoch(self, train_loader):
        """1 에폭 학습"""
        
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            # Skip unlabeled
            mask = labels != -1
            if mask.sum() == 0:
                continue
            
            images = images[mask].to(self.device)
            labels = labels[mask].to(self.device)
            
            # Forward
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 통계
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """검증"""
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                mask = labels != -1
                if mask.sum() == 0:
                    continue
                
                images = images[mask].to(self.device)
                labels = labels[mask].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=150):
        """전체 학습"""
        
        print("\n" + "="*80)
        print("🚀 모델 학습 시작")
        print("="*80 + "\n")
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 80)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Scheduler
            self.scheduler.step()
            
            # Logging
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%}")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}")
            
            self.history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'models/best_mt_exo_model.pth')
                print(f"✅ Best model saved! Acc: {best_acc:.2%}")
            
            # Early stopping
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("\n⚠️  Early stopping triggered!")
                break
        
        print("\n" + "="*80)
        print(f"✅ 학습 완료! Best Accuracy: {best_acc:.2%}")
        print("="*80 + "\n")
        
        # 히스토리 저장
        self.save_history()
        
        return best_acc
    
    def save_history(self):
        """학습 히스토리 저장"""
        
        df = pd.DataFrame(self.history)
        history_file = Path('models/training_history.csv')
        df.to_csv(history_file, index=False)
        
        print(f"📊 학습 히스토리 저장: {history_file}")


def main():
    """학습 실행"""
    
    print("\n" + "="*80)
    print("🎯 Production Model Training")
    print("신약 개발용 AI 모델 학습")
    print("="*80 + "\n")
    
    # Config
    config = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 32,
        'epochs': 150,
        'patience': 15
    }
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset
    manifest_path = "data/augmented_dataset/augmented_manifest.json"
    
    if not Path(manifest_path).exists():
        print(f"❌ 매니페스트 없음: {manifest_path}")
        print("   먼저 데이터 증강을 실행하세요:")
        print("   python scripts/augment_dataset.py")
        return
    
    dataset = CellImageDataset(manifest_path, transform=train_transform)
    
    # Train/Val Split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"📊 데이터셋:")
    print(f"   Train: {train_size} images")
    print(f"   Val: {val_size} images")
    
    # Trainer
    trainer = ProductionTrainer(config)
    
    # Train
    best_acc = trainer.train(train_loader, val_loader, epochs=config['epochs'])
    
    print("\n🎉 학습 완료!")
    print(f"   Best Accuracy: {best_acc:.2%}")
    print(f"   모델 저장: models/best_mt_exo_model.pth")


if __name__ == "__main__":
    main()
