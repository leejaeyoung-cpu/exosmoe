"""
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
    
    print("\n" + "="*80)
    print("🎓 Multi-Class Training")
    print("="*80 + "\n")
    
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
    print("\n🚀 학습 시작 (50 epochs)...\n")
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
    
    print("\n" + "="*80)
    print(f"✅ 학습 완료! Best Accuracy: {best_acc:.2%}")
    print("="*80)


if __name__ == "__main__":
    train()
