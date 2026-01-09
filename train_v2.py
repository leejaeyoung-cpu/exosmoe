"""
Train v2: 31개 검증된 실험 데이터 기반 학습
- 강력한 데이터 증강 (Aggressive Augmentation)
- ResNet50 Transfer Learning
- 실제 실험 조건 분류 (AG11515+Mt-exo 등)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
import copy
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# 설정
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExperimentDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform
        self.file_paths = dataframe['file_path'].values
        self.labels = dataframe['encoded_label'].values
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 에러 시 검은 이미지 반환 (학습 중단 방지)
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def get_transforms():
    # 강력한 데이터 증강
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_experiment_model():
    print(f"🚀 학습 시작 (Device: {DEVICE})")
    
    # 1. 데이터 로드
    manifest_path = "dataset_manifest.csv"
    if not os.path.exists(manifest_path):
        print(f"❌ 파일 없음: {manifest_path}")
        return
        
    df = pd.read_csv(manifest_path)
    print(f"📊 총 데이터 수: {len(df)}")
    
    # 레이블 인코딩
    le = LabelEncoder()
    df['encoded_label'] = le.fit_transform(df['label'])
    
    classes = le.classes_
    num_classes = len(classes)
    print(f"🏷️ 클래스 ({num_classes}개): {classes}")
    
    # Train/Val 분할 (데이터가 적으므로 Stratified Split 중요)
    # 데이터가 너무 적어서 일부 클래스는 1개일 수 있음 -> 예외 처리
    try:
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['encoded_label'], random_state=42)
    except ValueError:
        print("⚠️ 데이터가 너무 적어 Stratified Split 불가. 랜덤 분할합니다.")
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # 데이터셋 및 로더
    train_transform, val_transform = get_transforms()
    
    train_dataset = ExperimentDataset(train_df, transform=train_transform)
    val_dataset = ExperimentDataset(val_df, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. 모델 초기화 (ResNet50)
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    model = model.to(DEVICE)
    
    # 3. 학습 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    
    # 4. 학습 루프
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = []
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        
        # Val
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)
        
        print(f'Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # 기록
        history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'train_acc': epoch_acc.item(),
            'val_loss': val_loss,
            'val_acc': val_acc.item()
        })
        
        # Best Model 저장
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'best_model_v2.pth')
            print(f"  🏆 New Best Accuracy: {best_acc:.4f}")
            
        scheduler.step(val_loss)
        
        # 로그 저장
        pd.DataFrame(history).to_csv('training_log_v2.csv', index=False)
        
    time_elapsed = time.time() - start_time
    print(f'\n✅ 학습 완료! 소요 시간: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'🔥 Best Validation Accuracy: {best_acc:.4f}')

if __name__ == "__main__":
    train_experiment_model()
