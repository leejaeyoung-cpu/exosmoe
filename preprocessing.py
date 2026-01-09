import torch
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

class ImagePreprocessor:
    def __init__(self, size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.size = size
        self.mean = mean
        self.std = std
        
    def get_train_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
    def get_val_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

class OmicsPreprocessor:
    def __init__(self, log_transform=True, normalize=True):
        self.log_transform = log_transform
        self.normalize = normalize
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
    def fit_transform(self, df):
        # Assuming df contains only numerical feature columns
        data = df.values
        
        # 1. Impute missing values
        data = self.imputer.fit_transform(data)
        
        # 2. Log Transform (if applicable, e.g., for raw counts)
        if self.log_transform:
            # Add small epsilon to avoid log(0)
            data = np.log2(data + 1e-6)
            
        # 3. Normalize (Z-score)
        if self.normalize:
            data = self.scaler.fit_transform(data)
            
        return pd.DataFrame(data, columns=df.columns, index=df.index)
    
    def transform(self, df):
        data = df.values
        data = self.imputer.transform(data)
        if self.log_transform:
            data = np.log2(data + 1e-6)
        if self.normalize:
            data = self.scaler.transform(data)
        return pd.DataFrame(data, columns=df.columns, index=df.index)
