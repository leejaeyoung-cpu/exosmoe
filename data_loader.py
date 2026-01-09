import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from preprocessing import ImagePreprocessor, OmicsPreprocessor

class MelaExosomeDataset(Dataset):
    def __init__(self, manifest_df, omics_df=None, transform=None, mode='train'):
        self.manifest = manifest_df[manifest_df['split'] == mode].reset_index(drop=True)
        self.omics_df = omics_df
        self.transform = transform
        self.mode = mode
        
        self.labels = self.manifest['label'].unique().tolist()
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        
        # 1. Load Data (Image or Fused NPY)
        file_path = row['file_path']
        
        if file_path.endswith('.npy'):
            # Load Fused Data (H, W, 3)
            data = np.load(file_path)
            # Convert to Tensor (C, H, W)
            image = torch.from_numpy(data).permute(2, 0, 1).float()
            
            # Resize if needed (simple interpolation for now if dims don't match)
            # Ideally preprocessing should handle resizing before saving .npy
            if image.shape[1] != 224 or image.shape[2] != 224:
                image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
                
        else:
            # Standard Image Load
            try:
                img_pil = Image.open(file_path).convert('RGB')
                if self.transform:
                    image = self.transform(img_pil)
                else:
                    # Fallback transform
                    import torchvision.transforms as T
                    image = T.ToTensor()(img_pil)
            except Exception as e:
                print(f"Error loading image {file_path}: {e}")
                image = torch.zeros((3, 224, 224))
            
        # 2. Load Omics Data
        omics_vector = torch.tensor([])
        if self.omics_df is not None:
            label = row['label']
            # Custom Mapping Logic (from previous step)
            label_map = {
                'AG11515+control': 'Con-EXO.mean',
                'AG11515+Mt-exo': 'MT-EXOSOME.mean'
            }
            
            target_col = label_map.get(label)
            if target_col and target_col in self.omics_df.columns:
                omics_data = self.omics_df[target_col].values
                if np.isnan(omics_data).any():
                    omics_data = np.nan_to_num(omics_data)
                omics_vector = torch.tensor(omics_data, dtype=torch.float32)
            else:
                feature_dim = self.omics_df.shape[0] # Transposed shape
                omics_vector = torch.zeros(feature_dim, dtype=torch.float32)

        # 3. Label
        label_idx = self.label_to_idx.get(row['label'], -1)
        
        return {
            'image': image,
            'omics': omics_vector,
            'label': torch.tensor(label_idx, dtype=torch.long),
            'label_name': row['label']
        }

def get_dataloaders(manifest_path, omics_path=None, batch_size=32, num_workers=0):
    manifest_df = pd.read_csv(manifest_path)
    
    omics_processed = None
    if omics_path and os.path.exists(omics_path):
        try:
            # Load raw omics
            omics_processed = pd.read_excel(omics_path, index_col=0)
        except Exception as e:
            print(f"Warning: Failed to load omics data: {e}")

    img_prep = ImagePreprocessor()
    train_transform = img_prep.get_train_transforms()
    val_transform = img_prep.get_val_transforms()
    
    train_dataset = MelaExosomeDataset(manifest_df, omics_processed, transform=train_transform, mode='train')
    val_dataset = MelaExosomeDataset(manifest_df, omics_processed, transform=val_transform, mode='val')
    test_dataset = MelaExosomeDataset(manifest_df, omics_processed, transform=val_transform, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
