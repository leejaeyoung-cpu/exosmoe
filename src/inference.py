import torch
from model import MelaExosomeModel
from data_loader import get_dataloaders
from preprocessing import ImagePreprocessor
from PIL import Image
import pandas as pd
import numpy as np
import os

class InferenceEngine:
    def __init__(self, model_path='best_model.pth', manifest_path="dataset_manifest.csv", omics_path="data/Final_Analysis_Result/Final_Analysis_Result/data3.xlsx"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.manifest_path = manifest_path
        self.omics_path = omics_path
        self.model = None
        self.labels = None
        self.preprocessor = ImagePreprocessor()
        
    def load_model(self):
        if not os.path.exists(self.model_path):
            return False, "Model file not found."
            
        # Need to know num_classes and omics_dim to init model
        # We can get this from the dataset loader or config
        # For now, let's load the dataset to get metadata
        try:
            train_dl, _, _ = get_dataloaders(self.manifest_path, self.omics_path, batch_size=1)
            self.labels = train_dl.dataset.labels
            num_classes = len(self.labels)
            
            # Get omics dim from a sample
            sample = next(iter(train_dl))
            omics_dim = sample['omics'].shape[1]
            
            self.model = MelaExosomeModel(num_classes=num_classes, omics_dim=omics_dim)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            return True, "Model loaded successfully."
        except Exception as e:
            return False, str(e)
            
    def predict(self, image_file, omics_vector=None):
        if self.model is None:
            return None
            
        # Preprocess Image
        img = Image.open(image_file).convert('RGB')
        transform = self.preprocessor.get_val_transforms()
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        # Handle Omics
        if omics_vector is None:
            # If no omics provided, use zero vector (or mean if we had it)
            # We need to know the dimension.
            # This is a limitation of the current simple inference.
            # Ideally we should look up omics by some ID or use a default.
            # For now, let's assume zero vector if not provided.
            omics_dim = self.model.fc_omics.in_features
            omics_tensor = torch.zeros((1, omics_dim)).to(self.device)
        else:
            omics_tensor = torch.tensor(omics_vector).float().unsqueeze(0).to(self.device)
            
        with torch.no_grad():
            outputs = self.model(img_tensor, omics_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_p, top_class = probs.topk(1, dim=1)
            
        return {
            'class': self.labels[top_class.item()],
            'probability': top_p.item(),
            'all_probs': {label: prob.item() for label, prob in zip(self.labels, probs[0])}
        }
