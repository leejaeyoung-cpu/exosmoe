import torch
import torch.nn as nn
from torchvision import models

class MelaExosomeModel(nn.Module):
    def __init__(self, num_classes, omics_dim, fusion_dim=512):
        super(MelaExosomeModel, self).__init__()
        
        # 1. Image Branch (ResNet50)
        resnet = models.resnet50(pretrained=True)
        # Remove the last FC layer
        self.image_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.image_dim = 2048
        
        # 2. Omics Branch (MLP)
        self.omics_extractor = nn.Sequential(
            nn.Linear(omics_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.omics_out_dim = 256
        
        # 3. Fusion Layer
        self.fusion_dim = self.image_dim + self.omics_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(fusion_dim, num_classes)
        )
        
    def forward(self, image, omics):
        # Image Branch
        img_feat = self.image_extractor(image)
        img_feat = img_feat.view(img_feat.size(0), -1) # Flatten (N, 2048)
        
        # Omics Branch
        if omics.shape[1] > 0:
            omics_feat = self.omics_extractor(omics)
        else:
            # Handle case with no omics data (should be handled by loader, but safety check)
            omics_feat = torch.zeros(image.size(0), self.omics_out_dim).to(image.device)
            
        # Fusion
        combined = torch.cat((img_feat, omics_feat), dim=1)
        
        # Classification
        output = self.classifier(combined)
        return output
