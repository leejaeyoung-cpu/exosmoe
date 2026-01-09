import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloaders
from model import MelaExosomeModel
import os
import pandas as pd
import time
import copy
import argparse

def train_model(manifest_path, omics_path, num_epochs=100, batch_size=8, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on device: {device}")
    
    # 1. Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(manifest_path, omics_path, batch_size=batch_size)
    
    # Get dimensions from a sample batch
    sample_batch = next(iter(train_loader))
    num_classes = len(train_loader.dataset.labels)
    omics_dim = sample_batch['omics'].shape[1]
    print(f"Num Classes: {num_classes}, Omics Dim: {omics_dim}")
    print(f"Classes: {train_loader.dataset.labels}")
    
    # 2. Initialize Model
    model = MelaExosomeModel(num_classes=num_classes, omics_dim=omics_dim).to(device)
    
    # 3. Setup Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # 4. Training Loop
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for batch in dataloader:
                inputs = batch['image'].to(device)
                omics = batch['omics'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                
                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, omics)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + Optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"New best model saved with Acc: {best_acc:.4f}")
            
            if phase == 'val':
                scheduler.step(epoch_loss)
                history.append({'epoch': epoch+1, 'val_loss': epoch_loss, 'val_acc': epoch_acc.item()})
                # Update log file immediately for UI monitoring
                pd.DataFrame(history).to_csv('training_log.csv', index=False)

        print()
        
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    # Save history
    pd.DataFrame(history).to_csv('training_log.csv', index=False)
    print("Training log saved to training_log.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    manifest_file = "c:/Users/brook/Desktop/mi_exo_ai/dataset_manifest.csv"
    omics_file = "c:/Users/brook/Desktop/mi_exo_ai/data/Final_Analysis_Result/Final_Analysis_Result/data3.xlsx"
    
    train_model(manifest_file, omics_file, num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
