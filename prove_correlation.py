import torch
import torch.nn as nn
from torchvision import models
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import get_dataloaders
import os

def extract_image_features(dataloader, model, device):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch['image'].to(device)
            lbls = batch['label_name']
            
            # Extract features (2048-dim)
            feats = model(imgs)
            features.append(feats.cpu().numpy())
            labels.extend(lbls)
            
    return np.vstack(features), labels

def analyze_correlation(manifest_path, omics_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Setup Data Loader
    # Use a larger batch size for inference
    train_dl, _, _ = get_dataloaders(manifest_path, omics_path, batch_size=32)
    
    # 2. Setup Feature Extractor (ResNet50)
    # Remove the last classification layer to get raw features
    resnet = models.resnet50(pretrained=True)
    modules = list(resnet.children())[:-1] # Remove FC layer
    feature_extractor = nn.Sequential(*modules).to(device)
    
    print("Extracting Image Features...")
    img_features, img_labels = extract_image_features(train_dl, feature_extractor, device)
    img_features = img_features.reshape(img_features.shape[0], -1) # Flatten (N, 2048)
    print(f"Image Features Shape: {img_features.shape}")

    # 3. Get Omics Features (from the dataset)
    # We need to collect omics data corresponding to the images
    
    # Custom Mapping for this dataset
    label_map = {
        'AG11515+control': 'Con-EXO.mean',
        'AG11515+Mt-exo': 'MT-EXOSOME.mean'
    }
    
    # Load Omics Data Manually to ensure correct mapping
    try:
        omics_df = pd.read_excel(omics_path, index_col=0)
        print(f"Loaded Omics Data: {omics_df.shape}")
    except Exception as e:
        print(f"Error loading omics: {e}")
        return

    all_img_feats = []
    all_omics_feats = []
    all_labels = []
    
    print("Collecting paired features with mapping...")
    with torch.no_grad():
        for batch in train_dl:
            imgs = batch['image'].to(device)
            lbls = batch['label_name']
            
            # Extract image features
            i_feats = feature_extractor(imgs).cpu().numpy().reshape(imgs.shape[0], -1)
            
            for i, label in enumerate(lbls):
                if label in label_map and label_map[label] in omics_df.columns:
                    col_name = label_map[label]
                    omics_vec = omics_df[col_name].values
                    
                    # Handle NaNs if any
                    if np.isnan(omics_vec).any():
                        omics_vec = np.nan_to_num(omics_vec)
                        
                    all_img_feats.append(i_feats[i])
                    all_omics_feats.append(omics_vec)
                    all_labels.append(label)
            
    if not all_img_feats:
        print("Error: No paired image-omics data found after mapping.")
        return

    X_img = np.vstack(all_img_feats)
    X_omics = np.vstack(all_omics_feats)
    
    print(f"Paired Data: {X_img.shape[0]} samples (Filtered subset)")
    
    # 4. Dimensionality Reduction (PCA)
    print("Reducing dimensions for visualization...")
    n_samples = X_img.shape[0]
    n_features_img = X_img.shape[1]
    n_features_omics = X_omics.shape[1]
    
    n_components = min(n_samples, n_features_img, n_features_omics, 50)
    if n_components < 2:
        n_components = min(n_samples, 2)
        
    print(f"Using n_components={n_components} for PCA (Samples: {n_samples})")
    
    pca_img = PCA(n_components=n_components)
    pca_omics = PCA(n_components=n_components)
    
    X_img_pca = pca_img.fit_transform(X_img)
    X_omics_pca = pca_omics.fit_transform(X_omics)
    
    # 5. Correlation Analysis
    # Calculate Cosine Similarity between the two modalities
    similarity = cosine_similarity(X_img_pca, X_omics_pca)
    avg_sim = np.mean(np.diag(similarity))
    print(f"Average Cosine Similarity between aligned Image and Omics features: {avg_sim:.4f}")
    with open("correlation_score.txt", "w") as f:
        f.write(f"Average Cosine Similarity: {avg_sim:.4f}\n")
    
    # 6. Visualization (t-SNE)
    # We will plot t-SNE of Image features and color by Label
    # And t-SNE of Omics features and color by Label
    # If both separate the classes well, they are correlated in information content.
    
    perplexity = min(30, n_samples - 1)
    if perplexity < 1: perplexity = 1
    print(f"Using perplexity={perplexity} for t-SNE")
    
    tsne = TSNE(n_components=2, perplexity=perplexity, init='random', random_state=42)
    X_img_tsne = tsne.fit_transform(X_img_pca)
    X_omics_tsne = tsne.fit_transform(X_omics_pca)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=X_img_tsne[:,0], y=X_img_tsne[:,1], hue=all_labels, palette='viridis')
    plt.title("t-SNE of Image Features (ResNet50)")
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=X_omics_tsne[:,0], y=X_omics_tsne[:,1], hue=all_labels, palette='viridis')
    plt.title("t-SNE of Omics Features (PCA)")
    
    plt.tight_layout()
    plt.savefig('correlation_analysis_result.png')
    print("Saved visualization to correlation_analysis_result.png")

if __name__ == "__main__":
    manifest_file = "c:/Users/brook/Desktop/mi_exo_ai/dataset_manifest.csv"
    omics_file = "c:/Users/brook/Desktop/mi_exo_ai/data/Final_Analysis_Result/Final_Analysis_Result/data3.xlsx"
    
    analyze_correlation(manifest_file, omics_file)
