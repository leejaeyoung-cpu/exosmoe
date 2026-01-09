import pandas as pd
import os
import numpy as np

manifest_path = "c:/Users/brook/Desktop/mi_exo_ai/dataset_manifest.csv"
omics_path = "c:/Users/brook/Desktop/mi_exo_ai/data/Final_Analysis_Result/Final_Analysis_Result/data1-raw.data.xlsx"

print(f"Loading manifest from {manifest_path}...")
manifest = pd.read_csv(manifest_path)
print("Manifest Labels:", manifest['label'].unique())

print(f"\nLoading omics from {omics_path}...")
try:
    omics = pd.read_excel(omics_path, index_col=0)
    
    with open("debug_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Omics Index: {omics.index.tolist()[:10]} ...\n") # Truncate for brevity
        f.write(f"Omics Columns: {omics.columns.tolist()}\n")

        
        # Check intersection
        manifest_labels = set(manifest['label'].unique())
        omics_indices = set(omics.index)
        
        intersection = manifest_labels.intersection(omics_indices)
        f.write(f"\nIntersection (Matches): {intersection}\n")
        f.write(f"Missing in Omics: {manifest_labels - omics_indices}\n")
        
        # Check if we are getting zero vectors
        # Simulate data_loader logic
        f.write("\nSimulating Data Loader Lookup:\n")
        for label in manifest['label'].unique():
            if label in omics.index:
                data = omics.loc[label].values
                if len(data.shape) > 1:
                    data = np.mean(data, axis=0)
                f.write(f"Label '{label}': Found data shape {data.shape}, Mean val: {np.mean(data):.4f}\n")
            else:
                f.write(f"Label '{label}': NOT FOUND (would be zero vector)\n")

except Exception as e:
    print(f"Error loading omics: {e}")
