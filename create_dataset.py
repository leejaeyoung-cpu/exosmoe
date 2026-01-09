import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# 설정
BASE_DIR = r"c:/Users/brook/Desktop/mi_exo_ai"
IMAGE_DIR = os.path.join(BASE_DIR, "이미지")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(BASE_DIR, "dataset_manifest.csv")

# 지원하는 파일 확장자
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
DATA_EXTS = {'.csv', '.xlsx', '.xls', '.txt'}

def scan_directory(directory, file_types):
    data_list = []
    if not os.path.exists(directory):
        print(f"Warning: Directory not found: {directory}")
        return data_list

    for root, dirs, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in file_types:
                file_path = os.path.join(root, file)
                # 레이블은 바로 상위 폴더 이름으로 가정
                label = os.path.basename(root)
                data_list.append({
                    'file_path': file_path,
                    'file_name': file,
                    'label': label,
                    'type': 'image' if file_types == IMAGE_EXTS else 'tabular'
                })
    return data_list

def main():
    print("Scanning directories...")
    images = scan_directory(IMAGE_DIR, IMAGE_EXTS)
    tabular = scan_directory(DATA_DIR, DATA_EXTS)
    
    all_data = images + tabular
    
    if not all_data:
        print("No data found!")
        return

    df = pd.DataFrame(all_data)
    
    print(f"Found {len(images)} images and {len(tabular)} data files.")
    
    # Train/Val/Test Split (80/10/10)
    # Stratified split based on label if possible, otherwise random
    try:
        train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    except ValueError:
        print("Warning: Stratified split failed (likely due to single class). Using random split.")
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Assign split column
    df.loc[train_df.index, 'split'] = 'train'
    df.loc[val_df.index, 'split'] = 'val'
    df.loc[test_df.index, 'split'] = 'test'
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"Dataset manifest saved to {OUTPUT_FILE}")
    print(df.groupby(['type', 'split', 'label']).size())

if __name__ == "__main__":
    main()
