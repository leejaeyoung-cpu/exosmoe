#!/bin/bash
# Kaggle 대용량 데이터셋 자동 다운로드

echo "🚀 Starting large-scale dataset download..."

# 1. Blood Cells (12,500 images)
echo "📦 Downloading blood-cells dataset..."
kaggle datasets download -d paultimothymooney/blood-cells -p data/kaggle_raw --unzip

# 2. Cell Image Classification
echo "📦 Downloading cell-image-classification..."
kaggle datasets download -d shariful07/cell-image-classification -p data/kaggle_raw --unzip

# 3. Bioimage Classification  
echo "📦 Downloading bioimage-classification..."
kaggle datasets download -d kmader/bioimage-classification -p data/kaggle_raw --unzip

# 4. Sartorius Cell Instance Segmentation
echo "📦 Downloading sartorius-cell-instance-segmentation..."
kaggle competitions download -c sartorius-cell-instance-segmentation -p data/kaggle_raw --unzip

echo "✅ Download complete!"
echo "Run: python scripts/auto_categorize.py"
