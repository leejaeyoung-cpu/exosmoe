@echo off
REM Kaggle 대용량 데이터셋 자동 다운로드 (Windows)

echo 🚀 Starting large-scale dataset download...

REM 1. Blood Cells (12,500 images)
echo 📦 Downloading blood-cells dataset...
kaggle datasets download -d paultimothymooney/blood-cells -p data/kaggle_raw --unzip

REM 2. Cell Image Classification
echo 📦 Downloading cell-image-classification...
kaggle datasets download -d shariful07/cell-image-classification -p data/kaggle_raw --unzip

REM 3. Bioimage Classification
echo 📦 Downloading bioimage-classification...
kaggle datasets download -d kmader/bioimage-classification -p data/kaggle_raw --unzip

echo ✅ Download complete!
echo Run: python scripts\auto_categorize.py
