import pandas as pd
import os
import glob

def analyze_excel(path):
    print(f"\n--- Analyzing {os.path.basename(path)} ---")
    try:
        df = pd.read_excel(path, nrows=5)
        print("Columns:", df.columns.tolist())
        print("First 2 rows:")
        print(df.head(2).to_string())
    except Exception as e:
        print(f"Error reading Excel: {e}")

def analyze_html(path):
    print(f"\n--- Analyzing {os.path.basename(path)} ---")
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Simple extraction of title or headers
            if "<title>" in content:
                print("Title:", content.split("<title>")[1].split("</title>")[0])
            # Print first 500 chars to guess content
            print("Snippet:", content[:500].replace('\n', ' '))
    except Exception as e:
        print(f"Error reading HTML: {e}")

def list_images(path):
    print(f"\n--- Images in {os.path.basename(path)} ---")
    exts = ['*.jpg', '*.png', '*.tif', '*.tiff']
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(path, ext)))
    print(f"Found {len(files)} images.")
    if files:
        print("Sample:", [os.path.basename(f) for f in files[:5]])

# Paths
base_dir = r"c:\Users\brook\Desktop\mi_exo_ai\data"
final_res_dir = os.path.join(base_dir, "Final_Analysis_Result", "Final_Analysis_Result")
kegg_dir = os.path.join(base_dir, "HB00014503_GO_KEGG", "Final_Analysis_Result")
huvec_dir = os.path.join(base_dir, "HUVEC progerin over")

# 1. Analyze Excel Files
excel_files = [
    os.path.join(final_res_dir, "data1-raw.data.xlsx"),
    os.path.join(final_res_dir, "data2.xlsx"),
    os.path.join(final_res_dir, "data3.xlsx"),
    os.path.join(kegg_dir, "data3.xlsx") # This one is large
]

for f in excel_files:
    if os.path.exists(f):
        analyze_excel(f)

# 2. Analyze HTML
html_files = [
    os.path.join(final_res_dir, "Analysis_Result.html"),
    os.path.join(kegg_dir, "Analysis_Result.html")
]

for f in html_files:
    if os.path.exists(f):
        analyze_html(f)

# 3. List Images
if os.path.exists(huvec_dir):
    list_images(huvec_dir)
