import os
import glob

# Target pattern
target = "pages/3_*_분석_및_예측.py"
files = glob.glob(target)

for f in files:
    try:
        os.remove(f)
        print(f"Deleted: {f}")
    except Exception as e:
        print(f"Error deleting {f}: {e}")
