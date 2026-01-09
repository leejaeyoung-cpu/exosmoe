import pandas as pd
import json

# Load the microarray data
excel_path = r'c:\Users\brook\Desktop\mi_exo_ai\data\Final_Analysis_Result\Final_Analysis_Result\data3.xlsx'
df = pd.read_excel(excel_path)

# Save basic info to JSON
info = {
    "shape": df.shape,
    "columns": df.columns.tolist(),
    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    "sample_rows": df.head(10).to_dict('records')
}

output_path = r'c:\Users\brook\Desktop\mi_exo_ai\data_structure.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print(f"Data structure saved to: {output_path}")
print(f"Shape: {df.shape}")
print(f"Columns saved: {len(df.columns)}")
