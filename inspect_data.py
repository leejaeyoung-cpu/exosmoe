import pandas as pd
import numpy as np

# Load data
df = pd.read_excel(r'c:\Users\brook\Desktop\mi_exo_ai\data\Final_Analysis_Result\Final_Analysis_Result\data3.xlsx')

print("="*80)
print("MT-EXO vs Control MicroArray Data Analysis")
print("="*80)

print(f"\n📊 Dataset Shape: {df.shape}")
print(f"   - Total miRNAs: {df.shape[0]}")
print(f"   - Total columns: {df.shape[1]}")

print("\n📋 Column Names:")
for i, col in enumerate(df.columns):
    print(f"   {i:2d}. {col}")

print("\n🔬 First 5 rows sample:")
print(df.head())

print("\n📈 Data types:")
print(df.dtypes)

# Check for fold change and p-value columns
print("\n🔍 Looking for key analysis columns...")
fc_cols = [col for col in df.columns if 'fold' in col.lower() or 'fc' in col.lower()]
pval_cols = [col for col in df.columns if 'p-value' in col.lower() or 'pval' in col.lower()]
mt_cols = [col for col in df.columns if 'mt' in col.lower() or 'melatonin' in col.lower()]
con_cols = [col for col in df.columns if 'con' in col.lower() or 'control' in col.lower()]

print(f"\n   Fold Change columns: {fc_cols}")
print(f"   P-value columns: {pval_cols}")
print(f"   MT-EXO columns: {mt_cols}")
print(f"   Control columns: {con_cols}")

# Check for miRNA identifiers
print("\n🧬 miRNA identifiers sample:")
if 'Transcript' in df.columns:
    print(df['Transcript'].head(10).tolist())
elif 'ProbeID' in df.columns:
    print(df['ProbeID'].head(10).tolist())

print("\n" + "="*80)
