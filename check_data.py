"""
현재 데이터셋 분석 및 학습 준비
"""
import pandas as pd
from pathlib import Path

# 데이터 로드
df = pd.read_csv('dataset_manifest.csv')

print("=" * 80)
print("📊 현재 데이터셋 분석")
print("=" * 80 + "\n")

# 클래스 분포
print("클래스별 분포:")
print(df['label'].value_counts())

print(f"\n총 클래스 수: {df['label'].nunique()}")
print(f"총 이미지 수: {len(df)}")

# Split 분포
print("\n데이터 분할:")
print(df['split'].value_counts())

# 클래스 목록
print("\n전체 클래스 목록:")
for idx, label in enumerate(sorted(df['label'].unique())):
    print(f"  {idx}: {label}")

print("\n" + "=" * 80)
