"""
CKD-CVD miRNA 데이터 심층 분석 및 신규 miRNA 설계 가능성 평가
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 경로
data_dir = Path(r'C:\Users\brook\Desktop\mi_exo_ai\CKD_CVD')

# 1. 데이터 로드
print("="*80)
print("CKD-CVD miRNA 데이터 심층 분석")
print("="*80)

# 파일 1: 99개 후보 miRNA
df_candidates = pd.read_excel(data_dir / '1. CKD_CVD_exosome_miRNA_candidates.xlsx')
print(f"\n✅ 파일 1 로드: {len(df_candidates)}개 후보 miRNA")
print(f"   컬럼 수: {len(df_candidates.columns)}")

# 파일 2: 가중치 시스템
df_weights = pd.read_excel(data_dir / '2. CKD_CVD_miRNA_TopN_GoNoGo.xlsx')
print(f"✅ 파일 2 로드: 가중치 시스템 ({len(df_weights)} 카테고리)")

# 파일 3: 최종 칵테일
df_cocktail = pd.read_excel(data_dir / '3. CKD_CVD_final_cocktail_minimal_set.xlsx')
print(f"✅ 파일 3 로드: Core-2 최종 칵테일 ({len(df_cocktail)}개 miRNA)")

print("\n" + "="*80)
print("1. 후보 miRNA 기본 통계")
print("="*80)

# 기본 통계
print(f"\n📊 Fold Change 통계:")
print(f"   평균: {df_candidates['FC_MT_vs_Con'].mean():.2f}")
print(f"   중앙값: {df_candidates['FC_MT_vs_Con'].median():.2f}")
print(f"   최대: {df_candidates['FC_MT_vs_Con'].max():.2f}")
print(f"   최소: {df_candidates['FC_MT_vs_Con'].min():.2f}")
print(f"   표준편차: {df_candidates['FC_MT_vs_Con'].std():.2f}")

# 경로 분석용 컬럼 찾기
pathway_cols = [col for col in df_candidates.columns if '_Npath' in col]
print(f"\n📊 경로 관련 컬럼 ({len(pathway_cols)}개):")
for col in pathway_cols:
    print(f"   - {col}")

# 총 경로 수 계산
if 'total_pathways' not in df_candidates.columns:
    df_candidates['total_pathways'] = df_candidates[pathway_cols].sum(axis=1)

print(f"\n📊 총 경로 수 통계:")
print(f"   평균: {df_candidates['total_pathways'].mean():.2f}")
print(f"   중앙값: {df_candidates['total_pathways'].median():.2f}")
print(f"   최대: {df_candidates['total_pathways'].max():.0f}")
print(f"   최소: {df_candidates['total_pathways'].min():.0f}")

print("\n" + "="*80)
print("2. 상위 miRNA 분석 (FC 기준 Top 20)")
print("="*80)

top20_fc = df_candidates.nlargest(20, 'FC_MT_vs_Con')
print("\n상위 20개 miRNA:")
print(top20_fc[['miRNA', 'FC_MT_vs_Con', 'total_pathways']].to_string(index=False))

print("\n" + "="*80)
print("3. 경로 커버리지 분석 (경로 수 기준 Top 20)")
print("="*80)

top20_path = df_candidates.nlargest(20, 'total_pathways')
print("\n경로 커버리지 상위 20개:")
print(top20_path[['miRNA', 'FC_MT_vs_Con', 'total_pathways']].to_string(index=False))

print("\n" + "="*80)
print("4. 카테고리별 경로 분석")
print("="*80)

# 카테고리별 통계
category_stats = []
for col in pathway_cols:
    category_name = col.split('_Npath')[0].split('·')[0]
    stats = {
        '카테고리': category_name,
        '평균 경로수': df_candidates[col].mean(),
        '최대 경로수': df_candidates[col].max(),
        '상위10 합계': df_candidates.nlargest(10, col)[col].sum()
    }
    category_stats.append(stats)

df_cat_stats = pd.DataFrame(category_stats)
print("\n카테고리별 통계:")
print(df_cat_stats.to_string(index=False))

print("\n" + "="*80)
print("5. 상관관계 분석")
print("="*80)

# FC와 경로 수의 상관관계
correlation = df_candidates['FC_MT_vs_Con'].corr(df_candidates['total_pathways'])
print(f"\nFold Change vs 총 경로 수 상관계수: {correlation:.3f}")

if abs(correlation) < 0.3:
    print("→ 약한 상관관계: FC가 높다고 경로가 많은 것은 아님")
elif abs(correlation) < 0.7:
    print("→ 중간 상관관계: 일부 연관성 있음")
else:
    print("→ 강한 상관관계: FC와 경로 수가 밀접히 연관")

print("\n" + "="*80)
print("6. Core-2 선정 miRNA 상세 분석")
print("="*80)

core2_names = df_cocktail['miRNA'].tolist() if 'miRNA' in df_cocktail.columns else []
print(f"\nCore-2 miRNA: {core2_names}")

for mirna in core2_names[:2]:  # 처음 2개만
    if mirna in df_candidates['miRNA'].values:
        data = df_candidates[df_candidates['miRNA'] == mirna].iloc[0]
        print(f"\n🎯 {mirna}:")
        print(f"   FC: {data['FC_MT_vs_Con']:.2f}")
        print(f"   총 경로: {int(data['total_pathways'])}")
        print(f"   순위 (FC): {df_candidates['FC_MT_vs_Con'].rank(ascending=False)[data.name]:.0f}위 / 99개")
        print(f"   순위 (경로): {df_candidates['total_pathways'].rank(ascending=False)[data.name]:.0f}위 / 99개")

print("\n" + "="*80)
print("7. 신규 miRNA 설계 후보 발굴")
print("="*80)

# 고성능 후보 선별 기준
high_fc_threshold = df_candidates['FC_MT_vs_Con'].quantile(0.75)  # 상위 25%
high_path_threshold = df_candidates['total_pathways'].quantile(0.75)  # 상위 25%

print(f"\n선별 기준:")
print(f"   FC 임계값: {high_fc_threshold:.2f} (상위 25%)")
print(f"   경로 임계값: {high_path_threshold:.0f} (상위 25%)")

# 고성능 후보
high_performers = df_candidates[
    (df_candidates['FC_MT_vs_Con'] >= high_fc_threshold) &
    (df_candidates['total_pathways'] >= high_path_threshold)
]

print(f"\n🔥 고성능 후보 ({len(high_performers)}개):")
print(high_performers[['miRNA', 'FC_MT_vs_Con', 'total_pathways']].to_string(index=False))

# Core-2에 없는 신규 후보
if core2_names:
    novel_candidates = high_performers[~high_performers['miRNA'].isin(core2_names)]
    print(f"\n💡 신규 miRNA 설계 후보 (Core-2에 포함되지 않은 고성능 후보: {len(novel_candidates)}개):")
    if len(novel_candidates) > 0:
        print(novel_candidates[['miRNA', 'FC_MT_vs_Con', 'total_pathways']].to_string(index=False))
    else:
        print("   → Core-2가 이미 최적 조합")

print("\n" + "="*80)
print("8. 균형 분석 (FC vs 경로 스코어)")
print("="*80)

# 정규화
df_candidates['FC_normalized'] = (df_candidates['FC_MT_vs_Con'] - df_candidates['FC_MT_vs_Con'].min()) / \
                                  (df_candidates['FC_MT_vs_Con'].max() - df_candidates['FC_MT_vs_Con'].min())
df_candidates['path_normalized'] = (df_candidates['total_pathways'] - df_candidates['total_pathways'].min()) / \
                                    (df_candidates['total_pathways'].max() - df_candidates['total_pathways'].min())

# 균형 점수 (두 지표의 조화 평균)
df_candidates['balance_score'] = 2 * (df_candidates['FC_normalized'] * df_candidates['path_normalized']) / \
                                  (df_candidates['FC_normalized'] + df_candidates['path_normalized'] + 1e-10)

top10_balanced = df_candidates.nlargest(10, 'balance_score')
print("\n균형 점수 상위 10개 (FC와 경로 모두 고려):")
print(top10_balanced[['miRNA', 'FC_MT_vs_Con', 'total_pathways', 'balance_score']].to_string(index=False))

print("\n" + "="*80)
print("분석 완료!")
print("="*80)

# 결과 저장
output_dir = Path(r'C:\Users\brook\Desktop\mi_exo_ai\CKD_CVD')
df_candidates.to_excel(output_dir / 'CKD_CVD_분석결과_상세.xlsx', index=False)
print(f"\n✅ 상세 분석 결과 저장: {output_dir / 'CKD_CVD_분석결과_상세.xlsx'}")
