"""
CKD-CVD 타깃 엑소좀: 2D 벡터 스타일 최종 모식도 합성 스크립트
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import matplotlib.patheffects as path_effects
from PIL import Image
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 색상 (플랫 디자인)
COLOR_MIRNA1 = '#F39C12'   # miR-4739 (주황)
COLOR_MIRNA2 = '#2ECC71'   # miR-4651 (초록)
COLOR_EXO = '#3498DB'      # 엑소좀 (파랑)
COLOR_LIGAND = '#8E44AD'   # 리간드 (보라)
COLOR_TEXT = '#2C3E50'     # 텍스트 (진한 회색)

def add_flat_label(ax, x, y, text, color='white', bg_color='#2C3E50', fontsize=10):
    box = FancyBboxPatch((x - 80, y - 20), 160, 40, boxstyle="round,pad=10", 
                         fc=bg_color, ec='none', alpha=0.9)
    ax.add_patch(box)
    ax.text(x, y, text, fontsize=fontsize, color=color, ha='center', va='center', fontweight='bold')

# 1. 이미지 로드
img_path = 'CKD_CVD/CKD_CVD_2D_Vector_Base.png'
try:
    img = Image.open(img_path)
    width, height = img.size
except FileNotFoundError:
    print("이미지를 찾을 수 없습니다.")
    exit()

# Figure 생성
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
ax.imshow(img)
ax.axis('off')

# 좌표 설정 (이미지 분석 기반)
# 세포막: 상단 아치
# 핵: 하단 중앙 (y ~ 0.8*height)
# Mito (Green): 좌측 (x ~ 0.25*width, y ~ 0.6*height)
# NF-kB (Red): 중앙 상단 (x ~ 0.5*width, y ~ 0.4*height)
# Smad (Blue): 우측 (x ~ 0.75*width, y ~ 0.6*height)

# ========================================
# 1. Binding (상단 중앙)
# ========================================
exo_x, exo_y = width * 0.5, height * 0.15
# 엑소좀 (2D Flat Circle)
ax.add_patch(Circle((exo_x, exo_y), width*0.05, facecolor=COLOR_EXO, edgecolor='white', lw=3))
# 리간드 (Y자 결합 표현)
ax.plot([exo_x, exo_x], [exo_y + width*0.05, exo_y + width*0.1], color=COLOR_LIGAND, lw=5)

# 라벨
add_flat_label(ax, exo_x, exo_y - height*0.12, 'Targeted Exosome', bg_color=COLOR_EXO)
ax.text(exo_x + width*0.08, exo_y, 'Specific Binding', fontsize=12, fontweight='bold', color=COLOR_LIGAND)


# ========================================
# 2. Pathways
# ========================================

# A. 미토콘드리아 (좌측 Green) - Energy Recovery
mito_x, mito_y = width * 0.25, height * 0.6
# miR-4739
ax.add_patch(Circle((mito_x, mito_y - height*0.15), width*0.02, color=COLOR_MIRNA1))
ax.text(mito_x, mito_y - height*0.15, 'miR\n4739', ha='center', va='center', color='white', fontsize=9, fontweight='bold')
# 화살표 (활성)
ax.annotate('', xy=(mito_x, mito_y - height*0.05), xytext=(mito_x, mito_y - height*0.12),
            arrowprops=dict(arrowstyle='->', color='#27AE60', lw=4))
# 라벨
add_flat_label(ax, mito_x, mito_y + height*0.1, 'Energy Recovery', bg_color='#27AE60')


# B. 염증 (중앙 Red) - Inflammation Block
inf_x, inf_y = width * 0.5, height * 0.45
# miR-4651
ax.add_patch(Circle((inf_x - width*0.05, inf_y), width*0.02, color=COLOR_MIRNA2))
ax.text(inf_x - width*0.05, inf_y, 'miR\n4651', ha='center', va='center', color='white', fontsize=9, fontweight='bold')
# 화살표 (억제)
ax.annotate('', xy=(inf_x, inf_y), xytext=(inf_x - width*0.03, inf_y),
            arrowprops=dict(arrowstyle='-[', color='#C0392B', lw=4, mutation_scale=15))
# 라벨
add_flat_label(ax, inf_x, inf_y - height*0.12, 'Inflammation Block', bg_color='#C0392B')


# C. 섬유화 (우측 Blue) - Fibrosis Block
fib_x, fib_y = width * 0.75, height * 0.6
# miR-4739
ax.add_patch(Circle((fib_x, fib_y - height*0.15), width*0.02, color=COLOR_MIRNA1))
ax.text(fib_x, fib_y - height*0.15, 'miR\n4739', ha='center', va='center', color='white', fontsize=9, fontweight='bold')
# 화살표 (억제)
ax.annotate('', xy=(fib_x, fib_y - height*0.05), xytext=(fib_x, fib_y - height*0.12),
            arrowprops=dict(arrowstyle='-[', color='#2980B9', lw=4, mutation_scale=15))
# 라벨
add_flat_label(ax, fib_x, fib_y + height*0.1, 'Fibrosis Block', bg_color='#2980B9')


# ========================================
# 3. Nucleus (하단)
# ========================================
nuc_x, nuc_y = width * 0.5, height * 0.85
add_flat_label(ax, nuc_x, nuc_y, 'Gene Regulation', bg_color='#7F8C8D')
ax.text(nuc_x, nuc_y + height*0.08, 'Therapeutic Effect: CKD & CVD Prevention', 
        ha='center', fontsize=12, fontweight='bold', color='#2C3E50')

# 저장
plt.tight_layout()
plt.savefig('CKD_CVD/CKD_CVD_2D_Vector_Final.png', dpi=300, bbox_inches='tight', pad_inches=0)
print("2D 벡터 스타일 모식도 생성 완료: CKD_CVD/CKD_CVD_2D_Vector_Final.png")
