"""
CKD-CVD 타깃 엑소좀: 학회 발표용 전문 의학 일러스트레이션 (완전 재작성)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch, Rectangle
import matplotlib.patheffects as path_effects
from PIL import Image
import numpy as np

# 폰트 설정
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# 색상
COLOR_MIRNA1 = '#E67E22'
COLOR_MIRNA2 = '#27AE60'
COLOR_INHIBIT = '#C0392B'
COLOR_ACTIVATE = '#2ECC71'

# 1. 이미지 로드
img_path = 'CKD_CVD/CKD_CVD_Correct_Cell_Base.png'
try:
    img = Image.open(img_path)
    w, h = img.size
except FileNotFoundError:
    print("Base image not found.")
    exit()

# Figure 생성
fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(111)
ax.imshow(img)
ax.axis('off')

# ========================================
# 1. Binding Site - 우측 상단 확대 박스
# ========================================
box_x, box_y = w*0.7, h*0.05
box_w, box_h = w*0.28, h*0.25

# 박스
ax.add_patch(Rectangle((box_x, box_y), box_w, box_h,
                       facecolor='white', edgecolor='#3498DB', linewidth=4, alpha=1.0, zorder=10))

# 제목
ax.text(box_x + box_w/2, box_y + box_h*0.15, 'Binding Site', 
        ha='center', fontsize=16, fontweight='bold', color='#3498DB', zorder=11)

# 엑소좀 리간드
ax.add_patch(Rectangle((box_x + box_w*0.15, box_y + box_h*0.35), box_w*0.7, box_h*0.12,
                       facecolor='#8E44AD', edgecolor='none', alpha=1.0, zorder=11))
ax.text(box_x + box_w/2, box_y + box_h*0.41, 'Exosome Ligand', 
        ha='center', fontsize=13, fontweight='bold', color='white', zorder=12)

# 화살표
ax.text(box_x + box_w/2, box_y + box_h*0.55, '↓', 
        ha='center', fontsize=20, fontweight='bold', color='#2C3E50', zorder=11)

# 세포 수용체
ax.add_patch(Rectangle((box_x + box_w*0.15, box_y + box_h*0.7), box_w*0.7, box_h*0.12,
                       facecolor='#F39C12', edgecolor='none', alpha=1.0, zorder=11))
ax.text(box_x + box_w/2, box_y + box_h*0.76, 'Cell Receptor', 
        ha='center', fontsize=13, fontweight='bold', color='white', zorder=12)

# 메인 이미지 연결
ax.annotate('', xy=(box_x, box_y + box_h/2), xytext=(w*0.5 + w*0.05, h*0.15),
            arrowprops=dict(arrowstyle='->', color='#3498DB', lw=3, linestyle='--', zorder=9))

# ========================================
# 2. Exosome 라벨
# ========================================
ax.text(w*0.45, h*0.1, 'Targeted\nExosome', 
        ha='center', fontsize=18, fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#3498DB', edgecolor='white', linewidth=3, alpha=1.0))

# ========================================
# 3. miRNA Release
# ========================================
ax.text(w*0.15, h*0.32, 'miRNA\nCargo\nRelease', 
        ha='center', fontsize=16, fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#F39C12', edgecolor='white', linewidth=3, alpha=1.0))

# ========================================
# 4. Pathways (크고 명확하게)
# ========================================

# A. Inflammation - 좌측
inf_x, inf_y = w*0.25, h*0.52

# miRNA node
ax.add_patch(Circle((inf_x - w*0.1, inf_y - h*0.13), w*0.025,
                    facecolor=COLOR_MIRNA2, edgecolor='white', linewidth=3))
ax.text(inf_x - w*0.1, inf_y - h*0.13, 'miR\n4651', 
        ha='center', va='center', fontsize=11, color='white', fontweight='bold')

# 억제 화살표
ax.annotate('', xy=(inf_x - w*0.01, inf_y - h*0.02), xytext=(inf_x - w*0.07, inf_y - h*0.1),
            arrowprops=dict(arrowstyle='-[', color=COLOR_INHIBIT, lw=5, mutation_scale=18))

# 경로명
ax.text(inf_x, inf_y - h*0.2, 'NF-κB\nInhibition', 
        ha='center', fontsize=16, fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.8', facecolor=COLOR_INHIBIT, edgecolor='white', linewidth=3, alpha=1.0))

# 결과
ax.text(inf_x, inf_y + h*0.12, 'Anti-Inflammation\nTNFα ↓  IL-6 ↓', 
        ha='center', fontsize=14, fontweight='bold', color='#2C3E50',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor=COLOR_INHIBIT, linewidth=3, alpha=1.0))


# B. Fibrosis - 우측
fib_x, fib_y = w*0.75, h*0.52

# miRNA node
ax.add_patch(Circle((fib_x + w*0.1, fib_y - h*0.13), w*0.025,
                    facecolor=COLOR_MIRNA1, edgecolor='white', linewidth=3))
ax.text(fib_x + w*0.1, fib_y - h*0.13, 'miR\n4739', 
        ha='center', va='center', fontsize=11, color='white', fontweight='bold')

# 억제 화살표
ax.annotate('', xy=(fib_x + w*0.01, fib_y - h*0.02), xytext=(fib_x + w*0.07, fib_y - h*0.1),
            arrowprops=dict(arrowstyle='-[', color='#D35400', lw=5, mutation_scale=18))

# 경로명
ax.text(fib_x, fib_y - h*0.2, 'TGF-β/Smad\nBlock', 
        ha='center', fontsize=16, fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#D35400', edgecolor='white', linewidth=3, alpha=1.0))

# 결과
ax.text(fib_x, fib_y + h*0.12, 'Anti-Fibrosis\nCOL1A1 ↓  αSMA ↓', 
        ha='center', fontsize=14, fontweight='bold', color='#2C3E50',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='#D35400', linewidth=3, alpha=1.0))


# C. Mitochondrial - 중앙 하단
mito_x, mito_y = w*0.5, h*0.7

# miRNA node
ax.add_patch(Circle((mito_x, mito_y - h*0.15), w*0.025,
                    facecolor=COLOR_MIRNA1, edgecolor='white', linewidth=3))
ax.text(mito_x, mito_y - h*0.15, 'miR\n4739', 
        ha='center', va='center', fontsize=11, color='white', fontweight='bold')

# 활성화 화살표
ax.annotate('', xy=(mito_x, mito_y - h*0.02), xytext=(mito_x, mito_y - h*0.12),
            arrowprops=dict(arrowstyle='->', color=COLOR_ACTIVATE, lw=5))

# 경로명
ax.text(mito_x, mito_y - h*0.22, 'Mitochondrial\nProtection', 
        ha='center', fontsize=16, fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.8', facecolor=COLOR_ACTIVATE, edgecolor='white', linewidth=3, alpha=1.0))

# 결과
ax.text(mito_x, mito_y + h*0.1, 'Antioxidant\nROS ↓  ΔΨm ↑', 
        ha='center', fontsize=14, fontweight='bold', color='#2C3E50',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor=COLOR_ACTIVATE, linewidth=3, alpha=1.0))

# ========================================
# 5. 하단 정보
# ========================================

# Gene Regulation
ax.text(w*0.3, h*0.92, 'Gene Regulation', 
        ha='center', fontsize=16, fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#8E44AD', edgecolor='white', linewidth=3, alpha=1.0))

# Outcome
ax.text(w*0.7, h*0.92, 'CKD Block  |  CVD Prevention', 
        ha='center', fontsize=16, fontweight='bold', color='#2C3E50',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#ECF0F1', edgecolor='#34495E', linewidth=3, alpha=1.0))

# 저장
plt.tight_layout()
plt.savefig('CKD_CVD/CKD_CVD_Professional_Journal_Figure.png', dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
print("✓ 학회 발표용 모식도 생성 완료 (박스 문제 해결)")
