"""
CKD-CVD 타깃 엑소좀: 논문 스타일(2D Flat) 고품질 모식도 생성 스크립트
강조점: 정확한 바인딩, 물질 종류, 경로, 시너지 효과, 간결함
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Arc, PathPatch
from matplotlib.path import Path
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 색상 팔레트 (논문 스타일: 차분하고 명확한 색상)
COLOR_EXO_MEM = '#2874A6'   # 엑소좀 막 (진한 파랑)
COLOR_EXO_FILL = '#D6EAF8'  # 엑소좀 내부 (연한 파랑)
COLOR_CELL_MEM = '#95A5A6'  # 세포막 (회색)
COLOR_LIGAND = '#8E44AD'    # 리간드 (보라)
COLOR_RECEPTOR = '#F1C40F'  # 수용체 (노랑)
COLOR_MIRNA1 = '#E67E22'    # miR-4739 (주황)
COLOR_MIRNA2 = '#27AE60'    # miR-4651 (초록)
COLOR_PATH_INF = '#E74C3C'  # 염증 경로 (빨강)
COLOR_PATH_FIB = '#D35400'  # 섬유화 경로 (주황갈색)
COLOR_PATH_MITO = '#2ECC71' # 미토콘드리아 (초록)

# Figure 생성
fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(111)
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis('off')

# ========================================
# 1. 구조: 엑소좀 - 바인딩 - 세포
# ========================================

# 세포막 (수직선으로 표현하여 간결하게)
ax.add_patch(Rectangle((6, 0), 0.5, 10, facecolor=COLOR_CELL_MEM, alpha=0.3))
ax.plot([6, 6], [0, 10], color=COLOR_CELL_MEM, lw=2)
ax.plot([6.5, 6.5], [0, 10], color=COLOR_CELL_MEM, lw=2)
ax.text(6.25, 9.5, 'Cell Membrane', ha='center', va='center', rotation=90, fontsize=10, color='#555')

# 엑소좀 (좌측)
exo_center = (3, 5)
ax.add_patch(Circle(exo_center, 1.5, facecolor=COLOR_EXO_FILL, edgecolor=COLOR_EXO_MEM, lw=3))
ax.text(3, 7, 'Engineered Exosome', ha='center', fontweight='bold', fontsize=12, color=COLOR_EXO_MEM)

# 엑소좀 내부 miRNA
ax.text(2.5, 5.2, 'miR-4739', color=COLOR_MIRNA1, fontweight='bold', fontsize=10)
ax.plot([2.5, 2.9], [5.1, 5.1], color=COLOR_MIRNA1, lw=3) # miRNA 모양
ax.text(3.5, 4.8, 'miR-4651', color=COLOR_MIRNA2, fontweight='bold', fontsize=10)
ax.plot([3.1, 3.5], [4.7, 4.7], color=COLOR_MIRNA2, lw=3) # miRNA 모양

# 바인딩 부위 (확대 뷰 느낌)
# 리간드 (엑소좀 표면)
ligand_y = 5
ax.plot([4.5, 5.2], [ligand_y, ligand_y], color=COLOR_LIGAND, lw=4) # 줄기
ax.plot([5.2, 5.5], [ligand_y, ligand_y+0.3], color=COLOR_LIGAND, lw=4) # Y 상단
ax.plot([5.2, 5.5], [ligand_y, ligand_y-0.3], color=COLOR_LIGAND, lw=4) # Y 하단

# 수용체 (세포막)
ax.plot([6, 5.7], [ligand_y, ligand_y], color=COLOR_RECEPTOR, lw=4) # 줄기
ax.add_patch(Circle((5.7, ligand_y), 0.2, color=COLOR_RECEPTOR)) # 헤드

# 바인딩 텍스트
ax.text(5.25, 5.8, 'Specific Binding', ha='center', fontsize=10, fontweight='bold')
ax.text(5.25, 4.2, 'Targeting Ligand\n↔ Receptor', ha='center', fontsize=9, style='italic')


# ========================================
# 2. 세포 내 경로 (우측)
# ========================================

# 구역 나누기 (점선)
ax.plot([7, 17], [6.6, 6.6], color='#DDD', linestyle='--')
ax.plot([7, 17], [3.3, 3.3], color='#DDD', linestyle='--')

# --- A. 염증 (Inflammation) - 상단 ---
path_y_inf = 8.3
ax.text(7.5, path_y_inf, '1. Inflammation', fontsize=12, fontweight='bold', color=COLOR_PATH_INF)

# NF-kB
ax.add_patch(FancyBboxPatch((10, path_y_inf-0.4), 1.5, 0.8, boxstyle="round,pad=0.1", fc='#FADBD8', ec=COLOR_PATH_INF))
ax.text(10.75, path_y_inf, 'NF-κB', ha='center', va='center', fontweight='bold')

# 작용 (miR-4651)
ax.text(8.5, path_y_inf, 'miR-4651', color=COLOR_MIRNA2, fontweight='bold')
ax.annotate('', xy=(10, path_y_inf), xytext=(9.2, path_y_inf),
            arrowprops=dict(arrowstyle='-[', color=COLOR_MIRNA2, lw=3, mutation_scale=10))

# 결과
ax.annotate('', xy=(13, path_y_inf), xytext=(11.8, path_y_inf),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.text(13.2, path_y_inf, 'Cytokines ↓\n(TNFα, IL-6)', va='center', fontsize=11)


# --- B. 섬유화 (Fibrosis) - 중단 ---
path_y_fib = 5
ax.text(7.5, path_y_fib, '2. Fibrosis', fontsize=12, fontweight='bold', color=COLOR_PATH_FIB)

# TGF-b / Smad
ax.add_patch(FancyBboxPatch((10, path_y_fib-0.4), 1.5, 0.8, boxstyle="round,pad=0.1", fc='#FDEBD0', ec=COLOR_PATH_FIB))
ax.text(10.75, path_y_fib, 'TGF-β\nSmad2/3', ha='center', va='center', fontweight='bold', fontsize=9)

# 작용 (miR-4739)
ax.text(8.5, path_y_fib, 'miR-4739', color=COLOR_MIRNA1, fontweight='bold')
ax.annotate('', xy=(10, path_y_fib), xytext=(9.2, path_y_fib),
            arrowprops=dict(arrowstyle='-[', color=COLOR_MIRNA1, lw=3, mutation_scale=10))

# 결과
ax.annotate('', xy=(13, path_y_fib), xytext=(11.8, path_y_fib),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.text(13.2, path_y_fib, 'ECM Deposition ↓\n(COL1A1, αSMA)', va='center', fontsize=11)


# --- C. 항산화 (Antioxidant) - 하단 ---
path_y_mito = 1.7
ax.text(7.5, path_y_mito, '3. Antioxidant', fontsize=12, fontweight='bold', color=COLOR_PATH_MITO)

# Mitochondria
ax.add_patch(FancyBboxPatch((10, path_y_mito-0.4), 1.5, 0.8, boxstyle="round,pad=0.1", fc='#D4EFDF', ec=COLOR_PATH_MITO))
ax.text(10.75, path_y_mito, 'Mito-\nchondria', ha='center', va='center', fontweight='bold', fontsize=9)

# 작용 (miR-4739)
ax.text(8.5, path_y_mito, 'miR-4739', color=COLOR_MIRNA1, fontweight='bold')
ax.annotate('', xy=(10, path_y_mito), xytext=(9.2, path_y_mito),
            arrowprops=dict(arrowstyle='->', color=COLOR_MIRNA1, lw=3)) # 활성화/보호

# 결과
ax.annotate('', xy=(13, path_y_mito), xytext=(11.8, path_y_mito),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.text(13.2, path_y_mito, 'Function ↑\nROS ↓', va='center', fontsize=11)


# ========================================
# 3. 시너지 효과 (Synergy) - 우측 끝
# ========================================

synergy_x = 16
# Rectangle 대신 FancyBboxPatch 사용 (둥근 모서리)
ax.add_patch(FancyBboxPatch((15.5, 0.5), 2, 9, boxstyle="round,pad=0.2", facecolor='#F4F6F7', edgecolor='#BDC3C7'))
ax.text(synergy_x, 9, 'Synergistic\nEffect', ha='center', fontweight='bold', fontsize=12, color='#2C3E50')

# 연결선
ax.plot([15, 15.5], [path_y_inf, 8], color='#BDC3C7', linestyle=':')
ax.plot([15, 15.5], [path_y_fib, 5], color='#BDC3C7', linestyle=':')
ax.plot([15, 15.5], [path_y_mito, 2], color='#BDC3C7', linestyle=':')

# 시너지 내용
ax.text(synergy_x, 7.5, 'CKD\nProgression\nBlock', ha='center', va='center', fontweight='bold', fontsize=10, color='#C0392B')
ax.text(synergy_x, 5, 'Integrated\nTherapy', ha='center', va='center', fontweight='bold', fontsize=11, color='#2E86C1')
ax.text(synergy_x, 2.5, 'CVD\nPrevention', ha='center', va='center', fontweight='bold', fontsize=10, color='#27AE60')

# ========================================
# 4. 범례 (하단)
# ========================================
legend_y = 0.5
ax.text(2, legend_y, 'miR-4739: Multi-target', color=COLOR_MIRNA1, fontweight='bold', fontsize=10)
ax.text(5, legend_y, 'miR-4651: Anti-inflammation', color=COLOR_MIRNA2, fontweight='bold', fontsize=10)
ax.text(9, legend_y, '⊣ : Inhibition', fontweight='bold', fontsize=10)
ax.text(11, legend_y, '→ : Activation/Protection', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('CKD_CVD/CKD_CVD_Mechanism_Paper_Figure.png', dpi=300, bbox_inches='tight')
print("논문 스타일 모식도 생성 완료: CKD_CVD/CKD_CVD_Mechanism_Paper_Figure.png")
