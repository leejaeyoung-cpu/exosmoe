"""
CKD-CVD 타깃 엑소좀 작용 기전(MoA) 및 시그널 패스웨이 모식도 생성 스크립트
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, PathPatch
from matplotlib.path import Path
import matplotlib.lines as mlines
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 색상 팔레트
COLOR_BG_OUT = '#EBF5FB'   # 세포 외 공간 (연한 파랑)
COLOR_BG_CYTO = '#FDFEFE'  # 세포질 (흰색/아주 연한 회색)
COLOR_BG_NUC = '#F4ECF7'   # 핵 (연한 보라)
COLOR_MEMBRANE = '#AED6F1' # 세포막
COLOR_EXO = '#D6EAF8'      # 엑소좀
COLOR_MIRNA1 = '#E67E22'   # miR-4739 (주황)
COLOR_MIRNA2 = '#27AE60'   # miR-4651 (초록)
COLOR_INHIBIT = '#E74C3C'  # 억제 (빨강)
COLOR_ACTIVATE = '#2980B9' # 활성화 (파랑)

# Figure 생성
fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(111)
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis('off')

# ========================================
# 1. 세포 구조 그리기
# ========================================

# 세포막 (곡선)
verts = [
    (0, 8), (5, 9), (10, 8.5), (15, 9), (20, 8),  # 위쪽 곡선
    (20, 0), (0, 0), (0, 8)                       # 아래쪽 닫기
]
codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3, 
         Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
path = Path(verts, codes)
patch = PathPatch(path, facecolor=COLOR_BG_CYTO, lw=0)
ax.add_patch(patch)

# 세포막 선 (두껍게)
x = np.linspace(0, 20, 100)
y = 8.5 + 0.5 * np.sin(x) 
ax.plot(x, y, color=COLOR_MEMBRANE, linewidth=15, alpha=0.5)
ax.text(18, 9.2, '세포막 (Cell Membrane)', fontsize=12, color='#5D6D7E', fontweight='bold')

# 세포 외 공간 배경
ax.add_patch(Rectangle((0, 8), 20, 4, facecolor=COLOR_BG_OUT, alpha=0.3, zorder=-1))
ax.text(1, 11.5, '세포 외 공간 (Extracellular Space)', fontsize=12, color='#5D6D7E', fontweight='bold')

# 핵 (Nucleus)
nuc_circle = Circle((15, 3), 4, facecolor=COLOR_BG_NUC, edgecolor='#D7BDE2', linewidth=3, alpha=0.5)
ax.add_patch(nuc_circle)
ax.text(15, 6.5, '핵 (Nucleus)', fontsize=14, color='#884EA0', fontweight='bold', ha='center')

# ========================================
# 2. 타깃 엑소좀 및 진입
# ========================================

# 엑소좀 (세포 밖)
exo_x, exo_y = 4, 10.5
exo = Circle((exo_x, exo_y), 1.2, facecolor=COLOR_EXO, edgecolor='#3498DB', linewidth=3)
ax.add_patch(exo)

# 타깃 리간드 (Y자 모양)
for angle in range(0, 360, 45):
    rad = np.radians(angle)
    lx = exo_x + 1.2 * np.cos(rad)
    ly = exo_y + 1.2 * np.sin(rad)
    # Y shape
    ax.plot([lx, lx + 0.2*np.cos(rad)], [ly, ly + 0.2*np.sin(rad)], color='#8E44AD', lw=3)
    ax.plot([lx + 0.2*np.cos(rad), lx + 0.3*np.cos(rad+0.5)], 
            [ly + 0.2*np.sin(rad), ly + 0.3*np.sin(rad+0.5)], color='#8E44AD', lw=3)
    ax.plot([lx + 0.2*np.cos(rad), lx + 0.3*np.cos(rad-0.5)], 
            [ly + 0.2*np.sin(rad), ly + 0.3*np.sin(rad-0.5)], color='#8E44AD', lw=3)

# 엑소좀 내부 miRNA
ax.text(exo_x-0.4, exo_y+0.2, 'miR-4739', fontsize=9, color=COLOR_MIRNA1, fontweight='bold')
ax.text(exo_x-0.4, exo_y-0.2, 'miR-4651', fontsize=9, color=COLOR_MIRNA2, fontweight='bold')

# 타겟팅 설명 박스
ax.text(exo_x, exo_y+1.8, 'CKD-CVD 타깃 엑소좀', ha='center', fontsize=12, fontweight='bold')
bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="#8E44AD", lw=2)
ax.text(exo_x+2.5, exo_y, 'Targeting:\n- 신장 (Kidney)\n- 심장 (Heart)', ha='left', va='center', 
        bbox=bbox_props, fontsize=10)

# 진입 화살표
ax.annotate('', xy=(4, 7.5), xytext=(4, 9.2),
            arrowprops=dict(facecolor='#3498DB', shrink=0.05, lw=0))
ax.text(4.2, 8.5, 'Endocytosis / Fusion', fontsize=10, color='#34495E')

# ========================================
# 3. 세포 내 miRNA 방출
# ========================================

# 방출된 miRNA들
rel_x, rel_y = 4, 6
ax.text(rel_x, rel_y, 'miRNA 방출', ha='center', fontsize=11, fontweight='bold')

# miR-4739 (주황) 경로 시작점
ax.text(6, 6.5, 'miR-4739', color=COLOR_MIRNA1, fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', fc='white', ec=COLOR_MIRNA1))
# miR-4651 (초록) 경로 시작점
ax.text(6, 5.0, 'miR-4651', color=COLOR_MIRNA2, fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', fc='white', ec=COLOR_MIRNA2))

# ========================================
# 4. 시그널 패스웨이 (Signaling Pathways)
# ========================================

# --- A. 염증 경로 (Inflammation) ---
# 위치: 중앙 상단
inf_x, inf_y = 10, 6.5
ax.text(inf_x, inf_y+0.5, '염증 경로 (Inflammation)', fontsize=11, fontweight='bold', color='#C0392B')

# NF-kB complex
ax.add_patch(FancyBboxPatch((inf_x-1, inf_y-1), 2, 1, boxstyle="round,pad=0.1", fc='#FADBD8', ec='#C0392B'))
ax.text(inf_x, inf_y-0.5, 'NF-κB', ha='center', fontweight='bold')

# 화살표: 핵으로 이동
ax.annotate('', xy=(13, 5), xytext=(11, 6),
            arrowprops=dict(facecolor='#C0392B', arrowstyle='->', lw=2))

# 핵 내부: 염증 유전자 전사
ax.text(15, 4.5, 'Inflammatory Genes\n(TNFα, IL-6)', ha='center', fontsize=10, color='#C0392B')

# 억제 표시 (T-bar)
# miR-4651 -> NF-kB
ax.annotate('', xy=(9, 6), xytext=(7, 5.2),
            arrowprops=dict(arrowstyle='-[', color=COLOR_MIRNA2, lw=3, mutation_scale=10))
ax.text(8, 5.8, '억제', color=COLOR_MIRNA2, fontsize=9, fontweight='bold')

# --- B. 섬유화 경로 (Fibrosis) ---
# 위치: 우측 하단 (핵 근처)
fib_x, fib_y = 11, 2.5
ax.text(fib_x, fib_y+1.2, '섬유화 경로 (Fibrosis)', fontsize=11, fontweight='bold', color='#D35400')

# TGF-beta Receptor (세포막에)
ax.plot([10, 10], [8, 9], color='#D35400', lw=4) # 수용체
ax.text(10, 9.2, 'TGF-βR', ha='center', fontsize=9)

# Smad signaling
ax.annotate('', xy=(11, 3.5), xytext=(10, 8),
            arrowprops=dict(facecolor='#D35400', arrowstyle='->', lw=1.5, linestyle='dashed'))
ax.add_patch(FancyBboxPatch((fib_x-0.8, fib_y), 1.6, 0.8, boxstyle="round,pad=0.1", fc='#FDEBD0', ec='#D35400'))
ax.text(fib_x, fib_y+0.4, 'Smad2/3', ha='center', fontweight='bold')

# 핵으로 이동 및 전사
ax.annotate('', xy=(13.5, 3), xytext=(11.8, 2.9),
            arrowprops=dict(facecolor='#D35400', arrowstyle='->', lw=2))
ax.text(15, 2.5, 'Fibrosis Genes\n(COL1A1, αSMA)', ha='center', fontsize=10, color='#D35400')

# 억제 표시 (T-bar)
# miR-4739 -> Smad/TGF-b
ax.annotate('', xy=(10.5, 5.5), xytext=(7, 6.5),
            arrowprops=dict(arrowstyle='-[', color=COLOR_MIRNA1, lw=3, mutation_scale=10))
ax.text(9, 6.2, '억제', color=COLOR_MIRNA1, fontsize=9, fontweight='bold')


# --- C. 항산화/미토콘드리아 (Antioxidant) ---
# 위치: 좌측 하단
mito_x, mito_y = 5, 2
# 미토콘드리아 그리기
mito = FancyBboxPatch((mito_x-1.5, mito_y-0.8), 3, 1.6, boxstyle="round,pad=0.1", fc='#D4EFDF', ec='#27AE60')
ax.add_patch(mito)
# 크리스테 무늬
ax.plot([mito_x-1, mito_x-0.5, mito_x, mito_x+0.5, mito_x+1], 
        [mito_y, mito_y+0.4, mito_y-0.4, mito_y+0.4, mito_y], color='#27AE60', lw=1)
ax.text(mito_x, mito_y-1.2, '미토콘드리아\n(Mitochondria)', ha='center', fontweight='bold')

# ROS 감소 표시
ax.text(mito_x, mito_y+0.3, 'ROS ↓', ha='center', fontsize=12, fontweight='bold', color='#27AE60')
ax.text(mito_x, mito_y-0.3, 'ΔΨm ↑', ha='center', fontsize=12, fontweight='bold', color='#27AE60')

# 작용 표시
# miR-4739 -> Antioxidant pathways (FoxO, Nrf2 등 활성화 또는 손상 억제)
# 여기서는 '보호/회복' 화살표로 표시
ax.annotate('', xy=(mito_x, mito_y+1), xytext=(6.5, 6.3),
            arrowprops=dict(arrowstyle='->', color=COLOR_MIRNA1, lw=3))
ax.text(4.5, 4.5, '기능 회복\n(Mitophagy)', color=COLOR_MIRNA1, fontsize=9, fontweight='bold')


# ========================================
# 5. 범례 및 요약
# ========================================

legend_x, legend_y = 16.5, 10
ax.add_patch(Rectangle((legend_x, legend_y-2.5), 3.2, 3, facecolor='white', edgecolor='gray', alpha=0.9))
ax.text(legend_x+1.6, legend_y+0.2, '범례 (Legend)', ha='center', fontweight='bold')

# miR-4739
ax.add_patch(Circle((legend_x+0.3, legend_y-0.5), 0.15, color=COLOR_MIRNA1))
ax.text(legend_x+0.6, legend_y-0.6, 'miR-4739\n(다중 커버)', fontsize=9, va='center')

# miR-4651
ax.add_patch(Circle((legend_x+0.3, legend_y-1.2), 0.15, color=COLOR_MIRNA2))
ax.text(legend_x+0.6, legend_y-1.3, 'miR-4651\n(염증/내피 특화)', fontsize=9, va='center')

# 억제/활성화
ax.plot([legend_x+0.2, legend_x+0.5], [legend_y-1.8, legend_y-1.8], color=COLOR_INHIBIT, lw=2)
ax.plot([legend_x+0.5], [legend_y-1.8], marker='|', color=COLOR_INHIBIT, markersize=8, markeredgewidth=2)
ax.text(legend_x+0.6, legend_y-1.8, '억제 (Inhibition)', fontsize=9, va='center')

ax.annotate('', xy=(legend_x+0.5, legend_y-2.2), xytext=(legend_x+0.2, legend_y-2.2),
            arrowprops=dict(arrowstyle='->', color=COLOR_ACTIVATE, lw=2))
ax.text(legend_x+0.6, legend_y-2.2, '활성화/이동', fontsize=9, va='center')

plt.tight_layout()
plt.savefig('CKD_CVD/CKD_CVD_MoA_Signaling_Pathway.png', dpi=300, bbox_inches='tight')
print("MoA 다이어그램 생성 완료: CKD_CVD/CKD_CVD_MoA_Signaling_Pathway.png")
