"""
CKD-CVD 타깃 엑소좀: 참조 이미지 스타일(Clean Vector/Semi-3D) 구조화 모식도 생성 스크립트
특징: 바인딩 부위, 시그널 패스웨이, 구조화된 내용, 간단한 이유(Why) 포함
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge, PathPatch
from matplotlib.path import Path
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 색상 팔레트 (참조 이미지와 유사한 톤)
COLOR_MEMBRANE = '#6EBE73'  # 녹색 계열 (참조 이미지 세포막)
COLOR_CYTOPLASM = '#E8F5E9' # 연한 녹색 배경
COLOR_NUCLEUS = '#D7D9DA'   # 회색/은색 (참조 이미지 핵)
COLOR_EXO = '#3498DB'       # 엑소좀 (파랑)
COLOR_LIGAND = '#8E44AD'    # 리간드 (보라)
COLOR_RECEPTOR = '#F1C40F'  # 수용체 (노랑/골드)
COLOR_NODE_RED = '#C0392B'  # 붉은색 노드 (RAS/RAF 계열 느낌 -> 염증)
COLOR_NODE_BLUE = '#2980B9' # 파란색 노드 (PI3K 계열 느낌 -> 섬유화 억제)
COLOR_NODE_PURPLE = '#8E44AD' # 보라색 노드 (STAT/WNT 계열 -> 전사인자)

# Figure 생성
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111)
ax.set_xlim(0, 20)
ax.set_ylim(0, 15)
ax.axis('off')

# ========================================
# 1. 세포 구조 (Cell Structure)
# ========================================

# 세포질 (배경) - 큰 원형
cytoplasm = Circle((10, 5), 9.5, facecolor=COLOR_CYTOPLASM, edgecolor='none', alpha=0.5)
ax.add_patch(cytoplasm)

# 세포막 (Cell Membrane) - 상단 아치형
# 두께감 있는 아치 그리기
theta1, theta2 = 30, 150
membrane_outer = Wedge((10, 5), 9.5, theta1, theta2, width=0.8, facecolor=COLOR_MEMBRANE, edgecolor='#2E7D32', lw=2)
ax.add_patch(membrane_outer)

# 세포막 텍스트
ax.text(2, 13, '표적 세포 (Target Cell)\n(Kidney/Heart)', fontsize=14, fontweight='bold', color='#2E7D32')
ax.text(18, 13, '세포막\nCELL MEMBRANE', fontsize=10, ha='center', color='#2E7D32')

# 핵 (Nucleus) - 하단 중앙 구체
nucleus = Circle((10, 2), 3.5, facecolor='white', edgecolor='#7F8C8D', lw=3) # 3D 느낌을 위해 그라데이션 대신 심플하게
ax.add_patch(nucleus)
# 핵 구멍 표현 (Pores)
import random
random.seed(42)
for _ in range(15):
    rx = 10 + random.uniform(-2.5, 2.5)
    ry = 2 + random.uniform(-2.5, 2.5)
    if (rx-10)**2 + (ry-2)**2 < 3.0**2:
        ax.add_patch(Circle((rx, ry), 0.15, facecolor='#BDC3C7', edgecolor='none'))

ax.text(10, -0.5, '핵\nNUCLEUS', ha='center', fontsize=12, fontweight='bold', color='#555')
ax.text(10, 2, 'DNA / Gene Expression', ha='center', fontsize=10, color='#999')


# ========================================
# 2. 바인딩 부위 (Binding Site)
# ========================================

# 수용체 (Receptor) - 세포막에 박혀있는 형태
rec_x, rec_y = 10, 14.2 # 세포막 상단 중앙
# 수용체 다리
ax.plot([rec_x-0.3, rec_x-0.3], [rec_y-1, rec_y+0.5], color=COLOR_RECEPTOR, lw=5)
ax.plot([rec_x+0.3, rec_x+0.3], [rec_y-1, rec_y+0.5], color=COLOR_RECEPTOR, lw=5)
# 수용체 헤드 (컵 모양) - Wedge 사용
ax.add_patch(Wedge((rec_x, rec_y+0.5), 0.6, 180, 360, width=0.15, color=COLOR_RECEPTOR))

# 엑소좀 (Exosome) - 수용체에 결합
exo_x, exo_y = 10, 16.5
ax.add_patch(Circle((exo_x, exo_y), 1.2, facecolor=COLOR_EXO, edgecolor='#2980B9', lw=2))
# 리간드 (Ligand) - 엑소좀에서 나와 수용체로
ax.plot([exo_x, exo_x], [exo_y-1.2, rec_y+0.5], color=COLOR_LIGAND, lw=4) # 결합

# 라벨 & 이유
ax.text(exo_x, exo_y+1.5, 'Engineered Exosome', ha='center', fontweight='bold', fontsize=12, color=COLOR_EXO)
ax.text(13.5, 16, '1. 정밀 타겟팅 (Targeting)', fontsize=12, fontweight='bold', color='#2C3E50')
ax.text(13.5, 15.3, 'Why? 약물을 질환 부위(신장/심장)로만\n정확하게 배달하기 위해', fontsize=10, color='#555', 
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#BDC3C7', alpha=0.9))


# ========================================
# 3. 내부 시그널 패스웨이 (Internal Pathways)
# ========================================

# --- A. 염증 경로 (Inflammation) - 좌측 ---
# 흐름: miR-4651 -> NF-kB 억제
path_x_inf = 6
ax.text(path_x_inf, 11, '2. 염증 차단 (Anti-Inflammation)', ha='center', fontsize=11, fontweight='bold', color=COLOR_NODE_RED)
ax.text(path_x_inf, 10.3, 'Why? 사이토카인 폭풍을 막아\n조직 손상을 방지하기 위해', ha='center', fontsize=9, color='#555',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=COLOR_NODE_RED, alpha=0.8))

# 노드들
# miR-4651
ax.add_patch(Circle((path_x_inf, 9), 0.6, facecolor='#2ECC71', edgecolor='none'))
ax.text(path_x_inf, 9, 'miR\n4651', ha='center', va='center', color='white', fontweight='bold', fontsize=9)

# NF-kB (Target)
ax.add_patch(Circle((path_x_inf, 6.5), 0.7, facecolor=COLOR_NODE_RED, edgecolor='none'))
ax.text(path_x_inf, 6.5, 'NF-κB', ha='center', va='center', color='white', fontweight='bold')

# 화살표 (억제)
ax.annotate('', xy=(path_x_inf, 7.3), xytext=(path_x_inf, 8.3),
            arrowprops=dict(arrowstyle='-[', color=COLOR_NODE_RED, lw=3, mutation_scale=10))
ax.text(path_x_inf+0.5, 7.8, '억제', fontsize=9, fontweight='bold', color=COLOR_NODE_RED)

# 핵으로의 이동 차단 (X 표시)
ax.annotate('', xy=(8, 4), xytext=(path_x_inf, 5.7),
            arrowprops=dict(arrowstyle='->', color='#BDC3C7', lw=2, linestyle='--'))
ax.text(7.5, 4.8, 'X', fontsize=14, fontweight='bold', color='red')


# --- B. 섬유화 경로 (Fibrosis) - 우측 ---
# 흐름: miR-4739 -> TGF-b/Smad 억제
path_x_fib = 14
ax.text(path_x_fib, 11, '3. 섬유화 억제 (Anti-Fibrosis)', ha='center', fontsize=11, fontweight='bold', color=COLOR_NODE_BLUE)
ax.text(path_x_fib, 10.3, 'Why? 장기가 딱딱하게 굳어\n기능을 잃는 것을 막기 위해', ha='center', fontsize=9, color='#555',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=COLOR_NODE_BLUE, alpha=0.8))

# 노드들
# miR-4739
ax.add_patch(Circle((path_x_fib, 9), 0.6, facecolor='#F39C12', edgecolor='none'))
ax.text(path_x_fib, 9, 'miR\n4739', ha='center', va='center', color='white', fontweight='bold', fontsize=9)

# Smad (Target)
ax.add_patch(Circle((path_x_fib, 6.5), 0.7, facecolor=COLOR_NODE_BLUE, edgecolor='none'))
ax.text(path_x_fib, 6.5, 'Smad\n2/3', ha='center', va='center', color='white', fontweight='bold')

# 화살표 (억제)
ax.annotate('', xy=(path_x_fib, 7.3), xytext=(path_x_fib, 8.3),
            arrowprops=dict(arrowstyle='-[', color=COLOR_NODE_BLUE, lw=3, mutation_scale=10))
ax.text(path_x_fib+0.5, 7.8, '억제', fontsize=9, fontweight='bold', color=COLOR_NODE_BLUE)

# 핵으로의 이동 차단 (X 표시)
ax.annotate('', xy=(12, 4), xytext=(path_x_fib, 5.7),
            arrowprops=dict(arrowstyle='->', color='#BDC3C7', lw=2, linestyle='--'))
ax.text(12.5, 4.8, 'X', fontsize=14, fontweight='bold', color='red')


# --- C. 미토콘드리아 (Mitochondria) - 중앙 ---
# 흐름: miR-4739 -> 기능 회복
path_x_mito = 10
ax.text(path_x_mito, 8.5, '4. 에너지 회복 (Antioxidant)', ha='center', fontsize=11, fontweight='bold', color='#27AE60')
ax.text(path_x_mito, 7.8, 'Why? 세포의 에너지 공장을 고쳐\n활력을 되찾기 위해', ha='center', fontsize=9, color='#555',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#27AE60', alpha=0.8))

# 미토콘드리아 그림
ax.add_patch(FancyBboxPatch((9, 5.5), 2, 1, boxstyle="round,pad=0.1", fc='#D4EFDF', ec='#27AE60'))
ax.plot([9.3, 9.6, 10, 10.4, 10.7], [5.5, 6.0, 5.0, 6.0, 5.5], color='#27AE60', lw=1) # 크리스테

# 화살표 (활성화)
ax.annotate('', xy=(10, 6.7), xytext=(13.2, 8.5), # miR-4739에서 옴
            arrowprops=dict(arrowstyle='->', color='#F39C12', lw=2, linestyle=':')) 
ax.text(11.5, 7.2, '보호', fontsize=9, color='#F39C12', fontweight='bold')


# ========================================
# 4. 핵 내부 (Gene Regulation)
# ========================================
# DNA 나선
x = np.linspace(8, 12, 100)
y1 = 2 + 0.3 * np.sin(x * 5)
y2 = 2 + 0.3 * np.sin(x * 5 + np.pi)
ax.plot(x, y1, color='#3498DB', lw=2, alpha=0.7)
ax.plot(x, y2, color='#E74C3C', lw=2, alpha=0.7)

# 결과 텍스트
ax.text(8, 2.5, '염증 유전자 OFF', fontsize=10, color='#C0392B', fontweight='bold')
ax.text(12, 2.5, '섬유화 유전자 OFF', fontsize=10, color='#2980B9', fontweight='bold')
ax.text(10, 1.0, '세포 정상화 (Normalization)', ha='center', fontsize=11, fontweight='bold', color='#2C3E50')


plt.tight_layout()
plt.savefig('CKD_CVD/CKD_CVD_Structured_Mechanism.png', dpi=300, bbox_inches='tight')
print("구조화된 모식도 생성 완료: CKD_CVD/CKD_CVD_Structured_Mechanism.png")
