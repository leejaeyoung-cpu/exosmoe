"""
CKD-CVD 타깃 엑소좀: 포토샵 스타일 베이스 + 구조화된 정보 합성 스크립트
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch, Wedge
import matplotlib.patheffects as path_effects
from PIL import Image
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 색상
COLOR_MIRNA1 = '#F39C12'   # miR-4739 (주황)
COLOR_MIRNA2 = '#2ECC71'   # miR-4651 (초록)
COLOR_EXO = '#3498DB'      # 엑소좀 (파랑)
COLOR_LIGAND = '#8E44AD'   # 리간드 (보라)

def add_text_box(ax, x, y, title, content, color, width_scale=0.15, height_scale=0.08):
    # 제목
    title_txt = ax.text(x, y, title, fontsize=13, fontweight='bold', color=color, ha='center')
    title_txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
    
    # 내용 박스 (Why)
    box = FancyBboxPatch((x - width_scale/2*img_width, y - height_scale*img_height - 0.02*img_height),
                         width_scale*img_width, height_scale*img_height,
                         boxstyle="round,pad=0.02", fc='white', ec=color, alpha=0.9)
    ax.add_patch(box)
    
    content_txt = ax.text(x, y - height_scale/2*img_height - 0.02*img_height, content, 
                          fontsize=10, color='#555', ha='center', va='center')

# 1. 이미지 로드
img_path = 'CKD_CVD/CKD_CVD_Photoshop_Structured_Base.png'
try:
    img = Image.open(img_path)
    img_width, img_height = img.size
except FileNotFoundError:
    print("이미지를 찾을 수 없습니다.")
    exit()

# Figure 생성
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
ax.imshow(img)
ax.axis('off')

# 좌표 설정 (이미지 분석 기반)
# 세포막: 상단 아치 (y ~ 0.2*height)
# 핵: 하단 중앙 (y ~ 0.8*height)
# NF-kB (Red): 좌측 (x ~ 0.25*width, y ~ 0.5*height)
# Mito (Green): 중앙 (x ~ 0.5*width, y ~ 0.6*height)
# Smad (Blue): 우측 (x ~ 0.75*width, y ~ 0.5*height)

# ========================================
# 1. Binding (상단 중앙)
# ========================================
# 엑소좀 그리기 (이미지에 없으므로)
exo_x, exo_y = img_width * 0.5, img_height * 0.15
ax.add_patch(Circle((exo_x, exo_y), img_width*0.04, facecolor=COLOR_EXO, edgecolor='white', lw=2, alpha=0.9))
# 리간드
ax.plot([exo_x, exo_x], [exo_y + img_width*0.04, exo_y + img_width*0.08], color=COLOR_LIGAND, lw=4)

# 텍스트
add_text_box(ax, exo_x, exo_y - img_height*0.1, 
             '1. 정밀 타겟팅', 'Why? 신장/심장 정확 배달', COLOR_EXO)


# ========================================
# 2. Pathways
# ========================================

# A. 염증 (좌측 Red)
inf_x, inf_y = img_width * 0.25, img_height * 0.45
# miR-4651 아이콘
ax.add_patch(Circle((inf_x, inf_y - img_height*0.15), img_width*0.015, color=COLOR_MIRNA2))
ax.text(inf_x, inf_y - img_height*0.15, 'miR\n4651', ha='center', va='center', color='white', fontsize=9, fontweight='bold')
# 화살표 (억제)
ax.annotate('', xy=(inf_x, inf_y), xytext=(inf_x, inf_y - img_height*0.12),
            arrowprops=dict(arrowstyle='-[', color='#C0392B', lw=4, mutation_scale=15))
# 텍스트
add_text_box(ax, inf_x, inf_y - img_height*0.25, 
             '2. 염증 차단', 'Why? 사이토카인 폭풍 방지', '#C0392B')


# B. 미토콘드리아 (중앙 Green)
mito_x, mito_y = img_width * 0.5, img_height * 0.6
# miR-4739 아이콘
ax.add_patch(Circle((mito_x, mito_y - img_height*0.15), img_width*0.015, color=COLOR_MIRNA1))
ax.text(mito_x, mito_y - img_height*0.15, 'miR\n4739', ha='center', va='center', color='white', fontsize=9, fontweight='bold')
# 화살표 (활성)
ax.annotate('', xy=(mito_x, mito_y), xytext=(mito_x, mito_y - img_height*0.12),
            arrowprops=dict(arrowstyle='->', color='#27AE60', lw=4))
# 텍스트
add_text_box(ax, mito_x, mito_y - img_height*0.25, 
             '4. 에너지 회복', 'Why? 세포 활력 회복', '#27AE60')


# C. 섬유화 (우측 Blue)
fib_x, fib_y = img_width * 0.75, img_height * 0.45
# miR-4739 아이콘
ax.add_patch(Circle((fib_x, fib_y - img_height*0.15), img_width*0.015, color=COLOR_MIRNA1))
ax.text(fib_x, fib_y - img_height*0.15, 'miR\n4739', ha='center', va='center', color='white', fontsize=9, fontweight='bold')
# 화살표 (억제)
ax.annotate('', xy=(fib_x, fib_y), xytext=(fib_x, fib_y - img_height*0.12),
            arrowprops=dict(arrowstyle='-[', color='#2980B9', lw=4, mutation_scale=15))
# 텍스트
add_text_box(ax, fib_x, fib_y - img_height*0.25, 
             '3. 섬유화 억제', 'Why? 장기 경화 방지', '#2980B9')


# ========================================
# 3. Nucleus (하단)
# ========================================
nuc_x, nuc_y = img_width * 0.5, img_height * 0.85
# 텍스트
add_text_box(ax, nuc_x, nuc_y + img_height*0.05, 
             '5. 유전자 조절', 'Why? 근본 원인 교정', '#8E44AD')

# 저장
plt.tight_layout()
plt.savefig('CKD_CVD/CKD_CVD_Photoshop_Structured_Final.png', dpi=300, bbox_inches='tight', pad_inches=0)
print("포토샵 스타일 구조화 모식도 생성 완료: CKD_CVD/CKD_CVD_Photoshop_Structured_Final.png")
