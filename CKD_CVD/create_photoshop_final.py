"""
CKD-CVD 타깃 엑소좀: 포토샵 스타일 3D 이미지 + 정보 오버레이 합성 스크립트
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import matplotlib.patheffects as path_effects
from PIL import Image
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 색상
COLOR_MIRNA1 = '#F39C12'   # miR-4739 (주황)
COLOR_MIRNA2 = '#2ECC71'   # miR-4651 (초록)
COLOR_TEXT_STROKE = 'black'

def add_text(ax, x, y, text, color='white', size=12, weight='bold', ha='center'):
    txt = ax.text(x, y, text, fontsize=size, color=color, fontweight=weight, ha=ha)
    txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])

# 1. 이미지 로드
img_path = 'CKD_CVD/CKD_CVD_Photoshop_Base.png'
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

# 좌표계: (0,0) 좌상단, (width, height) 우하단
# 이미지 분석 기반 좌표 설정 (대략적 위치)
# 엑소좀: 상단 중앙 우측 (width*0.65, height*0.15)
# 미토콘드리아: 좌측 중단 (width*0.2, height*0.5)
# NF-kB: 우측 중단 (width*0.65, height*0.5)
# 핵: 하단 중앙 (width*0.5, height*0.85)

exo_x, exo_y = width * 0.62, height * 0.15
mito_x, mito_y = width * 0.25, height * 0.55
nfkb_x, nfkb_y = width * 0.65, height * 0.55
nuc_x, nuc_y = width * 0.5, height * 0.85

# ========================================
# 1. Binding (상단)
# ========================================
add_text(ax, exo_x, exo_y - height*0.12, 'Targeted Exosome', size=16, color='#AED6F1')
add_text(ax, exo_x, exo_y + height*0.12, 'Specific Receptor Binding', size=12, color='#D7BDE2')

# ========================================
# 2. miRNA Release (세포질)
# ========================================
rel_x, rel_y = width * 0.45, height * 0.35
add_text(ax, rel_x, rel_y, 'Cargo Release', size=14, color='white')

# miR-4739 (주황)
ax.add_patch(Circle((rel_x - width*0.03, rel_y + height*0.05), width*0.01, color=COLOR_MIRNA1))
add_text(ax, rel_x - width*0.03, rel_y + height*0.09, 'miR-4739', size=12, color=COLOR_MIRNA1)

# miR-4651 (초록)
ax.add_patch(Circle((rel_x + width*0.03, rel_y + height*0.05), width*0.01, color=COLOR_MIRNA2))
add_text(ax, rel_x + width*0.03, rel_y + height*0.09, 'miR-4651', size=12, color=COLOR_MIRNA2)

# ========================================
# 3. Pathways
# ========================================

# A. Inflammation (NF-kB) - 우측 붉은 단백질
# miR-4651 -> NF-kB
ax.annotate('', xy=(nfkb_x, nfkb_y), xytext=(rel_x + width*0.05, rel_y + height*0.08),
            arrowprops=dict(arrowstyle='-[', color=COLOR_MIRNA2, lw=4, mutation_scale=10))
add_text(ax, nfkb_x, nfkb_y - height*0.1, 'Inflammation Block\n(NF-κB 억제)', size=14, color='#FF9999')

# B. Antioxidant (Mitochondria) - 좌측 녹색 소기관
# miR-4739 -> Mito
ax.annotate('', xy=(mito_x, mito_y - height*0.05), xytext=(rel_x - width*0.05, rel_y + height*0.08),
            arrowprops=dict(arrowstyle='->', color=COLOR_MIRNA1, lw=4))
add_text(ax, mito_x, mito_y - height*0.15, 'Mitochondrial Protection\n(ROS 감소, 기능 회복)', size=14, color='#99FF99')

# C. Fibrosis (Nucleus/General) - 핵으로 가는 신호 차단
# miR-4739 -> TGF-b (가상의 경로 표시)
fib_x, fib_y = width * 0.85, height * 0.4
add_text(ax, fib_x, fib_y, 'Fibrosis Block\n(TGF-β/Smad)', size=14, color='#FFCC99')
ax.annotate('', xy=(fib_x - width*0.05, fib_y), xytext=(rel_x + width*0.05, rel_y),
            arrowprops=dict(arrowstyle='-[', color=COLOR_MIRNA1, lw=3, linestyle='dashed'))

# ========================================
# 4. Nucleus (Gene Regulation)
# ========================================
add_text(ax, nuc_x, nuc_y, 'Therapeutic Gene Regulation', size=16, color='#E1BEE7')
add_text(ax, nuc_x, nuc_y + height*0.05, 'CKD 진행 억제 & CVD 예방', size=14, color='white')

# ========================================
# 5. Synergy Badge
# ========================================
# 우측 하단
syn_x, syn_y = width * 0.85, height * 0.8
ax.add_patch(FancyBboxPatch((syn_x - width*0.1, syn_y - height*0.1), width*0.2, height*0.2,
                            boxstyle="round,pad=0.05", fc='white', ec='#3498DB', alpha=0.8))
add_text(ax, syn_x, syn_y - height*0.05, 'Synergistic Effect', size=14, color='#2E86C1')
add_text(ax, syn_x, syn_y + height*0.02, '● Kidney Protection\n● Heart Protection', size=11, color='black', weight='normal')

# 저장
plt.tight_layout()
plt.savefig('CKD_CVD/CKD_CVD_Photoshop_Final.png', dpi=300, bbox_inches='tight', pad_inches=0)
print("최종 포토샵 스타일 모식도 생성 완료: CKD_CVD/CKD_CVD_Photoshop_Final.png")
