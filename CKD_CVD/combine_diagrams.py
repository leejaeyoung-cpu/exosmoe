"""
CKD-CVD 타깃 엑소좀: 3D 배경 이미지와 시그널 패스웨이 합성 스크립트
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
from PIL import Image

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 색상 팔레트 (가독성을 위해 밝은 색이나 테두리 사용)
COLOR_TEXT = 'white'
COLOR_TEXT_STROKE = 'black'
COLOR_MIRNA1 = '#F39C12'   # miR-4739 (밝은 주황)
COLOR_MIRNA2 = '#2ECC71'   # miR-4651 (밝은 초록)
COLOR_INHIBIT = '#E74C3C'  # 억제 (빨강)

def text_with_stroke(x, y, text, fontsize, color, ax, ha='center', va='center', fontweight='bold'):
    import matplotlib.patheffects as path_effects
    txt = ax.text(x, y, text, fontsize=fontsize, color=color, ha=ha, va=va, fontweight=fontweight)
    txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])

# 1. 배경 이미지 로드
img_path = 'CKD_CVD/CKD_CVD_세포_신호전달_배경.png'
try:
    img = Image.open(img_path)
    width, height = img.size
except FileNotFoundError:
    print("배경 이미지를 찾을 수 없습니다. 더미 이미지를 생성합니다.")
    img = np.zeros((1024, 1792, 3), dtype=np.uint8)
    img[:, :, 2] = 100 # Blue tint
    width, height = 1792, 1024

# Figure 생성
fig = plt.figure(figsize=(16, 9)) # 16:9 비율
ax = fig.add_subplot(111)
ax.imshow(img)
ax.axis('off')

# 좌표 정규화 (이미지 크기에 맞게)
# 이미지 구조 가정: 위쪽=세포막, 아래쪽=핵, 중간=세포질
# 좌표계: (0,0)은 좌상단, (width, height)는 우하단

# ========================================
# 2. 요소 오버레이
# ========================================

# A. 엑소좀 진입 (상단 세포막 근처)
# 세포막이 위쪽에 있다고 가정
exo_x, exo_y = width * 0.3, height * 0.15
text_with_stroke(exo_x, exo_y - height*0.05, '타깃 엑소좀\n(Targeted Exosome)', 14, 'white', ax)

# 진입 화살표
arrow_enter = FancyArrowPatch((exo_x, exo_y), (exo_x, exo_y + height*0.15),
                             arrowstyle='->', mutation_scale=25, 
                             color='white', lw=3)
ax.add_patch(arrow_enter)
text_with_stroke(exo_x + width*0.02, exo_y + height*0.08, '내포작용\n(Endocytosis)', 10, 'white', ax, ha='left')


# B. miRNA 방출 (세포질 상단)
rel_x, rel_y = width * 0.3, height * 0.35
text_with_stroke(rel_x, rel_y, 'miRNA 방출', 14, 'yellow', ax)

# miR-4739 (주황)
ax.add_patch(Circle((rel_x - width*0.05, rel_y + height*0.05), width*0.01, color=COLOR_MIRNA1))
text_with_stroke(rel_x - width*0.05, rel_y + height*0.09, 'miR-4739', 12, COLOR_MIRNA1, ax)

# miR-4651 (초록)
ax.add_patch(Circle((rel_x + width*0.05, rel_y + height*0.05), width*0.01, color=COLOR_MIRNA2))
text_with_stroke(rel_x + width*0.05, rel_y + height*0.09, 'miR-4651', 12, COLOR_MIRNA2, ax)


# C. 염증 경로 (NF-kB) - 보통 세포질에 위치
# 배경 이미지에서 붉은색 단백질 구조가 NF-kB일 가능성 높음 (프롬프트 기반)
# 대략 중앙 좌측이나 중앙에 배치
inf_x, inf_y = width * 0.35, height * 0.55
text_with_stroke(inf_x, inf_y, '염증 경로 (Inflammation)', 12, '#FF9999', ax)

# 억제 표시 (miR-4651 -> NF-kB)
ax.annotate('', xy=(inf_x, inf_y + height*0.05), xytext=(rel_x + width*0.05, rel_y + height*0.1),
            arrowprops=dict(arrowstyle='-[', color=COLOR_MIRNA2, lw=4, mutation_scale=10))
text_with_stroke(inf_x + width*0.02, inf_y - height*0.02, 'NF-κB 억제', 12, COLOR_MIRNA2, ax)


# D. 미토콘드리아/항산화 (Mitochondria) - 보통 녹색으로 표현됨
# 좌측 하단이나 우측 하단에 위치
mito_x, mito_y = width * 0.75, height * 0.6
text_with_stroke(mito_x, mito_y, '항산화/미토콘드리아', 12, '#99FF99', ax)

# 기능 회복 표시 (miR-4739 -> Mito)
ax.annotate('', xy=(mito_x, mito_y - height*0.05), xytext=(rel_x - width*0.05, rel_y + height*0.1),
            arrowprops=dict(arrowstyle='->', color=COLOR_MIRNA1, lw=4))
text_with_stroke(mito_x, mito_y + height*0.05, '기능 회복 (Mitophagy)\nROS 감소', 11, COLOR_MIRNA1, ax)


# E. 섬유화 경로 (TGF-b) - 핵 근처나 세포막 수용체
fib_x, fib_y = width * 0.8, height * 0.25
text_with_stroke(fib_x, fib_y, '섬유화 경로 (Fibrosis)', 12, '#FFCC99', ax)

# 억제 표시 (miR-4739 -> TGF-b)
ax.annotate('', xy=(fib_x, fib_y + height*0.05), xytext=(rel_x, rel_y + height*0.05),
            arrowprops=dict(arrowstyle='-[', color=COLOR_MIRNA1, lw=4, mutation_scale=10, linestyle='dashed'))
text_with_stroke(fib_x, fib_y + height*0.1, 'TGF-β/Smad 억제', 12, COLOR_MIRNA1, ax)


# F. 핵 (Nucleus) - 하단 중앙
nuc_x, nuc_y = width * 0.5, height * 0.85
text_with_stroke(nuc_x, nuc_y, '핵 (Nucleus)', 16, '#E1BEE7', ax)

# 유전자 발현 조절 텍스트
text_with_stroke(nuc_x - width*0.15, nuc_y, '염증 유전자 ↓\n(TNFα, IL-6)', 11, '#FF9999', ax)
text_with_stroke(nuc_x + width*0.15, nuc_y, '섬유화 유전자 ↓\n(COL1A1, αSMA)', 11, '#FFCC99', ax)


# 3. 저장
plt.tight_layout()
output_path = 'CKD_CVD/CKD_CVD_MoA_Combined.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
print(f"합성 이미지 생성 완료: {output_path}")
