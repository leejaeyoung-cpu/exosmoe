"""
CKD-CVD 치료용 타깃 엑소좀 칵테일 개발 전략 모식도 생성 스크립트
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Wedge
import matplotlib.lines as mlines
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False

# 색상 팔레트
COLOR_PRIMARY = '#2E5C8A'  # 진한 파랑
COLOR_SECONDARY = '#3EACB0'  # 청록색
COLOR_ACCENT1 = '#E67E22'  # 주황색
COLOR_ACCENT2 = '#27AE60'  # 초록색
COLOR_LIGHT = '#ECF0F1'  # 연한 회색
COLOR_GRID = '#BDC3C7'  # 중간 회색

# Figure 생성 (큰 캔버스)
fig = plt.figure(figsize=(20, 14))
ax = fig.add_subplot(111)
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')

# ========================================
# 제목
# ========================================
title_text = 'CKD-CVD 치료용 타깃 엑소좀 칵테일 개발 전략'
ax.text(10, 13.3, title_text, ha='center', va='top', 
        fontsize=24, fontweight='bold', color=COLOR_PRIMARY)

# ========================================
# 상단: 전체 프로세스 플로우 (Step-0 → Step-6)
# ========================================

# Step 0-1: 99개 후보 발굴
step1_x, step1_y = 1.5, 11
step1_box = FancyBboxPatch((step1_x-0.7, step1_y-0.8), 1.4, 1.6, 
                           boxstyle="round,pad=0.1", 
                           edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=3)
ax.add_patch(step1_box)
ax.text(step1_x, step1_y+0.4, 'Step-0~1', ha='center', va='center', 
        fontsize=11, fontweight='bold', color=COLOR_PRIMARY)
ax.text(step1_x, step1_y, '초기 후보\n발굴', ha='center', va='center', 
        fontsize=13, fontweight='bold')
ax.text(step1_x, step1_y-0.5, '99개', ha='center', va='center', 
        fontsize=18, fontweight='bold', color=COLOR_ACCENT1)

# 화살표 1
arrow1 = FancyArrowPatch((step1_x+0.8, step1_y), (4.2, step1_y),
                        arrowstyle='->', mutation_scale=30, 
                        linewidth=3, color=COLOR_PRIMARY)
ax.add_patch(arrow1)

# Step 2: 경로 분석 & 가중치
step2_x, step2_y = 5.5, 11
step2_box = FancyBboxPatch((step2_x-1.2, step2_y-0.8), 2.4, 1.6, 
                           boxstyle="round,pad=0.1", 
                           edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=3)
ax.add_patch(step2_box)
ax.text(step2_x, step2_y+0.4, 'Step-2', ha='center', va='center', 
        fontsize=11, fontweight='bold', color=COLOR_PRIMARY)
ax.text(step2_x, step2_y, '경로 분석\n가중치 적용', ha='center', va='center', 
        fontsize=13, fontweight='bold')

# 6개 카테고리 작은 박스
categories = [
    ('염증', 0.25, '#E74C3C'),
    ('섬유화', 0.25, '#E67E22'),
    ('항산화', 0.20, '#F39C12'),
    ('내피', 0.20, '#3498DB'),
    ('CVD', 0.10, '#9B59B6'),
    ('노화', 0.05, '#95A5A6')
]
cat_y_start = step2_y - 0.5
for i, (name, weight, color) in enumerate(categories[:3]):
    x_pos = step2_x - 0.8 + i * 0.8
    rect = Rectangle((x_pos-0.3, cat_y_start-0.15), 0.6, 0.3, 
                     facecolor=color, edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(x_pos, cat_y_start, f'{name}\n{weight}', ha='center', va='center', 
            fontsize=8, color='white', fontweight='bold')

for i, (name, weight, color) in enumerate(categories[3:]):
    x_pos = step2_x - 0.8 + (i+3) * 0.8
    rect = Rectangle((x_pos-0.3, cat_y_start-0.45), 0.6, 0.3, 
                     facecolor=color, edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(x_pos, cat_y_start-0.3, f'{name}\n{weight}', ha='center', va='center', 
            fontsize=8, color='white', fontweight='bold')

# 화살표 2
arrow2 = FancyArrowPatch((step2_x+1.3, step2_y), (8.5, step2_y),
                        arrowstyle='->', mutation_scale=30, 
                        linewidth=3, color=COLOR_PRIMARY)
ax.add_patch(arrow2)

# Step 3-4: Core-2 선정
step3_x, step3_y = 10.5, 11
step3_box = FancyBboxPatch((step3_x-1.2, step3_y-0.8), 2.4, 1.6, 
                           boxstyle="round,pad=0.1", 
                           edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=3)
ax.add_patch(step3_box)
ax.text(step3_x, step3_y+0.4, 'Step-3~4', ha='center', va='center', 
        fontsize=11, fontweight='bold', color=COLOR_PRIMARY)
ax.text(step3_x, step3_y+0.15, 'Core-2 선정', ha='center', va='center', 
        fontsize=13, fontweight='bold')

# miRNA 원형 뱃지
miRNA1_circle = Circle((step3_x-0.5, step3_y-0.35), 0.3, 
                       facecolor='#E67E22', edgecolor='white', linewidth=3)
ax.add_patch(miRNA1_circle)
ax.text(step3_x-0.5, step3_y-0.35, 'miR\n4739', ha='center', va='center', 
        fontsize=8, color='white', fontweight='bold')

miRNA2_circle = Circle((step3_x+0.5, step3_y-0.35), 0.3, 
                       facecolor='#27AE60', edgecolor='white', linewidth=3)
ax.add_patch(miRNA2_circle)
ax.text(step3_x+0.5, step3_y-0.35, 'miR\n4651', ha='center', va='center', 
        fontsize=8, color='white', fontweight='bold')

# 화살표 3
arrow3 = FancyArrowPatch((step3_x+1.3, step3_y), (13.5, step3_y),
                        arrowstyle='->', mutation_scale=30, 
                        linewidth=3, color=COLOR_PRIMARY)
ax.add_patch(arrow3)

# Step 5-6: 타깃 엑소좀
step4_x, step4_y = 15.5, 11
step4_box = FancyBboxPatch((step4_x-1.2, step4_y-0.8), 2.4, 1.6, 
                           boxstyle="round,pad=0.1", 
                           edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=3)
ax.add_patch(step4_box)
ax.text(step4_x, step4_y+0.4, 'Step-5~6', ha='center', va='center', 
        fontsize=11, fontweight='bold', color=COLOR_PRIMARY)
ax.text(step4_x, step4_y+0.15, '타깃 엑소좀\n제작', ha='center', va='center', 
        fontsize=13, fontweight='bold')

# 엑소좀 단순 표현 (원 + 표면 마커)
exo_circle = Circle((step4_x, step4_y-0.35), 0.35, 
                    facecolor=COLOR_SECONDARY, edgecolor=COLOR_PRIMARY, linewidth=3, alpha=0.7)
ax.add_patch(exo_circle)
# 표면 타깃팅 펩타이드 (작은 삼각형)
for angle in [0, 60, 120, 180, 240, 300]:
    rad = np.radians(angle)
    x_tri = step4_x + 0.35 * np.cos(rad)
    y_tri = step4_y - 0.35 + 0.35 * np.sin(rad)
    triangle = mpatches.RegularPolygon((x_tri, y_tri), 3, radius=0.08, 
                                       facecolor='#E74C3C', edgecolor='white', linewidth=1)
    ax.add_patch(triangle)

# 내부 miRNA 표시
ax.text(step4_x, step4_y-0.35, 'miRNA\nCargo', ha='center', va='center', 
        fontsize=8, color='white', fontweight='bold')

# ========================================
# 중단: 선별 근거 및 기준
# ========================================

# 제목
ax.text(10, 8.8, '선별 근거 및 기준', ha='center', va='center', 
        fontsize=18, fontweight='bold', color=COLOR_PRIMARY)

# 3개 박스: 발현 데이터, 경로 커버리지, 기능 검증
criteria_y = 7.5
criteria_boxes = [
    ('발현 데이터', ['FC > 30', 'MT_mean 양호', '재현성 확보'], 3),
    ('경로 커버리지', ['총 경로 > 50개', '6개 카테고리 균형', '가중치 점수화'], 10),
    ('기능 검증', ['Primary\nreadouts 설정', 'In vitro 검증', 'Go/No-Go 기준'], 17)
]

for title, items, x_pos in criteria_boxes:
    box = FancyBboxPatch((x_pos-1.8, criteria_y-1.2), 3.6, 2.4, 
                         boxstyle="round,pad=0.15", 
                         edgecolor=COLOR_SECONDARY, facecolor='white', linewidth=2.5)
    ax.add_patch(box)
    
    # 박스 제목
    title_bg = Rectangle((x_pos-1.65, criteria_y+0.8), 3.3, 0.35, 
                         facecolor=COLOR_SECONDARY, edgecolor='none')
    ax.add_patch(title_bg)
    ax.text(x_pos, criteria_y+0.97, title, ha='center', va='center', 
            fontsize=13, fontweight='bold', color='white')
    
    # 항목들
    for i, item in enumerate(items):
        y_offset = criteria_y + 0.3 - i * 0.55
        ax.plot([x_pos-1.5, x_pos-1.3], [y_offset, y_offset], 
               color=COLOR_ACCENT1, linewidth=3)
        ax.text(x_pos-1.1, y_offset, item, ha='left', va='center', 
               fontsize=10)

# ========================================
# 하단: 타깃 엑소좀 전략
# ========================================

# 제목
ax.text(10, 4.8, '타깃 엑소좀 전략', ha='center', va='center', 
        fontsize=18, fontweight='bold', color=COLOR_PRIMARY)

# 중앙 큰 엑소좀
exo_main_x, exo_main_y = 10, 3
exo_main = Circle((exo_main_x, exo_main_y), 1.2, 
                  facecolor='#D5DBDB', edgecolor=COLOR_PRIMARY, linewidth=4, alpha=0.8)
ax.add_patch(exo_main)

# 외부 막 (이중층 표현)
exo_outer = Circle((exo_main_x, exo_main_y), 1.2, 
                   fill=False, edgecolor=COLOR_PRIMARY, linewidth=6)
ax.add_patch(exo_outer)
exo_inner = Circle((exo_main_x, exo_main_y), 1.15, 
                   fill=False, edgecolor=COLOR_PRIMARY, linewidth=2, linestyle='--')
ax.add_patch(exo_inner)

# 표면 타깃팅 펩타이드 (더 많이)
for angle in range(0, 360, 30):
    rad = np.radians(angle)
    x_pep = exo_main_x + 1.2 * np.cos(rad)
    y_pep = exo_main_y + 1.2 * np.sin(rad)
    
    # Y자 형태 타깃팅 마커
    pep_line = mlines.Line2D([x_pep, x_pep + 0.3*np.cos(rad)], 
                            [y_pep, y_pep + 0.3*np.sin(rad)],
                            linewidth=3, color='#E74C3C')
    ax.add_line(pep_line)
    
    pep_circle = Circle((x_pep + 0.3*np.cos(rad), y_pep + 0.3*np.sin(rad)), 
                       0.08, facecolor='#E74C3C', edgecolor='white', linewidth=1.5)
    ax.add_patch(pep_circle)

# 내부 miRNA 화물
ax.text(exo_main_x, exo_main_y+0.3, 'miRNA 칵테일', ha='center', va='center', 
        fontsize=13, fontweight='bold', color=COLOR_PRIMARY)
ax.text(exo_main_x-0.3, exo_main_y-0.1, '● miR-4739', ha='left', va='center', 
        fontsize=10, color='#E67E22', fontweight='bold')
ax.text(exo_main_x-0.3, exo_main_y-0.4, '● miR-4651', ha='left', va='center', 
        fontsize=10, color='#27AE60', fontweight='bold')
ax.text(exo_main_x, exo_main_y-0.75, '비율 1:1', ha='center', va='center', 
        fontsize=9, style='italic')

# 크기 표시
ax.text(exo_main_x, exo_main_y-1.5, '50-150 nm', ha='center', va='center', 
        fontsize=10, style='italic', color=COLOR_GRID)

# 좌측: 표면 변형
surface_x = 5.5
ax.text(surface_x, exo_main_y+1, '엑소좀 표면 변형', ha='center', va='center', 
        fontsize=12, fontweight='bold', color=COLOR_SECONDARY,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=COLOR_SECONDARY, linewidth=2))

surface_items = [
    '🎯 신장 타깃 펩타이드',
    '🎯 내피 타깃 리간드',
    '🏷️ CD63/CD81/CD9'
]
for i, item in enumerate(surface_items):
    y_pos = exo_main_y + 0.4 - i * 0.4
    ax.text(surface_x, y_pos, item, ha='center', va='center', fontsize=9)

# 화살표: 좌측 박스 → 엑소좀
arrow_left = FancyArrowPatch((surface_x+1.5, exo_main_y), (exo_main_x-1.3, exo_main_y),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color=COLOR_SECONDARY, linestyle='--')
ax.add_patch(arrow_left)

# 우측: 치료 효과
effect_x = 14.5
ax.text(effect_x, exo_main_y+1, '치료 효과', ha='center', va='center', 
        fontsize=12, fontweight='bold', color=COLOR_ACCENT2,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=COLOR_ACCENT2, linewidth=2))

effect_items = [
    ('🫘', '신장 섬유화 억제'),
    ('❤️', '심혈관 보호'),
    ('🔥', '염증 조절'),
    ('⚡', '항산화 증진')
]
for i, (icon, text) in enumerate(effect_items):
    y_pos = exo_main_y + 0.5 - i * 0.4
    ax.text(effect_x-0.5, y_pos, icon, ha='center', va='center', fontsize=14)
    ax.text(effect_x+0.5, y_pos, text, ha='left', va='center', fontsize=9)

# 화살표: 엑소좀 → 우측 박스
arrow_right = FancyArrowPatch((exo_main_x+1.3, exo_main_y), (effect_x-2, exo_main_y),
                             arrowstyle='->', mutation_scale=20, 
                             linewidth=2, color=COLOR_ACCENT2, linestyle='--')
ax.add_patch(arrow_right)

# ========================================
# 최하단: Primary Readouts
# ========================================

readout_y = 0.8
ax.text(10, readout_y+0.5, 'Primary Readouts (효능 평가 지표)', ha='center', va='center', 
        fontsize=14, fontweight='bold', color=COLOR_PRIMARY)

readouts = [
    ('TNFα/IL-6', '염증', '#E74C3C'),
    ('ROS/ΔΨm', '산화스트레스', '#F39C12'),
    ('COL1A1/αSMA', '섬유화', '#E67E22'),
    ('VCAM1/ICAM1', '내피기능', '#3498DB')
]

total_width = 12
box_width = total_width / len(readouts)
start_x = 10 - total_width/2

for i, (marker, category, color) in enumerate(readouts):
    x_pos = start_x + i * box_width + box_width/2
    
    box = FancyBboxPatch((x_pos - box_width/2 + 0.1, readout_y-0.25), 
                         box_width - 0.2, 0.5, 
                         boxstyle="round,pad=0.05", 
                         facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(box)
    
    ax.text(x_pos, readout_y+0.05, marker, ha='center', va='center', 
           fontsize=10, fontweight='bold', color='white')
    ax.text(x_pos, readout_y-0.15, f'({category})', ha='center', va='center', 
           fontsize=8, color='white', style='italic')

# 저장
plt.tight_layout()
plt.savefig('CKD_CVD/CKD_CVD_타깃_엑소좀_개발_모식도.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("모식도가 성공적으로 생성되었습니다!")
print("파일 위치: CKD_CVD/CKD_CVD_타깃_엑소좀_개발_모식도.png")

plt.show()
