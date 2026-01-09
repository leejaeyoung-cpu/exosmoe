"""
NOVA 작용 기전 인포그래픽 생성 (Photoshop-quality)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrow, Rectangle, Wedge
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path

# 한글 폰트
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Figure
fig = plt.figure(figsize=(20, 14), facecolor='white')
ax = fig.add_subplot(111)
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')

# Colors
COLOR_PRIMARY = '#3498db'
COLOR_SUCCESS = '#27ae60'
COLOR_DANGER = '#e74c3c'
COLOR_WARNING = '#f39c12'
COLOR_DARK = '#2c3e50'

# ===== HEADER =====
# Title box
title_box = FancyBboxPatch(
    (1, 12.5), 18, 1.3,
    boxstyle="round,pad=0.1",
    facecolor=COLOR_PRIMARY,
    edgecolor='white',
    linewidth=3
)
ax.add_patch(title_box)

ax.text(10, 13.5, 'NOVA 신약 작용 기전 (Mechanism of Action)',
        ha='center', va='center', fontsize=28, fontweight='bold', color='white')
ax.text(10, 12.9, 'Multi-Target Kinase Inhibitor | CKD-CVD 치료제',
        ha='center', va='center', fontsize=16, color='white')

# ===== MAIN DIAGRAM =====
# Left: Normal Pathology (Red/Orange)
pathology_x = 3
pathology_y_start = 10
pathology_spacing = 2

pathologies = [
    ('염증 신호 ↑', 'TNF-α, IL-6, p-p65', COLOR_DANGER),
    ('산화 스트레스 ↑', 'ROS, 지질 과산화', COLOR_WARNING),
    ('섬유화 진행 ↑', 'COL1A1, α-SMA', '#e67e22'),
    ('혈관 손상 ↑', 'VCAM1, ICAM1', '#c0392b')
]

ax.text(pathology_x, 11, '정상 병리 (CKD-CVD)',
        ha='center', fontsize=14, fontweight='bold', color=COLOR_DARK)

for i, (name, markers, color) in enumerate(pathologies):
    y = pathology_y_start - i * pathology_spacing
    
    # Box
    box = FancyBboxPatch(
        (pathology_x - 1.5, y - 0.4), 3, 0.8,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor=COLOR_DARK,
        linewidth=2,
        alpha=0.8
    )
    ax.add_patch(box)
    
    ax.text(pathology_x, y+0.15, name, ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    ax.text(pathology_x, y-0.15, markers, ha='center', va='center',
            fontsize=9, color='white')

# Center: NOVA Molecule (Blue Hexagon)
center_x = 10
center_y = 6.5

# Hexagon
hexagon = mpatches.RegularPolygon(
    (center_x, center_y), 6, radius=1.5,
    facecolor=COLOR_PRIMARY,
    edgecolor='white',
    linewidth=4
)
ax.add_patch(hexagon)

ax.text(center_x, center_y + 0.3, 'NOVA', ha='center', va='center',
        fontsize=24, fontweight='bold', color='white')
ax.text(center_x, center_y - 0.3, 'Drug', ha='center', va='center',
        fontsize=18, color='white')

# Arrows from NOVA to pathways with BLOCK symbols
for i, (name, markers, color) in enumerate(pathologies):
    target_y = pathology_y_start - i * pathology_spacing
    
    # Arrow
    arrow = FancyArrow(
        center_x - 1.2, center_y + 0.5 if i < 2 else center_y - 0.5,
        pathology_x + 1.8 - (center_x - 1.2), target_y - (center_y + 0.5 if i < 2 else center_y - 0.5),
        width=0.15, head_width=0.3, head_length=0.3,
        facecolor=COLOR_DANGER, edgecolor=COLOR_DARK, linewidth=2
    )
    ax.add_patch(arrow)
    
    # Block symbol
    mid_x = (center_x - 1.2 + pathology_x + 1.8) / 2
    mid_y = (center_y + 0.5 if i < 2 else center_y - 0.5) + (target_y - (center_y + 0.5 if i < 2 else center_y - 0.5)) / 2
    
    block_circle = Circle((mid_x, mid_y), 0.4, facecolor='white', edgecolor=COLOR_DANGER, linewidth=3)
    ax.add_patch(block_circle)
    ax.text(mid_x, mid_y, '🚫', ha='center', va='center', fontsize=20)

# Right: Treatment Result (Green)
result_x = 17
result_y_start = 10

results = [
    ('염증 ↓', '65% 감소', COLOR_SUCCESS),
    ('ROS ↓', '67% 감소', COLOR_SUCCESS),
    ('섬유화 ↓', '55% 감소', COLOR_SUCCESS),
    ('내피 보호 ↑', '64% 개선', COLOR_SUCCESS)
]

ax.text(result_x, 11, 'NOVA 치료 후',
        ha='center', fontsize=14, fontweight='bold', color=COLOR_DARK)

for i, (name, effect, color) in enumerate(results):
    y = result_y_start - i * pathology_spacing
    
    # Box
    box = FancyBboxPatch(
        (result_x - 1.5, y - 0.4), 3, 0.8,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor=COLOR_DARK,
        linewidth=2,
        alpha=0.8
    )
    ax.add_patch(box)
    
    ax.text(result_x, y+0.15, name, ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    ax.text(result_x, y-0.15, effect, ha='center', va='center',
            fontsize=10, color='white')
    
    # Checkmark
    ax.text(result_x + 1.8, y, '✅', ha='center', va='center', fontsize=20)

# ===== MOLECULAR MECHANISMS (Bottom boxes) =====
box_y = 3
box_spacing = 4.5
box_width = 4
box_height = 2.5

mechanisms = [
    {
        'title': 'NF-κB p65 억제',
        'mechanism': 'DNA 결합 차단',
        'effect': 'TNF-α/IL-6 ↓65%',
        'color': '#9b59b6'
    },
    {
        'title': 'TGF-β R1 차단',
        'mechanism': 'ATP 포켓 점유',
        'effect': 'SMAD2/3 ↓55%',
        'color': '#3498db'
    },
    {
        'title': 'NOX4 억제',
        'mechanism': 'NADPH 차단',
        'effect': 'ROS ↓67%',
        'color': '#16a085'
    },
    {
        'title': 'VCAM1/ICAM1 ↓',
        'mechanism': '세포 부착 감소',
        'effect': '내피 보호 ↑64%',
        'color': '#27ae60'
    }
]

for i, mech in enumerate(mechanisms):
    box_x = 1.5 + i * box_spacing
    
    # Box
    box = FancyBboxPatch(
        (box_x, box_y), box_width, box_height,
        boxstyle="round,pad=0.1",
        facecolor=mech['color'],
        edgecolor=COLOR_DARK,
        linewidth=2,
        alpha=0.9
    )
    ax.add_patch(box)
    
    # Content
    ax.text(box_x + box_width/2, box_y + box_height - 0.4,
            mech['title'],
            ha='center', va='top', fontsize=13, fontweight='bold', color='white')
    
    ax.text(box_x + box_width/2, box_y + box_height/2 + 0.2,
            mech['mechanism'],
            ha='center', va='center', fontsize=11, color='white')
    
    ax.text(box_x + box_width/2, box_y + 0.4,
            mech['effect'],
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            color='yellow')

# ===== BOTTOM STATS BAR =====
stats_box = Rectangle((0.5, 0.2), 19, 1.3, facecolor='#ecf0f1', edgecolor=COLOR_DARK, linewidth=2)
ax.add_patch(stats_box)

stats = [
    ('치료 지수 (TI)', '26.7', '매우 안전'),
    ('시너지 지수 (CI)', '0.59', '강력한 시너지'),
    ('경구 흡수율', '60-80%', '우수'),
    ('타겟', 'CKD+CVD', '다중 경로')
]

for i, (label, value, desc) in enumerate(stats):
    x = 2 + i * 4.5
    
    ax.text(x, 1.2, label, ha='left', va='top',
            fontsize=11, fontweight='bold', color=COLOR_DARK)
    ax.text(x, 0.8, value, ha='left', va='center',
            fontsize=16, fontweight='bold', color=COLOR_PRIMARY)
    ax.text(x, 0.4, desc, ha='left', va='bottom',
            fontsize=9, color='#7f8c8d')

# Save
output_path = Path(r"C:\Users\brook\Downloads\NOVA_Mechanism_Infographic.png")
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ 인포그래픽 저장 완료: {output_path}")
plt.close()

print("\n" + "="*80)
print("NOVA 작용 기전 인포그래픽 생성 완료!")
print("="*80)
print(f"파일 위치: {output_path}")
print(f"파일 크기: {output_path.stat().st_size / 1024:.1f} KB")
print("\n✨ 전문적인 포토샵 스타일 인포그래픽이 생성되었습니다!")
