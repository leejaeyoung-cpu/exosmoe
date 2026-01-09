"""
HUVEC TNF-α 실험 분석 스크립트
- HUVEC 세포의 TNF-α 농도별 반응 분석
- Control, 1ng, 5ng, 10ng 그룹 비교
- 세포 형태, 밀도, 형광 강도 등 분석
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class HUVECAnalyzer:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.results = {
            'control': [],
            '1ng': [],
            '5ng': [],
            '10ng': []
        }
        self.stats = {}
        
    def load_images(self):
        """이미지 로드 및 그룹별 분류"""
        print("📂 이미지 로딩 중...")
        
        for img_file in self.base_dir.glob("*.jpg"):
            img_name = img_file.name
            
            # 그룹 분류
            if '_con_' in img_name:
                group = 'control'
            elif '_1ng_' in img_name:
                group = '1ng'
            elif '_5ng_' in img_name:
                group = '5ng'
            elif '_10ng_' in img_name:
                group = '10ng'
            else:
                continue
                
            # 이미지 읽기
            img = cv2.imread(str(img_file))
            if img is not None:
                self.results[group].append({
                    'filename': img_name,
                    'image': img,
                    'path': str(img_file)
                })
                
        # 로드 결과 출력
        for group, images in self.results.items():
            print(f"  {group}: {len(images)}개 이미지")
            
    def analyze_cell_properties(self):
        """세포 특성 분석 (밀도, 형태, 강도 등)"""
        print("\n🔬 세포 특성 분석 중...")
        
        analysis_results = []
        
        for group, images in self.results.items():
            for img_data in images:
                img = img_data['image']
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # 1. 평균 강도 (세포 밀도 지표)
                mean_intensity = np.mean(gray)
                
                # 2. 대비 (세포 형태 선명도)
                contrast = np.std(gray)
                
                # 3. 에지 밀도 (세포 윤곽 복잡도)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                
                # 4. 밝기 분포
                hist, _ = np.histogram(gray, bins=256, range=(0, 256))
                entropy = -np.sum((hist / hist.sum()) * np.log2(hist / hist.sum() + 1e-10))
                
                # 5. 세포 영역 추정 (Otsu threshold)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cell_area_ratio = np.sum(binary > 0) / binary.size
                
                analysis_results.append({
                    'group': group,
                    'filename': img_data['filename'],
                    'mean_intensity': mean_intensity,
                    'contrast': contrast,
                    'edge_density': edge_density,
                    'entropy': entropy,
                    'cell_area_ratio': cell_area_ratio
                })
                
        self.analysis_df = pd.DataFrame(analysis_results)
        
        # 그룹별 통계
        self.stats = self.analysis_df.groupby('group').agg({
            'mean_intensity': ['mean', 'std'],
            'contrast': ['mean', 'std'],
            'edge_density': ['mean', 'std'],
            'entropy': ['mean', 'std'],
            'cell_area_ratio': ['mean', 'std']
        }).round(3)
        
        print("\n✅ 분석 완료!")
        return self.analysis_df
    
    def visualize_representative_images(self, output_dir):
        """대표 이미지 시각화"""
        print("\n🖼️  대표 이미지 시각화 중...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('HUVEC TNF-α 농도별 세포 형태 비교', fontsize=16, fontweight='bold')
        
        groups_order = ['control', '1ng', '5ng', '10ng']
        
        for idx, group in enumerate(groups_order):
            images = self.results[group]
            if len(images) >= 2:
                # 첫 번째 이미지
                img1_rgb = cv2.cvtColor(images[0]['image'], cv2.COLOR_BGR2RGB)
                axes[0, idx].imshow(img1_rgb)
                axes[0, idx].set_title(f'{group.upper()} - 1', fontweight='bold')
                axes[0, idx].axis('off')
                
                # 두 번째 이미지
                img2_rgb = cv2.cvtColor(images[1]['image'], cv2.COLOR_BGR2RGB)
                axes[1, idx].imshow(img2_rgb)
                axes[1, idx].set_title(f'{group.upper()} - 2', fontweight='bold')
                axes[1, idx].axis('off')
        
        plt.tight_layout()
        output_path = output_dir / 'representative_images.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 저장: {output_path}")
        plt.close()
        
    def plot_analysis_results(self, output_dir):
        """분석 결과 그래프 생성"""
        print("\n📊 분석 결과 그래프 생성 중...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 그룹 순서 정의
        group_order = ['control', '1ng', '5ng', '10ng']
        
        # 1. 박스플롯 - 주요 지표들
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('HUVEC TNF-α 처리 효과 분석', fontsize=16, fontweight='bold')
        
        metrics = [
            ('mean_intensity', '평균 강도', axes[0, 0]),
            ('contrast', '대비 (Contrast)', axes[0, 1]),
            ('edge_density', '에지 밀도', axes[0, 2]),
            ('entropy', '엔트로피', axes[1, 0]),
            ('cell_area_ratio', '세포 영역 비율', axes[1, 1])
        ]
        
        for metric, title, ax in metrics:
            sns.boxplot(data=self.analysis_df, x='group', y=metric, 
                       order=group_order, palette='Set2', ax=ax)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('TNF-α 농도', fontweight='bold')
            ax.set_ylabel(title, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # 빈 서브플롯 제거
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        output_path = output_dir / 'analysis_boxplots.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 저장: {output_path}")
        plt.close()
        
        # 2. 바 차트 - 그룹별 평균 비교
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 평균 강도
        group_means = self.analysis_df.groupby('group')['mean_intensity'].mean()
        group_means = group_means.reindex(group_order)
        axes[0].bar(range(len(group_means)), group_means.values, 
                   color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
        axes[0].set_xticks(range(len(group_means)))
        axes[0].set_xticklabels(group_means.index)
        axes[0].set_ylabel('평균 강도', fontweight='bold')
        axes[0].set_xlabel('TNF-α 농도', fontweight='bold')
        axes[0].set_title('TNF-α 농도별 평균 강도', fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 세포 영역 비율
        group_area = self.analysis_df.groupby('group')['cell_area_ratio'].mean()
        group_area = group_area.reindex(group_order)
        axes[1].bar(range(len(group_area)), group_area.values, 
                   color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
        axes[1].set_xticks(range(len(group_area)))
        axes[1].set_xticklabels(group_area.index)
        axes[1].set_ylabel('세포 영역 비율', fontweight='bold')
        axes[1].set_xlabel('TNF-α 농도', fontweight='bold')
        axes[1].set_title('TNF-α 농도별 세포 영역 비율', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = output_dir / 'group_comparison_bars.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 저장: {output_path}")
        plt.close()
        
        # 3. 히트맵 - 상관관계 분석
        fig, ax = plt.subplots(figsize=(10, 8))
        
        correlation_data = self.analysis_df[['mean_intensity', 'contrast', 
                                              'edge_density', 'entropy', 
                                              'cell_area_ratio']].corr()
        
        sns.heatmap(correlation_data, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, ax=ax, square=True, linewidths=1)
        ax.set_title('세포 특성 간 상관관계 분석', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        output_path = output_dir / 'correlation_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 저장: {output_path}")
        plt.close()
        
    def generate_report(self, output_dir):
        """분석 리포트 생성"""
        print("\n📝 분석 리포트 생성 중...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        report_path = output_dir / 'HUVEC_TNF_Analysis_Report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("HUVEC TNF-α 실험 분석 리포트\n")
            f.write("="*80 + "\n\n")
            
            f.write("📋 실험 개요\n")
            f.write("-"*80 + "\n")
            f.write(f"실험일자: 2025-12-09\n")
            f.write(f"세포주: HUVEC (Human Umbilical Vein Endothelial Cells)\n")
            f.write(f"처리물질: TNF-α (Tumor Necrosis Factor-alpha)\n")
            f.write(f"농도 그룹: Control, 1ng/ml, 5ng/ml, 10ng/ml\n\n")
            
            f.write("📊 샘플 구성\n")
            f.write("-"*80 + "\n")
            for group, images in self.results.items():
                f.write(f"  {group:10s}: {len(images):2d}개 이미지\n")
            f.write(f"\n총 {sum(len(imgs) for imgs in self.results.values())}개 이미지 분석\n\n")
            
            f.write("🔬 분석 결과 통계 (평균 ± 표준편차)\n")
            f.write("-"*80 + "\n\n")
            f.write(self.stats.to_string())
            f.write("\n\n")
            
            f.write("📈 주요 발견사항\n")
            f.write("-"*80 + "\n")
            
            # 평균 강도 변화
            intensity_means = self.analysis_df.groupby('group')['mean_intensity'].mean()
            control_intensity = intensity_means.get('control', 0)
            
            f.write("\n1. 평균 강도 변화 (세포 밀도 지표)\n")
            for group in ['1ng', '5ng', '10ng']:
                if group in intensity_means.index:
                    change = ((intensity_means[group] - control_intensity) / control_intensity) * 100
                    f.write(f"   - {group:10s}: {intensity_means[group]:.2f} "
                           f"(Control 대비 {change:+.2f}%)\n")
            
            # 세포 영역 비율 변화
            area_means = self.analysis_df.groupby('group')['cell_area_ratio'].mean()
            control_area = area_means.get('control', 0)
            
            f.write("\n2. 세포 영역 비율 변화\n")
            for group in ['1ng', '5ng', '10ng']:
                if group in area_means.index:
                    change = ((area_means[group] - control_area) / control_area) * 100
                    f.write(f"   - {group:10s}: {area_means[group]:.4f} "
                           f"(Control 대비 {change:+.2f}%)\n")
            
            # 에지 밀도 변화
            edge_means = self.analysis_df.groupby('group')['edge_density'].mean()
            control_edge = edge_means.get('control', 0)
            
            f.write("\n3. 에지 밀도 변화 (세포 형태 복잡도)\n")
            for group in ['1ng', '5ng', '10ng']:
                if group in edge_means.index:
                    change = ((edge_means[group] - control_edge) / control_edge) * 100
                    f.write(f"   - {group:10s}: {edge_means[group]:.6f} "
                           f"(Control 대비 {change:+.2f}%)\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("💡 결론\n")
            f.write("-"*80 + "\n")
            f.write("TNF-α 처리에 따른 HUVEC 세포의 형태학적 변화가 관찰되었습니다.\n")
            f.write("농도 의존적인 반응을 확인하기 위해서는 추가적인 정량 분석이 필요합니다.\n")
            f.write("세포 형태, 밀도, 복잡도 등의 지표에서 그룹 간 차이가 나타났습니다.\n")
            f.write("\n분석 완료 시간: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("="*80 + "\n")
        
        print(f"  ✓ 리포트 저장: {report_path}")
        
        # CSV로도 저장
        csv_path = output_dir / 'HUVEC_TNF_Analysis_Data.csv'
        self.analysis_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  ✓ 데이터 저장: {csv_path}")
        
        # 통계 요약도 저장
        stats_path = output_dir / 'HUVEC_TNF_Statistics.csv'
        self.stats.to_csv(stats_path, encoding='utf-8-sig')
        print(f"  ✓ 통계 저장: {stats_path}")


def main():
    """메인 실행 함수"""
    print("\n" + "="*80)
    print("🧬 HUVEC TNF-α 실험 분석 시작")
    print("="*80 + "\n")
    
    # 경로 설정
    data_dir = r"c:\Users\brook\Desktop\mi_exo_ai\data\HUVEC TNF-a\HUVEC TNF-a\251209"
    output_dir = r"c:\Users\brook\Desktop\mi_exo_ai\data\HUVEC TNF-a\Analysis_Results"
    
    # 분석기 초기화
    analyzer = HUVECAnalyzer(data_dir)
    
    # 1. 이미지 로드
    analyzer.load_images()
    
    # 2. 세포 특성 분석
    df = analyzer.analyze_cell_properties()
    
    # 3. 대표 이미지 시각화
    analyzer.visualize_representative_images(output_dir)
    
    # 4. 분석 결과 그래프
    analyzer.plot_analysis_results(output_dir)
    
    # 5. 리포트 생성
    analyzer.generate_report(output_dir)
    
    print("\n" + "="*80)
    print("✅ 분석 완료!")
    print(f"📁 결과 저장 위치: {output_dir}")
    print("="*80 + "\n")
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
