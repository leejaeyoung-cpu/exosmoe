"""
Advanced Image Analyzer
세포 이미지 분석 및 기능 추론 엔진

Features:
- 세포 형태학적 특성 분석 (밀도, 크기, 형태, 복잡도)
- 기능별 특징 추출 (항산화, 항섬유화, 항염증, 혈관형성, 세포증식)
- 다중 이미지 배치 처리
- AI 기반 기능 점수 예측
- 상세 분석 리포트 생성
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class AdvancedImageAnalyzer:
    """고급 세포 이미지 분석 클래스"""
    
    FUNCTION_FEATURES = {
        'antioxidant': {
            'description': '항산화 기능 지표',
            'markers': ['세포 생존율', '형태 건강도', '막 완전성']
        },
        'anti_fibrotic': {
            'description': '항섬유화 기능 지표',
            'markers': ['세포 밀도', '콜라겐 침착', '섬유화 패턴']
        },
        'anti_inflammatory': {
            'description': '항염증 기능 지표',
            'markers': ['세포 형태 변화', '에지 복잡도', '염증 마커']
        },
        'angiogenic': {
            'description': '혈관형성 기능 지표',
            'markers': ['관 형성 구조', '네트워크 패턴', '분지 밀도']
        },
        'proliferation': {
            'description': '세포증식 기능 지표',
            'markers': ['세포 수', '밀집도', '증식 속도']
        }
    }
    
    def __init__(self):
        self.results = []
        self.analysis_df = None
        
    def load_image(self, image_path: str) -> np.ndarray:
        """이미지 로드"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"이미지 로드 실패: {image_path}")
        return img
    
    def analyze_basic_properties(self, img: np.ndarray, image_name: str = "Unknown") -> Dict:
        """기본 세포 특성 분석"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. 평균 강도 (세포 밀도 지표)
        mean_intensity = np.mean(gray)
        
        # 2. 대비 (세포 형태 선명도)
        contrast = np.std(gray)
        
        # 3. 에지 밀도 (세포 윤곽 복잡도)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 4. 엔트로피 (밝기 분포의 복잡도)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist_normalized = hist / hist.sum()
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
        
        # 5. 세포 영역 추정 (Otsu threshold)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cell_area_ratio = np.sum(binary > 0) / binary.size
        
        # 6. 텍스처 특징 (GLCM 기반 - 간소화 버전)
        texture_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 7. 형태학적 특징
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_cells = len(contours)
        
        # 평균 세포 크기
        if num_cells > 0:
            avg_cell_size = np.mean([cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 50])
            cell_size_std = np.std([cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 50])
        else:
            avg_cell_size = 0
            cell_size_std = 0
        
        return {
            'image_name': image_name,
            'mean_intensity': mean_intensity,
            'contrast': contrast,
            'edge_density': edge_density,
            'entropy': entropy,
            'cell_area_ratio': cell_area_ratio,
            'texture_variance': texture_variance,
            'num_cells': num_cells,
            'avg_cell_size': avg_cell_size,
            'cell_size_std': cell_size_std
        }
    
    def infer_function_scores(self, properties: Dict) -> Dict:
        """
        이미지 특성으로부터 기능 점수 추론
        
        규칙 기반 추론 알고리즘:
        - 각 기능별로 관련 특성을 조합하여 점수 계산
        - 점수는 0-1 사이로 정규화
        """
        scores = {}
        
        # 1. 항산화 점수 (세포 건강도 기반)
        # 높은 mean_intensity, 낮은 edge_density, 높은 cell_area_ratio
        antioxidant_score = (
            (properties['mean_intensity'] / 255) * 0.4 +
            (1 - properties['edge_density'] * 10) * 0.3 +
            properties['cell_area_ratio'] * 0.3
        )
        scores['antioxidant_score'] = np.clip(antioxidant_score, 0, 1)
        
        # 2. 항섬유화 점수 (세포 밀도와 균일성)
        # 높은 cell_area_ratio, 낮은 texture_variance
        anti_fibrotic_score = (
            properties['cell_area_ratio'] * 0.5 +
            (1 - min(properties['texture_variance'] / 1000, 1)) * 0.5
        )
        scores['anti_fibrotic_score'] = np.clip(anti_fibrotic_score, 0, 1)
        
        # 3. 항염증 점수 (세포 형태 안정성)
        # 낮은 edge_density, 높은 entropy (균일한 분포)
        anti_inflammatory_score = (
            (1 - properties['edge_density'] * 10) * 0.5 +
            (properties['entropy'] / 8) * 0.5
        )
        scores['anti_inflammatory_score'] = np.clip(anti_inflammatory_score, 0, 1)
        
        # 4. 혈관형성 점수 (네트워크 구조)
        # 높은 edge_density, 높은 num_cells
        angiogenic_score = (
            properties['edge_density'] * 5 * 0.6 +
            min(properties['num_cells'] / 100, 1) * 0.4
        )
        scores['angiogenic_score'] = np.clip(angiogenic_score, 0, 1)
        
        # 5. 세포증식 점수 (세포 수와 밀집도)
        # 높은 num_cells, 높은 cell_area_ratio
        proliferation_score = (
            min(properties['num_cells'] / 100, 1) * 0.6 +
            properties['cell_area_ratio'] * 0.4
        )
        scores['proliferation_score'] = np.clip(proliferation_score, 0, 1)
        
        # 주요 기능 결정
        if max(scores.values()) > 0:
            primary_func_key = max(scores, key=scores.get)
            primary_func_names = {
                'antioxidant_score': '항산화',
                'anti_fibrotic_score': '항섬유화',
                'anti_inflammatory_score': '항염증',
                'angiogenic_score': '혈관형성',
                'proliferation_score': '세포증식'
            }
            scores['primary_function'] = primary_func_names[primary_func_key]
            scores['primary_score'] = scores[primary_func_key]
            scores['max_function_score'] = scores[primary_func_key]  # UI 호환성
        else:
            scores['primary_function'] = '미분류'
            scores['primary_score'] = 0
            scores['max_function_score'] = 0  # UI 호환성
        
        return scores
    
    def analyze_image(self, image_path: str) -> Dict:
        """단일 이미지 전체 분석"""
        img = self.load_image(image_path)
        image_name = Path(image_path).name
        
        # 기본 특성 분석
        properties = self.analyze_basic_properties(img, image_name)
        
        # 기능 점수 추론
        scores = self.infer_function_scores(properties)
        
        # 결과 통합
        result = {**properties, **scores}
        result['image_path'] = str(image_path)
        
        return result
    
    def analyze_batch(self, image_paths: List[str], group_name: str = "Default") -> pd.DataFrame:
        """여러 이미지 배치 분석"""
        print(f"\n🔬 {len(image_paths)}개 이미지 분석 중...")
        
        results = []
        for i, img_path in enumerate(image_paths, 1):
            try:
                result = self.analyze_image(img_path)
                result['group'] = group_name
                results.append(result)
                print(f"  ✓ [{i}/{len(image_paths)}] {Path(img_path).name}")
            except Exception as e:
                print(f"  ✗ [{i}/{len(image_paths)}] {Path(img_path).name} - 오류: {e}")
        
        self.results.extend(results)
        self.analysis_df = pd.DataFrame(self.results)
        
        print(f"✅ 분석 완료: {len(results)}개 성공\n")
        return self.analysis_df
    
    def compare_groups(self, df: pd.DataFrame) -> Dict:
        """그룹 간 비교 분석"""
        if 'group' not in df.columns:
            return {}
        
        comparison = {}
        groups = df['group'].unique()
        
        for func in ['antioxidant', 'anti_fibrotic', 'anti_inflammatory', 'angiogenic', 'proliferation']:
            score_col = f'{func}_score'
            comparison[func] = {}
            
            for group in groups:
                group_data = df[df['group'] == group][score_col]
                comparison[func][group] = {
                    'mean': group_data.mean(),
                    'std': group_data.std(),
                    'count': len(group_data)
                }
        
        return comparison
    
    def visualize_results(self, df: pd.DataFrame, output_dir: str):
        """분석 결과 시각화"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print("📊 결과 시각화 중...")
        
        # 1. 기능별 점수 분포
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('세포 이미지 기능 분석 결과', fontsize=16, fontweight='bold')
        
        score_cols = [
            ('antioxidant_score', '항산화'),
            ('anti_fibrotic_score', '항섬유화'),
            ('anti_inflammatory_score', '항염증'),
            ('angiogenic_score', '혈관형성'),
            ('proliferation_score', '세포증식')
        ]
        
        for idx, (col, title) in enumerate(score_cols):
            row, col_idx = idx // 3, idx % 3
            ax = axes[row, col_idx]
            
            if 'group' in df.columns:
                df.boxplot(column=col, by='group', ax=ax)
                ax.set_title(title, fontweight='bold')
                ax.set_xlabel('그룹')
            else:
                df[col].hist(bins=20, ax=ax, color='skyblue', edgecolor='black')
                ax.set_title(title, fontweight='bold')
                ax.set_xlabel('점수')
            
            ax.set_ylabel('기능 점수')
            ax.grid(True, alpha=0.3)
        
        axes[1, 2].axis('off')
        plt.tight_layout()
        
        viz_path = output_path / 'function_scores_distribution.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 저장: {viz_path}")
        plt.close()
        
        # 2. 상관관계 히트맵
        fig, ax = plt.subplots(figsize=(10, 8))
        
        score_data = df[[col for col, _ in score_cols]]
        correlation = score_data.corr()
        
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=ax, square=True, linewidths=1,
                   xticklabels=[t for _, t in score_cols],
                   yticklabels=[t for _, t in score_cols])
        ax.set_title('기능 점수 간 상관관계', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        corr_path = output_path / 'function_correlation.png'
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 저장: {corr_path}")
        plt.close()
        
        print("✅ 시각화 완료\n")
    
    def generate_report(self, df: pd.DataFrame, output_dir: str):
        """분석 리포트 생성"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        report_path = output_path / 'Image_Analysis_Report.txt'
        
        print("📝 리포트 생성 중...")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("세포 이미지 기능 분석 리포트\n")
            f.write("="*80 + "\n\n")
            
            f.write("📊 분석 개요\n")
            f.write("-"*80 + "\n")
            f.write(f"총 이미지 수: {len(df)}\n")
            
            if 'group' in df.columns:
                f.write(f"그룹 수: {df['group'].nunique()}\n")
                f.write("\n그룹별 이미지 수:\n")
                for group, count in df['group'].value_counts().items():
                    f.write(f"  - {group}: {count}개\n")
            
            f.write("\n🔬 기능별 평균 점수\n")
            f.write("-"*80 + "\n")
            
            functions = {
                'antioxidant_score': '항산화',
                'anti_fibrotic_score': '항섬유화',
                'anti_inflammatory_score': '항염증',
                'angiogenic_score': '혈관형성',
                'proliferation_score': '세포증식'
            }
            
            for col, name in functions.items():
                mean_score = df[col].mean()
                std_score = df[col].std()
                f.write(f"\n{name}:\n")
                f.write(f"  평균: {mean_score:.3f} ± {std_score:.3f}\n")
                
                if 'group' in df.columns:
                    for group in df['group'].unique():
                        group_mean = df[df['group'] == group][col].mean()
                        f.write(f"    {group}: {group_mean:.3f}\n")
            
            f.write("\n📈 주요 발견사항\n")
            f.write("-"*80 + "\n")
            
            # 가장 우수한 기능
            func_means = {name: df[col].mean() for col, name in functions.items()}
            best_func = max(func_means, key=func_means.get)
            f.write(f"\n가장 높은 기능: {best_func} (평균 점수: {func_means[best_func]:.3f})\n")
            
            # 주요 기능 분포
            f.write("\n주요 기능 분포:\n")
            for func, count in df['primary_function'].value_counts().items():
                percentage = (count / len(df)) * 100
                f.write(f"  - {func}: {count}개 ({percentage:.1f}%)\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("💡 결론\n")
            f.write("-"*80 + "\n")
            f.write("세포 이미지 기반 기능 분석이 완료되었습니다.\n")
            f.write("AI 추론 모델을 통해 각 이미지의 기능적 특성을 평가했습니다.\n")
            f.write("\n분석 완료 시간: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("="*80 + "\n")
        
        print(f"  ✓ 리포트 저장: {report_path}")
        
        # CSV로도 저장
        csv_path = output_path / 'Image_Analysis_Data.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  ✓ 데이터 저장: {csv_path}")
        
        print("✅ 리포트 생성 완료\n")


def main():
    """예제 실행"""
    print("\n" + "="*80)
    print("🔬 고급 이미지 분석 시스템 테스트")
    print("="*80 + "\n")
    
    # 테스트용 경로
    test_dir = r"c:\Users\brook\Desktop\mi_exo_ai\data\HUVEC TNF-a\HUVEC TNF-a\251209"
    output_dir = r"c:\Users\brook\Desktop\mi_exo_ai\data\Advanced_Image_Analysis"
    
    # 분석기 초기화
    analyzer = AdvancedImageAnalyzer()
    
    # 이미지 파일 찾기
    test_path = Path(test_dir)
    if test_path.exists():
        image_files = list(test_path.glob("*.jpg"))
        
        if image_files:
            # 그룹별로 분류하여 분석
            groups = {}
            for img_file in image_files:
                name = img_file.name
                if '_con_' in name:
                    group = 'Control'
                elif '_1ng_' in name:
                    group = '1ng TNF-α'
                elif '_5ng_' in name:
                    group = '5ng TNF-α'
                elif '_10ng_' in name:
                    group = '10ng TNF-α'
                else:
                    group = 'Other'
                
                if group not in groups:
                    groups[group] = []
                groups[group].append(str(img_file))
            
            # 각 그룹 분석
            for group, files in groups.items():
                analyzer.analyze_batch(files, group)
            
            # 결과 확인
            df = analyzer.analysis_df
            print(f"📋 분석 결과 요약:")
            print(df.groupby('group')[['antioxidant_score', 'anti_fibrotic_score', 
                                       'anti_inflammatory_score', 'angiogenic_score', 
                                       'proliferation_score']].mean())
            
            # 시각화
            analyzer.visualize_results(df, output_dir)
            
            # 리포트 생성
            analyzer.generate_report(df, output_dir)
            
            print("\n" + "="*80)
            print("✅ 분석 완료!")
            print(f"📁 결과 저장 위치: {output_dir}")
            print("="*80 + "\n")
        else:
            print("⚠️  이미지 파일을 찾을 수 없습니다.")
    else:
        print(f"⚠️  테스트 디렉토리가 없습니다: {test_dir}")


if __name__ == "__main__":
    main()
