"""
Advanced Cellpose Integration for MT-EXO Analysis
세포 세그멘테이션 및 고급 특징 추출
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from cellpose import models, core
import warnings
warnings.filterwarnings('ignore')


class AdvancedCellposeProcessor:
    """Cellpose 기반 고급 세포 분석"""
    
    def __init__(self, model_type='cyto2', use_gpu=True):
        """
        Args:
            model_type: 'cyto', 'cyto2', 'nuclei' 등
            use_gpu: GPU 사용 여부
        """
        self.use_gpu = use_gpu and core.use_gpu()
        print(f"🔬 Cellpose 초기화: GPU={self.use_gpu}")
        
        # Cellpose 모델 로드 (올바른 API 사용)
        self.model = models.CellposeModel(
            gpu=self.use_gpu,
            model_type=model_type
        )
        
    def segment_cells(self, image: np.ndarray, diameter: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        세포 세그멘테이션
        
        Args:
            image: RGB or Grayscale image
            diameter: 예상 세포 직경 (None이면 자동)
            
        Returns:
            masks: 세포 마스크 (0=배경, 1,2,3...=각 세포)
            flows: Flow 맵
        """
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Cellpose 실행
        result = self.model.eval(
            gray,
            diameter=diameter,
            channels=[0, 0],  # Grayscale
            flow_threshold=0.4,
            cellprob_threshold=0.0
        )
        
        # Cellpose 버전에 따라 반환값이 다름
        if len(result) == 4:
            masks, flows, styles, diams = result
        else:
            masks, flows, styles = result
        
        return masks, flows[0]
    
    def extract_cell_features(self, image: np.ndarray, masks: np.ndarray) -> List[Dict]:
        """
        각 세포별 상세 특징 추출
        
        Returns:
            List of feature dicts for each cell
        """
        features = []
        num_cells = masks.max()
        
        for cell_id in range(1, num_cells + 1):
            # 마스크 추출
            cell_mask = (masks == cell_id).astype(np.uint8)
            
            # 기본 특징
            area = np.sum(cell_mask)
            
            if area < 50:  # 너무 작은 영역 무시
                continue
            
            # 컨투어 찾기 (OpenCV 버전 호환)
            contours_result = cv2.findContours(
                cell_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            # OpenCV 3.x returns 3 values, OpenCV 4.x returns 2 values
            if len(contours_result) == 3:
                _, contours, _ = contours_result
            else:
                contours, _ = contours_result
            
            if len(contours) == 0:
                continue
            
            contour = contours[0]
            
            # 1. 형태학적 특징
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # 2. 경계 상자
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # 3. 타원 피팅
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (cx, cy), (ma, MA), angle = ellipse
                eccentricity = np.sqrt(1 - (min(ma, MA) / max(ma, MA)) ** 2) if max(ma, MA) > 0 else 0
            else:
                cx, cy, ma, MA, angle, eccentricity = 0, 0, 0, 0, 0, 0
            
            # 4. 볼록 껍질
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # 5. 강도 특징 (원본 이미지 필요)
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            cell_pixels = gray[cell_mask > 0]
            mean_intensity = np.mean(cell_pixels)
            std_intensity = np.std(cell_pixels)
            
            # 6. 텍스처 특징
            # Haralick 특징 간소화 버전
            cell_region = gray[y:y+h, x:x+w] * cell_mask[y:y+h, x:x+w]
            texture_variance = np.var(cell_region[cell_region > 0])
            
            features.append({
                'cell_id': cell_id,
                'area': float(area),
                'perimeter': float(perimeter),
                'circularity': float(circularity),
                'aspect_ratio': float(aspect_ratio),
                'eccentricity': float(eccentricity),
                'solidity': float(solidity),
                'centroid_x': float(cx),
                'centroid_y': float(cy),
                'major_axis': float(MA),
                'minor_axis': float(ma),
                'orientation': float(angle),
                'mean_intensity': float(mean_intensity),
                'std_intensity': float(std_intensity),
                'texture_variance': float(texture_variance),
                'bbox_x': int(x),
                'bbox_y': int(y),
                'bbox_w': int(w),
                'bbox_h': int(h)
            })
        
        return features
    
    def create_feature_vector(self, features: List[Dict]) -> np.ndarray:
        """
        전체 이미지의 특징 벡터 생성 (AI 모델 입력용)
        
        Returns:
            Feature vector (1D numpy array)
        """
        if len(features) == 0:
            return np.zeros(20)
        
        # 통계적 집계
        areas = [f['area'] for f in features]
        circularities = [f['circularity'] for f in features]
        eccentricities = [f['eccentricity'] for f in features]
        intensities = [f['mean_intensity'] for f in features]
        textures = [f['texture_variance'] for f in features]
        
        feature_vec = np.array([
            len(features),                    # 세포 수
            np.mean(areas),                   # 평균 면적
            np.std(areas),                    # 면적 표준편차
            np.mean(circularities),           # 평균 원형도
            np.std(circularities),            # 원형도 표준편차
            np.mean(eccentricities),          # 평균 이심률
            np.std(eccentricities),           # 이심률 표준편차
            np.mean(intensities),             # 평균 강도
            np.std(intensities),              # 강도 표준편차
            np.mean(textures),                # 평균 텍스처
            np.std(textures),                 # 텍스처 표준편차
            sum(areas),                       # 총 세포 면적
            sum(areas) / (features[0]['bbox_w'] * features[0]['bbox_h'])  # 밀도
            if features else 0,
            np.min(areas) if len(areas) > 0 else 0,    # 최소 세포 크기
            np.max(areas) if len(areas) > 0 else 0,    # 최대 세포 크기
            np.percentile(areas, 25) if len(areas) > 0 else 0,   # Q1
            np.percentile(areas, 50) if len(areas) > 0 else 0,   # Q2 (중앙값)
            np.percentile(areas, 75) if len(areas) > 0 else 0,   # Q3
            np.mean([f['solidity'] for f in features]),          # 평균 밀실도
            np.mean([f['aspect_ratio'] for f in features])       # 평균 종횡비
        ])
        
        return feature_vec
    
    def visualize_segmentation(self, image: np.ndarray, masks: np.ndarray, 
                              save_path: Optional[str] = None) -> np.ndarray:
        """
        세그멘테이션 결과 시각화
        
        Returns:
            Overlay image
        """
        from cellpose import plot
        
        # Overlay 생성
        overlay = plot.mask_overlay(image, masks)
        
        if save_path:
            cv2.imwrite(save_path, overlay)
        
        return overlay
    
    def process_image(self, image_path: str, visualize: bool = False) -> Dict:
        """
        전체 파이프라인: 이미지 → 세그멘테이션 → 특징 추출
        
        Returns:
            결과 딕셔너리
        """
        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지 로드 실패: {image_path}")
        
        # 세그멘테이션
        masks, flows = self.segment_cells(image)
        
        # 특징 추출
        cell_features = self.extract_cell_features(image, masks)
        feature_vector = self.create_feature_vector(cell_features)
        
        result = {
            'image_path': str(image_path),
            'num_cells': len(cell_features),
            'masks': masks,
            'flows': flows,
            'cell_features': cell_features,
            'feature_vector': feature_vector,
            'image_shape': image.shape
        }
        
        # 시각화
        if visualize:
            overlay = self.visualize_segmentation(image, masks)
            result['overlay'] = overlay
        
        return result


def main():
    """테스트 실행"""
    print("\n" + "="*80)
    print("🔬 Advanced Cellpose Processor 테스트")
    print("="*80 + "\n")
    
    # 프로세서 초기화
    processor = AdvancedCellposeProcessor(model_type='cyto2', use_gpu=True)
    
    # 테스트 이미지 찾기
    test_dir = Path(r"c:\Users\brook\Desktop\mi_exo_ai\data\HUVEC TNF-a\HUVEC TNF-a\251209")
    
    if test_dir.exists():
        test_images = list(test_dir.glob("*.jpg"))[:3]
        
        print(f"📁 {len(test_images)}개 테스트 이미지 발견\n")
        
        for img_path in test_images:
            print(f"\n처리 중: {img_path.name}")
            
            try:
                result = processor.process_image(str(img_path), visualize=False)
                
                print(f"  ✓ 세포 수: {result['num_cells']}")
                print(f"  ✓ 특징 벡터 크기: {len(result['feature_vector'])}")
                print(f"  ✓ 평균 세포 면적: {result['feature_vector'][1]:.1f} pixels")
                print(f"  ✓ 평균 원형도: {result['feature_vector'][3]:.3f}")
                
            except Exception as e:
                print(f"  ✗ 오류: {e}")
    else:
        print(f"⚠️  테스트 디렉토리 없음: {test_dir}")
    
    print("\n" + "="*80)
    print("✅ 테스트 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
