"""
MT-EXO AI Inference Engine
Cellpose + Deep Learning 통합 추론 시스템
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import sys

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.advanced_cellpose_processor import AdvancedCellposeProcessor
from src.mt_exo_model import MTEXOClassifier, ExplainableAI


class MTEXOInferenceEngine:
    """
    MT-EXO 통합 추론 엔진
    
    Pipeline:
    1. 이미지 입력
    2. Cellpose 세그멘테이션
    3. 특징 추출
    4. 딥러닝 추론
    5. Grad-CAM 설명
    """
    
    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        print(f"🔧 Inference Engine 초기화: Device={self.device}")
        
        # Cellpose 프로세서
        self.cellpose = AdvancedCellposeProcessor(model_type='cyto2', use_gpu=use_gpu)
        
        # 딥러닝 모델
        self.model = MTEXOClassifier(num_classes=5, pretrained=True)
        
        # 학습된 모델 자동 로드 (우선순위: multiclass > quick_trained)
        multiclass_model_path = Path("models/multiclass_model.pth")
        quick_model_path = Path("models/quick_trained_model.pth")
        
        if model_path and Path(model_path).exists():
            print(f"✓ 모델 로드: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        elif multiclass_model_path.exists():
            print(f"✓ 멀티클래스 모델 로드: {multiclass_model_path}")
            self.model.load_state_dict(torch.load(multiclass_model_path, map_location=self.device))
            print("  🎯 5개 기능 전체 분류 가능!")
        elif quick_model_path.exists():
            print(f"✓ 학습된 모델 로드: {quick_model_path}")
            self.model.load_state_dict(torch.load(quick_model_path, map_location=self.device))
            print("  🎯 신뢰도 향상 모델 적용!")
        else:
            print("⚠️  사전 학습된 모델 없음 - ImageNet 가중치 사용")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Explainable AI
        self.xai = ExplainableAI(self.model)
        
        # 클래스 이름
        self.class_names = self.model.class_names
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        이미지 전처리 (딥러닝 입력용)
        
        Args:
            image: BGR image (H, W, 3)
            
        Returns:
            tensor: (1, 3, 224, 224)
        """
        # BGR → RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 리사이즈
        image_resized = cv2.resize(image_rgb, (224, 224))
        
        # 정규화 (ImageNet 평균/표준편차)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        image_normalized = (image_resized / 255.0 - mean) / std
        
        # Transpose: (H, W, C) → (C, H, W)
        image_transposed = np.transpose(image_normalized, (2, 0, 1))
        
        # To tensor: (1, C, H, W)
        tensor = torch.FloatTensor(image_transposed).unsqueeze(0)
        
        return tensor
    
    def predict(self, image_path: str, explain: bool = True) -> Dict:
        """
        단일 이미지 추론
        
        Args:
            image_path: 이미지 경로
            explain: Grad-CAM 생성 여부
            
        Returns:
            결과 딕셔너리
        """
        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지 로드 실패: {image_path}")
        
        # Cellpose 세그멘테이션
        print("  🔬 Cellpose 세그멘테이션...")
        cellpose_result = self.cellpose.process_image(image_path, visualize=False)
        
        # 전처리
        input_tensor = self.preprocess_image(image)
        input_tensor = input_tensor.to(self.device)
        
        # 딥러닝 추론
        print("  🤖 AI 추론...")
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
        
        predicted_class = predicted.item()
        predicted_name = self.class_names[predicted_class]
        confidence_score = confidence.item()
        
        # 결과 구성
        result = {
            'image_path': str(image_path),
            'timestamp': datetime.now().isoformat(),
            
            # Cellpose 결과
            'cellpose': {
                'num_cells': cellpose_result['num_cells'],
                'feature_vector': cellpose_result['feature_vector'].tolist(),
                'cell_features': cellpose_result['cell_features']
            },
            
            # AI 예측
            'prediction': {
                'class_id': predicted_class,
                'class_name': predicted_name,
                'confidence': confidence_score,
                'probabilities': {
                    name: prob.item() 
                    for name, prob in zip(self.class_names, probabilities[0])
                }
            }
        }
        
        # Grad-CAM 설명
        if explain:
            print("  💡 설명 생성 (Grad-CAM)...")
            input_tensor.requires_grad = True
            heatmap = self.xai.generate_heatmap(input_tensor, target_class=predicted_class)
            
            # 오버레이 생성
            overlay = self.xai.overlay_heatmap(image, heatmap, alpha=0.4)
            
            result['explanation'] = {
                'heatmap': heatmap.tolist(),
                'overlay': overlay
            }
        
        return result
    
    def batch_predict(self, image_paths: List[str], explain: bool = False) -> List[Dict]:
        """배치 추론"""
        results = []
        
        print(f"\n🔄 {len(image_paths)}개 이미지 배치 추론 시작...\n")
        
        for i, img_path in enumerate(image_paths, 1):
            print(f"[{i}/{len(image_paths)}] {Path(img_path).name}")
            
            try:
                result = self.predict(img_path, explain=explain)
                results.append(result)
                
                print(f"  ✓ 예측: {result['prediction']['class_name']} "
                      f"(신뢰도: {result['prediction']['confidence']:.3f})")
                
            except Exception as e:
                print(f"  ✗ 오류: {e}")
                results.append({
                    'image_path': str(img_path),
                    'error': str(e)
                })
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """결과 저장"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 오버레이 이미지 제외 (JSON 직렬화 불가)
        results_json = []
        for r in results:
            r_copy = r.copy()
            if 'explanation' in r_copy and 'overlay' in r_copy['explanation']:
                del r_copy['explanation']['overlay']
            results_json.append(r_copy)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 결과 저장: {output_file}")


def main():
    """테스트 실행"""
    print("\n" + "="*80)
    print("🚀 MT-EXO AI Inference Engine 테스트")
    print("="*80 + "\n")
    
    # 엔진 초기화
    engine = MTEXOInferenceEngine(use_gpu=True)
    
    # 테스트 이미지 찾기
    test_dir = Path(r"c:\Users\brook\Desktop\mi_exo_ai\data\HUVEC TNF-a\HUVEC TNF-a\251209")
    
    if test_dir.exists():
        test_images = list(test_dir.glob("*.jpg"))[:3]
        
        if test_images:
            # 배치 추론
            results = engine.batch_predict([str(p) for p in test_images], explain=False)
            
            # 결과 요약
            print("\n" + "="*80)
            print("📊 분석 결과 요약")
            print("="*80 + "\n")
            
            for r in results:
                if 'prediction' in r:
                    print(f"이미지: {Path(r['image_path']).name}")
                    print(f"  예측: {r['prediction']['class_name']}")
                    print(f"  신뢰도: {r['prediction']['confidence']:.3f}")
                    print(f"  세포 수: {r['cellpose']['num_cells']}")
                    print()
            
            # 저장
            output_path = "data/AI_Inference_Results/inference_results.json"
            engine.save_results(results, output_path)
            
        else:
            print("⚠️  테스트 이미지 없음")
    else:
        print(f"⚠️  테스트 디렉토리 없음: {test_dir}")
    
    print("\n" + "="*80)
    print("✅ 테스트 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
