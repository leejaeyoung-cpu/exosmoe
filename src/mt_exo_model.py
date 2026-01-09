"""
Advanced MT-EXO Deep Learning Model
ResNet50 + Attention + Explainable AI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Dict, Optional
import numpy as np


class SpatialAttention(nn.Module):
    """공간적 주의 메커니즘 (중요 영역 강조)"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 채널 차원에서 평균과 최대값
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # Attention map 생성
        attention = self.sigmoid(self.conv(x_cat))
        
        return x * attention


class ChannelAttention(nn.Module):
    """채널 주의 메커니즘 (중요 특징 강조)"""
    
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        
        # Max pooling
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Attention
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * attention


class MTEXOClassifier(nn.Module):
    """
    MT-EXO 기능 분류 모델
    
    Architecture:
    - Backbone: ResNet50 (ImageNet pretrained)
    - Attention: Spatial + Channel Attention
    - Classifier: 5개 기능 분류
    """
    
    def __init__(self, num_classes=5, pretrained=True, dropout=0.5):
        super(MTEXOClassifier, self).__init__()
        
        # ResNet50 백본
        resnet = models.resnet50(pretrained=pretrained)
        
        # 백본 레이어 추출 (FC 제외)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Attention 모듈
        self.channel_attention = ChannelAttention(2048)
        self.spatial_attention = SpatialAttention()
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier Head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2048, num_classes)
        
        # 클래스 이름
        self.class_names = [
            '항산화',
            '항섬유화', 
            '항염증',
            '혈관형성',
            '세포증식'
        ]
    
    def forward(self, x):
        # 백본 통과
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Attention 적용
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        
        # Pooling & Flatten
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Dropout & Classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def predict_with_confidence(self, x):
        """예측 + 신뢰도 점수"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            
            confidence, predicted = torch.max(probabilities, dim=1)
            
        return predicted, confidence, probabilities


class ExplainableAI:
    """설명 가능한 AI (Grad-CAM)"""
    
    def __init__(self, model: MTEXOClassifier):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Hook 등록
        self.model.layer4.register_forward_hook(self.save_activation)
        self.model.layer4.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Forward hook: activation 저장"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Backward hook: gradient 저장"""
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Grad-CAM 히트맵 생성
        
        Args:
            input_tensor: 입력 이미지 (1, 3, H, W)
            target_class: None이면 예측 클래스, 지정하면 해당 클래스
            
        Returns:
            heatmap: numpy array (H, W)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Grad-CAM 계산
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        # 히트맵
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)  # ReLU
        heatmap /= torch.max(heatmap)  # Normalize
        
        return heatmap.cpu().numpy()
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4):
        """
        원본 이미지에 히트맵 오버레이
        
        Args:
            image: 원본 이미지 (H, W, 3) numpy array
            heatmap: Grad-CAM 히트맵 (H, W)
            alpha: 투명도
            
        Returns:
            overlayed image
        """
        import cv2
        
        # 히트맵 리사이즈
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # 컬러맵 적용
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), 
            cv2.COLORMAP_JET
        )
        
        # 오버레이
        overlay = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
        
        return overlay


def create_model(pretrained=True, num_classes=5):
    """모델 생성 헬퍼 함수"""
    model = MTEXOClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=0.5
    )
    return model


def test_model():
    """모델 테스트"""
    print("\n" + "="*80)
    print("🤖 MT-EXO Deep Learning Model 테스트")
    print("="*80 + "\n")
    
    # 모델 생성
    print("모델 생성 중...")
    model = create_model(pretrained=True, num_classes=5)
    
    # 모델 정보
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ 총 파라미터: {total_params:,}")
    print(f"✓ 학습 가능 파라미터: {trainable_params:,}")
    print(f"✓ 클래스 수: 5 ({', '.join(model.class_names)})")
    
    # 더미 입력으로 테스트
    print("\n순전파 테스트 중...")
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch=1, RGB, 224x224
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        predicted, confidence, probabilities = model.predict_with_confidence(dummy_input)
    
    print(f"✓ 출력 shape: {output.shape}")
    print(f"✓ 예측 클래스: {model.class_names[predicted.item()]}")
    print(f"✓ 신뢰도: {confidence.item():.3f}")
    
    print("\n클래스별 확률:")
    for i, (name, prob) in enumerate(zip(model.class_names, probabilities[0])):
        print(f"  {i+1}. {name:12s}: {prob.item():.3f}")
    
    # Explainable AI 테스트
    print("\nGrad-CAM 테스트 중...")
    xai = ExplainableAI(model)
    
    dummy_input.requires_grad = True
    heatmap = xai.generate_heatmap(dummy_input)
    
    print(f"✓ 히트맵 shape: {heatmap.shape}")
    print(f"✓ 히트맵 범위: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    
    print("\n" + "="*80)
    print("✅ 모델 테스트 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_model()
