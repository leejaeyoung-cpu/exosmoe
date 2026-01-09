"""
🤖 AI 추론 분석 페이지
Cellpose + Deep Learning 기반 MT-EXO 기능 자동 분류
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import cv2
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils import set_page_config, sidebar_info

# 페이지 설정
set_page_config("AI 추론 분석")

st.title("🤖 AI 추론 분석")
st.markdown("### Cellpose + Deep Learning 기반 세포 기능 자동 분류")

# Session state 초기화
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

# 정보 박스
with st.expander("ℹ️ 시스템 정보", expanded=False):
    st.markdown("""
    ### 🔬 분석 파이프라인
    
    1. **Cellpose 세그멘테이션**
       - 세포 자동 감지 및 분리
       - 20차원 특징 벡터 추출
    
    2. **딥러닝 분류**
       - ResNet50 + Attention Mechanism
       - 5개 기능 분류 (항산화, 항섬유화, 항염증, 혈관형성, 세포증식)
    
    3. **설명 가능한 AI**
       - Grad-CAM으로 중요 영역 시각화
       - 신뢰도 점수 제공
    """)

# 탭 구성
tab1, tab2, tab3 = st.tabs(["📤 이미지 업로드", "🔬 HUVEC 데이터 분석", "📊 분석 결과"])

# === 탭 1: 이미지 업로드 ===
with tab1:
    st.header("이미지 업로드 및 분석")
    
    uploaded_files = st.file_uploader(
        "세포 이미지를 업로드하세요",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )
    
    col1, col2 = st.columns(2)
    with col1:
        run_cellpose = st.checkbox("Cellpose 세그멘테이션", value=True)
    with col2:
        run_gradcam = st.checkbox("Grad-CAM 설명", value=False)
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)}개 이미지 업로드됨")
        
        if st.button("🚀 AI 분석 시작", type="primary", use_container_width=True):
            
            # 모델 로딩
            with st.spinner("AI 엔진 초기화 중..."):
                try:
                    from src.mt_exo_inference import MTEXOInferenceEngine
                    engine = MTEXOInferenceEngine(use_gpu=True)
                    st.success("✅ AI 엔진 로드 완료!")
                except Exception as e:
                    st.error(f"❌ 엔진 로드 실패: {e}")
                    st.stop()
            
            # 임시 저장
            temp_dir = project_root / "data" / "temp_ai_inference"
            temp_dir.mkdir(exist_ok=True, parents=True)
            
            image_paths = []
            for uploaded_file in uploaded_files:
                file_path = temp_dir / uploaded_file.name
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                image_paths.append(str(file_path))
            
            # 분석 실행
            with st.spinner(f"AI 추론 중... ({len(image_paths)}개 이미지)"):
                try:
                    results = engine.batch_predict(image_paths, explain=run_gradcam)
                    
                    # 세션에 저장
                    st.session_state.analysis_results = results
                    st.session_state.show_results = True
                    
                    st.success(f"✅ {len(results)}개 이미지 분석 완료!")
                    
                    # 페이지 재실행하여 결과 표시
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ 분석 실패: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.stop()
    
    # 결과 표시
    if st.session_state.show_results and len(st.session_state.analysis_results) > 0:
        st.markdown("---")
        st.subheader("📊 분석 결과")
        
        for i, result in enumerate(st.session_state.analysis_results):
            if 'prediction' not in result:
                continue
            
            with st.container():
                st.markdown(f"### 📷 이미지 #{i+1}")
                
                col1, col2, col3 = st.columns([2, 2, 3])
                
                with col1:
                    # 원본 이미지
                    try:
                        img = Image.open(result['image_path'])
                        st.image(img, caption="원본 이미지", use_container_width=True)
                    except:
                        st.warning("이미지 표시 실패")
                
                with col2:
                    # 예측 결과
                    st.metric("🎯 예측 기능", result['prediction']['class_name'])
                    st.metric("📊 신뢰도", f"{result['prediction']['confidence']:.1%}")
                    
                    if 'cellpose' in result:
                        st.metric("🔬 검출 세포", result['cellpose']['num_cells'])
                
                with col3:
                    # 확률 분포
                    probs = result['prediction']['probabilities']
                    prob_df = pd.DataFrame({
                        '기능': list(probs.keys()),
                        '확률': list(probs.values())
                    })
                    
                    fig = px.bar(
                        prob_df, 
                        x='기능', 
                        y='확률',
                        title='기능별 확률 분포',
                        color='확률',
                        color_continuous_scale='Viridis',
                        range_y=[0, 1]
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True, key=f"prob_chart_{i}")
                
                st.divider()
        
        # 초기화 버튼
        if st.button("🔄 새로운 분석 시작"):
            st.session_state.analysis_results = []
            st.session_state.show_results = False
            st.rerun()

# === 탭 2: HUVEC 데이터 분석 ===
with tab2:
    st.header("HUVEC TNF-α 데이터 자동 분석")
    
    st.info("💡 기존 HUVEC 데이터를 AI 모델로 자동 분류합니다.")
    
    huvec_dir = project_root / "data" / "HUVEC TNF-a" / "HUVEC TNF-a" / "251209"
    
    if huvec_dir.exists():
        image_files = list(huvec_dir.glob("*.jpg"))
        st.write(f"📁 발견된 이미지: {len(image_files)}개")
        
        num_analyze = st.slider("분석할 이미지 수", 1, min(len(image_files), 20), 6)
        
        if st.button("🔬 자동 분석 시작", type="primary"):
            with st.spinner("AI 엔진 로딩..."):
                try:
                    from src.mt_exo_inference import MTEXOInferenceEngine
                    engine = MTEXOInferenceEngine(use_gpu=True)
                    
                    selected_images = [str(f) for f in image_files[:num_analyze]]
                    
                    with st.spinner(f"{num_analyze}개 이미지 분석 중..."):
                        results = engine.batch_predict(selected_images, explain=False)
                    
                    # 결과 데이터프레임
                    df_results = pd.DataFrame([
                        {
                            '이미지': Path(r['image_path']).name,
                            '예측 기능': r['prediction']['class_name'],
                            '신뢰도': r['prediction']['confidence'],
                            '세포 수': r['cellpose']['num_cells']
                        }
                        for r in results if 'prediction' in r
                    ])
                    
                    st.success(f"✅ {len(df_results)}개 분석 완료!")
                    
                    # 요약
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("평균 신뢰도", f"{df_results['신뢰도'].mean():.1%}")
                    with col2:
                        st.metric("평균 세포 수", f"{df_results['세포 수'].mean():.0f}")
                    with col3:
                        most_common = df_results['예측 기능'].mode()[0]
                        st.metric("주요 기능", most_common)
                    
                    # 테이블
                    st.dataframe(df_results, use_container_width=True, height=400)
                    
                    # 분포 차트
                    fig = px.histogram(
                        df_results, 
                        x='예측 기능',
                        title='기능 분류 분포',
                        color='예측 기능'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ 오류: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.warning(f"⚠️ HUVEC 데이터가 없습니다: {huvec_dir}")

# === 탭 3: 분석 결과 ===
with tab3:
    st.header("저장된 분석 결과")
    
    results_path = project_root / "data" / "AI_Inference_Results" / "inference_results.json"
    
    if results_path.exists():
        import json
        with open(results_path, 'r', encoding='utf-8') as f:
            saved_results = json.load(f)
        
        st.success(f"✅ {len(saved_results)}개 분석 기록 발견")
        
        # 결과 표시
        for i, result in enumerate(saved_results, 1):
            if 'prediction' in result:
                with st.expander(f"#{i} - {Path(result['image_path']).name}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**예측 정보**")
                        st.write(f"- 기능: {result['prediction']['class_name']}")
                        st.write(f"- 신뢰도: {result['prediction']['confidence']:.3f}")
                        st.write(f"- 분석 시각: {result['timestamp']}")
                    
                    with col2:
                        st.write("**Cellpose 정보**")
                        st.write(f"- 세포 수: {result['cellpose']['num_cells']}")
                        st.write(f"- 특징 차원: {len(result['cellpose']['feature_vector'])}")
    else:
        st.info("💡 아직 저장된 분석 결과가 없습니다.")

# 사이드바
sidebar_info()

# 모델 상태
with st.sidebar:
    st.header("🤖 모델 상태")
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        
        if gpu_available:
            st.success("✅ GPU 사용 가능")
            st.write(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("⚠️ CPU 모드")
        
        st.write(f"PyTorch: {torch.__version__}")
        
    except:
        st.error("❌ PyTorch 미설치")
    
    st.divider()
    
    st.markdown("""
    ### 💡 사용 팁
    
    - GPU 사용 시 훨씬 빠름
    - 배치 처리로 여러 이미지 동시 분석
    - Grad-CAM으로 AI 판단 근거 확인
    """)
