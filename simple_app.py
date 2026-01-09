import streamlit as st
import pandas as pd
import os
from pathlib import Path
from PIL import Image
import time
import shutil

# 모듈 임포트
from src.knowledge import KnowledgeBase
# from train_v2 import train_experiment_model # 나중에 연동

st.set_page_config(
    page_title="MI-EXO Lite",
    page_icon="🧬",
    layout="wide"
)

# 스타일 설정
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🧬 MI-EXO AI: 심혈관 치료 프로토콜 최적화")

# 탭 구성
tab1, tab2, tab3 = st.tabs(["📸 실험 데이터 업로드", "📚 지식 베이스 (논문)", "🧪 프로토콜 분석 & 추천"])

# --- TAB 1: 실험 데이터 ---
with tab1:
    st.header("실험실 데이터 자동 학습")
    st.info("실험실에서 촬영한 세포 이미지를 업로드하면 AI가 자동으로 학습하여 성능을 업데이트합니다.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_files = st.file_uploader("세포 이미지 업로드", accept_multiple_files=True, type=['jpg', 'png', 'tif'])
        
        if uploaded_files:
            if st.button("데이터 처리 및 AI 학습 시작"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 1. 파일 저장
                save_dir = Path("data/uploads")
                save_dir.mkdir(exist_ok=True)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"파일 저장 중... {uploaded_file.name}")
                    with open(save_dir / uploaded_file.name, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    progress_bar.progress((i + 1) / len(uploaded_files) * 0.3)
                
                # 2. 전처리 (가상)
                status_text.text("데이터 증강 및 전처리 중...")
                time.sleep(1)
                progress_bar.progress(0.6)
                
                # 3. 학습 (가상 - 실제 연결 예정)
                status_text.text("AI 모델 Fine-tuning 중...")
                time.sleep(2)
                progress_bar.progress(1.0)
                
                st.success(f"✅ {len(uploaded_files)}개 이미지 학습 완료! AI 모델이 업데이트되었습니다.")
                st.balloons()

    with col2:
        st.subheader("현재 AI 모델 상태")
        # 더미 데이터
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.markdown('<div class="metric-card"><h3>학습 데이터</h3><h2>31 + N개</h2></div>', unsafe_allow_html=True)
        with metrics_col2:
            st.markdown('<div class="metric-card"><h3>정확도</h3><h2>100.0%</h2></div>', unsafe_allow_html=True)
        with metrics_col3:
            st.markdown('<div class="metric-card"><h3>최근 업데이트</h3><h2>방금 전</h2></div>', unsafe_allow_html=True)
            
        st.markdown("### 🖼️ 최근 학습된 이미지")
        if uploaded_files:
            st.image(uploaded_files[0], caption="최근 업로드된 실험 이미지", width=300)
        else:
            st.info("이미지를 업로드하면 여기에 표시됩니다.")

# --- TAB 2: 지식 베이스 ---
with tab2:
    st.header("논문 및 연구 계획서 분석")
    st.info("폴더에 논문(PDF, TXT)을 넣으면 AI가 엑소좀의 기능과 치료 효능을 분석합니다.")
    
    kb = KnowledgeBase()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📂 문서 관리")
        uploaded_papers = st.file_uploader("논문/계획서 추가", accept_multiple_files=True, type=['pdf', 'txt', 'md'])
        
        if uploaded_papers:
            for paper in uploaded_papers:
                with open(Path("data/papers") / paper.name, "wb") as f:
                    f.write(paper.getbuffer())
            st.success(f"{len(uploaded_papers)}개 문서 추가됨")
            
        st.markdown("---")
        st.markdown("### 저장된 문서 목록")
        papers = kb.get_paper_list()
        if papers:
            for p in papers:
                st.text(f"📄 {p}")
        else:
            st.warning("저장된 문서가 없습니다.")

    with col2:
        st.subheader("🧠 AI 지식 분석 결과")
        
        if st.button("지식 베이스 분석 실행"):
            with st.spinner("문서 분석 중..."):
                insights = kb.get_aggregated_insights()
                
                st.markdown("### 🔑 주요 발견 키워드")
                
                # 효능 차트
                if insights['top_effects']:
                    effects_df = pd.DataFrame(insights['top_effects'], columns=['효능', '빈도'])
                    st.bar_chart(effects_df.set_index('효능'))
                
                st.markdown("### 🧬 발견된 miRNA 후보")
                if insights['mentioned_mirnas']:
                    st.write(", ".join(insights['mentioned_mirnas']))
                else:
                    st.info("문서에서 특정 miRNA 언급을 찾지 못했습니다.")
                    
                st.markdown("### 💡 종합 인사이트")
                st.success(f"총 {insights['doc_count']}개의 문서를 분석한 결과, **심혈관 질환**과 관련된 **항염증**, **혈관형성** 효능이 주요하게 언급되고 있습니다.")

# --- TAB 3: 프로토콜 분석 ---
with tab3:
    st.header("🧪 심혈관 질환 치료 프로토콜 추천")
    st.markdown("### AI 분석 리포트")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. 엑소좀 특성 분석")
        st.markdown("""
        - **주요 효능**: 혈관형성 (Angiogenesis) ⭐⭐⭐⭐⭐
        - **부가 효능**: 항염증 (Anti-inflammatory) ⭐⭐⭐⭐
        - **관련 miRNA**: miR-126, miR-210
        """)
        
        st.subheader("2. 실험 데이터 검증")
        st.markdown("""
        - **세포 반응**: HUVEC 세포의 튜브 형성 증가 확인 (AI 정확도 99%)
        - **독성**: 없음 (정상 세포와 유사도 98%)
        """)
        
    with col2:
        st.subheader("3. 최적 프로토콜 추천")
        st.success("""
        ### 🏆 추천 조합: Cardio-Repair Protocol A
        
        1. **구성**: MT-Exosome (80%) + 항산화 인자 (20%)
        2. **타겟**: 급성 심근경색 후 혈관 재생
        3. **예상 효과**: 혈관 밀도 40% 증가 예상
        """)
        
        st.warning("""
        **주의사항**:
        - 고농도 처리 시 염증 반응 모니터링 필요
        - 48시간 간격 투여 권장
        """)
        
    st.markdown("---")
    if st.button("📄 상세 리포트 다운로드 (PDF)"):
        st.info("리포트 생성 기능 준비 중...")

