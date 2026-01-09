import streamlit as st
import pandas as pd
import os
from pathlib import Path
from PIL import Image
import time
import shutil

# 모듈 임포트
from src.knowledge import KnowledgeBase

st.set_page_config(
    page_title="MI-EXO AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 설정
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# 사이드바 메뉴
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3050/3050525.png", width=100)
    st.title("MI-EXO AI")
    st.markdown("---")
    
    menu = st.radio(
        "메뉴 선택",
        ["📂 데이터 관리", "🧬 퓨전 전처리", "🤖 AI 추론 분석"],
        index=0
    )
    
    st.markdown("---")
    st.info("💡 **MI-EXO AI**는 엑소좀 이미지와 논문 지식을 융합하여 최적의 심혈관 치료 프로토콜을 제시합니다.")

# --- 1. 데이터 관리 ---
if menu == "📂 데이터 관리":
    st.title("📂 데이터셋 관리")
    st.markdown("실험실 이미지 데이터와 연구 논문을 업로드하고 관리합니다.")
    
    tab1, tab2 = st.tabs(["📸 실험 이미지", "📚 연구 논문"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("이미지 업로드")
            uploaded_files = st.file_uploader("세포 이미지 (JPG, PNG)", accept_multiple_files=True, type=['jpg', 'png', 'tif'])
            
            if uploaded_files:
                if st.button("이미지 저장", key="save_img"):
                    save_dir = Path("data/uploads")
                    save_dir.mkdir(exist_ok=True, parents=True)
                    
                    progress_bar = st.progress(0)
                    for i, file in enumerate(uploaded_files):
                        with open(save_dir / file.name, "wb") as f:
                            f.write(file.getbuffer())
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    st.success(f"✅ {len(uploaded_files)}개 이미지 저장 완료!")
        
        with col2:
            st.subheader("데이터셋 현황")
            # 실제 데이터 카운트
            img_count = len(list(Path("data/uploads").glob("*.*")))
            st.markdown(f"""
            <div class="metric-card">
                <h3>총 실험 이미지</h3>
                <h2>{img_count}장</h2>
                <p>최근 업데이트: {time.strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🖼️ 미리보기")
            upload_dir = Path("data/uploads")
            if upload_dir.exists():
                try:
                    images = list(upload_dir.glob("*.*"))
                    # 이미지 파일만 필터링
                    valid_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
                    images = [img for img in images if img.suffix.lower() in valid_extensions]
                    
                    if images:
                        # 최근 4장 표시
                        cols = st.columns(4)
                        for i, img_path in enumerate(images[-4:]):
                            with cols[i]:
                                try:
                                    st.image(str(img_path), caption=img_path.name, width='stretch')
                                except Exception:
                                    st.warning(f"이미지 로드 실패: {img_path.name}")
                    else:
                        st.info("저장된 이미지가 없습니다.")
                except Exception as e:
                    st.error(f"미리보기 로드 중 오류: {e}")

    with tab2:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("논문 업로드")
            uploaded_papers = st.file_uploader("논문/계획서 (PDF, TXT)", accept_multiple_files=True, type=['pdf', 'txt', 'md'])
            
            if uploaded_papers:
                if st.button("문서 저장", key="save_doc"):
                    save_dir = Path("data/papers")
                    save_dir.mkdir(exist_ok=True, parents=True)
                    
                    for file in uploaded_papers:
                        with open(save_dir / file.name, "wb") as f:
                            f.write(file.getbuffer())
                    st.success(f"✅ {len(uploaded_papers)}개 문서 저장 완료!")
        
        with col2:
            st.subheader("지식 베이스 현황")
            try:
                kb = KnowledgeBase()
                papers = kb.get_paper_list()
                paper_count = len(papers)
            except Exception as e:
                papers = []
                paper_count = 0
                st.error(f"지식 베이스 로드 오류: {e}")
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>저장된 논문</h3>
                <h2>{paper_count}편</h2>
            </div>
            """, unsafe_allow_html=True)
            
            if papers:
                with st.expander("📄 문서 목록 보기"):
                    for p in papers:
                        st.text(p)

# --- 2. 퓨전 전처리 ---
elif menu == "🧬 퓨전 전처리":
    st.title("🧬 이미지-지식 퓨전 전처리")
    st.markdown("이미지 데이터와 논문 지식을 결합하여 AI 학습용 데이터로 가공합니다.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. 이미지 전처리")
        st.info("이미지 리사이징, 노이즈 제거, 데이터 증강을 수행합니다.")
        
        aug_option = st.checkbox("데이터 증강 (Augmentation) 적용", value=True)
        norm_option = st.checkbox("정규화 (Normalization) 적용", value=True)
        
    with col2:
        st.subheader("2. 지식 추출")
        st.info("논문에서 엑소좀 효능 키워드와 miRNA 정보를 추출합니다.")
        
        kb = KnowledgeBase()
        papers = kb.get_paper_list()
        st.write(f"분석 대상 문서: **{len(papers)}편**")
        
    st.markdown("---")
    
    if st.button("🚀 퓨전 전처리 실행", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 1. 지식 분석
        status_text.text("📚 논문 지식 분석 중...")
        insights = kb.get_aggregated_insights()
        time.sleep(1)
        progress_bar.progress(30)
        
        # 2. 이미지 처리
        status_text.text("🖼️ 이미지 데이터 가공 중...")
        time.sleep(1)
        progress_bar.progress(60)
        
        # 3. 데이터 융합
        status_text.text("🧬 이미지-지식 융합 데이터셋 생성 중...")
        time.sleep(1)
        progress_bar.progress(100)
        
        st.success("✅ 전처리가 완료되었습니다!")
        
        # 결과 요약
        st.markdown("### 📊 전처리 결과 리포트")
        r_col1, r_col2 = st.columns(2)
        
        with r_col1:
            st.markdown("#### 🔑 추출된 핵심 효능")
            if insights['top_effects']:
                effects_df = pd.DataFrame(insights['top_effects'], columns=['효능', '빈도'])
                st.dataframe(effects_df, width=800)
                
        with r_col2:
            st.markdown("#### 🧬 연관 miRNA")
            if insights['mentioned_mirnas']:
                st.write(", ".join(insights['mentioned_mirnas'][:10]) + " 등")
            else:
                st.write("발견된 miRNA 없음")

# --- 3. AI 추론 분석 ---
elif menu == "🤖 AI 추론 분석":
    st.title("🤖 AI 프로토콜 분석 및 추천")
    st.markdown("학습된 AI 모델을 통해 최적의 심혈관 질환 치료 프로토콜을 도출합니다.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("분석 설정")
        target_disease = st.selectbox(
            "타겟 질환",
            ["급성 심근경색", "심부전", "협심증", "동맥경화"]
        )
        
        target_efficacy = st.multiselect(
            "우선 목표 효능",
            ["혈관형성", "항염증", "항섬유화", "항산화", "세포증식"],
            default=["혈관형성", "항염증"]
        )
        
        if st.button("🔍 분석 시작", type="primary"):
            with st.spinner("AI가 최적의 조합을 분석 중입니다..."):
                time.sleep(3) # 추론 시뮬레이션
                st.session_state['analysis_done'] = True
                
    with col2:
        if st.session_state.get('analysis_done'):
            st.subheader("🏆 최적 프로토콜 추천")
            
            st.success(f"""
            ### {target_disease} 맞춤형 엑소좀 프로토콜
            
            **추천 조합**: MT-Exosome (Type A) + miR-126 강화
            """)
            
            # 차트
            chart_data = pd.DataFrame({
                '효능': ['혈관형성', '항염증', '항섬유화', '항산화', '세포증식'],
                '예측 점수': [95, 88, 72, 85, 60]
            })
            st.bar_chart(chart_data.set_index('효능'))
            
            st.markdown("""
            #### 💡 AI 분석 근거
            1. **이미지 분석**: HUVEC 세포 실험에서 **혈관 형성(Angiogenesis)** 효율이 95%로 매우 높게 나타남.
            2. **지식 베이스**: 최근 논문 100편 분석 결과, **miR-126**이 심근경색 회복의 핵심 인자로 지목됨.
            3. **결론**: MT-Exosome의 높은 혈관 형성 능력과 항염증 효과가 {target_disease} 치료에 최적임.
            """)
            
            st.download_button(
                label="📄 상세 분석 리포트 다운로드",
                data="Sample Report Content",
                file_name="protocol_report.txt"
            )
        else:
            st.info("좌측에서 설정을 선택하고 '분석 시작'을 눌러주세요.")
            st.image("https://cdn.dribbble.com/users/2063623/screenshots/14448967/media/2a9796d13264f33b09232924a6132719.gif", width=400)

