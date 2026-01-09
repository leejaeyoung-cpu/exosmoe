import streamlit as st
from src.utils import set_page_config, sidebar_info, load_css

# Page Setup
set_page_config("대시보드")
load_css()

# Header
st.title("🧬 Mela-Exosome AI 플랫폼")
st.markdown("### 환영합니다! 👋")
st.markdown("이 플랫폼은 **멀티모달 딥러닝**을 활용하여 엑소좀 데이터를 분석합니다.")

# Dashboard Widgets
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📂 데이터 관리")
    st.write("이미지와 오믹스 데이터를 업로드하고 관리합니다.")
    st.page_link("pages/1_📂_데이터_관리.py", label="데이터 관리 바로가기", icon="📂")

with col2:
    st.markdown("### 🫀 CKD-CVD miRNA")
    st.write("만성 신장질환·심혈관질환 치료용 miRNA 선별")
    st.page_link("pages/8_🫀_CKD_CVD_miRNA_선별.py", label="CKD-CVD 선별", icon="🫀")

with col3:
    st.markdown("### 💊 농도 추론")
    st.write("엑소좀 치료제 최적 농도 및 투여 프로토콜 계산")
    st.page_link("pages/9_💊_농도_추론_최적화.py", label="농도 최적화", icon="💊")

st.markdown("---")

st.subheader("🔬 추가 기능")
col4, col5, col6, col7 = st.columns(4)

with col4:
    st.page_link("pages/2_🚀_모델_학습.py", label="모델 학습", icon="🚀")

with col5:
    st.page_link("pages/6_🧬_MT_EXO_분석.py", label="MT-EXO 분석", icon="🧬")

with col6:
    st.page_link("pages/7_🤖_AI_추론_분석.py", label="AI 추론 분석", icon="🤖")

with col7:
    st.page_link("pages/5_🔬_연구_검증_플랫폼.py", label="연구 검증", icon="🔬")

st.markdown("---")

# System Status (Placeholder)
st.subheader("📊 시스템 현황")
col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("학습된 모델 수", "1 개", "최신: best_model.pth")
col_s2.metric("총 데이터 샘플", "확인 필요", "dataset_manifest.csv")
col_s3.metric("GPU 상태", "사용 가능", "CUDA")

sidebar_info()
