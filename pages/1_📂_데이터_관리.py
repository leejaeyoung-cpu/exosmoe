import streamlit as st
import pandas as pd
import plotly.express as px
from src.utils import set_page_config, sidebar_info, load_config
from src.data_manager import DataManager

set_page_config("데이터 관리")
config = load_config()

st.title("📂 데이터 관리")

# Initialize Data Manager
dm = DataManager(config['paths']['manifest'])

tab1, tab2 = st.tabs(["📊 데이터 통계", "➕ 데이터 추가"])

with tab1:
    st.subheader("현재 데이터셋 현황")
    df = dm.get_manifest()
    
    if not df.empty:
        # Stats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 클래스별 분포")
            counts = df['label'].value_counts().reset_index()
            counts.columns = ['Label', 'Count']
            fig = px.pie(counts, values='Count', names='Label', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("#### 데이터 미리보기")
            st.dataframe(df, height=400)
    else:
        st.warning("데이터가 없습니다. 데이터를 추가해주세요.")

with tab2:
    st.subheader("새로운 데이터 업로드")
    
    with st.form("upload_form"):
        uploaded_files = st.file_uploader("이미지 파일 선택", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
        
        label = st.selectbox("레이블(Label) 선택", config['classes'])
        new_label = st.text_input("또는 새로운 레이블 입력 (선택사항)")
        
        if new_label:
            label = new_label
            
        split = st.selectbox("데이터셋 분할", ["train", "val", "test"])
        
        submitted = st.form_submit_button("업로드 및 저장")
        
        if submitted and uploaded_files:
            with st.spinner("파일 저장 중..."):
                success = dm.add_files(uploaded_files, label, split)
                if success:
                    st.success(f"{len(uploaded_files)}개 파일이 성공적으로 추가되었습니다!")
                    st.rerun()
                else:
                    st.error("파일 저장 실패")

sidebar_info()
