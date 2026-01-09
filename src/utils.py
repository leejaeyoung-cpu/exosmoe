import streamlit as st
import yaml
import os

def load_config():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_page_config(page_title):
    st.set_page_config(
        page_title=f"{page_title} | Mela-Exosome AI",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def sidebar_info():
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ℹ️ 정보")
        st.info(
            """
            **Mela-Exosome AI Platform**
            
            이 프로그램은 엑소좀 이미지와 오믹스 데이터를 
            분석하여 진단/예측하는 
            AI 시스템입니다.
            """
        )
        st.markdown("---")
        st.caption("Developed by Google Deepmind Team")

def load_css():
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            font-weight: bold;
        }
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
        </style>
    """, unsafe_allow_html=True)
