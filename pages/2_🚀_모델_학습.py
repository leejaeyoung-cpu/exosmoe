import streamlit as st
import pandas as pd
import time
import plotly.express as px
from src.utils import set_page_config, sidebar_info, load_config
from src.trainer import Trainer

set_page_config("모델 학습")
config = load_config()

st.title("🚀 모델 학습")

# Singleton Trainer (using st.session_state to persist across reruns)
if 'trainer' not in st.session_state:
    st.session_state.trainer = Trainer()

trainer = st.session_state.trainer

# Sidebar Config
with st.sidebar:
    st.header("학습 설정")
    epochs = st.number_input("Epochs", min_value=1, value=config['training']['default_epochs'])
    batch_size = st.number_input("Batch Size", min_value=1, value=config['training']['default_batch_size'])
    lr = st.number_input("Learning Rate", value=config['training']['default_learning_rate'], format="%.4f")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("제어 패널")
    if trainer.is_running():
        st.info("학습이 진행 중입니다... 🏃‍♂️")
        if st.button("학습 중지 🛑", type="primary"):
            trainer.stop_training()
            st.rerun()
    else:
        st.success("대기 중")
        if st.button("학습 시작 ▶️"):
            trainer.start_training(epochs, batch_size, lr)
            st.rerun()

with col2:
    st.subheader("학습 현황")
    log_placeholder = st.empty()
    
    # Auto-refresh loop if running
    if trainer.is_running():
        while trainer.is_running():
            df = trainer.get_logs()
            if not df.empty:
                with log_placeholder.container():
                    # Plot Loss
                    fig = px.line(df, x='epoch', y='val_loss', title='Validation Loss')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Plot Acc
                    fig2 = px.line(df, x='epoch', y='val_acc', title='Validation Accuracy')
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    st.dataframe(df.tail(5), use_container_width=True)
            
            time.sleep(2)
            
    else:
        # Show final logs if exists
        df = trainer.get_logs()
        if not df.empty:
            st.line_chart(df.set_index('epoch')[['val_loss', 'val_acc']])
            st.dataframe(df)

sidebar_info()
