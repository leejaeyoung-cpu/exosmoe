import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils import set_page_config, sidebar_info, load_config
from src.fusion_prep import FusionPreprocessor
from src.data_manager import DataManager

set_page_config("퓨전 전처리")
config = load_config()

st.title("🧬 3D 퓨전 데이터 생성")
st.markdown("""
이 도구는 **Cellpose**를 사용하여 원본 이미지에서 세포 마스크를 추출하고, 
**[원본 + 마스크 + 윤곽선]**을 결합하여 3차원 텐서 데이터를 생성합니다.
""")

# Initialize Preprocessor
@st.cache_resource
def get_fusion_preprocessor_v2():
    return FusionPreprocessor()

preprocessor = get_fusion_preprocessor_v2()
dm = DataManager(config['paths']['manifest'])

tab1, tab2 = st.tabs(["🖼️ 단일 이미지 테스트", "📦 전체 데이터셋 변환"])

with tab1:
    st.subheader("이미지 업로드 및 테스트")
    uploaded_file = st.file_uploader("테스트할 이미지 업로드", type=['jpg', 'png', 'tif'])
    
    if uploaded_file:
        # Save temp
        temp_path = os.path.join("data", "temp_upload.png")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.image(temp_path, caption="원본 이미지", width=300)
        
        if st.button("퓨전 데이터 생성 실행 ⚡"):
            with st.spinner("Cellpose 분석 중... (GPU 가속)"):
                try:
                    save_path, fused_data, mask = preprocessor.process_image(temp_path)
                    
                    st.success("분석 완료!")
                    
                    # Visualize Channels
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(fused_data[:,:,0], caption="Ch1: 원본 (Normalized)", clamp=True)
                    with col2:
                        st.image(fused_data[:,:,1], caption="Ch2: 세포 마스크", clamp=True)
                    with col3:
                        st.image(fused_data[:,:,2], caption="Ch3: 세포 윤곽선", clamp=True)
                        
                    st.info(f"데이터 저장됨: `{save_path}`")
                    st.write(f"데이터 형태(Shape): {fused_data.shape}")
                    
                except Exception as e:
                    st.error(f"오류 발생: {e}")

with tab2:
    st.subheader("데이터셋 일괄 변환")
    st.warning("주의: 데이터셋의 모든 이미지를 변환합니다. 시간이 소요될 수 있습니다.")
    
    if st.button("전체 변환 시작"):
        df = dm.get_manifest()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        converted_count = 0
        new_manifest_rows = []
        
        import time
        start_time = time.time()
        total_files = len(df)
        
        for i, row in df.iterrows():
            if row['type'] == 'image' and not row['file_path'].endswith('.npy'):
                # Calculate ETA
                elapsed = time.time() - start_time
                if i > 0:
                    avg_time_per_file = elapsed / i
                    remaining_files = total_files - i
                    eta_seconds = avg_time_per_file * remaining_files
                    eta_str = f"{int(eta_seconds // 60)}분 {int(eta_seconds % 60)}초"
                else:
                    eta_str = "계산 중..."
                    
                status_text.markdown(f"**처리 중:** `{row['file_name']}`\n\n⏳ 예상 남은 시간: **{eta_str}**")
                
                try:
                    save_path, _, _ = preprocessor.process_image(row['file_path'])
                    
                    # Add new entry to manifest or replace?
                    # Let's add as a new entry with type 'fused_image'
                    new_manifest_rows.append({
                        'file_path': save_path,
                        'file_name': os.path.basename(save_path),
                        'label': row['label'],
                        'type': 'image', # Keep as image so loader picks it up, but loader handles .npy
                        'split': row['split']
                    })
                    converted_count += 1
                except Exception as e:
                    print(f"Failed {row['file_name']}: {e}")
            
            progress_bar.progress((i + 1) / total_files)
            
        if new_manifest_rows:
            # Update manifest
            new_df = pd.DataFrame(new_manifest_rows)
            # Option: Replace old images or Append? 
            # User wants to use this technique, so let's Append for now to allow comparison, 
            # or we can create a separate manifest.
            # Let's Append to the main manifest but maybe user should filter.
            # For simplicity, let's just append.
            dm.df = pd.concat([dm.df, new_df], ignore_index=True)
            dm.save_manifest()
            st.success(f"{converted_count}개 이미지 변환 완료! Manifest 업데이트됨.")
        else:
            st.info("변환할 이미지가 없거나 이미 변환되었습니다.")

sidebar_info()
