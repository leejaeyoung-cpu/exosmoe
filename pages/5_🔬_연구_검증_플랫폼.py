import streamlit as st
import pandas as pd
import plotly.express as px
import os
from src.utils import set_page_config, sidebar_info
from src.analytics import ResearchAnalytics, AutoAnalyzer
from src.fusion_prep import FusionPreprocessor

set_page_config("연구 검증 플랫폼")
analytics = ResearchAnalytics()

# Initialize Session State
if 'fusion_prep' not in st.session_state:
    st.session_state.fusion_prep = FusionPreprocessor()

st.title("🔬 연구 검증 플랫폼")
st.markdown("신약 개발 연구 계획에 따른 **데이터 검증 및 최적화** 도구입니다.")

# Create Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 세포 증식능 (CPDL)", "✅ 엑소좀 QC 판정", "🧪 효능/독성 평가", "🧬 miRNA 후보 도출"])

# --- Tab 1: CPDL ---
with tab1:
    st.header("1. 멜라토닌 농도별 세포 증식능 확인")
    st.info("CPDL = (log(NH) - log(NI)) / log(2)")
    
    # Input Data
    st.subheader("실험 데이터 입력")
    
    # Example Data
    default_data = pd.DataFrame({
        'Concentration (uM)': [0, 1, 10, 100, 200],
        'Time (h)': [48, 48, 48, 48, 48],
        'N_Initial': [10000, 10000, 10000, 10000, 10000],
        'N_Harvested': [35000, 42000, 55000, 48000, 20000]
    })
    
    edited_df = st.data_editor(default_data, num_rows="dynamic")
    
    if st.button("CPDL 분석 실행"):
        result_df, best_cond = analytics.analyze_proliferation(edited_df)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(result_df, x='Concentration (uM)', y='CPDL', 
                         title="농도별 세포 증식능 (CPDL)", color='CPDL')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.success(f"🏆 최적 농도: **{best_cond['Concentration (uM)']} uM**")
            st.metric("최대 CPDL", f"{best_cond['CPDL']:.2f}")
            st.write(result_df)

# --- Tab 2: QC ---
with tab2:
    st.header("2. 엑소좀 품질 관리 (QC)")
    st.markdown("생산된 엑소좀 Lot의 품질 적합성을 판정합니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        size = st.number_input("평균 입자 크기 (nm)", value=120.0)
        zeta = st.number_input("제타 전위 (mV)", value=-25.0)
    with col2:
        dna = st.number_input("DNA 잔존량 (pg/10^8)", value=30.0)
        viability = st.number_input("세포 생존율 (%)", value=95.0)
        
    if st.button("QC 판정 실행"):
        data = {'size': size, 'zeta': zeta, 'dna': dna, 'viability': viability}
        passed, results = analytics.evaluate_qc(data)
        
        if passed:
            st.success("✅ **적합 (Pass)**: 모든 기준을 충족합니다.")
        else:
            st.error("❌ **부적합 (Fail)**: 기준 미달 항목이 있습니다.")
            
        # Detail Table
        res_list = []
        for key, val in results.items():
            res_list.append({
                '항목': key,
                '측정값': val['value'],
                '기준': val['criteria'],
                '판정': 'Pass' if val['pass'] else 'Fail'
            })
        st.table(pd.DataFrame(res_list))

# --- Tab 3: Efficacy/Toxicity ---
with tab3:
    st.header("3. 효능 및 독성 평가")
    
    st.subheader("독성 평가 (MTT Assay)")
    # Simple visualization for Toxicity
    tox_data = pd.DataFrame({
        'Group': ['Control', 'Low Dose', 'Mid Dose', 'High Dose', 'Positive Ctrl (DMSO)'],
        'Viability (%)': [100, 98, 95, 92, 15]
    })
    st.write("예시 데이터:")
    st.dataframe(tox_data)
    
    fig_tox = px.bar(tox_data, x='Group', y='Viability (%)', color='Viability (%)', 
                     range_y=[0, 120], title="세포 독성 평가 결과")
    # Add threshold line
    fig_tox.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="Safety Limit (90%)")
    st.plotly_chart(fig_tox, use_container_width=True)
    
    st.subheader("효능 검증 (이미지 분석)")
    st.info("조직 염색 이미지를 업로드하여 섬유화 면적을 계산할 수 있습니다. (준비 중)")

# --- Tab 4: miRNA Discovery ---
with tab4:
    st.header("4. 심혈관 질환 치료용 핵심 miRNA 발굴")
    st.markdown("""
    **마이크로어레이(Microarray)** 결과를 분석하여 멜라토닌 처리에 의해 **증가(Upregulated)**된 
    핵심 miRNA를 도출하고, AI 기반으로 기능을 예측합니다.
    """)
    
    # Upload Data (Microarray OR Image)
    uploaded_files = st.file_uploader("데이터 업로드 (Microarray Excel/CSV 또는 세포 이미지)", type=['xlsx', 'csv', 'jpg', 'png', 'tif'], accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.markdown(f"---")
            st.markdown(f"### 📄 파일 분석: **{uploaded_file.name}**")
            
            try:
                # Case A: Image File
                if uploaded_file.type.startswith('image') or uploaded_file.name.endswith(('.jpg', '.png', '.tif')):
                    st.info("🖼️ 이미지 파일이 감지되었습니다. 퓨전 분석(Cellpose)을 시작합니다.")
                    
                    # Save temp
                    temp_path = os.path.join("data", uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    with st.spinner(f"세포 분석 중... ({uploaded_file.name})"):
                        save_path, fused_data, mask = st.session_state.fusion_prep.process_image(temp_path)
                        
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(temp_path, caption="원본 이미지", use_column_width=True)
                    with col2:
                        st.image(fused_data[:,:,1], caption="AI 분석 마스크 (Nuclei/Cell)", use_column_width=True)
                        
                    st.success("✅ 이미지 분석 완료! 세포 구조가 성공적으로 추출되었습니다.")
                    
                # Case B: Microarray Data
                else:
                    if uploaded_file.name.endswith('.csv'):
                        df_mirna = pd.read_csv(uploaded_file)
                    else:
                        df_mirna = pd.read_excel(uploaded_file)
                        
                    st.write("마이크로어레이 데이터 미리보기:", df_mirna.head())
                    
                    # Column Selection
                    cols = df_mirna.columns.tolist()
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        name_col = st.selectbox(f"miRNA 이름 컬럼 ({uploaded_file.name})", cols, index=0, key=f"name_{uploaded_file.name}")
                    with col2:
                        ctrl_col = st.selectbox(f"Control 값 ({uploaded_file.name})", cols, index=1, key=f"ctrl_{uploaded_file.name}")
                    with col3:
                        treat_col = st.selectbox(f"MT-EXO 값 ({uploaded_file.name})", cols, index=2, key=f"treat_{uploaded_file.name}")
                        
                    if st.button(f"후보 물질 도출 실행 🧬 ({uploaded_file.name})", key=f"btn_{uploaded_file.name}"):
                        # Prepare DF
                        analysis_df = df_mirna[[name_col, ctrl_col, treat_col]].copy()
                        analysis_df.columns = ['miRNA_Name', 'Control', 'MT_EXO']
                        
                        full_df, sig_df = analytics.analyze_microarray(analysis_df, 'Control', 'MT_EXO')
                        
                        # 1. Volcano Plot (Simplified: Log2FC vs Score)
                        st.subheader("📊 발현 차이 분석 (Differential Expression)")
                        fig_vol = px.scatter(full_df, x='Log2FC', y='MT_EXO', hover_data=['miRNA_Name'],
                                             color='Log2FC', title="Fold Change 분포",
                                             color_continuous_scale='RdBu_r')
                        st.plotly_chart(fig_vol, use_container_width=True)
                        
                        # 2. Top Candidates
                        st.subheader("🏆 Top 10 유력 후보 miRNA")
                        st.markdown("항산화, 항섬유화, 혈관형성 등 **심혈관 질환 치료**와 관련된 기능을 가진 후보군입니다.")
                        
                        top_candidates = sig_df.head(10)
                        st.dataframe(top_candidates[['miRNA_Name', 'Log2FC', 'Predicted_Function']].style.background_gradient(subset=['Log2FC'], cmap='Greens'))
                        
                        # 3. AI Insight
                        st.info(f"""
                        **AI 분석 리포트**:
                        총 {len(sig_df)}개의 miRNA가 유의미하게 증가했습니다.
                        그 중 **{top_candidates.iloc[0]['miRNA_Name']}**가 가장 강력한 후보이며, 
                        예측된 기능은 **'{top_candidates.iloc[0]['Predicted_Function']}'** 입니다.
                        """)
                    
            except Exception as e:
                st.error(f"데이터 분석 중 오류 발생 ({uploaded_file.name}): {e}")
    else:
        # Demo Data Button
        if st.button("데모 데이터로 테스트"):
            demo_data = pd.DataFrame({
                'miRNA_Name': ['hsa-miR-21-5p', 'hsa-miR-126-3p', 'hsa-miR-146a-5p', 'hsa-miR-155', 'hsa-let-7a'],
                'Control_Signal': [100, 50, 80, 120, 200],
                'MT_EXO_Signal': [500, 300, 160, 100, 220] # miR-21 (5x), miR-126 (6x), miR-146a (2x)
            })
            # Save and reload logic simulated
            st.write("데모 데이터 로드됨:", demo_data)
            
            analysis_df = demo_data.copy()
            analysis_df.columns = ['miRNA_Name', 'Control', 'MT_EXO']
            full_df, sig_df = analytics.analyze_microarray(analysis_df, 'Control', 'MT_EXO')
            
            st.subheader("🏆 Top 후보 (데모 결과)")
            st.dataframe(sig_df[['miRNA_Name', 'Log2FC', 'Predicted_Function']])

# --- Sidebar Auto-Analysis ---
st.sidebar.markdown("---")
st.sidebar.header("🤖 자동 분석 (Auto-Analysis)")
if st.sidebar.button("전체 데이터 스캔 및 분석"):
    analyzer = AutoAnalyzer()
    
    with st.spinner("데이터 폴더 스캔 중..."):
        summary = analyzer.scan_and_analyze()
        
    st.toast("분석 완료!", icon="✅")
    
    # Display Summary in a Modal or Expander
    with st.expander("📊 자동 분석 리포트", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("발견된 마이크로어레이 파일", f"{len(summary['mirna_files'])}개")
        with col2:
            st.metric("발견된 이미지 파일", f"{summary['fusion_ready']}개")
        
        # File Selection for Detailed Analysis
        if summary['mirna_files']:
            st.markdown("---")
            st.markdown("**📂 분석할 파일 선택:**")
            selected_files = st.multiselect(
                "분석에 포함할 마이크로어레이 파일:",
                summary['mirna_files'],
                default=summary['mirna_files']
            )
            
            if selected_files:
                # Re-filter candidates based on selection
                filtered_candidates = summary['candidates'][summary['candidates']['Source_File'].isin([os.path.basename(f) for f in selected_files])]
                
                if not filtered_candidates.empty:
                    st.subheader("🌟 통합된 유력 후보 물질 (Top Candidates)")
                    st.dataframe(filtered_candidates.head(20))
                    
                    # Download Report
                    csv = filtered_candidates.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "후보 물질 리포트 다운로드 (CSV)",
                        csv,
                        "auto_analysis_candidates.csv",
                        "text/csv"
                    )
                else:
                    st.warning("선택된 파일에서 유의미한 후보를 찾지 못했습니다.")
        
        # Show Image Files
        if summary['image_files']:
            st.markdown("---")
            st.markdown("**📸 감지된 이미지 파일 목록:**")
            img_df = pd.DataFrame(summary['image_files'], columns=['File Path'])
            img_df['File Name'] = img_df['File Path'].apply(lambda x: os.path.basename(x))
            st.dataframe(img_df[['File Name', 'File Path']], height=150)

sidebar_info()
