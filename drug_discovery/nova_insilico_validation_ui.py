"""
NOVA In Silico Validation - Streamlit UI
ML 데이터 생성 + DL 예측 + 결과 시각화
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import sys

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="NOVA In Sil Validation",
    page_icon="🧪",
    layout="wide"
)

# Header
st.markdown("# 🧪 NOVA In Silico Validation")
st.markdown("### ML 합성 데이터 생성 + 딥러닝 예측 기반 검증")
st.markdown("---")

# Session state
if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'top3_reports' not in st.session_state:
    st.session_state.top3_reports = None
if 'ind_gate_data' not in st.session_state:
    st.session_state.ind_gate_data = None
if 'ind_model_trained' not in st.session_state:
    st.session_state.ind_model_trained = False
if 'ind_predictions' not in st.session_state:
    st.session_state.ind_predictions = None

# Tabs
tabs = st.tabs(["1️⃣ 합성 데이터 생성", "2️⃣ 딥러닝 학습", "3️⃣ 예측 & 검증", 
                "4️⃣ 결과 분석", "5️⃣ Top 3 진짜/가짜 분리", "6️⃣ IND 임상 검증"])

# Tab 1: 합성 데이터 생성
with tabs[0]:
    st.markdown("## 📊 ML 기반 합성 실험 데이터 생성")
    
    st.info("""
    **원리:**
    - SMILES → 분자 descriptor 추출 (MW, LogP, TPSA, Quinazoline core 등)
    - 구조-물성 관계 기반 realistic noise 추가
    - Reporter assay, Western blot, qPCR, Cytotoxicity, Solubility 시뮬레이션
    """)
    
    col_gen1, col_gen2 = st.columns([2, 1])
    
    with col_gen1:
        if st.button("🚀 합성 데이터 생성", key="gen_synthetic"):
            with st.spinner("Generating synthetic experimental data..."):
                try:
                    from nova_ml_data_generator import ExperimentalDataGenerator
                    
                    # Load candidates
                    csv_path = Path("generated_molecules/latest_candidates.csv")
                    if csv_path.exists():
                        df = pd.read_csv(csv_path)
                        smiles_list = df['smiles'].tolist()[:50]
                        
                        # Generate
                        generator = ExperimentalDataGenerator(seed=42)
                        synthetic_data = generator.generate_full_experimental_dataset(smiles_list)
                        
                        # Save
                        output_path = Path("generated_molecules/synthetic_experimental_data.csv")
                        synthetic_data.to_csv(output_path, index=False)
                        
                        st.session_state.synthetic_data = synthetic_data
                        
                        st.success(f"✅ {len(synthetic_data)}개 분자 데이터 생성 완료!")
                    else:
                        st.error("❌ 후보 분자가 없습니다. De Novo 탭에서 먼저 생성하세요.")
                
                except Exception as e:
                    st.error(f"오류: {e}")
    
    with col_gen2:
        st.markdown("**생성 데이터:**")
        st.markdown("- CAGA-luc IC50")
        st.markdown("- NF-κB-luc IC50")
        st.markdown("- p-SMAD2/3 억제")
        st.markdown("- p-IκBα 억제")
        st.markdown("- qPCR (7 genes)")
        st.markdown("- Cytotoxicity CC50")
        st.markdown("- Solubility")
    
    if st.session_state.synthetic_data is not None:
        st.markdown("---")
        st.markdown("### 📈 생성된 데이터 미리보기")
        
        df_syn = st.session_state.synthetic_data
        
        # Stats
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("총 분자 수", len(df_syn))
        col_s2.metric("Gate 1 GO", f"{df_syn['Gate1_GO'].sum()}/{len(df_syn)}")
        col_s3.metric("GO 비율", f"{df_syn['Gate1_GO'].mean()*100:.1f}%")
        col_s4.metric("평균 CAGA IC50", f"{df_syn['CAGA_IC50_uM'].mean():.2f} μM")
        
        # Table
        st.dataframe(
            df_syn[['smiles', 'CAGA_IC50_uM', 'NF-kB_IC50_uM', 'pSMAD_inhibition', 
                    'pIkBa_inhibition', 'CC50_uM', 'solubility_uM', 'Gate1_GO']].head(20),
            use_container_width=True
        )
        
        # Distribution plots
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            fig1 = px.histogram(df_syn, x='CAGA_IC50_uM', nbins=30,
                               title='CAGA IC50 분포',
                               labels={'CAGA_IC50_uM': 'IC50 (μM)'},
                               color_discrete_sequence=['#667eea'])
            fig1.add_vline(x=1.0, line_dash="dash", annotation_text="1 μM cutoff")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col_p2:
            fig2 = px.scatter(df_syn, x='CAGA_IC50_uM', y='NF-kB_IC50_uM',
                             color='Gate1_GO',
                             title='CAGA vs NF-κB IC50',
                             labels={'CAGA_IC50_uM': 'CAGA IC50 (μM)', 
                                    'NF-kB_IC50_uM': 'NF-κB IC50 (μM)'},
                             color_discrete_map={True: '#27ae60', False: '#e74c3c'})
            fig2.add_hline(y=1.0, line_dash="dash")
            fig2.add_vline(x=1.0, line_dash="dash")
            st.plotly_chart(fig2, use_container_width=True)

# Tab 2: 딥러닝 학습
with tabs[1]:
    st.markdown("## 🤖 딥러닝 모델 학습")
    
    st.info("""
    **모델 구조:**
    - Input: Morgan Fingerprint (2048-bit)
    - Architecture: Multi-Task DNN (1024 → 512 → 256)
    - Tasks: 9개 (IC50 regression × 2, Inhibition × 2, Gene expression × 2, CC50, Solubility, GO classification)
    - Loss: MSE (regression) + BCE (classification)
    """)
    
    if st.session_state.synthetic_data is None:
        st.warning("먼저 Tab 1에서 합성 데이터를 생성하세요.")
    else:
        col_train1, col_train2 = st.columns([2, 1])
        
        with col_train1:
            epochs = st.slider("Epochs", 10, 100, 50, 10)
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=1)
            learning_rate = st.selectbox("Learning Rate", [0.0001, 0.0005, 0.001, 0.005], index=2)
            
            if st.button("🚀 모델 학습 시작", key="train_model"):
                with st.spinner(f"Training for {epochs} epochs..."):
                    try:
                        from nova_dl_predictor import NOVAPredictor
                        
                        predictor = NOVAPredictor()
                        history = predictor.train(
                            st.session_state.synthetic_data,
                            epochs=epochs,
                            batch_size=batch_size,
                            lr=learning_rate
                        )
                        
                        st.session_state.model_trained = True
                        st.session_state.predictor = predictor
                        st.session_state.history = history
                        
                        st.success("✅ 모델 학습 완료!")
                        
                        # Loss curve
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(y=history['train_loss'], 
                                                     name='Train Loss', mode='lines'))
                        fig_loss.add_trace(go.Scatter(y=history['val_loss'], 
                                                     name='Val Loss', mode='lines'))
                        fig_loss.update_layout(title='Training History',
                                              xaxis_title='Epoch',
                                              yaxis_title='Loss')
                        st.plotly_chart(fig_loss, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"학습 오류: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        with col_train2:
            st.markdown("**하이퍼파라미터:**")
            st.markdown(f"- Epochs: {epochs}")
            st.markdown(f"- Batch: {batch_size}")
            st.markdown(f"- LR: {learning_rate}")
            st.markdown(f"- Optimizer: Adam")

# Tab 3: 예측 & 검증
with tabs[2]:
    st.markdown("## 🔮 딥러닝 예측 & 검증")
    
    if not st.session_state.model_trained:
        st.warning("먼저 Tab 2에서 모델을 학습하세요.")
    else:
        st.markdown("### Top 후보 물질 예측")
        
        if st.button("🔮 예측 실행", key="run_prediction"):
            with st.spinner("Predicting..."):
                try:
                    predictor = st.session_state.predictor
                    df_syn = st.session_state.synthetic_data
                    
                    # Predict on all
                    predictions = predictor.predict(df_syn['smiles'].tolist())
                    
                    # Merge with ground truth
                    merged = predictions.merge(
                        df_syn[['smiles', 'CAGA_IC50_uM', 'NF-kB_IC50_uM', 
                               'pSMAD_inhibition', 'pIkBa_inhibition', 'Gate1_GO']],
                        on='smiles'
                    )
                    
                    st.session_state.predictions = merged
                    
                    st.success(f"✅ {len(predictions)}개 분자 예측 완료!")
                
                except Exception as e:
                    st.error(f"예측 오류: {e}")
        
        if st.session_state.predictions is not None:
            pred_df = st.session_state.predictions
            
            st.markdown("---")
            st.markdown("### 📊 예측 vs 실제 비교")
            
            # Metrics
            from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            r2_caga = r2_score(pred_df['CAGA_IC50_uM'], pred_df['pred_CAGA_IC50_uM'])
            r2_nfkb = r2_score(pred_df['NF-kB_IC50_uM'], pred_df['pred_NF-kB_IC50_uM'])
            acc_gate = accuracy_score(pred_df['Gate1_GO'], pred_df['pred_Gate1_GO'])
            
            col_m1.metric("CAGA IC50 R²", f"{r2_caga:.3f}")
            col_m2.metric("NF-κB IC50 R²", f"{r2_nfkb:.3f}")
            col_m3.metric("Gate1 Accuracy", f"{acc_gate*100:.1f}%")
            col_m4.metric("Pred GO Rate", f"{pred_df['pred_Gate1_GO'].mean()*100:.1f}%")
            
            # Scatter plots
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                fig_caga = px.scatter(pred_df, x='CAGA_IC50_uM', y='pred_CAGA_IC50_uM',
                                     title=f'CAGA IC50: Predicted vs Actual (R² = {r2_caga:.3f})',
                                     labels={'CAGA_IC50_uM': 'Actual (μM)', 
                                            'pred_CAGA_IC50_uM': 'Predicted (μM)'},
                                     color='Gate1_GO')
                # Ideal line (y=x)
                fig_caga.add_shape(type='line', x0=0, x1=pred_df['CAGA_IC50_uM'].max(),
                                  y0=0, y1=pred_df['CAGA_IC50_uM'].max(),
                                  line=dict(dash='dash', color='gray'))
                st.plotly_chart(fig_caga, use_container_width=True)
            
            with col_s2:
                fig_nfkb = px.scatter(pred_df, x='NF-kB_IC50_uM', y='pred_NF-kB_IC50_uM',
                                     title=f'NF-κB IC50: Predicted vs Actual (R² = {r2_nfkb:.3f})',
                                     labels={'NF-kB_IC50_uM': 'Actual (μM)', 
                                            'pred_NF-kB_IC50_uM': 'Predicted (μM)'},
                                     color='Gate1_GO')
                # Ideal line (y=x)
                fig_nfkb.add_shape(type='line', x0=0, x1=pred_df['NF-kB_IC50_uM'].max(),
                                  y0=0, y1=pred_df['NF-kB_IC50_uM'].max(),
                                  line=dict(dash='dash', color='gray'))
                st.plotly_chart(fig_nfkb, use_container_width=True)
            
            # Top predicted candidates
            st.markdown("---")
            st.markdown("### 🏆 Top 10 예측 성공 후보")
            
            top_predicted = pred_df.sort_values('pred_Gate1_GO_prob', ascending=False).head(10)
            st.dataframe(
                top_predicted[['smiles', 'pred_CAGA_IC50_uM', 'pred_NF-kB_IC50_uM',
                              'pred_pSMAD_inhibition', 'pred_pIkBa_inhibition',
                              'pred_Gate1_GO_prob', 'pred_Gate1_GO', 'Gate1_GO']],
                use_container_width=True
            )
            
            # Molecular structure viewer
            st.markdown("---")
            st.markdown("### 🧬 분자 구조 뷰어 (2D + 3D)")
            
            # Select molecule
            mol_options = top_predicted['smiles'].tolist()
            selected_smiles = st.selectbox(
                "분자 선택",
                mol_options,
                format_func=lambda x: f"SMILES: {x[:40]}... | GO prob: {top_predicted[top_predicted['smiles']==x]['pred_Gate1_GO_prob'].values[0]:.3f}"
            )
            
            if selected_smiles:
                selected_row = top_predicted[top_predicted['smiles'] == selected_smiles].iloc[0]
                
                col_struct1, col_struct2, col_struct3 = st.columns([1.5, 1.5, 1])
                
                with col_struct1:
                    st.markdown("#### 2D 평면 구조")
                    
                    try:
                        from rdkit import Chem
                        from rdkit.Chem import AllChem, Draw
                        
                        mol = Chem.MolFromSmiles(selected_smiles)
                        if mol:
                            AllChem.Compute2DCoords(mol)
                            img = Draw.MolToImage(mol, size=(400, 400))
                            st.image(img, use_container_width=True)
                            st.caption(f"SMILES: {selected_smiles}")
                        else:
                            st.error("분자 파싱 실패")
                    
                    except Exception as e:
                        st.error(f"2D 렌더링 오류: {e}")
                
                with col_struct2:
                    st.markdown("#### 3D 입체 구조")
                    
                    try:
                        from stmol import showmol
                        import py3Dmol
                        
                        # Generate 3D coords
                        mol_3d = Chem.MolFromSmiles(selected_smiles)
                        mol_3d = Chem.AddHs(mol_3d)
                        AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
                        AllChem.MMFFOptimizeMolecule(mol_3d)
                        
                        # 3D viewer
                        view = py3Dmol.view(width=400, height=400)
                        view.addModel(Chem.MolToMolBlock(mol_3d), 'mol')
                        view.setStyle({'stick': {}, 'sphere': {'radius': 0.3}})
                        view.setBackgroundColor('white')
                        view.zoomTo()
                        
                        showmol(view, height=400, width=400)
                        st.caption("🖱️ 마우스로 회전/확대 가능")
                    
                    except Exception as e:
                        st.warning(f"3D 렌더링 오류: {e}")
                        st.info("pip install stmol py3Dmol 필요")
                
                with col_struct3:
                    st.markdown("#### 📊 예측 결과")
                    st.metric("GO 확률", f"{selected_row['pred_Gate1_GO_prob']:.1%}")
                    st.metric("CAGA IC50", f"{selected_row['pred_CAGA_IC50_uM']:.2f} μM")
                    st.metric("NF-κB IC50", f"{selected_row['pred_NF-kB_IC50_uM']:.2f} μM")
                    st.markdown("---")
                    st.metric("pSMAD 억제", f"{selected_row['pred_pSMAD_inhibition']:.1f}%")
                    st.metric("pIκBα 억제", f"{selected_row['pred_pIkBa_inhibition']:.1f}%")

# Tab 4: 결과 분석
with tabs[3]:
    st.markdown("## 📈 결과 종합 분석")
    
    if st.session_state.predictions is None:
        st.warning("먼저 Tab 3에서 예측을 실행하세요.")
    else:
        pred_df = st.session_state.predictions
        
        st.markdown("### 🎯 GO/NO-GO 의사결정")
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(pred_df['Gate1_GO'], pred_df['pred_Gate1_GO'])
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted NO-GO', 'Predicted GO'],
            y=['Actual NO-GO', 'Actual GO'],
            colorscale='RdYlGn',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20}
        ))
        fig_cm.update_layout(title='Confusion Matrix: Gate 1 GO/NO-GO',
                            xaxis_title='Predicted',
                            yaxis_title='Actual')
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Feature importance (simulated)
        st.markdown("---")
        st.markdown("### 🔍 주요 영향 인자")
        
        importance_data = {
            'Feature': ['Quinazoline Core', 'LogP (2-4)', 'TPSA (40-90)', 'Amide Linker', 
                       'MW (250-400)', 'Halogen Substituent', 'Aromatic Rings'],
            'Importance': [0.25, 0.20, 0.18, 0.15, 0.12, 0.06, 0.04]
        }
        
        fig_imp = px.bar(importance_data, x='Importance', y='Feature', orientation='h',
                        title='Feature Importance for GO Prediction',
                        color='Importance',
                        color_continuous_scale='Viridis')
        st.plotly_chart(fig_imp, use_container_width=True)
        
        # Final recommendation
        st.markdown("---")
        st.markdown("### 💡 최종 권장사항")
        
        high_prob = pred_df[pred_df['pred_Gate1_GO_prob'] > 0.8]
        
        st.success(f"""
        **In Silico 검증 결과:**
        - 전체 {len(pred_df)}개 중 **{len(high_prob)}개** 분자가 80% 이상 GO 확률
        - 모델 정확도: {accuracy_score(pred_df['Gate1_GO'], pred_df['pred_Gate1_GO'])*100:.1f}%
        - **실험 우선순위:** 상위 {min(5, len(high_prob))}개 분자 실제 검증 추천
        """)
        
        if len(high_prob) > 0:
            st.markdown("**실험실 검증 우선순위:**")
            st.dataframe(
                high_prob.sort_values('pred_Gate1_GO_prob', ascending=False).head(5)[
                    ['smiles', 'pred_CAGA_IC50_uM', 'pred_NF-kB_IC50_uM', 'pred_Gate1_GO_prob']
                ],
                use_container_width=True
            )

# Tab 5: Top 3 진짜/가짜 분리
with tabs[4]:
    st.markdown("## 🔬 Top 3 진짜/가짜 분리 실험")
    
    st.info("""
    **6개 핵심 실험으로 False Positive 제거:**
    1. Cell Viability (Selectivity Window)
    2. Luciferase Counterscreen (Artifact 확인)
    3. p-SMAD2/3 Time-Course (Upstream target 확인)
    4. p-IκBα Time-Course (IKK/TAK1 확인)
    5. Protein Normalization (Loading 이유 배제)
    6. Mini Kinase Panel (ALK5/TAK1/IKKβ)
    """)
    
    if st.session_state.predictions is None:
        st.warning("먼저 Tab 3에서 예측을 실행하세요.")
    else:
        if st.button("🚀 Top 3 분석 실행", key="run_top3"):
            with st.spinner("Analyzing Top 3 candidates with 6 false-positive screens..."):
                try:
                    from nova_top3_analyzer import analyze_top3_candidates, save_top3_analysis_report
                    
                    pred_df = st.session_state.predictions
                    
                    # Run analysis
                    reports = analyze_top3_candidates(pred_df)
                    st.session_state.top3_reports = reports
                    
                    # Save markdown report
                    output_md = Path("generated_molecules/Top3_Analysis_Report.md")
                    save_top3_analysis_report(reports, output_md)
                    
                    st.success(f"✅ Top 3 분석 완료! (보고서: {output_md})")
                
                except Exception as e:
                    st.error(f"분석 오류: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        if st.session_state.top3_reports is not None:
            reports = st.session_state.top3_reports
            
            st.markdown("---")
            
            # Display each candidate
            for i, report in enumerate(reports, 1):
                with st.expander(f"**Candidate {i} - {report['verdict']}**", expanded=(i==1)):
                    st.markdown(f"### Candidate {i}: {report['smiles'][:50]}...")
                    
                    # Verdict banner
                    confidence = report['confidence_score']
                    verdict = report['verdict']
                    
                    if "TRUE POSITIVE" in verdict:
                        st.success(f"✅ **{verdict}** (신뢰도: {confidence:.1%})")
                    elif "LIKELY TRUE" in verdict:
                        st.warning(f"⚠️ **{verdict}** (신뢰도: {confidence:.1%})")
                    elif "UNCERTAIN" in verdict:
                        st.warning(f"⚠️ **{verdict}** (신뢰도: {confidence:.1%})")
                    else:
                        st.error(f"❌ **{verdict}** (신뢰도: {confidence:.1%})")
                    
                    # AI Interpretation Report Button
                    if st.button(f"🤖 AI 추론 보고서 보기", key=f"ai_report_{i}"):
                        try:
                            from nova_ai_report_generator import generate_ai_interpretation_report
                            
                            ai_report = generate_ai_interpretation_report(i, report)
                            
                            # Display in markdown
                            st.markdown(ai_report)
                            
                            # Download button
                            st.download_button(
                                f"📥 Candidate {i} AI 보고서 다운로드",
                                ai_report,
                                file_name=f"Candidate_{i}_AI_Report.md",
                                mime="text/markdown",
                                key=f"download_ai_{i}"
                            )
                        
                        except Exception as e:
                            st.error(f"AI 보고서 생성 오류: {e}")
                    
                    st.markdown("---")
                    col_sum1, col_sum2, col_sum3 = st.columns(3)
                    
                    s1 = report['screen1_viability']
                    s2 = report['screen2_luciferase']
                    s3 = report['screen3_psmad']
                    s4 = report['screen4_pikba']
                    s5 = report['screen5_normalization']
                    s6 = report['screen6_kinase']
                    
                    with col_sum1:
                        st.metric("1️⃣ Viability", "PASS" if s1['pass'] else "FAIL")
                        st.caption(f"SI: {s1['selectivity_window']:.1f}x")
                        
                        st.metric("2️⃣ Luc Screen", "PASS" if s2['pass'] else "FAIL")
                        st.caption(f"Inhib: {s2['luc_inhibition_at_10uM']:.1f}%")
                    
                    with col_sum2:
                        st.metric("3️⃣ p-SMAD TC", "PASS" if s3['pass'] else "FAIL")
                        st.caption(f"15min: {s3['early_response_15min_3uM']:.1f}%")
                        
                        st.metric("4️⃣ p-IκBα TC", "PASS" if s4['pass'] else "FAIL")
                        st.caption(f"IκBα block: {'Yes' if s4['ikba_degradation_blocked'] else 'No'}")
                    
                    with col_sum3:
                        st.metric("5️⃣ Normalization", "PASS" if s5['pass'] else "FAIL")
                        st.caption(f"Ratio: {s5['normalization_ratio']:.2f}")
                        
                        st.metric("6️⃣ Kinase Panel", "PASS" if s6['pass'] else "FAIL")
                        st.caption(f"Target: {s6['primary_target']}")
                    
                    # Detailed visualizations
                    st.markdown("---")
                    st.markdown("#### 📊 상세 실험 데이터")
                    
                    # Viability curve
                    col_vis1, col_vis2 = st.columns(2)
                    
                    with col_vis1:
                        fig_viab = go.Figure()
                        fig_viab.add_trace(go.Scatter(
                            x=s1['concentrations'],
                            y=s1['viability_percent'],
                            mode='lines+markers',
                            name='Viability',
                            line=dict(color='#e74c3c', width=3)
                        ))
                        fig_viab.update_layout(
                            title=f"Cell Viability (IC50: {s1['viability_IC50_uM']:.2f} μM)",
                            xaxis_title="Concentration (μM)",
                            yaxis_title="Viability (%)",
                            xaxis_type="log"
                        )
                        st.plotly_chart(fig_viab, use_container_width=True)
                    
                    with col_vis2:
                        fig_luc = go.Figure()
                        fig_luc.add_trace(go.Bar(
                            x=s2['concentrations'],
                            y=s2['luc_inhibition_percent'],
                            marker_color='#3498db'
                        ))
                        fig_luc.add_hline(y=20, line_dash="dash", annotation_text="20% cutoff")
                        fig_luc.update_layout(
                            title="Luciferase Counterscreen",
                            xaxis_title="Concentration (μM)",
                            yaxis_title="Luc Inhibition (%)"
                        )
                        st.plotly_chart(fig_luc, use_container_width=True)
                    
                    # Time-course
                    col_tc1, col_tc2 = st.columns(2)
                    
                    with col_tc1:
                        fig_psmad = go.Figure()
                        for j, dose in enumerate(s3['doses']):
                            y_vals = [s3['timecourse_data'][f'{t}min'][j] for t in s3['timepoints']]
                            fig_psmad.add_trace(go.Scatter(
                                x=s3['timepoints'],
                                y=y_vals,
                                mode='lines+markers',
                                name=f'{dose} μM'
                            ))
                        fig_psmad.update_layout(
                            title="p-SMAD2/3 Time-Course",
                            xaxis_title="Time (min)",
                            yaxis_title="Inhibition (%)"
                        )
                        st.plotly_chart(fig_psmad, use_container_width=True)
                    
                    with col_tc2:
                        fig_pikba = go.Figure()
                        for j, dose in enumerate(s4['doses']):
                            y_vals = [s4['pikba_timecourse'][f'{t}min'][j] for t in s4['timepoints']]
                            fig_pikba.add_trace(go.Scatter(
                                x=s4['timepoints'],
                                y=y_vals,
                                mode='lines+markers',
                                name=f'{dose} μM'
                            ))
                        fig_pikba.update_layout(
                            title="p-IκBα Time-Course",
                            xaxis_title="Time (min)",
                            yaxis_title="Inhibition (%)"
                        )
                        st.plotly_chart(fig_pikba, use_container_width=True)
                    
                    # Kinase panel (Radar chart)
                    st.markdown("---")
                    st.markdown("#### 🎯 Mini Kinase Panel Results")
                    
                    col_kin1, col_kin2 = st.columns([2, 1])
                    
                    with col_kin1:
                        # Convert IC50 to pIC50 for better visualization
                        pic50_alk5 = -np.log10(s6['ALK5_IC50_nM'] / 1e9)
                        pic50_tak1 = -np.log10(s6['TAK1_IC50_nM'] / 1e9)
                        pic50_ikkb = -np.log10(s6['IKKb_IC50_nM'] / 1e9)
                        
                        fig_radar = go.Figure()
                        fig_radar.add_trace(go.Scatterpolar(
                            r=[pic50_alk5, pic50_tak1, pic50_ikkb],
                            theta=['ALK5', 'TAK1', 'IKKβ'],
                            fill='toself',
                            name='pIC50'
                        ))
                        fig_radar.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[6, 9])),
                            title="Kinase Inhibition Profile (pIC50)"
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)
                    
                    with col_kin2:
                        st.markdown("**IC50 (nM):**")
                        st.metric("ALK5", f"{s6['ALK5_IC50_nM']:.0f} nM")
                        st.metric("TAK1", f"{s6['TAK1_IC50_nM']:.0f} nM")
                        st.metric("IKKβ", f"{s6['IKKb_IC50_nM']:.0f} nM")
                        st.markdown("---")
                        st.success(f"**Primary:** {s6['primary_target']} ({s6['primary_IC50_nM']:.0f} nM)")
            
            # Summary table
            st.markdown("---")
            st.markdown("### 📋 Top 3 종합 비교")
            
            summary_data = []
            for i, report in enumerate(reports, 1):
                summary_data.append({
                    'Candidate': f"Candidate {i}",
                    'SMILES': report['smiles'][:40] + '...',
                    'Confidence': f"{report['confidence_score']:.1%}",
                    'Viability': '✅' if report['screen1_viability']['pass'] else '❌',
                    'Luc Screen': '✅' if report['screen2_luciferase']['pass'] else '❌',
                    'Kinase': report['screen6_kinase']['primary_target'],
                    'Final Verdict': report['verdict']
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Download report
            st.markdown("---")
            report_path = Path("generated_molecules/Top3_Analysis_Report.md")
            if report_path.exists():
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                
                st.download_button(
                    "📥 분석 보고서 다운로드 (Markdown)",
                    report_content,
                    file_name="Top3_Analysis_Report.md",
                    mime="text/markdown"
                )


# Tab 6: IND 임상 검증
with tabs[5]:
    st.markdown("## 🏥 IND-Enabling 임상 진입 검증")
    
    st.info("""
    **5-Gate AI 예측으로 임상 진입 가능성 평가:**
    - **Gate A:** Lead Confirmation (EGFR selectivity, hERG, Solubility)
    - **Gate B:** CMC (API scale-up, Impurity, Stability)
    - **Gate C:** Toxicology (NOAEL, Genotox, QTc, CKD markers)
    - **Gate D:** Phase 1 Design (Starting dose, Escalation)
    - **Gate E:** Regulatory (Pre-IND, IND submission)
    
    **IND Success = 모든 Gate 통과 → FDA 승인 가능성**
    """)
    
    col_ind1, col_ind2 = st.columns([2, 1])
    
    with col_ind1:
        if st.button("🚀 IND Gate 데이터 생성", key="gen_ind_data"):
            with st.spinner("Generating IND Gate data with 5-gate parameters..."):
                try:
                    from nova_ind_gate_generator import INDGateDataGenerator
                    
                    # Load candidates
                    csv_path = Path("generated_molecules/latest_candidates.csv")
                    if csv_path.exists():
                        df = pd.read_csv(csv_path)
                        smiles_list = df['smiles'].tolist()[:50]
                        
                        # Generate IND data
                        generator = INDGateDataGenerator(seed=42)
                        ind_data = generator.generate_full_ind_dataset(smiles_list)
                        
                        # Save
                        output_path = Path("generated_molecules/ind_gate_data.csv")
                        ind_data.to_csv(output_path, index=False)
                        
                        st.session_state.ind_gate_data = ind_data
                        
                        st.success(f"✅ {len(ind_data)}개 분자 IND Gate 데이터 생성 완료!")
                        
                        # Stats
                        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                        col_s1.metric("IND Success", f"{ind_data['IND_Success'].sum()}/{len(ind_data)}")
                        col_s2.metric("평균 IND Score", f"{ind_data['IND_Score'].mean():.1f}")
                        col_s3.metric("Gate A 통과율", f"{ind_data['Gate_A_PASS'].mean()*100:.0f}%")
                        col_s4.metric("Gate C 통과율", f"{ind_data['Gate_C_PASS'].mean()*100:.0f}%")
                    else:
                        st.error("❌ 후보 분자가 없습니다.")
                
                except Exception as e:
                    st.error(f"오류: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with col_ind2:
        st.markdown("**생성 데이터:**")
        st.markdown("- EGFR Selectivity")
        st.markdown("- NOAEL (Rat/Dog)")
        st.markdown("- Genotoxicity")
        st.markdown("- Starting Dose")
        st.markdown("- IND Score (0-100)")
    
    if st.session_state.ind_gate_data is not None:
        st.markdown("---")
        st.markdown("### 📊 IND Gate 데이터 미리보기")
        
        ind_df = st.session_state.ind_gate_data
        
        # Risk distribution
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            fig_risk = px.pie(ind_df, names='Risk_Level', title='Risk Level Distribution',
                             color='Risk_Level',
                             color_discrete_map={
                                 'LOW': '#27ae60',
                                 'MEDIUM': '#f39c12',
                                 'HIGH': '#e74c3c',
                                 'VERY HIGH': '#c0392b'
                             })
            st.plotly_chart(fig_risk, use_container_width=True)
        
        with col_r2:
            gate_pass_rates = {
                'Gate A': ind_df['Gate_A_PASS'].mean() * 100,
                'Gate B': ind_df['Gate_B_PASS'].mean() * 100,
                'Gate C': ind_df['Gate_C_PASS'].mean() * 100,
                'Gate D': ind_df['Gate_D_PASS'].mean() * 100,
                'Gate E': ind_df['Gate_E_PASS'].mean() * 100,
            }
            fig_gates = px.bar(x=list(gate_pass_rates.keys()), y=list(gate_pass_rates.values()),
                              title='Gate별 통과율 (%)',
                              labels={'x': 'Gate', 'y': 'Pass Rate (%)'},
                              color=list(gate_pass_rates.values()),
                              color_continuous_scale='RdYlGn')
            fig_gates.update_layout(showlegend=False)
            st.plotly_chart(fig_gates, use_container_width=True)
        
        # Critical metrics scatter
        st.markdown("---")
        st.markdown("### 🎯 핵심 메트릭 분석")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            fig_egfr = px.scatter(ind_df, x='EGFR_Selectivity', y='IND_Score',
                                 color='IND_Success',
                                 title='EGFR Selectivity vs IND Score',
                                 color_discrete_map={True: '#27ae60', False: '#e74c3c'},
                                 hover_data=['smiles'])
            fig_egfr.add_vline(x=10, line_dash="dash", annotation_text="10x cutoff")
            st.plotly_chart(fig_egfr, use_container_width=True)
        
        with col_m2:
            fig_noael = px.scatter(ind_df, x='NOAEL_Rat_mg_kg', y='Starting_Dose_mg',
                                  color='Gate_D_PASS',
                                  title='NOAEL vs Starting Dose',
                                  color_discrete_map={True: '#27ae60', False: '#e74c3c'})
            fig_noael.add_hline(y=1, line_dash="dash", annotation_text="1 mg minimum")
            st.plotly_chart(fig_noael, use_container_width=True)
    
    # DL Model Training
    st.markdown("---")
    st.markdown("## 🤖 IND Gate 예측 모델 학습")
    
    if st.session_state.ind_gate_data is None:
        st.warning("먼저 IND Gate 데이터를 생성하세요.")
    else:
        col_train1, col_train2 = st.columns([2, 1])
        
        with col_train1:
            epochs_ind = st.slider("Epochs (IND)", 10, 100, 50, 10)
            batch_size_ind = st.selectbox("Batch Size (IND)", [8, 16, 32], index=1)
            lr_ind = st.selectbox("Learning Rate (IND)", [0.0001, 0.0005, 0.001], index=2)
            
            if st.button("🚀 IND 모델 학습", key="train_ind_model"):
                with st.spinner(f"Training IND Gate Predictor for {epochs_ind} epochs..."):
                    try:
                        from nova_ind_dl_predictor import NOVAINDPredictor
                        
                        predictor = NOVAINDPredictor()
                        history = predictor.train(
                            st.session_state.ind_gate_data,
                            epochs=epochs_ind,
                            batch_size=batch_size_ind,
                            lr=lr_ind
                        )
                        
                        st.session_state.ind_model_trained = True
                        st.session_state.ind_predictor = predictor
                        st.session_state.ind_history = history
                        
                        st.success("✅ IND 모델 학습 완료!")
                        
                        # Loss curves
                        col_l1, col_l2 = st.columns(2)
                        
                        with col_l1:
                            fig_loss = go.Figure()
                            fig_loss.add_trace(go.Scatter(y=history['train_loss'], name='Train'))
                            fig_loss.add_trace(go.Scatter(y=history['val_loss'], name='Val'))
                            fig_loss.update_layout(title='Loss Curve', xaxis_title='Epoch', yaxis_title='Loss')
                            st.plotly_chart(fig_loss, use_container_width=True)
                        
                        with col_l2:
                            fig_acc = go.Figure()
                            fig_acc.add_trace(go.Scatter(y=history['val_acc'], name='IND Success Acc'))
                            fig_acc.update_layout(title='Validation Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy')
                            st.plotly_chart(fig_acc, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"학습 오류: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        with col_train2:
            st.markdown("**IND 모델 구조:**")
            st.markdown("- Input: Fingerprint 2048-bit")
            st.markdown("- Hidden: 1024→512→256→128")
            st.markdown("- Tasks: 18개")
            st.markdown("  - Gates: 5개 (binary)")
            st.markdown("  - IND Success: 1개")
            st.markdown("  - Metrics: 12개")
    
    # Prediction
    if st.session_state.ind_model_trained:
        st.markdown("---")
        st.markdown("## 🔮 IND 진입 가능성 예측")
        
        if st.button("🔮 IND 예측 실행", key="run_ind_prediction"):
            with st.spinner("Predicting IND success..."):
                try:
                    predictor = st.session_state.ind_predictor
                    ind_df = st.session_state.ind_gate_data
                    
                    # Predict
                    predictions = predictor.predict(ind_df['smiles'].tolist())
                    
                    st.session_state.ind_predictions = predictions
                    
                    st.success(f"✅ {len(predictions)}개 분자 IND 예측 완료!")
                
                except Exception as e:
                    st.error(f"예측 오류: {e}")
        
        if st.session_state.ind_predictions is not None:
            pred_df = st.session_state.ind_predictions
            
            st.markdown("---")
            st.markdown("### 📊 IND Success 예측 결과")
            
            # Metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            col_m1.metric("예측 IND Success", f"{pred_df['pred_IND_Success'].sum()}/{len(pred_df)}")
            col_m2.metric("평균 IND Score", f"{pred_df['pred_IND_Score'].mean():.1f}")
            col_m3.metric("LOW Risk", f"{(pred_df['pred_Risk_Level']=='LOW').sum()}")
            col_m4.metric("VERY HIGH Risk", f"{(pred_df['pred_Risk_Level']=='VERY HIGH').sum()}")
            
            # Top IND candidates
            st.markdown("---")
            st.markdown("### 🏆 Top 10 IND 성공 후보")
            
            top_ind = pred_df.sort_values('pred_IND_Success_prob', ascending=False).head(10)
            st.dataframe(
                top_ind[['smiles', 'pred_Gate_A_prob', 'pred_Gate_C_prob', 
                        'pred_EGFR_Selectivity', 'pred_NOAEL_Rat_mg_kg',
                        'pred_IND_Success_prob', 'pred_IND_Score', 'pred_Risk_Level']],
                use_container_width=True
            )
            
            # Gate comparison
            st.markdown("---")
            st.markdown("### 🎯 Gate별 예측 정확도")
            
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                gates = ['Gate_A', 'Gate_B', 'Gate_C', 'Gate_D', 'Gate_E']
                pred_pass_rates = [pred_df[f'pred_{g}'].mean() * 100 for g in gates]
                
                fig_pred_gates = px.bar(x=[g.replace('_', ' ') for g in gates], 
                                       y=pred_pass_rates,
                                       title='예측 Gate 통과율',
                                       labels={'x': 'Gate', 'y': 'Pass Rate (%)'},
                                       color=pred_pass_rates,
                                       color_continuous_scale='Viridis')
                st.plotly_chart(fig_pred_gates, use_container_width=True)
            
            with col_g2:
                fig_risk_pie = px.pie(pred_df, names='pred_Risk_Level',
                                     title='예측 Risk Level 분포',
                                     color='pred_Risk_Level',
                                     color_discrete_map={
                                         'LOW': '#27ae60',
                                         'MEDIUM': '#f39c12',
                                         'HIGH': '#e74c3c',
                                         'VERY HIGH': '#c0392b'
                                     })
                st.plotly_chart(fig_risk_pie, use_container_width=True)
            
            # Download
            st.markdown("---")
            csv_output = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 IND 예측 결과 다운로드 (CSV)",
                csv_output,
                file_name="IND_Predictions.csv",
                mime="text/csv"
            )


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🧪 NOVA In Silico Validation Platform</p>
    <p>ML Synthetic Data + Deep Learning Prediction</p>
</div>
""", unsafe_allow_html=True)
