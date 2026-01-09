"""
NOVA De Novo Designer v2.0
Learning-based Molecule Generation with IND Gate Constraints
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import sys

# Add parent
sys.path.insert(0, str(Path(__file__).parent))

# RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    RDKIT_AVAILABLE = True
except:
    RDKIT_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="NOVA De Novo v2.0",
    page_icon="🧬",
    layout="wide"
)

# Header
st.markdown("# 🧬 NOVA De Novo Designer v2.0")
st.markdown("### Learning from Candidate 1 + IND Gate Constraints")
st.markdown("---")

# Sidebar: Constraints
st.sidebar.markdown("## ⚙️ Generation Constraints")

st.sidebar.markdown("### 🎯 Critical Constraints")
egfr_min = st.sidebar.slider("EGFR Selectivity (min)", 1.0, 50.0, 10.0, 1.0)
st.sidebar.caption(f"Target: ≥{egfr_min}x (vs ALK5)")

synthesis_min = st.sidebar.slider("Synthesis Score (min)", 50, 100, 70, 5)
st.sidebar.caption("HATU coupling feasibility")

lipinski_filter = st.sidebar.checkbox("Lipinski Rule", value=True)

st.sidebar.markdown("### 📊 Generation Settings")
n_molecules = st.sidebar.slider("Number of Molecules", 10, 100, 50, 10)

# Session state
if 'v2_molecules' not in st.session_state:
    st.session_state.v2_molecules = None
if 'efficacy_results' not in st.session_state:
    st.session_state.efficacy_results = None

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ Candidate 1 학습",
    "2️⃣ 2세대 생성",
    "3️⃣ 비교 분석",
    "4️⃣ Top 10 선정",
    "5️⃣ 치료 효과 AI 예측"
])

# Tab 1: Candidate 1 Review
with tab1:
    st.markdown("## 📚 Candidate 1 분석 및 학습")
    
    col_ref1, col_ref2 = st.columns([1, 1])
    
    with col_ref1:
        st.markdown("### ✅ 성공 요인")
        
        st.success("""
        **Dual Pathway Inhibitor:**
        - ALK5: 82 nM
        - TAK1: 145 nM
        - IKKβ: 99 nM
        
        **기능:**
        - TGF-β/SMAD 억제 (섬유화)
        - NF-κB 억제 (염증)
        - CKD 컨셉 최적
        """)
        
        st.markdown("**합성 Feasibility:**")
        st.success("""
        - Route: HATU coupling
        - Yield: 60-80%
        - Scale-up: CRO-friendly
        - Cost: ~$3K/g
        """)
    
    with col_ref2:
        st.markdown("### ⚠️ 개선 필요")
        
        st.error("""
        **EGFR Off-Target Risk:**
        - Quinazoline → EGFR 억제 가능성 높음
        - Selectivity < 10x 우려
        - **이것이 프로젝트의 가장 큰 리스크!**
        """)
        
        st.markdown("**개선 전략:**")
        st.warning("""
        1. Quinazoline 6-position 변형 (6-F, 6-Cl)
        2. Scaffold hopping (Pyrimidine, Pyrazolo-pyrimidine)
        3. Benzoyl 최적화 (F vs Cl)
        4. EGFR selectivity를 생성 시 explicit constraint로!
        """)
    
    # Structure visualization
    st.markdown("---")
    st.markdown("### 🧪 Candidate 1 Structure")
    
    reference_smiles = "COc1ccc(C(=O)Nc2ncnc3ccccc23)cc1Cl"
    
    if RDKIT_AVAILABLE:
        mol = Chem.MolFromSmiles(reference_smiles)
        if mol:
            AllChem.Compute2DCoords(mol)
            img = Draw.MolToImage(mol, size=(600, 400))
            col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
            with col_img2:
                st.image(img)
                st.caption(f"**SMILES:** {reference_smiles}")
    
    # Key learnings
    st.markdown("---")
    st.markdown("### 💡 핵심 학습 내용")
    
    col_learn1, col_learn2, col_learn3 = st.columns(3)
    
    with col_learn1:
        st.info("""
        **Scaffold:**
        - Quinazoline core → Kinase hinge binding
        - Amide linker → 안정성
        - Benzoyl cap → Selectivity tuning
        """)
    
    with col_learn2:
        st.info("""
        **Property Optimization:**
        - MW: ~314 Da (ideal)
        - LogP: ~3.5 (good)
        - TPSA: ~70 Å² (drug-like)
        - Lipinski: ✅ PASS
        """)
    
    with col_learn3:
        st.info("""
        **Next Generation Target:**
        - EGFR selectivity ≥ 15x
        - ALK5/TAK1 유지
        - Synthesis score ≥ 70
        - IND pass prob ≥ 80%
        """)

# Tab 2: Generation
with tab2:
    st.markdown("## 🚀 2세대 분자 생성")
    
    st.info(f"""
    **현재 Constraint 설정:**
    - EGFR Selectivity: ≥ {egfr_min}x
    - Synthesis Score: ≥ {synthesis_min}
    - Lipinski Rule: {'ON' if lipinski_filter else 'OFF'}
    - Target Molecules: {n_molecules}
    """)
   
    if st.button("🚀 2세대 분자 생성", key="gen_v2"):
        with st.spinner("Generating improved molecules..."):
            try:
                from nova_generator_v2 import NOVAGeneratorV2
                
                generator = NOVAGeneratorV2(seed=42)
                molecules_v2 = generator.generate_with_constraints(
                    egfr_min=egfr_min,
                    synthesis_min=synthesis_min,
                    lipinski=lipinski_filter,
                    n_molecules=n_molecules
                )
                
                st.session_state.v2_molecules = molecules_v2
                
                # Save
                output_path = Path("generated_molecules/nova_v2_candidates.csv")
                molecules_v2.to_csv(output_path, index=False)
                
                st.success(f"✅ {len(molecules_v2)}개 분자 생성 완료!")
                
                # Quick stats
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("평균 EGFR Selectivity", f"{molecules_v2['egfr_selectivity_pred'].mean():.1f}x")
                col_s2.metric("평균 Synthesis Score", f"{molecules_v2['synthesis_score'].mean():.0f}")
                col_s3.metric("평균 IND Pass Prob", f"{molecules_v2['ind_pass_prob'].mean():.0%}")
                col_s4.metric("Lipinski Pass", f"{molecules_v2['lipinski_pass'].sum()}/{len(molecules_v2)}")
            
            except Exception as e:
                st.error(f"생성 오류: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    if st.session_state.v2_molecules is not None:
        df_v2 = st.session_state.v2_molecules
        
        st.markdown("---")
        st.markdown("### 📊 생성 결과")
        
        # Scaffold distribution
        col_dist1, col_dist2 = st.columns(2)
        
        with col_dist1:
            fig_scaffold = px.pie(df_v2, names='scaffold', title='Scaffold Distribution')
            st.plotly_chart(fig_scaffold, use_container_width=True)
        
        with col_dist2:
            fig_egfr = px.histogram(df_v2, x='egfr_selectivity_pred', nbins=20,
                                   title='EGFR Selectivity Distribution',
                                   labels={'egfr_selectivity_pred': 'EGFR Selectivity (x)'})
            fig_egfr.add_vline(x=10, line_dash="dash", annotation_text="10x cutoff")
            st.plotly_chart(fig_egfr, use_container_width=True)
        
        # Property scatter
        st.markdown("---")
        st.markdown("### 🎯 특성 분석")
        
        col_scat1, col_scat2 = st.columns(2)
        
        with col_scat1:
            fig_ind = px.scatter(df_v2, x='egfr_selectivity_pred', y='ind_pass_prob',
                                color='scaffold', size='synthesis_score',
                                title='EGFR Selectivity vs IND Pass Probability',
                                labels={'egfr_selectivity_pred': 'EGFR Selectivity (x)',
                                       'ind_pass_prob': 'IND Pass Prob'})
            fig_ind.add_vline(x=10, line_dash="dash")
            fig_ind.add_hline(y=0.8, line_dash="dash")
            st.plotly_chart(fig_ind, use_container_width=True)
        
        with col_scat2:
            fig_prop = px.scatter(df_v2, x='logp', y='mw',
                                 color='lipinski_pass',
                                 title='Property Space (MW vs LogP)',
                                 labels={'logp': 'LogP', 'mw': 'MW (Da)'},
                                 color_discrete_map={True: '#27ae60', False: '#e74c3c'})
            fig_prop.add_hline(y=500, line_dash="dash", annotation_text="MW 500")
            fig_prop.add_vline(x=5, line_dash="dash", annotation_text="LogP 5")
            st.plotly_chart(fig_prop, use_container_width=True)
        
        # Table
        st.markdown("---")
        st.markdown("### 📋 전체 후보 리스트")
        st.dataframe(
            df_v2[['id', 'scaffold', 'egfr_selectivity_pred', 'synthesis_score',
                  'ind_pass_prob', 'mw', 'logp', 'lipinski_pass']],
            use_container_width=True
        )

# Tab 3: Comparison
with tab3:
    st.markdown("## 📊 1세대 vs 2세대 비교")
    
    if st.session_state.v2_molecules is None:
        st.warning("먼저 Tab 2에서 2세대 분자를 생성하세요.")
    else:
        df_v2 = st.session_state.v2_molecules
        
        # Create comparison data
        comparison_data = {
            'Metric': [
                'EGFR Selectivity',
                'Synthesis Score',
                'IND Pass Prob',
                'Lipinski Pass Rate'
            ],
            'Candidate 1 (v1)': [
                '< 10x (예상)',
                '80',
                '50%',
                '100%'
            ],
            'Average v2': [
                f"{df_v2['egfr_selectivity_pred'].mean():.1f}x",
                f"{df_v2['synthesis_score'].mean():.0f}",
                f"{df_v2['ind_pass_prob'].mean():.0%}",
                f"{df_v2['lipinski_pass'].mean():.0%}"
            ]
        }
        
        st.markdown("### 📈 핵심 메트릭 비교")
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        # Improvement visualization
        st.markdown("---")
        st.markdown("### 🎯 개선 사항")
        
        col_imp1, col_imp2, col_imp3 = st.columns(3)
        
        avg_egfr = df_v2['egfr_selectivity_pred'].mean()
        egfr_improvement = ((avg_egfr - 7) / 7) * 100  # Assume v1 = 7x
        
        col_imp1.metric(
            "EGFR Selectivity",
            f"{avg_egfr:.1f}x",
            delta=f"+{egfr_improvement:.0f}% vs v1"
        )
        
        avg_ind = df_v2['ind_pass_prob'].mean()
        ind_improvement = ((avg_ind - 0.5) / 0.5) * 100
        
        col_imp2.metric(
            "IND Pass Prob",
            f"{avg_ind:.0%}",
            delta=f"+{ind_improvement:.0f}% vs v1"
        )
        
        high_conf = (df_v2['egfr_selectivity_pred'] >= 15).sum()
        
        col_imp3.metric(
            "High Confidence (EGFR ≥15x)",
            f"{high_conf}",
            delta=f"{high_conf/len(df_v2):.0%} of total"
        )
        
        # Risk level comparison
        st.markdown("---")
        st.markdown("### ⚠️ Risk Level Analysis")
        
        # Assign risk levels
        def assign_risk(row):
            if row['egfr_selectivity_pred'] >= 15 and row['ind_pass_prob'] >= 0.8:
                return "LOW"
            elif row['egfr_selectivity_pred'] >= 10 and row['ind_pass_prob'] >= 0.6:
                return "MEDIUM"
            else:
                return "HIGH"
        
        df_v2['risk_level'] = df_v2.apply(assign_risk, axis=1)
        
        fig_risk = px.pie(df_v2, names='risk_level', title='Risk Level Distribution (v2)',
                         color='risk_level',
                         color_discrete_map={
                             'LOW': '#27ae60',
                             'MEDIUM': '#f39c12',
                             'HIGH': '#e74c3c'
                         })
        st.plotly_chart(fig_risk, use_container_width=True)

# Tab 4: Top 10
with tab4:
    st.markdown("## 🏆 Top 10 후보 선정")
    
    if st.session_state.v2_molecules is None:
        st.warning("먼저 Tab 2에서 2세대 분자를 생성하세요.")
    else:
        df_v2 = st.session_state.v2_molecules
        
        # Composite score
        df_v2['composite_score'] = (
            df_v2['egfr_selectivity_pred'] / 50 * 30 +  # 30%
            df_v2['synthesis_score'] / 100 * 20 +  # 20%
            df_v2['ind_pass_prob'] * 50  # 50%
        )
        
        top10 = df_v2.nlargest(10, 'composite_score')
        
        st.markdown("### 📊 Top 10 Candidates")
        st.dataframe(
            top10[['id', 'scaffold', 'egfr_selectivity_pred', 'synthesis_score',
                  'ind_pass_prob', 'composite_score', 'smiles']],
            use_container_width=True
        )
        
        # Detailed view
        st.markdown("---")
        st.markdown("### 🔍 상세 분석")
        
        selected_id = st.selectbox("후보 선택", top10['id'].tolist())
        
        if selected_id:
            selected = top10[top10['id'] == selected_id].iloc[0]
            
            col_det1, col_det2 = st.columns([1, 1])
            
            with col_det1:
                st.markdown(f"#### {selected_id}")
                st.markdown(f"**Scaffold:** {selected['scaffold']}")
                st.markdown(f"**SMILES:** `{selected['smiles']}`")
                
                # Properties
                st.markdown("**Properties:**")
                st.markdown(f"- MW: {selected['mw']:.1f} Da")
                st.markdown(f"- LogP: {selected['logp']:.2f}")
                st.markdown(f"- TPSA: {selected['tpsa']:.1f} Å²")
                st.markdown(f"- Lipinski: {'✅ PASS' if selected['lipinski_pass'] else '❌ FAIL'}")
                
                # Scores
                st.markdown("**Scores:**")
                st.metric("EGFR Selectivity", f"{selected['egfr_selectivity_pred']:.1f}x")
                st.metric("Synthesis Score", f"{selected['synthesis_score']:.0f}")
                st.metric("IND Pass Prob", f"{selected['ind_pass_prob']:.0%}")
                st.metric("Composite Score", f"{selected['composite_score']:.1f}")
            
            with col_det2:
                # Structure
                if RDKIT_AVAILABLE:
                    mol = Chem.MolFromSmiles(selected['smiles'])
                    if mol:
                        AllChem.Compute2DCoords(mol)
                        img = Draw.MolToImage(mol, size=(500, 400))
                        st.image(img)
                        st.caption("2D Structure")
                
                # Rationale
                st.markdown("**Design Rationale:**")
                st.info(selected['rationale'])
        
        # Download
        st.markdown("---")
        csv_output = top10.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Top 10 다운로드 (CSV)",
            csv_output,
            file_name="nova_v2_top10.csv",
            mime="text/csv"
        )

# Tab 5: Therapeutic Efficacy Prediction
with tab5:
    st.markdown("## 🏥 CKD 치료 효과 AI 예측")
    
    st.info("""
    **AI 기반 UUO 모델 시뮬레이션:**
    - 섬유화 감소 (Fibrosis reduction)
    - 염증 억제 (Inflammation suppression)
    - 신장 기능 개선 (Renal function improvement)
    - 14일 Time-course
    - Dose-response curve
    
    **Literature-based PK/PD modeling**
    """)
    
    if st.session_state.v2_molecules is None:
        st.warning("먼저 Tab 2에서 2세대 분자를 생성하세요.")
    else:
        df_v2 = st.session_state.v2_molecules
        
        # Select candidate
        st.markdown("### 🔍 후보 선택")
        
        # Top 10 for selection
        df_v2['composite_score'] = (
            df_v2['egfr_selectivity_pred'] / 50 * 30 +
            df_v2['synthesis_score'] / 100 * 20 +
            df_v2['ind_pass_prob'] * 50
        )
        top10 = df_v2.nlargest(10, 'composite_score')
        
        selected_id = st.selectbox("치료 효과를 예측할 후보 선택", top10['id'].tolist(), key="efficacy_select")
        
        # Simulation parameters
        col_param1, col_param2 = st.columns(2)
        
        with col_param1:
            dose_mg_kg = st.slider("투여 용량 (mg/kg)", 1, 100, 30, 1)
            duration_days = st.slider("투여 기간 (days)", 7, 21, 14, 1)
        
        with col_param2:
            st.markdown("**가정된 Kinase IC50:**")
            st.markdown("- ALK5: 82 nM (Candidate 1 기준)")
            st.markdown("- TAK1: 145 nM")
            st.markdown("- IKKβ: 99 nM")
            st.caption("실제 값은 실험으로 확인 필요")
        
        if st.button("🏥 치료 효과 예측 실행", key="run_efficacy"):
            with st.spinner("Simulating therapeutic effects..."):
                try:
                    from nova_efficacy_predictor import CKDEfficacyPredictor
                    
                    predictor = CKDEfficacyPredictor()
                    
                    # Assume kinase IC50 (would be from actual data)
                    alk5_ic50 = 82
                    tak1_ic50 = 145
                    ikkb_ic50 = 99
                    
                    # Full prediction
                    efficacy = predictor.predict_full_efficacy(
                        alk5_ic50_nM=alk5_ic50,
                        tak1_ic50_nM=tak1_ic50,
                        ikkb_ic50_nM=ikkb_ic50,
                        dose_mg_kg=dose_mg_kg,
                        duration_days=duration_days
                    )
                    
                    st.session_state.efficacy_results = efficacy
                    
                    st.success("✅ 치료 효과 예측 완료!")
                
                except Exception as e:
                    st.error(f"예측 오류: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        if st.session_state.efficacy_results is not None:
            efficacy = st.session_state.efficacy_results
            
            st.markdown("---")
            st.markdown("### 📊 예측 결과")
            
            # Overall metrics
            col_eff1, col_eff2, col_eff3, col_eff4 = st.columns(4)
            
            col_eff1.metric(
                "Overall Efficacy Score",
                f"{efficacy['efficacy_score']:.1f}/100"
            )
            
            col_eff2.metric(
                "Fibrosis Reduction",
                f"{efficacy['fibrosis']['fibrosis_area_reduction']:.1f}%"
            )
            
            col_eff3.metric(
                "Inflammation Reduction",
                f"{efficacy['inflammation']['inflammation_reduction']:.1f}%"
            )
            
            col_eff4.metric(
                "Success (≥30%)",
                "✅ YES" if efficacy['success'] else "❌ NO"
            )
            
            # Detailed results
            st.markdown("---")
            st.markdown("### 🔬 상세 결과")
            
            col_det1, col_det2 = st.columns(2)
            
            with col_det1:
                st.markdown("#### 섬유화 (Fibrosis)")
                
                fib = efficacy['fibrosis']
                
                fig_fib = go.Figure()
                fig_fib.add_trace(go.Bar(
                    x=['Fibrosis Area', 'COL1A1', 'FN1', 'α-SMA'],
                    y=[
                        fib['fibrosis_area_reduction'],
                        fib['col1a1_reduction'],
                        fib['fn1_reduction'],
                        fib['acta2_reduction']
                    ],
                    marker_color=['#3498db', '#2ecc71', '#27ae60', '#16a085']
                ))
                fig_fib.update_layout(
                    title='섬유화 마커 감소 (%)',
                    yaxis_title='Reduction (%)',
                    yaxis_range=[0, 100]
                )
                fig_fib.add_hline(y=30, line_dash="dash", annotation_text="Pirfenidone 수준")
                st.plotly_chart(fig_fib, use_container_width=True)
                
                # Metrics
                st.metric("ALK5 Occupancy", f"{fib['occupancy']:.1f}%")
                st.metric("Cmax (estimated)", f"{fib['cmax_nM']:.0f} nM")
            
            with col_det2:
                st.markdown("#### 염증 (Inflammation)")
                
                inf = efficacy['inflammation']
                
                fig_inf = go.Figure()
                fig_inf.add_trace(go.Bar(
                    x=['Overall', 'CCL2', 'IL-6', 'ICAM1', 'F4/80'],
                    y=[
                        inf['inflammation_reduction'],
                        inf['ccl2_reduction'],
                        inf['il6_reduction'],
                        inf['icam1_reduction'],
                        inf['f480_reduction']
                    ],
                    marker_color=['#e74c3c', '#c0392b', '#e67e22', '#d35400', '#c0392b']
                ))
                fig_inf.update_layout(
                    title='염증 마커 감소 (%)',
                    yaxis_title='Reduction (%)',
                    yaxis_range=[0, 100]
                )
                st.plotly_chart(fig_inf, use_container_width=True)
                
                # Occupancy
                st.metric("TAK1 Occupancy", f"{inf['tak1_occupancy']:.1f}%")
                st.metric("IKKβ Occupancy", f"{inf['ikkb_occupancy']:.1f}%")
            
            # Renal function
            st.markdown("---")
            st.markdown("### 🩺 신장 기능 (Renal Function)")
            
            col_ren1, col_ren2 = st.columns(2)
            
            func = efficacy['renal_function']
            
            with col_ren1:
                # Creatinine
                fig_cr = go.Figure()
                
                fig_cr.add_trace(go.Bar(
                    x=['Sham', 'Vehicle (UUO)', f'Treated ({dose_mg_kg} mg/kg)'],
                    y=[0.5, func['baseline_creatinine'], func['final_creatinine']],
                    marker_color=['#2ecc71', '#e74c3c', '#3498db']
                ))
                
                fig_cr.update_layout(
                    title='Serum Creatinine (mg/dL)',
                    yaxis_title='Creatinine (mg/dL)',
                    yaxis_range=[0, 2.5]
                )
                
                st.plotly_chart(fig_cr, use_container_width=True)
                
                st.metric(
                    "Creatinine Improvement",
                    f"{func['creatinine_improvement']:.2f} mg/dL",
                    delta=f"-{func['creatinine_reduction_percent']:.1f}%"
                )
            
            with col_ren2:
                # BUN
                fig_bun = go.Figure()
                
                fig_bun.add_trace(go.Bar(
                    x=['Sham', 'Vehicle (UUO)', f'Treated ({dose_mg_kg} mg/kg)'],
                    y=[25, func['baseline_bun'], func['final_bun']],
                    marker_color=['#2ecc71', '#e74c3c', '#3498db']
                ))
                
                fig_bun.update_layout(
                    title='Blood Urea Nitrogen (mg/dL)',
                    yaxis_title='BUN (mg/dL)',
                    yaxis_range=[0, 100]
                )
                
                st.plotly_chart(fig_bun, use_container_width=True)
                
                st.metric(
                    "BUN Improvement",
                    f"{func['bun_improvement']:.1f} mg/dL"
                )
            
            # Time-course
            st.markdown("---")
            st.markdown("### 📈 Time-Course Analysis")
            
            tc = efficacy['time_course']
            
            fig_tc = go.Figure()
            
            fig_tc.add_trace(go.Scatter(
                x=tc['day'],
                y=tc['fibrosis_reduction'],
                mode='lines+markers',
                name='Fibrosis Reduction',
                line=dict(color='#3498db', width=3)
            ))
            
            fig_tc.add_trace(go.Scatter(
                x=tc['day'],
                y=tc['inflammation_reduction'],
                mode='lines+markers',
                name='Inflammation Reduction',
                line=dict(color='#e74c3c', width=3)
            ))
            
            fig_tc.add_hline(y=30, line_dash="dash", annotation_text="Success threshold (30%)")
            
            fig_tc.update_layout(
                title=f'14-Day Time-Course @ {dose_mg_kg} mg/kg',
                xaxis_title='Day',
                yaxis_title='Reduction (%)',
                yaxis_range=[0, 100]
            )
            
            st.plotly_chart(fig_tc, use_container_width=True)
            
            # Gene expression time-course
            st.markdown("---")
            st.markdown("### 🧬 유전자 발현 Time-Course")
            
            fig_gene = go.Figure()
            
            fig_gene.add_trace(go.Scatter(
                x=tc['day'],
                y=tc['col1a1_reduction'],
                mode='lines+markers',
                name='COL1A1 (fibrosis)',
                line=dict(color='#2ecc71', width=2)
            ))
            
            fig_gene.add_trace(go.Scatter(
                x=tc['day'],
                y=tc['ccl2_reduction'],
                mode='lines+markers',
                name='CCL2 (inflammation)',
                line=dict(color='#e67e22', width=2)
            ))
            
            fig_gene.add_trace(go.Scatter(
                x=tc['day'],
                y=tc['acta2_reduction'],
                mode='lines+markers',
                name='α-SMA (myofibroblast)',
                line=dict(color='#9b59b6', width=2)
            ))
            
            fig_gene.update_layout(
                title='주요 유전자 발현 변화',
                xaxis_title='Day',
                yaxis_title='Reduction from baseline (%)',
                yaxis_range=[0, 100]
            )
            
            st.plotly_chart(fig_gene, use_container_width=True)
            
            # Comparison with Pirfenidone
            st.markdown("---")
            st.markdown("### 🎯 Pirfenidone 대비 성능")
            
            col_comp1, col_comp2, col_comp3 = st.columns(3)
            
            pirfenidone_fibrosis = 30  # Literature standard
            nova_fibrosis = efficacy['fibrosis']['fibrosis_area_reduction']
            
            fold_improvement = nova_fibrosis / pirfenidone_fibrosis
            
            col_comp1.metric(
                "Pirfenidone (표준)",
                f"{pirfenidone_fibrosis}%",
                delta="Benchmark"
            )
            
            col_comp2.metric(
                f"NOVA Candidate @ {dose_mg_kg} mg/kg",
                f"{nova_fibrosis:.1f}%",
                delta=f"{nova_fibrosis - pirfenidone_fibrosis:+.1f}%"
            )
            
            col_comp3.metric(
                "Fold Improvement",
                f"{fold_improvement:.2f}x",
                delta="vs Pirfenidone"
            )
            
            # Interpretation
            if fold_improvement >= 1.5:
                st.success(f"""
                ✅ **Outstanding Performance!**
                - {fold_improvement:.1f}배 superior to Pirfenidone
                - Strong clinical potential
                - Proceed to IND-enabling studies
                """)
            elif fold_improvement >= 1.2:
                st.success(f"""
                ✅ **Good Performance!**
                - {fold_improvement:.1f}배 better than Pirfenidi
                - Clinically meaningful improvement
                - Recommend in vivo validation
                """)
            elif fold_improvement >= 1.0:
                st.warning("""
                ⚠️ **Comparable Performance**
                - Similar to Pirfenidone
                - Dual pathway may offer advantages
                - Consider dose optimization
                """)
            else:
                st.error("""
                ❌ **Below Standard**
                - Weaker than Pirfenidone
                - Formulation or dosing issue?
                - Re-evaluate or try next candidate
                """)
            
            # Download
            st.markdown("---")
            
            # Create summary report
            summary_data = {
                'Parameter': [
                    'Dose (mg/kg)',
                    'Duration (days)',
                    'Fibrosis Reduction (%)',
                    'Inflammation Reduction (%)',
                    'Creatinine Improvement (mg/dL)',
                    'BUN Improvement (mg/dL)',
                    'Efficacy Score',
                    'Success (≥30%)',
                    'vs Pirfenidone (fold)'
                ],
                'Value': [
                    dose_mg_kg,
                    duration_days,
                    f"{efficacy['fibrosis']['fibrosis_area_reduction']:.1f}",
                    f"{efficacy['inflammation']['inflammation_reduction']:.1f}",
                    f"{func['creatinine_improvement']:.2f}",
                    f"{func['bun_improvement']:.1f}",
                    f"{efficacy['efficacy_score']:.1f}",
                    "YES" if efficacy['success'] else "NO",
                    f"{fold_improvement:.2f}"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            
            csv_eff = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 치료 효과 예측 결과 다운로드 (CSV)",
                csv_eff,
                file_name=f"efficacy_prediction_{selected_id}.csv",
                mime="text/csv"
            )


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🧬 NOVA De Novo Designer v2.0</p>
    <p>Learning-based Molecule Generation with IND Gate Constraints</p>
</div>
""", unsafe_allow_html=True)
