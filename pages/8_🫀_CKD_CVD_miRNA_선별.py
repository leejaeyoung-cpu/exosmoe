import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# 경로 추가
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import set_page_config, load_css

# 페이지 설정
set_page_config("CKD-CVD miRNA 선별")
load_css()

# 제목
st.title("🫀 CKD-CVD 치료용 miRNA 선별 플랫폼")
st.markdown("### 만성 신장질환 & 심혈관질환 통합 치료를 위한 miRNA 후보 발굴")

# 사이드바
st.sidebar.header("⚙️ 설정")

# 탭 생성
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 후보 데이터", 
    "⚖️ 가중치 시스템", 
    "🔬 경로 분석", 
    "✅ 선별 결과",
    "📈 시각화"
])

# ========================================
# TAB 1: 후보 데이터 로드
# ========================================
with tab1:
    st.header("1. miRNA 후보 데이터 로드")
    
    # 파일 업로더
    uploaded_file = st.file_uploader(
        "CKD-CVD miRNA 후보 엑셀 파일 업로드 (99개 후보)",
        type=['xlsx', 'csv'],
        key='ckd_cvd_upload'
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df_candidates = pd.read_excel(uploaded_file)
            else:
                df_candidates = pd.read_csv(uploaded_file)
            
            st.success(f"✅ 데이터 로드 완료: {len(df_candidates)}개 miRNA")
            
            # 데이터 미리보기
            st.subheader("데이터 미리보기")
            st.dataframe(df_candidates.head(10), use_container_width=True)
            
            # 통계 요약
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("총 후보 수", len(df_candidates))
            col2.metric("평균 FC", f"{df_candidates['FC_MT_vs_Con'].mean():.2f}")
            col3.metric("최대 FC", f"{df_candidates['FC_MT_vs_Con'].max():.2f}")
            col4.metric("컬럼 수", len(df_candidates.columns))
            
            # Session state에 저장
            st.session_state['df_candidates'] = df_candidates
            
        except Exception as e:
            st.error(f"❌ 데이터 로드 실패: {e}")
    else:
        st.info("👆 엑셀 파일을 업로드하여 시작하세요. (예: CKD_CVD_exosome_miRNA_candidates.xlsx)")

# ========================================
# TAB 2: 가중치 시스템
# ========================================
with tab2:
    st.header("2. 치료 카테고리 가중치 설정")
    
    st.markdown("""
    **CKD-CVD 치료의 6개 핵심 카테고리:**
    - **염증 (Inflammation)**: 사이토카인 폭풍 차단
    - **섬유화 (Fibrosis)**: 장기 경화 방지
    - **항산화 (Antioxidant)**: 산화 스트레스 감소
    - **내피 기능 (Endothelial)**: 혈관 건강 회복
    - **CVD 보호**: 심혈관 합병증 예방
    - **노화/손상 반응**: 장기적 질환 관리
    """)
    
    st.subheader("가중치 조정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        w_inflam = st.slider("염증 (Inflammation)", 0.0, 0.5, 0.25, 0.05)
        w_fib = st.slider("섬유화 (Fibrosis)", 0.0, 0.5, 0.25, 0.05)
        w_anti = st.slider("항산화 (Antioxidant)", 0.0, 0.5, 0.20, 0.05)
    
    with col2:
        w_endo = st.slider("내피 기능 (Endothelial)", 0.0, 0.5, 0.20, 0.05)
        w_cvd = st.slider("CVD 보호", 0.0, 0.5, 0.10, 0.05)
        w_sen = st.slider("노화/손상", 0.0, 0.5, 0.05, 0.05)
    
    # 가중치 합계 확인
    total_weight = w_inflam + w_fib + w_anti + w_endo + w_cvd + w_sen
    
    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"⚠️ 가중치 합계: {total_weight:.2f} (1.0이 되어야 합니다)")
    else:
        st.success(f"✅ 가중치 합계: {total_weight:.2f}")
    
    # 가중치 시각화
    weights_df = pd.DataFrame({
        '카테고리': ['염증', '섬유화', '항산화', '내피기능', 'CVD', '노화'],
        '가중치': [w_inflam, w_fib, w_anti, w_endo, w_cvd, w_sen]
    })
    
    fig = px.bar(weights_df, x='카테고리', y='가중치', 
                 title='카테고리별 가중치 분포',
                 color='가중치', color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)
    
    # Session state에 저장
    st.session_state['weights'] = {
        'inflam': w_inflam,
        'fib': w_fib,
        'anti': w_anti,
        'endo': w_endo,
        'cvd': w_cvd,
        'sen': w_sen
    }

# ========================================
# TAB 3: 경로 분석
# ========================================
with tab3:
    st.header("3. KEGG 경로 분석")
    
    if 'df_candidates' in st.session_state:
        df = st.session_state['df_candidates']
        
        st.subheader("데이터 컬럼 확인")
        st.write("현재 데이터 컬럼:", df.columns.tolist())
        
        # 실제 컬럼명 매핑 (한글 컬럼명 사용)
        pathway_cols = {}
        col_mapping = {
            '항염증': '항염증·면역조절_Npath',
            '항산화': '항산화·미토콘드리아/대사 항상성_Npath',
            '내피': '혈관신생·내피기능/혈류반응_Npath',
            '섬유화': '신장 섬유화·ECM/EMT 억제_Npath',
            'CVD': '심혈관 합병증/죽상동맥경화·심근보호_Npath',
            '노화': '세포사멸·노화/손상 반응_Npath'
        }
        
        # 실제 존재하는 컬럼만 매핑
        for key, col in col_mapping.items():
            matching_cols = [c for c in df.columns if col.split('_')[0] in c and '_Npath' in c]
            if matching_cols:
                pathway_cols[key] = matching_cols[0]
        
        if pathway_cols:
            st.success(f"✅ {len(pathway_cols)}개 경로 컬럼 발견")
            
            # 경로 수 통계
            stats_data = []
            for category, col in pathway_cols.items():
                stats_data.append({
                    '카테고리': category,
                    '평균 경로 수': df[col].mean(),
                    '최대 경로 수': df[col].max(),
                    '총 경로 수 (상위 10)': df.nlargest(10, col)[col].sum()
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
            # 총 경로 수 계산
            if 'total_pathways' not in df.columns:
                df['total_pathways'] = df[list(pathway_cols.values())].sum(axis=1)
            
            # 상위 miRNA 경로 커버리지
            st.subheader("상위 10개 miRNA의 경로 커버리지")
            
            top10 = df.nlargest(10, 'total_pathways')[['miRNA', 'FC_MT_vs_Con'] + list(pathway_cols.values()) + ['total_pathways']]
            
            st.dataframe(top10, use_container_width=True)
            
            # 히트맵
            fig = go.Figure(data=go.Heatmap(
                z=top10[list(pathway_cols.values())].values,
                x=list(pathway_cols.keys()),
                y=top10['miRNA'],
                colorscale='YlGnBu'
            ))
            fig.update_layout(title='상위 10개 miRNA의 카테고리별 경로 분포',
                             xaxis_title='카테고리',
                             yaxis_title='miRNA')
            st.plotly_chart(fig, use_container_width=True)
            
            # Session state에 저장
            st.session_state['pathway_cols'] = pathway_cols
        else:
            st.error("❌ 경로 컬럼을 찾을 수 없습니다. 데이터 형식을 확인하세요.")
        
    else:
        st.warning("⚠️ 먼저 '후보 데이터' 탭에서 데이터를 로드하세요.")

# ========================================
# TAB 4: 선별 결과
# ========================================
with tab4:
    st.header("4. miRNA 선별 결과")
    
    if 'df_candidates' in st.session_state and 'pathway_cols' in st.session_state:
        df = st.session_state['df_candidates']
        pathway_cols = st.session_state['pathway_cols']
        weights = st.session_state.get('weights', {
            'inflam': 0.25, 'fib': 0.25, 'anti': 0.20, 
            'endo': 0.20, 'cvd': 0.10, 'sen': 0.05
        })
        
        st.subheader("선별 기준")
        
        col1, col2 = st.columns(2)
        with col1:
            min_fc = st.number_input("최소 Fold Change", min_value=1.0, value=30.0, step=5.0)
            min_total_pathways = st.number_input("최소 총 경로 수", min_value=10, value=50, step=10)
        
        with col2:
            top_n = st.number_input("상위 N개 선택", min_value=1, max_value=20, value=5, step=1)
        
        if st.button("🔍 miRNA 선별 실행", type="primary"):
            # 가중치 매핑 (한글 카테고리 → 가중치)
            weight_mapping = {
                '항염증': weights['inflam'],
                '섬유화': weights['fib'],
                '항산화': weights['anti'],
                '내피': weights['endo'],
                'CVD': weights['cvd'],
                '노화': weights['sen']
            }
            
            # 가중치 점수 계산
            df['weighted_score'] = 0
            for category, col in pathway_cols.items():
                if category in weight_mapping:
                    df['weighted_score'] += df[col] * weight_mapping[category]
            
            # total_pathways가 없으면 계산
            if 'total_pathways' not in df.columns:
                df['total_pathways'] = df[list(pathway_cols.values())].sum(axis=1)
            
            # 필터링
            filtered = df[
                (df['FC_MT_vs_Con'] >= min_fc) & 
                (df['total_pathways'] >= min_total_pathways)
            ].nlargest(top_n, 'weighted_score')
            
            st.success(f"✅ {len(filtered)}개 miRNA 선별 완료!")
            
            # 결과 표시
            result_cols = ['miRNA', 'FC_MT_vs_Con', 'total_pathways', 'weighted_score'] + list(pathway_cols.values())
            st.dataframe(filtered[result_cols], use_container_width=True)
            
            # Core-2 권장
            if len(filtered) >= 2:
                st.subheader("🎯 Core-2 칵테일 권장")
                
                # 전략: 가장 높은 점수 + 상호 보완적 경로
                core1 = filtered.iloc[0]
                
                # 상호 보완성 계산
                complementarity_scores = []
                for idx in range(1, len(filtered)):
                    candidate = filtered.iloc[idx]
                    comp_score = 0
                    for category, col in pathway_cols.items():
                        if category in weight_mapping:
                            if core1[col] < candidate[col]:
                                comp_score += (candidate[col] - core1[col]) * weight_mapping[category]
                    complementarity_scores.append(comp_score)
                
                core2_idx = np.argmax(complementarity_scores) + 1
                core2 = filtered.iloc[core2_idx]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**miRNA #1**: {core1['miRNA']}")
                    st.write(f"- FC: {core1['FC_MT_vs_Con']:.2f}")
                    st.write(f"- 총 경로: {int(core1['total_pathways'])}")
                    st.write(f"- 가중치 점수: {core1['weighted_score']:.2f}")
                
                with col2:
                    st.info(f"**miRNA #2**: {core2['miRNA']}")
                    st.write(f"- FC: {core2['FC_MT_vs_Con']:.2f}")
                    st.write(f"- 총 경로: {int(core2['total_pathways'])}")
                    st.write(f"- 가중치 점수: {core2['weighted_score']:.2f}")
                
                # Session state에 저장
                st.session_state['core2'] = {
                    'miRNA1': core1['miRNA'],
                    'miRNA2': core2['miRNA']
                }
                st.session_state['filtered_results'] = filtered
            
    else:
        st.warning("⚠️ 먼저 데이터를 로드하고 경로 분석을 완료하세요.")

# ========================================
# TAB 5: 시각화 및 저장
# ========================================
with tab5:
    st.header("5. 결과 시각화 및 저장")
    
    if 'df_candidates' in st.session_state:
        df = st.session_state['df_candidates']
        
        # 시각화 섹션
        st.subheader("📊 시각화")
        
        # FC vs Total Pathways 산점도
        fig_scatter = px.scatter(
            df, 
            x='FC_MT_vs_Con', 
            y='total_pathways',
            hover_data=['miRNA'],
            title='Fold Change vs 총 경로 수',
            labels={'FC_MT_vs_Con': 'Fold Change', 'total_pathways': '총 경로 수'},
            color='weighted_score' if 'weighted_score' in df.columns else None
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Core-2 강조
        if 'core2' in st.session_state:
            core2 = st.session_state['core2']
            st.success(f"🎯 Core-2 칵테일: **{core2['miRNA1']}** + **{core2['miRNA2']}**")
        
        # 저장 섹션
        st.markdown("---")
        st.subheader("💾 결과 저장")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📄 데이터 저장")
            
            # 1. 전체 후보 데이터 (CSV)
            if st.button("📥 전체 후보 데이터 다운로드 (CSV)"):
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="CSV 다운로드",
                    data=csv,
                    file_name="CKD_CVD_miRNA_candidates_analyzed.csv",
                    mime="text/csv"
                )
            
            # 2. 선별된 결과 (Excel)
            if 'filtered_results' in st.session_state:
                if st.button("📥 선별된 miRNA 결과 다운로드 (Excel)"):
                    from io import BytesIO
                    
                    filtered = st.session_state['filtered_results']
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        filtered.to_excel(writer, sheet_name='Selected_miRNAs', index=False)
                    
                    st.download_button(
                        label="Excel 다운로드",
                        data=buffer.getvalue(),
                        file_name="CKD_CVD_Selected_miRNAs.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        with col2:
            st.markdown("#### 📊 보고서 저장")
            
            # 3. Core-2 정보 (Markdown)
            if 'core2' in st.session_state and 'filtered_results' in st.session_state:
                if st.button("📥 분석 보고서 다운로드 (Markdown)"):
                    core2 = st.session_state['core2']
                    filtered = st.session_state['filtered_results']
                    weights = st.session_state.get('weights', {})
                    
                    # Markdown 보고서 생성
                    report = f"""# CKD-CVD miRNA 선별 분석 보고서

## 1. 분석 개요

- **분석 일시**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- **총 후보 수**: {len(df)}개
- **선별된 miRNA 수**: {len(filtered)}개

## 2. 가중치 설정

| 카테고리 | 가중치 |
|---------|--------|
| 염증 (Inflammation) | {weights.get('inflam', 0.25):.2f} |
| 섬유화 (Fibrosis) | {weights.get('fib', 0.25):.2f} |
| 항산화 (Antioxidant) | {weights.get('anti', 0.20):.2f} |
| 내피 기능 (Endothelial) | {weights.get('endo', 0.20):.2f} |
| CVD 보호 | {weights.get('cvd', 0.10):.2f} |
| 노화/손상 | {weights.get('sen', 0.05):.2f} |

## 3. Core-2 칵테일 최종 선정

### 🎯 miRNA #1: {core2['miRNA1']}

"""
                    # Core-2 상세 정보 추가
                    core1_info = filtered[filtered['miRNA'] == core2['miRNA1']].iloc[0]
                    core2_info = filtered[filtered['miRNA'] == core2['miRNA2']].iloc[0]
                    
                    report += f"""
- **Fold Change**: {core1_info['FC_MT_vs_Con']:.2f}
- **총 경로 수**: {int(core1_info['total_pathways'])}
- **가중치 점수**: {core1_info['weighted_score']:.2f}

### 🎯 miRNA #2: {core2['miRNA2']}

- **Fold Change**: {core2_info['FC_MT_vs_Con']:.2f}
- **총 경로 수**: {int(core2_info['total_pathways'])}
- **가중치 점수**: {core2_info['weighted_score']:.2f}

## 4. 선정 근거

Core-2 조합은 다음과 같은 시너지 효과를 제공합니다:

1. **상호 보완성**: 두 miRNA가 서로 다른 경로를 강화하여 치료 효과 극대화
2. **높은 발현**: 두 miRNA 모두 높은 Fold Change로 충분한 치료 농도 확보 가능
3. **광범위한 경로 커버리지**: CKD-CVD의 주요 병리 기전을 포괄적으로 타겟팅

## 5. 다음 단계

1. **in vitro 검증**: 신장 세포주 및 심근세포에서 효능 확인
2. **엑소좀 로딩**: Core-2 miRNA를 엑소좀에 효율적으로 탑재
3. **동물 실험**: CKD-CVD 마우스 모델에서 치료 효과 검증

---

**분석 플랫폼**: Mela-Exosome AI - CKD-CVD miRNA 선별 모듈
"""
                    
                    st.download_button(
                        label="Markdown 보고서 다운로드",
                        data=report,
                        file_name="CKD_CVD_Analysis_Report.md",
                        mime="text/markdown"
                    )
            
            # 4. 시각화 저장 안내
            st.info("💡 **시각화 저장 방법**: 각 그래프 우측 상단의 📷 아이콘을 클릭하여 이미지로 저장할 수 있습니다.")
    
    else:
        st.warning("⚠️ 먼저 데이터를 로드하세요.")

# Footer
st.markdown("---")
st.markdown("**CKD-CVD miRNA 선별 플랫폼** | Powered by Mela-Exosome AI")
