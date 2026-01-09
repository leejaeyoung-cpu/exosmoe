"""
MT-EXO miRNA 분석 페이지 (간소화 버전)
기능: 분석 결과 보기만 (분석 실행은 별도로)
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path

# 페이지 설정
st.set_page_config(page_title="MT-EXO 분석", layout="wide")

st.title("🧬 MT-EXO miRNA 분석")
st.markdown("### 멜라토닌 전처리 엑소좀의 miRNA 기능 분석")

# 경로
project_root = Path(__file__).parent.parent
RESULTS_PATH = project_root / "data" / "MT_EXO_Analysis_Results"

# 결과 파일 확인
sig_path = RESULTS_PATH / "MT_EXO_Significant_Candidates.csv"
summary_path = RESULTS_PATH / "Function_Summary.json"

if not sig_path.exists():
    st.warning("⚠️ 분석 결과가 없습니다. 터미널에서 다음 명령어를 실행하세요:")
    st.code("python src\\mirna_functional_analyzer.py")
    st.stop()

# 데이터 로드
try:
    df_sig = pd.read_csv(sig_path)
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    st.success(f"✅ 분석 결과 로드 완료: {len(df_sig)}개 miRNA")
    
except Exception as e:
    st.error(f"❌ 데이터 로드 실패: {e}")
    st.stop()

# 탭 구성
tab1, tab2 = st.tabs(["📊 전체 결과", "🎯 기능별 필터"])

# === 탭 1: 전체 결과 ===
with tab1:
    st.subheader("📋 분석 결과 요약")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 miRNA", f"{len(df_sig)}개")
    
    with col2:
        avg_fc = df_sig['MT-EXOSOME/Con-EXO.fc'].mean()
        st.metric("평균 Fold Change", f"{avg_fc:.2f}x")
    
    with col3:
        high_fc = (df_sig['MT-EXOSOME/Con-EXO.fc'] > 10).sum()
        st.metric("고배수 증가 (FC>10)", f"{high_fc}개")
    
    with col4:
        annotated = (df_sig['primary_function'] != '미분류 (신규 후보)').sum()
        st.metric("기능 주석됨", f"{annotated}개")
    
    # 전체 테이블
    st.subheader("📊 전체 miRNA 목록")
    
    # 표시할 컬럼 선택 (존재하는 것만)
    display_cols = []
    possible_cols = [
        'miRNA', 
        'MT-EXOSOME/Con-EXO.fc', 
        'Log2FC',
        'primary_function',
        'Candidate_Score'
    ]
    
    for col in possible_cols:
        if col in df_sig.columns:
            display_cols.append(col)
    
    st.dataframe(
        df_sig[display_cols].sort_values('Candidate_Score', ascending=False),
        use_container_width=True,
        height=500
    )

# === 탭 2: 기능별 필터 ===
with tab2:
    st.subheader("🎯 기능별 Top 후보")
    
    # 기능 선택
    func_names = ["항산화", "항섬유화", "항염증", "혈관형성", "세포증식"]
    func_map = {
        "항산화": "antioxidant",
        "항섬유화": "anti_fibrotic",
        "항염증": "anti_inflammatory",
        "혈관형성": "angiogenic",
        "세포증식": "proliferation"
    }
    
    selected_func = st.selectbox("기능 선택", func_names)
    
    func_key = func_map[selected_func]
    score_col = f"{func_key}_score"
    
    if score_col in df_sig.columns:
        # 해당 기능 점수가 있는 것만 필터링
        filtered = df_sig[df_sig[score_col] > 0].sort_values(score_col, ascending=False)
        
        st.info(f"💡 {selected_func} 기능을 가진 miRNA: {len(filtered)}개")
        
        if len(filtered) > 0:
            # 컬럼 준비
            show_cols = ['miRNA', 'MT-EXOSOME/Con-EXO.fc', 'Log2FC', score_col, 'Candidate_Score']
            show_cols = [c for c in show_cols if c in filtered.columns]
            
            st.dataframe(
                filtered[show_cols].head(20),
                use_container_width=True,
                height=400
            )
        else:
            st.warning(f"⚠️ {selected_func} 기능을 가진 miRNA가 없습니다.")
    else:
        st.error(f"❌ {score_col} 컬럼이 없습니다.")

# 사이드바 - 기능별 요약
with st.sidebar:
    st.header("📈 기능별 요약")
    
    for name in func_names:
        if name in summary:
            count = summary[name]['total_count']
            if count > 0:
                st.metric(name, f"{count}개")

# 도움말
with st.expander("💡 사용 방법"):
    st.markdown("""
    ### 분석 결과 보는 법
    
    1. **전체 결과 탭**
       - 모든 miRNA를 Candidate Score 순으로 표시
       - 상단 메트릭에서 전체 통계 확인
    
    2. **기능별 필터 탭**  
       - 특정 기능(항산화, 항염증 등)을 가진 miRNA만 필터링
       - 해당 기능의 점수가 높은 순으로 정렬
    
    3. **사이드바**
       - 각 기능별로 몇 개의 miRNA가 있는지 요약
    
    ### 새로 분석하려면
    
    터미널에서 실행:
    ```
    python src\\mirna_functional_analyzer.py
    ```
    """)
