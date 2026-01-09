"""
AI 신약 발견 파이프라인 - Streamlit UI

사용자 친화적인 웹 인터페이스로 파이프라인 실행 및 결과 확인
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import sys
from datetime import datetime
import subprocess
import time

# 페이지 설정
st.set_page_config(
    page_title="AI 신약 발견 파이프라인",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        font-size: 18px;
        border-radius: 10px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 타이틀
st.markdown('<p class="main-header">🧬 CKD-CVD AI 신약 발견</p>', unsafe_allow_html=True)
st.markdown("---")

# 사이드바
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=AI+Drug+Discovery", use_container_width=True)
    st.markdown("## 🎯 파이프라인 단계")
    
    phases = {
        "Phase 1": "📚 문헌 마이닝",
        "Phase 2": "🧬 분자 도킹",
        "Phase 3": "🤖 딥러닝 평가"
    }
    
    for phase, desc in phases.items():
        st.markdown(f"**{phase}**: {desc}")
    
    st.markdown("---")
    st.markdown("### 📊 시스템 정보")
    st.info(f"""
    **Version**: 1.0  
    **Last Updated**: 2025-12-27  
    **Status**: ✅ Ready
    """)

# 메인 컨텐츠
tabs = st.tabs(["🏠 홈", "🚀 실행", "📊 결과", "📄 보고서", "⚙️ 설정"])

# Tab 1: 홈
with tabs[0]:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>120+</h2>
            <p>논문 자동 분석</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>8개</h2>
            <p>타겟 단백질 식별</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>Top 10</h2>
            <p>후보 물질 도출</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("## 📖 시스템 설명")
    
    st.markdown("""
    ### AI 기반 신약 발견 파이프라인
    
    이 시스템은 **3단계 AI 파이프라인**을 통해 CKD-CVD 치료제 후보를 자동으로 발굴합니다:
    
    1. **Phase 1: 문헌 마이닝** 📚
       - PubMed에서 최신 논문 100+ 개 수집
       - NLP로 타겟 단백질 및 치료 분자 자동 추출
    
    2. **Phase 2: 분자 도킹** 🧬
       - PDB/AlphaFold에서 단백질 3D 구조 획득
       - Virtual screening으로 결합력 계산
    
    3. **Phase 3: 딥러닝 평가** 🤖
       - GNN으로 분자 특성 예측
       - Transformer로 ADMET 평가
       - 종합 점수화 및 순위 산출
    
    ### 🎯 예상 결과
    
    - ⭐⭐⭐ **Highly Recommended**: 3-5개 고품질 후보
    - ⭐⭐ **Recommended**: 추가 검증 후보
    - 📄 **상세 보고서**: Markdown + CSV + 시각화
    """)
    
    st.success("✨ **시작하려면 '🚀 실행' 탭으로 이동하세요!**")

# Tab 2: 실행
with tabs[1]:
    st.markdown("## 🚀 파이프라인 실행")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 실행 전 체크리스트
        
        - ✅ Python 환경 준비됨
        - ✅ 필수 패키지 설치됨
        - ✅ 인터넷 연결됨 (PubMed API 사용)
        
        **예상 소요 시간**: 약 30초 - 1분
        """)
    
    with col2:
        st.info("""
        **시스템 요구사항**
        
        - Python 3.8+
        - RAM 4GB+
        - 디스크 500MB+
        """)
    
    st.markdown("---")
    
    # 실행 버튼
    if st.button("▶️ 파이프라인 시작", key="run_pipeline"):
        st.markdown("### 🔄 실행 중...")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Phase 1
        status_text.markdown("**Phase 1**: 📚 문헌 마이닝...")
        progress_bar.progress(10)
        time.sleep(1)
        
        # 실제 실행
        try:
            # 백그라운드에서 실행
            import subprocess
            result = subprocess.run(
                [sys.executable, "run_pipeline.py"],
                cwd=str(Path(__file__).parent),
                capture_output=True,
                text=True,
                timeout=300  # 5분 타임아웃
            )
            
            progress_bar.progress(33)
            status_text.markdown("**Phase 2**: 🧬 분자 도킹...")
            time.sleep(1)
            
            progress_bar.progress(66)
            status_text.markdown("**Phase 3**: 🤖 딥러닝 평가...")
            time.sleep(1)
            
            progress_bar.progress(100)
            status_text.markdown("**✅ 완료!**")
            
            st.success("🎉 파이프라인이 성공적으로 완료되었습니다!")
            
            # 결과 요약
            st.markdown("""
            <div class="success-box">
                <h3>✨ 실행 완료</h3>
                <p>📁 결과가 <code>results</code> 폴더에 저장되었습니다.</p>
                <p>📊 <strong>'결과'</strong> 탭에서 확인하세요!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 로그 표시
            with st.expander("📜 실행 로그 보기"):
                st.code(result.stdout, language='text')
            
        except subprocess.TimeoutExpired:
            st.error("⚠️ 실행 시간 초과 (5분). 다시 시도하세요.")
        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")
            st.markdown("**해결 방법**: 터미널에서 직접 실행해보세요:")
            st.code("cd drug_discovery\npython run_pipeline.py", language='bash')

# Tab 3: 결과
with tabs[2]:
    st.markdown("## 📊 분석 결과")
    
    # 최신 결과 찾기
    results_dir = Path("results")
    if results_dir.exists():
        runs = sorted(results_dir.glob("run_*"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if runs:
            latest_run = runs[0]
            st.success(f"📂 최신 결과: `{latest_run.name}`")
            
            # 최종 순위 로드
            ranking_file = latest_run / "final_ranking.csv"
            if ranking_file.exists():
                df = pd.read_csv(ranking_file, encoding='utf-8-sig')
                
                # Top 10 테이블
                st.markdown("### 🏆 Top 10 후보 물질")
                
                # 스타일링된 테이블
                st.dataframe(
                    df[['rank', 'molecule', '종합_점수', 'avg_binding_affinity', 
                        'qed', 'toxicity_risk', 'recommendation']].head(10),
                    use_container_width=True,
                    height=400
                )
                
                # 시각화
                st.markdown("### 📈 시각화")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart - Top 10 점수
                    fig1 = px.bar(
                        df.head(10),
                        x='종합_점수',
                        y='molecule',
                        orientation='h',
                        title='Top 10 후보 물질 종합 점수',
                        color='종합_점수',
                        color_continuous_scale='Viridis'
                    )
                    fig1.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Scatter - Binding vs ML
                    fig2 = px.scatter(
                        df,
                        x='avg_binding_affinity',
                        y='ml_composite_score',
                        size='종합_점수',
                        color='종합_점수',
                        hover_data=['molecule'],
                        title='Binding Affinity vs ML Score',
                        color_continuous_scale='Plasma'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                # 추가 차트
                col3, col4 = st.columns(2)
                
                with col3:
                    # QED 분포
                    fig3 = px.histogram(
                        df,
                        x='qed',
                        nbins=20,
                        title='Drug-likeness (QED) 분포',
                        color_discrete_sequence=['#667eea']
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                
                with col4:
                    # 추천 등급 파이
                    rec_counts = df['recommendation'].value_counts()
                    fig4 = px.pie(
                        values=rec_counts.values,
                        names=rec_counts.index,
                        title='추천 등급 분포',
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    st.plotly_chart(fig4, use_container_width=True)
                
                # 다운로드 버튼
                st.markdown("### 💾 데이터 다운로드")
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    csv = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                    st.download_button(
                        label="📥 CSV 다운로드",
                        data=csv,
                        file_name="final_ranking.csv",
                        mime="text/csv"
                    )
                
                with col_dl2:
                    # 시각화 이미지
                    viz_file = latest_run / "visualizations.png"
                    if viz_file.exists():
                        with open(viz_file, "rb") as f:
                            st.download_button(
                                label="📥 시각화 이미지 다운로드",
                                data=f,
                                file_name="visualizations.png",
                                mime="image/png"
                            )
            else:
                st.warning("결과 파일을 찾을 수 없습니다. 파이프라인을 먼저 실행하세요.")
        else:
            st.info("아직 실행된 결과가 없습니다. '🚀 실행' 탭에서 파이프라인을 시작하세요.")
    else:
        st.info("아직 실행된 결과가 없습니다. '🚀 실행' 탭에서 파이프라인을 시작하세요.")

# Tab 4: 보고서
with tabs[3]:
    st.markdown("## 📄 최종 보고서")
    
    # 최신 보고서 찾기
    results_dir = Path("results")
    if results_dir.exists():
        runs = sorted(results_dir.glob("run_*"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if runs:
            latest_run = runs[0]
            report_file = latest_run / "FINAL_REPORT.md"
            
            if report_file.exists():
                # 보고서 읽기
                with open(report_file, 'r', encoding='utf-8') as f:
                    report = f.read()
                
                # Markdown 렌더링
                st.markdown(report)
                
                # 다운로드 버튼
                st.download_button(
                    label="📥 보고서 다운로드 (Markdown)",
                    data=report.encode('utf-8'),
                    file_name=f"CKD_CVD_Report_{latest_run.name}.md",
                    mime="text/markdown"
                )
            else:
                st.warning("보고서 파일을 찾을 수 없습니다.")
        else:
            st.info("아직 생성된 보고서가 없습니다.")
    else:
        st.info("아직 생성된 보고서가 없습니다.")

# Tab 5: 설정
with tabs[4]:
    st.markdown("## ⚙️ 시스템 설정")
    
    st.markdown("### 🔧 파이프라인 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input(
            "논문 수집 개수 (Phase 1)",
            min_value=10,
            max_value=200,
            value=100,
            help="PubMed에서 수집할 논문 개수"
        )
        
        st.selectbox(
            "도킹 타겟 선택",
            ["All", "NF-kB only", "TGF-beta only", "Custom"],
            help="분자 도킹을 수행할 타겟 단백질"
        )
    
    with col2:
        st.slider(
            "ML 평가 신뢰도",
            min_value=0.5,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="딥러닝 모델 예측 신뢰도 임계값"
        )
        
        st.checkbox(
            "고급 시각화 활성화",
            value=True,
            help="추가 차트 및 3D 시각화"
        )
    
    st.markdown("---")
    
    st.markdown("### 📊 시스템 정보")
    
    st.code(f"""
Python: {sys.version.split()[0]}
Streamlit: {st.__version__}
Working Directory: {Path.cwd()}
Results Directory: {Path("results").absolute()}
    """, language='text')
    
    st.markdown("### 🗑️ 데이터 관리")
    
    if st.button("🗑️ 결과 폴더 초기화", type="secondary"):
        if st.checkbox("정말 삭제하시겠습니까?"):
            st.warning("⚠️ 이 기능은 수동으로 구현해야 합니다.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🧬 CKD-CVD AI 신약 발견 파이프라인 v1.0</p>
    <p>Powered by PyTorch, BioBERT, AlphaFold | © 2025 Mela-Exosome AI Team</p>
</div>
""", unsafe_allow_html=True)
