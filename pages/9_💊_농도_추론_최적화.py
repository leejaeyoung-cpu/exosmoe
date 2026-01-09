import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
from io import BytesIO
from datetime import datetime

# 경로 추가
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import set_page_config, load_css
from src.concentration_inference import (
    ConcentrationInferenceModel, 
    PharmacokineticsParams,
    format_scientific,
    particles_to_mass
)

# 페이지 설정
set_page_config("엑소좀 농도 추론")
load_css()

# 제목
st.title("💊 엑소좀 치료제 농도 추론 및 최적화")
st.markdown("### CKD-CVD miRNA 칵테일의 최적 농도 및 투여 프로토콜 계산")

# 사이드바
st.sidebar.header("⚙️ 설정")

# 탭 생성
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Core-2 데이터",
    "🧪 모델 설정",
    "📐 농도 추론",
    "📈 시뮬레이션",
    "💾 프로토콜 생성"
])

# ========================================
# TAB 1: Core-2 데이터 확인
# ========================================
with tab1:
    st.header("1. Core-2 miRNA 칵테일 데이터")
    
    # Session state에서 데이터 로드 시도
    if 'df_candidates' in st.session_state and 'core2' in st.session_state:
        df = st.session_state['df_candidates']
        core2 = st.session_state['core2']
        
        st.success(f"✅ CKD-CVD 선별 페이지에서 데이터를 로드했습니다!")
        
        st.subheader("🎯 선정된 Core-2 miRNA")
        
        # Core-2 정보 표시
        col1, col2 = st.columns(2)
        
        with col1:
            mirna1 = core2['miRNA1']
            data1 = df[df['miRNA'] == mirna1].iloc[0]
            
            st.info(f"**miRNA #1: {mirna1}**")
            st.metric("Fold Change", f"{data1['FC_MT_vs_Con']:.2f}")
            st.metric("총 경로 수", int(data1['total_pathways']))
            st.metric("가중치 점수", f"{data1['weighted_score']:.2f}")
        
        with col2:
            mirna2 = core2['miRNA2']
            data2 = df[df['miRNA'] == mirna2].iloc[0]
            
            st.info(f"**miRNA #2: {mirna2}**")
            st.metric("Fold Change", f"{data2['FC_MT_vs_Con']:.2f}")
            st.metric("총 경로 수", int(data2['total_pathways']))
            st.metric("가중치 점수", f"{data2['weighted_score']:.2f}")
        
        # 비교 차트
        st.subheader("📊 Core-2 비교")
        
        comparison_df = pd.DataFrame({
            'miRNA': [mirna1, mirna2],
            'Fold Change': [data1['FC_MT_vs_Con'], data2['FC_MT_vs_Con']],
            '총 경로 수': [data1['total_pathways'], data2['total_pathways']],
            '가중치 점수': [data1['weighted_score'], data2['weighted_score']]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Fold Change',
            x=comparison_df['miRNA'],
            y=comparison_df['Fold Change'],
            yaxis='y',
            offsetgroup=1
        ))
        fig.add_trace(go.Bar(
            name='총 경로 수',
            x=comparison_df['miRNA'],
            y=comparison_df['총 경로 수'],
            yaxis='y2',
            offsetgroup=2
        ))
        
        fig.update_layout(
            title='Core-2 miRNA 특성 비교',
            yaxis=dict(title='Fold Change'),
            yaxis2=dict(title='총 경로 수', overlaying='y', side='right'),
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Session state에 농도 추론용 데이터 저장
        st.session_state['concentration_mirnas'] = [mirna1, mirna2]
        st.session_state['concentration_data'] = df
        
    else:
        st.warning("⚠️ CKD-CVD miRNA 선별 페이지에서 먼저 Core-2를 선정하세요.")
        st.info("**또는** 수동으로 miRNA를 입력할 수 있습니다:")
        
        col1, col2 = st.columns(2)
        with col1:
            manual_mirna1 = st.text_input("miRNA #1", value="hsa-miR-4739")
        with col2:
            manual_mirna2 = st.text_input("miRNA #2", value="hsa-miR-4651")
        
        if st.button("수동 입력 적용"):
            st.session_state['concentration_mirnas'] = [manual_mirna1, manual_mirna2]
            st.success("✅ 수동 입력 적용 완료!")

# ========================================
# TAB 2: 농도 모델 설정
# ========================================
with tab2:
    st.header("2. 농도 모델 파라미터 설정")
    
    st.subheader("🧬 엑소좀 및 miRNA 파라미터")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**엑소좀 농도 범위**")
        exo_min = st.number_input(
            "최소 농도 (particles/mL, 과학적 표기)",
            min_value=1e6,
            max_value=1e15,
            value=1e8,
            format="%.2e"
        )
        exo_max = st.number_input(
            "최대 농도 (particles/mL, 과학적 표기)",
            min_value=1e6,
            max_value=1e15,
            value=1e12,
            format="%.2e"
        )
        
        loading_efficiency = st.slider(
            "miRNA 로딩 효율 (%)",
            min_value=5.0,
            max_value=50.0,
            value=20.0,
            step=5.0
        ) / 100.0
    
    with col2:
        st.markdown("**투여 경로 및 환자 정보**")
        route = st.selectbox(
            "투여 경로",
            options=['IV', 'IP', 'SC'],
            format_func=lambda x: {
                'IV': 'IV - 정맥주사 (생체이용률 100%)',
                'IP': 'IP - 복강주사 (생체이용률 80%)',
                'SC': 'SC - 피하주사 (생체이용률 60%)'
            }[x],
            index=0
        )
        
        patient_weight = st.number_input(
            "환자 체중 (kg)",
            min_value=30.0,
            max_value=150.0,
            value=70.0,
            step=5.0
        )
    
    st.markdown("---")
    st.subheader("⚗️ 약동학(PK) 파라미터")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        half_life = st.number_input(
            "반감기 (시간)",
            min_value=1.0,
            max_value=72.0,
            value=24.0,
            step=1.0
        )
    
    with col2:
        volume_dist = st.number_input(
            "분포 용적 (L/kg)",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.01
        )
    
    with col3:
        clearance = st.number_input(
            "청소율 (L/hr/kg)",
            min_value=0.001,
            max_value=1.0,
            value=0.05,
            step=0.01
        )
    
    st.markdown("---")
    st.subheader("📊 용량-반응 모델 파라미터")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hill_coefficient = st.slider(
            "Hill 계수 (기울기)",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1
        )
    
    with col2:
        target_efficacy = st.slider(
            "목표 효능 (%)",
            min_value=50.0,
            max_value=95.0,
            value=70.0,
            step=5.0
        )
    
    # PK 파라미터 객체 생성 및 저장
    pk_params = PharmacokineticsParams(
        half_life=half_life,
        volume_distribution=volume_dist,
        clearance=clearance
    )
    
    st.session_state['pk_params'] = pk_params
    st.session_state['model_settings'] = {
        'exo_conc_range': (exo_min, exo_max),
        'loading_efficiency': loading_efficiency,
        'route': route,
        'patient_weight': patient_weight,
        'hill_coefficient': hill_coefficient,
        'target_efficacy': target_efficacy
    }
    
    st.success("✅ 모델 파라미터 설정 완료")

# ========================================
# TAB 3: 농도 추론 실행
# ========================================
with tab3:
    st.header("3. 농도 추론 알고리즘 실행")
    
    if 'concentration_mirnas' in st.session_state and 'concentration_data' in st.session_state:
        mirnas = st.session_state['concentration_mirnas']
        df_data = st.session_state['concentration_data']
        
        # 가중치 로드
        weights = st.session_state.get('weights', {
            'inflam': 0.25,
            'fib': 0.25,
            'anti': 0.20,
            'endo': 0.20,
            'cvd': 0.10,
            'sen': 0.05
        })
        
        if st.button("🔍 농도 추론 실행", type="primary"):
            with st.spinner("농도를 계산하고 있습니다..."):
                # 모델 초기화
                model = ConcentrationInferenceModel(df_data, weights)
                
                # PK 파라미터 적용
                if 'pk_params' in st.session_state:
                    model.pk_params = st.session_state['pk_params']
                
                # 개별 miRNA 농도 계산
                st.subheader("📊 개별 miRNA 농도 추정")
                
                results = []
                for mirna in mirnas:
                    base_conc = model.estimate_base_concentration(mirna)
                    ec50 = model.calculate_ec50(mirna)
                    ti_info = model.calculate_therapeutic_index(mirna)
                    
                    results.append({
                        'miRNA': mirna,
                        '기준 농도 (particles/mL)': base_conc,
                        'EC50 (particles/mL)': ec50,
                        'ED50 (particles/mL)': ti_info['ED50_particles_per_mL'],
                        'TD50 (particles/mL)': ti_info['TD50_particles_per_mL'],
                        '치료 지수 (TI)': ti_info['therapeutic_index'],
                        '안전성 평가': ti_info['safety_assessment']
                    })
                
                results_df = pd.DataFrame(results)
                
                # 과학적 표기법으로 포맷팅
                display_df = results_df.copy()
                for col in ['기준 농도 (particles/mL)', 'EC50 (particles/mL)', 
                           'ED50 (particles/mL)', 'TD50 (particles/mL)']:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2e}")
                
                st.dataframe(display_df, use_container_width=True)
                
                # 최적 혼합 비율 계산
                st.subheader("⚖️ 최적 혼합 비율")
                
                ratios = model.optimize_combination_ratio(mirnas, target_efficacy)
                
                ratio_df = pd.DataFrame({
                    'miRNA': list(ratios.keys()),
                    '혼합 비율': list(ratios.values()),
                    '퍼센트 (%)': [v*100 for v in ratios.values()]
                })
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.dataframe(ratio_df, use_container_width=True)
                
                with col2:
                    fig = px.pie(
                        ratio_df,
                        values='혼합 비율',
                        names='miRNA',
                        title='Core-2 miRNA 혼합 비율'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # 시너지 효과 계산
                if len(mirnas) == 2:
                    st.subheader("🔬 시너지 효과 분석")
                    
                    # 여러 농도에서 CI 계산
                    test_concs = np.logspace(9, 11, 10)
                    ci_values = [
                        model.calculate_synergy_index(mirnas[0], mirnas[1], c, ratios)
                        for c in test_concs
                    ]
                    
                    ci_df = pd.DataFrame({
                        '농도 (particles/mL)': test_concs,
                        'Combination Index (CI)': ci_values
                    })
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=ci_df['농도 (particles/mL)'],
                        y=ci_df['Combination Index (CI)'],
                        mode='lines+markers',
                        name='CI'
                    ))
                    
                    # CI = 1 기준선
                    fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                                 annotation_text="CI=1 (상가 효과)")
                    fig.add_hline(y=0.7, line_dash="dash", line_color="green",
                                 annotation_text="CI=0.7 (시너지)")
                    
                    fig.update_layout(
                        title='Combination Index vs 농도',
                        xaxis_type='log',
                        xaxis_title='농도 (particles/mL)',
                        yaxis_title='Combination Index',
                        yaxis_range=[0, 2]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    avg_ci = np.mean(ci_values)
                    if avg_ci < 0.7:
                        st.success(f"✅ 강한 시너지 효과! (평균 CI = {avg_ci:.2f})")
                    elif avg_ci < 1.0:
                        st.info(f"👍 시너지 효과 (평균 CI = {avg_ci:.2f})")
                    elif avg_ci == 1.0:
                        st.warning(f"⚠️ 상가 효과 (평균 CI = {avg_ci:.2f})")
                    else:
                        st.error(f"❌ 길항 효과 (평균 CI = {avg_ci:.2f})")
                
                # Session state에 저장
                st.session_state['inference_results'] = results_df
                st.session_state['mixture_ratios'] = ratios
                st.session_state['inference_model'] = model
    else:
        st.warning("⚠️ 먼저 Tab 1에서 miRNA 데이터를 로드하세요.")

# ========================================
# TAB 4: 농도-반응 시뮬레이션
# ========================================
with tab4:
    st.header("4. 농도-반응 곡선 시뮬레이션")
    
    if 'inference_model' in st.session_state and 'concentration_mirnas' in st.session_state:
        model = st.session_state['inference_model']
        mirnas = st.session_state['concentration_mirnas']
        
        st.subheader("📈 개별 miRNA 농도-반응 곡선")
        
        # 농도 범위 설정
        col1, col2 = st.columns(2)
        with col1:
            sim_min = st.number_input(
                "시뮬레이션 최소 농도",
                min_value=1e6,
                max_value=1e14,
                value=1e8,
                format="%.2e",
                key='sim_min'
            )
        with col2:
            sim_max = st.number_input(
                "시뮬레이션 최대 농도",
                min_value=1e7,
                max_value=1e15,
                value=1e12,
                format="%.2e",
                key='sim_max'
            )
        
        # 각 miRNA의 농도-반응 곡선 시뮬레이션
        all_curves = []
        for mirna in mirnas:
            curve_df = model.simulate_dose_response(
                mirna,
                conc_range=(sim_min, sim_max),
                n_points=100
            )
            all_curves.append(curve_df)
        
        combined_df = pd.concat(all_curves, ignore_index=True)
        
        # 플롯
        fig = px.line(
            combined_df,
            x='concentration',
            y='response',
            color='miRNA',
            title='Core-2 miRNA 농도-반응 곡선',
            labels={
                'concentration': '농도 (particles/mL)',
                'response': '치료 효과 (%)'
            },
            log_x=True
        )
        
        # EC50 마커 추가
        for mirna in mirnas:
            ec50 = model.calculate_ec50(mirna)
            fig.add_vline(
                x=ec50,
                line_dash="dash",
                annotation_text=f"{mirna} EC50",
                annotation_position="top"
            )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # 약동학 프로파일
        st.subheader("⏱️ 약동학(PK) 프로파일")
        
        if 'mixture_ratios' in st.session_state:
            ratios = st.session_state['mixture_ratios']
            settings = st.session_state.get('model_settings', {})
            
            # 투여량 계산
            protocol = model.generate_dosing_protocol(
                mirnas,
                patient_weight=settings.get('patient_weight', 70),
                route=settings.get('route', 'IV'),
                target_efficacy=settings.get('target_efficacy', 70)
            )
            
            # PK 프로파일 시뮬레이션
            time_hours = np.linspace(0, 72, 100)
            pk_df = model.simulate_pk_profile(
                dose_particles=protocol['dose_per_administration_particles'],
                patient_weight=settings.get('patient_weight', 70),
                route=settings.get('route', 'IV'),
                time_hours=time_hours
            )
            
            fig_pk = px.line(
                pk_df,
                x='time_hours',
                y='concentration_particles_per_mL',
                title='투여 후 혈중 농도 변화',
                labels={
                    'time_hours': '시간 (hours)',
                    'concentration_particles_per_mL': '농도 (particles/mL)'
                },
                log_y=True
            )
            
            # 반감기 마커
            half_life = model.pk_params.half_life
            fig_pk.add_vline(
                x=half_life,
                line_dash="dash",
                line_color="red",
                annotation_text=f"반감기 ({half_life}h)"
            )
            
            st.plotly_chart(fig_pk, use_container_width=True)
            
            # 정보 표시
            col1, col2, col3 = st.columns(3)
            col1.metric("Cmax", format_scientific(pk_df['concentration_particles_per_mL'].max()))
            col2.metric("반감기", f"{half_life:.1f} 시간")
            col3.metric("72시간 후 농도", format_scientific(pk_df['concentration_particles_per_mL'].iloc[-1]))
    
    else:
        st.warning("⚠️ 먼저 Tab 3에서 농도 추론을 실행하세요.")

# ========================================
# TAB 5: 투여 프로토콜 생성
# ========================================
with tab5:
    st.header("5. 최적 투여 프로토콜 생성")
    
    if 'inference_model' in st.session_state and 'concentration_mirnas' in st.session_state:
        model = st.session_state['inference_model']
        mirnas = st.session_state['concentration_mirnas']
        
        st.subheader("⚙️ 프로토콜 파라미터")
        
        col1, col2 = st.columns(2)
        
        with col1:
            protocol_weight = st.number_input(
                "환자 체중 (kg)",
                min_value=30.0,
                max_value=150.0,
                value=70.0,
                step=5.0,
                key='protocol_weight'
            )
            
            protocol_route = st.selectbox(
                "투여 경로",
                options=['IV', 'IP', 'SC'],
                index=0,
                key='protocol_route'
            )
        
        with col2:
            protocol_efficacy = st.slider(
                "목표 효능 (%)",
                min_value=50.0,
                max_value=95.0,
                value=70.0,
                step=5.0,
                key='protocol_efficacy'
            )
            
            treatment_days = st.number_input(
                "총 치료 기간 (일)",
                min_value=7,
                max_value=90,
                value=28,
                step=7,key='treatment_days'
            )
        
        if st.button("📋 프로토콜 생성", type="primary"):
            with st.spinner("프로토콜을 생성하고 있습니다..."):
                protocol = model.generate_dosing_protocol(
                    mirnas,
                    patient_weight=protocol_weight,
                    route=protocol_route,
                    target_efficacy=protocol_efficacy,
                    treatment_duration_days=treatment_days
                )
                
                st.success("✅ 투여 프로토콜 생성 완료!")
                
                # 프로토콜 요약
                st.subheader("📊 프로토콜 요약")
                
                col1, col2, col3 = st.columns(3)
                
                col1.metric(
                    "1회 투여량",
                    format_scientific(protocol['dose_per_administration_particles']) + " particles"
                )
                col2.metric(
                    "투여 간격",
                    f"{protocol['dosing_interval_hours']:.1f} 시간"
                )
                col3.metric(
                    "총 투여 횟수",
                    f"{protocol['total_doses']} 회"
                )
                
                # 상세 정보
                st.subheader("📝 상세 프로토콜")
                
                protocol_details = f"""
## 엑소좀 치료제 투여 프로토콜

### 기본 정보
- **날짜**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **환자 체중**: {protocol['patient_weight_kg']} kg
- **투여 경로**: {protocol['route']}
- **생체이용률**: {protocol['bioavailability']*100:.0f}%

### miRNA 조성 (Core-2 Cocktail)
"""
                for mirna, ratio in protocol['miRNA_composition'].items():
                    particles = protocol['individual_mirna_doses'][mirna]
                    protocol_details += f"- **{mirna}**: {ratio*100:.1f}% ({format_scientific(particles)} particles)\n"
                
                protocol_details += f"""

### 농도 정보
- **목표 농도**: {format_scientific(protocol['target_concentration_particles_per_mL'])} particles/mL
- **보정 농도** (생체이용률 반영): {format_scientific(protocol['adjusted_concentration_particles_per_mL'])} particles/mL
- **투여 용적**: {protocol['dose_volume_mL_per_kg']} mL/kg

### 투여 일정
- **1회 투여량**: {format_scientific(protocol['dose_per_administration_particles'])} particles
- **투여 간격**: {protocol['dosing_interval_hours']:.1f} 시간 (1일 {protocol['doses_per_day']:.1f}회)
- **치료 기간**: {protocol['treatment_duration_days']} 일
- **총 투여 횟수**: {protocol['total_doses']} 회

### 제조 지침

#### 엑소좀 준비
1. **MSC 배양**: Core-2 miRNA를 발현하도록 조작된 MSC 배양
2. **엑소좀 분리**: 초원심분리법 또는 크기 배제 크로마토그래피
3. **miRNA 로딩**: 전기천공법으로 miRNA 탑재 (효율 20%)
4. **농축**: 목표 농도까지 농축
5. **품질 관리**: 
   - 크기 분석 (NTA, DLS)
   - 마커 확인 (CD63, CD81, CD9)
   - miRNA 정량 (qPCR)

#### 투여 전 준비
1. 엑소좀 해동 (4°C에서 천천히)
2. 용량 계산 및 희석
3. 필터링 (0.22 μm)
4. 투여 직전 사용

### 모니터링 지표

#### 안전성 모니터링 (매 투여시)
- 활력 징후 (혈압, 맥박, 체온)
- 주사 부위 반응
- 알레르기 반응 관찰

#### 효능 모니터링 (주 1회)
- 혈청 크레아티닌
- eGFR
- 단백뇨
- 염증 마커 (CRP, IL-6)

#### 심화 평가 (월 1회)
- 신장 기능 종합 평가
- 심혈관 기능 평가
- 안전성 혈액 검사

### 주의사항
⚠️ **중요**: 이 프로토콜은 in silico 모델 기반 예측입니다. 실제 임상 적용 전 반드시 다음을 수행하세요:
- In vitro 효능 검증
- In vivo 동물 실험
- 독성 평가
- 임상시험심사위원회(IRB) 승인

---
*생성 시스템: Mela-Exosome AI - 농도 추론 모듈*
"""
                
                st.markdown(protocol_details)
                
                # 다운로드 버튼
                st.subheader("💾 다운로드")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📥 프로토콜 다운로드 (Markdown)",
                        data=protocol_details,
                        file_name=f"Exosome_Protocol_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown"
                    )
                
                with col2:
                    # Excel로 데이터 저장
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        # 프로토콜 요약
                        summary_df = pd.DataFrame({
                            '항목': [
                                '환자 체중 (kg)',
                                '투여 경로',
                                '목표 효능 (%)',
                                '1회 투여량 (particles)',
                                '투여 간격 (시간)',
                                '치료 기간 (일)',
                                '총 투여 횟수'
                            ],
                            '값': [
                                protocol['patient_weight_kg'],
                                protocol['route'],
                                protocol_efficacy,
                                format_scientific(protocol['dose_per_administration_particles']),
                                f"{protocol['dosing_interval_hours']:.1f}",
                                protocol['treatment_duration_days'],
                                protocol['total_doses']
                            ]
                        })
                        summary_df.to_excel(writer, sheet_name='요약', index=False)
                        
                        # miRNA 조성
                        composition_df = pd.DataFrame({
                            'miRNA': list(protocol['miRNA_composition'].keys()),
                            '비율 (%)': [v*100 for v in protocol['miRNA_composition'].values()],
                            '투여량 (particles)': [protocol['individual_mirna_doses'][m] for m in protocol['miRNA_composition'].keys()]
                        })
                        composition_df.to_excel(writer, sheet_name='miRNA_조성', index=False)
                    
                    st.download_button(
                        label="📥 프로토콜 다운로드 (Excel)",
                        data=buffer.getvalue(),
                        file_name=f"Exosome_Protocol_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                # Session state에 저장
                st.session_state['final_protocol'] = protocol
    
    else:
        st.warning("⚠️ 먼저 Tab 3에서 농도 추론을 실행하세요.")

# Footer
st.markdown("---")
st.markdown("**엑소좀 농도 추론 및 최적화 시스템** | Powered by Mela-Exosome AI")
