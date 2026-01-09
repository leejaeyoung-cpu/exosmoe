"""
miRNA 기반 신규 분자 설계 - Enhanced Streamlit UI
분자 구조, 작용 기전, 경로 차단 시각화 포함
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import sys

# 모듈 import
sys.path.insert(0, str(Path(__file__).parent))

# RDKit import (필수)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw, Descriptors, Crippen, Lipinski
    from rdkit.Chem import rdMolDescriptors, QED
    RDKIT_AVAILABLE = True
except:
    RDKIT_AVAILABLE = False
    st.error("⚠️ RDKit이 필요합니다: `conda install -c conda-forge rdkit`")

# Generator import (수정된 버전 사용)
try:
    from denovo_molecule_generator import AdvancedMoleculeGenerator, MoleculeEvaluator
    GENERATOR_AVAILABLE = True
except:
    GENERATOR_AVAILABLE = False

# 페이지 설정
st.set_page_config(
    page_title="De Novo Drug Design",
    page_icon="🧬",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px;
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

# Header
st.markdown('<p class="main-header">🧬 De Novo Molecule Design</p>', unsafe_allow_html=True)
st.markdown("### AI 기반 신약 후보 물질 설계 시스템")
st.markdown("---")

# Session State
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'generated' not in st.session_state:
    st.session_state.generated = False

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ 설정")
    n_molecules = st.slider("생성할 분자 수", 50, 500, 150, 50)
    show_top_n = st.slider("표시할 후보 수", 5, 30, 15, 5)
    
    st.markdown("---")
    st.markdown("## 📊 타겟 단백질")
    st.info("""
    **NF-κB p65**
    - 역할: 염증 조절
    - PDB: 1VKX
    - Druggability: ⭐⭐⭐⭐
    
    **TGF-β R1**
    - 역할: 섬유화 조절
    - PDB: 3FAA
    - Druggability: ⭐⭐⭐⭐⭐
    """)

# Main Tabs
tabs = st.tabs(["🧬 분자 생성", "📊 결과 분석", "🎯 작용 기전", "💾 데이터"])

# Tab 1: 분자 생성
with tabs[0]:
    st.markdown("## 🧬 신규 분자 생성")
    
    col_gen1, col_gen2 = st.columns([2, 1])
    
    with col_gen1:
        if st.button("🚀 AI 분자 생성 시작", key="gen_btn"):
            if not GENERATOR_AVAILABLE:
                st.error("Generator 모듈을 불러올 수 없습니다.")
            else:
                progress = st.progress(0)
                status = st.empty()
                
                try:
                    status.markdown("**Step 1/3:** 분자 생성 중...")
                    progress.progress(33)
                    
                    generator = AdvancedMoleculeGenerator()
                    candidates = generator.generate_molecules_for_target(
                        'NF-κB p65', {}, n_molecules=n_molecules
                    )
                    
                    status.markdown("**Step 2/3:** 평가 및 필터링...")
                    progress.progress(66)
                    
                    evaluator = MoleculeEvaluator()
                    df = evaluator.filter_and_rank(candidates, 'NF-κB p65')
                    
                    status.markdown("**Step 3/3:** 저장 중...")
                    progress.progress(100)
                    
                    Path("generated_molecules").mkdir(exist_ok=True)
                    df.to_csv("generated_molecules/latest_candidates.csv", index=False)
                    
                    st.session_state.results_df = df
                    st.session_state.generated = True
                    
                    status.markdown("**✅ 완료!**")
                    st.success(f"🎉 {len(df)}개 고유 분자 생성 완료!")
                    
                except Exception as e:
                    st.error(f"오류: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with col_gen2:
        st.info(f"""
        **설정:**
        - 생성 수: {n_molecules}
        - 표시 수: {show_top_n}
        
        **필터:**
        - MW: 250-600
        - Rings: ≥ 2
        - 연결된 단일 분자
        """)

# Tab 2: 결과 분석 (Updated with 3D)
with tabs[1]:
    st.markdown("## 📊 생성 결과 분석")
    
    if not st.session_state.generated:
        st.info("먼저 '🧬 분자 생성' 탭에서 분자를 생성하세요.")
    else:
        df = st.session_state.results_df
        
        # Stats
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("총 분자 수", len(df))
        col_s2.metric("평균 MW", f"{df['mw'].mean():.1f} Da")
        col_s3.metric("평균 QED", f"{df['qed'].mean():.2f}")
        col_s4.metric("최고 점수", f"{df['final_score'].max():.3f}")
        
        st.markdown("---")
        
        # Top N 후보
        st.markdown(f"### 🏆 Top {show_top_n} 신약 후보")
        
        top_n_molecules = df.head(show_top_n)
        
        # Table
        st.dataframe(
            top_n_molecules[['id', 'smiles', 'mw', 'logp', 'qed', 'binding', 'final_score']],
            use_container_width=True,
            height=300
        )
        
        # Detailed molecule view
        st.markdown("---")
        st.markdown("### 🔬 분자 상세 분석 (2D & 3D)")
        
        # Select molecule
        mol_options = top_n_molecules['id'].tolist()
        selected_id = st.selectbox(
            "분자 선택",
            mol_options,
            format_func=lambda x: f"{x} | Score: {df[df['id']==x]['final_score'].values[0]:.3f}"
        )
        
        mol_row = df[df['id'] == selected_id].iloc[0]
        
        col_m1, col_m2, col_m3 = st.columns([1.5, 1.5, 1])
        
        with col_m1:
            st.markdown(f"#### 2D 구조 ({mol_row['id']})")
            if RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(mol_row['smiles'])
                if mol:
                    AllChem.Compute2DCoords(mol)
                    img = Draw.MolToImage(mol, size=(350, 350))
                    st.image(img, use_container_width=True)
        
        with col_m2:
            st.markdown(f"#### 3D 입체 구조")
            try:
                from stmol import showmol
                import py3Dmol
                
                # Generate 3D coords
                mol_3d = Chem.MolFromSmiles(mol_row['smiles'])
                mol_3d = Chem.AddHs(mol_3d)
                AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
                AllChem.MMFFOptimizeMolecule(mol_3d)
                
                # View
                view = py3Dmol.view(width=350, height=350)
                view.addModel(Chem.MolToMolBlock(mol_3d), 'mol')
                view.setStyle({'stick': {}})
                view.setBackgroundColor('white')
                view.zoomTo()
                
                showmol(view, height=350, width=350)
                st.caption("🖱️ 마우스로 회전/확대 가능")
                
            except Exception as e:
                st.warning(f"3D 렌더링 오류: {e}")
                st.info("pip install stmol py3Dmol 필요")

        with col_m3:
            st.markdown("#### 📊 물성 & 평가")
            st.metric("분자량", f"{mol_row['mw']:.1f} Da")
            st.metric("LogP", f"{mol_row['logp']:.2f}")
            st.metric("TPSA", f"{mol_row['tpsa']:.1f}")
            st.markdown("---")
            st.metric("Binding Score", f"{mol_row['binding']:.3f}")
            st.metric("Drug-likeness", f"{mol_row['qed']:.3f}")

# Tab 3: 작용 기전 (Revised)
with tabs[2]:
    st.markdown("## 🎯 작용 기전 (Revised Mechanism)")
    
    st.info("""
    **💡 과학적 타겟 재정의 (Scientific Update)**
    
    전사인자(NF-κB, SMAD)는 직접 저해가 어렵습니다.
    따라서 NOVA 분자는 상위 **'Druggable Kinase Node'**를 공략합니다.
    """)
    
    col_targets1, col_targets2 = st.columns(2)
    
    with col_targets1:
        st.markdown("### 🎯 Primary Targets (Kinases)")
        st.markdown("""
        1. **TGFBR1 (ALK5)** 🛑
           - 역할: TGF-β 수용체 키나아제
           - 효과: p-SMAD2/3 인산화 차단
           - 결과: **섬유화 억제**
           
        2. **TAK1 (MAP3K7)** 🛑
           - 역할: 염증/섬유화 교차 노드
           - 효과: IKK 및 p38 활성화 차단
           - 결과: **염증 & 섬유화 동시 억제**
           
        3. **IKKβ (IKBKB)** 🛑
           - 역할: NF-κB 활성화 효소
           - 효과: IκBα 분해 억제
           - 결과: **NF-κB 핵 이동 차단 (염증 억제)**
        """)
        
    with col_targets2:
        st.markdown("### 💊 분자 설계 전략")
        st.markdown("""
        - **Scaffold**: Quinazoline-Amide (Kinase Hinge Binder)
        - **Binding Mode**: ATP Competitive Inhibition
        - **Selectivity**: Gatekeeper residue 공략
        """)

    st.markdown("---")
    st.markdown("### 🔗 다중 경로 차단 메커니즘")

    # Professional matplotlib infographic (Updated Text)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch, Circle, FancyArrow
        import io
        from PIL import Image as PILImage
        import platform

        # 한글 폰트 설정
        if platform.system() == 'Windows':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        elif platform.system() == 'Darwin':
            plt.rcParams['font.family'] = 'AppleGothic'
        else:
            plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        
        # Create figure
        fig_mpl = plt.figure(figsize=(12, 8), facecolor='white')
        ax_mpl = fig_mpl.add_subplot(111)
        ax_mpl.set_xlim(0, 12)
        ax_mpl.set_ylim(0, 8)
        ax_mpl.axis('off')
        
        # Colors
        COLOR_PRIMARY = '#3498db'
        COLOR_SUCCESS = '#27ae60'
        COLOR_DANGER = '#e74c3c'
        COLOR_WARNING = '#f39c12'
        
        # Left: Upstream Kinases (Targets)
        targets = [
            ('IKKβ / TAK1', '염증 신호 개시', COLOR_DANGER, 6.5),
            ('NOX4', 'ROS 생성', COLOR_WARNING, 5.0),
            ('TGFBR1 (ALK5)', '섬유화 신호', '#e67e22', 3.5),
            ('NF-κB (Indirect)', '혈관 염증', '#c0392b', 2.0)
        ]
        
        for i, (name, desc, color, y) in enumerate(targets):
            # Target box
            box = FancyBboxPatch(
                (0.5, y-0.4), 2.8, 0.8,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor='white',
                linewidth=2,
                alpha=0.85
            )
            ax_mpl.add_patch(box)
            
            ax_mpl.text(1.9, y+0.15, name, ha='center', va='center',
                       fontsize=11, fontweight='bold', color='white')
            ax_mpl.text(1.9, y-0.15, desc, ha='center', va='center',
                       fontsize=8, color='white', style='italic')

        # Center: Drug molecule
        center_x, center_y = 6, 4
        hexagon = mpatches.RegularPolygon(
            (center_x, center_y), 6, radius=1.2,
            facecolor=COLOR_PRIMARY,
            edgecolor='white',
            linewidth=3
        )
        ax_mpl.add_patch(hexagon)
        
        ax_mpl.text(center_x, center_y+0.25, "NOVA", ha='center', va='center',
                   fontsize=18, fontweight='bold', color='white')
        ax_mpl.text(center_x, center_y-0.25, 'Kinase Inhibitor', ha='center', va='center',
                   fontsize=10, color='white')
        
        # Arrows
        for i, (name, desc, color, y) in enumerate(targets):
            arrow = FancyArrow(
                center_x - 0.9, center_y, 
                3.7 - (center_x - 0.9), y - center_y,
                width=0.12, head_width=0.25, head_length=0.25,
                facecolor=COLOR_DANGER, edgecolor='white', linewidth=1.5
            )
            ax_mpl.add_patch(arrow)
            
            mid_x = (center_x - 0.9 + 3.7) / 2
            mid_y = center_y + (y - center_y) / 2
            
            block = Circle((mid_x, mid_y), 0.35, facecolor='white', edgecolor=COLOR_DANGER, linewidth=2.5)
            ax_mpl.add_patch(block)
            ax_mpl.text(mid_x, mid_y, '🚫', ha='center', va='center', fontsize=16)

        # Right: Downstream Effects
        effects = [
            ('p-IκBα ↓', 'NF-κB 활성 억제', COLOR_SUCCESS, 6.5),
            ('ROS ↓', '산화 스트레스 감소', COLOR_SUCCESS, 5.0),
            ('p-SMAD2/3 ↓', '섬유화 유전자 억제', COLOR_SUCCESS, 3.5),
            ('VCAM1 ↓', '내피세포 보호', COLOR_SUCCESS, 2.0)
        ]
        
        for i, (name, desc, color, y) in enumerate(effects):
            box = FancyBboxPatch(
                (9, y-0.4), 2.5, 0.8,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor='white',
                linewidth=2,
                alpha=0.85
            )
            ax_mpl.add_patch(box)
            
            ax_mpl.text(10.25, y+0.15, name, ha='center', va='center',
                       fontsize=11, fontweight='bold', color='white')
            ax_mpl.text(10.25, y-0.15, desc, ha='center', va='center',
                       fontsize=9, color='white')
            
            ax_mpl.text(8.5, y, '✅', ha='center', va='center', fontsize=18)

        # Title
        title_box = FancyBboxPatch(
            (0.5, 7.3), 11, 0.6,
            boxstyle="round,pad=0.05",
            facecolor=COLOR_PRIMARY,
            edgecolor='white',
            linewidth=2
        )
        ax_mpl.add_patch(title_box)
        ax_mpl.text(6, 7.6, 'NOVA: Multi-Kinase Inhibition Mechanism',
                   ha='center', va='center', fontsize=16, fontweight='bold', color='white')

        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img_mech = PILImage.open(buf)
        
        st.image(img_mech, use_container_width=True)
        plt.close(fig_mpl)
        
    except Exception as e:
        st.error(f"인포그래픽 생성 오류: {e}")
        st.code(str(e))

# Tab 4: 데이터
with tabs[3]:
    st.markdown("## 💾 데이터 다운로드")
    
    if not st.session_state.generated:
        st.info("데이터가 없습니다.")
    else:
        df = st.session_state.results_df
        
        col_d1, col_d2, col_d3 = st.columns(3)
        
        with col_d1:
            csv_all = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 전체 데이터 CSV",
                csv_all,
                "all_molecules.csv",
                "text/csv"
            )
        
        with col_d2:
            csv_top = df.head(show_top_n).to_csv(index=False).encode('utf-8')
            st.download_button(
                f"📥 Top {show_top_n} CSV",
                csv_top,
                f"top{show_top_n}_molecules.csv",
                "text/csv"
            )
        
        with col_d3:
            smiles_txt = "\n".join(df['smiles'].tolist())
            st.download_button(
                "📥 SMILES TXT",
                smiles_txt,
                "smiles_list.txt",
                "text/plain"
            )
        
        st.markdown("---")
        st.info(f"💾 {len(df)}개 분자 데이터 준비 완료")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🧬 De Novo Molecule Design Platform v2.0</p>
    <p>Powered by RDKit + Chemical Reactions + AI</p>
</div>
""", unsafe_allow_html=True)
