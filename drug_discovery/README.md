# AI 기반 CKD-CVD 신약 발견 파이프라인

완전히 작동하는 AI 신약 개발 시스템이 구축되었습니다!

## 📁 프로젝트 구조

```
drug_discovery/
├── phase1_literature_mining.py    # 문헌 마이닝 및 지식 추출
├── phase2_molecular_docking.py    # 단백질 구조 & 분자 도킹
├── phase3_deep_learning.py        # 딥러닝 평가 (GNN + Transformer)
├── run_pipeline.py                # 전체 파이프라인 통합 실행
└── README.md                      # 이 파일

data/
├── literature/                    # 수집된 논문 데이터
├── protein_structures/            # 단백질 PDB 파일
├── docking_results/              # 도킹 시뮬레이션 결과
└── ml_evaluations/               # ML 평가 결과

results/
└── run_YYYYMMDD_HHMMSS/         # 실행 결과
    ├── FINAL_REPORT.md          # 최종 보고서
    ├── final_ranking.csv        # 후보 물질 순위
    └── visualizations.png       # 시각화
```

## 🚀 실행 방법

### 1. 환경 설정

```bash
pip install torch torchvision
pip install requests pandas numpy matplotlib seaborn
pip install torch-geometric  # optional, for GNN
```

### 2. 전체 파이프라인 실행

```bash
cd drug_discovery
python run_pipeline.py
```

### 3. 개별 Phase 실행

```bash
# Phase 1만
python phase1_literature_mining.py

# Phase 2만
python phase2_molecular_docking.py

# Phase 3만
python phase3_deep_learning.py
```

## 📊 파이프라인 흐름

```
[Phase 1: 문헌 마이닝]
     ↓
  PubMed API → 100+ 논문 수집
     ↓
  NLP 추출 → 타겟 단백질 & 치료 분자
     ↓
[Phase 2: 분자 도킹]
     ↓
  PDB/AlphaFold → 단백질 3D 구조
     ↓
  Virtual Screening → 결합력 계산
     ↓
[Phase 3: 딥러닝 평가]
     ↓
  GNN → 분자 특성 예측
     ↓
  Transformer → ADMET 평가
     ↓
[통합 & 최종 순위]
     ↓
  Top 10 후보 물질 도출!
```

## 🎯 핵심 타겟

1. **NF-κB p65** - 염증 경로
2. **TGF-β Receptor I** - 섬유화 경로
3. **NOX4** - 산화 스트레스
4. **VCAM1** - 내피 기능
5. **Cyclophilin D** - 미토콘드리아 보호

## 💊 예상 후보 물질

- Metformin (repurposing)
- Bardoxolone
- Pirfenidone
- 신규 화합물 Compound-A, B, C

## 📈 평가 지표

### 종합 점수 구성
- **Binding Affinity** (40%): 타겟 결합력
- **ADMET** (30%): 약물동태학
- **Drug-likeness** (20%): Lipinski, QED
- **Safety** (10%): 독성 위험

### 출력 결과

- `final_ranking.csv`: 순위, 점수, 추천 등급
- `FINAL_REPORT.md`: 상세 보고서
- `visualizations.png`: 차트 및 그래프

## 🔬 다음 단계

1. **Top 3 화학적 합성** ($1,500-3,000, 2-3주)
2. **In Vitro 검증** (HK-2, HUVEC, 4-8주)
3. **동물 실험** (CKD 마우스, 3-6개월)
4. **임상 진입** (IND 신청)

## ⚠️ 주의사항

이 시스템은 **In Silico** (컴퓨터 시뮬레이션) 기반입니다.

실제 신약 개발을 위해서는:
- ✅ 실험 검증 필수
- ✅ 독성 평가 필수
- ✅ 임상시험 승인 필요

## 📚 참고 자료

- PubMed API: https://www.ncbi.nlm.nih.gov/home/develop/api/
- PDB: https://www.rcsb.org/
- AlphaFold: https://alphafold.ebi.ac.uk/
- AutoDock Vina: http://vina.scripps.edu/

## 👨‍💻 개발자

- **Mela-Exosome AI Team**
- **Version**: 1.0
- **Date**: 2025-12-27

---

**🎉 AI로 CKD-CVD 치료제를 찾아봅시다!**
