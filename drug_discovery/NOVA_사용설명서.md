# NOVA Drug Discovery System - 사용 설명서

## 📦 시스템 구성

NOVA는 **AI 기반 신약 설계 플랫폼**으로 두 가지 핵심 모듈로 구성됩니다:

1. **NOVA De Novo Designer** - 분자 생성 및 설계
2. **NOVA In Silico Validation** - AI 실험 검증 및 분석

---

## 🚀 빠른 시작 (Quick Start)

### 방법 1: 통합 실행 (권장)

1. **`NOVA_Complete_Start.bat`** 더블클릭
2. 자동으로 두 시스템이 실행되고 브라우저가 열립니다
   - De Novo: `http://localhost:8502`
   - In Silico: `http://localhost:8503`

### 방법 2: 개별 실행

**De Novo Designer만 실행:**
- `NOVA_DeNovo_Start.bat` 더블클릭

**In Silico Validation만 실행:**
- `NOVA_InSilico_Start.bat` 더블클릭

---

## 📋 필수 요구사항

### 시스템 요구사항
- **OS:** Windows 10/11
- **RAM:** 8GB 이상 (16GB 권장)
- **디스크:** 2GB 이상 여유 공간

### 소프트웨어 요구사항
- **Python:** 3.8 이상 ([다운로드](https://www.python.org/downloads/))
- **인터넷:** 최초 실행 시 필요 (패키지 설치)

> **참고:** 배치 파일이 자동으로 필요한 Python 패키지를 설치합니다.

---

## 📚 모듈별 사용법

### 1️⃣ NOVA De Novo Designer (Port 8502)

**기능:**
- AI 기반 분자 생성
- 2D/3D 구조 시각화
- 작용 기전(MOA) 분석
- 물성 예측 (MW, LogP, TPSA 등)

**워크플로우:**
```
1. 🧬 분자 생성
   - 타겟 설정
   - 생성 파라미터 조정
   - AI 분자 생성 실행

2. 📊 결과 분석
   - Top N 후보 확인
   - 2D/3D 구조 확인
   - Lipinski Rule 검증

3. 🎯 작용 기전
   - Kinase 타겟 확인 (ALK5, TAK1, IKKβ)
   - 경로 차단 메커니즘 시각화

4. 💾 데이터 다운로드
   - CSV 내보내기
```

---

### 2️⃣ NOVA In Silico Validation (Port 8503)

**기능:**
- ML 합성 실험 데이터 생성
- 딥러닝 예측 모델 학습
- False Positive 제거 (6개 실험)
- AI 추론 보고서 자동 생성

**워크플로우:**
```
1️⃣ 합성 데이터 생성
   - De Novo에서 생성한 분자 로드
   - ML로 실험 데이터 시뮬레이션
   - Reporter assay, Western blot, qPCR 등

2️⃣ 딥러닝 학습
   - Multi-Task DNN 학습
   - IC50 regression + GO classification

3️⃣ 예측 & 검증
   - SMILES → 실험 결과 예측
   - R² score, Accuracy 확인
   - Top 10 후보 추출

4️⃣ 결과 분석
   - Confusion Matrix
   - Feature Importance

5️⃣ Top 3 진짜/가짜 분리 ⭐
   - 6개 False Positive 제거 실험
   - AI 추론 보고서 생성
   - Go/No-Go 의사결정
```

---

## 🔬 Top 3 진짜/가짜 분리 (핵심 기능)

### 6개 핵심 실험:

| # | 실험 | 목적 | PASS 기준 |
|---|------|------|----------|
| 1️⃣ | Cell Viability | Selectivity 확인 | SI > 10x |
| 2️⃣ | Luciferase Screen | Artifact 제거 | < 20% @ 10 μM |
| 3️⃣ | p-SMAD2/3 TC | Upstream target | > 40% @ 15 min |
| 4️⃣ | p-IκBα TC | IKK/TAK1 확인 | IκBα degradation blocked |
| 5️⃣ | Protein Norm | Loading 배제 | Ratio 0.8-1.2 |
| 6️⃣ | Kinase Panel | Primary target | ≥1 IC50 < 200 nM |

### AI 추론 보고서 포함:

- **역할 분류:** Dual / Fibrosis-Focused / Questionable
- **상세 해석:** 각 실험의 의미와 메커니즘
- **리스크 평가:** EGFR off-target 등
- **다음 단계:** Selectivity panel → Cell validation → In vivo
- **Go/No-Go 의사결정:** 명확한 기준과 trigger

---

## 🛠️ 문제 해결 (Troubleshooting)

### 1. Python이 설치되지 않은 경우

**증상:** "Python is not installed" 오류

**해결:**
1. https://www.python.org/downloads/ 접속
2. Python 3.8 이상 다운로드
3. 설치 시 **"Add Python to PATH"** 체크 필수
4. 재부팅 후 다시 실행

### 2. 패키지 설치 실패

**증상:** "Failed to install packages" 오류

**해결:**
```cmd
python -m pip install --upgrade pip
pip install streamlit pandas plotly numpy rdkit torch scikit-learn stmol py3Dmol matplotlib seaborn ipython_genutils
```

### 3. 포트가 이미 사용 중인 경우

**증상:** "Port 8502 is already in use" 오류

**해결:**
```cmd
# 실행 중인 Streamlit 프로세스 종료
taskkill /F /IM streamlit.exe

# 또는 다른 포트로 실행
streamlit run denovo_ui.py --server.port 8504
```

### 4. 3D 구조가 표시되지 않는 경우

**증상:** "3D 렌더링 오류" 메시지

**해결:**
```cmd
pip install stmol py3Dmol ipython_genutils nbformat ipywidgets
```

### 5. 한글이 깨지는 경우

**증상:** 그래프에서 한글이 ☐☐☐으로 표시

**해결:**
- Windows: 맑은 고딕 폰트 설치 확인
- 시스템 재부팅

---

## 📖 주요 용어 설명

| 용어 | 설명 |
|------|------|
| **SMILES** | 분자 구조를 텍스트로 표현하는 방식 |
| **IC50** | 50% 억제에 필요한 농도 (낮을수록 강력) |
| **LogP** | 지용성 척도 (2-4가 이상적) |
| **TPSA** | 극성 표면적 (40-90이 이상적) |
| **QED** | Drug-likeness 점수 (0-1, 높을수록 좋음) |
| **ALK5** | TGFBR1, 섬유화 경로의 핵심 kinase |
| **TAK1** | MAP3K7, 염증/섬유화 교차 노드 |
| **IKKβ** | NF-κB 활성화 kinase, 염증 핵심 |
| **SI** | Selectivity Index (독성 IC50 / 효능 IC50) |

---

## 💡 활용 전략

### 신약 개발 파이프라인에서의 활용

```
Phase 1: De Novo Design
├─ AI로 수천 개 분자 생성
├─ 물성 필터링 (Lipinski)
└─ Top 50 후보 선별

Phase 2: In Silico Validation (비용/시간 80% 절감)
├─ ML 합성 데이터 생성
├─ DL 예측 모델로 스크리닝
├─ Top 10 추출
└─ Top 3 False Positive 제거

Phase 3: 실제 실험 (Top 3만)
├─ Selectivity panel
├─ Cell validation
└─ In vivo (UUO model)

Result: $50K → $10K, 6개월 → 2개월
```

---

## 📞 지원 및 문의

**기술 지원:**
- 프로젝트 디렉토리: `c:\Users\brook\Desktop\mi_exo_ai\drug_discovery\`
- 로그 파일: `.streamlit/` 폴더 확인

**업데이트:**
```cmd
cd c:\Users\brook\Desktop\mi_exo_ai\drug_discovery
git pull origin main
```

---

## 📝 라이선스 및 인용

**NOVA Drug Discovery System v1.0**
- AI-Powered De Novo Design + In Silico Validation
- 2025 NOVA Therapeutics

**인용:**
```
NOVA In Silico Validation System
AI-Based False Positive Screening for Drug Discovery
Version 1.0 (2025)
```

---

## 🎯 다음 버전 로드맵

- [ ] ADMET 예측 강화
- [ ] 3D Docking simulation
- [ ] Cloud deployment (AWS/Azure)
- [ ] Multi-language support (English)
- [ ] API endpoint for integration
- [ ] Automated report generation (PDF)

---

**Last Updated:** 2025-12-28  
**Version:** 1.0.0
