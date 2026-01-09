"""
NOVA AI 추론 보고서 생성기
각 Candidate의 역할, 강점, 리스크, 다음 단계를 AI가 분석
"""

from pathlib import Path
from typing import Dict

def generate_ai_interpretation_report(candidate_num: int, report: Dict) -> str:
    """각 Candidate에 대한 AI 추론 보고서 생성"""
    
    s1 = report['screen1_viability']
    s2 = report['screen2_luciferase']
    s3 = report['screen3_psmad']
    s4 = report['screen4_pikba']
    s5 = report['screen5_normalization']
    s6 = report['screen6_kinase']
    
    confidence = report['confidence_score']
    verdict = report['verdict']
    
    # Role classification
    role, role_desc = classify_candidate_role(s3, s4, s6, s5)
    
    md = f"""# AI 추론 보고서: Candidate {candidate_num}

## 🎯 최종 판정

**Verdict:** {verdict}  
**신뢰도:** {confidence:.1%} ({int(confidence*6)}/6 tests passed)  
**역할:** {role}  

---

## 📋 Executive Summary

{role_desc}

---

## 🔬 6개 실험 상세 해석

### 1️⃣ Cell Viability Counterscreen: **{'✅ PASS' if s1['pass'] else '❌ FAIL'}**

**결과:**
- Viability IC50: **{s1['viability_IC50_uM']:.2f} μM**
- Reporter IC50 (예상): ~0.15 μM
- Selectivity Window: **{s1['selectivity_window']:.1f}x**

**해석:**
"""
    
    if s1['selectivity_window'] > 100:
        md += f"""
- ✅ **매우 우수한 선택성** (> 100x)
- 리포터 IC50와 독성 IC50가 **{s1['selectivity_window']:.0f}배** 차이
- False positive(독성 때문에 신호 감소) 가능성 **매우 낮음**
- CKD 환자에서 치료 창(Therapeutic window) 확보 유리
"""
    elif s1['selectivity_window'] > 10:
        md += f"""
- ✅ **적절한 선택성** (10-100x)
- 리포터 억제가 독성 때문이 아님
- 추가 독성 연구 필요하지만, lead로 진행 가능
"""
    else:
        md += f"""
- ❌ **선택성 부족** (< 10x)
- 리포터 IC50와 독성 IC50가 겹침
- **False positive 리스크 높음** → 재검토 필요
"""
    
    md += f"""

---

### 2️⃣ Luciferase Counterscreen: **{'✅ PASS' if s2['pass'] else '❌ FAIL'}**

**결과:**
- Constitutive Luc 억제 @ 10 μM: **{s2['luc_inhibition_at_10uM']:.1f}%**

**해석:**
"""
    
    if s2['luc_inhibition_at_10uM'] < 10:
        md += f"""
- ✅ **매우 깨끗한 신호** (< 10%)
- Luciferase 효소 자체는 거의 건드리지 않음
- Reporter assay IC50가 **진짜 타겟 억제**를 반영
"""
    elif s2['luc_inhibition_at_10uM'] < 20:
        md += f"""
- ✅ **허용 가능** (10-20%)
- 약간의 luciferase 억제가 있지만 문제 수준은 아님
- Reporter IC50 해석 시 주의 필요
"""
    else:
        md += f"""
- ❌ **Artifact 리스크** (> 20%)
- Luciferase 자체를 억제할 가능성
- Reporter IC50의 신뢰도 **하락** → 다른 실험으로 재확인 필수
"""
    
    md += f"""

---

### 3️⃣ p-SMAD2/3 Time-Course: **{'✅ PASS' if s3['pass'] else '❌ FAIL'}**

**결과:**
- 15 min @ 3 μM: **{s3['early_response_15min_3uM']:.1f}% 억제**
- Interpretation: **{s3['interpretation']}**

**해석:**
"""
    
    if s3['early_response_15min_3uM'] > 70:
        md += f"""
- ✅ **매우 빠르고 강력한 억제** (> 70% @ 15 min)
- **Upstream target (Receptor/ALK5 근처)** 가능성 높음
- TGF-β 신호의 "근원지"를 차단하는 형태
- 섬유화 억제 효능 **강력할 것으로 예상**

**메커니즘 추정:**
```
TGF-β → [TGFBR1/ALK5 ← COMPOUND 억제] → p-SMAD2/3 ↓ → 섬유화 유전자 ↓
```
"""
    elif s3['early_response_15min_3uM'] > 40:
        md += f"""
- ✅ **적절한 억제** (40-70% @ 15 min)
- Upstream target 가능성 있음
- 추가 time-course (5 min, 10 min) 및 dose-response 권장
"""
    else:
        md += f"""
- ❌ **느리거나 약한 억제** (< 40% @ 15 min)
- Downstream target이거나 간접 효과 가능성
- TAK1/IKKβ를 통한 교차 억제일 수 있음
"""
    
    md += f"""

---

### 4️⃣ p-IκBα + IκBα Degradation: **{'✅ PASS' if s4['pass'] else '❌ FAIL'}**

**결과:**
- IκBα Degradation Blocked: **{'Yes' if s4['ikba_degradation_blocked'] else 'No'}**
- Interpretation: **{s4['interpretation']}**

**해석:**
"""
    
    if s4['ikba_degradation_blocked']:
        md += f"""
- ✅ **NF-κB 축을 실제로 차단**
- TNF-α 자극 시 IκBα가 분해되어야 하는데, 이를 막음
- **IKKβ 또는 TAK1 저해** 가능성 높음

**메커니즘 추정:**
```
TNF-α → [TAK1/IKKβ ← COMPOUND 억제] → p-IκBα ↓ → IκBα 유지 → NF-κB 핵 이동 차단
```

- 염증성 사이토카인(IL-6, CCL2) 억제 효능 기대
- CKD에서 **염증 + 섬유화 동시 제어** 가능
"""
    else:
        md += f"""
- ❌ **NF-κB 축 억제 불확실**
- IκBα degradation이 정상적으로 진행
- Reporter IC50는 나왔지만 **기능적 억제는 약함**
- TAK1/IKKβ가 아닌 다른 경로일 가능성
"""
    
    md += f"""

---

### 5️⃣ Protein Normalization Check: **{'✅ PASS' if s5['pass'] else '❌ FAIL'}**

**결과:**
- Total Protein: {s5['total_protein_fold']:.2f}
- Housekeeping: {s5['housekeeping_fold']:.2f}
- Normalization Ratio: **{s5['normalization_ratio']:.2f}**

**해석:**
"""
    
    if 0.9 <= s5['normalization_ratio'] <= 1.1:
        md += f"""
- ✅ **완벽한 정규화** (0.9-1.1)
- 웨스턴 블롯 신호 감소가 **단백질 로딩 문제가 아님**
- p-SMAD2/3, p-IκBα 억제가 **진짜 효과**
"""
    elif 0.8 <= s5['normalization_ratio'] <= 1.2:
        md += f"""
- ✅ **허용 가능** (0.8-1.2)
- 약간의 변동은 있지만 큰 문제 없음
"""
    else:
        md += f"""
- ❌ **정규화 이슈** (< 0.8 or > 1.2)
- 웨스턴 신호 변화가 **로딩/세포수 변화** 때문일 가능성
- 동일 조건으로 **재실험 권장** (β-actin, GAPDH 확인)
"""
    
    md += f"""

---

### 6️⃣ Mini Kinase Panel (ALK5, TAK1, IKKβ): **{'✅ PASS' if s6['pass'] else '❌ FAIL'}**

**결과:**
- **ALK5 (TGFBR1):** {s6['ALK5_IC50_nM']:.0f} nM
- **TAK1 (MAP3K7):** {s6['TAK1_IC50_nM']:.0f} nM
- **IKKβ (IKBKB):** {s6['IKKb_IC50_nM']:.0f} nM

**Primary Target:** **{s6['primary_target']}** ({s6['primary_IC50_nM']:.0f} nM)

**해석:**
"""
    
    primary = s6['primary_target']
    primary_ic50 = s6['primary_IC50_nM']
    
    all_sub200 = all([s6['ALK5_IC50_nM'] < 200, s6['TAK1_IC50_nM'] < 200, s6['IKKb_IC50_nM'] < 200])
    
    if all_sub200:
        md += f"""
- ✅ **Triple Kinase Inhibitor** (모두 < 200 nM)
- {primary}가 가장 강하지만 (**{primary_ic50:.0f} nM**)
- TAK1, IKKβ도 동시에 억제 → **Polypharmacology**

**장점:**
- TGF-β/SMAD (ALK5) + NF-κB (TAK1/IKKβ) **동시 차단**
- CKD에서 섬유화 + 염증 **synergistic 억제** 기대
- "One drug, dual pathway" 컨셉

**리스크:**
- Selectivity 문제 가능성 (EGFR, other kinases 확인 필수)
- Kinome-wide panel 권장
"""
    elif primary_ic50 < 100:
        md += f"""
- ✅ **매우 강력한 {primary} 억제제** (< 100 nM)
- Selective inhibitor로 최적화 가능
- {primary} 특이적 효과 확인 용이

**장점:**
- 명확한 MoA
- 특허성 강화 가능

**리스크:**
- Single pathway만 억제 시 효능 제한 가능성
"""
    else:
        md += f"""
- ❌ **Kinase 억제 약함** (모두 > 200 nM)
- Reporter IC50는 낮았지만, kinase IC50는 높음
- **간접 효과** 또는 **다른 타겟** 가능성
- 추가 kinase panel (확장) 권장
"""
    
    md += f"""

---

## 💡 종합 해석 및 역할

### **{role}**

{get_detailed_role_interpretation(role, s1, s2, s3, s4, s5, s6)}

---

## 🚀 다음 단계 권장사항

{get_next_steps_recommendation(role, candidate_num, s6)}

---

## ⚠️ 주요 리스크 및 대응

{get_risk_assessment(role, s1, s2, s5, s6)}

---

## 📊 Go/No-Go 의사결정

{get_go_nogo_decision(role, confidence, candidate_num)}

---

**생성 시각:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**AI 추론 엔진:** NOVA In Silico Validation System v1.0
"""
    
    return md


def classify_candidate_role(s3, s4, s6, s5) -> tuple:
    """Candidate 역할 분류"""
    
    psmad_strong = s3['early_response_15min_3uM'] > 70
    pikba_blocked = s4['ikba_degradation_blocked']
    all_kinase_sub200 = all([s6['ALK5_IC50_nM'] < 200, s6['TAK1_IC50_nM'] < 200, s6['IKKb_IC50_nM'] < 200])
    norm_ok = s5['pass']
    
    if psmad_strong and pikba_blocked and all_kinase_sub200 and norm_ok:
        return "Dual Pathway Inhibitor (우선순위 1)", "섬유화(ALK5) + 염증(TAK1/IKKβ) 동시 억제 리드"
    elif psmad_strong and s6['ALK5_IC50_nM'] < 100 and not pikba_blocked:
        return "Fibrosis-Focused Lead (섬유화 특화)", "ALK5 중심 강력한 섬유화 억제, 염증 효과는 제한적"
    elif not norm_ok:
        return "Questionable (재검토 필요)", "단백질 정규화 이슈로 데이터 신뢰도 재확인 필요"
    elif not s3['pass'] or not s4['pass']:
        return "Weak Lead (약한 리드)", "기능적 억제가 약하거나 불확실함"
    else:
        return "Moderate Lead (중간 리드)", "일부 실험 통과, 추가 검증 필요"


def get_detailed_role_interpretation(role, s1, s2, s3, s4, s5, s6) -> str:
    """역할별 상세 해석"""
    
    if "Dual Pathway" in role:
        return f"""
**이 Candidate를 "Dual Inhibitor"로 보는 근거:**

1. **ALK5 (TGFBR1):** {s6['ALK5_IC50_nM']:.0f} nM → 섬유화 경로 직접 차단
2. **TAK1/IKKβ:** {s6['TAK1_IC50_nM']:.0f} / {s6['IKKb_IC50_nM']:.0f} nM → 염증 경로 차단
3. **p-SMAD2/3 조기 억제:** {s3['early_response_15min_3uM']:.1f}% @ 15 min
4. **IκBα degradation 차단:** Yes

**CKD 컨셉 적합성:**
- TGF-β/SMAD (섬유화) + NF-κB (염증) = CKD 핵심 2대 경로
- 둘 다 커버 → **Synergistic 효과** 기대
- "One drug, dual benefit"

**경쟁사 대비 차별점:**
- Pirfenidone: 기전 불명확, 효능 제한적
- NAC: 항산화제, 직접 타겟 없음
- 본 후보: **명확한 kinase target + dual pathway**
"""
    
    elif "Fibrosis-Focused" in role:
        return f"""
**이 Candidate를 "섬유화 특화"로 보는 근거:**

1. **ALK5 매우 강력:** {s6['ALK5_IC50_nM']:.0f} nM (< 100 nM)
2. **p-SMAD2/3 조기 억제:** {s3['early_response_15min_3uM']:.1f}%
3. **IκBα 차단은 약함:** p-IκBα time-course FAIL

**장점:**
- ALK5 selectivity 높음 → 특허성 강화
- 섬유화 억제 효능 **극대화** 가능
- Off-target 리스크 낮을 가능성

**단점:**
- 염증 제어 효과 제한적
- CKD에서 "염증 + 섬유화" 동시 필요 시 효능 부족 가능

**활용 전략:**
- Candidate 1 (Dual)과 **병행 개발**
- 섬유화 dominant CKD 환자 타게팅
- Combination therapy의 섬유화 파트너
"""
    
    elif "Questionable" in role:
        return f"""
**재검토가 필요한 이유:**

1. **Protein Normalization FAIL:** {s5['normalization_ratio']:.2f}
   - 웨스턴 신호 감소가 로딩/세포수 변화 때문일 가능성
   
2. **데이터 신뢰도 하락:**
   - p-SMAD2/3, p-IκBα 억제가 **Artifact**일 수 있음
   
**대응:**
- 동일 조건으로 **재실험** (β-actin, GAPDH 정규화 명확히)
- 다른 세포주에서 Cross-validation
- 재실험 후 5/6 → 6/6으로 개선 시 GO 가능
"""
    
    else:
        return f"""
**이 Candidate의 한계:**

- 일부 실험만 통과
- 기능적 억제 weak 또는 inconsistent
- Lead로 진행하기엔 리스크 높음

**활용 가능성:**
- SAR (Structure-Activity Relationship) 학습용
- Hit-to-Lead 최적화 출발점
- Scaffold로만 활용
"""


def get_next_steps_recommendation(role, candidate_num, s6) -> str:
    """다음 단계 권장"""
    
    if "Dual" in role:
        return f"""
### A. Selectivity 정리 (필수, 2주)
**목적:** "좋은 dual"인지 kinome promiscuous인지 구분

1. **확장 Kinase Panel (30-100 kinases)**
   - EGFR, HER2, CDK2, Aurora A 등 대표 오프타겟
   - Selectivity ratio > 10x 목표
   
2. **EGFR/HER2 우선 확인**
   - Quinazoline 계열 → EGFR 억제 리스크
   - IC50 > 1 μM 목표

**판정:**
- EGFR selectivity > 10x → **GO**
- EGFR < 200 nM → **재최적화** 또는 DROP

### B. CKD 세포 기능 검증 (3-4주)
**세포주:** HK-2 (필수) + Renal fibroblast 또는 Podocyte

**측정 항목:**
1. Fibrosis markers (qPCR/Western)
   - COL1A1, FN1, ACTA2 (α-SMA), CTGF
   
2. Inflammation markers
   - CCL2 (MCP-1), IL-6, ICAM1
   
3. Phospho-Western
   - p-SMAD2/3 dose-response (0.1-10 μM)
   - p-p65 dose-response

**판정:**
- ≥3 genes 40% ↓ → **GO to In Vivo**

### C. In Vivo Proof-of-Concept (8-12주)
**모델:** UUO (Unilateral Ureteral Obstruction) - Mouse

**군:**
- Vehicle
- Pirfenidone (30 mg/kg, Positive control)
- Candidate {candidate_num} (10, 30 mg/kg)

**Endpoints:**
- Masson's Trichrome (섬유화 면적)
- IHC: α-SMA, F4/80 (대식구)
- qPCR: COL1A1, CCL2
- Serum Cr, BUN

**Go 기준:**
- Fibrosis ↓ ≥ 30% vs vehicle @ 30 mg/kg
"""
    
    elif "Fibrosis-Focused" in role:
        return f"""
### A. ALK5 Selectivity 극대화 (우선)
- ALK5 vs TAK1/IKKβ selectivity 확인
- > 10x selectivity → "ALK5 특이적" 포지셔닝

### B. 섬유화 모델 특화 검증
- Renal fibroblast에서 TGF-β induced COL1A1 억제
- Dose-response IC50 정밀 측정

### C. Combination 가능성 검토
- Candidate 1 (Dual) + Candidate {candidate_num} (Fibrosis) 병용 효과
"""
    
    else:
        return f"""
### A. 재실험 (필수)
- Protein normalization 정확히
- 독립적 replicate 3회

### B. 재평가 후 결정
- 재실험 결과가 일관되면 GO
- 여전히 inconsistent → DROP
"""


def get_risk_assessment(role, s1, s2, s5, s6) -> str:
    """리스크 평가"""
    
    risks = []
    
    if s1['selectivity_window'] < 20:
        risks.append("- ⚠️ **독성 리스크:** Selectivity window 좁음 → 세포 독성 재확인")
    
    if s2['luc_inhibition_at_10uM'] > 15:
        risks.append("- ⚠️ **Luciferase artifact:** Reporter IC50 과대평가 가능성")
    
    if not s5['pass']:
        risks.append("- 🚨 **데이터 신뢰도:** Protein normalization 이슈 → 재실험 필수")
    
    if s6['ALK5_IC50_nM'] < 100 and s6['IKKb_IC50_nM'] < 100:
        risks.append("- ⚠️ **EGFR off-target:** Quinazoline → EGFR 억제 리스크 높음")
    
    if not risks:
        risks.append("- ✅ **주요 리스크 없음:** 모든 counterscreen 통과")
    
    return "\n".join(risks)


def get_go_nogo_decision(role, confidence, candidate_num) -> str:
    """Go/No-Go 의사결정"""
    
    if "Dual" in role and confidence >= 0.83:
        return f"""
### ✅ **GO - 우선순위 1**

**이유:**
1. Dual pathway (섬유화 + 염증) 동시 커버
2. 6/6 또는 5/6 tests PASS
3. CKD 컨셉에 최적

**의사결정:**
- **Candidate {candidate_num}를 주력(Primary Lead)**으로 즉시 진행
- Selectivity panel + Cell validation 병행
- 6개월 내 IND-enabling study 목표

**No-Go Trigger:**
- EGFR IC50 < 200 nM (selectivity 부족)
- HK-2 세포 독성 (CC50 < 3 μM)
- UUO에서 효능 미달 (섬유화 ↓ < 20%)
"""
    
    elif "Fibrosis-Focused" in role:
        return f"""
### ⚠️ **GO - 백업 리드 (Backup Lead)**

**이유:**
1. ALK5 매우 강력 (섬유화 특화)
2. Dual은 아니지만, 단일 pathway 명확
3. Candidate 1과 병행 가치

**의사결정:**
- Candidate 1 (Dual)이 **실패 시** 대안
- 또는 **섬유화 dominant CKD** 타게팅
- Combination partner로 활용

**No-Go Trigger:**
- ALK5 selectivity 부족 (vs EGFR < 10x)
- 섬유화 억제 효능 weak
"""
    
    else:
        return f"""
### 🛑 **HOLD - 재검토 후 결정**

**이유:**
1. 데이터 품질 이슈 (normalization)
2. 또는 기능적 억제 불충분
3. Confidence {confidence:.1%} (너무 낮음)

**의사결정:**
- 재실험 후 재평가
- SAR 학습용으로만 활용
- Lead로는 부적합

**재평가 조건:**
- Protein normalization 통과
- 추가 replicate에서 일관성 확인
- → 5/6 이상 → GO로 전환 가능
"""


# Pandas import for timestamp
import pandas as pd

if __name__ == "__main__":
    print("AI Interpretation Report Generator Ready")
