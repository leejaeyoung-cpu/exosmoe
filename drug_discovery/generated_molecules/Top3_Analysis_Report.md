# NOVA Top 3 후보 물질 분석 보고서

## Executive Summary

상위 3개 예측 성공 후보에 대한 **6개 False Positive 제거 실험** 시뮬레이션 결과입니다.

---

## Candidate 1

**SMILES:** `O=C(Nc1ncnc2ccc(Cl)cc12)c1cccnc1`

**Final Verdict:** **TRUE POSITIVE - High Confidence Lead**

**Confidence Score:** 100.0% (6/6 tests passed)

### 실험 결과 요약

#### 1️⃣ Cell Viability Counterscreen: **✅ PASS**
- Viability IC50: 18.06 μM
- Selectivity Window: 36.5x
- Risk: LOW

#### 2️⃣ Luciferase Counterscreen: **✅ PASS**
- Luc Inhibition @ 10 μM: 14.2%
- Risk: LOW

#### 3️⃣ p-SMAD2/3 Time-Course: **✅ PASS**
- Early Response (15 min, 3 μM): 83.0%
- Interpretation: Upstream target (Receptor/ALK5)

#### 4️⃣ p-IκBα Time-Course: **✅ PASS**
- IκBα Degradation Blocked: Yes
- Interpretation: IKK/TAK1 target

#### 5️⃣ Protein Normalization Check: **✅ PASS**
- Normalization Ratio: 1.01
- Risk: LOW

#### 6️⃣ Mini Kinase Panel: **✅ PASS**
- ALK5 IC50: 82 nM
- TAK1 IC50: 145 nM
- IKKβ IC50: 99 nM
- **Primary Target:** ALK5 (82 nM)
- Conclusion: ALK5 inhibitor

---

## Candidate 2

**SMILES:** `COc1ccc(C(=O)Nc2ncnc3ccccc23)cc1Cl`

**Final Verdict:** **TRUE POSITIVE - High Confidence Lead**

**Confidence Score:** 100.0% (6/6 tests passed)

### 실험 결과 요약

#### 1️⃣ Cell Viability Counterscreen: **✅ PASS**
- Viability IC50: 10.96 μM
- Selectivity Window: 66.9x
- Risk: LOW

#### 2️⃣ Luciferase Counterscreen: **✅ PASS**
- Luc Inhibition @ 10 μM: 9.9%
- Risk: LOW

#### 3️⃣ p-SMAD2/3 Time-Course: **✅ PASS**
- Early Response (15 min, 3 μM): 95.0%
- Interpretation: Upstream target (Receptor/ALK5)

#### 4️⃣ p-IκBα Time-Course: **✅ PASS**
- IκBα Degradation Blocked: Yes
- Interpretation: IKK/TAK1 target

#### 5️⃣ Protein Normalization Check: **✅ PASS**
- Normalization Ratio: 0.91
- Risk: LOW

#### 6️⃣ Mini Kinase Panel: **✅ PASS**
- ALK5 IC50: 57 nM
- TAK1 IC50: 67 nM
- IKKβ IC50: 71 nM
- **Primary Target:** ALK5 (57 nM)
- Conclusion: ALK5 inhibitor

---

## Candidate 3

**SMILES:** `COc1ccc(C(=O)Nc2cccnc2)cc1Cl`

**Final Verdict:** **TRUE POSITIVE - High Confidence Lead**

**Confidence Score:** 83.3% (5/6 tests passed)

### 실험 결과 요약

#### 1️⃣ Cell Viability Counterscreen: **✅ PASS**
- Viability IC50: 25.50 μM
- Selectivity Window: 39.0x
- Risk: LOW

#### 2️⃣ Luciferase Counterscreen: **✅ PASS**
- Luc Inhibition @ 10 μM: 9.9%
- Risk: LOW

#### 3️⃣ p-SMAD2/3 Time-Course: **✅ PASS**
- Early Response (15 min, 3 μM): 81.3%
- Interpretation: Upstream target (Receptor/ALK5)

#### 4️⃣ p-IκBα Time-Course: **✅ PASS**
- IκBα Degradation Blocked: Yes
- Interpretation: IKK/TAK1 target

#### 5️⃣ Protein Normalization Check: **✅ PASS**
- Normalization Ratio: 1.11
- Risk: LOW

#### 6️⃣ Mini Kinase Panel: **❌ FAIL**
- ALK5 IC50: 445 nM
- TAK1 IC50: 607 nM
- IKKβ IC50: 545 nM
- **Primary Target:** ALK5 (445 nM)
- Conclusion: No clear kinase target

---

