"""
NOVA Top 3 분석 및 진짜/가짜 분리 실험 시뮬레이터
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class FalsePositiveScreener:
    """False Positive 제거를 위한 6개 핵심 실험"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
    
    def screen_1_cell_viability(self, mol_features: Dict, reporter_ic50: float) -> Dict:
        """
        실험 1: Cell Viability Counterscreen
        리포터 IC50와 겹치면 가짜 양성
        """
        logp = mol_features['logp']
        mw = mol_features['mw']
        
        # 독성 예측 (LogP > 4.5, MW > 450 → 독성 증가)
        toxicity_factor = 1.0
        if logp > 4.5:
            toxicity_factor *= 0.5
        if mw > 450:
            toxicity_factor *= 0.7
        
        # Base viability IC50 (μM)
        base_viability_ic50 = np.random.lognormal(mean=2.5, sigma=0.8) * toxicity_factor
        
        # Noise
        viability_ic50 = base_viability_ic50 + np.random.normal(0, 0.5)
        viability_ic50 = max(viability_ic50, 1.0)
        
        # Selectivity window
        selectivity_window = viability_ic50 / reporter_ic50
        
        # Dose-response curve
        concentrations = [0.05, 0.1, 0.3, 1, 3, 10]  # μM
        viability_responses = []
        
        for conc in concentrations:
            viability = 100 / (1 + (viability_ic50 / conc)**1.0)
            viability = 100 - viability  # Convert to % viability
            viability += np.random.normal(0, 3)
            viability = np.clip(viability, 0, 100)
            viability_responses.append(viability)
        
        # 판정: Selectivity window > 10x → PASS
        pass_criteria = selectivity_window > 10
        
        return {
            'viability_IC50_uM': viability_ic50,
            'selectivity_window': selectivity_window,
            'concentrations': concentrations,
            'viability_percent': viability_responses,
            'pass': pass_criteria,
            'risk': 'LOW' if pass_criteria else 'HIGH (False Positive Risk!)'
        }
    
    def screen_2_luciferase_counterscreen(self, mol_features: Dict) -> Dict:
        """
        실험 2: Constitutive Luciferase Counterscreen
        루시퍼레이스 자체를 억제하는지 확인
        """
        # Quinazoline 계열은 luciferase inhibition 가능성 있음
        has_quinazoline = mol_features.get('has_quinazoline', False)
        
        # Base inhibition (should be low)
        base_inhib = np.random.uniform(5, 15)  # 5-15% (normal range)
        
        if has_quinazoline:
            # Quinazoline → luciferase 억제 가능성 약간 상승
            base_inhib += np.random.uniform(0, 10)
        
        # Noise
        luc_inhibition = base_inhib + np.random.normal(0, 3)
        luc_inhibition = np.clip(luc_inhibition, 0, 50)
        
        # Dose-response
        concentrations = [0.3, 1, 3, 10]  # μM
        luc_responses = []
        
        for conc in concentrations:
            response = luc_inhibition * (conc / 10)  # Dose-dependent
            response += np.random.normal(0, 3)
            response = np.clip(response, 0, 60)
            luc_responses.append(response)
        
        # 판정: < 20% inhibition @ 10 μM → PASS
        pass_criteria = luc_responses[-1] < 20
        
        return {
            'luc_inhibition_at_10uM': luc_responses[-1],
            'concentrations': concentrations,
            'luc_inhibition_percent': luc_responses,
            'pass': pass_criteria,
            'risk': 'LOW' if pass_criteria else 'HIGH (Luciferase Artifact!)'
        }
    
    def screen_3_psmad_timecourse(self, mol_features: Dict, ic50: float) -> Dict:
        """
        실험 3: p-SMAD2/3 Dose-Response + Time-Course
        빠른 시간대 억제 → 수용체 근처 타겟 가능성
        """
        # Time points (min)
        timepoints = [15, 30, 60, 120]
        
        # Doses (μM)
        doses = [0.3, 1, 3]
        
        results = {}
        
        for time in timepoints:
            time_responses = []
            
            for dose in doses:
                # Time-dependent response (빠를수록 강함)
                time_factor = 1.0 if time == 15 else (15 / time)**0.5
                
                # Dose-response
                base_inhib = 100 / (1 + (ic50 / dose)**1.0)
                inhib = base_inhib * time_factor
                inhib += np.random.normal(0, 5)
                inhib = np.clip(inhib, 0, 95)
                
                time_responses.append(inhib)
            
            results[f'{time}min'] = time_responses
        
        # 판정: 15 min에서 이미 > 40% 억제 (고농도) → 수용체 근처 타겟
        early_response = results['15min'][-1]  # 3 μM @ 15 min
        
        pass_criteria = early_response > 40
        
        return {
            'timepoints': timepoints,
            'doses': doses,
            'timecourse_data': results,
            'early_response_15min_3uM': early_response,
            'pass': pass_criteria,
            'interpretation': 'Upstream target (Receptor/ALK5)' if pass_criteria else 'Downstream target'
        }
    
    def screen_4_pikba_timecourse(self, mol_features: Dict, ic50: float) -> Dict:
        """
        实验 4: p-IκBα + p-p65 Time-Course + IκBα Degradation
        """
        timepoints = [15, 30, 60, 120]
        doses = [0.3, 1, 3]
        
        # p-IκBα
        pikba_results = {}
        for time in timepoints:
            time_responses = []
            for dose in doses:
                time_factor = 1.0 if time == 15 else (15 / time)**0.5
                base_inhib = 100 / (1 + (ic50 / dose)**1.0)
                inhib = base_inhib * time_factor
                inhib += np.random.normal(0, 5)
                inhib = np.clip(inhib, 0, 95)
                time_responses.append(inhib)
            pikba_results[f'{time}min'] = time_responses
        
        # IκBα degradation (should be blocked)
        degradation_blocked = pikba_results['30min'][-1] > 50  # 3 μM @ 30 min
        
        return {
            'timepoints': timepoints,
            'doses': doses,
            'pikba_timecourse': pikba_results,
            'ikba_degradation_blocked': degradation_blocked,
            'pass': degradation_blocked,
            'interpretation': 'IKK/TAK1 target' if degradation_blocked else 'Unclear'
        }
    
    def screen_5_protein_normalization(self, mol_features: Dict) -> Dict:
        """
        실험 5: Total Protein / Housekeeping Normalization
        웨스턴 신호 감소가 단백질 로딩 문제인지 확인
        """
        # Normally should be 1.0 (no change)
        total_protein = np.random.normal(1.0, 0.05)
        housekeeping = np.random.normal(1.0, 0.08)
        
        # If compound is toxic → protein loading ↓
        logp = mol_features['logp']
        if logp > 4.5:
            total_protein *= np.random.uniform(0.85, 0.95)
            housekeeping *= np.random.uniform(0.80, 0.90)
        
        # Normalization ratio
        norm_ratio = total_protein / housekeeping
        
        # 판정: 0.8 ~ 1.2 → PASS (정상)
        pass_criteria = 0.8 <= norm_ratio <= 1.2
        
        return {
            'total_protein_fold': total_protein,
            'housekeeping_fold': housekeeping,
            'normalization_ratio': norm_ratio,
            'pass': pass_criteria,
            'risk': 'LOW' if pass_criteria else 'HIGH (Loading Issue!)'
        }
    
    def screen_6_mini_kinase_panel(self, mol_features: Dict) -> Dict:
        """
        실험 6: Mini Kinase Panel (ALK5, TAK1, IKKβ)
        """
        has_quinazoline = mol_features.get('has_quinazoline', False)
        has_amide = mol_features.get('has_amide', False)
        logp = mol_features['logp']
        
        # Quinazoline + Amide → Kinase inhibitor 유리
        kinase_bonus = 0.5 if (has_quinazoline and has_amide) else 1.5
        logp_bonus = 1.0 if 2.5 <= logp <= 4.0 else 1.3
        
        # IC50 (nM)
        alk5_ic50 = np.random.lognormal(5.0, 0.5) * kinase_bonus * logp_bonus
        tak1_ic50 = np.random.lognormal(5.3, 0.5) * kinase_bonus * logp_bonus
        ikkb_ic50 = np.random.lognormal(5.2, 0.5) * kinase_bonus * logp_bonus
        
        # Determine primary target
        ic50s = {'ALK5': alk5_ic50, 'TAK1': tak1_ic50, 'IKKβ': ikkb_ic50}
        primary_target = min(ic50s, key=ic50s.get)
        
        # 판정: 1개 이상 < 200 nM → PASS
        pass_criteria = any(ic50 < 200 for ic50 in ic50s.values())
        
        return {
            'ALK5_IC50_nM': alk5_ic50,
            'TAK1_IC50_nM': tak1_ic50,
            'IKKb_IC50_nM': ikkb_ic50,
            'primary_target': primary_target,
            'primary_IC50_nM': ic50s[primary_target],
            'pass': pass_criteria,
            'conclusion': f'{primary_target} inhibitor' if pass_criteria else 'No clear kinase target'
        }
    
    def generate_comprehensive_report(self, smiles: str, mol_features: Dict, 
                                     reporter_ic50: float) -> Dict:
        """전체 6개 실험 실행 및 종합 판정"""
        
        # Run all 6 screens
        screen1 = self.screen_1_cell_viability(mol_features, reporter_ic50)
        screen2 = self.screen_2_luciferase_counterscreen(mol_features)
        screen3 = self.screen_3_psmad_timecourse(mol_features, reporter_ic50)
        screen4 = self.screen_4_pikba_timecourse(mol_features, reporter_ic50)
        screen5 = self.screen_5_protein_normalization(mol_features)
        screen6 = self.screen_6_mini_kinase_panel(mol_features)
        
        # Overall decision
        all_pass = (
            screen1['pass'] and
            screen2['pass'] and
            screen3['pass'] and
            screen4['pass'] and
            screen5['pass'] and
            screen6['pass']
        )
        
        # Confidence score
        pass_count = sum([
            screen1['pass'], screen2['pass'], screen3['pass'],
            screen4['pass'], screen5['pass'], screen6['pass']
        ])
        
        confidence = pass_count / 6.0
        
        # Final verdict
        if confidence >= 0.83:  # 5/6 or 6/6
            verdict = "TRUE POSITIVE - High Confidence Lead"
        elif confidence >= 0.67:  # 4/6
            verdict = "LIKELY TRUE - Requires Follow-up"
        elif confidence >= 0.50:  # 3/6
            verdict = "UNCERTAIN - Significant Risk"
        else:
            verdict = "FALSE POSITIVE - DROP"
        
        return {
            'smiles': smiles,
            'screen1_viability': screen1,
            'screen2_luciferase': screen2,
            'screen3_psmad': screen3,
            'screen4_pikba': screen4,
            'screen5_normalization': screen5,
            'screen6_kinase': screen6,
            'overall_pass': all_pass,
            'confidence_score': confidence,
            'verdict': verdict
        }


def analyze_top3_candidates(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """상위 3개 분자 상세 분석"""
    
    from nova_ml_data_generator import ExperimentalDataGenerator
    
    # Top 3 by GO probability
    top3 = predictions_df.sort_values('pred_Gate1_GO_prob', ascending=False).head(3)
    
    screener = FalsePositiveScreener(seed=42)
    gen = ExperimentalDataGenerator(seed=42)
    
    reports = []
    
    for idx, row in top3.iterrows():
        smiles = row['smiles']
        
        # Extract features
        mol_features = gen.extract_molecular_features(smiles)
        if not mol_features:
            continue
        
        # Use predicted IC50
        reporter_ic50 = row['pred_CAGA_IC50_uM']
        
        # Generate comprehensive report
        report = screener.generate_comprehensive_report(smiles, mol_features, reporter_ic50)
        
        reports.append(report)
    
    return reports


def save_top3_analysis_report(reports: List[Dict], output_path: Path):
    """Top 3 분석 보고서를 Markdown으로 저장"""
    
    md_content = "# NOVA Top 3 후보 물질 분석 보고서\n\n"
    md_content += "## Executive Summary\n\n"
    md_content += "상위 3개 예측 성공 후보에 대한 **6개 False Positive 제거 실험** 시뮬레이션 결과입니다.\n\n"
    md_content += "---\n\n"
    
    for i, report in enumerate(reports, 1):
        md_content += f"## Candidate {i}\n\n"
        md_content += f"**SMILES:** `{report['smiles']}`\n\n"
        md_content += f"**Final Verdict:** **{report['verdict']}**\n\n"
        md_content += f"**Confidence Score:** {report['confidence_score']:.1%} ({int(report['confidence_score']*6)}/6 tests passed)\n\n"
        
        md_content += "### 실험 결과 요약\n\n"
        
        # Screen 1
        s1 = report['screen1_viability']
        md_content += f"#### 1️⃣ Cell Viability Counterscreen: **{'✅ PASS' if s1['pass'] else '❌ FAIL'}**\n"
        md_content += f"- Viability IC50: {s1['viability_IC50_uM']:.2f} μM\n"
        md_content += f"- Selectivity Window: {s1['selectivity_window']:.1f}x\n"
        md_content += f"- Risk: {s1['risk']}\n\n"
        
        # Screen 2
        s2 = report['screen2_luciferase']
        md_content += f"#### 2️⃣ Luciferase Counterscreen: **{'✅ PASS' if s2['pass'] else '❌ FAIL'}**\n"
        md_content += f"- Luc Inhibition @ 10 μM: {s2['luc_inhibition_at_10uM']:.1f}%\n"
        md_content += f"- Risk: {s2['risk']}\n\n"
        
        # Screen 3
        s3 = report['screen3_psmad']
        md_content += f"#### 3️⃣ p-SMAD2/3 Time-Course: **{'✅ PASS' if s3['pass'] else '❌ FAIL'}**\n"
        md_content += f"- Early Response (15 min, 3 μM): {s3['early_response_15min_3uM']:.1f}%\n"
        md_content += f"- Interpretation: {s3['interpretation']}\n\n"
        
        # Screen 4
        s4 = report['screen4_pikba']
        md_content += f"#### 4️⃣ p-IκBα Time-Course: **{'✅ PASS' if s4['pass'] else '❌ FAIL'}**\n"
        md_content += f"- IκBα Degradation Blocked: {'Yes' if s4['ikba_degradation_blocked'] else 'No'}\n"
        md_content += f"- Interpretation: {s4['interpretation']}\n\n"
        
        # Screen 5
        s5 = report['screen5_normalization']
        md_content += f"#### 5️⃣ Protein Normalization Check: **{'✅ PASS' if s5['pass'] else '❌ FAIL'}**\n"
        md_content += f"- Normalization Ratio: {s5['normalization_ratio']:.2f}\n"
        md_content += f"- Risk: {s5['risk']}\n\n"
        
        # Screen 6
        s6 = report['screen6_kinase']
        md_content += f"#### 6️⃣ Mini Kinase Panel: **{'✅ PASS' if s6['pass'] else '❌ FAIL'}**\n"
        md_content += f"- ALK5 IC50: {s6['ALK5_IC50_nM']:.0f} nM\n"
        md_content += f"- TAK1 IC50: {s6['TAK1_IC50_nM']:.0f} nM\n"
        md_content += f"- IKKβ IC50: {s6['IKKb_IC50_nM']:.0f} nM\n"
        md_content += f"- **Primary Target:** {s6['primary_target']} ({s6['primary_IC50_nM']:.0f} nM)\n"
        md_content += f"- Conclusion: {s6['conclusion']}\n\n"
        
        md_content += "---\n\n"
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"✅ Top 3 분석 보고서 저장: {output_path}")


if __name__ == "__main__":
    # Test
    print("Testing False Positive Screener...")
    
    # Load predictions (if exists)
    pred_path = Path("generated_molecules/predictions_with_structures.csv")
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        
        print("\n🔬 Analyzing Top 3 candidates...")
        reports = analyze_top3_candidates(pred_df)
        
        # Save report
        output_md = Path("generated_molecules/Top3_Analysis_Report.md")
        save_top3_analysis_report(reports, output_md)
        
        print(f"\n📊 분석 완료! {len(reports)}개 후보 평가")
    else:
        print("❌ Predictions file not found. Run nova_insilico_validation_ui.py first.")
