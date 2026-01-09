"""
NOVA IND-Enabling ML Data Generator
5 Gate 파라미터를 포함한 임상 진입 가능성 예측 데이터 생성
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from nova_ml_data_generator import ExperimentalDataGenerator

class INDGateDataGenerator(ExperimentalDataGenerator):
    """IND Gate 기준 포함 확장 데이터 생성기"""
    
    def __init__(self, seed=42):
        super().__init__(seed)
    
    def generate_gate_a_data(self, mol_features: Dict) -> Dict:
        """
        Gate A: Lead Confirmation
        - EGFR selectivity
        - hERG IC50
        - Solubility
        - Reference standard purity
        """
        logp = mol_features['logp']
        mw = mol_features['mw']
        has_quinazoline = mol_features.get('has_quinazoline', False)
        
        # EGFR Selectivity (critical!)
        # Quinazoline → EGFR 억제 리스크
        if has_quinazoline:
            # Quinazoline은 EGFR 억제 가능성 높음
            egfr_ic50_nM = np.random.lognormal(mean=5.5, sigma=0.8)  # ~250 nM
            alk5_ic50_nM = np.random.lognormal(mean=4.5, sigma=0.5)  # ~90 nM
        else:
            egfr_ic50_nM = np.random.lognormal(mean=7.0, sigma=1.0)  # ~1000 nM
            alk5_ic50_nM = np.random.lognormal(mean=5.0, sigma=0.5)  # ~150 nM
        
        egfr_selectivity = egfr_ic50_nM / alk5_ic50_nM
        
        # hERG IC50
        # LogP 높으면 hERG 억제 증가
        base_herg = np.random.lognormal(mean=8.0, sigma=1.0)  # ~3 μM
        if logp > 4.5:
            base_herg *= 0.5
        herg_ic50_uM = base_herg / 1000  # Convert to μM
        
        # Thermodynamic Solubility
        # LogP 높으면 용해도 낮음
        base_sol = np.random.lognormal(mean=4.0, sigma=0.8)  # ~50 μM
        if logp > 4.0:
            base_sol *= 0.3
        elif logp < 2.5:
            base_sol *= 2.0
        
        thermo_solubility_uM = max(base_sol, 1.0)
        
        # Reference Standard Purity
        purity = np.random.normal(99.2, 0.3)
        purity = np.clip(purity, 98.0, 99.9)
        
        # Gate A 판정
        gate_a_pass = (
            egfr_selectivity > 10 and
            herg_ic50_uM > 10 and
            thermo_solubility_uM > 50 and
            purity >= 99.0
        )
        
        return {
            'EGFR_IC50_nM': egfr_ic50_nM,
            'ALK5_IC50_nM': alk5_ic50_nM,
            'EGFR_Selectivity': egfr_selectivity,
            'hERG_IC50_uM': herg_ic50_uM,
            'Thermo_Solubility_uM': thermo_solubility_uM,
            'Ref_Std_Purity': purity,
            'Gate_A_PASS': gate_a_pass
        }
    
    def generate_gate_b_data(self, mol_features: Dict) -> Dict:
        """
        Gate B: CMC
        - API scale-up success
        - Impurity profile
        - Stability (6M)
        """
        mw = mol_features['mw']
        
        # API Scale-up Success (g → 100g)
        # MW 높거나 복잡하면 어려움
        complexity_factor = 1.0
        if mw > 400:
            complexity_factor *= 0.7
        
        scaleup_success_prob = np.random.uniform(0.7, 0.95) * complexity_factor
        scaleup_success = np.random.random() < scaleup_success_prob
        
        # Overall Yield
        if scaleup_success:
            overall_yield = np.random.normal(45, 10)
        else:
            overall_yield = np.random.normal(25, 10)
        overall_yield = np.clip(overall_yield, 10, 80)
        
        # Impurity Profile
        # 각 불순물 %
        max_single_impurity = np.random.lognormal(mean=-3.0, sigma=0.5)  # ~0.05%
        total_impurities = max_single_impurity * np.random.uniform(2, 4)
        
        # Residual Solvents (ppm)
        residual_solvents_ppm = np.random.lognormal(mean=5.0, sigma=1.0)  # ~150 ppm
        
        # Stability (6M @ 25°C/60% RH)
        # Assay 변화 %
        stability_assay_change = np.random.normal(1.5, 1.0)  # ~1.5% 감소
        stability_assay_change = np.clip(abs(stability_assay_change), 0, 5)
        
        # Stability impurity increase
        stability_impurity_increase = np.random.normal(0.05, 0.03)
        stability_impurity_increase = np.clip(stability_impurity_increase, 0, 0.2)
        
        # Gate B 판정
        gate_b_pass = (
            scaleup_success and
            overall_yield > 40 and
            max_single_impurity < 0.10 and
            total_impurities < 0.5 and
            residual_solvents_ppm < 5000 and
            stability_assay_change < 3.0 and
            (max_single_impurity + stability_impurity_increase) < 0.15
        )
        
        return {
            'API_ScaleUp_Success': scaleup_success,
            'Overall_Yield_Percent': overall_yield,
            'Max_Single_Impurity_Percent': max_single_impurity,
            'Total_Impurities_Percent': total_impurities,
            'Residual_Solvents_ppm': residual_solvents_ppm,
            'Stability_Assay_Change_Percent': stability_assay_change,
            'Stability_Impurity_Increase': stability_impurity_increase,
            'Gate_B_PASS': gate_b_pass
        }
    
    def generate_gate_c_data(self, mol_features: Dict) -> Dict:
        """
        Gate C: Toxicology
        - NOAEL (Rat, Dog)
        - Genotoxicity (Ames, in vitro, in vivo)
        - QTc prolongation
        - CKD markers
        """
        logp = mol_features['logp']
        mw = mol_features['mw']
        
        # NOAEL (Rat, 2-week)
        # LogP > 4.5 → 독성 증가
        base_noael_rat = np.random.lognormal(mean=3.5, sigma=0.5)  # ~30 mg/kg
        if logp > 4.5:
            base_noael_rat *= 0.5
        elif logp < 2.5:
            base_noael_rat *= 1.5
        
        noael_rat_mg_kg = max(base_noael_rat, 1.0)
        
        # NOAEL (Dog, 2-week)
        noael_dog_mg_kg = noael_rat_mg_kg * np.random.uniform(0.7, 1.3)
        
        # Genotoxicity
        # Ames Test
        ames_positive_prob = 0.05  # 5% baseline
        if mw > 500:
            ames_positive_prob += 0.05
        ames_positive = np.random.random() < ames_positive_prob
        
        # In vitro Chromosome Aberration
        in_vitro_positive_prob = 0.08
        in_vitro_positive = np.random.random() < in_vitro_positive_prob
        
        # In vivo Micronucleus
        in_vivo_positive_prob = 0.03
        if ames_positive:
            in_vivo_positive_prob += 0.15
        in_vivo_positive = np.random.random() < in_vivo_positive_prob
        
        genotox_all_negative = not (ames_positive or in_vitro_positive or in_vivo_positive)
        
        # QTc Prolongation (% change from baseline)
        # hERG와 상관
        herg_ic50 = mol_features.get('hERG_IC50_uM', 10.0)
        if herg_ic50 < 10:
            qtc_prolongation = np.random.normal(8, 3)
        elif herg_ic50 < 30:
            qtc_prolongation = np.random.normal(3, 2)
        else:
            qtc_prolongation = np.random.normal(1, 1)
        
        qtc_prolongation = np.clip(abs(qtc_prolongation), 0, 20)
        
        # CKD-specific markers
        # KIM-1 fold change
        kim1_fold = np.random.lognormal(mean=0.2, sigma=0.3)  # ~1.2x
        
        # NGAL fold change
        ngal_fold = np.random.lognormal(mean=0.1, sigma=0.3)  # ~1.1x
        
        # Kidney histopathology score (0-4)
        kidney_histo_score = np.random.poisson(lam=0.5)
        kidney_histo_score = min(kidney_histo_score, 4)
        
        # Gate C 판정
        gate_c_pass = (
            noael_rat_mg_kg > 10 and
            noael_dog_mg_kg > 10 and
            genotox_all_negative and
            qtc_prolongation < 5 and
            kim1_fold < 2.0 and
            ngal_fold < 2.0 and
            kidney_histo_score <= 1
        )
        
        return {
            'NOAEL_Rat_mg_kg': noael_rat_mg_kg,
            'NOAEL_Dog_mg_kg': noael_dog_mg_kg,
            'Ames_Positive': ames_positive,
            'InVitro_CA_Positive': in_vitro_positive,
            'InVivo_MN_Positive': in_vivo_positive,
            'Genotox_All_Negative': genotox_all_negative,
            'QTc_Prolongation_Percent': qtc_prolongation,
            'KIM1_Fold_Change': kim1_fold,
            'NGAL_Fold_Change': ngal_fold,
            'Kidney_Histo_Score': kidney_histo_score,
            'Gate_C_PASS': gate_c_pass
        }
    
    def generate_gate_d_data(self, gate_c_data: Dict) -> Dict:
        """
        Gate D: Phase 1 Design
        - Starting dose (calculated from NOAEL)
        - Dose escalation feasibility
        """
        noael_rat = gate_c_data['NOAEL_Rat_mg_kg']
        
        # HED (Human Equivalent Dose)
        # HED = NOAEL × (Animal Wt / Human Wt)^0.33
        # Rat: 0.2 kg, Human: 60 kg
        hed_factor = (0.2 / 60) ** 0.33
        hed_mg_kg = noael_rat * hed_factor
        
        # Starting dose (HED / Safety Factor)
        safety_factor = np.random.uniform(8, 12)  # 보통 10
        starting_dose_mg_kg = hed_mg_kg / safety_factor
        starting_dose_mg = starting_dose_mg_kg * 70  # 70 kg adult
        
        # Escalation steps possible (Modified Fibonacci)
        # Starting dose가 너무 낮으면 escalation 어려움
        if starting_dose_mg > 5:
            escalation_steps = np.random.randint(5, 8)
        elif starting_dose_mg > 1:
            escalation_steps = np.random.randint(3, 6)
        else:
            escalation_steps = np.random.randint(1, 4)
        
        # Gate D 판정
        gate_d_pass = (
            starting_dose_mg >= 1 and
            escalation_steps >= 3
        )
        
        return {
            'HED_mg_kg': hed_mg_kg,
            'Starting_Dose_mg_kg': starting_dose_mg_kg,
            'Starting_Dose_mg': starting_dose_mg,
            'Safety_Factor': safety_factor,
            'Escalation_Steps_Possible': escalation_steps,
            'Gate_D_PASS': gate_d_pass
        }
    
    def generate_gate_e_data(self, all_gates_data: Dict) -> Dict:
        """
        Gate E: Regulatory
        - Pre-IND meeting success
        - IND submission readiness
        """
        # Pre-IND Meeting Success Probability
        # Gate A-D가 모두 통과면 높음
        base_pre_ind_success = 0.7
        
        if all_gates_data['Gate_A_PASS']:
            base_pre_ind_success += 0.1
        if all_gates_data['Gate_B_PASS']:
            base_pre_ind_success += 0.1
        if all_gates_data['Gate_C_PASS']:
            base_pre_ind_success += 0.1
        
        pre_ind_success = np.random.random() < base_pre_ind_success
        
        # IND Package Completeness (%)
        if pre_ind_success:
            ind_completeness = np.random.normal(90, 5)
        else:
            ind_completeness = np.random.normal(75, 10)
        
        ind_completeness = np.clip(ind_completeness, 50, 100)
        
        # FDA Review Outcome
        # Clinical Hold 확률
        clinical_hold_prob = 0.2  # Baseline 20%
        if all_gates_data.get('EGFR_Selectivity', 0) < 5:
            clinical_hold_prob += 0.3
        if not all_gates_data.get('Genotox_All_Negative', False):
            clinical_hold_prob += 0.4
        if all_gates_data.get('QTc_Prolongation_Percent', 0) > 10:
            clinical_hold_prob += 0.3
        
        clinical_hold = np.random.random() < clinical_hold_prob
        
        # Gate E 판정
        gate_e_pass = (
            pre_ind_success and
            ind_completeness >= 80 and
            not clinical_hold
        )
        
        return {
            'Pre_IND_Meeting_Success': pre_ind_success,
            'IND_Completeness_Percent': ind_completeness,
            'Clinical_Hold_Issued': clinical_hold,
            'Gate_E_PASS': gate_e_pass
        }
    
    def generate_full_ind_dataset(self, smiles_list: list) -> pd.DataFrame:
        """
        전체 IND Gate 데이터셋 생성
        """
        all_results = []
        
        for smiles in smiles_list:
            mol_features = self.extract_molecular_features(smiles)
            if not mol_features:
                continue
            
            # Generate all gates
            gate_a = self.generate_gate_a_data(mol_features)
            gate_b = self.generate_gate_b_data(mol_features)
            
            # Gate C needs hERG from Gate A
            mol_features['hERG_IC50_uM'] = gate_a['hERG_IC50_uM']
            gate_c = self.generate_gate_c_data(mol_features)
            
            gate_d = self.generate_gate_d_data(gate_c)
            
            # Gate E needs all previous
            all_gates_data = {**gate_a, **gate_b, **gate_c, **gate_d}
            gate_e = self.generate_gate_e_data(all_gates_data)
            
            # Overall IND Success
            ind_success = (
                gate_a['Gate_A_PASS'] and
                gate_b['Gate_B_PASS'] and
                gate_c['Gate_C_PASS'] and
                gate_d['Gate_D_PASS'] and
                gate_e['Gate_E_PASS']
            )
            
            # Calculate IND Score (0-100)
            gate_scores = [
                gate_a['Gate_A_PASS'],
                gate_b['Gate_B_PASS'],
                gate_c['Gate_C_PASS'],
                gate_d['Gate_D_PASS'],
                gate_e['Gate_E_PASS']
            ]
            ind_score = sum(gate_scores) / 5 * 100
            
            # Risk Level
            if ind_score >= 80:
                risk_level = "LOW"
            elif ind_score >= 60:
                risk_level = "MEDIUM"
            elif ind_score >= 40:
                risk_level = "HIGH"
            else:
                risk_level = "VERY HIGH"
            
            result = {
                'smiles': smiles,
                'MW': mol_features['mw'],
                'LogP': mol_features['logp'],
                'TPSA': mol_features['tpsa'],
                
                # Gate A
                **gate_a,
                
                # Gate B
                **gate_b,
                
                # Gate C
                **gate_c,
                
                # Gate D
                **gate_d,
                
                # Gate E
                **gate_e,
                
                # Overall
                'IND_Success': ind_success,
                'IND_Score': ind_score,
                'Risk_Level': risk_level
            }
            
            all_results.append(result)
        
        return pd.DataFrame(all_results)


def main():
    """메인 실행"""
    # Load NOVA candidates
    csv_path = Path("generated_molecules/latest_candidates.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        smiles_list = df['smiles'].tolist()[:50]
        
        print(f"📊 Generating IND Gate data for {len(smiles_list)} molecules...")
        
        generator = INDGateDataGenerator(seed=42)
        ind_data = generator.generate_full_ind_dataset(smiles_list)
        
        # Save
        output_path = Path("generated_molecules/ind_gate_data.csv")
        ind_data.to_csv(output_path, index=False)
        
        print(f"✅ Saved to: {output_path}")
        print(f"📈 IND Success rate: {ind_data['IND_Success'].sum() / len(ind_data) * 100:.1f}%")
        print(f"📊 Average IND Score: {ind_data['IND_Score'].mean():.1f}")
        
        print("\nRisk Level Distribution:")
        print(ind_data['Risk_Level'].value_counts())
        
        print("\nGate Pass Rates:")
        for gate in ['Gate_A_PASS', 'Gate_B_PASS', 'Gate_C_PASS', 'Gate_D_PASS', 'Gate_E_PASS']:
            print(f"  {gate}: {ind_data[gate].sum() / len(ind_data) * 100:.1f}%")
    
    else:
        print("❌ No candidate molecules found. Run denovo_ui.py first.")


if __name__ == "__main__":
    main()
