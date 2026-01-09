"""
NOVA Lead Validation - ML 합성 데이터 생성기
실험 프로토콜 기반 realistic synthetic data 생성
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
from rdkit.Chem import rdMolDescriptors
import random
from pathlib import Path
from typing import Dict, List, Tuple

class ExperimentalDataGenerator:
    """실험 데이터 시뮬레이터"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_reporter_assay_data(self, mol_features: Dict, assay_type: str) -> Dict:
        """
        Reporter assay (CAGA-luc, NF-κB-luc) 데이터 생성
        
        Parameters:
        - mol_features: 분자 특성 (MW, LogP, TPSA, etc.)
        - assay_type: 'CAGA' or 'NF-kB'
        
        Returns:
        - IC50, curve_data, quality_score
        """
        # Quinazoline core가 있으면 유리
        has_quinazoline = mol_features.get('has_quinazoline', False)
        has_amide = mol_features.get('has_amide', False)
        
        # 기본 IC50 예측 (LogP, MW, TPSA 기반)
        logp = mol_features['logp']
        mw = mol_features['mw']
        tpsa = mol_features['tpsa']
        
        # Optimal range: LogP 2-4, MW 250-400, TPSA 40-90
        logp_factor = 1.0 if 2 <= logp <= 4 else 1.5
        mw_factor = 1.0 if 250 <= mw <= 400 else 1.3
        tpsa_factor = 1.0 if 40 <= tpsa <= 90 else 1.4
        
        # Quinazoline + Amide → Kinase inhibitor 유리
        scaffold_bonus = 0.5 if (has_quinazoline and has_amide) else 1.0
        
        # Base IC50 (nM)
        base_ic50 = np.random.lognormal(mean=6.0, sigma=0.8)  # ~400 nM median
        ic50 = base_ic50 * logp_factor * mw_factor * tpsa_factor * scaffold_bonus
        
        # Assay-specific adjustment
        if assay_type == 'CAGA':
            ic50 *= np.random.uniform(0.8, 1.2)  # TGF-β pathway
        elif assay_type == 'NF-kB':
            ic50 *= np.random.uniform(0.7, 1.3)  # NF-κB pathway
        
        # Dose-response curve (8-point)
        concentrations = [0.1, 0.3, 1, 3, 10, 30, 100, 300]  # μM
        responses = []
        
        for conc in concentrations:
            # Hill equation: Response = 100 / (1 + (IC50/conc)^Hill)
            hill_slope = np.random.uniform(0.8, 1.2)
            response = 100 / (1 + (ic50 / (conc * 1000))**hill_slope)  # nM 변환
            
            # Noise 추가 (실험 오차)
            noise = np.random.normal(0, 5)
            response = np.clip(response + noise, 0, 100)
            responses.append(response)
        
        # Quality score (R² equivalent)
        quality = np.random.uniform(0.85, 0.98)
        
        return {
            'IC50_nM': ic50,
            'IC50_uM': ic50 / 1000,
            'concentrations': concentrations,
            'responses': responses,
            'hill_slope': hill_slope,
            'quality_score': quality,
            'pass_criteria': ic50 < 1000  # < 1 μM
        }
    
    def generate_western_blot_data(self, mol_features: Dict, target: str) -> Dict:
        """
        Western blot 정량 데이터 생성
        
        Parameters:
        - target: 'p-SMAD2/3', 'p-IkBa', 'p-p65'
        """
        # IC50와 상관관계 (낮은 IC50 → 높은 억제)
        logp = mol_features['logp']
        potency_factor = 1.0 if 2.5 <= logp <= 4.0 else 0.7
        
        # Base inhibition at 1 μM
        base_inhibition = np.random.uniform(40, 80) * potency_factor
        
        # Target-specific
        if target == 'p-SMAD2/3':
            inhibition = base_inhibition * np.random.uniform(0.9, 1.1)
        elif target in ['p-IkBa', 'p-p65']:
            inhibition = base_inhibition * np.random.uniform(0.85, 1.15)
        
        # Noise
        inhibition = np.clip(inhibition + np.random.normal(0, 5), 0, 95)
        
        return {
            'target': target,
            'inhibition_percent': inhibition,
            'treatment_conc_uM': 1.0,
            'pass_criteria': inhibition >= 50  # ≥50% 억제
        }
    
    def generate_qpcr_data(self, mol_features: Dict, gene_panel: List[str]) -> pd.DataFrame:
        """
        qPCR gene expression 데이터 생성
        
        Parameters:
        - gene_panel: ['COL1A1', 'FN1', 'ACTA2', 'CCL2', 'IL6', ...]
        """
        logp = mol_features['logp']
        potency = 1.0 if 2.5 <= logp <= 4.0 else 0.6
        
        results = []
        for gene in gene_panel:
            # Fold change (vehicle = 1.0)
            vehicle_fc = 1.0
            
            # Stimulated (TGF-β or TNF-α)
            if gene in ['COL1A1', 'FN1', 'ACTA2', 'CTGF']:
                stimulated_fc = np.random.uniform(3.5, 6.0)  # Fibrosis genes 상승
            elif gene in ['CCL2', 'IL6', 'ICAM1', 'VCAM1']:
                stimulated_fc = np.random.uniform(4.0, 7.0)  # Inflammation genes 상승
            else:
                stimulated_fc = np.random.uniform(2.0, 4.0)
            
            # Treatment (NOVA compound)
            # 좋은 compound → fold change를 baseline에 가깝게
            reduction = np.random.uniform(0.4, 0.7) * potency  # 40-70% 감소
            treated_fc = stimulated_fc * (1 - reduction)
            
            # Noise
            treated_fc += np.random.normal(0, 0.2)
            
            percent_reduction = ((stimulated_fc - treated_fc) / (stimulated_fc - vehicle_fc)) * 100
            
            results.append({
                'gene': gene,
                'vehicle_fc': vehicle_fc,
                'stimulated_fc': stimulated_fc,
                'treated_fc': treated_fc,
                'percent_reduction': percent_reduction,
                'pass': percent_reduction >= 40  # ≥40% 감소
            })
        
        return pd.DataFrame(results)
    
    def generate_cytotoxicity_data(self, mol_features: Dict) -> Dict:
        """
        MTT cytotoxicity 데이터 생성
        """
        # LogP 높으면 독성 증가 경향
        logp = mol_features['logp']
        mw = mol_features['mw']
        
        # Base CC50 (μM)
        base_cc50 = np.random.lognormal(mean=3.0, sigma=0.7)  # ~20 μM median
        
        # LogP > 4.5 → 독성 증가
        if logp > 4.5:
            base_cc50 *= 0.5
        elif logp < 2.0:
            base_cc50 *= 1.5
        
        # MW > 450 → 독성 증가 가능성
        if mw > 450:
            base_cc50 *= 0.8
        
        cc50 = base_cc50 + np.random.normal(0, 3)
        cc50 = max(cc50, 1.0)  # Minimum 1 μM
        
        # Selectivity Index (SI = CC50 / IC50)
        # 여기선 IC50 평균을 0.5 μM로 가정
        avg_ic50 = 0.5
        si = cc50 / avg_ic50
        
        return {
            'CC50_uM': cc50,
            'cell_line': 'HK-2',
            'selectivity_index': si,
            'pass_criteria': cc50 > 10  # > 10 μM
        }
    
    def generate_solubility_data(self, mol_features: Dict) -> Dict:
        """
        Kinetic solubility 데이터 생성
        """
        logp = mol_features['logp']
        tpsa = mol_features['tpsa']
        
        # LogP 높으면 용해도 낮음
        # TPSA 높으면 용해도 증가
        base_solubility = np.random.lognormal(mean=3.5, sigma=0.8)  # ~33 μM median
        
        # LogP penalty
        if logp > 4.0:
            base_solubility *= 0.4
        elif logp > 3.5:
            base_solubility *= 0.7
        elif logp < 2.5:
            base_solubility *= 1.5
        
        # TPSA bonus
        if tpsa > 70:
            base_solubility *= 1.3
        elif tpsa < 50:
            base_solubility *= 0.8
        
        solubility = base_solubility + np.random.normal(0, 5)
        solubility = max(solubility, 1.0)
        
        return {
            'solubility_uM': solubility,
            'buffer': 'PBS pH 7.4',
            'temperature_C': 37,
            'pass_criteria': solubility > 30  # > 30 μM
        }
    
    def extract_molecular_features(self, smiles: str) -> Dict:
        """SMILES에서 분자 특성 추출"""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        
        # Quinazoline detection (rough)
        quinazoline_smarts = Chem.MolFromSmarts('c1ncnc2ccccc12')
        has_quinazoline = mol.HasSubstructMatch(quinazoline_smarts) if quinazoline_smarts else False
        
        # Amide detection
        amide_smarts = Chem.MolFromSmarts('C(=O)N')
        has_amide = mol.HasSubstructMatch(amide_smarts) if amide_smarts else False
        
        features = {
            'smiles': smiles,
            'mw': Descriptors.MolWt(mol),
            'logp': Crippen.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'has_quinazoline': has_quinazoline,
            'has_amide': has_amide
        }
        
        return features
    
    def generate_full_experimental_dataset(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        전체 실험 데이터셋 생성
        """
        all_results = []
        
        for smiles in smiles_list:
            mol_features = self.extract_molecular_features(smiles)
            if not mol_features:
                continue
            
            # Generate all assay data
            caga_data = self.generate_reporter_assay_data(mol_features, 'CAGA')
            nfkb_data = self.generate_reporter_assay_data(mol_features, 'NF-kB')
            
            psmad_data = self.generate_western_blot_data(mol_features, 'p-SMAD2/3')
            pikba_data = self.generate_western_blot_data(mol_features, 'p-IkBa')
            
            fibrosis_genes = ['COL1A1', 'FN1', 'ACTA2', 'CTGF']
            inflammation_genes = ['CCL2', 'IL6', 'ICAM1']
            
            qpcr_fib = self.generate_qpcr_data(mol_features, fibrosis_genes)
            qpcr_inf = self.generate_qpcr_data(mol_features, inflammation_genes)
            
            cytotox = self.generate_cytotoxicity_data(mol_features)
            solubility = self.generate_solubility_data(mol_features)
            
            # Aggregate
            result = {
                'smiles': smiles,
                'MW': mol_features['mw'],
                'LogP': mol_features['logp'],
                'TPSA': mol_features['tpsa'],
                
                # Reporter assays
                'CAGA_IC50_uM': caga_data['IC50_uM'],
                'CAGA_pass': caga_data['pass_criteria'],
                'NF-kB_IC50_uM': nfkb_data['IC50_uM'],
                'NF-kB_pass': nfkb_data['pass_criteria'],
                
                # Western blot
                'pSMAD_inhibition': psmad_data['inhibition_percent'],
                'pSMAD_pass': psmad_data['pass_criteria'],
                'pIkBa_inhibition': pikba_data['inhibition_percent'],
                'pIkBa_pass': pikba_data['pass_criteria'],
                
                # qPCR (average reduction)
                'fibrosis_genes_avg_reduction': qpcr_fib['percent_reduction'].mean(),
                'inflammation_genes_avg_reduction': qpcr_inf['percent_reduction'].mean(),
                'qPCR_pass': (qpcr_fib['pass'].sum() + qpcr_inf['pass'].sum()) >= 4,
                
                # Cytotox & Solubility
                'CC50_uM': cytotox['CC50_uM'],
                'SI': cytotox['selectivity_index'],
                'cytotox_pass': cytotox['pass_criteria'],
                'solubility_uM': solubility['solubility_uM'],
                'solubility_pass': solubility['pass_criteria'],
            }
            
            # Overall GO/NO-GO
            gate1_pass = (
                result['CAGA_pass'] and
                result['NF-kB_pass'] and
                result['pSMAD_pass'] and
                result['pIkBa_pass'] and
                result['cytotox_pass'] and
                result['solubility_pass']
            )
            
            result['Gate1_GO'] = gate1_pass
            
            all_results.append(result)
        
        return pd.DataFrame(all_results)


def main():
    """메인 실행"""
    # Load NOVA candidates
    csv_path = Path("generated_molecules/latest_candidates.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        smiles_list = df['smiles'].tolist()[:50]  # Top 50
        
        print(f"📊 Generating synthetic experimental data for {len(smiles_list)} molecules...")
        
        generator = ExperimentalDataGenerator(seed=42)
        experimental_data = generator.generate_full_experimental_dataset(smiles_list)
        
        # Save
        output_path = Path("generated_molecules/synthetic_experimental_data.csv")
        experimental_data.to_csv(output_path, index=False)
        
        print(f"✅ Saved to: {output_path}")
        print(f"📈 Gate 1 GO rate: {experimental_data['Gate1_GO'].sum() / len(experimental_data) * 100:.1f}%")
        print("\nSample preview:")
        print(experimental_data[['smiles', 'CAGA_IC50_uM', 'NF-kB_IC50_uM', 'Gate1_GO']].head(10))
    
    else:
        print("❌ No candidate molecules found. Run denovo_ui.py first.")


if __name__ == "__main__":
    main()
