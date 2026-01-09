"""
AI-Driven Drug Discovery Pipeline for CKD-CVD
Phase 2: Target Protein 3D Structure and Molecular Docking

AlphaFold2 또는 PDB에서 단백질 구조 획득 및 분자 도킹 시뮬레이션
"""

import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json

class ProteinStructureManager:
    """
    단백질 3D 구조 관리
    """
    
    def __init__(self):
        self.output_dir = Path("data/protein_structures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CKD-CVD 핵심 타겟
        self.targets = {
            'NF-kB_p65': {
                'uniprot': 'Q04206',
                'pdb_ids': ['1NFI', '1VKX', '1LE5'],
                'function': 'Inflammatory transcription factor',
                'druggability': 0.82
            },
            'TGF-beta_R1': {
                'uniprot': 'P36897',
                'pdb_ids': ['1PY5', '3FAA', '2WOT'],
                'function': 'Fibrosis receptor kinase',
                'druggability': 0.91
            },
            'NOX4': {
                'uniprot': 'Q9NPH5',
                'pdb_ids': [],  # homology model needed
                'function': 'ROS generator',
                'druggability': 0.75
            },
            'VCAM1': {
                'uniprot': 'P19320',
                'pdb_ids': ['1VSC'],
                'function': 'Endothelial adhesion molecule',
                'druggability': 0.68
            },
            'Cyclophilin_D': {
                'uniprot': 'P30405',
                'pdb_ids': ['2Z6W'],
                'function': 'Mitochondrial permeability',
                'druggability': 0.87
            }
        }
    
    def fetch_pdb_structure(self, pdb_id: str) -> bool:
        """
        PDB에서 구조 파일 다운로드
        
        Args:
            pdb_id: PDB ID (예: '1NFI')
            
        Returns:
            성공 여부
        """
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        output_file = self.output_dir / f"{pdb_id}.pdb"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with open(output_file, 'w') as f:
                f.write(response.text)
            
            print(f"   ✅ {pdb_id}.pdb 다운로드 완료")
            return True
            
        except Exception as e:
            print(f"   ❌ {pdb_id} 다운로드 실패: {e}")
            return False
    
    def get_alphafold_structure(self, uniprot_id: str) -> bool:
        """
        AlphaFold DB에서 예측 구조 다운로드
        
        Args:
            uniprot_id: UniProt ID
            
        Returns:
            성공 여부
        """
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        output_file = self.output_dir / f"AF_{uniprot_id}.pdb"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with open(output_file, 'w') as f:
                f.write(response.text)
            
            print(f"   ✅ AlphaFold {uniprot_id} 다운로드 완료")
            return True
            
        except Exception as e:
            print(f"   ⚠️ AlphaFold {uniprot_id} 다운로드 실패: {e}")
            return False
    
    def prepare_all_structures(self):
        """
        모든 타겟 단백질 구조 준비
        """
        print("\n" + "="*70)
        print("단백질 3D 구조 다운로드")
        print("="*70)
        
        for target_name, info in self.targets.items():
            print(f"\n🎯 {target_name}:")
            
            # PDB 구조 우선
            success = False
            if info['pdb_ids']:
                for pdb_id in info['pdb_ids']:
                    if self.fetch_pdb_structure(pdb_id):
                        success = True
                        break
            
            # PDB 실패시 AlphaFold 사용
            if not success:
                print(f"   → AlphaFold 예측 구조 시도...")
                self.get_alphafold_structure(info['uniprot'])


class MolecularDockingSimulator:
    """
    분자 도킹 시뮬레이션 (간소화 버전)
    
    실제 구현시 AutoDock Vina 또는 Schrödinger 사용 권장
    여기서는 개념적 프레임워크 제공
    """
    
    def __init__(self):
        self.structures_dir = Path("data/protein_structures")
        self.results_dir = Path("data/docking_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def define_binding_sites(self) -> Dict:
        """
        각 타겟의 결합 부위 정의
        
        Returns:
            타겟별 binding pocket 좌표
        """
        binding_sites = {
            'NF-kB_p65': {
                'center': [10.5, -5.2, 3.8],  # DNA binding domain
                'size': [20, 20, 20],  # Angstroms
                'key_residues': ['Lys221', 'Arg246', 'Arg302']
            },
            'TGF-beta_R1': {
                'center': [8.2, 12.1, -4.5],  # ATP binding pocket
                'size': [15, 15, 15],
                'key_residues': ['Lys232', 'Glu245', 'Asp351']
            },
            'Cyclophilin_D': {
                'center': [5.1, -2.8, 6.3],  # Active site
                'size': [18, 18, 18],
                'key_residues': ['Arg55', 'Phe60', 'Trp121']
            }
        }
        
        return binding_sites
    
    def calculate_binding_score(self, molecule_data: Dict, target: str) -> float:
        """
        간소화된 결합 점수 계산
        
        실제로는 AutoDock Vina의 scoring function 사용
        여기서는 개념적 점수만 제공
        
        Args:
            molecule_data: 분자 정보
            target: 타겟 단백질
            
        Returns:
            Binding affinity (kcal/mol, 음수일수록 강함)
        """
        # 실제 도킹 대신 placeholder 점수
        # 실전에서는 Vina, AutoDock, Glide 등 사용
        
        base_score = np.random.uniform(-12.0, -4.0)
        
        # 분자 크기 보정
        if 'mw' in molecule_data:
            if 300 < molecule_data['mw'] < 500:
                base_score -= 1.0  # bonus
        
        # Lipophilicity 보정
        if 'logp' in molecule_data:
            if 2 < molecule_data['logp'] < 4:
                base_score -= 0.5  # bonus
        
        return round(base_score, 2)
    
    def dock_library(self, molecules: List[Dict], target: str) -> pd.DataFrame:
        """
        분자 라이브러리를 타겟에 도킹
        
        Args:
            molecules: 분자 리스트
            target: 타겟 단백질
            
        Returns:
            도킹 결과 DataFrame
        """
        print(f"\n🔬 {target}에 대한 도킹 시뮬레이션...")
        
        results = []
        
        for mol in molecules:
            binding_score = self.calculate_binding_score(mol, target)
            
            results.append({
                'molecule': mol.get('name', 'Unknown'),
                'target': target,
                'binding_affinity': binding_score,
                'estimated_ki': self.affinity_to_ki(binding_score),
                'druggability': mol.get('druggability', 0.5)
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('binding_affinity')  # 낮을수록 좋음
        
        print(f"   ✅ {len(df)}개 분자 도킹 완료")
        print(f"   🏆 최고 결합력: {df.iloc[0]['binding_affinity']} kcal/mol")
        
        return df
    
    @staticmethod
    def affinity_to_ki(affinity_kcal_mol: float, temp_k: float = 298.15) -> float:
        """
        결합 에너지를 Ki(해리 상수)로 변환
        
        ΔG = RT ln(Ki)
        Ki = exp(ΔG / RT)
        
        Args:
            affinity_kcal_mol: 결합 에너지 (kcal/mol)
            temp_k: 온도 (Kelvin)
            
        Returns:
            Ki (M)
        """
        R = 1.987e-3  # kcal/(mol·K)
        ki = np.exp(affinity_kcal_mol / (R * temp_k))
        return ki


class VirtualScreening:
    """
    Virtual Screening 파이프라인
    """
    
    def __init__(self):
        self.docker = MolecularDockingSimulator()
        self.output_dir = Path("data/screening_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_candidate_molecules(self) -> List[Dict]:
        """
        후보 분자 로드
        
        실제로는 ZINC, PubChem에서 수백만개 로드
        여기서는 예시 데이터
        """
        # 기존 알려진 약물 + 새로운 후보
        candidates = [
            {'name': 'Metformin', 'mw': 129.16, 'logp': -1.43, 'druggability': 0.85},
            {'name': 'Bardoxolone', 'mw': 505.7, 'logp': 6.2, 'druggability': 0.78},
            {'name': 'Pirfenidone', 'mw': 185.22, 'logp': 0.5, 'druggability': 0.82},
            {'name': 'Curcumin', 'mw': 368.38, 'logp': 3.2, 'druggability': 0.65},
            {'name': 'Resveratrol', 'mw': 228.25, 'logp': 3.1, 'druggability': 0.70},
            {'name': 'Losartan', 'mw': 422.91, 'logp': 4.3, 'druggability': 0.88},
            {'name': 'NAC', 'mw': 163.20, 'logp': -1.6, 'druggability': 0.60},
            # 신규 후보 (가상)
            {'name': 'Compound-A', 'mw': 385.5, 'logp': 2.8, 'druggability': 0.75},
            {'name': 'Compound-B', 'mw': 412.3, 'logp': 3.5, 'druggability': 0.80},
            {'name': 'Compound-C', 'mw': 328.9, 'logp': 2.1, 'druggability': 0.72},
        ]
        
        return candidates
    
    def run_multi_target_screening(self) -> Dict[str, pd.DataFrame]:
        """
        다중 타겟 스크리닝 실행
        
        Returns:
            타겟별 도킹 결과
        """
        print("\n" + "="*70)
        print("Virtual Screening 시작")
        print("="*70)
        
        molecules = self.load_candidate_molecules()
        print(f"\n💊 총 {len(molecules)}개 후보 분자")
        
        targets = ['NF-kB_p65', 'TGF-beta_R1', 'Cyclophilin_D']
        
        all_results = {}
        
        for target in targets:
            results_df = self.docker.dock_library(molecules, target)
            all_results[target] = results_df
            
            # 저장
            output_file = self.output_dir / f"{target}_docking_results.csv"
            results_df.to_csv(output_file, index=False)
            print(f"   💾 저장: {output_file}")
        
        # 종합 순위
        self.rank_multi_target_hits(all_results)
        
        return all_results
    
    def rank_multi_target_hits(self, results: Dict[str, pd.DataFrame]):
        """
        다중 타겟 결과 종합 순위
        """
        print("\n" + "="*70)
        print("Multi-Target Ranking")
        print("="*70)
        
        # 각 분자의 모든 타겟에 대한 평균 점수
        molecules = set()
        for df in results.values():
            molecules.update(df['molecule'])
        
        rankings = []
        
        for mol in molecules:
            scores = []
            for target, df in results.items():
                mol_data = df[df['molecule'] == mol]
                if not mol_data.empty:
                    scores.append(mol_data['binding_affinity'].values[0])
            
            if scores:
                rankings.append({
                    'molecule': mol,
                    'avg_binding': np.mean(scores),
                    'best_binding': min(scores),
                    'n_targets': len(scores)
                })
        
        rankings = sorted(rankings, key=lambda x: x['avg_binding'])
        
        print(f"\n🏆 Top 5 Multi-Target Hits:\n")
        for i, r in enumerate(rankings[:5], 1):
            print(f"   {i}. {r['molecule']}")
            print(f"      평균 결합력: {r['avg_binding']:.2f} kcal/mol")
            print(f"      최고 결합력: {r['best_binding']:.2f} kcal/mol")
            print(f"      타겟 적중: {r['n_targets']}/3\n")
        
        # 저장
        pd.DataFrame(rankings).to_csv(
            self.output_dir / "multi_target_ranking.csv", 
            index=False
        )


def main():
    """
    Phase 2 메인 실행
    """
    print("\n" + "="*70)
    print("AI 기반 CKD-CVD 신약 발견 파이프라인")
    print("Phase 2: 단백질 구조 및 분자 도킹")
    print("="*70)
    
    # Step 1: 단백질 구조 준비
    struct_mgr = ProteinStructureManager()
    struct_mgr.prepare_all_structures()
    
    # Step 2: Virtual Screening
    screener = VirtualScreening()
    results = screener.run_multi_target_screening()
    
    print("\n" + "="*70)
    print("✅ Phase 2 완료!")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = main()
