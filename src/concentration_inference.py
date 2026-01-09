"""
엑소좀 치료제 농도 추론 모듈
CKD-CVD miRNA 칵테일의 최적 농도를 추론하는 AI 시스템
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class PharmacokineticsParams:
    """약동학 파라미터"""
    half_life: float = 24.0  # 반감기 (시간)
    volume_distribution: float = 0.1  # 분포 용적 (L/kg)
    clearance: float = 0.05  # 청소율 (L/hr/kg)
    bioavailability: Dict[str, float] = None  # 투여 경로별 생체이용률
    
    def __post_init__(self):
        if self.bioavailability is None:
            self.bioavailability = {
                'IV': 1.0,  # 정맥주사
                'IP': 0.8,  # 복강주사
                'SC': 0.6   # 피하주사
            }


@dataclass
class DoseResponseParams:
    """용량-반응 파라미터"""
    emax: float = 100.0  # 최대 효과 (%)
    ec50: float = 1e10   # 반수효과농도 (particles/mL)
    hill_coefficient: float = 1.0  # Hill 계수
    baseline: float = 0.0  # 기저 반응


class ConcentrationInferenceModel:
    """
    엑소좀 치료제 농도 추론 모델
    
    miRNA의 Fold Change, 경로 커버리지, 가중치 점수를 바탕으로
    최적의 엑소좀 농도, 투여량, 투여 프로토콜을 계산합니다.
    """
    
    def __init__(self, mirna_data: pd.DataFrame, weights: Dict[str, float]):
        """
        Args:
            mirna_data: miRNA 선별 결과 (컬럼: miRNA, FC_MT_vs_Con, total_pathways, weighted_score 등)
            weights: 6개 카테고리 가중치 {'inflam': 0.25, 'fib': 0.25, ...}
        """
        self.data = mirna_data.set_index('miRNA') if 'miRNA' in mirna_data.columns else mirna_data
        self.weights = weights
        self.pk_params = PharmacokineticsParams()
        
        # 표준 범위 설정
        self.exosome_conc_range = (1e8, 1e12)  # particles/mL
        self.mirna_loading_efficiency = 0.2  # 20% 기본값
        
    def estimate_base_concentration(self, mirna_name: str) -> float:
        """
        개별 miRNA의 기준 농도 추정
        
        FC, 경로 커버리지, 가중치 점수를 통합하여 계산
        높은 점수 → 낮은 농도로도 효과적
        
        Args:
            mirna_name: miRNA 이름 (예: hsa-miR-4739)
            
        Returns:
            기준 엑소좀 농도 (particles/mL)
        """
        if mirna_name not in self.data.index:
            raise ValueError(f"miRNA '{mirna_name}' not found in data")
        
        fc = self.data.loc[mirna_name, 'FC_MT_vs_Con']
        pathways = self.data.loc[mirna_name, 'total_pathways']
        score = self.data.loc[mirna_name, 'weighted_score']
        
        # FC 정규화 (log scale, 범위: 1-250)
        fc_normalized = np.log10(fc + 1) / np.log10(250)
        
        # 경로 커버리지 정규화 (최대값 기준)
        max_pathways = self.data['total_pathways'].max()
        path_normalized = pathways / max_pathways
        
        # 가중치 점수 정규화
        max_score = self.data['weighted_score'].max()
        score_normalized = score / max_score
        
        # 통합 효능 지수 (0~1, 높을수록 강력)
        # FC: 40%, 경로: 30%, 점수: 30%
        efficacy_index = (0.4 * fc_normalized + 
                         0.3 * path_normalized + 
                         0.3 * score_normalized)
        
        # 기준 농도 계산 (역관계: 효능 높으면 낮은 농도)
        # 범위: 1e9 ~ 5e10 particles/mL
        min_conc = 1e9
        max_conc = 5e10
        base_conc = max_conc - (max_conc - min_conc) * efficacy_index
        
        return base_conc
    
    def calculate_ec50(self, mirna_name: str, safety_factor: float = 2.0) -> float:
        """
        반수효과농도(EC50) 계산
        
        EC50은 50% 치료 효과를 내는 농도
        기준 농도의 일정 비율로 추정
        
        Args:
            mirna_name: miRNA 이름
            safety_factor: 안전 계수 (기본 2.0, 실제 효과 농도를 높게 추정)
            
        Returns:
            EC50 (particles/mL)
        """
        base_conc = self.estimate_base_concentration(mirna_name)
        
        # EC50은 기준 농도보다 약간 낮게 설정
        # safety_factor를 나누면 더 낮은 농도에서 효과
        ec50 = base_conc / safety_factor
        
        return ec50
    
    def hill_equation(self, 
                     concentration: np.ndarray, 
                     emax: float, 
                     ec50: float, 
                     hill_coef: float = 1.0,
                     baseline: float = 0.0) -> np.ndarray:
        """
        Hill equation for dose-response curve
        
        Response = baseline + (Emax × C^n) / (EC50^n + C^n)
        
        Args:
            concentration: 농도 배열
            emax: 최대 효과
            ec50: 반수효과농도
            hill_coef: Hill 계수 (기울기, 기본 1.0)
            baseline: 기저 반응
            
        Returns:
            반응 값 배열
        """
        response = baseline + (emax * concentration**hill_coef) / \
                   (ec50**hill_coef + concentration**hill_coef)
        return response
    
    def simulate_dose_response(self,
                              mirna_name: str,
                              conc_range: Optional[Tuple[float, float]] = None,
                              n_points: int = 100) -> pd.DataFrame:
        """
        개별 miRNA의 농도-반응 곡선 시뮬레이션
        
        Args:
            mirna_name: miRNA 이름
            conc_range: 농도 범위 (min, max), None이면 기본 범위 사용
            n_points: 데이터 포인트 수
            
        Returns:
            DataFrame (컬럼: concentration, response)
        """
        if conc_range is None:
            conc_range = self.exosome_conc_range
        
        # 로그 스케일로 농도 생성
        concentrations = np.logspace(
            np.log10(conc_range[0]), 
            np.log10(conc_range[1]), 
            n_points
        )
        
        ec50 = self.calculate_ec50(mirna_name)
        
        # Hill equation으로 반응 계산
        responses = self.hill_equation(
            concentrations, 
            emax=100.0,  # 100% 최대 효과
            ec50=ec50,
            hill_coef=1.0
        )
        
        return pd.DataFrame({
            'concentration': concentrations,
            'response': responses,
            'miRNA': mirna_name
        })
    
    def optimize_combination_ratio(self, 
                                   mirna_list: List[str],
                                   target_efficacy: float = 70.0) -> Dict[str, float]:
        """
        miRNA 칵테일의 최적 혼합 비율 계산
        
        전략: 각 miRNA의 EC50 기반으로 비율 설정
        EC50이 낮은 miRNA는 적은 비율로도 효과적
        
        Args:
            mirna_list: miRNA 이름 리스트
            target_efficacy: 목표 효능 (%, 기본 70%)
            
        Returns:
            miRNA별 혼합 비율 딕셔너리 (합: 1.0)
        """
        ec50_values = {m: self.calculate_ec50(m) for m in mirna_list}
        
        # 역수를 취하면 EC50이 낮을수록 높은 가중치
        inverse_ec50 = {m: 1.0/ec50 for m, ec50 in ec50_values.items()}
        
        # 정규화하여 합이 1.0이 되도록
        total = sum(inverse_ec50.values())
        ratios = {m: inv_ec50/total for m, inv_ec50 in inverse_ec50.items()}
        
        return ratios
    
    def calculate_synergy_index(self,
                                mirna1: str,
                                mirna2: str,
                                concentration: float,
                                ratio: Dict[str, float]) -> float:
        """
        두 miRNA 조합의 시너지 효과 계산
        
        Combination Index (CI):
        CI < 1: 시너지 (synergism)
        CI = 1: 상가 효과 (additive)
        CI > 1: 길항 효과 (antagonism)
        
        Args:
            mirna1, mirna2: miRNA 이름
            concentration: 총 엑소좀 농도
            ratio: 혼합 비율
            
        Returns:
            Combination Index
        """
        ec50_1 = self.calculate_ec50(mirna1)
        ec50_2 = self.calculate_ec50(mirna2)
        
        # 각 miRNA의 실제 농도
        conc_1 = concentration * ratio[mirna1]
        conc_2 = concentration * ratio[mirna2]
        
        # Combination Index 계산 (Chou-Talalay method 간소화 버전)
        ci = (conc_1 / ec50_1) + (conc_2 / ec50_2)
        
        # 이상적인 시너지는 CI < 0.7
        return ci
    
    def generate_dosing_protocol(self,
                                mirna_list: List[str],
                                patient_weight: float = 70.0,
                                route: str = 'IV',
                                target_efficacy: float = 70.0,
                                treatment_duration_days: int = 28) -> Dict:
        """
        투여 프로토콜 생성
        
        Args:
            mirna_list: 사용할 miRNA 리스트
            patient_weight: 환자 체중 (kg)
            route: 투여 경로 ('IV', 'IP', 'SC')
            target_efficacy: 목표 효능 (%)
            treatment_duration_days: 치료 기간 (일)
            
        Returns:
            투여 프로토콜 딕셔너리
        """
        # 최적 혼합 비율 계산
        ratios = self.optimize_combination_ratio(mirna_list, target_efficacy)
        
        # 평균 EC50 계산 (가중 평균)
        weighted_ec50 = sum(
            self.calculate_ec50(m) * ratios[m] 
            for m in mirna_list
        )
        
        # 목표 농도 계산 (목표 효능을 위한 농도)
        # Hill equation 역산: C = EC50 × (E/(Emax-E))^(1/n)
        # n=1, Emax=100 가정
        target_conc = weighted_ec50 * (target_efficacy / (100 - target_efficacy))
        
        # 생체이용률 보정
        bioavailability = self.pk_params.bioavailability[route]
        adjusted_conc = target_conc / bioavailability
        
        # 투여 용적 (mL/kg)
        dose_volume = 5.0  # 5 mL/kg (표준)
        
        # 총 투여량 (particles)
        total_particles = adjusted_conc * dose_volume * patient_weight
        
        # 투여 빈도 계산 (반감기 기반)
        # 반감기가 24시간이면 1일 1회
        dosing_interval_hours = self.pk_params.half_life
        doses_per_day = 24 / dosing_interval_hours
        
        # 총 투여 횟수
        total_doses = int(treatment_duration_days * doses_per_day)
        
        protocol = {
            'miRNA_composition': ratios,
            'target_concentration_particles_per_mL': target_conc,
            'adjusted_concentration_particles_per_mL': adjusted_conc,
            'dose_volume_mL_per_kg': dose_volume,
            'total_dose_particles': total_particles,
            'dose_per_administration_particles': total_particles,
            'patient_weight_kg': patient_weight,
            'route': route,
            'bioavailability': bioavailability,
            'dosing_interval_hours': dosing_interval_hours,
            'doses_per_day': doses_per_day,
            'treatment_duration_days': treatment_duration_days,
            'total_doses': total_doses,
            'individual_mirna_doses': {
                m: total_particles * ratio 
                for m, ratio in ratios.items()
            }
        }
        
        return protocol
    
    def calculate_therapeutic_index(self,
                                    mirna_name: str,
                                    ed50_factor: float = 1.0,
                                    td50_factor: float = 10.0) -> Dict:
        """
        치료 지수(Therapeutic Index) 계산
        
        TI = TD50 / ED50
        TD50: 50% 독성 용량
        ED50: 50% 효과 용량
        
        TI > 10: 안전한 약물
        TI < 3: 위험한 약물
        
        Args:
            mirna_name: miRNA 이름
            ed50_factor: ED50 계산 계수 (EC50 대비)
            td50_factor: TD50 계산 계수 (EC50 대비)
            
        Returns:
            치료 지수 정보 딕셔너리
        """
        ec50 = self.calculate_ec50(mirna_name)
        
        # ED50 ≈ EC50 (비슷하다고 가정)
        ed50 = ec50 * ed50_factor
        
        # TD50은 EC50의 10배로 추정 (안전 마진)
        td50 = ec50 * td50_factor
        
        ti = td50 / ed50
        
        return {
            'miRNA': mirna_name,
            'EC50_particles_per_mL': ec50,
            'ED50_particles_per_mL': ed50,
            'TD50_particles_per_mL': td50,
            'therapeutic_index': ti,
            'safety_assessment': 'Safe' if ti > 10 else ('Moderate' if ti > 3 else 'Risky')
        }
    
    def simulate_pk_profile(self,
                           dose_particles: float,
                           patient_weight: float,
                           route: str = 'IV',
                           time_hours: np.ndarray = None) -> pd.DataFrame:
        """
        약동학 프로파일 시뮬레이션 (단순 1-compartment 모델)
        
        C(t) = (Dose/Vd) × e^(-k×t)
        k = ln(2) / t_half
        
        Args:
            dose_particles: 투여 입자 수
            patient_weight: 환자 체중 (kg)
            route: 투여 경로
            time_hours: 시간 배열 (시간), None이면 0-72시간
            
        Returns:
            DataFrame (컬럼: time_hours, concentration_particles_per_mL)
        """
        if time_hours is None:
            time_hours = np.linspace(0, 72, 100)
        
        # 파라미터
        vd = self.pk_params.volume_distribution * patient_weight * 1000  # mL
        k = np.log(2) / self.pk_params.half_life  # 제거 속도 상수
        bioavailability = self.pk_params.bioavailability[route]
        
        # 초기 농도
        c0 = (dose_particles * bioavailability) / vd
        
        # 농도-시간 곡선
        concentrations = c0 * np.exp(-k * time_hours)
        
        return pd.DataFrame({
            'time_hours': time_hours,
            'concentration_particles_per_mL': concentrations
        })


def format_scientific(value: float) -> str:
    """과학적 표기법으로 포맷팅"""
    return f"{value:.2e}"


def particles_to_mass(particles: float, 
                     particles_per_μg: float = 1e9) -> float:
    """
    입자 수를 질량으로 변환
    
    Args:
        particles: 입자 수
        particles_per_μg: μg 당 입자 수 (기본 1e9)
        
    Returns:
        질량 (μg)
    """
    return particles / particles_per_μg
