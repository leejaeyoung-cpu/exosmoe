"""
NOVA Therapeutic Efficacy Predictor
AI-based CKD Treatment Effect Simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List

class CKDEfficacyPredictor:
    """
    CKD 치료 효과 예측 시뮬레이터
    
    Based on:
    - ALK5, TAK1, IKKβ inhibition
    - UUO model endpoints
    - Literature-based dose-response
    """
    
    def __init__(self, seed=42):
        np.random.seed(seed)
    
    def predict_fibrosis_reduction(self, 
                                   alk5_ic50_nM: float,
                                   dose_mg_kg: float,
                                   duration_days: int = 14) -> Dict:
        """
        섬유화 감소 예측
        
        Parameters:
        - alk5_ic50_nM: ALK5 IC50 (nM)
        - dose_mg_kg: 투여 용량 (mg/kg)
        - duration_days: 투여 기간 (일)
        
        Returns:
        - Fibrosis area reduction %
        - COL1A1 gene reduction %
        - α-SMA reduction %
        """
        
        # Exposure estimation (simplified PK model)
        # Assume: Bioavailability ~50%, Cmax/dose ~100 ng/mL per mg/kg
        cmax_ng_mL = dose_mg_kg * 100 * 0.5
        
        # Convert to nM (MW ~314 Da)
        mw = 314
        cmax_nM = cmax_ng_mL / mw * 1000
        
        # Occupancy (simple Emax model)
        # Occupancy = Cmax / (IC50 + Cmax)
        occupancy = cmax_nM / (alk5_ic50_nM + cmax_nM)
        
        # Efficacy (occupancy-dependent)
        # Literature: ALK5 inhibitor → 30-60% fibrosis reduction in UUO
        max_fibrosis_reduction = 60  # %
        fibrosis_reduction = max_fibrosis_reduction * occupancy
        
        # Duration effect (saturation after 14 days)
        duration_factor = min(duration_days / 14, 1.0)
        fibrosis_reduction *= duration_factor
        
        # Gene expression
        # COL1A1, FN1, ACTA2 typically ~40-70% reduction
        col1a1_reduction = fibrosis_reduction * np.random.uniform(0.8, 1.2)
        fn1_reduction = fibrosis_reduction * np.random.uniform(0.7, 1.1)
        acta2_reduction = fibrosis_reduction * np.random.uniform(0.6, 1.0)
        
        # Add noise
        fibrosis_reduction += np.random.normal(0, 5)
        fibrosis_reduction = np.clip(fibrosis_reduction, 0, 80)
        
        return {
            'fibrosis_area_reduction': fibrosis_reduction,
            'col1a1_reduction': np.clip(col1a1_reduction, 0, 90),
            'fn1_reduction': np.clip(fn1_reduction, 0, 90),
            'acta2_reduction': np.clip(acta2_reduction, 0, 90),
            'occupancy': occupancy * 100,
            'cmax_nM': cmax_nM
        }
    
    def predict_inflammation_reduction(self,
                                       tak1_ic50_nM: float,
                                       ikkb_ic50_nM: float,
                                       dose_mg_kg: float) -> Dict:
        """
        염증 감소 예측
        
        Based on TAK1/IKKβ inhibition → NF-κB suppression
        """
        
        cmax_ng_mL = dose_mg_kg * 100 * 0.5
        mw = 314
        cmax_nM = cmax_ng_mL / mw * 1000
        
        # Dual occupancy (TAK1 + IKKβ)
        tak1_occupancy = cmax_nM / (tak1_ic50_nM + cmax_nM)
        ikkb_occupancy = cmax_nM / (ikkb_ic50_nM + cmax_nM)
        
        # Synergy (dual pathway → stronger effect)
        combined_occupancy = (tak1_occupancy + ikkb_occupancy) / 2
        synergy_factor = 1.2  # 20% synergy
        
        effective_occupancy = min(combined_occupancy * synergy_factor, 1.0)
        
        # Inflammation markers
        # Literature: NF-κB inhibitor → 40-70% reduction in CCL2, IL-6
        max_inflammation_reduction = 70
        inflammation_reduction = max_inflammation_reduction * effective_occupancy
        
        ccl2_reduction = inflammation_reduction * np.random.uniform(0.9, 1.1)
        il6_reduction = inflammation_reduction * np.random.uniform(0.8, 1.0)
        icam1_reduction = inflammation_reduction * np.random.uniform(0.7, 0.9)
        
        # Macrophage infiltration (F4/80)
        f480_reduction = inflammation_reduction * np.random.uniform(0.6, 0.8)
        
        return {
            'inflammation_reduction': np.clip(inflammation_reduction, 0, 80),
            'ccl2_reduction': np.clip(ccl2_reduction, 0, 90),
            'il6_reduction': np.clip(il6_reduction, 0, 90),
            'icam1_reduction': np.clip(icam1_reduction, 0, 90),
            'f480_reduction': np.clip(f480_reduction, 0, 80),
            'tak1_occupancy': tak1_occupancy * 100,
            'ikkb_occupancy': ikkb_occupancy * 100
        }
    
    def predict_renal_function_improvement(self,
                                          fibrosis_reduction: float,
                                          inflammation_reduction: float) -> Dict:
        """
        신장 기능 개선 예측
        
        Based on fibrosis + inflammation reduction
        """
        
        # Creatinine reduction (inversely related to fibrosis/inflammation)
        # UUO baseline Cr ~2.0 mg/dL (vs Sham ~0.5)
        baseline_cr = 2.0
        sham_cr = 0.5
        
        # Improvement proportional to (fibrosis + inflammation) / 2
        combined_reduction = (fibrosis_reduction + inflammation_reduction) / 2
        
        cr_reduction_percent = combined_reduction * 0.6  # ~60% correlation
        cr_improvement = (baseline_cr - sham_cr) * cr_reduction_percent / 100
        
        final_cr = baseline_cr - cr_improvement
        
        # BUN (similar pattern)
        baseline_bun = 80  # mg/dL
        sham_bun = 25
        bun_reduction_percent = combined_reduction * 0.55
        bun_improvement = (baseline_bun - sham_bun) * bun_reduction_percent / 100
        final_bun = baseline_bun - bun_improvement
        
        return {
            'baseline_creatinine': baseline_cr,
            'final_creatinine': final_cr,
            'creatinine_improvement': cr_improvement,
            'creatinine_reduction_percent': cr_reduction_percent,
            'baseline_bun': baseline_bun,
            'final_bun': final_bun,
            'bun_improvement': bun_improvement
        }
    
    def generate_time_course(self,
                            alk5_ic50_nM: float,
                            tak1_ic50_nM: float,
                            ikkb_ic50_nM: float,
                            dose_mg_kg: float,
                            duration_days: int = 14) -> pd.DataFrame:
        """
        Time-course simulation (0 → 14 days)
        
        Returns DataFrame with daily progression
        """
        
        timepoints = np.arange(0, duration_days + 1)
        results = []
        
        for day in timepoints:
            # Fibrosis (accumulates over time)
            fib = self.predict_fibrosis_reduction(alk5_ic50_nM, dose_mg_kg, day)
            
            # Inflammation (faster response)
            inf = self.predict_inflammation_reduction(tak1_ic50_nM, ikkb_ic50_nM, dose_mg_kg)
            
            # Scale by day (inflammation peaks at day 3-5, fibrosis at day 10-14)
            inflammation_day_factor = min(day / 5, 1.0)
            fibrosis_day_factor = min(day / 12, 1.0)
            
            # Function (lagging indicator)
            function_day_factor = min(day / 14, 1.0)
            
            fib_scaled = fib['fibrosis_area_reduction'] * fibrosis_day_factor
            inf_scaled = inf['inflammation_reduction'] * inflammation_day_factor
            
            func = self.predict_renal_function_improvement(fib_scaled, inf_scaled)
            
            results.append({
                'day': day,
                'fibrosis_reduction': fib_scaled,
                'inflammation_reduction': inf_scaled,
                'creatinine': func['final_creatinine'],
                'col1a1_reduction': fib['col1a1_reduction'] * fibrosis_day_factor,
                'ccl2_reduction': inf['ccl2_reduction'] * inflammation_day_factor,
                'acta2_reduction': fib['acta2_reduction'] * fibrosis_day_factor
            })
        
        return pd.DataFrame(results)
    
    def predict_full_efficacy(self,
                             alk5_ic50_nM: float,
                             tak1_ic50_nM: float,
                             ikkb_ic50_nM: float,
                             dose_mg_kg: float = 30,
                             duration_days: int = 14) -> Dict:
        """
        종합 치료 효과 예측
        """
        
        # Fibrosis
        fib = self.predict_fibrosis_reduction(alk5_ic50_nM, dose_mg_kg, duration_days)
        
        # Inflammation
        inf = self.predict_inflammation_reduction(tak1_ic50_nM, ikkb_ic50_nM, dose_mg_kg)
        
        # Renal function
        func = self.predict_renal_function_improvement(
            fib['fibrosis_area_reduction'],
            inf['inflammation_reduction']
        )
        
        # Time-course
        tc = self.generate_time_course(alk5_ic50_nM, tak1_ic50_nM, ikkb_ic50_nM, dose_mg_kg, duration_days)
        
        # Overall efficacy score (0-100)
        efficacy_score = (
            fib['fibrosis_area_reduction'] * 0.4 +
            inf['inflammation_reduction'] * 0.3 +
            func['creatinine_reduction_percent'] * 0.3
        )
        
        # Success criteria (literature-based)
        # Pirfenidone (standard): ~30% fibrosis reduction
        success_threshold = 30
        success = fib['fibrosis_area_reduction'] >= success_threshold
        
        return {
            'fibrosis': fib,
            'inflammation': inf,
            'renal_function': func,
            'time_course': tc,
            'efficacy_score': efficacy_score,
            'success': success,
            'success_threshold': success_threshold,
            'dose_mg_kg': dose_mg_kg,
            'duration_days': duration_days
        }


def simulate_dose_response_curve(alk5_ic50_nM: float,
                                 tak1_ic50_nM: float,
                                 ikkb_ic50_nM: float) -> pd.DataFrame:
    """
    Dose-response curve (1-100 mg/kg)
    """
    
    predictor = CKDEfficacyPredictor()
    doses = [1, 3, 10, 30, 100]
    results = []
    
    for dose in doses:
        efficacy = predictor.predict_full_efficacy(
            alk5_ic50_nM, tak1_ic50_nM, ikkb_ic50_nM, dose_mg_kg=dose
        )
        
        results.append({
            'dose_mg_kg': dose,
            'fibrosis_reduction': efficacy['fibrosis']['fibrosis_area_reduction'],
            'inflammation_reduction': efficacy['inflammation']['inflammation_reduction'],
            'creatinine_improvement': efficacy['renal_function']['creatinine_improvement'],
            'efficacy_score': efficacy['efficacy_score']
        })
    
    return pd.DataFrame(results)


def main():
    """메인 실행 (테스트)"""
    print("🏥 NOVA Therapeutic Efficacy Predictor\n")
    
    # Candidate 1 parameters (ALK5 82 nM, TAK1 145 nM, IKKβ 99 nM)
    alk5_ic50 = 82
    tak1_ic50 = 145
    ikkb_ic50 = 99
    
    predictor = CKDEfficacyPredictor()
    
    print("Simulating Candidate 1 @ 30 mg/kg, 14 days...\n")
    
    efficacy = predictor.predict_full_efficacy(alk5_ic50, tak1_ic50, ikkb_ic50, dose_mg_kg=30)
    
    print("📊 Fibrosis:")
    print(f"  Area Reduction: {efficacy['fibrosis']['fibrosis_area_reduction']:.1f}%")
    print(f"  COL1A1 ↓: {efficacy['fibrosis']['col1a1_reduction']:.1f}%")
    print(f"  α-SMA ↓: {efficacy['fibrosis']['acta2_reduction']:.1f}%")
    
    print("\n🔥 Inflammation:")
    print(f"  Reduction: {efficacy['inflammation']['inflammation_reduction']:.1f}%")
    print(f"  CCL2 ↓: {efficacy['inflammation']['ccl2_reduction']:.1f}%")
    print(f"  IL-6 ↓: {efficacy['inflammation']['il6_reduction']:.1f}%")
    
    print("\n🩺 Renal Function:")
    print(f"  Baseline Cr: {efficacy['renal_function']['baseline_creatinine']:.2f} mg/dL")
    print(f"  Final Cr: {efficacy['renal_function']['final_creatinine']:.2f} mg/dL")
    print(f"  Improvement: {efficacy['renal_function']['creatinine_improvement']:.2f} mg/dL")
    
    print(f"\n✅ Overall Efficacy Score: {efficacy['efficacy_score']:.1f}/100")
    print(f"✅ Success (≥30% fibrosis reduction): {'YES' if efficacy['success'] else 'NO'}")
    
    print("\n📈 Dose-Response Curve:")
    dr_curve = simulate_dose_response_curve(alk5_ic50, tak1_ic50, ikkb_ic50)
    print(dr_curve)


if __name__ == "__main__":
    main()
