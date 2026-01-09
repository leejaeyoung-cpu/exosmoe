"""
NOVA Animal Dosing Schedule Generator
UUO Model 실험 자동 스케줄링
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import plotly.figure_factory as ff
import plotly.graph_objects as go

class UUODosingScheduler:
    """
    UUO 동물 모델 실험 스케줄 생성기
    
    Features:
    - 투약 스케줄 (PO/IV, QD/BID)
    - 체중 측정
    - 혈액/뇨 샘플링
    - Sacrifice timeline
    - Daily checklist
    """
    
    def __init__(self, start_date: str = None):
        if start_date:
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            self.start_date = datetime.now()
    
    def generate_uuo_protocol(self,
                              n_mice_per_group: int = 8,
                              groups: List[str] = None,
                              route: str = 'PO',  # PO or IV
                              frequency: str = 'QD',  # QD (once daily) or BID (twice daily)
                              duration_days: int = 14,
                              doses_mg_kg: List[float] = None) -> Dict:
        """
        UUO 실험 프로토콜 생성
        
        Parameters:
        - n_mice_per_group: 군당 마우스 수
        - groups: 실험 군 (None이면 기본값)
        - route: 투여 경로 (PO/IV)
        - frequency: 투여 빈도 (QD/BID)
        - duration_days: 실험 기간
        - doses_mg_kg: 투여 용량 리스트
        
        Returns:
        - 전체 실험 프로토콜 딕셔너리
        """
        
        # Default groups
        if groups is None:
            if doses_mg_kg is None:
                doses_mg_kg = [10, 30]
            
            groups = [
                f"{dose} mg/kg" for dose in doses_mg_kg
            ]
            groups = ['Sham', 'Vehicle (UUO)', 'Pirfenidone 30 mg/kg'] + groups
        
        total_mice = n_mice_per_group * len(groups)
        
        # Timeline
        day_minus_7 = self.start_date - timedelta(days=7)  # Acclimation start
        day_0 = self.start_date  # UUO surgery
        day_1 = self.start_date + timedelta(days=1)  # First dose
        day_final = self.start_date + timedelta(days=duration_days)  # Sacrifice
        
        protocol = {
            'experiment_info': {
                'title': 'UUO Model - CKD Fibrosis & Inflammation',
                'total_mice': total_mice,
                'n_per_group': n_mice_per_group,
                'groups': groups,
                'n_groups': len(groups),
                'route': route,
                'frequency': frequency,
                'duration_days': duration_days,
                'start_date': self.start_date.strftime('%Y-%m-%d'),
                'end_date': day_final.strftime('%Y-%m-%d')
            },
            'timeline': {
                'acclimation_start': day_minus_7,
                'surgery_day': day_0,
                'first_dose': day_1,
                'sacrifice_day': day_final,
                'total_duration': (day_final - day_minus_7).days
            },
            'groups': self._generate_group_details(groups, n_mice_per_group, doses_mg_kg)
        }
        
        return protocol
    
    def _generate_group_details(self, groups, n_per_group, doses_mg_kg):
        """각 군의 상세 정보"""
        group_details = []
        
        for i, group_name in enumerate(groups):
            if 'Sham' in group_name:
                treatment = 'Sham surgery (no UUO)'
                dose = 0
                formulation = 'N/A'
            elif 'Vehicle' in group_name:
                treatment = 'Vehicle (control)'
                dose = 0
                formulation = 'Saline or vehicle'
            elif 'Pirfenidone' in group_name:
                treatment = 'Positive control'
                dose = 30
                formulation = '0.5% CMC suspension'
            else:
                treatment = 'Test compound (NOVA)'
                dose = doses_mg_kg[i - 3] if doses_mg_kg else 10
                formulation = 'To be determined (suspension or solution)'
            
            group_details.append({
                'group_id': i + 1,
                'group_name': group_name,
                'n_mice': n_per_group,
                'treatment': treatment,
                'dose_mg_kg': dose,
                'formulation': formulation,
                'cage_numbers': f"Cage {i*2+1}-{i*2+2}" if n_per_group > 4 else f"Cage {i+1}"
            })
        
        return group_details
    
    def generate_dosing_schedule(self, protocol: Dict) -> pd.DataFrame:
        """
        일별 투약 스케줄 생성
        """
        
        duration = protocol['experiment_info']['duration_days']
        frequency = protocol['experiment_info']['frequency']
        groups = protocol['experiment_info']['groups']
        
        schedule = []
        
        # Day 0: Surgery
        schedule.append({
            'Day': 0,
            'Date': protocol['timeline']['surgery_day'].strftime('%Y-%m-%d'),
            'Activity': 'UUO Surgery',
            'Groups': 'All (except Sham)',
            'Time': '09:00-12:00',
            'Notes': 'Sham: sham surgery without UUO. Anesthesia: Isoflurane 2%',
            'Responsible': 'Surgeon'
        })
        
        # Day 1 ~ Day N: Dosing
        for day in range(1, duration + 1):
            current_date = protocol['timeline']['surgery_day'] + timedelta(days=day)
            
            if frequency == 'QD':
                schedule.append({
                    'Day': day,
                    'Date': current_date.strftime('%Y-%m-%d'),
                    'Activity': 'Dosing (QD)',
                    'Groups': 'All dosing groups',
                    'Time': '09:00',
                    'Notes': f'Route: {protocol["experiment_info"]["route"]}. Dose volume: 10 mL/kg',
                    'Responsible': 'Technician A'
                })
            else:  # BID
                schedule.append({
                    'Day': day,
                    'Date': current_date.strftime('%Y-%m-%d'),
                    'Activity': 'Dosing (BID - AM)',
                    'Groups': 'All dosing groups',
                    'Time': '09:00',
                    'Notes': f'Route: {protocol["experiment_info"]["route"]}, 1st dose',
                    'Responsible': 'Technician A'
                })
                schedule.append({
                    'Day': day,
                    'Date': current_date.strftime('%Y-%m-%d'),
                    'Activity': 'Dosing (BID - PM)',
                    'Groups': 'All dosing groups',
                    'Time': '17:00',
                    'Notes': f'Route: {protocol["experiment_info"]["route"]}, 2nd dose',
                    'Responsible': 'Technician B'
                })
            
            # Body weight (every 3-4 days)
            if day % 3 == 0 or day == duration:
                schedule.append({
                    'Day': day,
                    'Date': current_date.strftime('%Y-%m-%d'),
                    'Activity': 'Body Weight',
                    'Groups': 'All',
                    'Time': '08:30',
                    'Notes': 'Weigh before dosing. Record in data sheet',
                    'Responsible': 'Technician A'
                })
            
            # Blood sampling (Day 7)
            if day == 7:
                schedule.append({
                    'Day': day,
                    'Date': current_date.strftime('%Y-%m-%d'),
                    'Activity': 'Blood Sampling (interim)',
                    'Groups': 'All',
                    'Time': '10:00',
                    'Notes': 'Retro-orbital, 100 μL. For Cr/BUN analysis',
                    'Responsible': 'Technician B'
                })
        
        # Final day: Sacrifice
        final_date = protocol['timeline']['sacrifice_day']
        
        schedule.append({
            'Day': duration,
            'Date': final_date.strftime('%Y-%m-%d'),
            'Activity': 'Blood Sampling (terminal)',
            'Groups': 'All',
            'Time': '09:00',
            'Notes': 'Cardiac puncture under deep anesthesia. 500-800 μL',
            'Responsible': 'Technician A+B'
        })
        
        schedule.append({
            'Day': duration,
            'Date': final_date.strftime('%Y-%m-%d'),
            'Activity': 'Sacrifice & Tissue Collection',
            'Groups': 'All',
            'Time': '09:30-12:00',
            'Notes': 'Kidney, liver, heart. Fix in 10% formalin. Snap-freeze for RNA/protein',
            'Responsible': 'All team'
        })
        
        return pd.DataFrame(schedule)
    
    def generate_sample_collection_plan(self, protocol: Dict) -> pd.DataFrame:
        """
        샘플 수집 계획
        """
        
        n_per_group = protocol['experiment_info']['n_per_group']
        groups = protocol['experiment_info']['groups']
        
        samples = []
        
        for group_info in protocol['groups']:
            group_name = group_info['group_name']
            n_mice = group_info['n_mice']
            
            for mouse_id in range(1, n_mice + 1):
                samples.append({
                    'Group': group_name,
                    'Mouse_ID': f"{group_info['group_id']}-{mouse_id:02d}",
                    'Blood_Day7': f"Tube_{group_info['group_id']}-{mouse_id:02d}-D7",
                    'Blood_Day14': f"Tube_{group_info['group_id']}-{mouse_id:02d}-D14",
                    'Kidney_Left': f"Kidney_L_{group_info['group_id']}-{mouse_id:02d}",
                    'Kidney_Right': f"Kidney_R_{group_info['group_id']}-{mouse_id:02d}",
                    'Liver': f"Liver_{group_info['group_id']}-{mouse_id:02d}",
                    'Processing': 'L kidney: formalin (histology), R kidney: snap-freeze (RNA/protein)'
                })
        
        return pd.DataFrame(samples)
    
    def generate_daily_checklist(self, protocol: Dict, target_day: int) -> Dict:
        """
        특정 일자의 작업 체크리스트
        """
        
        duration = protocol['experiment_info']['duration_days']
        current_date = protocol['timeline']['surgery_day'] + timedelta(days=target_day)
        
        checklist = {
            'date': current_date.strftime('%Y-%m-%d'),
            'day': target_day,
            'tasks': []
        }
        
        if target_day == 0:
            checklist['tasks'] = [
                '☐ 마우스 체중 측정 및 무작위 배정 확인',
                '☐ 수술 기구 멸균 완료 확인',
                '☐ Isoflurane 마취 시스템 체크',
                '☐ UUO 수술 (왼쪽 요관 결찰)',
                '☐ Sham 수술 (요관 노출만, 결찰 없음)',
                '☐ 수술 후 보온 패드 배치',
                '☐ 진통제 투여 (Buprenorphine 0.05 mg/kg, SC)',
                '☐ 수술 기록지 작성 (시간, 마취 시간, 특이사항)'
            ]
        elif target_day >= 1 and target_day <= duration:
            checklist['tasks'].append('☐ 케이지 체크 (사망/이상 징후 확인)')
            
            if target_day % 3 == 0 or target_day == duration:
                checklist['tasks'].append('☐ 체중 측정 (08:30)')
            
            checklist['tasks'].append(f'☐ 투약 준비 ({protocol["experiment_info"]["route"]}, 09:00)')
            checklist['tasks'].append('☐ Dosing 실시 (각 군 확인, 체중 기반 용량 계산)')
            checklist['tasks'].append('☐ Dosing 기록지 작성')
            
            if protocol['experiment_info']['frequency'] == 'BID':
                checklist['tasks'].append('☐ 2차 투약 (17:00)')
            
            if target_day == 7:
                checklist['tasks'].append('☐ 혈액 샘플링 (retro-orbital, 100 μL)')
                checklist['tasks'].append('☐ 혈청 분리 (3000 rpm, 10 min)')
                checklist['tasks'].append('☐ Cr/BUN 분석 또는 -80°C 보관')
            
            if target_day == duration:
                checklist['tasks'].extend([
                    '',
                    '=== Final Day Tasks ===',
                    '☐ 마지막 투약 (if needed)',
                    '☐ Terminal blood sampling (cardiac puncture)',
                    '☐ Sacrifice (CO2 or cervical dislocation)',
                    '☐ 신장 적출 (L/R kidney)',
                    '☐ 간, 심장 적출 (필요 시)',
                    '☐ Tissue processing:',
                    '  - L kidney → 10% formalin (histology)',
                    '  - R kidney → snap-freeze in LN2 → -80°C',
                    '☐ 샘플 라벨링 및 기록',
                    '☐ 데이터 정리 및 백업'
                ])
        
        return checklist
    
    def generate_gantt_chart_data(self, protocol: Dict) -> List[Dict]:
        """
        Gantt chart용 데이터 생성
        """
        
        timeline = protocol['timeline']
        duration = protocol['experiment_info']['duration_days']
        
        tasks = []
        
        # Acclimation
        tasks.append({
            'Task': 'Acclimation',
            'Start': timeline['acclimation_start'],
            'Finish': timeline['surgery_day'],
            'Resource': 'Preparation'
        })
        
        # Surgery
        tasks.append({
            'Task': 'UUO Surgery',
            'Start': timeline['surgery_day'],
            'Finish': timeline['surgery_day'] + timedelta(hours=6),
            'Resource': 'Surgery'
        })
        
        # Dosing period
        tasks.append({
            'Task': 'Daily Dosing',
            'Start': timeline['first_dose'],
            'Finish': timeline['sacrifice_day'],
            'Resource': 'Treatment'
        })
        
        # Monitoring
        tasks.append({
            'Task': 'Body Weight Monitoring',
            'Start': timeline['first_dose'],
            'Finish': timeline['sacrifice_day'],
            'Resource': 'Monitoring'
        })
        
        # Interim sampling
        day_7 = timeline['surgery_day'] + timedelta(days=7)
        tasks.append({
            'Task': 'Blood Sampling (Day 7)',
            'Start': day_7,
            'Finish': day_7 + timedelta(hours=2),
            'Resource': 'Sampling'
        })
        
        # Terminal
        tasks.append({
            'Task': 'Terminal Sacrifice',
            'Start': timeline['sacrifice_day'],
            'Finish': timeline['sacrifice_day'] + timedelta(hours=4),
            'Resource': 'Endpoint'
        })
        
        return tasks


def main():
    """메인 실행 (테스트)"""
    print("🧪 NOVA UUO Dosing Schedule Generator\n")
    
    scheduler = UUODosingScheduler(start_date='2025-01-15')
    
    # Generate protocol
    protocol = scheduler.generate_uuo_protocol(
        n_mice_per_group=8,
        route='PO',
        frequency='QD',
        duration_days=14,
        doses_mg_kg=[10, 30]
    )
    
    print("📋 Experiment Info:")
    print(f"  Total mice: {protocol['experiment_info']['total_mice']}")
    print(f"  Groups: {protocol['experiment_info']['n_groups']}")
    print(f"  Duration: {protocol['experiment_info']['duration_days']} days")
    print(f"  Route: {protocol['experiment_info']['route']}")
    print(f"  Frequency: {protocol['experiment_info']['frequency']}")
    
    print("\n🗓️ Timeline:")
    print(f"  Acclimation: {protocol['timeline']['acclimation_start'].strftime('%Y-%m-%d')}")
    print(f"  Surgery (Day 0): {protocol['timeline']['surgery_day'].strftime('%Y-%m-%d')}")
    print(f"  First Dose (Day 1): {protocol['timeline']['first_dose'].strftime('%Y-%m-%d')}")
    print(f"  Sacrifice (Day 14): {protocol['timeline']['sacrifice_day'].strftime('%Y-%m-%d')}")
    
    # Dosing schedule
    print("\n📅 Dosing Schedule:")
    schedule = scheduler.generate_dosing_schedule(protocol)
    print(schedule.head(10))
    
    # Sample collection
    print("\n🧬 Sample Collection Plan:")
    samples = scheduler.generate_sample_collection_plan(protocol)
    print(samples.head(5))
    
    # Daily checklist (Day 1)
    print("\n✅ Day 1 Checklist:")
    checklist = scheduler.generate_daily_checklist(protocol, target_day=1)
    print(f"Date: {checklist['date']}")
    for task in checklist['tasks']:
        print(f"  {task}")


if __name__ == "__main__":
    main()
