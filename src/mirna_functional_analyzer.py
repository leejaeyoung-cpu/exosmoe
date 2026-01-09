"""
miRNA Functional Analyzer
MT-EXO vs Control-EXO miRNA 분석 및 유력 후보 도출 엔진

Features:
- miRNA 기능 분류 (항산화, 항섬유화, 항염증, 혈관형성, 세포증식)
- MT-EXO vs Control 비교 분석
- AI 기반 후보 스코어링 및 우선순위 도출
- 시그널 경로 매핑
- 상세 분석 리포트 생성
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class miRNA_FunctionalAnalyzer:
    """miRNA 기능 분석 및 후보 도출 클래스"""
    
    FUNCTION_CATEGORIES = {
        'antioxidant': '항산화',
        'anti_fibrotic': '항섬유화',
        'anti_inflammatory': '항염증',
        'angiogenic': '혈관형성',
        'proliferation': '세포증식'
    }
    
    EVIDENCE_SCORES = {
        'High': 1.0,
        'Medium': 0.7,
        'Low': 0.4
    }
    
    def __init__(self, database_path: str):
        """
        Args:
            database_path: miRNA 기능 데이터베이스 JSON 파일 경로
        """
        self.database_path = Path(database_path)
        self.database = self._load_database()
        self.mirna_data = self.database['mirna_functions']
        self.pathway_data = self.database['pathway_database']
        self.weights = self.database['scoring_weights']
        self.thresholds = self.database['thresholds']
        
    def _load_database(self) -> Dict:
        """데이터베이스 로드"""
        with open(self.database_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_microarray_data(self, excel_path: str) -> pd.DataFrame:
        """
        마이크로어레이 데이터 로드 및 전처리
        
        Args:
            excel_path: Excel 파일 경로
            
        Returns:
            전처리된 DataFrame
        """
        df = pd.read_excel(excel_path)
        
        # 컬럼명 정리
        df.columns = df.columns.str.strip()
        
        # 필수 컬럼 확인
        required_cols = [
            'Transcript ID(Array Design)',
            'MT-EXOSOME/Con-EXO.fc',
            'Con-EXO.mean',
            'MT-EXOSOME.mean'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"필수 컬럼 누락: {missing_cols}")
        
        # miRNA 이름 표준화
        df['miRNA'] = df['Transcript ID(Array Design)'].str.strip()
        
        # Log2 Fold Change 계산 (fc는 linear fold change로 가정)
        df['Log2FC'] = np.log2(df['MT-EXOSOME/Con-EXO.fc'].abs() + 1e-10)
        df['Log2FC'] = df['Log2FC'] * np.sign(df['MT-EXOSOME/Con-EXO.fc'])
        
        # Detection 여부
        if 'MT-EXOSOME/Con-EXO.detected' in df.columns:
            df['Detected'] = df['MT-EXOSOME/Con-EXO.detected']
        else:
            df['Detected'] = True
        
        # 발현량 차이
        df['Expression_Diff'] = df['MT-EXOSOME.mean'] - df['Con-EXO.mean']
        
        return df
    
    def annotate_functions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        miRNA에 기능 주석 추가
        
        Args:
            df: 마이크로어레이 데이터 DataFrame
            
        Returns:
            기능 주석이 추가된 DataFrame
        """
        annotations = []
        
        for _, row in df.iterrows():
            mirna = row['miRNA']
            annotation = self._get_mirna_annotation(mirna)
            annotation['miRNA'] = mirna
            annotations.append(annotation)
        
        annotation_df = pd.DataFrame(annotations)
        result = df.merge(annotation_df, on='miRNA', how='left')
        
        return result
    
    def _get_mirna_annotation(self, mirna: str) -> Dict:
        """개별 miRNA 주석 정보 추출"""
        # 데이터베이스에서 정확한 매칭 시도
        if mirna in self.mirna_data:
            mirna_info = self.mirna_data[mirna]
        else:
            # Partial matching (변형 이름 처리)
            matched = None
            for db_mirna in self.mirna_data.keys():
                if db_mirna.replace('-3p', '').replace('-5p', '') in mirna or \
                   mirna.replace('-3p', '').replace('-5p', '') in db_mirna:
                    matched = db_mirna
                    break
            
            if matched:
                mirna_info = self.mirna_data[matched]
            else:
                # Unknown miRNA
                return self._create_unknown_annotation()
        
        # 기능별 점수 추출
        functions = mirna_info.get('functions', {})
        annotation = {
            'antioxidant_score': functions.get('antioxidant', {}).get('score', 0),
            'anti_fibrotic_score': functions.get('anti_fibrotic', {}).get('score', 0),
            'anti_inflammatory_score': functions.get('anti_inflammatory', {}).get('score', 0),
            'angiogenic_score': functions.get('angiogenic', {}).get('score', 0),
            'proliferation_score': functions.get('proliferation', {}).get('score', 0),
            'max_function_score': 0,
            'primary_function': 'Unknown',
            'targets': ', '.join(mirna_info.get('targets', [])),
            'pathways': ', '.join(mirna_info.get('pathways', [])),
            'therapeutic_potential': mirna_info.get('therapeutic_potential', 'Unknown'),
            'mirbase_id': mirna_info.get('mirbase_id', 'Unknown')
        }
        
        # 주요 기능 결정
        function_scores = {
            'antioxidant': annotation['antioxidant_score'],
            'anti_fibrotic': annotation['anti_fibrotic_score'],
            'anti_inflammatory': annotation['anti_inflammatory_score'],
            'angiogenic': annotation['angiogenic_score'],
            'proliferation': annotation['proliferation_score']
        }
        
        if max(function_scores.values()) > 0:
            primary_func = max(function_scores, key=function_scores.get)
            annotation['primary_function'] = self.FUNCTION_CATEGORIES[primary_func]
            annotation['max_function_score'] = function_scores[primary_func]
        
        return annotation
    
    def _create_unknown_annotation(self) -> Dict:
        """알려지지 않은 miRNA의 기본 주석"""
        return {
            'antioxidant_score': 0,
            'anti_fibrotic_score': 0,
            'anti_inflammatory_score': 0,
            'angiogenic_score': 0,
            'proliferation_score': 0,
            'max_function_score': 0,
            'primary_function': '미분류 (신규 후보)',
            'targets': 'Unknown',
            'pathways': 'Under investigation',
            'therapeutic_potential': 'Low',
            'mirbase_id': 'Unknown'
        }
    
    def calculate_candidate_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        AI 기반 후보 점수 계산
        
        종합 점수 = (Fold Change 점수 × 0.35) + (기능 점수 × 0.25) + 
                    (증거 수준 × 0.25) + (치료 잠재력 × 0.15)
        """
        # Fold Change 점수 (정규화)
        max_fc = df['Log2FC'].abs().max()
        df['FC_score'] = df['Log2FC'].abs() / max_fc if max_fc > 0 else 0
        
        # 치료 잠재력 점수
        therapeutic_map = {'High': 1.0, 'Medium': 0.7, 'Low': 0.4, 'Unknown': 0.2}
        df['Therapeutic_score'] = df['therapeutic_potential'].map(therapeutic_map).fillna(0.2)
        
        # 증거 점수 (최대 기능 점수를 증거 수준으로 간주)
        df['Evidence_score'] = df['max_function_score']
        
        # 종합 점수 계산
        df['Candidate_Score'] = (
            df['FC_score'] * self.weights['fold_change_weight'] +
            df['max_function_score'] * self.weights['function_score_weight'] +
            df['Evidence_score'] * self.weights['evidence_weight'] +
            df['Therapeutic_score'] * self.weights['therapeutic_potential_weight']
        )
        
        # MT-EXO에서 upregulated된 경우에만 양수 점수
        df.loc[df['Log2FC'] < 0, 'Candidate_Score'] *= -1
        
        return df
    
    def filter_significant_mirnas(self, df: pd.DataFrame, 
                                  min_fc: Optional[float] = None,
                                  upregulated_only: bool = True) -> pd.DataFrame:
        """
        유의미한 miRNA 필터링
        
        Args:
            df: 분석된 DataFrame
            min_fc: 최소 fold change (None이면 기본값 사용)
            upregulated_only: MT-EXO에서 증가한 것만 선택
            
        Returns:
            필터링된 DataFrame
        """
        if min_fc is None:
            min_fc = self.thresholds['min_fold_change']
        
        filtered = df[df['Detected'] == True].copy()
        
        if upregulated_only:
            filtered = filtered[filtered['Log2FC'] > 0]
        
        filtered = filtered[filtered['Log2FC'].abs() >= np.log2(min_fc)]
        
        return filtered.sort_values('Candidate_Score', ascending=False)
    
    def get_top_candidates_by_function(self, df: pd.DataFrame, 
                                      function: str, 
                                      top_n: int = 10) -> pd.DataFrame:
        """
        특정 기능별 Top 후보 miRNA 추출
        
        Args:
            df: 분석된 DataFrame
            function: 기능 카테고리 (antioxidant, anti_fibrotic 등)
            top_n: 상위 N개
            
        Returns:
            Top 후보 DataFrame
        """
        score_col = f'{function}_score'
        
        if score_col not in df.columns:
            raise ValueError(f"Invalid function: {function}")
        
        # 해당 기능 점수가 있는 miRNA만 선택
        candidates = df[df[score_col] > 0].copy()
        
        # MT-EXO에서 upregulated된 것만
        candidates = candidates[candidates['Log2FC'] > 0]
        
        # 해당 기능 점수로 정렬
        candidates = candidates.sort_values(score_col, ascending=False)
        
        return candidates.head(top_n)
    
    def generate_function_summary(self, df: pd.DataFrame) -> Dict:
        """
        기능별 요약 통계 생성
        
        Returns:
            기능별 통계 딕셔너리
        """
        summary = {}
        
        for func_key, func_name in self.FUNCTION_CATEGORIES.items():
            score_col = f'{func_key}_score'
            
            # 해당 기능을 가진 miRNA
            candidates = df[(df[score_col] > 0) & (df['Log2FC'] > 0)]
            
            summary[func_name] = {
                'total_count': len(candidates),
                'mean_fold_change': candidates['MT-EXOSOME/Con-EXO.fc'].mean(),
                'mean_function_score': candidates[score_col].mean(),
                'top_3_mirnas': candidates.nlargest(3, score_col)['miRNA'].tolist(),
                'top_3_scores': candidates.nlargest(3, score_col)[score_col].tolist()
            }
        
        return summary
    
    def export_analysis_results(self, df: pd.DataFrame, output_dir: str):
        """
        분석 결과를 여러 형식으로 저장
        
        Args:
            df: 분석된 DataFrame
            output_dir: 출력 디렉토리
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 1. 전체 분석 결과
        full_path = output_path / 'MT_EXO_Full_Analysis.csv'
        df.to_csv(full_path, index=False, encoding='utf-8-sig')
        print(f"✓ 전체 분석 결과 저장: {full_path}")
        
        # 2. 유의미한 후보만
        significant = self.filter_significant_mirnas(df)
        sig_path = output_path / 'MT_EXO_Significant_Candidates.csv'
        significant.to_csv(sig_path, index=False, encoding='utf-8-sig')
        print(f"✓ 유의미한 후보 저장: {sig_path}")
        
        # 3. 기능별 Top 후보
        for func_key, func_name in self.FUNCTION_CATEGORIES.items():
            top_candidates = self.get_top_candidates_by_function(df, func_key, top_n=10)
            func_path = output_path / f'Top_Candidates_{func_name}.csv'
            top_candidates.to_csv(func_path, index=False, encoding='utf-8-sig')
            print(f"✓ {func_name} Top 후보 저장: {func_path}")
        
        # 4. 요약 통계
        summary = self.generate_function_summary(df)
        summary_path = output_path / 'Function_Summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✓ 기능별 요약 저장: {summary_path}")
        
        print(f"\n📁 모든 결과가 {output_dir}에 저장되었습니다.")


def main():
    """메인 실행 함수"""
    print("\n" + "="*80)
    print("🧬 MT-EXO miRNA 기능 분석 시작")
    print("="*80 + "\n")
    
    # 경로 설정
    database_path = r"c:\Users\brook\Desktop\mi_exo_ai\data\mirna_function_database.json"
    data_path = r"c:\Users\brook\Desktop\mi_exo_ai\data\Final_Analysis_Result\Final_Analysis_Result\data3.xlsx"
    output_dir = r"c:\Users\brook\Desktop\mi_exo_ai\data\MT_EXO_Analysis_Results"
    
    # 분석기 초기화
    print("📚 miRNA 기능 데이터베이스 로딩...")
    analyzer = miRNA_FunctionalAnalyzer(database_path)
    print(f"   ✓ {len(analyzer.mirna_data)}개 miRNA 정보 로드 완료")
    print(f"   ✓ {len(analyzer.pathway_data)}개 시그널 경로 로드 완료\n")
    
    # 데이터 로드
    print("📊 MT-EXO 마이크로어레이 데이터 로딩...")
    df = analyzer.load_microarray_data(data_path)
    print(f"   ✓ {len(df)}개 miRNA 데이터 로드 완료\n")
    
    # 기능 주석
    print("🔬 miRNA 기능 주석 추가 중...")
    df = analyzer.annotate_functions(df)
    annotated_count = (df['primary_function'] != '미분류 (신규 후보)').sum()
    print(f"   ✓ {annotated_count}개 miRNA에 기능 주석 추가 완료\n")
    
    # 후보 점수 계산
    print("🎯 후보 점수 계산 중...")
    df = analyzer.calculate_candidate_score(df)
    print(f"   ✓ 후보 점수 계산 완료\n")
    
    # 유의미한 후보 필터링
    print("🔍 유의미한 후보 필터링 중...")
    significant = analyzer.filter_significant_mirnas(df, upregulated_only=True)
    print(f"   ✓ {len(significant)}개 유의미한 후보 발견\n")
    
    # 기능별 Top 후보
    print("🏆 기능별 Top 후보:")
    print("-"*80)
    for func_key, func_name in analyzer.FUNCTION_CATEGORIES.items():
        top = analyzer.get_top_candidates_by_function(df, func_key, top_n=5)
        print(f"\n[{func_name}] Top 5:")
        if len(top) > 0:
            for i, (_, row) in enumerate(top.iterrows(), 1):
                print(f"  {i}. {row['miRNA']:20s} | FC: {row['MT-EXOSOME/Con-EXO.fc']:6.2f} | "
                      f"Score: {row[f'{func_key}_score']:.2f} | "
                      f"Candidate Score: {row['Candidate_Score']:.3f}")
        else:
            print("  (해당 기능 후보 없음)")
    
    # 결과 저장
    print("\n" + "="*80)
    print("💾 분석 결과 저장 중...")
    print("="*80 + "\n")
    analyzer.export_analysis_results(df, output_dir)
    
    print("\n" + "="*80)
    print("✅ MT-EXO miRNA 기능 분석 완료!")
    print("="*80 + "\n")
    
    return analyzer, df, significant


if __name__ == "__main__":
    analyzer, df, significant = main()
