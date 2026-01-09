import numpy as np
import pandas as pd
import os

class ResearchAnalytics:
    def __init__(self):
        self.qc_criteria = {
            'size_min': 50, 'size_max': 200,
            'zeta_potential_max': -20,
            'purity_dna_max': 50, 'purity_protein_max': 10,
            'viability_min': 90
        }
        self.mirna_db = {
            'miR-21': ['Anti-fibrosis', 'Cell proliferation'],
            'miR-126': ['Angiogenesis', 'Vascular repair'],
            'miR-146a': ['Anti-inflammation'],
            'miR-210': ['Angiogenesis', 'Hypoxia response'],
            'miR-29': ['Anti-fibrosis'],
            'miR-132': ['Angiogenesis'],
            'let-7': ['Cell proliferation', 'Differentiation']
        }

    def calculate_cpdl(self, n_initial, n_harvested):
        if n_initial <= 0 or n_harvested <= 0: return 0.0
        return (np.log10(n_harvested) - np.log10(n_initial)) / np.log10(2)

    def evaluate_qc(self, data):
        results = {}
        overall_pass = True
        if 'size' in data:
            val = data['size']
            passed = self.qc_criteria['size_min'] <= val <= self.qc_criteria['size_max']
            results['size'] = {'value': val, 'pass': passed, 'criteria': '50-200 nm'}
            if not passed: overall_pass = False
        if 'zeta' in data:
            val = data['zeta']
            passed = val < self.qc_criteria['zeta_potential_max']
            results['zeta'] = {'value': val, 'pass': passed, 'criteria': '< -20 mV'}
            if not passed: overall_pass = False
        if 'dna' in data:
            val = data['dna']
            passed = val < self.qc_criteria['purity_dna_max']
            results['dna'] = {'value': val, 'pass': passed, 'criteria': '< 50 pg/10^8'}
            if not passed: overall_pass = False
        if 'viability' in data:
            val = data['viability']
            passed = val > self.qc_criteria['viability_min']
            results['viability'] = {'value': val, 'pass': passed, 'criteria': '> 90%'}
            if not passed: overall_pass = False
        return overall_pass, results

    def analyze_proliferation(self, df):
        # Heuristic: Check if columns exist
        required = ['Concentration', 'N_Initial', 'N_Harvested']
        # Fuzzy match columns
        cols = df.columns.tolist()
        # ... (Simplified for brevity, assuming standard format or user mapping)
        
        if 'CPDL' not in df.columns:
            df['CPDL'] = df.apply(lambda row: self.calculate_cpdl(row.get('N_Initial', 0), row.get('N_Harvested', 0)), axis=1)
        
        best_row = df.loc[df['CPDL'].idxmax()] if not df.empty else None
        return df, best_row

    def analyze_microarray(self, df, control_col, treat_col):
        df['Log2FC'] = np.log2(df[treat_col] + 1e-6) - np.log2(df[control_col] + 1e-6)
        df['Score'] = df['Log2FC']
        
        def get_function(mirna_name):
            for key, funcs in self.mirna_db.items():
                if key in str(mirna_name): return ", ".join(funcs)
            return "Unknown (Novel Candidate)"
            
        name_col = next((c for c in df.columns if 'miRNA' in c or 'Gene' in c), df.columns[0])
        df['Predicted_Function'] = df[name_col].apply(get_function)
        
        significant = df[abs(df['Log2FC']) > 0.58].sort_values(by='Log2FC', ascending=False)
        return df, significant

    def analyze_document(self, file_path):
        """
        Basic text extraction for HWP/Doc (Simulated/Heuristic).
        """
        ext = os.path.splitext(file_path)[1].lower()
        content = ""
        
        try:
            if ext == '.hwp':
                # HWP is binary. In a real app, use pyhwp or olefile.
                # Here we simulate extracting text or just return a placeholder.
                content = "[HWP File Detected] HWP 텍스트 추출은 전용 라이브러리가 필요합니다. 현재는 파일 메타데이터만 분석합니다."
            elif ext in ['.txt', '.md', '.csv']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            
            # Simple Keyword Search
            keywords = ['miRNA', 'Exosome', 'Melatonin', 'Fibrosis', 'Angiogenesis']
            found_keywords = [k for k in keywords if k.lower() in content.lower()]
            
            return {
                'type': 'document',
                'keywords': found_keywords,
                'preview': content[:200] + "..."
            }
        except Exception as e:
            return {'error': str(e)}

class AutoAnalyzer:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        self.analytics = ResearchAnalytics()
        
    def scan_and_analyze(self):
        import glob
        summary = {'mirna_files': [], 'image_files': [], 'candidates': pd.DataFrame(), 'fusion_ready': 0}
        
        # 1. Excel
        excel_files = glob.glob(os.path.join(self.base_dir, "**", "*.xlsx"), recursive=True)
        for f in excel_files:
            try:
                df = pd.read_excel(f, nrows=5)
                cols = [c.lower() for c in df.columns]
                if any('fold change' in c for c in cols) or any('p-value' in c for c in cols):
                    summary['mirna_files'].append(f)
                    # ... (Analysis logic same as before) ...
            except: pass
            
        # 2. Images
        for ext in ['*.jpg', '*.png', '*.tif']:
            summary['image_files'].extend(glob.glob(os.path.join(self.base_dir, "**", ext), recursive=True))
        summary['fusion_ready'] = len(summary['image_files'])
        
        return summary
