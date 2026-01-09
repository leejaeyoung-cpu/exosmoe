"""
MI-EXO Knowledge Base Module
논문과 연구 계획서에서 엑소좀 기능 및 치료 효능 정보를 추출합니다.
"""

import os
from pathlib import Path
import re
from collections import Counter

class KnowledgeBase:
    def __init__(self, papers_dir="data/papers"):
        self.papers_dir = Path(papers_dir)
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        
        # 관심 키워드 정의
        self.keywords = {
            '심혈관': ['cardiovascular', 'heart', 'cardiac', 'vascular', 'angiogenesis', '혈관'],
            '항염증': ['anti-inflammatory', 'inflammation', 'immune', 'cytokine', '염증'],
            '항섬유화': ['anti-fibrotic', 'fibrosis', 'collagen', '섬유화'],
            '항산화': ['antioxidant', 'ros', 'oxidative', 'stress', '산화'],
            '세포증식': ['proliferation', 'growth', 'regeneration', '증식'],
            '엑소좀': ['exosome', 'ev', 'vesicle', 'mirna', 'mir-']
        }

    def get_paper_list(self):
        """저장된 논문/문서 목록 반환"""
        extensions = ['*.md', '*.txt', '*.pdf']
        files = []
        for ext in extensions:
            files.extend(self.papers_dir.glob(ext))
        return [f.name for f in files]

    def analyze_document(self, filename):
        """문서 내용을 분석하여 주요 키워드 추출"""
        file_path = self.papers_dir / filename
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
                
            results = {}
            found_mirnas = set(re.findall(r'mir-\d+[a-z]?', content))
            
            # 카테고리별 관련성 점수 계산
            for category, terms in self.keywords.items():
                score = 0
                for term in terms:
                    score += content.count(term)
                if score > 0:
                    results[category] = score
            
            return {
                'filename': filename,
                'scores': results,
                'mirnas': list(found_mirnas),
                'summary': content[:200] + "..." # 간단 요약
            }
            
        except Exception as e:
            return {'error': str(e)}

    def get_aggregated_insights(self):
        """모든 문서의 정보를 종합"""
        files = self.get_paper_list()
        total_scores = Counter()
        all_mirnas = set()
        
        for file in files:
            analysis = self.analyze_document(file)
            if 'error' not in analysis:
                total_scores.update(analysis['scores'])
                all_mirnas.update(analysis['mirnas'])
                
        return {
            'top_effects': total_scores.most_common(3),
            'mentioned_mirnas': list(all_mirnas),
            'doc_count': len(files)
        }
