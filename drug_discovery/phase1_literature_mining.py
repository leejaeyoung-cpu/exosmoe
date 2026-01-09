"""
AI-Driven Drug Discovery Pipeline for CKD-CVD
Phase 1: Literature Mining and Knowledge Extraction

이 모듈은 PubMed에서 자동으로 논문을 검색하고 핵심 정보를 추출합니다.
"""

import requests
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
import time
from xml.etree import ElementTree as ET

class LiteratureMiner:
    """
    PubMed API를 사용한 문헌 마이닝
    """
    
    def __init__(self, email="research@example.com"):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.email = email
        self.output_dir = Path("data/literature")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def search_papers(self, query: str, max_results: int = 100) -> List[str]:
        """
        PubMed에서 논문 검색
        
        Args:
            query: 검색어
            max_results: 최대 결과 수
            
        Returns:
            PubMed ID 리스트
        """
        print(f"🔍 검색 중: '{query}'...")
        
        search_url = f"{self.base_url}esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'email': self.email,
            # 최근 5년 논문만
            'reldate': 1825,  # 5 years in days
            'datetype': 'pdat'
        }
        
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            pmids = data.get('esearchresult', {}).get('idlist', [])
            print(f"   ✅ {len(pmids)}개 논문 발견")
            return pmids
            
        except Exception as e:
            print(f"   ❌ 검색 실패: {e}")
            return []
    
    def fetch_abstracts(self, pmids: List[str]) -> List[Dict]:
        """
        PubMed ID로 초록 가져오기
        
        Args:
            pmids: PubMed ID 리스트
            
        Returns:
            논문 정보 리스트 (제목, 초록, 저자, 저널 등)
        """
        print(f"📥 {len(pmids)}개 논문 초록 다운로드 중...")
        
        fetch_url = f"{self.base_url}efetch.fcgi"
        papers = []
        
        # API rate limit을 위해 배치 처리
        batch_size = 50
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i+batch_size]
            params = {
                'db': 'pubmed',
                'id': ','.join(batch),
                'retmode': 'xml',
                'email': self.email
            }
            
            try:
                response = requests.get(fetch_url, params=params)
                response.raise_for_status()
                
                # XML 파싱
                root = ET.fromstring(response.content)
                
                for article in root.findall('.//PubmedArticle'):
                    try:
                        # PMID
                        pmid = article.find('.//PMID').text
                        
                        # 제목
                        title_elem = article.find('.//ArticleTitle')
                        title = title_elem.text if title_elem is not None else "No title"
                        
                        # 초록
                        abstract_elem = article.find('.//AbstractText')
                        abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"
                        
                        # 저널
                        journal_elem = article.find('.//Journal/Title')
                        journal = journal_elem.text if journal_elem is not None else "Unknown"
                        
                        # 년도
                        year_elem = article.find('.//PubDate/Year')
                        year = year_elem.text if year_elem is not None else "Unknown"
                        
                        papers.append({
                            'pmid': pmid,
                            'title': title,
                            'abstract': abstract,
                            'journal': journal,
                            'year': year
                        })
                        
                    except Exception as e:
                        print(f"   ⚠️ 논문 파싱 오류: {e}")
                        continue
                
                # API rate limit 준수
                time.sleep(0.5)
                print(f"   진행: {min(i+batch_size, len(pmids))}/{len(pmids)}")
                
            except Exception as e:
                print(f"   ❌ 배치 다운로드 실패: {e}")
                continue
        
        print(f"   ✅ 총 {len(papers)}개 논문 정보 수집 완료")
        return papers
    
    def mine_ckd_cvd_literature(self, papers_per_query: int = 20) -> pd.DataFrame:
        """
        CKD-CVD 관련 문헌 종합 수집
        
        Returns:
            논문 정보 DataFrame
        """
        print("\n" + "="*70)
        print("CKD-CVD 문헌 마이닝 시작")
        print("="*70)
        
        queries = [
            "chronic kidney disease drug discovery",
            "cardiovascular disease therapeutic targets",
            "NF-kappa B inhibitor kidney",
            "TGF-beta antagonist renal fibrosis",
            "mitochondrial protection chronic kidney disease",
            "endothelial dysfunction cardiovascular disease treatment",
            "oxidative stress kidney disease therapy",
            "inflammation kidney cardiovascular disease",
        ]
        
        all_papers = []
        all_pmids = set()
        
        for query in queries:
            pmids = self.search_papers(query, max_results=papers_per_query)
            
            # 중복 제거
            new_pmids = [pmid for pmid in pmids if pmid not in all_pmids]
            if new_pmids:
                papers = self.fetch_abstracts(new_pmids)
                all_papers.extend(papers)
                all_pmids.update(new_pmids)
            
            time.sleep(1)  # API rate limit
        
        df = pd.DataFrame(all_papers)
        
        # 저장
        output_file = self.output_dir / "ckd_cvd_literature.csv"
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 저장: {output_file}")
        print(f"📊 총 {len(df)}개 고유 논문 수집")
        
        return df


class KnowledgeExtractor:
    """
    논문에서 핵심 지식 추출
    (간소화 버전 - 키워드 기반)
    """
    
    def __init__(self):
        # CKD-CVD 관련 핵심 타겟
        self.targets = {
            'NF-κB': ['NF-kappa B', 'NF-kappaB', 'NFKB', 'p65', 'RelA'],
            'TGF-β': ['TGF-beta', 'TGF-β', 'TGFB', 'transforming growth factor'],
            'NOX4': ['NADPH oxidase 4', 'NOX4'],
            'VCAM1': ['VCAM-1', 'VCAM1', 'vascular cell adhesion'],
            'ICAM1': ['ICAM-1', 'ICAM1', 'intercellular adhesion'],
            'mTOR': ['mTOR', 'mammalian target of rapamycin'],
            'AMPK': ['AMPK', 'AMP-activated protein kinase'],
            'Nrf2': ['Nrf2', 'NRF2', 'nuclear factor erythroid'],
        }
        
        # 치료 메커니즘
        self.mechanisms = {
            'inhibitor': ['inhibit', 'suppress', 'block', 'antagonist'],
            'activator': ['activate', 'enhance', 'agonist', 'induce'],
            'modulator': ['modulate', 'regulate', 'control'],
        }
    
    def extract_targets(self, papers_df: pd.DataFrame) -> pd.DataFrame:
        """
        논문에서 타겟 단백질 언급 추출
        
        Returns:
            타겟별 언급 빈도 및 관련 논문
        """
        print("\n" + "="*70)
        print("타겟 단백질 추출")
        print("="*70)
        
        target_mentions = []
        
        for idx, row in papers_df.iterrows():
            text = f"{row['title']} {row['abstract']}".lower()
            
            for target, keywords in self.targets.items():
                for keyword in keywords:
                    if keyword.lower() in text:
                        target_mentions.append({
                            'pmid': row['pmid'],
                            'target': target,
                            'keyword': keyword,
                            'title': row['title'],
                            'year': row['year']
                        })
                        break  # 하나만 찾으면 됨
        
        df_targets = pd.DataFrame(target_mentions)
        
        # 타겟별 통계
        if not df_targets.empty:
            target_stats = df_targets['target'].value_counts()
            print(f"\n📊 타겟 언급 빈도:")
            for target, count in target_stats.items():
                print(f"   {target}: {count}회")
        
        return df_targets
    
    def extract_molecules(self, papers_df: pd.DataFrame) -> List[Dict]:
        """
        논문에서 잠재적 치료 분자 추출
        (기존 약물 및 화합물)
        
        Returns:
            분자 정보 리스트
        """
        print("\n" + "="*70)
        print("치료 분자 추출")
        print("="*70)
        
        # 알려진 약물/화합물 키워드
        known_molecules = {
            'Metformin': ['diabetes', 'AMPK'],
            'Bardoxolone': ['Nrf2', 'oxidative stress'],
            'Pirfenidone': ['fibrosis', 'TGF-beta'],
            'Losartan': ['angiotensin', 'fibrosis'],
            'Curcumin': ['NF-kappa B', 'inflammation'],
            'Resveratrol': ['oxidative', 'mitochondria'],
            'N-acetylcysteine': ['antioxidant', 'glutathione'],
        }
        
        molecules = []
        
        for molecule, contexts in known_molecules.items():
            count = 0
            pmids = []
            
            for idx, row in papers_df.iterrows():
                text = f"{row['title']} {row['abstract']}".lower()
                
                if molecule.lower() in text:
                    # Context 확인
                    relevant = any(ctx.lower() in text for ctx in contexts)
                    if relevant:
                        count += 1
                        pmids.append(row['pmid'])
            
            if count > 0:
                molecules.append({
                    'molecule': molecule,
                    'mentions': count,
                    'pmids': pmids[:5],  # 상위 5개만
                    'context': ', '.join(contexts)
                })
        
        # 정렬
        molecules = sorted(molecules, key=lambda x: x['mentions'], reverse=True)
        
        print(f"\n💊 발견된 치료 분자 ({len(molecules)}개):")
        for mol in molecules[:10]:
            print(f"   {mol['molecule']}: {mol['mentions']}회 언급")
        
        return molecules


def main():
    """
    메인 실행 함수
    """
    print("\n" + "="*70)
    print("AI 기반 CKD-CVD 신약 발견 파이프라인")
    print("Phase 1: 문헌 마이닝 및 지식 추출")
    print("="*70)
    
    # Step 1: 문헌 수집
    miner = LiteratureMiner()
    papers_df = miner.mine_ckd_cvd_literature(papers_per_query=15)
    
    # Step 2: 지식 추출
    extractor = KnowledgeExtractor()
    
    # 타겟 추출
    targets_df = extractor.extract_targets(papers_df)
    targets_df.to_csv("data/literature/extracted_targets.csv", index=False, encoding='utf-8-sig')
    
    # 분자 추출
    molecules = extractor.extract_molecules(papers_df)
    pd.DataFrame(molecules).to_csv("data/literature/extracted_molecules.csv", index=False, encoding='utf-8-sig')
    
    print("\n" + "="*70)
    print("✅ Phase 1 완료!")
    print(f"   📄 총 논문: {len(papers_df)}개")
    print(f"   🎯 타겟: {len(targets_df)}개 언급")
    print(f"   💊 분자: {len(molecules)}개 발견")
    print("="*70)
    
    return papers_df, targets_df, molecules


if __name__ == "__main__":
    papers, targets, molecules = main()
