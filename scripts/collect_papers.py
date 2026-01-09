"""
논문 데이터 자동 수집기 (PubMed API 활용)
주제: Exosome & Cardiovascular Disease Therapy
목표: 최신 논문 100편의 초록 수집
"""

import requests
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

def collect_nature_level_papers():
    print("="*80)
    print("📚 네이처급 엑소좀 논문 데이터 수집 시작")
    print("="*80 + "\n")
    
    # 저장 경로
    save_dir = Path("data/papers")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 검색 키워드
    query = "exosome cardiovascular therapy[Title/Abstract]"
    
    # 1. 논문 ID 검색 (ESearch)
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    search_url = f"{base_url}/esearch.fcgi?db=pubmed&term={query}&retmax=100&sort=date&retmode=json"
    
    try:
        print("🔍 논문 검색 중...")
        response = requests.get(search_url)
        data = response.json()
        id_list = data['esearchresult']['idlist']
        print(f"✅ {len(id_list)}개의 최신 논문 발견!")
        
        # 2. 상세 정보 수집 (EFetch)
        ids = ",".join(id_list)
        fetch_url = f"{base_url}/efetch.fcgi?db=pubmed&id={ids}&retmode=xml"
        
        print("📥 데이터 다운로드 및 분석 중...")
        response = requests.get(fetch_url)
        root = ET.fromstring(response.content)
        
        count = 0
        for article in tqdm(root.findall(".//PubmedArticle")):
            try:
                # 제목
                title = article.find(".//ArticleTitle").text
                
                # 초록
                abstract_list = article.findall(".//AbstractText")
                abstract = "\n".join([t.text for t in abstract_list if t.text])
                
                # 저널명
                journal = article.find(".//Title").text
                
                # 연도
                year = article.find(".//PubDate/Year")
                if year is None:
                    year = "2024" # 기본값
                else:
                    year = year.text
                
                # 파일 저장
                safe_title = "".join([c for c in title if c.isalnum() or c in (' ', '-', '_')]).rstrip()
                filename = f"{year}_{safe_title[:50]}.txt"
                
                content = f"""Title: {title}
Journal: {journal}
Year: {year}

Abstract:
{abstract}

Keywords: Exosome, Cardiovascular, Therapy, miRNA
"""
                with open(save_dir / filename, "w", encoding="utf-8") as f:
                    f.write(content)
                
                count += 1
                
            except Exception as e:
                continue
                
        print(f"\n✅ 총 {count}개의 논문 데이터가 지식 베이스에 등록되었습니다!")
        print(f"📂 저장 위치: {save_dir.absolute()}")
        
        return count
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return 0

if __name__ == "__main__":
    collect_nature_level_papers()
