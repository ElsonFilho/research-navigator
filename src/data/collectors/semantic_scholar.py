"""
Semantic Scholar API integration for citation enrichment
"""

import requests
import time
from typing import List, Dict, Optional
from pathlib import Path


class SemanticScholarEnricher:
    """Enrich papers with Semantic Scholar metadata"""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["x-api-key"] = api_key
    
    def _search_by_arxiv_id(self, arxiv_id: str) -> Optional[Dict]:
        clean_id = arxiv_id.split('v')[0]
        url = f"{self.BASE_URL}/paper/arXiv:{clean_id}"
        params = {"fields": "paperId,title,citationCount,influentialCitationCount,publicationDate"}
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
            else:
                print(f"⚠️ S2 API error {response.status_code} for {arxiv_id}")
                return None
        except Exception as e:
            print(f"⚠️ Error fetching S2 data for {arxiv_id}: {e}")
            return None
    
    def enrich_paper(self, paper: Dict) -> Dict:
        arxiv_id = paper.get("arxiv_id")
        if not arxiv_id:
            return paper
        s2_data = self._search_by_arxiv_id(arxiv_id)
        if s2_data:
            paper["s2_paper_id"] = s2_data.get("paperId")
            paper["citation_count"] = s2_data.get("citationCount", 0)
            paper["influential_citation_count"] = s2_data.get("influentialCitationCount", 0)
            print(f"   ✓ Enriched: {paper['title'][:50]}... ({paper['citation_count']} citations)")
        else:
            paper["s2_paper_id"] = None
            paper["citation_count"] = 0
            paper["influential_citation_count"] = 0
            print(f"   ⚠️ Not found in S2: {paper['title'][:50]}...")
        return paper
    
    def enrich_papers(self, papers: List[Dict], delay: float = 0.5) -> List[Dict]:
        print(f"\n📚 Enriching {len(papers)} papers with Semantic Scholar data...")
        enriched_papers = []
        for i, paper in enumerate(papers, 1):
            print(f"\n[{i}/{len(papers)}] Processing: {paper['title'][:60]}...")
            enriched_paper = self.enrich_paper(paper)
            enriched_papers.append(enriched_paper)
            if i < len(papers):
                time.sleep(delay)
        print(f"\n✅ Enrichment complete!")
        found_count = sum(1 for p in enriched_papers if p.get("s2_paper_id"))
        total_citations = sum(p.get("citation_count", 0) for p in enriched_papers)
        print(f"\n📊 Summary:")
        print(f"   • Papers found in S2: {found_count}/{len(papers)}")
        print(f"   • Total citations: {total_citations}")
        print(f"   • Average citations: {total_citations/len(papers):.1f}")
        return enriched_papers
    
    def save_enriched_papers(self, papers: List[Dict], output_path: Path):
        import json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, default=str)
        print(f"\n💾 Saved {len(papers)} enriched papers to {output_path}")
