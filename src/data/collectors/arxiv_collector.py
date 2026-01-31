"""
arXiv paper collector for Swiss AI research - FIXED VERSION
"""

import arxiv
from typing import List, Dict
from datetime import datetime
import time
from pathlib import Path


class ArxivCollector:
    """Collect papers from arXiv API"""
    
    SWISS_INSTITUTIONS = [
        "ETH Zurich",
        "EPFL",
        "University of Zurich",
        "IDSIA",
    ]
    
    AI_CATEGORIES = [
        "cs.AI",
        "cs.LG",
        "cs.CV",
        "cs.CL",
        "cs.RO",
        "cs.NE",
        "stat.ML",
    ]
    
    def __init__(self, max_results_per_institution: int = 50):
        self.max_results_per_institution = max_results_per_institution
        self.client = arxiv.Client()
    
    def _extract_author_info(self, arxiv_author) -> Dict:
        author_str = str(arxiv_author)
        return {"name": author_str.strip(), "affiliations": []}
    
    def collect_papers(self, start_year: int = 2020) -> List[Dict]:
        all_papers = []
        seen_ids = set()
        
        print(f"🔍 Searching arXiv for Swiss AI papers (from {start_year})...")
        
        # Search for each institution separately
        for institution in self.SWISS_INSTITUTIONS:
            print(f"\n🏛️ Searching for: {institution}")
            
            # Build query: institution AND (AI categories)
            cat_query = " OR ".join([f"cat:{cat}" for cat in self.AI_CATEGORIES])
            query = f'({cat_query}) AND all:"{institution}"'
            
            search = arxiv.Search(
                query=query,
                max_results=self.max_results_per_institution,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            try:
                results = list(self.client.results(search))
                print(f"   📥 Found {len(results)} papers")
                
                for result in results:
                    # Check publication year
                    if result.published.year < start_year:
                        continue
                    
                    # Skip duplicates
                    paper_id = result.entry_id.split("/")[-1]
                    if paper_id in seen_ids:
                        continue
                    
                    seen_ids.add(paper_id)
                    
                    # Extract paper data
                    paper_data = {
                        "arxiv_id": paper_id,
                        "title": result.title,
                        "authors": [self._extract_author_info(author) for author in result.authors],
                        "institution": institution,
                        "abstract": result.summary,
                        "categories": result.categories,
                        "published": result.published,
                        "updated": result.updated,
                        "arxiv_url": result.entry_id,
                        "pdf_url": result.pdf_url,
                    }
                    
                    all_papers.append(paper_data)
                    print(f"   ✓ {result.title[:60]}...")
                
                # Be nice to arXiv API
                time.sleep(3)
                
            except Exception as e:
                print(f"   ⚠️ Error searching {institution}: {e}")
                continue
        
        print(f"\n✅ Found {len(all_papers)} total Swiss AI papers")
        return all_papers
    
    def save_papers(self, papers: List[Dict], output_path: Path):
        import json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, default=str)
        print(f"💾 Saved {len(papers)} papers to {output_path}")
