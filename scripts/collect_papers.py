"""
Main data collection script for Research Navigator

Orchestrates:
1. arXiv paper collection (Swiss AI institutions)
2. Semantic Scholar enrichment (citations)
3. Data validation and storage
"""

# Add project root to Python path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from datetime import datetime
import json
from collections import Counter

from src.data.collectors.arxiv_collector import ArxivCollector
from src.data.collectors.semantic_scholar import SemanticScholarEnricher


def analyze_collection(papers: list) -> dict:
    """Generate statistics about collected papers"""
    
    stats = {
        "total_papers": len(papers),
        "date_range": {
            "earliest": min(p["published"] for p in papers),
            "latest": max(p["published"] for p in papers),
        },
        "institutions": dict(Counter(p["institution"] for p in papers)),
        "categories": dict(Counter(
            cat for p in papers for cat in p.get("categories", [])
        )),
        "total_citations": sum(p.get("citation_count", 0) for p in papers),
        "avg_citations": sum(p.get("citation_count", 0) for p in papers) / len(papers) if papers else 0,
    }
    
    return stats


def print_collection_summary(papers: list, stats: dict):
    """Pretty print collection summary"""
    
    print("\n" + "="*70)
    print("📊 COLLECTION SUMMARY")
    print("="*70)
    
    print(f"\n📚 Total Papers: {stats['total_papers']}")
    print(f"📅 Date Range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
    
    print(f"\n🏛️ By Institution:")
    for inst, count in sorted(stats['institutions'].items(), key=lambda x: x[1], reverse=True):
        print(f"   • {inst}: {count} papers")
    
    print(f"\n🔬 Top Research Categories:")
    top_cats = sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True)[:10]
    for cat, count in top_cats:
        print(f"   • {cat}: {count}")
    
    print(f"\n📈 Citations:")
    print(f"   • Total: {stats['total_citations']}")
    print(f"   • Average per paper: {stats['avg_citations']:.1f}")
    
    if papers:
        # Find most cited paper
        most_cited = max(papers, key=lambda p: p.get("citation_count", 0))
        print(f"\n🌟 Most Cited Paper:")
        print(f"   • {most_cited['title']}")
        print(f"   • {most_cited.get('citation_count', 0)} citations")
        print(f"   • {most_cited['institution']}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Collect Swiss AI research papers from arXiv and enrich with Semantic Scholar"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=200,
        help="Maximum papers to fetch from arXiv (default: 200)"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2020,
        help="Only collect papers from this year onwards (default: 2020)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for collected data (default: data/raw)"
    )
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Skip Semantic Scholar enrichment (faster, but no citations)"
    )
    parser.add_argument(
        "--s2-api-key",
        type=str,
        default=None,
        help="Semantic Scholar API key (optional, for higher rate limits)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 RESEARCH NAVIGATOR - DATA COLLECTION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"   • Max results from arXiv: {args.max_results}")
    print(f"   • Start year: {args.start_year}")
    print(f"   • Output directory: {args.output_dir}")
    print(f"   • Enrich with S2: {not args.skip_enrichment}")
    print()
    
    # Step 1: Collect from arXiv
    print("\n" + "="*70)
    print("STEP 1: Collecting papers from arXiv")
    print("="*70)
    
    collector = ArxivCollector(max_results_per_institution=args.max_results)
    papers = collector.collect_papers(start_year=args.start_year)
    
    if not papers:
        print("\n❌ No papers collected. Exiting.")
        return
    
    # Save raw arXiv data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arxiv_output = args.output_dir / f"arxiv_papers_{timestamp}.json"
    collector.save_papers(papers, arxiv_output)
    
    # Step 2: Enrich with Semantic Scholar (optional)
    if not args.skip_enrichment:
        print("\n" + "="*70)
        print("STEP 2: Enriching with Semantic Scholar")
        print("="*70)
        
        enricher = SemanticScholarEnricher(api_key=args.s2_api_key)
        papers = enricher.enrich_papers(papers)
        
        # Save enriched data
        enriched_output = args.output_dir / f"papers_enriched_{timestamp}.json"
        enricher.save_enriched_papers(papers, enriched_output)
    
    # Step 3: Analyze and summarize
    print("\n" + "="*70)
    print("STEP 3: Analysis")
    print("="*70)
    
    stats = analyze_collection(papers)
    print_collection_summary(papers, stats)
    
    # Save statistics
    stats_output = args.output_dir / f"collection_stats_{timestamp}.json"
    with open(stats_output, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"\n💾 Statistics saved to: {stats_output}")
    
    print("\n✅ Data collection complete!")
    print(f"\n📂 Output files:")
    print(f"   • Raw arXiv data: {arxiv_output}")
    if not args.skip_enrichment:
        print(f"   • Enriched data: {enriched_output}")
    print(f"   • Statistics: {stats_output}")
    print()


if __name__ == "__main__":
    main()