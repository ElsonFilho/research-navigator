"""
Citation extraction utilities
Extracts and counts citations from text responses
"""
import re
from typing import List, Dict, Any


class CitationExtractor:
    """Extract citations from academic text"""
    
    def __init__(self):
        # Citation patterns (order matters - most specific first)
        self.patterns = {
            # (Author et al., 2020) or (SingleAuthor, 2020)
            'author_year_paren': r'\([A-Z][A-Za-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)',
            # Author et al. (2020) or SingleAuthor (2020)
            'author_year_inline': r'\b[A-Z][A-Za-z]+\s+(?:et\s+al\.\s*)?\(\d{4}\)',
            # [1], [2], etc.
            'numbered': r'\[\d+\]'
        }
    
    def extract_citations(self, text: str) -> Dict[str, Any]:
        """
        Extract all citations from text
        
        Returns:
            {
                'citations': List of unique citation strings,
                'count': Total number of unique citations,
                'by_pattern': Dict with counts per pattern type
            }
        """
        all_citations = []
        by_pattern = {}
        
        for pattern_name, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            by_pattern[pattern_name] = len(matches)
            all_citations.extend(matches)
        
        # Get unique citations
        unique_citations = list(set(all_citations))
        
        return {
            'citations': unique_citations,
            'count': len(unique_citations),
            'by_pattern': by_pattern,
            'total_mentions': len(all_citations)  # Including duplicates
        }
    
    def format_citations_for_display(self, citations: List[str]) -> str:
        """Format citation list for display"""
        if not citations:
            return "No citations found"
        
        return ", ".join(sorted(citations))


def extract_baseline_citations(response_text: str) -> int:
    """
    Quick function to get citation count from baseline response
    
    Args:
        response_text: The baseline response text
        
    Returns:
        Number of unique citations found
    """
    extractor = CitationExtractor()
    result = extractor.extract_citations(response_text)
    return result['count']
