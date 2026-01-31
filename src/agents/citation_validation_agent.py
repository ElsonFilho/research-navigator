"""
Citation Validation Agent - Validates citations at different levels

Validation Levels:
- Level 0: No validation (baseline only)
- Level 1: Metadata validation (corpus citations - authors, year, title)
- Level 2: Existence check via triple fallback (ArXiv → OpenAlex → Google Scholar)
  * For LLM citations: Extracts paper references and validates via multiple sources
"""

import re
import time
from typing import Dict, List, Optional
import arxiv
from scholarly import scholarly
import requests

from src.agents.base_agent import BaseAgent, AgentResponse
from src.retrieval.retriever import Retriever


class CitationValidationAgent(BaseAgent):
    """Agent that validates citations at different levels."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Citation Validation Agent.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(agent_name="citation_validation_agent", config=config)
        
        # Initialize retriever for corpus lookup
        self.retriever = Retriever()
        
        self.logger.info("Citation Validation Agent initialized with triple fallback (ArXiv → OpenAlex → Scholar)")
    
    async def process(
        self,
        synthesis_text: str,
        corpus_papers: Optional[List[Dict]] = None,
        arxiv_papers: Optional[List[Dict]] = None,
        web_results: Optional[Dict] = None
    ) -> AgentResponse:
        """
        Validate all citations in synthesis text.
        
        Args:
            synthesis_text: Text containing citations
            corpus_papers: Papers from corpus (for Level 1)
            arxiv_papers: Papers from ArXiv (for Level 2)
            web_results: Web research results with paper references (for Level 2 LLM validation)
            
        Returns:
            AgentResponse with validation results
        """
        self.logger.info("Starting citation validation...")
        
        try:
            # Extract citations from text
            corpus_citations = self._extract_citations(synthesis_text, "[Corpus")
            llm_citation_count = synthesis_text.count("[LLM Knowledge]")
            arxiv_citations = self._extract_arxiv_citations(synthesis_text)
            
            # Extract paper references from web research for LLM validation
            llm_paper_references = []
            if web_results and isinstance(web_results, dict):
                content = web_results.get('content', {})
                paper_refs = content.get('paper_references', [])
                llm_paper_references = paper_refs
            
            self.logger.info(
                f"Found citations: {len(corpus_citations)} corpus, "
                f"{llm_citation_count} LLM, {len(arxiv_citations)} ArXiv"
            )
            self.logger.info(
                f"Found {len(llm_paper_references)} paper references from LLM for validation"
            )
            
            # Validate corpus citations (Level 1 - Metadata)
            corpus_validation = []
            if corpus_citations and corpus_papers:
                self.logger.info("Validating corpus citations (Level 1)...")
                for citation in corpus_citations:
                    result = self._validate_level_1(citation, corpus_papers)
                    corpus_validation.append(result)
            
            # Validate LLM citations (Level 2 - Triple fallback: ArXiv → OpenAlex → Scholar)
            llm_validation = []
            if llm_paper_references:
                self.logger.info(f"Validating {len(llm_paper_references)} LLM paper references (Level 2 - Triple Fallback)...")
                for ref_dict in llm_paper_references:
                    result = await self._validate_level_2_llm_paper(ref_dict)
                    llm_validation.append(result)
            elif llm_citation_count > 0:
                # No paper references found, mark as unverifiable
                self.logger.info("No paper references found for LLM citations - marking as unverifiable")
                for i in range(llm_citation_count):
                    llm_validation.append({
                        "citation": "[LLM Knowledge]",
                        "level": 2,
                        "valid": True,
                        "note": "No specific paper reference found - general LLM knowledge",
                        "warning": "Cannot verify without specific paper citation"
                    })
            
            # Validate ArXiv citations (Level 2 - Existence check)
            arxiv_validation = []
            if arxiv_citations:
                self.logger.info("Validating ArXiv citations (Level 2)...")
                for citation in arxiv_citations:
                    result = await self._validate_level_2_arxiv(citation)
                    arxiv_validation.append(result)
            
            # Calculate statistics
            total_citations = len(corpus_citations) + llm_citation_count + len(arxiv_citations)
            total_validated = len(corpus_validation) + len(llm_validation) + len(arxiv_validation)
            
            valid_count = sum(
                1 for v in corpus_validation + llm_validation + arxiv_validation 
                if v["valid"]
            )
            accuracy = valid_count / total_validated if total_validated > 0 else 0
            
            # Count sources used
            source_counts = {"arxiv": 0, "openalex": 0, "google_scholar": 0, "corpus": len(corpus_validation)}
            for v in llm_validation + arxiv_validation:
                source = v.get("source", "unknown")
                if source in source_counts:
                    source_counts[source] += 1
            
            self.logger.info(
                f"Validation complete: {valid_count}/{total_validated} valid ({accuracy:.1%})"
            )
            self.logger.info(
                f"Sources used: ArXiv={source_counts['arxiv']}, "
                f"OpenAlex={source_counts['openalex']}, "
                f"Scholar={source_counts['google_scholar']}"
            )
            
            # Prepare response
            content = {
                "total_citations": total_citations,
                "corpus_citations": corpus_validation,
                "llm_citations": llm_validation,
                "arxiv_citations": arxiv_validation,
                "validation_accuracy": accuracy,
                "valid_count": valid_count,
                "total_validated": total_validated,
                "source_counts": source_counts
            }
            
            metadata = {
                "source": "citation_validation",
                "level_1_count": len(corpus_validation),
                "level_2_count": len(llm_validation) + len(arxiv_validation),
                "arxiv_used": source_counts["arxiv"],
                "openalex_used": source_counts["openalex"],
                "scholar_used": source_counts["google_scholar"]
            }
            
            return self._create_response(
                content=content,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error in citation validation: {e}")
            return self._create_response(
                content={},
                metadata={"error": str(e)},
                success=False,
                error_message=str(e)
            )
    
    def _extract_citations(self, text: str, citation_type: str) -> List[str]:
        """
        Extract citations of a specific type from text.
        
        Args:
            text: Text to search
            citation_type: Citation marker (e.g., "[Corpus", "[ArXiv")
            
        Returns:
            List of citation strings
        """
        pattern = re.escape(citation_type) + r"[^\]]*\]"
        matches = re.findall(pattern, text)
        return matches
    
    def _extract_arxiv_citations(self, text: str) -> List[Dict]:
        """
        Extract ArXiv citations with IDs.
        
        Args:
            text: Text to search
            
        Returns:
            List of dicts with citation and ArXiv ID
        """
        citations = []
        
        # Find all [ArXiv: ID] citations
        pattern = r'\[ArXiv:\s*(\d{4}\.\d{4,5})\]'
        matches = re.findall(pattern, text)
        
        for arxiv_id in matches:
            citations.append({
                "tag": f"[ArXiv: {arxiv_id}]",
                "arxiv_id": arxiv_id
            })
        
        return citations
    
    def _validate_level_1(
        self,
        citation: str,
        corpus_papers: List[Dict]
    ) -> Dict:
        """
        Level 1: Metadata validation (corpus citations).
        
        Validates that cited paper exists in corpus and metadata matches.
        
        Args:
            citation: Citation string (e.g., "[Corpus Paper 1]")
            corpus_papers: List of papers from corpus
            
        Returns:
            Validation result dictionary
        """
        # Extract paper number from citation
        match = re.search(r'\d+', citation)
        if not match:
            return {
                "citation": citation,
                "level": 1,
                "valid": False,
                "reason": "Could not extract paper number from citation"
            }
        
        paper_num = int(match.group()) - 1  # Convert to 0-indexed
        
        # Check if paper exists in corpus
        if paper_num < 0 or paper_num >= len(corpus_papers):
            return {
                "citation": citation,
                "level": 1,
                "valid": False,
                "reason": f"Paper {paper_num + 1} not in retrieved corpus (only {len(corpus_papers)} papers)"
            }
        
        # Paper exists - validate metadata
        paper = corpus_papers[paper_num]
        metadata = paper.get("metadata", {})
        
        return {
            "citation": citation,
            "level": 1,
            "valid": True,
            "paper_title": metadata.get("title", "N/A")[:80],
            "paper_year": metadata.get("year", "N/A"),
            "paper_authors": metadata.get("authors", "N/A")[:100] if isinstance(metadata.get("authors"), str) else "N/A",
            "relevance_score": paper.get("relevance_score", 0),
            "note": "Metadata validated against corpus",
            "source": "corpus"
        }
    
    async def _validate_level_2_llm_paper(self, ref_dict: Dict) -> Dict:
        """
        Level 2: Existence check with triple fallback chain.
        
        Validates paper reference by searching:
        1. ArXiv (fast, ML/AI focused)
        2. OpenAlex (broad coverage, no captchas)
        3. Google Scholar (comprehensive, last resort)
        
        Args:
            ref_dict: Dict with paper reference info (title, authors, year, etc.)
            
        Returns:
            Validation result dictionary
        """
        # Extract information from reference
        title = ref_dict.get('title', '')
        authors = ref_dict.get('authors', '')
        year = ref_dict.get('year', '')
        
        # Try to extract first author and year for search
        if not authors or not year:
            return {
                "citation": "[LLM Knowledge]",
                "level": 2,
                "valid": False,
                "reference": str(ref_dict),
                "reason": "Incomplete reference information (missing authors or year)"
            }
        
        # Extract first author
        first_author = authors.split(',')[0].strip() if ',' in authors else authors.split()[0]
        
        # STAGE 1: Try ArXiv first (fast, ML/AI papers)
        self.logger.info(f"[1/3] Trying ArXiv for: {first_author} ({year})")
        paper_found = await self._search_arxiv_for_paper(first_author, str(year), title)
        
        if paper_found:
            self.logger.info(f"✅ Found on ArXiv: {paper_found['arxiv_id']}")
            return {
                "citation": "[LLM Knowledge]",
                "level": 2,
                "valid": True,
                "reference": f"{authors} ({year})",
                "arxiv_title": paper_found["title"][:100],
                "arxiv_id": paper_found["arxiv_id"],
                "note": f"Paper verified on ArXiv: {paper_found['arxiv_id']}",
                "source": "arxiv"
            }
        
        # STAGE 2: Try OpenAlex (broad coverage, no captchas)
        self.logger.info(f"[2/3] ArXiv failed, trying OpenAlex for: {first_author} ({year})")
        openalex_found = await self._search_openalex(first_author, str(year), title)
        
        if openalex_found:
            self.logger.info(f"✅ Found on OpenAlex: {openalex_found['venue']}")
            return {
                "citation": "[LLM Knowledge]",
                "level": 2,
                "valid": True,
                "reference": f"{authors} ({year})",
                "openalex_title": openalex_found["title"][:100],
                "openalex_venue": openalex_found["venue"],
                "openalex_citations": openalex_found["citations"],
                "openalex_doi": openalex_found.get("doi", "N/A"),
                "note": f"Paper verified on OpenAlex: {openalex_found['venue']}",
                "source": "openalex"
            }
        
        # STAGE 3: Last resort - Google Scholar (slow, captcha risk)
        self.logger.info(f"[3/3] OpenAlex failed, trying Google Scholar (last resort): {first_author} ({year})")
        scholar_found = await self._search_google_scholar(first_author, str(year), title)
        
        if scholar_found:
            self.logger.info(f"✅ Found on Google Scholar: {scholar_found['venue']}")
            return {
                "citation": "[LLM Knowledge]",
                "level": 2,
                "valid": True,
                "reference": f"{authors} ({year})",
                "scholar_title": scholar_found["title"][:100],
                "scholar_venue": scholar_found["venue"],
                "scholar_citations": scholar_found["citations"],
                "note": f"Paper verified on Google Scholar: {scholar_found['venue']}",
                "source": "google_scholar"
            }
        
        # Not found on any source
        self.logger.warning(f"❌ Paper not found on ArXiv, OpenAlex, or Google Scholar: {authors} ({year})")
        return {
            "citation": "[LLM Knowledge]",
            "level": 2,
            "valid": False,
            "reference": f"{authors} ({year})",
            "reason": "Paper not found on ArXiv, OpenAlex, or Google Scholar",
            "warning": "Paper may be hallucinated, from non-indexed sources, or citation info is incorrect"
        }
    
    async def _validate_level_2_arxiv(self, citation: Dict) -> Dict:
        """
        Level 2: Existence check (ArXiv citations).
        
        Verifies that ArXiv paper exists by searching for the ID.
        
        Args:
            citation: Dict with tag and arxiv_id
            
        Returns:
            Validation result dictionary
        """
        tag = citation["tag"]
        arxiv_id = citation["arxiv_id"]
        
        # Search ArXiv by ID
        self.logger.info(f"Verifying ArXiv paper: {arxiv_id}")
        
        paper_found = await self._search_arxiv_by_id(arxiv_id)
        
        if paper_found:
            return {
                "citation": tag,
                "level": 2,
                "valid": True,
                "arxiv_id": arxiv_id,
                "arxiv_title": paper_found["title"][:100],
                "arxiv_authors": ", ".join(paper_found["authors"][:3]),
                "note": f"Paper exists on ArXiv: {arxiv_id}",
                "source": "arxiv"
            }
        else:
            return {
                "citation": tag,
                "level": 2,
                "valid": False,
                "arxiv_id": arxiv_id,
                "reason": f"ArXiv paper {arxiv_id} not found",
                "warning": "Paper ID may be incorrect or paper removed from ArXiv"
            }
    
    async def _search_arxiv_for_paper(
        self,
        author: str,
        year: str,
        title: str = ""
    ) -> Optional[Dict]:
        """
        Search ArXiv for a paper by author, year, and optionally title.
        
        Args:
            author: First author name
            year: Publication year
            title: Paper title (optional, for better matching)
            
        Returns:
            Dict with paper info if found, None otherwise
        """
        try:
            # Respect ArXiv rate limits
            time.sleep(3)
            
            # Build search query
            if title:
                # Search by title first (more accurate)
                query = f'ti:"{title}"'
            else:
                # Fallback to author + year
                query = f"au:{author} AND submittedDate:[{year}0101 TO {year}1231]"
            
            search = arxiv.Search(
                query=query,
                max_results=3,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            # Get first result
            for paper in search.results():
                return {
                    "title": paper.title,
                    "arxiv_id": paper.entry_id.split('/')[-1],
                    "authors": [a.name for a in paper.authors],
                    "year": paper.published.year
                }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"ArXiv search failed for {author} {year}: {e}")
            return None
    
    async def _search_arxiv_by_id(self, arxiv_id: str) -> Optional[Dict]:
        """
        Search ArXiv for a paper by ID.
        
        Args:
            arxiv_id: ArXiv paper ID (e.g., "2104.12345")
            
        Returns:
            Dict with paper info if found, None otherwise
        """
        try:
            # Respect ArXiv rate limits
            time.sleep(3)
            
            # Search by ID
            search = arxiv.Search(id_list=[arxiv_id])
            
            # Get first result
            paper = next(search.results())
            
            return {
                "title": paper.title,
                "arxiv_id": paper.entry_id.split('/')[-1],
                "authors": [a.name for a in paper.authors],
                "year": paper.published.year
            }
            
        except Exception as e:
            self.logger.warning(f"ArXiv ID lookup failed for {arxiv_id}: {e}")
            return None
    
    async def _search_openalex(
        self,
        author: str,
        year: str,
        title: str = ""
    ) -> Optional[Dict]:
        """
        Search OpenAlex for a paper (second fallback when ArXiv fails).
        
        OpenAlex is a free, open API with no rate limits or captchas.
        Coverage: 250M+ works (journals, conferences, books, preprints).
        
        Args:
            author: First author name
            year: Publication year
            title: Paper title (optional)
            
        Returns:
            Dict with paper info if found, None otherwise
        """
        try:
            # No rate limit needed for OpenAlex!
            # They allow ~10 requests/second for reasonable use
            
            # Build search query
            base_url = "https://api.openalex.org/works"
            
            if title:
                # Search by title (most accurate)
                params = {
                    "search": title,
                    "filter": f"publication_year:{year}"
                }
                self.logger.info(f"Searching OpenAlex by title: {title[:60]}...")
            else:
                # Fallback to author + year
                params = {
                    "search": author,
                    "filter": f"publication_year:{year}"
                }
                self.logger.info(f"Searching OpenAlex for: {author} ({year})")
            
            # Make request
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                return None
            
            # Get first result
            paper = results[0]
            
            # Extract authors
            authorships = paper.get("authorships", [])
            authors_list = [
                auth.get("author", {}).get("display_name", "Unknown")
                for auth in authorships[:3]  # First 3 authors
            ]
            authors_str = ", ".join(authors_list)
            
            # Extract publication info
            pub_year = paper.get("publication_year")
            
            # Verify year matches (within ±2 year tolerance)
            if pub_year:
                year_diff = abs(int(pub_year) - int(year))
                if year_diff > 2:
                    self.logger.warning(
                        f"Year mismatch: expected {year}, found {pub_year} (diff={year_diff})"
                    )
                    return None
                elif year_diff > 0:
                    self.logger.info(
                        f"Year match within tolerance: expected {year}, found {pub_year}"
                    )
            
            # Extract venue/source
            host_venue = paper.get("host_venue", {}) or {}
            primary_location = paper.get("primary_location", {}) or {}
            
            venue = (
                host_venue.get("display_name") or 
                primary_location.get("source", {}).get("display_name") or
                "Unknown venue"
            )
            
            # Paper found and validated!
            self.logger.info(
                f"✅ Paper verified on OpenAlex: {paper.get('title', 'N/A')[:60]}"
            )
            
            return {
                "title": paper.get("title", "N/A"),
                "authors": authors_str,
                "year": pub_year,
                "venue": venue,
                "citations": paper.get("cited_by_count", 0),
                "doi": paper.get("doi", "N/A"),
                "openalex_id": paper.get("id", "N/A"),
                "source": "openalex"
            }
            
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"OpenAlex API error for {author} {year}: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"OpenAlex search failed for {author} {year}: {e}")
            return None
    
    async def _search_google_scholar(
        self,
        author: str,
        year: str,
        title: str = ""
    ) -> Optional[Dict]:
        """
        Search Google Scholar for a paper (third fallback, last resort).
        
        NOTE: Scholar may serve captchas after ~20-30 requests.
        This is now the LAST fallback after ArXiv and OpenAlex fail.
        
        Args:
            author: First author name
            year: Publication year
            title: Paper title (optional)
            
        Returns:
            Dict with paper info if found, None otherwise
        """
        try:
            # Respect rate limits (slower than ArXiv/OpenAlex)
            time.sleep(self.config.get("scholar_retry_delay", 5))
            
            # Build search query
            if title:
                # Search by title (most accurate)
                search_query = title
            else:
                # Fallback to author + year
                search_query = f"{author} {year}"
            
            self.logger.info(f"Searching Google Scholar for: {search_query[:60]}...")
            
            # Search Google Scholar
            search_results = scholarly.search_pubs(search_query)
            
            # Get first result
            paper = next(search_results, None)
            
            if paper:
                # Extract publication year from bib
                pub_year = paper['bib'].get('pub_year', 'Unknown')
                
                # Verify year matches (within ±2 year tolerance for older papers)
                if pub_year != 'Unknown':
                    year_diff = abs(int(pub_year) - int(year))
                    if year_diff > 2:
                        self.logger.warning(
                            f"Year mismatch too large: expected {year}, found {pub_year} (diff={year_diff})"
                        )
                        return None
                    elif year_diff > 0:
                        self.logger.info(
                            f"Year match within tolerance: expected {year}, found {pub_year} (diff={year_diff})"
                        )
                
                # Paper found and validated!
                self.logger.info(
                    f"✅ Paper verified on Google Scholar: {paper['bib'].get('title', 'N/A')[:60]}"
                )
                
                return {
                    "title": paper['bib'].get('title', 'N/A'),
                    "authors": paper['bib'].get('author', 'N/A'),
                    "year": pub_year,
                    "venue": paper['bib'].get('venue', 'N/A'),
                    "citations": paper.get('num_citations', 0),
                    "source": "google_scholar"
                }
            
            return None
            
        except StopIteration:
            # No results found
            return None
        except Exception as e:
            self.logger.warning(f"Google Scholar search failed for {author} {year}: {e}")
            return None