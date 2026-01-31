"""
Web Research Agent - Combines LLM parametric knowledge with ArXiv search.

This agent:
1. Uses GPT-5.2 to synthesize knowledge from training data
2. Extracts paper references mentioned by the LLM
3. Optionally searches ArXiv for those specific papers
"""

import re
import time
from typing import Dict, List, Optional
from openai import OpenAI
import arxiv

from src.agents.base_agent import BaseAgent, AgentResponse
from src.rag.config import RAGConfig


class WebResearchAgent(BaseAgent):
    """Agent that combines LLM parametric knowledge with targeted ArXiv search."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Web Research Agent.
        
        Args:
            config: Configuration dictionary with:
                - search_arxiv: bool, whether to search ArXiv for mentioned papers
                - max_arxiv_results: int, max papers to fetch per reference
        """
        super().__init__(agent_name="web_research_agent", config=config)
        
        # Load OpenAI config
        self.rag_config = RAGConfig()
        self.client = OpenAI(api_key=self.rag_config.openai_api_key)
        
        # Agent configuration
        self.search_arxiv = self.config.get("search_arxiv", False)
        self.max_arxiv_results = self.config.get("max_arxiv_results", 3)
        
        self.logger.info(
            f"Web Research Agent initialized with search_arxiv={self.search_arxiv}"
        )
    
    async def process(self, query: str) -> AgentResponse:
        """
        Process query using LLM parametric knowledge + optional ArXiv search.
        
        Args:
            query: Research question
            
        Returns:
            AgentResponse with LLM synthesis and optional ArXiv papers
        """
        self.logger.info(f"Processing query: {query}...")
        
        try:
            # Step 1: Get LLM parametric knowledge
            self.logger.info("Step 1: Querying GPT-5.2 for parametric knowledge...")
            llm_response = self._query_llm(query)
            
            # Step 2: Extract paper references from LLM response
            self.logger.info("Step 2: Extracting paper references...")
            paper_references = self._extract_references(llm_response)
            self.logger.info(f"Found {len(paper_references)} paper references")
            
            # Step 3: Optionally search ArXiv for mentioned papers
            arxiv_papers = []
            if self.search_arxiv and paper_references:
                self.logger.info("Step 3: Searching ArXiv for mentioned papers...")
                arxiv_papers = await self._search_arxiv_for_papers(paper_references)
                self.logger.info(f"Retrieved {len(arxiv_papers)} papers from ArXiv")
            
            # Prepare response
            content = {
                "llm_synthesis": llm_response,
                "paper_references": paper_references,
                "arxiv_papers": arxiv_papers
            }
            
            metadata = {
                "source": "web_research",
                "llm_model": "gpt-5.2",
                "references_found": len(paper_references),
                "arxiv_papers_retrieved": len(arxiv_papers),
                "citation_level": "level_1"  # Parametric knowledge
            }
            
            self.logger.info(
                f"Web research complete: {len(paper_references)} refs, "
                f"{len(arxiv_papers)} ArXiv papers"
            )
            
            return self._create_response(
                content=content,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error in web research: {e}")
            return self._create_response(
                content={},
                metadata={"error": str(e)},
                success=False,
                error_message=str(e)
            )
    
    def _query_llm(self, query: str) -> str:
        """
        Query GPT-5.2 for parametric knowledge.
        
        Args:
            query: Research question
            
        Returns:
            LLM's synthesized response
        """
        system_prompt = (
            "You are an academic researcher writing a literature review. "
            "Synthesize dominant research directions relevant to the query, compare approaches, and highlight trade-offs."
            "Focus on the most influential and representative papers."
            "\n\n"
            "Citation Requirements:\n"
            "- Cite 4-6 key papers from your training data that are most relevant to the query\n"
            "- ALWAYS use this exact format: Author et al. (Year). \"Paper Title\"\n"
            "- Example: McMahan et al. (2017). \"Communication-Efficient Learning of Deep Networks from Decentralized Data\"\n"
            "- Include specific methods and findings where relevant\n"
            "\n"
            "Style Requirements:\n"
            "- Write in formal academic prose suitable for publication\n"
            "- Be comprehensive but concise (~600 words)\n"
            "- Avoid speculation and stick to well-established research\n"
            "- DO NOT include conversational elements like 'If you tell me...', 'Let me know...', or 'I can help...'\n"
            "- DO NOT end with offers to provide more information\n"
            "- Focus purely on synthesizing the research literature"
        )
        
        response = self.client.chat.completions.create(
            model="gpt-5.2",
            temperature=0.5,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
        
        return response.choices[0].message.content
    
    def _extract_references(self, text: str) -> List[Dict[str, str]]:
        """
        Extract paper references from LLM response.
        
        Looks for patterns like:
        - McMahan et al. (2017). "Paper Title"
        - Kairouz et al. (2021). "Title with quotes"
        - Smith and Jones (2020)
        
        Args:
            text: LLM response text
            
        Returns:
            List of reference dictionaries with 'authors', 'year', and 'title' (if available)
        """
        references = []
        
        # Pattern 1: Author et al. (Year). "Title"
        pattern_with_title = r'([A-Z][a-z]+(?:\s+et\s+al\.?)?)\s*\((\d{4})\)[.,\s]*["\']([^"\']+)["\']'
        matches_with_title = re.finditer(pattern_with_title, text)
        for match in matches_with_title:
            references.append({
                "authors": match.group(1),
                "year": match.group(2),
                "title": match.group(3).strip(),
                "citation_format": f"{match.group(1)} ({match.group(2)})"
            })
        
        # Pattern 2: (Author et al., Year). "Title"
        pattern_paren_with_title = r'\(([A-Z][a-z]+(?:\s+et\s+al\.?)?),\s*(\d{4})\)[.,\s]*["\']([^"\']+)["\']'
        matches_paren_with_title = re.finditer(pattern_paren_with_title, text)
        for match in matches_paren_with_title:
            references.append({
                "authors": match.group(1),
                "year": match.group(2),
                "title": match.group(3).strip(),
                "citation_format": f"({match.group(1)}, {match.group(2)})"
            })
        
        # Track what we already have (to avoid duplicates)
        seen = {(ref["authors"], ref["year"]) for ref in references}
        
        # Pattern 3: Author et al. (Year) - without title (fallback)
        pattern_no_title = r'([A-Z][a-z]+(?:\s+et\s+al\.?)?)\s*\((\d{4})\)'
        matches_no_title = re.finditer(pattern_no_title, text)
        for match in matches_no_title:
            key = (match.group(1), match.group(2))
            if key not in seen:
                references.append({
                    "authors": match.group(1),
                    "year": match.group(2),
                    "title": "",  # No title available
                    "citation_format": f"{match.group(1)} ({match.group(2)})"
                })
                seen.add(key)
        
        # Pattern 4: (Author et al., Year) - without title (fallback)
        pattern_paren_no_title = r'\(([A-Z][a-z]+(?:\s+et\s+al\.?)?),\s*(\d{4})\)'
        matches_paren_no_title = re.finditer(pattern_paren_no_title, text)
        for match in matches_paren_no_title:
            key = (match.group(1), match.group(2))
            if key not in seen:
                references.append({
                    "authors": match.group(1),
                    "year": match.group(2),
                    "title": "",  # No title available
                    "citation_format": f"({match.group(1)}, {match.group(2)})"
                })
                seen.add(key)
        
        # Remove exact duplicates while preserving order
        unique_refs = []
        seen_full = set()
        for ref in references:
            key = (ref["authors"], ref["year"], ref.get("title", ""))
            if key not in seen_full:
                seen_full.add(key)
                unique_refs.append(ref)
        
        return unique_refs
    
    async def _search_arxiv_for_papers(
        self, 
        references: List[Dict[str, str]]
    ) -> List[Dict]:
        """
        Search ArXiv for specific papers mentioned by LLM.
        
        Args:
            references: List of paper references (authors, year)
            
        Returns:
            List of ArXiv paper metadata
        """
        arxiv_papers = []
        
        for ref in references[:self.max_arxiv_results]:
            try:
                # Respect ArXiv rate limits (3 seconds between requests)
                time.sleep(3)
                
                # Build search query
                search_query = f"{ref['authors']} {ref['year']}"
                
                self.logger.info(f"Searching ArXiv for: {search_query}")
                
                # Search ArXiv
                search = arxiv.Search(
                    query=search_query,
                    max_results=1,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                
                # Get first result
                for paper in search.results():
                    arxiv_papers.append({
                        "title": paper.title,
                        "authors": [a.name for a in paper.authors],
                        "published": paper.published.strftime("%Y-%m-%d"),
                        "arxiv_id": paper.entry_id.split('/')[-1],
                        "pdf_url": paper.pdf_url,
                        "summary": paper.summary,
                        "matched_reference": ref["citation_format"]
                    })
                    break  # Only take first result
                    
            except Exception as e:
                self.logger.warning(f"Error searching ArXiv for {ref}: {e}")
                continue
        
        return arxiv_papers