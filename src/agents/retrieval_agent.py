"""
Retrieval Agent - Corpus Search
Wraps existing retriever.py to work within multi-agent system
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, AgentResponse, AgentConfig
from ..retrieval.retriever import Retriever


class RetrievalAgent(BaseAgent):
    """
    Retrieval Agent for corpus search (296 papers, 12,109 chunks)
    
    Responsibilities:
    - Semantic search in ChromaDB
    - Return top-k relevant papers
    - Provide relevance scores (converted from distance)
    - Include full text and metadata
    
    Citation Level: Level 2 (metadata validation)
    Source: Corpus (curated, high-quality)
    """
    
    def __init__(self, retriever: Retriever, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Retrieval Agent
        
        Args:
            retriever: Existing Retriever instance
            config: Optional configuration (top_k, threshold, etc.)
        """
        super().__init__("retrieval_agent", config)
        
        self.retriever = retriever
        self.top_k = config.get("top_k", AgentConfig.TOP_K_RESULTS) if config else AgentConfig.TOP_K_RESULTS
        self.threshold = config.get("threshold", AgentConfig.RELEVANCE_THRESHOLD) if config else AgentConfig.RELEVANCE_THRESHOLD
        
        self.logger.info(f"Retrieval Agent initialized with top_k={self.top_k}, threshold={self.threshold}")
    
    async def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Process retrieval query
        
        Args:
            query: Search query
            context: Optional context (not used for retrieval)
            
        Returns:
            AgentResponse with retrieved papers
        """
        self._log_processing(query)
        
        try:
            # Perform retrieval using existing retriever
            results = self.retriever.search(query, n_results=self.top_k)
            
            # Convert distance to relevance score (1 - distance/2)
            # Lower distance = higher relevance
            # Filter by relevance threshold
            filtered_results = []
            for r in results:
                relevance = 1 - (r.distance / 2)  # Convert distance to similarity
                if relevance >= self.threshold:
                    filtered_results.append({
                        'chunk_id': r.chunk_id,
                        'content': r.content,
                        'metadata': {
                            'title': r.paper_title,
                            'authors': r.authors,
                            'year': r.year,
                            'arxiv_id': r.arxiv_id,
                            'chunk_index': r.chunk_index
                        },
                        'relevance_score': relevance,
                        'distance': r.distance
                    })
            
            self.logger.info(
                f"Retrieved {len(results)} papers, "
                f"{len(filtered_results)} above threshold ({self.threshold})"
            )
            
            # Format metadata
            metadata = {
                "total_retrieved": len(results),
                "filtered_count": len(filtered_results),
                "top_k": self.top_k,
                "threshold": self.threshold,
                "source": "corpus",
                "citation_level": "level_2"  # Metadata validation
            }
            
            # Log top results
            if filtered_results:
                top_result = filtered_results[0]
                self.logger.info(
                    f"Top result: {top_result['metadata']['title']} "
                    f"(score: {top_result['relevance_score']:.3f})"
                )
            else:
                self.logger.warning("No results above relevance threshold")
            
            self._log_success(f"Retrieved {len(filtered_results)} relevant papers")
            
            return self._create_response(
                content=filtered_results,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            self._log_error(e)
            return self._create_response(
                content=[],
                metadata={"error": str(e)},
                success=False,
                error_message=f"Retrieval failed: {str(e)}"
            )
    
    def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific paper
        
        Args:
            paper_id: Paper identifier (arxiv_id)
            
        Returns:
            Paper details or None if not found
        """
        try:
            self.logger.info(f"Requesting details for paper: {paper_id}")
            # This would use retriever's get_by_id if available
            # For now, we'll note it as a future enhancement
            return None
        except Exception as e:
            self.logger.error(f"Error getting paper details: {e}")
            return None


# Example usage for testing
if __name__ == "__main__":
    import asyncio
    
    async def test_retrieval_agent():
        """Test the retrieval agent"""
        print("Testing Retrieval Agent...")
        print("✅ Retrieval Agent module loaded successfully")
        print("   Ready to wrap existing retriever.py")
    
    asyncio.run(test_retrieval_agent())