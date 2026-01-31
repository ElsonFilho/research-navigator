"""
Coordinator Agent - Orchestrates the multi-agent system

This agent:
1. Analyzes incoming queries (with Query Decomposition Agent)
2. Routes to appropriate agents (Retrieval, Web Research)
3. Handles both simple and complex (decomposed) queries
4. Runs agents in parallel for efficiency
5. Synthesizes results from multiple sources
6. Validates citations
7. Returns final response with metadata
"""

import asyncio
from typing import Dict, List, Optional

from src.agents.base_agent import BaseAgent, AgentResponse
from src.agents.query_decomposition_agent import QueryDecompositionAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.web_research_agent import WebResearchAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.agents.citation_validation_agent import CitationValidationAgent
from src.retrieval.retriever import Retriever


class CoordinatorAgent(BaseAgent):
    """
    Coordinator that orchestrates the multi-agent RAG system.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Coordinator Agent.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(agent_name="coordinator_agent", config=config)
        
        # Initialize retriever (shared across retrieval agent instances)
        self.retriever = Retriever()
        
        # Configuration
        self.top_k = self.config.get("top_k", 3)
        self.threshold = self.config.get("threshold", 0.4)
        self.use_web_research = self.config.get("use_web_research", True)
        self.use_query_decomposition = self.config.get("use_query_decomposition", True)
        self.validate_citations = self.config.get("validate_citations", True)
        
        self.logger.info(
            f"Coordinator initialized: top_k={self.top_k}, "
            f"web_research={self.use_web_research}, "
            f"query_decomposition={self.use_query_decomposition}, "
            f"validate_citations={self.validate_citations}"
        )
    
    async def process(self, query: str) -> AgentResponse:
        """
        Process query through the multi-agent system.
        
        Workflow:
        1. Query Decomposition (if enabled and needed)
        2. Parallel: Retrieval Agent + Web Research Agent (for each query/sub-query)
        3. Synthesis Agent (combines all results)
        4. Citation Validation Agent (validates synthesis)
        5. Return final response
        
        Args:
            query: User's research question
            
        Returns:
            AgentResponse with final synthesis and metadata
        """
        self.logger.info(f"Coordinator processing query: {query}")
        
        try:
            # ============================================================
            # PHASE 0: QUERY DECOMPOSITION (if enabled)
            # ============================================================
            queries_to_process = [query]  # Default: just original query
            decomposition_used = False
            
            if self.use_query_decomposition:
                self.logger.info("Phase 0: Analyzing query complexity...")
                
                decomposition_response = await self._run_query_decomposition(query)
                
                if decomposition_response.content.get("needs_decomposition", False):
                    sub_queries = decomposition_response.content.get("sub_queries", [])
                    if sub_queries:
                        queries_to_process = sub_queries
                        decomposition_used = True
                        self.logger.info(
                            f"Query decomposed into {len(sub_queries)} sub-queries"
                        )
                else:
                    self.logger.info("Query is simple - no decomposition needed")
            
            # ============================================================
            # PHASE 1: PARALLEL RETRIEVAL (for each query/sub-query)
            # ============================================================
            self.logger.info(
                f"Phase 1: Running parallel retrieval for "
                f"{len(queries_to_process)} queries..."
            )
            
            all_corpus_results = []
            all_web_results = []
            
            # Process each query (or sub-query)
            for i, q in enumerate(queries_to_process, 1):
                self.logger.info(f"Processing query {i}/{len(queries_to_process)}: {q[:60]}...")
                
                # Create tasks for parallel execution
                tasks = []
                
                # Task 1: Corpus retrieval (always run)
                retrieval_task = self._run_retrieval_agent(q)
                tasks.append(retrieval_task)
                
                # Task 2: Web research (if enabled)
                if self.use_web_research:
                    web_task = self._run_web_research_agent(q)
                    tasks.append(web_task)
                
                # Run both in parallel
                results = await asyncio.gather(*tasks)
                
                corpus_response = results[0]
                web_response = results[1] if len(results) > 1 else None
                
                # Collect results
                if corpus_response.success:
                    all_corpus_results.extend(corpus_response.content)
                
                if web_response and web_response.success:
                    all_web_results.append(web_response)
            
            self.logger.info(
                f"Phase 1 complete: collected {len(all_corpus_results)} corpus papers, "
                f"{len(all_web_results)} web results"
            )
            
            # ============================================================
            # PHASE 2: SYNTHESIS
            # ============================================================
            self.logger.info("Phase 2: Synthesizing all results...")
            
            # Combine all results for synthesis
            combined_corpus = {
                "success": True,
                "content": all_corpus_results,
                "metadata": {
                    "total_papers": len(all_corpus_results),
                    "queries_processed": len(queries_to_process)
                }
            }
            
            # Combine web results
            combined_web = None
            if all_web_results:
                # Merge all LLM syntheses
                all_llm_syntheses = []
                all_references = []
                
                for web_resp in all_web_results:
                    content = web_resp.content
                    if content.get("llm_synthesis"):
                        all_llm_syntheses.append(content["llm_synthesis"])
                    if content.get("paper_references"):
                        all_references.extend(content["paper_references"])
                
                combined_web = {
                    "success": True,
                    "content": {
                        "llm_synthesis": "\n\n---\n\n".join(all_llm_syntheses),
                        "paper_references": all_references,
                        "arxiv_papers": []
                    },
                    "metadata": {
                        "references_found": len(all_references)
                    }
                }
            
            synthesis_response = await self._run_synthesis_agent(
                query=query,  # Use ORIGINAL query for synthesis context
                corpus_results=combined_corpus,
                web_results=combined_web
            )
            
            if not synthesis_response.success:
                self.logger.error("Synthesis failed")
                return synthesis_response
            
            self.logger.info("Phase 2 complete: synthesis generated")
            
            # ============================================================
            # PHASE 3: CITATION VALIDATION
            # ============================================================
            validation_response = None
            if self.validate_citations:
                self.logger.info("Phase 3: Validating citations...")
                
                validation_response = await self._run_citation_validation(
                    synthesis_text=synthesis_response.content["synthesis"],
                    corpus_papers=all_corpus_results,
                    arxiv_papers=[],
                    web_results=all_web_results
                )
                
                self.logger.info(
                    f"Phase 3 complete: validation_accuracy="
                    f"{validation_response.content.get('validation_accuracy', 0):.1%}"
                )
            
            # ============================================================
            # PHASE 4: FINAL RESPONSE
            # ============================================================
            self.logger.info("Phase 4: Preparing final response...")
            
            # Get citation stats from synthesis agent
            citation_stats = synthesis_response.content.get("citation_stats", {})
            
            # Build final content
            content = {
                "query": query,
                "synthesis": synthesis_response.content["synthesis"],
                "query_decomposed": decomposition_used,
                "sub_queries_count": len(queries_to_process) if decomposition_used else 0,
                "corpus_papers_used": len(all_corpus_results),
                "web_knowledge_used": bool(all_web_results),
                "validation_accuracy": validation_response.content.get("validation_accuracy", None) if validation_response else None,
                # Get citation stats from synthesis agent (not validation agent)
                "total_citations": citation_stats.get("total_citations", 0),
                "corpus_citations": citation_stats.get("corpus_citations", 0),
                "llm_citations": citation_stats.get("llm_citations", 0),
                "arxiv_citations": citation_stats.get("arxiv_citations", 0),
                "sources_used": synthesis_response.content.get("sources_used", {})
            }
            
            # Build metadata
            metadata = {
                "source": "multi_agent_rag",
                "agents_used": ["retrieval", "synthesis"],
                "query_decomposed": decomposition_used,
                "queries_processed": len(queries_to_process),
                "corpus_papers": len(all_corpus_results),
                "web_research_enabled": self.use_web_research,
                "query_decomposition_enabled": self.use_query_decomposition,
                "citation_validation_enabled": self.validate_citations,
                # Add citation statistics to metadata
                "total_citations": citation_stats.get("total_citations", 0),
                "corpus_citations": citation_stats.get("corpus_citations", 0),
                "llm_citations": citation_stats.get("llm_citations", 0),
                "arxiv_citations": citation_stats.get("arxiv_citations", 0)
            }
            
            if decomposition_used:
                metadata["agents_used"].insert(0, "query_decomposition")
            
            if all_web_results:
                metadata["agents_used"].append("web_research")
                total_refs = sum(
                    wr.metadata.get("references_found", 0) 
                    for wr in all_web_results
                )
                metadata["web_references_found"] = total_refs
            
            if validation_response:
                metadata["agents_used"].append("citation_validation")
                metadata["validation_accuracy"] = validation_response.content.get("validation_accuracy", 0)
            
            self.logger.info(
                f"Coordinator complete: {len(content['synthesis'])} chars, "
                f"{content['total_citations']} citations "
                f"({content['corpus_citations']} corpus, {content['llm_citations']} LLM)"
            )
            
            return self._create_response(
                content=content,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error in coordinator: {e}")
            return self._create_response(
                content={},
                metadata={"error": str(e)},
                success=False,
                error_message=str(e)
            )
    
    async def _run_query_decomposition(self, query: str) -> AgentResponse:
        """Run query decomposition agent."""
        self.logger.info("Running Query Decomposition Agent...")
        
        agent = QueryDecompositionAgent()
        return await agent.process(query)
    
    async def _run_retrieval_agent(self, query: str) -> AgentResponse:
        """Run retrieval agent on corpus."""
        agent = RetrievalAgent(
            retriever=self.retriever,
            config={
                "top_k": self.top_k,
                "threshold": self.threshold
            }
        )
        
        return await agent.process(query)
    
    async def _run_web_research_agent(self, query: str) -> AgentResponse:
        """Run web research agent."""
        agent = WebResearchAgent(
            config={
                "search_arxiv": False  # Disable ArXiv for speed
            }
        )
        
        return await agent.process(query)
    
    async def _run_synthesis_agent(
        self,
        query: str,
        corpus_results: Dict,
        web_results: Optional[Dict]
    ) -> AgentResponse:
        """Run synthesis agent to combine sources."""
        self.logger.info("Running Synthesis Agent...")
        
        agent = SynthesisAgent()
        
        return await agent.process(
            query=query,
            corpus_results=corpus_results,
            web_results=web_results
        )
    
    async def _run_citation_validation(
        self,
        synthesis_text: str,
        corpus_papers: list,
        arxiv_papers: list,
        web_results: Optional[List[AgentResponse]] = None
    ) -> AgentResponse:
        """Run citation validation agent."""
        self.logger.info("Running Citation Validation Agent...")
        
        agent = CitationValidationAgent()

        # Combine web results for validation
        combined_web = None
        if web_results:
            # Get the first web result's content
            if len(web_results) > 0:
                combined_web = web_results[0].to_dict()
        
        return await agent.process(
            synthesis_text=synthesis_text,
            corpus_papers=corpus_papers,
            arxiv_papers=arxiv_papers,
            web_results=combined_web
        )