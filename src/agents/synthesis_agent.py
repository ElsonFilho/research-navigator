"""
Synthesis Agent - Combines multiple sources into coherent response

This agent:
1. Takes results from Retrieval Agent (corpus)
2. Takes results from Web Research Agent (LLM + ArXiv)
3. Synthesizes them into a unified, well-cited response
4. Highlights agreements/contradictions between sources
5. Provides proper attribution for each source
"""

from typing import Dict, List, Optional
from openai import OpenAI

from src.agents.base_agent import BaseAgent, AgentResponse
from src.rag.config import RAGConfig


class SynthesisAgent(BaseAgent):
    """Agent that synthesizes information from multiple sources."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Synthesis Agent.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(agent_name="synthesis_agent", config=config)
        
        # Load OpenAI config
        self.rag_config = RAGConfig()
        self.client = OpenAI(api_key=self.rag_config.openai_api_key)
        
        # Synthesis configuration - use GPT-5.2
        self.model = self.config.get("model", "gpt-5.2")
        self.temperature = self.config.get("temperature", 0.1)
        self.max_completion_tokens = self.config.get("max_completion_tokens", 5000)
        
        self.logger.info(
            f"Synthesis Agent initialized with model={self.model}, "
            f"temperature={self.temperature}, max_completion_tokens={self.max_completion_tokens}"
        )
    
    async def process(
        self, 
        query: str,
        corpus_results: Optional[Dict] = None,
        web_results: Optional[Dict] = None
    ) -> AgentResponse:
        """
        Synthesize information from multiple sources.
        
        Args:
            query: Original research question
            corpus_results: Results from Retrieval Agent
            web_results: Results from Web Research Agent
            
        Returns:
            AgentResponse with synthesized content
        """
        self.logger.info(f"Synthesizing results for query: {query}...")
        
        try:
            # Build synthesis prompt
            synthesis_prompt = self._build_synthesis_prompt(
                query=query,
                corpus_results=corpus_results,
                web_results=web_results
            )
            
            self.logger.info(f"Generating synthesis with {self.model}...")
            
            # Generate synthesis with GPT-5.2
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": synthesis_prompt
                    }
                ]
            )
            
            synthesis = response.choices[0].message.content
            
            self.logger.info(
                f"Synthesis complete: {len(synthesis)} chars, "
                f"{response.usage.total_tokens} tokens"
            )
            
            # Parse citations from the synthesis
            citation_stats = self._parse_citations(synthesis)
            
            # Prepare response
            content = {
                "synthesis": synthesis,
                "sources_used": self._count_sources(corpus_results, web_results),
                "citation_stats": citation_stats,
                "tokens_used": response.usage.total_tokens
            }
            
            metadata = {
                "source": "synthesis",
                "model": self.model,
                "corpus_papers_used": len(corpus_results) if corpus_results else 0,
                "web_knowledge_used": bool(web_results),
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_citations": citation_stats["total_citations"],
                "corpus_citations": citation_stats["corpus_citations"],
                "llm_citations": citation_stats["llm_citations"],
                "arxiv_citations": citation_stats["arxiv_citations"]
            }
            
            return self._create_response(
                content=content,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error in synthesis: {e}")
            return self._create_response(
                content={},
                metadata={"error": str(e)},
                success=False,
                error_message=str(e)
            )
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for synthesis."""
        return (
            "You are a research synthesis expert writing an academic literature review. "
            "Your task is to combine information from multiple sources into a coherent, "
            "well-structured response with proper academic citations.\n\n"
            "Citation Guidelines:\n"
            "1. ALWAYS use proper academic citation format: 'Author et al. (Year)'\n"
            "2. For corpus papers, extract author surname and year from metadata provided\n"
            "3. Cite naturally inline: 'Smith et al. (2023) demonstrated that...'\n"
            "4. Or parenthetically: 'Recent work shows improvements (Jones et al., 2024)'\n"
            "5. For multiple authors, use 'Author1 et al.' if more than 2 authors\n"
            "6. NEVER use placeholder tags like [Corpus Paper 1] or [LLM Knowledge]\n"
            "7. Highlight where sources agree or disagree\n"
            "8. Maintain academic rigor and precision\n\n"
            "Quality Requirements:\n"
            "- Write in natural academic prose\n"
            "- Target ~800 words for the main synthesis\n"
            "- Be comprehensive but concise\n"
            "- Integrate citations smoothly into sentences\n"
            "- Prioritize corpus papers (you have full text access)\n"
            "- Aim for 8-10 citations maximum\n\n"
            "MANDATORY: End your response with a 'References' section:\n"
            "- List ALL papers cited in the synthesis\n"
            "- Use standard academic format: Author, A. B., & Author, C. D. (Year). Title. Venue.\n"
            "- If venue is not provided or empty, omit it from the reference\n"
            "- Include [CORPUS] or [LLM] tag after each reference to indicate source\n"
            "- Example:\n"
            "  References:\n"
            "  \n"
            "  Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. ICML 2020. [CORPUS]\n"
            "  \n"
            "  Kingma, D. P., & Welling, M. (2014). Auto-encoding variational bayes. ICLR 2014. [LLM]\n"
        )
    
    def _build_synthesis_prompt(
        self,
        query: str,
        corpus_results: Optional[Dict],
        web_results: Optional[Dict]
    ) -> str:
        """
        Build the synthesis prompt from multiple sources.
        
        Args:
            query: Research question
            corpus_results: Retrieval agent results
            web_results: Web research agent results
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [f"Query: {query}\n"]
        
        # Add corpus results
        if corpus_results and corpus_results.get("content"):
            prompt_parts.append("\n=== CORPUS PAPERS (Full-Text Database) ===\n")
            for i, paper in enumerate(corpus_results["content"], 1):
                metadata = paper.get("metadata", {})
                prompt_parts.append(
                    f"[Corpus Paper {i}] [SOURCE: CORPUS]\n"
                    f"Title: {metadata.get('title', 'N/A')}\n"
                    f"Authors: {metadata.get('authors', 'N/A')}\n"
                    f"Year: {metadata.get('year', 'N/A')}\n"
                    f"Venue: {metadata.get('venue', '')}\n"
                    f"Institution: {metadata.get('institution', 'N/A')}\n"
                    f"Relevance: {paper.get('relevance_score', 0):.3f}\n"
                    f"Content: {paper.get('content', '')[:500]}...\n\n"
                )
        
        # Add web research results
        if web_results and web_results.get("content"):
            # LLM synthesis
            llm_synthesis = web_results["content"].get("llm_synthesis", "")
            if llm_synthesis:
                prompt_parts.append("\n=== LLM PARAMETRIC KNOWLEDGE [SOURCE: LLM] ===\n")
                prompt_parts.append(f"{llm_synthesis}\n\n")
                prompt_parts.append("NOTE: Papers mentioned above are from LLM training data and should be marked [LLM] in References.\n\n")
            
            # ArXiv papers
            arxiv_papers = web_results["content"].get("arxiv_papers", [])
            if arxiv_papers:
                prompt_parts.append("\n=== ARXIV PAPERS ===\n")
                for i, paper in enumerate(arxiv_papers, 1):
                    prompt_parts.append(
                        f"[ArXiv Paper {i}]\n"
                        f"Title: {paper.get('title', 'N/A')}\n"
                        f"Authors: {', '.join(paper.get('authors', [])[:3])}...\n"
                        f"Published: {paper.get('published', 'N/A')}\n"
                        f"ArXiv ID: {paper.get('arxiv_id', 'N/A')}\n"
                        f"Summary: {paper.get('summary', '')[:300]}...\n\n"
                    )
        
        prompt_parts.append(
            "\nTask: Write a fluid, natural academic literature review response "
            "(TARGET: 600 words) with proper academic citations.\n\n"
            "Requirements:\n"
            "- Use proper citation format: 'Author et al. (Year)' - extract from metadata above\n"
            "- Write in natural prose WITHOUT meta-commentary\n"
            "- DO NOT use placeholder tags like [Corpus Paper 1], [LLM Knowledge], or [ArXiv]\n"
            "- DO NOT use section headers like 'Main findings', 'Relation to', 'Key agreements'\n"
            "- Seamlessly integrate all sources into a cohesive narrative\n"
            "- Cite naturally: 'Smith et al. (2023) showed...' or '(Jones et al., 2024)'\n"
            "- Write as if for an academic journal's literature review section\n"
            "- Be comprehensive but concise - prioritize depth over breadth\n"
            "- Prioritize corpus papers (you have full text) - cite these more frequently\n"
            "- Aim for 8-10 total citations maximum\n"
            "- End with key insights and implications\n\n"
            "Citation Examples:\n"
            "- 'Recent work by Chen et al. (2024) demonstrates efficient architectures...'\n"
            "- 'Model compression techniques show promise (Zhang et al., 2023; Wang et al., 2024)'\n"
            "- 'Qu et al. (2024) achieved state-of-the-art results through...'\n\n"
            "Style: Academic, authoritative, fluid prose with natural inline citations."
        )
        
        return "".join(prompt_parts)
    
    def _count_sources(
        self,
        corpus_results: Optional[Dict],
        web_results: Optional[Dict]
    ) -> Dict[str, int]:
        """
        Count sources used in synthesis.
        
        Returns:
            Dictionary with source counts
        """
        counts = {
            "corpus_papers": 0,
            "arxiv_papers": 0,
            "llm_knowledge": 0
        }
        
        if corpus_results and corpus_results.get("content"):
            counts["corpus_papers"] = len(corpus_results["content"])
        
        if web_results and web_results.get("content"):
            if web_results["content"].get("llm_synthesis"):
                counts["llm_knowledge"] = 1
            counts["arxiv_papers"] = len(
                web_results["content"].get("arxiv_papers", [])
            )
        
        return counts
    
    def _parse_citations(self, synthesis: str) -> Dict[str, any]:
        """
        Parse the References section to count citations by source.
        
        Args:
            synthesis: The complete synthesis text with References section
            
        Returns:
            Dictionary with citation statistics
        """
        citation_stats = {
            "total_citations": 0,
            "corpus_citations": 0,
            "llm_citations": 0,
            "arxiv_citations": 0,
            "references_list": []
        }
        
        # Check if References section exists
        has_references = (
            "References:" in synthesis or 
            "## References" in synthesis or
            "\nReferences\n" in synthesis or
            synthesis.strip().endswith("References")
        )

        if not has_references:
            self.logger.warning("No References section found in synthesis")
            return citation_stats

        # Extract References section
        if "References:" in synthesis:
            refs_section = synthesis.split("References:")[-1]
        elif "## References" in synthesis:
            refs_section = synthesis.split("## References")[-1]
        elif "\nReferences\n" in synthesis:
            refs_section = synthesis.split("\nReferences\n")[-1]
        else:
            # Try splitting on just "References" at end
            refs_section = synthesis.split("References")[-1]
        
        # Count citations by source tags
        lines = refs_section.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
                
            if "[CORPUS]" in line:
                citation_stats["corpus_citations"] += 1
                citation_stats["references_list"].append({
                    "text": line.replace("[CORPUS]", "").strip(),
                    "source": "corpus"
                })
            elif "[LLM]" in line:
                citation_stats["llm_citations"] += 1
                citation_stats["references_list"].append({
                    "text": line.replace("[LLM]", "").strip(),
                    "source": "llm"
                })
            elif "[ARXIV]" in line or "[ArXiv]" in line:
                citation_stats["arxiv_citations"] += 1
                citation_stats["references_list"].append({
                    "text": line.replace("[ARXIV]", "").replace("[ArXiv]", "").strip(),
                    "source": "arxiv"
                })
        
        citation_stats["total_citations"] = (
            citation_stats["corpus_citations"] + 
            citation_stats["llm_citations"] + 
            citation_stats["arxiv_citations"]
        )
        
        self.logger.info(
            f"Parsed citations: {citation_stats['total_citations']} total "
            f"({citation_stats['corpus_citations']} corpus, "
            f"{citation_stats['llm_citations']} LLM, "
            f"{citation_stats['arxiv_citations']} ArXiv)"
        )
        
        return citation_stats