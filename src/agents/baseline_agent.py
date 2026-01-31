"""
Baseline Agent - Single GPT-5.2 without RAG
This serves as the comparison baseline for the multi-agent system.

Characteristics:
- Uses GPT-5.2 only
- No corpus retrieval
- No ArXiv search
- No citation validation
- Pure parametric knowledge from LLM training data
"""

from typing import Dict, Optional
from openai import OpenAI

from src.agents.base_agent import BaseAgent, AgentResponse
from src.rag.config import RAGConfig


class BaselineAgent(BaseAgent):
    """
    Single LLM baseline for comparison.
    
    This agent represents the "no RAG" baseline to compare against
    the multi-agent RAG system.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Baseline Agent.
        
        Args:
            config: Configuration dictionary (optional)
        """
        super().__init__(agent_name="baseline_agent", config=config)
        
        # Load OpenAI config
        self.rag_config = RAGConfig()
        self.client = OpenAI(api_key=self.rag_config.openai_api_key)
        
        # Baseline configuration
        self.model = self.config.get("model", "gpt-5.2")
        self.temperature = self.config.get("temperature", 0.5)
        
        self.logger.info(
            f"Baseline Agent initialized with model={self.model}, "
            f"temperature={self.temperature}"
        )
    
    async def process(self, query: str) -> AgentResponse:
        """
        Process query using only GPT-5.2 parametric knowledge.
        
        Args:
            query: Research question
            
        Returns:
            AgentResponse with LLM response only
        """
        self.logger.info(f"Processing query: {query}...")
        
        try:
            # Single LLM call - no retrieval, no tools
            self.logger.info("Generating response with GPT-5.2 (no RAG)...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            )
            
            answer = response.choices[0].message.content
            
            self.logger.info(
                f"Response generated: {len(answer)} chars, "
                f"{response.usage.total_tokens} tokens"
            )
            
            # Prepare response
            content = {
                "answer": answer,
                "tokens_used": response.usage.total_tokens
            }
            
            metadata = {
                "source": "baseline",
                "model": self.model,
                "temperature": self.temperature,
                "has_retrieval": False,
                "has_citations": False,
                "citation_level": "level_0",  # No validation
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
            
            return self._create_response(
                content=content,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error in baseline agent: {e}")
            return self._create_response(
                content={},
                metadata={"error": str(e)},
                success=False,
                error_message=str(e)
            )
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for the baseline agent.
        
        Returns:
            System prompt string
        """
        return (
            "You are an academic researcher writing a literature review. "
            "Synthesize dominant research directions relevant to the query, compare approaches, and highlight trade-offs. "
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
            "MANDATORY: End your response with a 'References' section:\n"
            "- List ALL papers cited in the synthesis\n"
            "- Use standard academic format: Author, A. B., & Author, C. D. (Year). Title. Venue.\n"
            "- If venue is not provided or empty, omit it from the reference\n"
            "- Example:\n"
            "  References:\n"
            "  \n"
            "  Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. ICML 2020.\n"
            "  \n"
            "  Kingma, D. P., & Welling, M. (2014). Auto-encoding variational bayes. ICLR 2014.\n"
        )