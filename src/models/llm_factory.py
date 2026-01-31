"""
Factory for creating LLM instances
"""
from typing import Optional
from .openai_llm import OpenAILLM
from .base import BaseLLM


class LLMFactory:
    """Factory for creating LLM instances"""
    
    @staticmethod
    def create(
        provider: str = "openai",
        model: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:
        """
        Create an LLM instance
        
        Args:
            provider: Provider name ('openai')
            model: Model name (if None, uses default for provider)
            **kwargs: Additional parameters passed to LLM constructor
            
        Returns:
            LLM instance
            
        Examples:
            # Create default nano model
            llm = LLMFactory.create()
            
            # Create specific model
            llm = LLMFactory.create(model="gpt-5.2")
            
            # Create with custom settings
            llm = LLMFactory.create(model="gpt-5-mini", temperature=0.3)
        """
        if provider == "openai":
            model = model or "gpt-5-nano"  # Default to nano
            return OpenAILLM(model=model, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    @staticmethod
    def create_for_agent(agent_name: str) -> BaseLLM:
        """
        Create LLM optimized for specific agent
        
        Args:
            agent_name: Name of agent (coordinator, retrieval, synthesis, etc.)
            
        Returns:
            LLM instance with appropriate model
        """
        # Model selection based on agent needs
        agent_models = {
            "coordinator": "gpt-5-mini",      # Routing logic
            "retrieval": "gpt-5-nano",        # Query rewriting
            "synthesis": "gpt-5.2",           # Multi-paper summaries
            "citation": "gpt-5.2",            # Accuracy critical
            "trend_analyzer": "gpt-5-mini",   # Analysis
            "query_decomposition": "gpt-5-mini",  # Reasoning
            "application_matcher": "gpt-5-nano"   # Simple matching
        }
        
        model = agent_models.get(agent_name, "gpt-5-nano")
        return LLMFactory.create(model=model)