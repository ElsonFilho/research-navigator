"""
Base class for all LLM providers
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class BaseLLM(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, model: str, temperature: float = 0.1, **kwargs):
        """
        Initialize LLM
        
        Args:
            model: Model identifier (e.g., 'gpt-5-nano')
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional provider-specific parameters
        """
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs
    
    @abstractmethod
    def chat(
        self, 
        messages: List[Dict[str, str]],
        max_completion_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Simple chat completion
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_completion_tokens: Maximum tokens in response
            **kwargs: Additional parameters for this call
            
        Returns:
            Response text as string
        """
        pass
    
    @abstractmethod
    def chat_with_response(
        self,
        messages: List[Dict[str, str]],
        max_completion_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chat completion with full response object
        
        Args:
            messages: List of message dicts
            max_completion_tokens: Maximum tokens in response
            **kwargs: Additional parameters
            
        Returns:
            Full response dict including usage info
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model}', temperature={self.temperature})"