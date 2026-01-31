"""
OpenAI LLM wrapper
"""
import os
from typing import List, Dict, Optional, Any
from openai import OpenAI


class OpenAILLM:
    """Wrapper for OpenAI models (GPT-4, GPT-5, etc.)"""
    
    def __init__(
        self,
        model: str = "gpt-5-nano",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize OpenAI LLM"""
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found.")
        
        self.client = OpenAI(api_key=self.api_key)
        
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_completion_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Simple chat completion - returns just the text"""
        response = self.chat_with_response(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            **kwargs
        )
        return response["content"]
    
    def chat_with_response(
        self,
        messages: List[Dict[str, str]],
        max_completion_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Chat completion with full response metadata"""
        params = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            **self.kwargs,
            **kwargs
        }
        
        if "nano" not in self.model.lower():
            params["temperature"] = self.temperature
        
        response = self.client.chat.completions.create(**params)
        
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens
        self.call_count += 1
        
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "finish_reason": response.choices[0].finish_reason
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get cumulative usage statistics"""
        return {
            "total_calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens
        }
    
    def reset_usage_stats(self):
        """Reset usage tracking counters"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
    
    def __repr__(self) -> str:
        return f"OpenAILLM(model='{self.model}', temperature={self.temperature})"