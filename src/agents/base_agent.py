"""
Base Agent Abstract Class
All agents in the multi-agent system inherit from this class
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@dataclass
class AgentResponse:
    """
    Standardized response format for all agents
    Ensures consistency across the multi-agent system
    """
    content: Any
    metadata: Dict[str, Any]
    agent_name: str
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "agent_name": self.agent_name,
            "success": self.success,
            "error_message": self.error_message
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the Research Navigator system
    
    Provides:
    - Standardized interface
    - Logging capabilities
    - Error handling structure
    - Response formatting
    """
    
    def __init__(self, agent_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base agent
        
        Args:
            agent_name: Unique identifier for the agent
            config: Optional configuration dictionary
        """
        self.agent_name = agent_name
        self.config = config or {}
        self.logger = logging.getLogger(agent_name)
        
        self.logger.info(f"Initialized {agent_name}")
    
    @abstractmethod
    async def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Process a query and return a response
        
        This is the main method that each agent must implement
        
        Args:
            query: The user query or task
            context: Optional context from previous agents
            
        Returns:
            AgentResponse: Standardized response object
        """
        pass
    
    def _create_response(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> AgentResponse:
        """
        Helper method to create standardized responses
        
        Args:
            content: The main content/result from the agent
            metadata: Additional metadata about the processing
            success: Whether the processing was successful
            error_message: Error message if processing failed
            
        Returns:
            AgentResponse: Standardized response object
        """
        return AgentResponse(
            content=content,
            metadata=metadata or {},
            agent_name=self.agent_name,
            success=success,
            error_message=error_message
        )
    
    def _log_processing(self, query: str):
        """Log the start of processing"""
        self.logger.info(f"Processing query: {query[:100]}...")
    
    def _log_success(self, message: str = "Processing completed successfully"):
        """Log successful processing"""
        self.logger.info(message)
    
    def _log_error(self, error: Exception):
        """Log errors during processing"""
        self.logger.error(f"Error in {self.agent_name}: {str(error)}", exc_info=True)


class AgentConfig:
    """
    Configuration helper for agents
    Centralizes common configuration parameters
    """
    
    # OpenAI Configuration
    OPENAI_MODEL = "gpt-5.2"  # As per architecture document
    OPENAI_TEMPERATURE = 0.3  # Recommended range: 0.2-0.4
    
    # Retrieval Configuration
    TOP_K_RESULTS = 5  # Number of papers to retrieve
    RELEVANCE_THRESHOLD = 0.5  # Minimum relevance score
    
    # ArXiv Configuration
    ARXIV_MAX_RESULTS = 5  # Papers per ArXiv search
    
    # Citation Validation Levels
    VALIDATION_LEVEL_0 = "none"  # No validation (baseline)
    VALIDATION_LEVEL_1 = "existence"  # Check paper exists
    VALIDATION_LEVEL_2 = "metadata"  # Check authors, year, title
    VALIDATION_LEVEL_3 = "content"  # Full text verification
    
    @classmethod
    def get_openai_config(cls) -> Dict[str, Any]:
        """Get OpenAI configuration"""
        return {
            "model": cls.OPENAI_MODEL,
            "temperature": cls.OPENAI_TEMPERATURE
        }
    
    @classmethod
    def get_retrieval_config(cls) -> Dict[str, Any]:
        """Get retrieval configuration"""
        return {
            "top_k": cls.TOP_K_RESULTS,
            "threshold": cls.RELEVANCE_THRESHOLD
        }
    
    @classmethod
    def get_arxiv_config(cls) -> Dict[str, Any]:
        """Get ArXiv configuration"""
        return {
            "max_results": cls.ARXIV_MAX_RESULTS
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Example concrete agent for testing
    class ExampleAgent(BaseAgent):
        async def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
            self._log_processing(query)
            
            try:
                # Simulate processing
                result = f"Processed: {query}"
                
                self._log_success()
                return self._create_response(
                    content=result,
                    metadata={"query_length": len(query)}
                )
                
            except Exception as e:
                self._log_error(e)
                return self._create_response(
                    content=None,
                    success=False,
                    error_message=str(e)
                )
    
    # Test the agent
    async def test():
        agent = ExampleAgent("test_agent")
        response = await agent.process("Test query")
        print(f"Success: {response.success}")
        print(f"Content: {response.content}")
        print(f"Metadata: {response.metadata}")
    
    asyncio.run(test())