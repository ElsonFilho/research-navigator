"""
Multi-Agent System for Research Navigator

This module contains all specialized agents for the multi-agent RAG architecture:
- Coordinator: Orchestrates agent interactions
- Retrieval Agent: Searches corpus (296 papers)
- Web Research Agent: Combines LLM knowledge + ArXiv search
- Synthesis Agent: Combines insights from all sources
- Decomposition Agent: Breaks complex queries into sub-queries
- Citation Validator: Validates citations at multiple levels
- Baseline Agent: Single LLM for comparison (GPT-5.2)

Architecture:
    User Query → Coordinator → [Retrieval + Web Research] → Synthesis → Citation Validation → Response

"""

from .base_agent import BaseAgent, AgentResponse, AgentConfig

__all__ = [
    "BaseAgent",
    "AgentResponse",
    "AgentConfig",
]
