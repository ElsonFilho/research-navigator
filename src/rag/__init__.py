"""
Research Navigator - RAG Module
Core RAG Infrastructure

This module provides retrieval-augmented generation capabilities for academic literature.
"""

from .config import (
    RAGConfig,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    LLM_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    COLLECTION_NAME,
    print_config_summary,
    validate_config
)

__version__ = "0.1.0"
__all__ = [
    "RAGConfig",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS", 
    "LLM_MODEL",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K",
    "COLLECTION_NAME",
    "print_config_summary",
    "validate_config"
]