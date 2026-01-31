"""
Simple RAG Pipeline for Research Navigator
Combines retrieval and generation into a single-agent baseline system.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from src.rag.config import RAGConfig
from src.retrieval.retriever import Retriever, RetrievalResult
from src.generation.generator import Generator


@dataclass
class RAGResponse:
    """Complete RAG response with answer, sources, and metadata."""
    query: str
    answer: str
    sources: List[RetrievalResult]
    response_time: float
    n_chunks_retrieved: int
    
    def format_full_response(self) -> str:
        """Format the complete response for display."""
        output = []
        output.append("=" * 80)
        output.append("QUERY")
        output.append("=" * 80)
        output.append(self.query)
        output.append("\n" + "=" * 80)
        output.append("ANSWER")
        output.append("=" * 80)
        output.append(self.answer)
        output.append("\n" + "=" * 80)
        output.append("SOURCES")
        output.append("=" * 80)
        
        for i, source in enumerate(self.sources, 1):
            output.append(f"\n[{i}] {source.paper_title}")
            output.append(f"    Authors: {source.authors}")
            output.append(f"    Year: {source.year}")
            output.append(f"    arXiv: {source.arxiv_id}")
            output.append(f"    Relevance: {1 - source.distance:.3f}")
        
        output.append("\n" + "=" * 80)
        output.append(f"Retrieved {self.n_chunks_retrieved} chunks in {self.response_time:.2f} seconds")
        output.append("=" * 80)
        
        return "\n".join(output)


class SimpleRAGPipeline:
    """
    Single-agent RAG pipeline (baseline for comparison).
    
    This will serve as the baseline to compare against the multi-agent
    architecture.
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize pipeline with retriever and generator."""
        self.config = config or RAGConfig()
        self.retriever = Retriever(config=self.config)
        self.generator = Generator(config=self.config)
        
    def query(
        self,
        question: str,
        n_chunks: int = 5,
        max_tokens: int = 10000,
        temperature: float = 0.3,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: User's research question
            n_chunks: Number of chunks to retrieve
            max_tokens: Maximum tokens in generated response
            temperature: Generation temperature (0.0-1.0)
            filter_dict: Optional metadata filters
            
        Returns:
            RAGResponse with answer and sources
        """
        start_time = time.time()
        
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.retriever.search(
            query=question,
            n_results=n_chunks,
            filter_dict=filter_dict
        )
        
        # Step 2: Generate response
        answer = self.generator.generate_response(
            query=question,
            retrieved_chunks=retrieved_chunks,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        response_time = time.time() - start_time
        
        return RAGResponse(
            query=question,
            answer=answer,
            sources=retrieved_chunks,
            response_time=response_time,
            n_chunks_retrieved=len(retrieved_chunks)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            **self.retriever.get_stats(),
            "generation_model": self.config.generation_model
        }


def run_example_queries():
    """Run some example queries to test the pipeline."""
    pipeline = SimpleRAGPipeline()
    
    print("Simple RAG Pipeline - Example Queries")
    print("=" * 80)
    print(f"System Stats: {pipeline.get_stats()}")
    print("=" * 80 + "\n")
    
    example_queries = [
        "What are the main challenges in multi-agent AI systems?",
        "How does retrieval-augmented generation improve language models?",
        "What are recent advances in federated learning?"
    ]
    
    for query in example_queries:
        print(f"\nProcessing: {query}")
        print("-" * 80)
        
        response = pipeline.query(query, n_chunks=3)
        print(response.format_full_response())
        print("\n" + "=" * 80 + "\n")
        
        # Small delay to avoid rate limits
        time.sleep(1)


if __name__ == "__main__":
    run_example_queries()