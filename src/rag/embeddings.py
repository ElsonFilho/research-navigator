"""
Research Navigator - Embeddings Generation Module
Core RAG Infrastructure

This module handles generating embeddings for paper chunks using OpenAI API.

Key features:
- Batch processing with progress tracking
- Cost estimation before processing
- Rate limiting to avoid API errors
- Retry logic for failed requests
- Embedding validation
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI

from .config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_BATCH_SIZE,
    API_RATE_LIMIT_DELAY,
    estimate_cost,
    get_latest_papers_file,
    SHOW_COST_ESTIMATES,
    DEBUG_MODE
)
from .chunker import TextChunker


class EmbeddingGenerator:
    """
    Generates embeddings for text chunks using OpenAI API.
    
    Handles batching, rate limiting, and error recovery.
    """
    
    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        rate_limit_delay: float = API_RATE_LIMIT_DELAY
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model: OpenAI embedding model name
            batch_size: Number of texts to embed in each batch
            rate_limit_delay: Delay between API calls in seconds
        """
        self.model = model
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        
        # Initialize OpenAI client
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize chunker
        self.chunker = TextChunker()
        
        if DEBUG_MODE:
            print(f"🔧 EmbeddingGenerator initialized:")
            print(f"   Model: {model}")
            print(f"   Dimensions: {EMBEDDING_DIMENSIONS}")
            print(f"   Batch size: {batch_size}")
            print(f"   Rate limit delay: {rate_limit_delay}s")
    
    def generate_embedding(self, text: str, retry_count: int = 3) -> Optional[List[float]]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            retry_count: Number of retries on failure
            
        Returns:
            Embedding vector (list of floats) or None on failure
        """
        for attempt in range(retry_count):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text
                )
                embedding = response.data[0].embedding
                
                # Validate embedding
                if len(embedding) != EMBEDDING_DIMENSIONS:
                    raise ValueError(
                        f"Expected {EMBEDDING_DIMENSIONS} dimensions, "
                        f"got {len(embedding)}"
                    )
                
                return embedding
                
            except Exception as e:
                if attempt < retry_count - 1:
                    if DEBUG_MODE:
                        print(f"   ⚠️  API error (attempt {attempt + 1}/{retry_count}): {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"   ❌ Failed to generate embedding after {retry_count} attempts: {e}")
                    return None
    
    def generate_embeddings_batch(
        self, 
        texts: List[str], 
        show_progress: bool = True
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress updates
            
        Returns:
            List of embeddings (same order as input texts)
        """
        embeddings = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if show_progress and (i % 10 == 0 or i == total - 1):
                print(f"   Progress: {i + 1}/{total} embeddings generated...")
            
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
            
            # Rate limiting
            if i < total - 1 and self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)
        
        return embeddings
    
    def embed_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for chunks and add them to chunk dictionaries.
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            show_progress: Whether to show progress updates
            
        Returns:
            Chunks with 'embedding' field added
        """
        print(f"\n🔢 Generating embeddings for {len(chunks)} chunks...")
        print(f"   Model: {self.model}")
        print("=" * 70)
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        start_time = time.time()
        embeddings = self.generate_embeddings_batch(texts, show_progress)
        elapsed_time = time.time() - start_time
        
        # Add embeddings to chunks
        embedded_chunks = []
        successful = 0
        failed = 0
        
        for chunk, embedding in zip(chunks, embeddings):
            if embedding is not None:
                chunk['embedding'] = embedding
                embedded_chunks.append(chunk)
                successful += 1
            else:
                failed += 1
        
        print("=" * 70)
        print(f"✅ Embedding generation complete!")
        print(f"   Successful: {successful}/{len(chunks)}")
        if failed > 0:
            print(f"   Failed: {failed}/{len(chunks)}")
        print(f"   Time elapsed: {elapsed_time:.1f}s")
        print(f"   Avg time per embedding: {elapsed_time / len(chunks):.2f}s")
        print()
        
        return embedded_chunks
    
    def load_and_chunk_papers(
        self, 
        papers_file: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Load papers from JSON and chunk them.
        
        Args:
            papers_file: Path to papers JSON file (auto-detects if None)
            
        Returns:
            List of chunks ready for embedding
        """
        # Auto-detect papers file if not provided
        if papers_file is None:
            papers_file = get_latest_papers_file()
        
        print(f"\n📂 Loading papers from: {papers_file.name}")
        
        # Load papers
        with open(papers_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        
        print(f"   Loaded {len(papers)} papers")
        
        # Chunk papers
        chunks = self.chunker.chunk_papers(papers)
        
        return chunks
    
    def estimate_embedding_cost(self, chunks: List[Dict[str, Any]]) -> float:
        """
        Estimate cost of embedding generation.
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            Estimated cost in USD
        """
        total_tokens = sum(chunk['tokens'] for chunk in chunks)
        cost = estimate_cost(total_tokens, self.model)
        return cost
    
    def process_papers(
        self, 
        papers_file: Optional[Path] = None,
        output_file: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Complete pipeline: load papers → chunk → embed.
        
        Args:
            papers_file: Path to papers JSON file (auto-detects if None)
            output_file: Path to save embedded chunks (optional)
            
        Returns:
            List of chunks with embeddings
        """
        print("\n" + "=" * 70)
        print("🚀 EMBEDDING GENERATION PIPELINE")
        print("=" * 70)
        
        # Step 1: Load and chunk
        chunks = self.load_and_chunk_papers(papers_file)
        
        # Step 2: Show cost estimate
        if SHOW_COST_ESTIMATES:
            cost = self.estimate_embedding_cost(chunks)
            print(f"\n💰 Cost Estimate:")
            print(f"   Total chunks: {len(chunks)}")
            print(f"   Total tokens: {sum(chunk['tokens'] for chunk in chunks):,}")
            print(f"   Estimated cost: ${cost:.4f}")
            print()
            
            # Ask for confirmation
            response = input("   Proceed with embedding generation? [y/N]: ")
            if response.lower() != 'y':
                print("   ❌ Cancelled by user")
                return []
        
        # Step 3: Generate embeddings
        embedded_chunks = self.embed_chunks(chunks)
        
        # Step 4: Save to file if requested
        if output_file:
            print(f"💾 Saving embedded chunks to: {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(embedded_chunks, f, indent=2)
            print(f"   ✅ Saved {len(embedded_chunks)} chunks")
        
        print("=" * 70)
        print("🎉 Pipeline complete!")
        print("=" * 70)
        
        return embedded_chunks


def generate_embeddings(
    papers_file: Optional[Path] = None,
    output_file: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to generate embeddings for papers.
    
    Args:
        papers_file: Path to papers JSON file (auto-detects if None)
        output_file: Path to save embedded chunks (optional)
        
    Returns:
        List of chunks with embeddings
    """
    generator = EmbeddingGenerator()
    return generator.process_papers(papers_file, output_file)


if __name__ == "__main__":
    """
    Run embedding generation when executed directly.
    
    This is for quick testing only - proper usage should be through
    the RAG Setup UI or as an imported module.
    """
    print("\n⚠️  Running embeddings.py directly")
    print("For production use, call this from RAG Setup UI or import as module\n")
    
    # Test with auto-detected papers file
    try:
        embedded_chunks = generate_embeddings()
        print(f"\n✅ Successfully generated embeddings for {len(embedded_chunks)} chunks")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. OPENAI_API_KEY is set in .env file")
        print("2. papers_with_fulltext_*.json exists in data/ folder")
        print("3. All dependencies are installed")