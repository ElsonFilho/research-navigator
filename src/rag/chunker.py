"""
Research Navigator - Text Chunking Module
Week 2: Core RAG Infrastructure

This module handles splitting academic papers into chunks for embedding generation.

Key features:
- Token-based chunking (using tiktoken for accurate OpenAI token counts)
- Semantic boundary preservation (paragraphs, sentences)
- Configurable overlap to maintain context
- Metadata preservation for each chunk
"""

import tiktoken
from typing import List, Dict, Any
from .config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    CHUNK_SEPARATORS,
    EMBEDDING_MODEL,
    LOG_CHUNK_STATS
)


class TextChunker:
    """
    Splits text into overlapping chunks while preserving semantic boundaries.
    
    Uses tiktoken for accurate token counting that matches OpenAI's models.
    """
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target size for each chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            min_chunk_size: Minimum chunk size (discard smaller chunks)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Initialize tokenizer for accurate token counting
        # Use cl100k_base encoding (used by text-embedding-3-* models)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        if LOG_CHUNK_STATS:
            print(f"📊 Chunker initialized:")
            print(f"   - Target chunk size: {chunk_size} tokens")
            print(f"   - Overlap: {chunk_overlap} tokens")
            print(f"   - Min size: {min_chunk_size} tokens")
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))
    
    def split_text_by_separators(self, text: str, separators: List[str]) -> List[str]:
        """
        Split text using a hierarchy of separators.
        
        Tries each separator in order until it finds one that works.
        
        Args:
            text: Text to split
            separators: List of separators in priority order
            
        Returns:
            List of text segments
        """
        if not separators:
            return [text]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by current separator
        splits = text.split(separator)
        
        # If we got meaningful splits, return them (keeping the separator)
        if len(splits) > 1:
            # Reconstruct with separator (except for last split)
            result = []
            for i, split in enumerate(splits):
                if i < len(splits) - 1:
                    result.append(split + separator)
                else:
                    result.append(split)
            return [s for s in result if s.strip()]
        
        # If no split occurred, try next separator
        return self.split_text_by_separators(text, remaining_separators)
    
    def merge_splits_into_chunks(self, splits: List[str]) -> List[str]:
        """
        Merge small splits into chunks of target size with overlap.
        
        This is where the magic happens:
        - Combines splits until reaching target chunk size
        - Adds overlap by including tokens from previous chunk
        - Respects semantic boundaries when possible
        - STRICTLY enforces maximum chunk size
        
        Args:
            splits: List of text segments to merge
            
        Returns:
            List of chunks with overlap
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for split in splits:
            split_tokens = self.count_tokens(split)
            
            # If this single split exceeds chunk size, we need to hard-split it
            if split_tokens > self.chunk_size:
                # Save current chunk if it has content
                if current_chunk:
                    chunk_text = "".join(current_chunk)
                    chunks.append(chunk_text)
                    current_chunk = []
                    current_tokens = 0
                
                # Hard split this oversized text into smaller pieces
                split_chunks = self._hard_split_text(split)
                chunks.extend(split_chunks)
                continue
            
            # If adding this split exceeds chunk size
            if current_tokens + split_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = "".join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                # Calculate how many tokens to keep from previous chunk
                overlap_text = self._get_overlap_text(chunk_text)
                current_chunk = [overlap_text, split]
                current_tokens = self.count_tokens(overlap_text) + split_tokens
            else:
                # Add to current chunk
                current_chunk.append(split)
                current_tokens += split_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = "".join(current_chunk)
            # Only add if it meets minimum size
            if self.count_tokens(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks
    
    def _hard_split_text(self, text: str) -> List[str]:
        """
        Hard split text that's too large into smaller chunks.
        
        This is a fallback for when text can't be split semantically.
        Splits at token boundaries to ensure chunks don't exceed limit.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks, each within size limit
        """
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        # Split tokens into chunks
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            if len(chunk_tokens) >= self.min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """
        Extract the last N tokens from text for overlap.
        
        Args:
            text: Source text
            
        Returns:
            Text containing approximately chunk_overlap tokens
        """
        tokens = self.tokenizer.encode(text)
        
        # If text is shorter than overlap size, return entire text
        if len(tokens) <= self.chunk_overlap:
            return text
        
        # Get last chunk_overlap tokens
        overlap_tokens = tokens[-self.chunk_overlap:]
        overlap_text = self.tokenizer.decode(overlap_tokens)
        
        return overlap_text
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Main entry point for chunking.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        # Handle empty or very short text
        if not text or not text.strip():
            return []
        
        text_tokens = self.count_tokens(text)
        
        # If text fits in one chunk, return as-is
        if text_tokens <= self.chunk_size:
            return [text]
        
        # Split by semantic boundaries
        splits = self.split_text_by_separators(text, CHUNK_SEPARATORS)
        
        # Merge splits into chunks with overlap
        chunks = self.merge_splits_into_chunks(splits)
        
        return chunks
    
    def chunk_paper(self, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a paper and add metadata to each chunk.
        
        Args:
            paper: Paper dictionary with 'full_text' and metadata
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Extract text (handle both 'full_text' and 'fulltext' keys)
        text = paper.get('full_text') or paper.get('fulltext', '')
        
        if not text:
            if LOG_CHUNK_STATS:
                print(f"   ⚠️  No text found for paper: {paper.get('title', 'Unknown')[:50]}")
            return []
        
        # Generate chunks
        text_chunks = self.chunk_text(text)
        
        # Create chunk dictionaries with metadata
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = {
                'text': chunk_text,
                'chunk_index': i,
                'total_chunks': len(text_chunks),
                'tokens': self.count_tokens(chunk_text),
                'metadata': {
                    'paper_id': paper.get('paper_id', paper.get('arxiv_id', 'unknown')),
                    'title': paper.get('title', 'Unknown'),
                    'authors': paper.get('authors', []),
                    'abstract': paper.get('abstract', ''),
                    'publication_date': paper.get('publication_date', paper.get('published', '')),
                    'arxiv_id': paper.get('arxiv_id', ''),
                    'institution': paper.get('institution', ''),
                    'chunk_index': i,
                    'total_chunks': len(text_chunks)
                }
            }
            chunks.append(chunk)
        
        if LOG_CHUNK_STATS:
            total_tokens = sum(c['tokens'] for c in chunks)
            avg_tokens = total_tokens / len(chunks) if chunks else 0
            print(f"   ✓ Chunked: {paper.get('title', 'Unknown')[:50]}...")
            print(f"      {len(chunks)} chunks, {total_tokens:,} tokens total, {avg_tokens:.0f} avg")
        
        return chunks
    
    def chunk_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple papers.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            List of all chunks from all papers
        """
        all_chunks = []
        
        print(f"\n📄 Chunking {len(papers)} papers...")
        print("=" * 70)
        
        for i, paper in enumerate(papers, 1):
            if i % 10 == 0 or i == 1:
                print(f"Progress: {i}/{len(papers)} papers processed...")
            
            chunks = self.chunk_paper(paper)
            all_chunks.extend(chunks)
        
        print("=" * 70)
        print(f"✅ Chunking complete!")
        print(f"   Total papers: {len(papers)}")
        print(f"   Total chunks: {len(all_chunks)}")
        print(f"   Avg chunks per paper: {len(all_chunks) / len(papers):.1f}")
        
        # Calculate token statistics
        total_tokens = sum(c['tokens'] for c in all_chunks)
        avg_tokens = total_tokens / len(all_chunks) if all_chunks else 0
        print(f"   Total tokens: {total_tokens:,}")
        print(f"   Avg tokens per chunk: {avg_tokens:.0f}")
        print()
        
        return all_chunks


# Convenience function for single-use chunking
def chunk_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience function to chunk papers with default settings.
    
    Args:
        papers: List of paper dictionaries
        
    Returns:
        List of chunks with metadata
    """
    chunker = TextChunker()
    return chunker.chunk_papers(papers)