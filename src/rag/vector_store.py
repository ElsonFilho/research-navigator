"""
Research Navigator - Vector Store Module
Week 2: Core RAG Infrastructure

This module handles storing and retrieving embeddings using ChromaDB.

Key features:
- Persistent storage of embeddings
- Metadata filtering (institution, date, etc.)
- Similarity search with configurable parameters
- Batch operations for efficiency
- Collection management
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from pathlib import Path

from .config import (
    CHROMADB_DIR,
    COLLECTION_NAME,
    DISTANCE_METRIC,
    TOP_K,
    MIN_SIMILARITY_SCORE,
    EMBEDDING_DIMENSIONS,
    DEBUG_MODE
)


class VectorStore:
    """
    Manages vector storage and retrieval using ChromaDB.
    
    Provides efficient similarity search with metadata filtering.
    """
    
    def __init__(
        self,
        persist_directory: Path = CHROMADB_DIR,
        collection_name: str = COLLECTION_NAME,
        distance_metric: str = DISTANCE_METRIC
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to store ChromaDB data
            collection_name: Name of the collection
            distance_metric: Distance metric for similarity ("cosine", "l2", "ip")
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        
        # Ensure persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        if DEBUG_MODE:
            print(f"🗄️  VectorStore initialized:")
            print(f"   Database: {self.persist_directory}")
            print(f"   Collection: {collection_name}")
            print(f"   Distance metric: {distance_metric}")
            print(f"   Existing items: {self.collection.count()}")
    
    def _get_or_create_collection(self):
        """
        Get existing collection or create a new one.
        
        Returns:
            ChromaDB collection
        """
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=None  # We provide embeddings directly
            )
            if DEBUG_MODE:
                print(f"   ✓ Loaded existing collection: {self.collection_name}")
        except Exception:
            # Create new collection
            metadata = {
                "description": "Research papers vector embeddings",
                "dimensions": EMBEDDING_DIMENSIONS,
                "distance_metric": self.distance_metric
            }
            
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata=metadata,
                embedding_function=None  # We provide embeddings directly
            )
            if DEBUG_MODE:
                print(f"   ✓ Created new collection: {self.collection_name}")
        
        return collection
    
    def add_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        show_progress: bool = True,
        batch_size: int = 5000
    ) -> int:
        """
        Add chunks with embeddings to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'embedding' and 'metadata'
            show_progress: Whether to show progress updates
            batch_size: Maximum number of chunks per batch (ChromaDB limit ~5461)
            
        Returns:
            Number of chunks successfully added
        """
        if not chunks:
            print("⚠️  No chunks to add")
            return 0
        
        print(f"\n💾 Adding {len(chunks)} chunks to vector store...")
        print(f"   Processing in batches of {batch_size}...")
        print("=" * 70)
        
        total_added = 0
        
        # Process in batches
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            if show_progress:
                print(f"   Batch {batch_start//batch_size + 1}: Processing chunks {batch_start+1}-{batch_end}...")
            
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for i, chunk in enumerate(batch_chunks):
                # Validate chunk has required fields
                if 'embedding' not in chunk:
                    if show_progress:
                        print(f"   ⚠️  Skipping chunk {batch_start + i}: no embedding")
                    continue
                
                if 'text' not in chunk:
                    if show_progress:
                        print(f"   ⚠️  Skipping chunk {batch_start + i}: no text")
                    continue
                
                # Generate unique ID
                paper_id = chunk.get('metadata', {}).get('paper_id', f'unknown_{batch_start + i}')
                chunk_idx = chunk.get('chunk_index', batch_start + i)
                chunk_id = f"{paper_id}_chunk_{chunk_idx}"
                
                # Prepare metadata (ChromaDB requires all values to be strings, ints, or floats)
                metadata = self._prepare_metadata(chunk.get('metadata', {}))
                metadata['chunk_index'] = chunk.get('chunk_index', batch_start + i)
                metadata['total_chunks'] = chunk.get('total_chunks', 1)
                metadata['tokens'] = chunk.get('tokens', 0)
                
                ids.append(chunk_id)
                embeddings.append(chunk['embedding'])
                documents.append(chunk['text'])
                metadatas.append(metadata)
            
            # Add batch to ChromaDB
            if ids:
                try:
                    if DEBUG_MODE and batch_start == 0:
                        print(f"\n🔍 DEBUG: Attempting to add batch of {len(ids)} items to ChromaDB")
                        print(f"   Sample ID: {ids[0]}")
                        print(f"   Sample embedding length: {len(embeddings[0])}")
                        print(f"   Sample document length: {len(documents[0])} chars")
                    
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas
                    )
                    
                    total_added += len(ids)
                    
                    if show_progress:
                        print(f"   ✓ Added {len(ids)} chunks from this batch")
                    
                except Exception as e:
                    print(f"❌ Error adding batch {batch_start//batch_size + 1}: {e}")
                    print(f"   Error type: {type(e).__name__}")
                    if DEBUG_MODE:
                        import traceback
                        traceback.print_exc()
                    # Continue with next batch instead of failing completely
                    continue
        
        print("=" * 70)
        print(f"✅ Successfully added {total_added} chunks to vector store")
        print(f"   Total items in collection: {self.collection.count()}")
        print()
        
        return total_added
    
    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare metadata for ChromaDB (convert complex types to strings).
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Cleaned metadata with ChromaDB-compatible types
        """
        cleaned = {}
        
        for key, value in metadata.items():
            # Convert lists to comma-separated strings
            if isinstance(value, list):
                cleaned[key] = ", ".join(str(v) for v in value)
            # Convert None to empty string
            elif value is None:
                cleaned[key] = ""
            # Keep strings, ints, floats as-is
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            # Convert everything else to string
            else:
                cleaned[key] = str(value)
        
        return cleaned
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = TOP_K,
        min_similarity: float = MIN_SIMILARITY_SCORE,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: Query vector (1536 dimensions)
            top_k: Number of results to return
            min_similarity: Minimum similarity score (0-1)
            metadata_filter: Optional metadata filters (e.g., {"institution": "ETH Zurich"})
            
        Returns:
            List of matching chunks with similarity scores
        """
        # Validate query embedding
        if len(query_embedding) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Query embedding must have {EMBEDDING_DIMENSIONS} dimensions, "
                f"got {len(query_embedding)}"
            )
        
        # Build ChromaDB where clause from metadata filter
        # ChromaDB requires specific syntax for where clauses
        where_clause = None
        if metadata_filter:
            # For simple filters, ChromaDB uses direct key-value pairs
            where_clause = metadata_filter
        
        # Query ChromaDB
        try:
            # Use where_document for text filtering, where for metadata
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            chunks = []
            
            # ChromaDB returns lists of lists, we want the first query's results
            documents = results['documents'][0] if results['documents'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            for doc, metadata, distance in zip(documents, metadatas, distances):
                # Convert distance to similarity score (for cosine: 1 - distance)
                if self.distance_metric == "cosine":
                    similarity = 1 - distance
                elif self.distance_metric == "l2":
                    similarity = 1 / (1 + distance)  # Normalize L2 distance
                else:  # inner product
                    similarity = distance
                
                # Filter by minimum similarity
                if similarity >= min_similarity:
                    chunks.append({
                        'text': doc,
                        'metadata': metadata,
                        'similarity': similarity,
                        'distance': distance
                    })
            
            return chunks
            
        except Exception as e:
            print(f"❌ Error searching vector store: {e}")
            return []
    
    def search_by_text(
        self,
        query_text: str,
        embedding_generator,
        top_k: int = TOP_K,
        min_similarity: float = MIN_SIMILARITY_SCORE,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using text query (generates embedding automatically).
        
        Args:
            query_text: Text query
            embedding_generator: EmbeddingGenerator instance
            top_k: Number of results to return
            min_similarity: Minimum similarity score
            metadata_filter: Optional metadata filters
            
        Returns:
            List of matching chunks with similarity scores
        """
        # Generate embedding for query
        query_embedding = embedding_generator.generate_embedding(query_text)
        
        if query_embedding is None:
            print("❌ Failed to generate query embedding")
            return []
        
        # Search with embedding
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            min_similarity=min_similarity,
            metadata_filter=metadata_filter
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        
        # Get sample metadata to understand data structure
        sample = None
        if count > 0:
            sample_results = self.collection.get(
                limit=1,
                include=["metadatas"]
            )
            if sample_results['metadatas']:
                sample = sample_results['metadatas'][0]
        
        stats = {
            'collection_name': self.collection_name,
            'total_chunks': count,
            'distance_metric': self.distance_metric,
            'embedding_dimensions': EMBEDDING_DIMENSIONS,
            'persist_directory': str(self.persist_directory),
            'sample_metadata_keys': list(sample.keys()) if sample else []
        }
        
        return stats
    
    def clear_collection(self, confirm: bool = False):
        """
        Clear all data from the collection.
        
        Args:
            confirm: Must be True to actually clear data (safety check)
        """
        if not confirm:
            print("⚠️  Clear cancelled: confirm parameter must be True")
            return
        
        count_before = self.collection.count()
        
        # Delete collection and recreate
        self.client.delete_collection(name=self.collection_name)
        self.collection = self._get_or_create_collection()
        
        print(f"🗑️  Cleared {count_before} chunks from collection '{self.collection_name}'")
    
    def delete_by_paper_id(self, paper_id: str) -> int:
        """
        Delete all chunks for a specific paper.
        
        Args:
            paper_id: Paper ID to delete
            
        Returns:
            Number of chunks deleted
        """
        try:
            # Get all chunks for this paper
            results = self.collection.get(
                where={"paper_id": paper_id},
                include=["metadatas"]
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                count = len(results['ids'])
                print(f"🗑️  Deleted {count} chunks for paper '{paper_id}'")
                return count
            else:
                print(f"ℹ️  No chunks found for paper '{paper_id}'")
                return 0
                
        except Exception as e:
            print(f"❌ Error deleting chunks: {e}")
            return 0


def create_vector_store(
    persist_directory: Path = CHROMADB_DIR,
    collection_name: str = COLLECTION_NAME
) -> VectorStore:
    """
    Convenience function to create a vector store.
    
    Args:
        persist_directory: Directory to store ChromaDB data
        collection_name: Name of the collection
        
    Returns:
        Initialized VectorStore instance
    """
    return VectorStore(
        persist_directory=persist_directory,
        collection_name=collection_name
    )


if __name__ == "__main__":
    """
    Test the vector store with sample data.
    """
    print("\n⚠️  Running vector_store.py directly")
    print("For production use, call this from RAG Setup UI or import as module\n")
    
    # Create test vector store
    print("Creating test vector store...")
    store = VectorStore()
    
    # Show stats
    stats = store.get_stats()
    print("\n📊 Vector Store Statistics:")
    print("=" * 70)
    for key, value in stats.items():
        print(f"   {key}: {value}")
    print("=" * 70)
    
    print("\n✅ Vector store is ready to use!")
    print("\nTo use with your papers:")
    print("1. Generate embeddings: python -m src.rag.embeddings")
    print("2. Store in ChromaDB: Use vector_store.add_chunks(embedded_chunks)")
    print("3. Query: Use vector_store.search_by_text(query, embedding_generator)")