"""
Retriever Module for Research Navigator
Handles querying the vector database and retrieving relevant chunks.
"""

import chromadb
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
from ..rag.config import RAGConfig


@dataclass
class RetrievalResult:
    """A single retrieval result with content and metadata."""
    chunk_id: str
    content: str
    paper_title: str
    authors: str
    year: str
    arxiv_id: str
    distance: float  # Lower is better (more similar)
    chunk_index: int
    
    def __str__(self):
        return (
            f"Paper: {self.paper_title}\n"
            f"Authors: {self.authors} ({self.year})\n"
            f"arXiv: {self.arxiv_id}\n"
            f"Relevance: {1 - self.distance:.3f}\n"
            f"Content:\n{self.content[:200]}...\n"
        )


class Retriever:
    """Retrieves relevant chunks from the vector database."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize retriever with ChromaDB and OpenAI client."""
        self.config = config or RAGConfig()
        
        # DEBUG: Print the path being used
        print(f"🔍 DEBUG: ChromaDB path = {self.config.vector_db_path}")
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.config.vector_db_path)
        )
        
        print(f"🔍 DEBUG: Collections found = {[c.name for c in self.chroma_client.list_collections()]}")
        
        self.collection = self.chroma_client.get_collection(
            name=self.config.collection_name
        )
        
        print(f"🔍 DEBUG: Collection '{self.config.collection_name}' has {self.collection.count()} chunks")
        
        # Initialize OpenAI client for query embeddings
        self.openai_client = OpenAI(api_key=self.config.openai_api_key)


    def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query string."""
        response = self.openai_client.embeddings.create(
            model=self.config.embedding_model,
            input=query
        )
        return response.data[0].embedding
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query string
            n_results: Number of results to return (default: 5)
            filter_dict: Optional metadata filters (e.g., {"year": "2024"})
            
        Returns:
            List of RetrievalResult objects, sorted by relevance
        """
        # Embed the query
        query_embedding = self._embed_query(query)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict,
            include=["documents", "metadatas", "distances"]
        )
        
        # Parse results
        retrieval_results = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            
            # Format authors properly
            authors_raw = metadata.get('authors', 'Unknown')
            if isinstance(authors_raw, str) and authors_raw != 'Unknown':
                try:
                    # The format is: "{'name': 'X', ...}, {'name': 'Y', ...}"
                    # Wrap in brackets to make it a valid list, then parse
                    authors_str = f"[{authors_raw}]"
                    import ast
                    authors_data = ast.literal_eval(authors_str)
                    
                    if isinstance(authors_data, list):
                        # Extract names from list of dicts
                        author_names = [
                            a.get('name', '') 
                            for a in authors_data 
                            if isinstance(a, dict) and a.get('name')
                        ]
                        authors = ', '.join(author_names) if author_names else 'Unknown'
                    else:
                        authors = str(authors_data)
                except (ValueError, SyntaxError) as e:
                    # If parsing fails, try to extract names with regex as fallback
                    import re
                    names = re.findall(r"'name':\s*'([^']+)'", authors_raw)
                    authors = ', '.join(names) if names else authors_raw
            else:
                authors = authors_raw
            
            # Extract year from publication_date
            year = metadata.get('year', 'Unknown')
            if year == 'Unknown' and 'publication_date' in metadata:
                pub_date = metadata.get('publication_date', '')
                if isinstance(pub_date, str) and len(pub_date) >= 4:
                    year = pub_date[:4]
            
            # Get distance and convert to similarity score
            distance = results['distances'][0][i]
            # For cosine distance (0 = identical, 2 = opposite), convert to similarity (0-1)
            # Cosine similarity = 1 - (cosine_distance / 2)
            # This gives: 0 distance → 1.0 similarity, 2 distance → 0.0 similarity
            similarity_score = 1.0 - (distance / 2.0)
            
            result = RetrievalResult(
                chunk_id=results['ids'][0][i],
                content=results['documents'][0][i],
                paper_title=metadata.get('title', 'Unknown'),
                authors=authors,
                year=year,
                arxiv_id=metadata.get('arxiv_id', 'Unknown'),
                distance=similarity_score,  # Now stores similarity (higher = better)
                chunk_index=metadata.get('chunk_index', -1)
            )
            retrieval_results.append(result)
        
        return retrieval_results
       
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        count = self.collection.count()
        
        # Get a sample to check metadata
        sample = self.collection.peek(limit=1)
        
        return {
            "total_chunks": count,
            "collection_name": self.config.collection_name,
            "embedding_model": self.config.embedding_model,
            "vector_db_path": str(self.config.vector_db_path)
        }


if __name__ == "__main__":
    # Quick test
    retriever = Retriever()
    print("Retriever Stats:")
    print(retriever.get_stats())
    
    print("\n" + "="*80)
    print("Test Query: 'multi-agent systems'")
    print("="*80 + "\n")
    
    results = retriever.search("multi-agent systems", n_results=3)
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(result)
        print()