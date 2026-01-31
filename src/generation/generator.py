"""
Generator Module for Research Navigator
Generates responses using OpenAI GPT models and retrieved context.
"""

from typing import List, Optional
from openai import OpenAI
from src.rag.config import RAGConfig
from src.retrieval.retriever import RetrievalResult


class Generator:
    """Generates responses using retrieved context and LLM."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize generator with OpenAI client."""
        self.config = config or RAGConfig()
        self.client = OpenAI(api_key=self.config.openai_api_key)
        
    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[RetrievalResult],
        max_tokens: int = 10000,
        temperature: float = 0.3,
        max_retries: int = 2
    ) -> str:
        """
        Generate a response using retrieved context.
        
        Args:
            query: User's question
            retrieved_chunks: List of relevant chunks from retriever
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-1.0)
            max_retries: Number of retries if response is empty
            
        Returns:
            Generated response string
        """
        print("\n" + "="*60)
        print("🔍 GENERATOR DEBUG - Starting generation")
        print("="*60)
        print(f"Query: {query[:100]}...")
        print(f"Number of chunks: {len(retrieved_chunks)}")
        print(f"max_tokens: {max_tokens}")
        print(f"temperature: {temperature} (NOTE: ignored by gpt-5-nano)")
        print(f"max_retries: {max_retries}")
        
        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"[Source {i}] {chunk.paper_title} ({chunk.authors}, {chunk.year})\n"
                f"{chunk.content}\n"
            )
        
        context = "\n---\n".join(context_parts)
        print(f"Context length: {len(context)} characters")
        
        # Create the prompt
        system_prompt = """You are Research Navigator, an AI assistant specialized in academic literature analysis.

Your role is to help researchers by:
1. Synthesizing information from multiple academic papers
2. Providing accurate, well-cited responses
3. Maintaining academic rigor and precision

Guidelines:
- Always cite sources using [Source X] notation
- If information conflicts across papers, acknowledge this
- If you cannot answer based on the provided context, say so
- Be concise but comprehensive
- Use academic language appropriate for researchers"""

        user_prompt = f"""Based on the following research papers, please answer this question:

Question: {query}

Context from relevant papers:
{context}

Please provide a well-structured answer with proper citations."""

        print(f"System prompt length: {len(system_prompt)} characters")
        print(f"User prompt length: {len(user_prompt)} characters")
        print(f"Total prompt length: {len(system_prompt) + len(user_prompt)} characters")

        # Retry logic for reasoning models that may return empty content
        for attempt in range(max_retries + 1):
            print(f"\n--- Attempt {attempt + 1}/{max_retries + 1} ---")
            
            try:
                # Generate response
                print("Calling OpenAI API...")
                response = self.client.chat.completions.create(
                    model=self.config.generation_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=max_tokens
                )
                
                # Debug response details
                print(f"✅ API call successful")
                print(f"Model used: {response.model}")
                print(f"Finish reason: {response.choices[0].finish_reason}")
                print(f"Total tokens: {response.usage.total_tokens}")
                print(f"Prompt tokens: {response.usage.prompt_tokens}")
                print(f"Completion tokens: {response.usage.completion_tokens}")
                
                # Check for reasoning tokens (gpt-5-nano specific)
                if hasattr(response.usage, 'completion_tokens_details'):
                    details = response.usage.completion_tokens_details
                    if hasattr(details, 'reasoning_tokens'):
                        print(f"Reasoning tokens: {details.reasoning_tokens}")
                
                content = response.choices[0].message.content
                print(f"Content received: {len(content) if content else 0} characters")
                
                if content:
                    print(f"Content preview: {content[:200]}...")
                else:
                    print("⚠️ Content is None or empty!")
                
                # If we got content, return it
                if content and len(content.strip()) > 0:
                    print("✅ SUCCESS - Returning response")
                    print("="*60 + "\n")
                    return content
                
                # If empty and we have retries left, try again
                if attempt < max_retries:
                    print(f"⚠️ Empty response on attempt {attempt + 1}, retrying...")
                    continue
                    
            except Exception as e:
                print(f"❌ ERROR on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries:
                    print("Retrying...")
                    continue
                else:
                    print("All retries exhausted")
                    raise
        
        # If all retries failed, return error message
        print("❌ FAILED - All retries returned empty responses")
        print("="*60 + "\n")
        return "Unable to generate response after multiple attempts."
    
    def format_sources(self, retrieved_chunks: List[RetrievalResult]) -> str:
        """
        Format retrieved chunks as a reference list.
        
        Args:
            retrieved_chunks: List of retrieval results
            
        Returns:
            Formatted reference string
        """
        if not retrieved_chunks:
            return "No sources retrieved."
        
        sources = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            sources.append(
                f"[{i}] {chunk.paper_title}\n"
                f"    Authors: {chunk.authors}\n"
                f"    Year: {chunk.year}\n"
                f"    arXiv: {chunk.arxiv_id}\n"
                f"    Relevance: {1 - chunk.distance:.3f}"
            )
        
        return "\n\n".join(sources)


if __name__ == "__main__":
    # Quick test
    from src.retrieval.retriever import Retriever
    
    # Retrieve some chunks
    retriever = Retriever()
    query = "What are the challenges in multi-agent systems?"
    chunks = retriever.search(query, n_results=3)
    
    # Generate response
    generator = Generator()
    response = generator.generate_response(query, chunks)
    
    print("Query:", query)
    print("\n" + "="*80)
    print("Response:")
    print("="*80)
    print(response)
    print("\n" + "="*80)
    print("Sources:")
    print("="*80)
    print(generator.format_sources(chunks))