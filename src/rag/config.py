"""
Research Navigator - RAG System Configuration
Core RAG Infrastructure

This configuration file contains all settings for:
- Embedding generation (OpenAI API)
- Text chunking parameters
- Vector database settings
- LLM generation settings
"""

import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMADB_DIR = DATA_DIR / "chromadb"
PAPERS_DIR = DATA_DIR

# Ensure directories exist
CHROMADB_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# OPENAI API SETTINGS
# ============================================================================

# Your OpenAI API key (should be set as environment variable)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM Model for generation
LLM_MODEL = "gpt-5-nano"  # For response generation

# EMBEDDING MODEL SETTINGS
# Easy to switch between models by changing this single line!
EMBEDDING_MODEL: Literal[
    "text-embedding-3-small",
    "text-embedding-3-large"
] = "text-embedding-3-small"

# Model specifications (automatically set based on EMBEDDING_MODEL)
EMBEDDING_CONFIGS = {
    "text-embedding-3-small": {
        "dimensions": 1536,
        "max_tokens": 8191,
        "cost_per_1m_tokens": 0.02,  # USD
        "description": "Efficient, good quality, recommended for development"
    },
    "text-embedding-3-large": {
        "dimensions": 3072,
        "max_tokens": 8191,
        "cost_per_1m_tokens": 0.13,  # USD
        "description": "Higher accuracy, larger dimensions, production-ready"
    }
}

# Get current model config
CURRENT_EMBEDDING_CONFIG = EMBEDDING_CONFIGS[EMBEDDING_MODEL]
EMBEDDING_DIMENSIONS = CURRENT_EMBEDDING_CONFIG["dimensions"]
EMBEDDING_MAX_TOKENS = CURRENT_EMBEDDING_CONFIG["max_tokens"]


# ============================================================================
# TEXT CHUNKING PARAMETERS
# ============================================================================

# Chunk size in tokens (not characters!)
# 800 tokens ≈ 3200 characters ≈ 2-3 paragraphs
CHUNK_SIZE = 800  # tokens

# Overlap between chunks to maintain context
# 100 tokens ≈ 400 characters ≈ 2-3 sentences
CHUNK_OVERLAP = 100  # tokens

# Minimum chunk size (discard smaller chunks)
MIN_CHUNK_SIZE = 100  # tokens

# Separators for text splitting (in order of preference)
CHUNK_SEPARATORS = [
    "\n\n",      # Paragraph breaks (highest priority)
    "\n",        # Line breaks
    ". ",        # Sentence endings
    "! ",        # Exclamation sentences
    "? ",        # Question sentences
    "; ",        # Semi-colons
    ", ",        # Commas
    " ",         # Spaces (last resort)
]


# ============================================================================
# CHROMADB SETTINGS
# ============================================================================

# Collection name in ChromaDB
COLLECTION_NAME = "research_papers"

# Distance metric for similarity search
# Options: "cosine", "l2", "ip" (inner product)
DISTANCE_METRIC = "cosine"

# Metadata fields to store with each chunk
METADATA_FIELDS = [
    "paper_id",           # Unique identifier
    "title",              # Paper title
    "authors",            # Author list
    "abstract",           # Paper abstract
    "publication_date",   # When published
    "arxiv_id",          # arXiv identifier
    "institution",        # Primary institution
    "chunk_index",        # Position in paper
    "total_chunks",       # Total chunks for this paper
]


# ============================================================================
# RETRIEVAL SETTINGS
# ============================================================================

# Number of chunks to retrieve for each query
TOP_K = 5  # Start with 5, can adjust based on testing

# Minimum similarity score threshold (0.0 to 1.0)
# Chunks below this threshold will be filtered out
MIN_SIMILARITY_SCORE = 0.5

# Maximum number of papers to include in context
# (even if more chunks are retrieved)
MAX_PAPERS_IN_CONTEXT = 3


# ============================================================================
# LLM GENERATION SETTINGS
# ============================================================================

# Temperature for generation (0.0 = deterministic, 1.0 = creative)
GENERATION_TEMPERATURE = 0.3  # Lower for factual responses

# Maximum tokens in generated response
MAX_RESPONSE_TOKENS = 1000

# System prompt template for RAG
SYSTEM_PROMPT = """You are an AI research assistant specialized in academic literature analysis.

Your task is to answer questions based ONLY on the provided research paper context.

Guidelines:
1. Be precise and cite specific papers when making claims
2. If information isn't in the context, say "Based on the provided papers, I don't have information about..."
3. Use academic language appropriate for researchers
4. When citing, use format: [Author et al., Year]
5. Synthesize information across multiple papers when relevant

Context from research papers:
{context}

User question: {query}
"""


# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

# Test queries file location
TEST_QUERIES_PATH = PROJECT_ROOT / "tests" / "evaluation_queries.json"

# Batch size for embedding generation (to avoid rate limits)
EMBEDDING_BATCH_SIZE = 100  # Papers per batch

# Rate limiting: wait time between API calls (seconds)
API_RATE_LIMIT_DELAY = 0.1  # 100ms between calls


# ============================================================================
# LOGGING AND DEBUGGING
# ============================================================================

# Verbosity level
DEBUG_MODE = True

# Log chunk statistics during processing
LOG_CHUNK_STATS = True

# Display cost estimates before processing
SHOW_COST_ESTIMATES = True


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_embedding_config(model_name: str = None) -> dict:
    """Get configuration for specified embedding model"""
    if model_name is None:
        model_name = EMBEDDING_MODEL
    return EMBEDDING_CONFIGS.get(model_name, EMBEDDING_CONFIGS["text-embedding-3-small"])


def estimate_cost(num_tokens: int, model_name: str = None) -> float:
    """Estimate cost in USD for embedding generation"""
    config = get_embedding_config(model_name)
    cost_per_token = config["cost_per_1m_tokens"] / 1_000_000
    return num_tokens * cost_per_token


def get_latest_papers_file() -> Path:
    """Find the most recent papers_with_fulltext_*.json file"""
    json_files = list(PAPERS_DIR.glob("papers_with_fulltext_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No papers_with_fulltext_*.json found in {PAPERS_DIR}")
    # Sort by modification time, get most recent
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    return latest_file


def print_config_summary():
    """Print current configuration summary"""
    print("\n" + "="*70)
    print("RESEARCH NAVIGATOR - RAG CONFIGURATION")
    print("="*70)
    print(f"\n📊 LLM Model: {LLM_MODEL}")
    print(f"🔢 Embedding Model: {EMBEDDING_MODEL}")
    print(f"   - Dimensions: {EMBEDDING_DIMENSIONS}")
    print(f"   - Cost: ${CURRENT_EMBEDDING_CONFIG['cost_per_1m_tokens']}/1M tokens")
    print(f"   - Description: {CURRENT_EMBEDDING_CONFIG['description']}")
    print(f"\n📝 Chunking:")
    print(f"   - Chunk Size: {CHUNK_SIZE} tokens (~{CHUNK_SIZE * 4} characters)")
    print(f"   - Overlap: {CHUNK_OVERLAP} tokens")
    print(f"   - Min Size: {MIN_CHUNK_SIZE} tokens")
    print(f"\n🗄️  Vector Database:")
    print(f"   - Location: {CHROMADB_DIR}")
    print(f"   - Collection: {COLLECTION_NAME}")
    print(f"   - Distance: {DISTANCE_METRIC}")
    print(f"\n🔍 Retrieval:")
    print(f"   - Top-K: {TOP_K}")
    print(f"   - Min Similarity: {MIN_SIMILARITY_SCORE}")
    print(f"   - Max Papers: {MAX_PAPERS_IN_CONTEXT}")
    print("="*70 + "\n")


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check API key
    if not OPENAI_API_KEY:
        errors.append("⚠️  OPENAI_API_KEY not set in environment variables")
    
    # Check paths
    if not DATA_DIR.exists():
        errors.append(f"⚠️  Data directory not found: {DATA_DIR}")
    
    # Check chunking parameters
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append(f"⚠️  Chunk overlap ({CHUNK_OVERLAP}) must be less than chunk size ({CHUNK_SIZE})")
    
    if errors:
        print("\n❌ Configuration Errors:")
        for error in errors:
            print(f"   {error}")
        return False
    
    print("✅ Configuration validated successfully!")
    return True

# ============================================================================
# RAG CONFIG CLASS (for new Week 3 components)
# ============================================================================

class RAGConfig:
    """Configuration class for RAG system components."""
    
    def __init__(self):
        # API Keys
        self.openai_api_key = OPENAI_API_KEY
        
        # Model settings
        self.embedding_model = EMBEDDING_MODEL
        self.generation_model = LLM_MODEL
        
        # Chunking settings
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
        self.min_chunk_size = MIN_CHUNK_SIZE
        self.chunk_separators = CHUNK_SEPARATORS
        
        # Vector database settings
        self.vector_db_path = CHROMADB_DIR
        self.collection_name = COLLECTION_NAME
        self.distance_metric = DISTANCE_METRIC
        
        # Retrieval settings
        self.top_k = TOP_K
        self.min_similarity_score = MIN_SIMILARITY_SCORE
        self.max_papers_in_context = MAX_PAPERS_IN_CONTEXT
        
        # Generation settings
        self.generation_temperature = GENERATION_TEMPERATURE
        self.max_response_tokens = MAX_RESPONSE_TOKENS
        self.system_prompt = SYSTEM_PROMPT

if __name__ == "__main__":
    """Test configuration when run directly"""
    print_config_summary()
    validate_config()