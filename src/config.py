"""
Configuration management for Research Navigator
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Central configuration"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    SRC_DIR = PROJECT_ROOT / "src"
    DATA_DIR = PROJECT_ROOT / "data"
    
    # Data directories
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    CHROMA_DIR = DATA_DIR / "chromadb"
    CACHE_DIR = DATA_DIR / "llm_cache"
    
    # Create directories
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    
    # Models
    PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gpt-4-turbo-preview")
    FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-3.5-turbo")
    
    # RAG settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    TOP_K = 10
    
    # Evaluation targets
    RETRIEVAL_TARGET = 0.85
    QUALITY_TARGET = 0.80
    LATENCY_TARGET = 5.0
    CITATION_TARGET = 0.95
    
    @classmethod
    def validate(cls):
        """Check configuration"""
        if not cls.OPENAI_API_KEY or cls.OPENAI_API_KEY == "your-key-here":
            print("⚠️  WARNING: OPENAI_API_KEY not set in .env file")
        return True

Config.validate()