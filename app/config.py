import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    # Application settings
    app_name: str = os.getenv("APP_NAME", "Private Research Agent")
    environment: str = os.getenv("ENVIRONMENT", "DEVELOPMENT")

    # Firecrawl
    firecrawl_api_key: str = os.getenv("FIRECRAWL_API_KEY", "")

    # Qdrant Vector Database
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "research_documents")

    # Embedding Model (Local)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Data directories
    data_raw_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("DATA_RAW_DIR", "data/raw")
    )
    data_processed_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("DATA_PROCESSED_DIR", "data/processed")
    )
    data_cache_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("DATA_CACHE_DIR", "data/cache")
    )

    # LLM Settings
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")  # "groq" or "ollama"

    # Groq (Cloud LLM)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Ollama (Local LLM)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    ollama_num_predict: int = int(os.getenv("OLLAMA_NUM_PREDICT", "2048"))
    ollama_temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))

    # Performance & Concurrency
    max_concurrent_scrapes: int = int(os.getenv("MAX_CONCURRENT_SCRAPES", "5"))
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))

    # Search Providers
    search_provider: str = os.getenv("SEARCH_PROVIDER", "firecrawl")  # "firecrawl" or "duckduckgo"
    fallback_search_provider: str = os.getenv("FALLBACK_SEARCH_PROVIDER", "duckduckgo")

    # Hybrid Retrieval (Dense BGE-M3 + Lexical BM25)
    dense_weight: float = float(os.getenv("DENSE_WEIGHT", "0.7"))
    sparse_weight: float = float(os.getenv("SPARSE_WEIGHT", "0.3"))

    # Reranker Settings
    reranker_enabled: bool = os.getenv("RERANKER_ENABLED", "true").lower() in ("true", "1", "yes")
    reranker_model: str = os.getenv("RERANKER_MODEL", "ms-marco-MiniLM-L-6-v2")
    reranker_top_k: int = int(os.getenv("RERANKER_TOP_K", "8"))
    reranker_candidates: int = int(os.getenv("RERANKER_CANDIDATES", "25"))

    # Relevance & Anti-Hallucination Threshold
    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.35"))
    max_research_iterations: int = int(os.getenv("MAX_RESEARCH_ITERATIONS", "3"))
    hybrid_top_k: int = int(os.getenv("HYBRID_TOP_K", "15"))

    # Uploads and Database
    data_uploads_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("DATA_UPLOADS_DIR", "data/uploads")
    )
    upload_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("DATA_UPLOADS_DIR", "data/uploads")
    )
    sqlite_db_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("SQLITE_DB_PATH", "data/conversations.db")
    )

    def ensure_directories(self) -> None:
        """Ensure all required data directories exist on the filesystem."""
        self.data_raw_dir.mkdir(parents=True, exist_ok=True)
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_uploads_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)


# Global settings singleton
settings = Settings()
settings.ensure_directories()
