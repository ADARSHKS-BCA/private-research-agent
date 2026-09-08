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
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    # Ollama (Local LLM)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:4b")

    def ensure_directories(self) -> None:
        """Ensure all required data directories exist on the filesystem."""
        self.data_raw_dir.mkdir(parents=True, exist_ok=True)
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)


# Global settings singleton
settings = Settings()
settings.ensure_directories()
