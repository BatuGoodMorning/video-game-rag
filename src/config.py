"""Configuration module for loading environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


class Config:
    """Application configuration loaded from environment variables."""

    # Pinecone
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "video-games")

    # Google Gemini
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # Embedding settings
    EMBEDDING_MODEL: str = "all-mpnet-base-v2"  # HuggingFace model
    EMBEDDING_DIMENSION: int = 768

    # Chunking settings
    SUMMARY_CHUNK_SIZE: int = 200  # tokens
    DETAIL_CHUNK_SIZE: int = 512  # tokens
    CHUNK_OVERLAP: int = 100  # tokens

    # Reranker settings
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_TOP_K: int = 5  # Final top-k after reranking
    INITIAL_RETRIEVAL_K: int = 30  # Initial retrieval before reranking

    # Data settings
    GAMES_PER_PLATFORM: int = 100
    DATA_DIR: Path = Path(__file__).parent.parent / "data"

    @classmethod
    def validate(cls) -> list[str]:
        """Validate required configuration. Returns list of missing keys."""
        missing = []
        if not cls.PINECONE_API_KEY:
            missing.append("PINECONE_API_KEY")
        if not cls.GOOGLE_API_KEY:
            missing.append("GOOGLE_API_KEY")
        return missing


config = Config()
