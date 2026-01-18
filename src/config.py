"""Configuration module using Pydantic Settings for environment-based config."""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Config(BaseSettings):
    """Application configuration loaded from environment variables.
    
    Environment variables can be set in .env file or system environment.
    """
    
    # Pinecone
    PINECONE_API_KEY: str = Field(
        default="",
        description="Pinecone API key for vector database"
    )
    PINECONE_INDEX_NAME: str = Field(
        default="video-games",
        description="Pinecone index name"
    )
    
    # Google Gemini / Vertex AI
    GOOGLE_API_KEY: str = Field(
        default="",
        description="Google API key for Gemini"
    )
    GOOGLE_PROJECT_ID: Optional[str] = Field(
        default=None,
        description="GCP project ID for Vertex AI (optional)"
    )
    GOOGLE_LOCATION: str = Field(
        default="us-central1",
        description="GCP location for Vertex AI"
    )
    USE_VERTEX_AI: bool = Field(
        default=False,
        description="Use Vertex AI instead of Gemini API"
    )
    
    # Embedding settings
    EMBEDDING_MODEL: str = Field(
        default="all-mpnet-base-v2",
        description="HuggingFace embedding model"
    )
    EMBEDDING_DIMENSION: int = Field(
        default=768,
        description="Embedding dimension"
    )
    USE_GPU: bool = Field(
        default=False,
        description="Use GPU for embeddings (if available)"
    )
    
    # Chunking settings
    SUMMARY_CHUNK_SIZE: int = Field(
        default=200,
        description="Summary chunk size in tokens"
    )
    DETAIL_CHUNK_SIZE: int = Field(
        default=512,
        description="Detail chunk size in tokens"
    )
    CHUNK_OVERLAP: int = Field(
        default=100,
        description="Chunk overlap in tokens"
    )
    
    # Reranker settings
    RERANKER_MODEL: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking"
    )
    RERANKER_TOP_K: int = Field(
        default=5,
        description="Final top-k after reranking"
    )
    INITIAL_RETRIEVAL_K: int = Field(
        default=30,
        description="Initial retrieval before reranking"
    )
    
    # Data settings
    GAMES_PER_PLATFORM: int = Field(
        default=100,
        description="Number of games to fetch per platform"
    )
    DATA_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data",
        description="Data directory path"
    )
    
    # API settings
    API_HOST: str = Field(
        default="0.0.0.0",
        description="API host"
    )
    API_PORT: int = Field(
        default=8000,
        description="API port"
    )
    API_WORKERS: int = Field(
        default=1,
        description="Number of API workers"
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    # Rate limiting
    RATE_LIMIT_CALLS: int = Field(
        default=100,
        description="Rate limit: calls per period"
    )
    RATE_LIMIT_PERIOD: int = Field(
        default=60,
        description="Rate limit: period in seconds"
    )
    
    # Tracing (Phoenix/OpenTelemetry)
    ENABLE_TRACING: bool = Field(
        default=True,
        description="Enable OpenTelemetry tracing"
    )
    PHOENIX_ENDPOINT: Optional[str] = Field(
        default=None,
        description="Phoenix collector endpoint (e.g., https://phoenix-xxx.run.app)"
    )
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = Field(
        default=None,
        description="OpenTelemetry OTLP endpoint"
    )
    
    # GCP Secret Manager
    USE_SECRET_MANAGER: bool = Field(
        default=False,
        description="Load secrets from GCP Secret Manager"
    )
    SECRET_MANAGER_PROJECT_ID: Optional[str] = Field(
        default=None,
        description="GCP project ID for Secret Manager"
    )
    
    # Model configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    def validate_required(self) -> list[str]:
        """Validate required configuration. Returns list of missing keys.
        
        Returns:
            List of missing required configuration keys
        """
        missing = []
        
        if not self.PINECONE_API_KEY:
            missing.append("PINECONE_API_KEY")
        
        if not self.GOOGLE_API_KEY and not self.USE_VERTEX_AI:
            missing.append("GOOGLE_API_KEY (or enable USE_VERTEX_AI)")
        
        if self.USE_VERTEX_AI and not self.GOOGLE_PROJECT_ID:
            missing.append("GOOGLE_PROJECT_ID (required for Vertex AI)")
        
        return missing
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.LOG_LEVEL.upper() in ["WARNING", "ERROR"]


# Global config instance
config = Config()


# Backward compatibility for existing code
def validate() -> list[str]:
    """Validate configuration (backward compatible function)."""
    return config.validate_required()
