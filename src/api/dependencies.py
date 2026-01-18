"""FastAPI dependencies for RAG components.

Implements singleton pattern for expensive resources like
embedders, vector stores, and LLM-based components.
"""

from typing import Optional
from functools import lru_cache
import logging

from fastapi import HTTPException, status

from src.config import config
from src.embeddings.embed import EmbeddingGenerator
from src.vectorstores.pinecone_store import PineconeStore
from src.rag.retriever import GameRetriever
from src.rag.chain import RAGChain
from src.rag.agent import GameRAGAgent
from src.rag.reranker import Reranker

logger = logging.getLogger(__name__)


class RAGComponents:
    """Container for RAG components (singleton)."""
    
    _instance: Optional["RAGComponents"] = None
    _initialized: bool = False
    
    embedder: Optional[EmbeddingGenerator] = None
    pinecone_store: Optional[PineconeStore] = None
    reranker: Optional[Reranker] = None
    retriever: Optional[GameRetriever] = None
    chain: Optional[RAGChain] = None
    agent: Optional[GameRAGAgent] = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls) -> "RAGComponents":
        """Initialize all RAG components.
        
        This is called once during app startup.
        """
        if cls._initialized:
            logger.info("RAG components already initialized")
            return cls._instance
        
        instance = cls()
        
        try:
            logger.info("Initializing RAG components...")
            
            # Validate config
            missing = config.validate()
            if missing:
                raise ValueError(f"Missing configuration: {', '.join(missing)}")
            
            # Initialize embedding generator
            logger.info("Loading embedding model...")
            instance.embedder = EmbeddingGenerator(
                model_key="mpnet",
                use_gpu=False,  # Cloud Run typically uses CPU
                google_api_key=config.GOOGLE_API_KEY,
            )
            logger.info(f"Embedding model loaded: {instance.embedder.dimension}D")
            
            # Initialize Pinecone
            logger.info("Connecting to Pinecone...")
            instance.pinecone_store = PineconeStore(
                api_key=config.PINECONE_API_KEY,
                index_name=config.PINECONE_INDEX_NAME,
                dimension=instance.embedder.dimension,
            )
            stats = instance.pinecone_store.get_stats()
            logger.info(f"Pinecone connected: {stats.get('total_vectors', 0)} vectors")
            
            # Initialize reranker
            logger.info("Loading reranker model...")
            instance.reranker = Reranker(model_name=config.RERANKER_MODEL)
            logger.info("Reranker loaded")
            
            # Initialize retriever
            instance.retriever = GameRetriever(
                embedding_generator=instance.embedder,
                pinecone_store=instance.pinecone_store,
                reranker=instance.reranker,
            )
            logger.info("Retriever initialized")
            
            # Initialize RAG chain
            instance.chain = RAGChain(
                retriever=instance.retriever,
                google_api_key=config.GOOGLE_API_KEY,
            )
            logger.info("RAG chain initialized")
            
            # Initialize LangGraph agent
            instance.agent = GameRAGAgent(
                retriever=instance.retriever,
                google_api_key=config.GOOGLE_API_KEY,
            )
            logger.info("LangGraph agent initialized")
            
            cls._initialized = True
            logger.info("✓ All RAG components initialized successfully")
            
            return instance
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}")
            raise RuntimeError(f"Component initialization failed: {e}") from e
    
    @classmethod
    def get(cls) -> "RAGComponents":
        """Get the singleton instance.
        
        Raises HTTPException if not initialized.
        """
        if not cls._initialized or cls._instance is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG components not initialized. Service is starting up."
            )
        return cls._instance
    
    @classmethod
    def is_ready(cls) -> bool:
        """Check if components are ready."""
        if not cls._initialized or cls._instance is None:
            return False
        
        instance = cls._instance
        return all([
            instance.embedder is not None,
            instance.pinecone_store is not None,
            instance.retriever is not None,
            instance.agent is not None,
        ])
    
    @classmethod
    def health_check(cls) -> dict[str, str]:
        """Get health status of components."""
        if not cls._initialized or cls._instance is None:
            return {"status": "not_initialized"}
        
        instance = cls._instance
        checks = {}
        
        # Check embedder
        checks["embedder"] = "ok" if instance.embedder else "error"
        
        # Check Pinecone
        try:
            if instance.pinecone_store:
                instance.pinecone_store.get_stats()
                checks["pinecone"] = "ok"
            else:
                checks["pinecone"] = "error"
        except Exception as e:
            checks["pinecone"] = f"error: {str(e)[:50]}"
        
        # Check reranker
        checks["reranker"] = "ok" if instance.reranker else "error"
        
        # Check retriever
        checks["retriever"] = "ok" if instance.retriever else "error"
        
        # Check agent
        checks["agent"] = "ok" if instance.agent else "error"
        
        return checks


# FastAPI dependency functions
def get_rag_components() -> RAGComponents:
    """FastAPI dependency to get RAG components."""
    return RAGComponents.get()


def get_retriever() -> GameRetriever:
    """FastAPI dependency to get retriever."""
    components = RAGComponents.get()
    if components.retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retriever not available"
        )
    return components.retriever


def get_agent() -> GameRAGAgent:
    """FastAPI dependency to get agent."""
    components = RAGComponents.get()
    if components.agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not available"
        )
    return components.agent


def get_chain() -> RAGChain:
    """FastAPI dependency to get RAG chain."""
    components = RAGComponents.get()
    if components.chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG chain not available"
        )
    return components.chain

