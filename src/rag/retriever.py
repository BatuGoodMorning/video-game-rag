"""Game retriever module using LangChain.

Supports both Pinecone and Qdrant vector stores with
unified interface for RAG pipeline.
"""

from typing import Optional, Literal
from dataclasses import dataclass
import time

import numpy as np

from src.embeddings.embed import EmbeddingGenerator
from src.vectorstores.pinecone_store import PineconeStore
from src.vectorstores.qdrant_store import QdrantStore


@dataclass
class RetrievalResult:
    """Result from retrieval with timing info."""
    
    chunks: list[dict]
    query: str
    vector_store: str
    latency_ms: float
    top_k: int
    filters: dict
    
    def get_texts(self) -> list[str]:
        """Extract text content from chunks."""
        return [c["metadata"].get("text", "") for c in self.chunks]
    
    def get_game_names(self) -> list[str]:
        """Extract unique game names."""
        names = [c["metadata"].get("game_name", "") for c in self.chunks]
        return list(dict.fromkeys(names))  # Preserve order, remove duplicates


class GameRetriever:
    """Unified retriever for game information."""
    
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        pinecone_store: Optional[PineconeStore] = None,
        qdrant_store: Optional[QdrantStore] = None,
    ):
        """Initialize retriever with stores.
        
        Args:
            embedding_generator: Embedding generator instance
            pinecone_store: Pinecone store instance
            qdrant_store: Qdrant store instance
        """
        self.embedder = embedding_generator
        self.pinecone = pinecone_store
        self.qdrant = qdrant_store
        
        if not pinecone_store and not qdrant_store:
            raise ValueError("At least one vector store must be provided")
    
    def retrieve(
        self,
        query: str,
        store: Literal["pinecone", "qdrant"] = "pinecone",
        top_k: int = 5,
        platform: Optional[str] = None,
        genre: Optional[str] = None,
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            store: Which store to use
            top_k: Number of results
            platform: Filter by platform (PC, PS5, Switch)
            genre: Filter by genre
            
        Returns:
            RetrievalResult with chunks and metadata
        """
        # Embed query
        query_embedding = self.embedder.embed_query(query)
        
        # Select store
        if store == "pinecone" and self.pinecone:
            vector_store = self.pinecone
        elif store == "qdrant" and self.qdrant:
            vector_store = self.qdrant
        else:
            raise ValueError(f"Store '{store}' not available")
        
        # Build filters
        filters = {}
        if platform:
            filters["platform"] = platform
        if genre:
            filters["genre"] = genre
        
        # Search with timing
        start_time = time.perf_counter()
        
        if platform:
            results = vector_store.search_by_platform(
                query_embedding, platform, top_k=top_k
            )
        elif genre:
            results = vector_store.search_by_genre(
                query_embedding, genre, top_k=top_k
            )
        else:
            results = vector_store.search(query_embedding, top_k=top_k)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return RetrievalResult(
            chunks=results,
            query=query,
            vector_store=store,
            latency_ms=latency_ms,
            top_k=top_k,
            filters=filters,
        )
    
    def retrieve_from_both(
        self,
        query: str,
        top_k: int = 5,
        platform: Optional[str] = None,
        genre: Optional[str] = None,
    ) -> tuple[RetrievalResult, RetrievalResult]:
        """Retrieve from both stores for comparison.
        
        Returns:
            Tuple of (pinecone_result, qdrant_result)
        """
        if not self.pinecone or not self.qdrant:
            raise ValueError("Both stores must be available for comparison")
        
        pinecone_result = self.retrieve(
            query, store="pinecone", top_k=top_k, platform=platform, genre=genre
        )
        qdrant_result = self.retrieve(
            query, store="qdrant", top_k=top_k, platform=platform, genre=genre
        )
        
        return pinecone_result, qdrant_result
    
    def compare_results(
        self,
        pinecone_result: RetrievalResult,
        qdrant_result: RetrievalResult,
    ) -> dict:
        """Compare results from both stores.
        
        Returns:
            Comparison statistics
        """
        pinecone_games = set(pinecone_result.get_game_names())
        qdrant_games = set(qdrant_result.get_game_names())
        
        common = pinecone_games & qdrant_games
        only_pinecone = pinecone_games - qdrant_games
        only_qdrant = qdrant_games - pinecone_games
        
        return {
            "pinecone_latency_ms": pinecone_result.latency_ms,
            "qdrant_latency_ms": qdrant_result.latency_ms,
            "latency_diff_ms": pinecone_result.latency_ms - qdrant_result.latency_ms,
            "common_games": list(common),
            "only_pinecone": list(only_pinecone),
            "only_qdrant": list(only_qdrant),
            "overlap_ratio": len(common) / max(len(pinecone_games), len(qdrant_games), 1),
        }

