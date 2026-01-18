"""Game retriever module for RAG pipeline.

Uses Pinecone vector store with optional reranking.
"""

from typing import Optional, Literal
from dataclasses import dataclass
import time

import numpy as np

from src.embeddings.embed import EmbeddingGenerator
from src.vectorstores.pinecone_store import PineconeStore


@dataclass
class RetrievalResult:
    """Result from retrieval with timing info."""
    
    chunks: list[dict]
    query: str
    latency_ms: float
    top_k: int
    filters: dict
    reranked: bool = False
    
    def get_texts(self) -> list[str]:
        """Extract text content from chunks."""
        return [c["metadata"].get("text", "") for c in self.chunks]
    
    def get_game_names(self) -> list[str]:
        """Extract unique game names."""
        names = [c["metadata"].get("game_name", "") for c in self.chunks]
        return list(dict.fromkeys(names))  # Preserve order, remove duplicates


class GameRetriever:
    """Retriever for game information using Pinecone."""
    
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        pinecone_store: PineconeStore,
        reranker=None,
    ):
        """Initialize retriever.
        
        Args:
            embedding_generator: Embedding generator instance
            pinecone_store: Pinecone store instance
            reranker: Optional reranker instance
        """
        self.embedder = embedding_generator
        self.pinecone = pinecone_store
        self.reranker = reranker
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        platform: Optional[str] = None,
        genre: Optional[str] = None,
        chunk_type: Optional[str] = None,
        game_name: Optional[str] = None,
        use_reranker: bool = True,
        initial_k: int = 30,
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Number of final results
            platform: Filter by platform (PC, PS5, Switch)
            genre: Filter by genre
            chunk_type: Filter by chunk type (summary, detail, similarity)
            game_name: Filter by specific game name
            use_reranker: Whether to use reranker (if available)
            initial_k: Number of initial results before reranking
            
        Returns:
            RetrievalResult with chunks and metadata
        """
        # Embed query
        query_embedding = self.embedder.embed_query(query)
        
        # Build filters
        filters = {}
        if platform:
            filters["platform"] = platform
        if genre:
            filters["genre"] = genre
        if chunk_type:
            filters["chunk_type"] = chunk_type
        if game_name:
            filters["game_name"] = game_name
        
        # Determine how many to retrieve initially
        retrieve_k = initial_k if (use_reranker and self.reranker) else top_k
        
        # Search with timing
        start_time = time.perf_counter()
        
        # Build Pinecone filter
        pinecone_filter = None
        if filters:
            pinecone_filter = {}
            if platform:
                pinecone_filter["platform"] = {"$eq": platform}
            if genre:
                pinecone_filter["genres"] = {"$in": [genre]}
            if chunk_type:
                pinecone_filter["chunk_type"] = {"$eq": chunk_type}
            if game_name:
                pinecone_filter["game_name"] = {"$eq": game_name}
        
        results = self.pinecone.search(
            query_embedding, 
            top_k=retrieve_k,
            filter=pinecone_filter
        )
        
        # Apply reranker if available
        reranked = False
        if use_reranker and self.reranker and len(results) > top_k:
            results = self.reranker.rerank(query, results, top_k=top_k)
            reranked = True
        else:
            results = results[:top_k]
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return RetrievalResult(
            chunks=results,
            query=query,
            latency_ms=latency_ms,
            top_k=top_k,
            filters=filters,
            reranked=reranked,
        )
    
    def retrieve_by_chunk_type(
        self,
        query: str,
        chunk_type: str,
        top_k: int = 5,
        game_name: Optional[str] = None,
    ) -> RetrievalResult:
        """Retrieve chunks of a specific type (summary, detail, similarity)."""
        return self.retrieve(
            query=query,
            top_k=top_k,
            chunk_type=chunk_type,
            game_name=game_name,
        )
    
    def retrieve_similarity_chunks(
        self,
        game_name: str,
        top_k: int = 3,
    ) -> RetrievalResult:
        """Retrieve similarity chunks for a specific game."""
        return self.retrieve(
            query=f"Games similar to {game_name}",
            top_k=top_k,
            chunk_type="similarity",
            game_name=game_name,
            use_reranker=False,  # Similarity chunks don't need reranking
        )
