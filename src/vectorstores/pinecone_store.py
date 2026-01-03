"""Pinecone vector store implementation using HNSW index.

Pinecone uses HNSW (Hierarchical Navigable Small World) algorithm
for approximate nearest neighbor search.
"""

import time
from typing import Optional
import numpy as np
from tqdm import tqdm

from pinecone import Pinecone, ServerlessSpec

from src.chunking.strategies import Chunk
from src.config import config


class PineconeStore:
    """Vector store using Pinecone with HNSW indexing."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        dimension: int = 768,
        metric: str = "cosine",
    ):
        """Initialize Pinecone client.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the index
            dimension: Embedding dimension
            metric: Distance metric (cosine, euclidean, dotproduct)
        """
        self.api_key = api_key or config.PINECONE_API_KEY
        self.index_name = index_name or config.PINECONE_INDEX_NAME
        self.dimension = dimension
        self.metric = metric
        
        if not self.api_key:
            raise ValueError("Pinecone API key is required")
        
        # Initialize client
        self.pc = Pinecone(api_key=self.api_key)
        self._index = None
    
    @property
    def index(self):
        """Get or create the index."""
        if self._index is None:
            self._ensure_index_exists()
            self._index = self.pc.Index(self.index_name)
        return self._index
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                ),
            )
            # Wait for index to be ready
            print("Waiting for index to be ready...")
            while not self.pc.describe_index(self.index_name).status.ready:
                time.sleep(1)
            print("Index ready!")
        else:
            print(f"Using existing index: {self.index_name}")
    
    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        batch_size: int = 100,
        namespace: str = "",
    ) -> int:
        """Upsert chunks with their embeddings.
        
        Args:
            chunks: List of Chunk objects
            embeddings: Numpy array of embeddings
            batch_size: Number of vectors per batch
            namespace: Pinecone namespace
            
        Returns:
            Number of vectors upserted
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                "id": chunk.id,
                "values": embedding.tolist(),
                "metadata": {
                    **chunk.to_metadata(),
                    "text": chunk.text,  # Store text for retrieval
                },
            })
        
        # Upsert in batches
        total_upserted = 0
        for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting to Pinecone"):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
            total_upserted += len(batch)
        
        return total_upserted
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        namespace: str = "",
        filter: Optional[dict] = None,
        include_metadata: bool = True,
    ) -> list[dict]:
        """Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            namespace: Pinecone namespace
            filter: Metadata filter dict
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of results with id, score, and metadata
        """
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_metadata=include_metadata,
        )
        
        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata if include_metadata else {},
            }
            for match in results.matches
        ]
    
    def search_by_platform(
        self,
        query_embedding: np.ndarray,
        platform: str,
        top_k: int = 10,
        namespace: str = "",
    ) -> list[dict]:
        """Search with platform filter."""
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            namespace=namespace,
            filter={"platform": {"$eq": platform}},
        )
    
    def search_by_genre(
        self,
        query_embedding: np.ndarray,
        genre: str,
        top_k: int = 10,
        namespace: str = "",
    ) -> list[dict]:
        """Search with genre filter."""
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            namespace=namespace,
            filter={"genres": {"$in": [genre]}},
        )
    
    def delete_all(self, namespace: str = ""):
        """Delete all vectors in namespace."""
        self.index.delete(delete_all=True, namespace=namespace)
        print(f"Deleted all vectors in namespace: {namespace or 'default'}")
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "namespaces": dict(stats.namespaces) if stats.namespaces else {},
            "index_fullness": stats.index_fullness,
        }
    
    def get_info(self) -> dict:
        """Get store information."""
        return {
            "type": "Pinecone",
            "algorithm": "HNSW",
            "index_name": self.index_name,
            "dimension": self.dimension,
            "metric": self.metric,
        }

