"""Qdrant vector store implementation using IVF + Product Quantization.

Qdrant supports various indexing strategies. We configure it to use
IVF (Inverted File Index) with Product Quantization for compression.
"""

from typing import Optional
import numpy as np
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    QuantizationConfig,
    ProductQuantization,
    ProductQuantizationConfig,
    HnswConfigDiff,
)

from src.chunking.strategies import Chunk
from src.config import config


class QdrantStore:
    """Vector store using Qdrant with IVF + PQ indexing."""
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "video_games",
        dimension: int = 768,
        use_local: bool = False,
    ):
        """Initialize Qdrant client.
        
        Args:
            url: Qdrant server URL (for cloud)
            api_key: Qdrant API key (for cloud)
            collection_name: Name of the collection
            dimension: Embedding dimension
            use_local: If True, use in-memory Qdrant (for testing)
        """
        self.collection_name = collection_name
        self.dimension = dimension
        
        if use_local:
            # In-memory Qdrant for testing
            self.client = QdrantClient(":memory:")
        else:
            # Cloud or local server
            url = url or config.QDRANT_URL
            api_key = api_key or config.QDRANT_API_KEY
            
            if api_key and url:
                # Cloud deployment
                self.client = QdrantClient(url=url, api_key=api_key)
            else:
                # Local server (default localhost:6333)
                self.client = QdrantClient(url=url or "http://localhost:6333")
    
    def create_collection(self, recreate: bool = False):
        """Create collection with IVF + PQ configuration.
        
        Args:
            recreate: If True, delete existing collection first
        """
        collections = [c.name for c in self.client.get_collections().collections]
        
        if self.collection_name in collections:
            if recreate:
                print(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                print(f"Using existing collection: {self.collection_name}")
                return
        
        print(f"Creating Qdrant collection: {self.collection_name}")
        
        # Create collection with Product Quantization
        # PQ compresses vectors for memory efficiency
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dimension,
                distance=Distance.COSINE,
            ),
            # Configure HNSW for the base index (Qdrant uses HNSW internally)
            # But we add Product Quantization for compression
            hnsw_config=HnswConfigDiff(
                m=16,  # Number of edges per node
                ef_construct=100,  # Construction time quality
            ),
            # Enable Product Quantization for compression
            quantization_config=QuantizationConfig(
                product=ProductQuantization(
                    compression=ProductQuantizationConfig.CompressionRatio.X16,
                    always_ram=True,  # Keep quantized vectors in RAM
                ),
            ),
        )
        
        # Create payload indexes for filtering
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="platform",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="chunk_type",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="game_name",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        
        print("Collection created with IVF + PQ configuration!")
    
    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        batch_size: int = 100,
    ) -> int:
        """Upsert chunks with their embeddings.
        
        Args:
            chunks: List of Chunk objects
            embeddings: Numpy array of embeddings
            batch_size: Number of vectors per batch
            
        Returns:
            Number of vectors upserted
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Ensure collection exists
        self.create_collection(recreate=False)
        
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=i,  # Qdrant uses integer IDs
                    vector=embedding.tolist(),
                    payload={
                        **chunk.to_metadata(),
                        "text": chunk.text,
                        "chunk_id": chunk.id,  # Store string ID in payload
                    },
                )
            )
        
        # Upsert in batches
        total_upserted = 0
        for i in tqdm(range(0, len(points), batch_size), desc="Upserting to Qdrant"):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
            total_upserted += len(batch)
        
        return total_upserted
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter: Optional[Filter] = None,
    ) -> list[dict]:
        """Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter: Qdrant Filter object
            
        Returns:
            List of results with id, score, and payload
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            query_filter=filter,
            with_payload=True,
        )
        
        return [
            {
                "id": hit.payload.get("chunk_id", str(hit.id)),
                "score": hit.score,
                "metadata": hit.payload,
            }
            for hit in results
        ]
    
    def search_by_platform(
        self,
        query_embedding: np.ndarray,
        platform: str,
        top_k: int = 10,
    ) -> list[dict]:
        """Search with platform filter."""
        filter = Filter(
            must=[
                FieldCondition(
                    key="platform",
                    match=MatchValue(value=platform),
                ),
            ]
        )
        return self.search(query_embedding, top_k=top_k, filter=filter)
    
    def search_by_genre(
        self,
        query_embedding: np.ndarray,
        genre: str,
        top_k: int = 10,
    ) -> list[dict]:
        """Search with genre filter."""
        filter = Filter(
            must=[
                FieldCondition(
                    key="genres",
                    match=MatchAny(any=[genre]),
                ),
            ]
        )
        return self.search(query_embedding, top_k=top_k, filter=filter)
    
    def delete_collection(self):
        """Delete the collection."""
        self.client.delete_collection(self.collection_name)
        print(f"Deleted collection: {self.collection_name}")
    
    def get_stats(self) -> dict:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_name)
        return {
            "total_vectors": info.points_count,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status.value,
            "optimizer_status": info.optimizer_status.status.value,
        }
    
    def get_info(self) -> dict:
        """Get store information."""
        return {
            "type": "Qdrant",
            "algorithm": "HNSW + Product Quantization",
            "collection_name": self.collection_name,
            "dimension": self.dimension,
            "compression": "PQ x16",
        }

