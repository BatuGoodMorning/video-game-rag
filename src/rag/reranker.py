"""Reranker module using cross-encoder for improved retrieval.

Uses a lightweight cross-encoder model to rerank retrieved chunks,
improving precision and reducing hallucinations.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker for improving retrieval quality."""
    
    # Lightweight, fast cross-encoder model
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """Initialize the reranker.
        
        Args:
            model_name: Cross-encoder model to use (default: ms-marco-MiniLM-L-6-v2)
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name or self.MODEL_NAME
        self._model = None
        self._device = device
    
    @property
    def model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            import torch
            
            # Determine device
            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading reranker model: {self.model_name}")
            self._model = CrossEncoder(self.model_name, device=self._device)
            logger.info(f"Reranker loaded on {self._device}")
            
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            raise RuntimeError(f"Could not load reranker model: {e}")
    
    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Rerank chunks using cross-encoder scores.
        
        Args:
            query: The user query
            chunks: List of chunk dicts with 'metadata' containing 'text'
            top_k: Number of top results to return
            
        Returns:
            Reranked list of chunks with updated scores
        """
        if not chunks:
            return []
        
        if len(chunks) <= top_k:
            return chunks
        
        # Extract texts from chunks
        texts = []
        for chunk in chunks:
            text = chunk.get("metadata", {}).get("text", "")
            if not text:
                text = str(chunk.get("metadata", {}))
            texts.append(text)
        
        # Create query-document pairs
        pairs = [(query, text) for text in texts]
        
        # Get cross-encoder scores
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fall back to original order
            return chunks[:top_k]
        
        # Combine chunks with scores
        scored_chunks = list(zip(chunks, scores))
        
        # Sort by cross-encoder score (descending)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Update scores in chunks and return top_k
        result = []
        for chunk, ce_score in scored_chunks[:top_k]:
            # Create a copy with updated score
            reranked_chunk = {
                **chunk,
                "score": float(ce_score),
                "original_score": chunk.get("score", 0),
                "reranked": True,
            }
            result.append(reranked_chunk)
        
        return result
    
    def score_pair(self, query: str, text: str) -> float:
        """Score a single query-text pair.
        
        Args:
            query: The query string
            text: The text to score against the query
            
        Returns:
            Relevance score (higher is more relevant)
        """
        try:
            score = self.model.predict([(query, text)])[0]
            return float(score)
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            return 0.0
    
    def get_info(self) -> dict:
        """Get information about the reranker."""
        return {
            "model_name": self.model_name,
            "device": self._device,
            "loaded": self._model is not None,
        }

