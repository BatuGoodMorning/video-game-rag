"""Caching layer for embeddings and responses."""

import hashlib
import json
import logging
from typing import Optional, Any
from functools import lru_cache

logger = logging.getLogger(__name__)


class InMemoryCache:
    """Simple in-memory LRU cache for embeddings and responses."""
    
    def __init__(self, maxsize: int = 1000):
        """Initialize cache.
        
        Args:
            maxsize: Maximum number of items to cache
        """
        self.maxsize = maxsize
        self._cache: dict[str, Any] = {}
        self._access_order: list[str] = []
        logger.info(f"Initialized in-memory cache (maxsize={maxsize})")
    
    def _make_key(self, prefix: str, data: Any) -> str:
        """Create cache key from data.
        
        Args:
            prefix: Key prefix (e.g., "emb", "resp")
            data: Data to hash
            
        Returns:
            Cache key
        """
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)
        
        hash_obj = hashlib.md5(content.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if key in self._cache:
            # Update access order (LRU)
            self._access_order.remove(key)
            self._access_order.append(key)
            logger.debug(f"Cache hit: {key[:20]}...")
            return self._cache[key]
        
        logger.debug(f"Cache miss: {key[:20]}...")
        return None
    
    def set(self, key: str, value: Any):
        """Set item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Evict oldest if at capacity
        if len(self._cache) >= self.maxsize and key not in self._cache:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
            logger.debug(f"Cache evicted: {oldest_key[:20]}...")
        
        self._cache[key] = value
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        logger.debug(f"Cache set: {key[:20]}...")
    
    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()
        logger.info("Cache cleared")
    
    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "utilization": len(self._cache) / self.maxsize if self.maxsize > 0 else 0
        }


class EmbeddingCache:
    """Cache for query embeddings."""
    
    def __init__(self, cache: InMemoryCache):
        """Initialize embedding cache.
        
        Args:
            cache: Underlying cache implementation
        """
        self.cache = cache
    
    def get_embedding(self, query: str) -> Optional[list[float]]:
        """Get cached embedding for query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector or None
        """
        key = self.cache._make_key("emb", query)
        return self.cache.get(key)
    
    def set_embedding(self, query: str, embedding: list[float]):
        """Cache embedding for query.
        
        Args:
            query: Query text
            embedding: Embedding vector
        """
        key = self.cache._make_key("emb", query)
        self.cache.set(key, embedding)


class ResponseCache:
    """Cache for query responses."""
    
    def __init__(self, cache: InMemoryCache):
        """Initialize response cache.
        
        Args:
            cache: Underlying cache implementation
        """
        self.cache = cache
    
    def get_response(self, query: str, filters: Optional[dict] = None) -> Optional[dict]:
        """Get cached response for query.
        
        Args:
            query: Query text
            filters: Query filters (platform, genre, etc.)
            
        Returns:
            Cached response or None
        """
        cache_data = {"query": query, "filters": filters or {}}
        key = self.cache._make_key("resp", cache_data)
        return self.cache.get(key)
    
    def set_response(self, query: str, response: dict, filters: Optional[dict] = None):
        """Cache response for query.
        
        Args:
            query: Query text
            response: Response data
            filters: Query filters
        """
        cache_data = {"query": query, "filters": filters or {}}
        key = self.cache._make_key("resp", cache_data)
        self.cache.set(key, response)


# Global cache instances
_global_cache: Optional[InMemoryCache] = None
_embedding_cache: Optional[EmbeddingCache] = None
_response_cache: Optional[ResponseCache] = None


def initialize_cache(maxsize: int = 1000):
    """Initialize global cache instances.
    
    Args:
        maxsize: Maximum cache size
    """
    global _global_cache, _embedding_cache, _response_cache
    
    _global_cache = InMemoryCache(maxsize=maxsize)
    _embedding_cache = EmbeddingCache(_global_cache)
    _response_cache = ResponseCache(_global_cache)
    
    logger.info("✓ Cache initialized")


def get_embedding_cache() -> Optional[EmbeddingCache]:
    """Get global embedding cache."""
    return _embedding_cache


def get_response_cache() -> Optional[ResponseCache]:
    """Get global response cache."""
    return _response_cache


def get_cache_stats() -> dict:
    """Get cache statistics."""
    if _global_cache:
        return _global_cache.stats()
    return {"size": 0, "maxsize": 0, "utilization": 0}


def clear_cache():
    """Clear all caches."""
    if _global_cache:
        _global_cache.clear()

