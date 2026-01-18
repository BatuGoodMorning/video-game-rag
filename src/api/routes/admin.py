"""Admin endpoints for system management (optional)."""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional

from src.api.dependencies import get_rag_components, RAGComponents

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.get(
    "/stats",
    summary="Get System Stats",
    description="Get statistics about the RAG system"
)
async def get_stats(components: RAGComponents = Depends(get_rag_components)):
    """Get system statistics."""
    try:
        stats = {}
        
        # Pinecone stats
        if components.pinecone_store:
            pinecone_stats = components.pinecone_store.get_stats()
            stats["pinecone"] = {
                "total_vectors": pinecone_stats.get("total_vectors", 0),
                "dimension": pinecone_stats.get("dimension", 0)
            }
        
        # Embedder stats
        if components.embedder:
            stats["embedder"] = {
                "model": components.embedder.model_name,
                "dimension": components.embedder.dimension
            }
        
        # Reranker stats
        if components.reranker:
            stats["reranker"] = {
                "model": components.reranker.model_name
            }
        
        return stats
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


@router.get(
    "/cache/stats",
    summary="Get Cache Stats",
    description="Get cache statistics"
)
async def get_cache_stats_endpoint():
    """Get cache statistics."""
    from src.api.cache import get_cache_stats
    
    stats = get_cache_stats()
    return {
        "cache": stats,
        "status": "enabled" if stats["maxsize"] > 0 else "disabled"
    }


@router.post(
    "/cache/clear",
    summary="Clear Cache",
    description="Clear embedding and response caches"
)
async def clear_cache_endpoint():
    """Clear application caches."""
    from src.api.cache import clear_cache
    
    try:
        clear_cache()
        return {
            "status": "success",
            "message": "Cache cleared successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )

