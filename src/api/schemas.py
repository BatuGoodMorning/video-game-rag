"""Pydantic schemas for API request/response models."""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for game query."""
    
    query: str = Field(..., min_length=1, max_length=500, description="User question about video games")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to retrieve")
    platform: Optional[Literal["PC", "PlayStation", "Switch"]] = Field(default=None, description="Filter by platform")
    genre: Optional[str] = Field(default=None, description="Filter by genre")
    use_reranker: bool = Field(default=True, description="Whether to use reranker")
    use_agent: bool = Field(default=True, description="Use LangGraph agent for complex queries")
    stream: bool = Field(default=False, description="Stream response (SSE)")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What are the best RPG games on Nintendo Switch?",
                    "top_k": 5,
                    "platform": "Switch",
                    "genre": "RPG",
                    "use_reranker": True,
                    "use_agent": True
                }
            ]
        }
    }


class ChunkMetadata(BaseModel):
    """Metadata for a retrieved chunk."""
    
    game_name: str
    platform: str
    chunk_type: str
    sales_millions: Optional[float] = None
    score: float


class QueryResponse(BaseModel):
    """Response model for game query."""
    
    answer: str = Field(..., description="Generated answer")
    sources: list[str] = Field(default_factory=list, description="Game names used as sources")
    query_type: Optional[str] = Field(default=None, description="Query type (simple/complex)")
    retrieval_latency_ms: Optional[float] = Field(default=None, description="Retrieval latency in milliseconds")
    total_latency_ms: Optional[float] = Field(default=None, description="Total query latency in milliseconds")
    reranked: bool = Field(default=False, description="Whether reranker was used")
    guardrail_status: Optional[str] = Field(default=None, description="Guardrail status (passed/warning/blocked)")
    guardrail_message: Optional[str] = Field(default=None, description="Guardrail message")
    chunks_retrieved: int = Field(default=0, description="Number of chunks retrieved")
    error: Optional[str] = Field(default=None, description="Error message if any")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: Literal["healthy", "unhealthy"]
    version: str
    components: dict[str, str] = Field(default_factory=dict)


class ReadinessResponse(BaseModel):
    """Readiness check response."""
    
    ready: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str
    detail: Optional[str] = None
    status_code: int

