"""Query endpoints for RAG pipeline."""

import time
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from src.api.schemas import QueryRequest, QueryResponse, ErrorResponse
from src.api.dependencies import get_agent, get_chain
from src.rag.agent import GameRAGAgent
from src.rag.chain import RAGChain

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query Game Information",
    description="Query the RAG system for video game information",
    responses={
        200: {"description": "Successful query response"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Service unavailable"}
    }
)
async def query_games(
    request: QueryRequest,
    agent: GameRAGAgent = Depends(get_agent),
    chain: RAGChain = Depends(get_chain)
):
    """Process a game query through the RAG pipeline.
    
    Args:
        request: Query request with question and filters
        agent: LangGraph agent (injected)
        chain: Simple RAG chain (injected)
    
    Returns:
        QueryResponse with answer and metadata
    """
    start_time = time.perf_counter()
    
    try:
        logger.info(f"Processing query: {request.query[:100]}...")
        
        if request.use_agent:
            # Use LangGraph agent for complex queries
            logger.info("Using LangGraph agent")
            result = agent.query(request.query)
            
            total_latency_ms = (time.perf_counter() - start_time) * 1000
            
            return QueryResponse(
                answer=result["answer"],
                sources=result.get("sources", []),
                query_type=result.get("query_type"),
                retrieval_latency_ms=None,  # Agent doesn't expose this separately
                total_latency_ms=total_latency_ms,
                reranked=True,  # Agent uses reranker by default
                guardrail_status=result.get("guardrail_status"),
                guardrail_message=result.get("guardrail_message"),
                chunks_retrieved=len(result.get("sources", [])),
                error=result.get("error")
            )
        
        else:
            # Use simple RAG chain
            logger.info("Using simple RAG chain")
            result = chain.query(
                question=request.query,
                top_k=request.top_k,
                platform=request.platform,
                genre=request.genre,
                use_reranker=request.use_reranker,
            )
            
            total_latency_ms = (time.perf_counter() - start_time) * 1000
            
            return QueryResponse(
                answer=result["answer"],
                sources=result.get("sources", []),
                query_type="simple",
                retrieval_latency_ms=result.get("retrieval_latency_ms"),
                total_latency_ms=total_latency_ms,
                reranked=result.get("reranked", False),
                guardrail_status="passed",
                guardrail_message=None,
                chunks_retrieved=len(result.get("chunks", [])),
                error=None
            )
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@router.post(
    "/query/stream",
    summary="Stream Query Response",
    description="Query with server-sent events (SSE) streaming"
)
async def stream_query(
    request: QueryRequest,
    agent: GameRAGAgent = Depends(get_agent),
):
    """Stream query response using SSE.
    
    Note: Currently returns the same response as non-streaming.
    Full streaming implementation would require async LLM streaming.
    """
    # For now, return a simple SSE stream
    # Full implementation would require async streaming from LangChain
    
    async def generate():
        """Generate SSE events."""
        try:
            result = agent.query(request.query)
            
            # Send the answer in chunks (simulated streaming)
            answer = result["answer"]
            chunk_size = 50
            
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i+chunk_size]
                yield f"data: {chunk}\n\n"
            
            # Send metadata as final event
            yield f"data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

