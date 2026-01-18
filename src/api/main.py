"""FastAPI main application for Video Game RAG."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from src.api import __version__
from src.api.dependencies import RAGComponents
from src.api.middleware import LoggingMiddleware, setup_cors, RateLimitMiddleware
from src.api.routes import health, query, admin
from src.api.schemas import ErrorResponse
from src.api.tracing import setup_tracing, instrument_langchain, instrument_fastapi
from src.config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events.
    
    This replaces the old @app.on_event("startup") pattern.
    """
    # Startup
    logger.info("=" * 80)
    logger.info(f"Starting Video Game RAG API v{__version__}")
    logger.info("=" * 80)
    
    try:
        # Load secrets from GCP Secret Manager (if enabled)
        from src.api.secrets import load_secrets_to_config
        load_secrets_to_config()
        
        # Initialize cache
        from src.api.cache import initialize_cache
        initialize_cache(maxsize=1000)
        
        # Setup tracing
        setup_tracing("video-game-rag-api")
        instrument_langchain()
        
        # Initialize RAG components
        RAGComponents.initialize()
        logger.info("✓ RAG components initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize RAG components: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Video Game RAG API")
    logger.info("Cleanup complete")


# Create FastAPI app
app = FastAPI(
    title="Video Game RAG API",
    description=(
        "A production-ready RAG (Retrieval-Augmented Generation) API "
        "for video game information with multi-hop reasoning, reranking, "
        "and guardrails."
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# Setup CORS
setup_cors(app)

# Setup tracing instrumentation for FastAPI
instrument_fastapi(app)

# Add middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    calls=config.RATE_LIMIT_CALLS,
    period=config.RATE_LIMIT_PERIOD,
)


# Include routers
app.include_router(health.router)
app.include_router(query.router)
app.include_router(admin.router)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed messages."""
    logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc.errors()),
            status_code=422
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred. Please try again later.",
            status_code=500
        ).model_dump()
    )


# Root endpoint
@app.get(
    "/",
    summary="API Root",
    description="Get API information and available endpoints"
)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Video Game RAG API",
        "version": __version__,
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "readiness": "/ready",
            "query": "/api/v1/query",
            "stream": "/api/v1/query/stream",
            "admin": "/api/v1/admin"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

