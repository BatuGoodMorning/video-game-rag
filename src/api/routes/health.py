"""Health and readiness check endpoints."""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from src.api.schemas import HealthResponse, ReadinessResponse
from src.api.dependencies import RAGComponents
from src.api import __version__

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Basic health check endpoint for liveness probe"
)
async def health_check():
    """Health check endpoint (liveness probe).
    
    Returns 200 OK if the service is running.
    """
    return HealthResponse(
        status="healthy",
        version=__version__,
        components={}
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness Check",
    description="Readiness check endpoint that verifies all components are initialized"
)
async def readiness_check():
    """Readiness check endpoint (readiness probe).
    
    Returns 200 OK if all components are ready to serve requests.
    Returns 503 Service Unavailable if components are still initializing.
    """
    is_ready = RAGComponents.is_ready()
    checks = RAGComponents.health_check()
    
    if is_ready:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=ReadinessResponse(
                ready=True,
                checks={k: v == "ok" for k, v in checks.items()},
                message="All systems operational"
            ).model_dump()
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ReadinessResponse(
                ready=False,
                checks={k: v == "ok" for k, v in checks.items()},
                message="Service is starting up or components are unavailable"
            ).model_dump()
        )


@router.get(
    "/health/components",
    summary="Component Health",
    description="Detailed health status of all components"
)
async def component_health():
    """Get detailed health status of all components."""
    checks = RAGComponents.health_check()
    is_ready = RAGComponents.is_ready()
    
    return {
        "ready": is_ready,
        "components": checks,
        "version": __version__
    }

