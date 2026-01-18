"""Middleware for FastAPI application."""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response with timing."""
        start_time = time.perf_counter()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Log response
            logger.info(
                f"Response: {request.method} {request.url.path} "
                f"status={response.status_code} latency={latency_ms:.2f}ms"
            )
            
            # Add latency header
            response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
            
            return response
        
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"Error: {request.method} {request.url.path} "
                f"error={str(e)} latency={latency_ms:.2f}ms",
                exc_info=True
            )
            raise


def setup_cors(app):
    """Setup CORS middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:8000",
            "http://localhost:8501",  # Streamlit
            # Add your production domains here
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware.
    
    Note: This is a basic implementation. For production, consider using
    redis-based rate limiting or a service like Cloud Armor.
    """
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        """Initialize rate limiter.
        
        Args:
            app: FastAPI app
            calls: Number of calls allowed per period
            period: Time period in seconds
        """
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.requests = {}  # {ip: [(timestamp, count)]}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check rate limit and process request."""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/ready"]:
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [
                (ts, count) for ts, count in self.requests[client_ip]
                if current_time - ts < self.period
            ]
        else:
            self.requests[client_ip] = []
        
        # Count requests in current period
        total_requests = sum(count for _, count in self.requests[client_ip])
        
        if total_requests >= self.calls:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return Response(
                content='{"error": "Rate limit exceeded. Try again later."}',
                status_code=429,
                media_type="application/json"
            )
        
        # Add current request
        self.requests[client_ip].append((current_time, 1))
        
        return await call_next(request)

