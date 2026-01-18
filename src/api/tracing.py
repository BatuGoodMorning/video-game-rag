"""OpenTelemetry and Phoenix tracing setup."""

import logging
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from src.config import config

logger = logging.getLogger(__name__)


def setup_tracing(service_name: str = "video-game-rag-api") -> Optional[TracerProvider]:
    """Setup OpenTelemetry tracing with Phoenix and/or Cloud Trace.
    
    Args:
        service_name: Name of the service for tracing
        
    Returns:
        TracerProvider if tracing is enabled, None otherwise
    """
    if not config.ENABLE_TRACING:
        logger.info("Tracing is disabled")
        return None
    
    try:
        logger.info("Setting up OpenTelemetry tracing...")
        
        # Create resource
        resource = Resource(attributes={
            SERVICE_NAME: service_name,
            "service.version": "1.0.0",
            "deployment.environment": "production" if config.is_production else "development"
        })
        
        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        
        # Setup Phoenix exporter if endpoint is configured
        if config.PHOENIX_ENDPOINT:
            logger.info(f"Configuring Phoenix exporter: {config.PHOENIX_ENDPOINT}")
            phoenix_exporter = OTLPSpanExporter(
                endpoint=f"{config.PHOENIX_ENDPOINT}/v1/traces",
                headers={}
            )
            tracer_provider.add_span_processor(
                BatchSpanProcessor(phoenix_exporter)
            )
            logger.info("✓ Phoenix exporter configured")
        
        # Setup OTLP exporter (for Cloud Trace or other backends)
        if config.OTEL_EXPORTER_OTLP_ENDPOINT:
            logger.info(f"Configuring OTLP exporter: {config.OTEL_EXPORTER_OTLP_ENDPOINT}")
            otlp_exporter = OTLPSpanExporter(
                endpoint=config.OTEL_EXPORTER_OTLP_ENDPOINT,
                headers={}
            )
            tracer_provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
            logger.info("✓ OTLP exporter configured")
        
        # Set as global tracer provider
        trace.set_tracer_provider(tracer_provider)
        
        logger.info("✓ OpenTelemetry tracing setup complete")
        return tracer_provider
        
    except Exception as e:
        logger.error(f"Failed to setup tracing: {e}")
        return None


def instrument_langchain():
    """Instrument LangChain with OpenInference.
    
    This automatically traces all LangChain/LangGraph operations.
    """
    if not config.ENABLE_TRACING:
        return
    
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        
        logger.info("Instrumenting LangChain...")
        LangChainInstrumentor().instrument()
        logger.info("✓ LangChain instrumentation complete")
        
    except ImportError:
        logger.warning(
            "openinference-instrumentation-langchain not installed. "
            "LangChain tracing will not be available."
        )
    except Exception as e:
        logger.error(f"Failed to instrument LangChain: {e}")


def instrument_fastapi(app):
    """Instrument FastAPI with OpenTelemetry.
    
    Args:
        app: FastAPI application instance
    """
    if not config.ENABLE_TRACING:
        return
    
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        
        logger.info("Instrumenting FastAPI...")
        FastAPIInstrumentor.instrument_app(app)
        logger.info("✓ FastAPI instrumentation complete")
        
    except ImportError:
        logger.warning(
            "opentelemetry-instrumentation-fastapi not installed. "
            "FastAPI tracing will not be available."
        )
    except Exception as e:
        logger.error(f"Failed to instrument FastAPI: {e}")


def get_tracer(name: str = __name__):
    """Get a tracer instance.
    
    Args:
        name: Tracer name (usually __name__)
        
    Returns:
        Tracer instance
    """
    return trace.get_tracer(name)

