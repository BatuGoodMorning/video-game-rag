"""Factory for creating LLM instances (Gemini API or Vertex AI)."""

import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from src.config import config

logger = logging.getLogger(__name__)


def create_llm(
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.7,
    google_api_key: Optional[str] = None,
) -> BaseChatModel:
    """Create LLM instance based on configuration.
    
    Args:
        model_name: Model name (e.g., "gemini-2.0-flash", "gemini-2.5-flash")
        temperature: Generation temperature
        google_api_key: Google API key (for Gemini API mode)
        
    Returns:
        LangChain chat model instance
        
    Raises:
        ValueError: If configuration is invalid
    """
    if config.USE_VERTEX_AI:
        logger.info(f"Creating Vertex AI LLM: {model_name}")
        return _create_vertex_ai_llm(model_name, temperature)
    else:
        logger.info(f"Creating Gemini API LLM: {model_name}")
        return _create_gemini_api_llm(model_name, temperature, google_api_key)


def _create_vertex_ai_llm(
    model_name: str,
    temperature: float
) -> BaseChatModel:
    """Create Vertex AI LLM instance.
    
    Args:
        model_name: Model name
        temperature: Generation temperature
        
    Returns:
        Vertex AI chat model
        
    Raises:
        ValueError: If Vertex AI configuration is missing
    """
    if not config.GOOGLE_PROJECT_ID:
        raise ValueError(
            "GOOGLE_PROJECT_ID is required for Vertex AI. "
            "Set USE_VERTEX_AI=false to use Gemini API instead."
        )
    
    try:
        from langchain_google_vertexai import ChatVertexAI
        
        logger.info(
            f"Initializing Vertex AI: project={config.GOOGLE_PROJECT_ID}, "
            f"location={config.GOOGLE_LOCATION}"
        )
        
        return ChatVertexAI(
            model_name=model_name,
            project=config.GOOGLE_PROJECT_ID,
            location=config.GOOGLE_LOCATION,
            temperature=temperature,
        )
        
    except ImportError:
        raise ImportError(
            "langchain-google-vertexai is required for Vertex AI. "
            "Install it with: pip install langchain-google-vertexai"
        )


def _create_gemini_api_llm(
    model_name: str,
    temperature: float,
    google_api_key: Optional[str]
) -> BaseChatModel:
    """Create Gemini API LLM instance.
    
    Args:
        model_name: Model name
        temperature: Generation temperature
        google_api_key: Google API key
        
    Returns:
        Gemini API chat model
        
    Raises:
        ValueError: If API key is missing
    """
    api_key = google_api_key or config.GOOGLE_API_KEY
    
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is required for Gemini API. "
            "Set USE_VERTEX_AI=true to use Vertex AI with service account instead."
        )
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )
        
    except ImportError:
        raise ImportError(
            "langchain-google-genai is required for Gemini API. "
            "Install it with: pip install langchain-google-genai"
        )

