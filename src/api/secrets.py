"""GCP Secret Manager integration."""

import logging
from typing import Optional

from src.config import config

logger = logging.getLogger(__name__)


def load_secret(secret_id: str, project_id: Optional[str] = None, version: str = "latest") -> Optional[str]:
    """Load a secret from GCP Secret Manager.
    
    Args:
        secret_id: Secret ID (e.g., "pinecone-api-key")
        project_id: GCP project ID (defaults to config.SECRET_MANAGER_PROJECT_ID)
        version: Secret version (default: "latest")
        
    Returns:
        Secret value as string, or None if not found/error
    """
    if not config.USE_SECRET_MANAGER:
        logger.debug("Secret Manager is disabled")
        return None
    
    project = project_id or config.SECRET_MANAGER_PROJECT_ID or config.GOOGLE_PROJECT_ID
    
    if not project:
        logger.warning("No project ID configured for Secret Manager")
        return None
    
    try:
        from google.cloud import secretmanager
        
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project}/secrets/{secret_id}/versions/{version}"
        
        logger.info(f"Loading secret: {secret_id}")
        response = client.access_secret_version(request={"name": name})
        secret_value = response.payload.data.decode("UTF-8")
        
        logger.info(f"✓ Secret loaded: {secret_id}")
        return secret_value
        
    except ImportError:
        logger.error(
            "google-cloud-secret-manager not installed. "
            "Install it with: pip install google-cloud-secret-manager"
        )
        return None
    
    except Exception as e:
        logger.error(f"Failed to load secret {secret_id}: {e}")
        return None


def load_secrets_to_config():
    """Load secrets from GCP Secret Manager and update config.
    
    This should be called during app startup if USE_SECRET_MANAGER is True.
    """
    if not config.USE_SECRET_MANAGER:
        logger.info("Secret Manager integration disabled")
        return
    
    logger.info("Loading secrets from GCP Secret Manager...")
    
    # Load Pinecone API key
    if not config.PINECONE_API_KEY:
        pinecone_key = load_secret("pinecone-api-key")
        if pinecone_key:
            config.PINECONE_API_KEY = pinecone_key
            logger.info("✓ Pinecone API key loaded from Secret Manager")
    
    # Load Google API key (if using Gemini API)
    if not config.USE_VERTEX_AI and not config.GOOGLE_API_KEY:
        google_key = load_secret("google-api-key")
        if google_key:
            config.GOOGLE_API_KEY = google_key
            logger.info("✓ Google API key loaded from Secret Manager")
    
    # Load Google Project ID (if using Vertex AI)
    if config.USE_VERTEX_AI and not config.GOOGLE_PROJECT_ID:
        project_id = load_secret("google-project-id")
        if project_id:
            config.GOOGLE_PROJECT_ID = project_id
            logger.info("✓ Google Project ID loaded from Secret Manager")
    
    logger.info("Secret loading complete")

