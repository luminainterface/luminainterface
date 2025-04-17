import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def get_mistral_api_key():
    """
    Securely retrieve the Mistral API key from environment variable.
    Returns None if not found.
    """
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        logger.warning("MISTRAL_API_KEY environment variable not set")
        return None
    return api_key

def setup_api_config():
    """
    Setup API configuration and verify required keys are present
    """
    if not get_mistral_api_key():
        logger.error("Missing required Mistral API key. Please set MISTRAL_API_KEY environment variable")
        return False
    return True 