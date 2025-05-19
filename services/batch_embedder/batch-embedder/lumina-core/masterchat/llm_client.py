from mistralai.async_client import MistralAsyncClient
from pydantic_settings import BaseSettings
from pydantic import Field
import logging
from typing import Optional, Dict, Any, AsyncGenerator, Union
import os

logger = logging.getLogger("masterchat.llm")

class Settings(BaseSettings):
    MISTRAL_API_KEY: Optional[str] = Field(None, env="MISTRAL_API_KEY")
    MISTRAL_MODEL: str = Field("mistral-medium", env="MISTRAL_MODEL")
    LLM_TEMP: float = Field(0.3, env="LLM_TEMP")

    class Config:
        env_file = None  # Disable .env file loading
        case_sensitive = True

settings = Settings()

# Initialize Mistral client
mistral_client = None

if settings.MISTRAL_API_KEY:
    mistral_client = MistralAsyncClient(api_key=settings.MISTRAL_API_KEY)
    logger.info("Mistral client initialized")
else:
    logger.warning("Mistral API key not found in environment")

async def chat(
    messages: list[Dict[str, str]],
    temperature: float = None,
    max_tokens: int = 1024,
    stream: bool = False
) -> Union[Dict[str, Any], AsyncGenerator]:
    """
    Send chat completion request to Mistral.
    """
    if not mistral_client:
        raise RuntimeError("Mistral API key not configured - check environment variables")
    
    try:
        return await mistral_client.chat(
            model=settings.MISTRAL_MODEL,
            messages=messages,
            temperature=temperature or settings.LLM_TEMP,
            max_tokens=max_tokens,
            stream=stream
        )
    except Exception as e:
        logger.error(f"Mistral API error: {e}")
        raise

def get_token_count(text: str) -> int:
    """Get token count for text using Mistral's tokenizer"""
    # TODO: Implement proper token counting when Mistral releases their tokenizer
    # For now, use a rough estimate: 1 token ≈ 4 chars
    return len(text) // 4

def get_cost_estimate(tokens: int) -> float:
    """Get estimated cost for token count"""
    # TODO: Update with actual Mistral pricing when available
    return tokens * 0.00001  # $0.01 per 1K tokens (estimated) 