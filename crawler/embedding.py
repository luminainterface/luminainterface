from sentence_transformers import SentenceTransformer
import asyncio
import logging

logger = logging.getLogger(__name__)

# Initialize the model
_model = SentenceTransformer("all-MiniLM-L6-v2")

async def embed(text: str) -> list[float]:
    """Embed text using sentence-transformers model."""
    try:
        # Run the CPU-bound embedding in a thread pool
        embedding = await asyncio.to_thread(_model.encode, text)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error embedding text: {str(e)}")
        raise 