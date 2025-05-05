import logging
import os
from apscheduler.schedulers.asyncio import AsyncIOScheduler

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def setup_scheduler():
    """Initialize and start the background task scheduler."""
    scheduler = AsyncIOScheduler()
    scheduler.start()
    return scheduler

def verify_api_key(api_key: str = None) -> bool:
    """Verify that the provided API key is valid."""
    expected_key = os.getenv("LUMINA_API_KEY")
    if not expected_key:
        # Skip API key verification if no key is set
        return True
    return api_key == expected_key 