import asyncio
import logging
import os
from app.core.crawler import Crawler
from app.core.concept_client import ConceptClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_concept_registration():
    """Test concept registration through both adapter and direct client"""
    # Initialize clients
    concept_dict_url = os.getenv("CONCEPT_DICT_URL", "http://localhost:8526")
    crawler = Crawler(
        redis_url="redis://localhost:6379",
        qdrant_url="http://localhost:6333",
        graph_api_url="http://localhost:8200",
        concept_dict_url=concept_dict_url
    )
    
    # Test concept
    test_concept = "Artificial Intelligence"
    
    try:
        # Try direct registration first
        logger.info("Testing direct concept registration...")
        success = await crawler.concept_client.add_concept(test_concept, {
            "term": test_concept,
            "definition": "The simulation of human intelligence by machines",
            "sources": ["test"],
            "usage_count": 0
        })
        logger.info(f"Direct registration {'succeeded' if success else 'failed'}")
        
        # Try crawling the concept
        logger.info("Testing concept crawling...")
        crawl_success = await crawler.crawl(test_concept)
        logger.info(f"Crawl {'succeeded' if crawl_success else 'failed'}")
        
        # Verify concept was registered
        concept_data = await crawler.concept_client.get_concept(test_concept)
        if concept_data:
            logger.info(f"Successfully retrieved concept: {concept_data}")
        else:
            logger.error("Failed to retrieve concept after registration")
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
    finally:
        # Cleanup
        await crawler.concept_client.client.aclose()

if __name__ == "__main__":
    asyncio.run(test_concept_registration()) 