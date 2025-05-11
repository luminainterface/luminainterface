from typing import List, Dict, Optional
import wikipediaapi
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class WikiClient:
    def __init__(self, user_agent: str = "LuminaCrawler/1.0", language: str = "en"):
        self.wiki = wikipediaapi.Wikipedia(
            user_agent=user_agent,
            language=language
        )
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_page(self, title: str) -> Optional[wikipediaapi.WikipediaPage]:
        """Fetch a Wikipedia page by title with retry logic"""
        try:
            page = self.wiki.page(title)
            if page.exists():
                return page
            return None
        except Exception as e:
            logger.error(f"Error fetching Wikipedia page {title}: {str(e)}")
            raise
            
    def get_links(self, page: wikipediaapi.WikipediaPage, max_links: int = 10) -> List[str]:
        """Get outgoing links from a Wikipedia page"""
        try:
            links = []
            for title in page.links.keys():
                if len(links) >= max_links:
                    break
                if not title.startswith(("File:", "Template:", "Category:", "Wikipedia:")):
                    links.append(title)
            return links
        except Exception as e:
            logger.error(f"Error getting links from page {page.title}: {str(e)}")
            return []

    def get_summary(self, page: wikipediaapi.WikipediaPage) -> str:
        """Get the summary of a Wikipedia page"""
        try:
            return page.summary
        except Exception as e:
            logger.error(f"Error getting summary for page {page.title}: {str(e)}")
            return ""

    def get_full_text(self, page: wikipediaapi.WikipediaPage) -> str:
        """Get the full text content of a Wikipedia page"""
        try:
            return page.text
        except Exception as e:
            logger.error(f"Error getting full text for page {page.title}: {str(e)}")
            return "" 