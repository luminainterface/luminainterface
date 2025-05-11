from typing import Dict, Any, List
import logging
import wikipedia
from .base_agent import BaseAgent

logger = logging.getLogger("masterchat.agents.crawl")

class CrawlAgent(BaseAgent):
    """Agent for crawling Wikipedia articles"""
    
    @property
    def name(self) -> str:
        return "CrawlAgent"
    
    @property
    def description(self) -> str:
        return "Crawls Wikipedia articles related to a given topic"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "topic": str,
            "max_nodes": int
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "articles": List[Dict[str, Any]]
        }
    
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the crawler agent"""
        # Validate input
        self.validate_input(input_data)
        
        topic = input_data["topic"]
        max_nodes = input_data["max_nodes"]
        
        try:
            # Search for articles
            search_results = wikipedia.search(topic, results=max_nodes)
            articles = []
            
            # Fetch article content
            for title in search_results:
                try:
                    page = wikipedia.page(title)
                    articles.append({
                        "title": page.title,
                        "url": page.url,
                        "content": page.content,
                        "summary": page.summary
                    })
                except wikipedia.exceptions.DisambiguationError as e:
                    # Handle disambiguation pages
                    logger.warning(f"Disambiguation page for {title}: {e.options}")
                    continue
                except wikipedia.exceptions.PageError:
                    logger.warning(f"Page not found: {title}")
                    continue
                except Exception as e:
                    logger.error(f"Error fetching {title}: {str(e)}")
                    continue
            
            # Validate output
            output = {"articles": articles}
            self.validate_output(output)
            
            return output
            
        except Exception as e:
            logger.error(f"Crawl error: {str(e)}")
            raise 