from typing import Dict, Any, List
from loguru import logger
from prometheus_client import Counter, Histogram
from ..utils.rate_limit import redis_rate_limit
from .base import BaseAgent
from ..llm.mistral import chat

# Prometheus metrics
SUMMARISE_REQUESTS = Counter(
    "summarise_requests_total",
    "Total number of summarise requests",
    ["status"]
)
SUMMARISE_DURATION = Histogram(
    "summarise_duration_seconds",
    "Time spent summarising",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

class SummariseAgent(BaseAgent):
    """Agent for summarizing Wikipedia articles."""
    
    def __init__(self):
        super().__init__(
            name="SummariseAgent",
            description="Summarizes Wikipedia articles using Mistral"
        )
    
    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the summarise agent.
        
        Args:
            payload: Dictionary containing:
                - articles: List[Dict], List of articles to summarize
                
        Returns:
            Dictionary containing:
                - status: str, "success" or "error"
                - summary: str, Summary if successful
                - facts: List[str], Interesting facts if successful
                - error: str, Error message if failed
        """
        # Validate input
        if "articles" not in payload:
            SUMMARISE_REQUESTS.labels(status="error").inc()
            return {
                "status": "error",
                "error": "Articles are required"
            }
        
        articles = payload["articles"]
        if not articles:
            SUMMARISE_REQUESTS.labels(status="error").inc()
            return {
                "status": "error",
                "error": "No articles to summarize"
            }
        
        try:
            # Rate limit by article count
            async with redis_rate_limit(f"summarise:{len(articles)}", 10, 60):
                with SUMMARISE_DURATION.time():
                    # Create prompt
                    prompt = self._create_prompt(articles)
                    
                    # Get summary from Mistral
                    response = await chat(messages=[{
                        "role": "user",
                        "content": prompt
                    }])
                    
                    # Parse response
                    summary, facts = self._parse_response(response)
                    
                    # Log success
                    SUMMARISE_REQUESTS.labels(status="success").inc()
                    
                    return {
                        "status": "success",
                        "summary": summary,
                        "facts": facts
                    }
                    
        except Exception as e:
            # Log error
            logger.error(f"Summarise failed: {str(e)}")
            SUMMARISE_REQUESTS.labels(status="error").inc()
            
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _create_prompt(self, articles: List[Dict[str, Any]]) -> str:
        """Create prompt for Mistral."""
        # Combine article contents
        content = "\n\n".join(article["content"] for article in articles)
        
        return f"""Please summarize the following Wikipedia articles and extract interesting facts:

{content}

Format your response as:
SUMMARY:
[Your summary here]

FACTS:
1. [Fact 1]
2. [Fact 2]
..."""
    
    def _parse_response(self, response: str) -> tuple[str, List[str]]:
        """Parse Mistral's response into summary and facts."""
        try:
            # Split into sections
            sections = response.split("\n\n")
            
            # Get summary
            summary = sections[0].replace("SUMMARY:", "").strip()
            
            # Get facts
            facts = []
            if len(sections) > 1:
                facts_text = sections[1].replace("FACTS:", "").strip()
                facts = [f.strip() for f in facts_text.split("\n") if f.strip()]
            
            return summary, facts
            
        except Exception as e:
            logger.error(f"Failed to parse response: {str(e)}")
            raise Exception("Failed to parse Mistral's response") 