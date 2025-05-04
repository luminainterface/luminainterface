from typing import Dict, Any, List
from loguru import logger
from prometheus_client import Counter, Histogram
from ..utils.rate_limit import redis_rate_limit
from .base import BaseAgent
from ..llm.mistral import chat
from ..memory.qdrant_store import similarity_search

# Prometheus metrics
QA_REQUESTS = Counter(
    "qa_requests_total",
    "Total number of QA requests",
    ["status"]
)
QA_DURATION = Histogram(
    "qa_duration_seconds",
    "Time spent answering questions",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

class QAAgent(BaseAgent):
    """Agent for answering questions using Wikipedia knowledge."""
    
    def __init__(self):
        super().__init__(
            name="QAAgent",
            description="Answers questions using Wikipedia knowledge"
        )
    
    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the QA agent.
        
        Args:
            payload: Dictionary containing:
                - question: str, The question to answer
                - articles: List[Dict], Optional, List of articles to use
                
        Returns:
            Dictionary containing:
                - status: str, "success" or "error"
                - answer: str, Answer if successful
                - sources: List[str], Sources used if successful
                - error: str, Error message if failed
        """
        # Validate input
        if "question" not in payload:
            QA_REQUESTS.labels(status="error").inc()
            return {
                "status": "error",
                "error": "Question is required"
            }
        
        question = payload["question"]
        
        try:
            # Rate limit by question
            async with redis_rate_limit(f"qa:{question}", 5, 60):
                with QA_DURATION.time():
                    # Get relevant articles
                    articles = payload.get("articles", [])
                    if not articles:
                        # Search Qdrant for relevant articles
                        articles = await self._search_articles(question)
                    
                    if not articles:
                        QA_REQUESTS.labels(status="error").inc()
                        return {
                            "status": "error",
                            "error": "No relevant articles found"
                        }
                    
                    # Create prompt
                    prompt = self._create_prompt(question, articles)
                    
                    # Get answer from Mistral
                    response = await chat(messages=[{
                        "role": "user",
                        "content": prompt
                    }])
                    
                    # Parse response
                    answer, sources = self._parse_response(response)
                    
                    # Log success
                    QA_REQUESTS.labels(status="success").inc()
                    
                    return {
                        "status": "success",
                        "answer": answer,
                        "sources": sources
                    }
                    
        except Exception as e:
            # Log error
            logger.error(f"QA failed: {str(e)}")
            QA_REQUESTS.labels(status="error").inc()
            
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _search_articles(self, question: str) -> List[Dict[str, Any]]:
        """Search Qdrant for relevant articles."""
        # TODO: Implement Qdrant search
        return []
    
    def _create_prompt(self, question: str, articles: List[Dict[str, Any]]) -> str:
        """Create prompt for Mistral."""
        # Combine article contents
        content = "\n\n".join(article["content"] for article in articles)
        
        return f"""Please answer the following question using the provided Wikipedia articles:

Question: {question}

Articles:
{content}

Format your response as:
ANSWER:
[Your answer here]

SOURCES:
1. [Source 1]
2. [Source 2]
..."""
    
    def _parse_response(self, response: str) -> tuple[str, List[str]]:
        """Parse Mistral's response into answer and sources."""
        try:
            # Split into sections
            sections = response.split("\n\n")
            
            # Get answer
            answer = sections[0].replace("ANSWER:", "").strip()
            
            # Get sources
            sources = []
            if len(sections) > 1:
                sources_text = sections[1].replace("SOURCES:", "").strip()
                sources = [s.strip() for s in sources_text.split("\n") if s.strip()]
            
            return answer, sources
            
        except Exception as e:
            logger.error(f"Failed to parse response: {str(e)}")
            raise Exception("Failed to parse Mistral's response") 