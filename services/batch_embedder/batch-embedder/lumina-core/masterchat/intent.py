import re
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("masterchat.intent")

# Prometheus metrics
from prometheus_client import Counter
wiki_queries_total = Counter('wiki_queries_total', 'Total number of Wikipedia queries')

def detect_wiki(question: str) -> bool:
    """Detect if question is asking for Wikipedia information"""
    # Check for explicit Wikipedia mentions
    if "wikipedia" in question.lower() or question.lower().startswith("wiki "):
        return True
    
    # Check for question patterns that typically indicate Wikipedia queries
    question_patterns = [
        r"^(who|what|where|when|why|how)\s+.*",
        r".*explain.*",
        r".*tell me about.*",
        r".*describe.*"
    ]
    
    return any(re.match(pattern, question.lower()) for pattern in question_patterns)

def extract_topic(question: str) -> str:
    """Extract topic from question"""
    # Remove common prefixes and Wikipedia mentions
    cleaned = question.lower()
    cleaned = re.sub(r"^(explain|tell me about|describe|what is|who is|where is|when is|why is|how is)\s+", "", cleaned)
    cleaned = re.sub(r"\s+from\s+wikipedia.*$", "", cleaned)
    cleaned = re.sub(r"^wiki\s+", "", cleaned)
    
    # Take the first noun phrase (simple heuristic)
    words = cleaned.split()
    if not words:
        return "General Knowledge"
    
    # Capitalize first letter of each word
    return " ".join(word.capitalize() for word in words)

async def route_intent(message: str) -> Optional[Dict[str, Any]]:
    """Route user message to appropriate intent handler"""
    
    # Check for Wikipedia-related questions
    wiki_patterns = [
        r"what is",
        r"who is",
        r"tell me about",
        r"explain",
        r"describe",
        r"define"
    ]
    
    for pattern in wiki_patterns:
        if re.match(pattern, message.lower()):
            logger.info(f"Detected Wikipedia intent: {message}")
            return {
                "mode": "wiki_qa",
                "payload": {
                    "topic": message,
                    "question": message
                }
            }
    
    logger.info(f"No intent detected for message: {message}")
    return None 