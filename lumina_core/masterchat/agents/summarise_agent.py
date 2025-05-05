from typing import Dict, Any, List
import logging
from .base_agent import BaseAgent, register_agent
from ..llm_client import chat

logger = logging.getLogger("masterchat.agents.summarise")

class SummariseAgent(BaseAgent):
    """Agent for summarizing Wikipedia articles"""
    
    def __init__(self):
        register_agent(self)

    @property
    def name(self) -> str:
        return "SummariseAgent"

    @property
    def description(self) -> str:
        return "Summarizes Wikipedia articles and extracts key facts"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "enum": ["summarise"]},
                "articles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["title", "content"]
                    }
                }
            },
            "required": ["cmd", "articles"]
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "facts": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["summary", "facts"]
        }

    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize Wikipedia articles"""
        try:
            # Validate input
            if payload["cmd"] != "summarise":
                raise ValueError("Invalid command for SummariseAgent")
            
            # Prepare prompt
            articles_text = "\n\n".join([
                f"Article: {article['title']}\n{article['content']}"
                for article in payload["articles"]
            ])
            
            prompt = f"""Please analyze these Wikipedia articles and provide:
1. A concise summary (2-3 paragraphs)
2. A list of key facts (5-7 bullet points)

Articles:
{articles_text}

Return your response in this JSON format:
{{
    "summary": "concise summary here",
    "facts": [
        "key fact 1",
        "key fact 2",
        ...
    ]
}}"""
            
            # Get summary from Mistral
            response = await chat(
                messages=[{"role": "system", "content": prompt}],
                temperature=0.3,
                max_tokens=1024
            )
            
            # Parse and return response
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"SummariseAgent error: {e}")
            raise 