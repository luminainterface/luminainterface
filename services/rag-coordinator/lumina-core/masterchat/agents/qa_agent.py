from typing import Dict, Any, List
import logging
from . import BaseAgent, register_agent
from ..llm_client import chat

logger = logging.getLogger("masterchat.agents.qa")

class QAAgent(BaseAgent):
    """Agent for answering questions using article summaries"""
    
    def __init__(self):
        register_agent(self)

    @property
    def name(self) -> str:
        return "QAAgent"

    @property
    def description(self) -> str:
        return "Answers questions using Wikipedia article summaries"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "enum": ["qa"]},
                "question": {"type": "string"},
                "context": {
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
            },
            "required": ["cmd", "question", "context"]
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "sources": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["answer", "confidence", "sources"]
        }

    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Answer a question using article summaries"""
        try:
            # Validate input
            if payload["cmd"] != "qa":
                raise ValueError("Invalid command for QAAgent")
            
            # Prepare prompt
            context = payload["context"]
            prompt = f"""Please answer the following question using only the provided context.
If the answer cannot be found in the context, say so.

Context Summary:
{context['summary']}

Key Facts:
{chr(10).join(f"- {fact}" for fact in context['facts'])}

Question: {payload['question']}

Return your response in this JSON format:
{{
    "answer": "your answer here",
    "confidence": 0.95,  # number between 0 and 1
    "sources": [
        "source 1",
        "source 2"
    ]
}}"""
            
            # Get answer from Mistral
            response = await chat(
                messages=[{"role": "system", "content": prompt}],
                temperature=0.3,
                max_tokens=1024
            )
            
            # Parse and return response
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"QAAgent error: {e}")
            raise 