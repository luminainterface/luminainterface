from typing import Dict, Any, List
import logging
import json
import aiohttp
from datetime import datetime
from .base_agent import BaseAgent, register_agent
from ..llm_client import chat
from jinja2 import Environment, FileSystemLoader
import os

logger = logging.getLogger("masterchat.agents.orchestrator")

class OrchestratorAgent(BaseAgent):
    """Agent for system orchestration and task management"""
    
    def __init__(self):
        super().__init__()
        self.env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "../prompts")))
        self.template = self.env.get_template("orchestrator.j2")
        
    @property
    def name(self) -> str:
        return "OrchestratorAgent"
    
    @property
    def description(self) -> str:
        return "Orchestrates system tasks and maintains system health"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "metrics": {
                    "type": "object",
                    "properties": {
                        "d_nodes": {"type": "integer"},
                        "d_edges": {"type": "integer"},
                        "entropy": {"type": "number"},
                        "redis_lag": {"type": "integer"},
                        "plan_p95": {"type": "number"},
                        "nn_recall": {"type": "number"}
                    }
                }
            },
            "required": ["metrics"]
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "crawl": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "prune": {
                    "type": ["string", "null"]
                },
                "focus": {
                    "type": ["object", "null"],
                    "properties": {
                        "cluster_id": {"type": "string"},
                        "reason": {"type": "string"}
                    }
                },
                "ui": {
                    "type": ["object", "null"],
                    "properties": {
                        "severity": {"type": "string"},
                        "msg": {"type": "string"}
                    }
                },
                "note": {"type": "string"}
            }
        }
    
    async def execute_crawl(self, topic: str, hops: int = 1, max_nodes: int = 40) -> bool:
        """Execute a crawl task"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:8400/crawl",
                    json={"seed": topic, "hops": hops, "max_nodes": max_nodes}
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Crawl execution error: {e}")
            return False
    
    async def check_embeddings(self) -> bool:
        """Check if embeddings are in Qdrant"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:6333/collections") as response:
                    if response.status == 200:
                        collections = await response.json()
                        return "embeddings" in collections
            return False
        except Exception as e:
            logger.error(f"Embedding check error: {e}")
            return False
    
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the orchestrator agent"""
        try:
            # Render prompt template with metrics
            prompt = self.template.render(**input_data["metrics"])
            
            # Get LLM response
            response = await chat(
                messages=[{"role": "system", "content": prompt}],
                temperature=0.3,
                max_tokens=1024
            )
            
            # Parse response
            task_list = json.loads(response.choices[0].message.content)
            
            # Execute tasks
            if task_list.get("crawl"):
                for topic in task_list["crawl"]:
                    success = await self.execute_crawl(topic)
                    if not success:
                        logger.error(f"Failed to execute crawl for topic: {topic}")
            
            # Check embeddings if needed
            if not await self.check_embeddings():
                logger.warning("Embeddings not found in Qdrant")
                # TODO: Enqueue re-embed batch
            
            return task_list
            
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            return {
                "crawl": [],
                "prune": None,
                "focus": None,
                "ui": {
                    "severity": "error",
                    "msg": f"Orchestrator error: {str(e)}"
                },
                "note": "Error in orchestrator execution"
            } 