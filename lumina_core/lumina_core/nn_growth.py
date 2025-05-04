from typing import List, Dict, Any
import redis
import time
from prometheus_client import Gauge
from loguru import logger

# Redis client
redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

# Prometheus metrics
SYNAPSE_NODES = Gauge(
    "synapse_nodes_total",
    "Total number of nodes in the neural network"
)
SYNAPSE_NODES_LAST_UPDATE = Gauge(
    "synapse_nodes_total_last_update",
    "Timestamp of last node addition"
)

def bump_node(agent_name: str) -> None:
    """
    Increment node count for an agent.
    
    Args:
        agent_name: Name of the agent that created the node
    """
    try:
        # Increment node count
        redis_client.hincrby("synapse_nodes", agent_name, 1)
        
        # Update total
        total = sum(int(count) for count in redis_client.hgetall("synapse_nodes").values())
        SYNAPSE_NODES.set(total)
        
        # Update timestamp
        now = int(time.time())
        redis_client.set("synapse_nodes_last_update", now)
        SYNAPSE_NODES_LAST_UPDATE.set(now)
        
    except Exception as e:
        logger.error(f"Failed to bump node count: {str(e)}")

def get_stats() -> Dict[str, Any]:
    """
    Get neural network statistics.
    
    Returns:
        Dictionary containing:
            - total_nodes: int, Total number of nodes
            - agents: Dict[str, int], Number of nodes per agent
    """
    try:
        # Get node counts
        agent_counts = redis_client.hgetall("synapse_nodes")
        total = sum(int(count) for count in agent_counts.values())
        
        return {
            "total_nodes": total,
            "agents": {agent: int(count) for agent, count in agent_counts.items()}
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        return {"total_nodes": 0, "agents": {}} 