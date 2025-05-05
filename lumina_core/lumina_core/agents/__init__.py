from .crawl_agent import CrawlAgent
from .summarise_agent import SummariseAgent
from .qa_agent import QAAgent

# Registry of available agents
AGENTS = {
    "crawl": CrawlAgent(),
    "summarise": SummariseAgent(),
    "qa": QAAgent()
}

def get_agent(name: str):
    """
    Get an agent by name.
    
    Args:
        name: The name of the agent to retrieve
        
    Returns:
        The requested agent instance
        
    Raises:
        KeyError: If the agent is not found
    """
    if name not in AGENTS:
        raise KeyError(f"Agent '{name}' not found")
    return AGENTS[name]

def list_agents():
    """
    List all available agents.
    
    Returns:
        Dictionary mapping agent names to their metadata
    """
    return {name: agent.to_dict() for name, agent in AGENTS.items()} 