from typing import Dict, Any, List
from lumina_core.masterchat.agents.base_agent import AGENT_REGISTRY

# Predefined plans for common workflows
PLANS = {
    "wiki_qa": [
        {
            "agent": "CrawlAgent",
            "description": "Crawl Wikipedia for relevant articles",
            "input": {"cmd": "crawl", "topic": "{topic}", "depth": 2}
        },
        {
            "agent": "SummariseAgent",
            "description": "Summarize crawled articles",
            "input": {"cmd": "summarise", "articles": "{crawl_result}"}
        },
        {
            "agent": "QAAgent",
            "description": "Answer user question using summaries",
            "input": {"cmd": "qa", "question": "{question}", "context": "{summarise_result}"}
        }
    ]
}

def get_agent(name: str) -> Any:
    """Get an agent by name"""
    if name not in AGENT_REGISTRY:
        raise KeyError(f"Agent not found: {name}")
    return AGENT_REGISTRY[name] 