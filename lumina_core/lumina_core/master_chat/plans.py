from typing import Dict, List, Any

# Static plans for different modes
PLANS: Dict[str, List[Dict[str, str]]] = {
    # Hard-coded three-step chain for wiki-qa
    "wiki_qa": [
        {"agent": "CrawlAgent"},
        {"agent": "SummariseAgent"},
        {"agent": "QAAgent"},
    ]
}

async def get_plan(mode: str) -> List[Dict[str, str]]:
    """
    Get the plan for a given mode.
    
    Args:
        mode: The mode to get the plan for
        
    Returns:
        List of agent steps to execute
    """
    return PLANS.get(mode, []) 