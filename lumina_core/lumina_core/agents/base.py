from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self):
        self.name = "BaseAgent"
        self.description = "Base agent class"
    
    @abstractmethod
    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent with the given payload.
        
        Args:
            payload: Dictionary containing the input data for the agent
            
        Returns:
            Dictionary containing the agent's output
        """
        pass
    
    def to_dict(self) -> Dict[str, str]:
        """
        Convert agent metadata to dictionary.
        
        Returns:
            Dictionary containing agent name and description
        """
        return {
            "name": self.name,
            "description": self.description
        } 