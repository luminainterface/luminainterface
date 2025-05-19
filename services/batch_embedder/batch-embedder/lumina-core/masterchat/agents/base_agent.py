from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging

logger = logging.getLogger("masterchat.agents")

# Global registry for all agents
AGENT_REGISTRY: Dict[str, Any] = {}

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self):
        """Initialize agent and register it"""
        register_agent(self)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Agent description"""
        pass
    
    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """Input schema for the agent"""
        pass
    
    @property
    @abstractmethod
    def output_schema(self) -> Dict[str, Any]:
        """Output schema for the agent"""
        pass
    
    @abstractmethod
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent with input data"""
        pass
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data against schema"""
        try:
            # Create a Pydantic model from the schema
            InputModel = type('InputModel', (BaseModel,), {
                '__annotations__': self.input_schema
            })
            # Validate input
            InputModel(**input_data)
            return True
        except Exception as e:
            raise ValueError(f"Invalid input: {str(e)}")
    
    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        """Validate output data against schema"""
        try:
            # Create a Pydantic model from the schema
            OutputModel = type('OutputModel', (BaseModel,), {
                '__annotations__': self.output_schema
            })
            # Validate output
            OutputModel(**output_data)
            return True
        except Exception as e:
            raise ValueError(f"Invalid output: {str(e)}")

def register_agent(agent: BaseAgent):
    """Register an agent in the global registry"""
    AGENT_REGISTRY[agent.name] = agent
    logger.info(f"Registered agent: {agent.name}")

def get_agent(name: str) -> BaseAgent:
    """Get an agent by name"""
    if name not in AGENT_REGISTRY:
        raise KeyError(f"Agent not found: {name}")
    return AGENT_REGISTRY[name] 