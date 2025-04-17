"""
Center Module for LUMINA V7

This module implements the Center component shown in the architecture,
managing Myth, Lore, and Echo components while coordinating with
the Voice Integration System.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from .node_consciousness_manager import NodeConsciousnessManager
from .learning_coordinator import LearningCoordinator
from .database_integration import DatabaseIntegration
from .voice_integration_system import ComponentType, ComponentState

# Configure logging
logger = logging.getLogger("lumina_v7.center")

class CenterComponentType(Enum):
    MYTH = "myth"
    LORE = "lore"
    ECHO = "echo"
    GLYPH = "glyph"
    VOICE = "voice"

class ProcessingState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    LEARNING = "learning"
    ERROR = "error"

@dataclass
class CenterState:
    active_components: Set[str]
    processing_state: ProcessingState
    current_myth: Optional[Dict[str, Any]]
    current_lore: Optional[Dict[str, Any]]
    echo_queue: List[Dict[str, Any]]
    metrics: Dict[str, Any]

class CenterModule:
    """
    Core module for managing the Center components of LUMINA v7.
    Coordinates Myth, Lore, and Echo components while integrating
    with the Voice Integration System.
    """
    
    def __init__(self,
                 node_manager: NodeConsciousnessManager,
                 learning_coord: LearningCoordinator,
                 db_integration: DatabaseIntegration,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Center Module.
        
        Args:
            node_manager: Node consciousness manager instance
            learning_coord: Learning coordinator instance
            db_integration: Database integration instance
            config: Configuration options
        """
        self.node_manager = node_manager
        self.learning_coord = learning_coord
        self.db = db_integration
        self.config = config or {}
        
        # Initialize state
        self.state = CenterState(
            active_components=set(),
            processing_state=ProcessingState.IDLE,
            current_myth=None,
            current_lore=None,
            echo_queue=[],
            metrics={
                "myths_processed": 0,
                "lore_integrated": 0,
                "echoes_generated": 0,
                "processing_time": 0.0
            }
        )
        
        # Component states
        self.components: Dict[str, ComponentState] = {}
        
        # Processing queues
        self.myth_queue = asyncio.Queue()
        self.lore_queue = asyncio.Queue()
        self.integration_queue = asyncio.Queue()
        
        # Initialize core components
        self._initialize_center_components()
        
        logger.info("Center Module initialized")
    
    def _initialize_center_components(self) -> None:
        """Initialize the core Center components."""
        # Initialize Myth component
        self.register_component(
            "center_myth",
            CenterComponentType.MYTH,
            {"mode": "processing", "priority": "high"}
        )
        
        # Initialize Lore component
        self.register_component(
            "center_lore",
            CenterComponentType.LORE,
            {"mode": "storage", "priority": "medium"}
        )
        
        # Initialize Echo component
        self.register_component(
            "center_echo",
            CenterComponentType.ECHO,
            {"mode": "output", "priority": "high"}
        )
        
        # Create initial connections
        self.connect_components("center_myth", "center_lore")
        self.connect_components("center_lore", "center_echo")
    
    def register_component(self,
                         component_id: str,
                         component_type: CenterComponentType,
                         config: Dict[str, Any]) -> bool:
        """
        Register a new component in the Center module.
        
        Args:
            component_id: Unique identifier for the component
            component_type: Type of the component
            config: Component configuration
            
        Returns:
            bool: True if registration was successful
        """
        if component_id in self.components:
            logger.warning(f"Component {component_id} already registered")
            return False
        
        try:
            # Create component state
            self.components[component_id] = ComponentState(
                active=True,
                connected=False,
                last_update=0.0,
                metrics={},
                current_task=None
            )
            
            # Register with node manager
            self.node_manager.register_node(
                component_id,
                {
                    "type": component_type.value,
                    "config": config
                }
            )
            
            # Add to active components
            self.state.active_components.add(component_id)
            
            # Register for learning if applicable
            if config.get("learning_enabled", False):
                self.learning_coord.register_learning_node(component_id)
            
            logger.info(f"Registered {component_type.value} component: {component_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering component {component_id}: {e}")
            return False
    
    def connect_components(self, source_id: str, target_id: str) -> bool:
        """
        Create a connection between two components.
        
        Args:
            source_id: ID of the source component
            target_id: ID of the target component
            
        Returns:
            bool: True if connection was created successfully
        """
        if source_id not in self.components or target_id not in self.components:
            logger.error("Invalid component IDs for connection")
            return False
        
        try:
            # Register connection with node manager
            self.node_manager.connect_nodes(source_id, target_id)
            
            # Update component states
            self.components[source_id].connected = True
            self.components[target_id].connected = True
            
            logger.info(f"Created connection: {source_id} -> {target_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating connection: {e}")
            return False
    
    async def process_myth(self, myth_data: Dict[str, Any]) -> None:
        """
        Process myth data through the Center module.
        
        Args:
            myth_data: Myth data to process
        """
        try:
            # Update state
            self.state.processing_state = ProcessingState.PROCESSING
            self.state.current_myth = myth_data
            
            # Queue myth for processing
            await self.myth_queue.put({
                "type": "myth_data",
                "data": myth_data,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            # Update component state
            self.components["center_myth"].current_task = "processing_myth"
            
            # Process myth
            await self._process_myth_queue()
            
            # Update metrics
            self.state.metrics["myths_processed"] += 1
            
        except Exception as e:
            logger.error(f"Error processing myth: {e}")
            self.state.processing_state = ProcessingState.ERROR
    
    async def integrate_lore(self, lore_data: Dict[str, Any]) -> None:
        """
        Integrate lore data into the system.
        
        Args:
            lore_data: Lore data to integrate
        """
        try:
            # Update state
            self.state.current_lore = lore_data
            
            # Queue lore for integration
            await self.lore_queue.put({
                "type": "lore_data",
                "data": lore_data,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            # Update component state
            self.components["center_lore"].current_task = "integrating_lore"
            
            # Process lore
            await self._process_lore_queue()
            
            # Update metrics
            self.state.metrics["lore_integrated"] += 1
            
        except Exception as e:
            logger.error(f"Error integrating lore: {e}")
            self.state.processing_state = ProcessingState.ERROR
    
    async def generate_center_echo(self, source_data: Dict[str, Any]) -> None:
        """
        Generate echo from the Center module.
        
        Args:
            source_data: Source data for echo generation
        """
        try:
            # Add to echo queue
            self.state.echo_queue.append(source_data)
            
            # Queue for integration
            await self.integration_queue.put({
                "type": "echo_generation",
                "data": source_data,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            # Update component state
            self.components["center_echo"].current_task = "generating_echo"
            
            # Process echo
            await self._process_integration_queue()
            
            # Update metrics
            self.state.metrics["echoes_generated"] += 1
            
        except Exception as e:
            logger.error(f"Error generating echo: {e}")
            self.state.processing_state = ProcessingState.ERROR
    
    async def _process_myth_queue(self) -> None:
        """Process items in the myth queue."""
        while not self.myth_queue.empty():
            myth_data = await self.myth_queue.get()
            start_time = asyncio.get_event_loop().time()
            
            try:
                # Process myth data
                processed_myth = await self._analyze_myth(myth_data["data"])
                
                # Store results
                self.db.store_analysis_result(
                    "myth_processing",
                    {
                        "input": myth_data,
                        "output": processed_myth,
                        "processing_time": asyncio.get_event_loop().time() - start_time
                    }
                )
                
                # Queue for lore integration
                await self.lore_queue.put({
                    "type": "processed_myth",
                    "data": processed_myth,
                    "timestamp": asyncio.get_event_loop().time()
                })
                
            except Exception as e:
                logger.error(f"Error processing myth queue item: {e}")
    
    async def _process_lore_queue(self) -> None:
        """Process items in the lore queue."""
        while not self.lore_queue.empty():
            lore_data = await self.lore_queue.get()
            start_time = asyncio.get_event_loop().time()
            
            try:
                # Process lore data
                integrated_lore = await self._integrate_lore(lore_data["data"])
                
                # Store results
                self.db.store_analysis_result(
                    "lore_integration",
                    {
                        "input": lore_data,
                        "output": integrated_lore,
                        "processing_time": asyncio.get_event_loop().time() - start_time
                    }
                )
                
                # Queue for echo generation if needed
                if integrated_lore.get("requires_echo", False):
                    await self.integration_queue.put({
                        "type": "integrated_lore",
                        "data": integrated_lore,
                        "timestamp": asyncio.get_event_loop().time()
                    })
                
            except Exception as e:
                logger.error(f"Error processing lore queue item: {e}")
    
    async def _process_integration_queue(self) -> None:
        """Process items in the integration queue."""
        while not self.integration_queue.empty():
            integration_data = await self.integration_queue.get()
            start_time = asyncio.get_event_loop().time()
            
            try:
                # Process integration data
                result = await self._integrate_data(integration_data["data"])
                
                # Store results
                self.db.store_analysis_result(
                    "data_integration",
                    {
                        "input": integration_data,
                        "output": result,
                        "processing_time": asyncio.get_event_loop().time() - start_time
                    }
                )
                
            except Exception as e:
                logger.error(f"Error processing integration queue item: {e}")
    
    async def _analyze_myth(self, myth_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze myth data.
        
        Args:
            myth_data: Myth data to analyze
            
        Returns:
            Processed myth data
        """
        # Implement myth analysis logic
        return {
            "original": myth_data,
            "analyzed": True,
            "timestamp": asyncio.get_event_loop().time()
        }
    
    async def _integrate_lore(self, lore_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate lore data.
        
        Args:
            lore_data: Lore data to integrate
            
        Returns:
            Integrated lore data
        """
        # Implement lore integration logic
        return {
            "original": lore_data,
            "integrated": True,
            "timestamp": asyncio.get_event_loop().time()
        }
    
    async def _integrate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate general data.
        
        Args:
            data: Data to integrate
            
        Returns:
            Integrated data
        """
        # Implement data integration logic
        return {
            "original": data,
            "integrated": True,
            "timestamp": asyncio.get_event_loop().time()
        }
    
    def get_center_status(self) -> Dict[str, Any]:
        """
        Get current Center module status.
        
        Returns:
            Dictionary with module status
        """
        return {
            "state": {
                "active_components": list(self.state.active_components),
                "processing_state": self.state.processing_state.value,
                "current_myth": self.state.current_myth,
                "current_lore": self.state.current_lore,
                "echo_queue_size": len(self.state.echo_queue)
            },
            "components": {
                comp_id: {
                    "state": comp_state.__dict__
                }
                for comp_id, comp_state in self.components.items()
            },
            "metrics": self.state.metrics,
            "queue_sizes": {
                "myth": self.myth_queue.qsize(),
                "lore": self.lore_queue.qsize(),
                "integration": self.integration_queue.qsize()
            }
        }
    
    async def shutdown(self) -> None:
        """Shutdown the Center module."""
        try:
            # Update state
            self.state.processing_state = ProcessingState.IDLE
            
            # Clear queues
            while not self.myth_queue.empty():
                await self.myth_queue.get()
            while not self.lore_queue.empty():
                await self.lore_queue.get()
            while not self.integration_queue.empty():
                await self.integration_queue.get()
            
            # Update component states
            for component in self.components.values():
                component.active = False
                component.connected = False
            
            # Clear active components
            self.state.active_components.clear()
            
            # Store final state
            self.db.store_system_state("center_module_shutdown", {
                "final_state": self.get_center_status(),
                "shutdown_time": asyncio.get_event_loop().time()
            })
            
            logger.info("Center Module shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# For standalone testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        # Create test instances
        node_manager = NodeConsciousnessManager()
        learning_coord = LearningCoordinator(node_manager, None)
        db_integration = DatabaseIntegration()
        
        # Create module
        center = CenterModule(
            node_manager,
            learning_coord,
            db_integration
        )
        
        # Test myth processing
        await center.process_myth({"type": "test_myth", "data": "test data"})
        
        # Test lore integration
        await center.integrate_lore({"type": "test_lore", "data": "test data"})
        
        # Test echo generation
        await center.generate_center_echo({"type": "test_echo", "data": "test data"})
        
        # Get status
        status = center.get_center_status()
        print(f"Center status: {status}")
        
        # Shutdown
        await center.shutdown()
    
    # Run test
    asyncio.run(main()) 