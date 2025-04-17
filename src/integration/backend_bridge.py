#!/usr/bin/env python3
"""
Backend Bridge Integration Module

This module provides the central integration point for all backend components of the Lumina Neural Network System.
It coordinates the initialization, connection, and communication between all major subsystems.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend_bridge")

@dataclass
class ComponentStatus:
    """Status information for a component"""
    initialized: bool = False
    running: bool = False
    error: Optional[str] = None
    mock: bool = False
    dependencies: List[str] = None

class BackendBridge:
    """
    Central integration point for all backend components.
    
    This class:
    1. Initializes all required components
    2. Manages component dependencies
    3. Handles inter-component communication
    4. Provides unified access to all subsystems
    5. Manages version compatibility
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Backend Bridge.
        
        Args:
            config: Configuration parameters for the bridge system
        """
        self.config = config or {}
        self.components = {}
        self.component_status = {}
        self.message_handlers = {}
        self.event_queues = {}
        
        # Initialize core components
        self._initialize_core_components()
        
    def _initialize_core_components(self):
        """Initialize all core components in the correct order"""
        initialization_order = [
            self._initialize_version_bridge,
            self._initialize_language_memory,
            self._initialize_neural_playground,
            self._initialize_visualization,
            self._initialize_consciousness,
            self._initialize_database
        ]
        
        for init_func in initialization_order:
            try:
                init_func()
            except Exception as e:
                logger.error(f"Failed to initialize component {init_func.__name__}: {e}")
                
    def _initialize_version_bridge(self):
        """Initialize the version bridge system"""
        from src.integration.version_bridge import VersionBridgeManager
        
        self.version_bridge = VersionBridgeManager(self.config)
        self.components["version_bridge"] = self.version_bridge
        self.component_status["version_bridge"] = ComponentStatus(
            initialized=True,
            running=False,
            dependencies=[]
        )
        
    def _initialize_language_memory(self):
        """Initialize the language memory system"""
        from src.language_memory_synthesis_integration import LanguageMemorySynthesisIntegration
        
        self.language_memory = LanguageMemorySynthesisIntegration()
        self.components["language_memory"] = self.language_memory
        self.component_status["language_memory"] = ComponentStatus(
            initialized=True,
            running=False,
            dependencies=["version_bridge"]
        )
        
    def _initialize_neural_playground(self):
        """Initialize the neural playground system"""
        from src.neural.playground import NeuralPlayground
        
        self.neural_playground = NeuralPlayground()
        self.components["neural_playground"] = self.neural_playground
        self.component_status["neural_playground"] = ComponentStatus(
            initialized=True,
            running=False,
            dependencies=["language_memory"]
        )
        
    def _initialize_visualization(self):
        """Initialize the visualization system"""
        from src.visualization.visualization_manager import VisualizationManager
        
        self.visualization = VisualizationManager()
        self.components["visualization"] = self.visualization
        self.component_status["visualization"] = ComponentStatus(
            initialized=True,
            running=False,
            dependencies=["neural_playground"]
        )
        
    def _initialize_consciousness(self):
        """Initialize the consciousness system"""
        from src.consciousness.consciousness_manager import ConsciousnessManager
        
        self.consciousness = ConsciousnessManager()
        self.components["consciousness"] = self.consciousness
        self.component_status["consciousness"] = ComponentStatus(
            initialized=True,
            running=False,
            dependencies=["neural_playground", "language_memory"]
        )
        
    def _initialize_database(self):
        """Initialize the database system"""
        from src.database.database_manager import DatabaseManager
        
        self.database = DatabaseManager()
        self.components["database"] = self.database
        self.component_status["database"] = ComponentStatus(
            initialized=True,
            running=False,
            dependencies=["version_bridge"]
        )
        
    def connect_components(self):
        """Connect all components together"""
        # Connect version bridge to other components
        self.version_bridge.connect_to_language_memory(self.language_memory)
        self.version_bridge.connect_to_neural_playground(self.neural_playground)
        self.version_bridge.connect_to_visualization(self.visualization)
        self.version_bridge.connect_to_consciousness(self.consciousness)
        self.version_bridge.connect_to_database(self.database)
        
        # Connect language memory to neural playground
        self.language_memory.connect_to_neural_playground(self.neural_playground)
        
        # Connect neural playground to visualization
        self.neural_playground.connect_to_visualization(self.visualization)
        
        # Connect consciousness to all components
        self.consciousness.connect_to_language_memory(self.language_memory)
        self.consciousness.connect_to_neural_playground(self.neural_playground)
        self.consciousness.connect_to_visualization(self.visualization)
        
        # Update component status
        for component in self.components.values():
            self.component_status[component.__class__.__name__].running = True
            
    def start(self):
        """Start all components"""
        for component in self.components.values():
            if hasattr(component, 'start'):
                component.start()
                
    def stop(self):
        """Stop all components"""
        for component in reversed(self.components.values()):
            if hasattr(component, 'stop'):
                component.stop()
                
    def get_component(self, component_name: str) -> Optional[Any]:
        """Get a specific component by name"""
        return self.components.get(component_name)
        
    def get_status(self) -> Dict[str, ComponentStatus]:
        """Get the status of all components"""
        return self.component_status
        
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register a message handler for a specific message type"""
        self.message_handlers[message_type] = handler
        
    def send_message(self, message_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to the appropriate handler"""
        if message_type in self.message_handlers:
            return self.message_handlers[message_type](data)
        return {"status": "error", "message": f"No handler for message type {message_type}"}
        
    def get_event_queue(self, queue_name: str) -> asyncio.Queue:
        """Get or create an event queue"""
        if queue_name not in self.event_queues:
            self.event_queues[queue_name] = asyncio.Queue()
        return self.event_queues[queue_name]
        
    async def process_events(self):
        """Process events from all queues"""
        while True:
            for queue_name, queue in self.event_queues.items():
                try:
                    event = await queue.get()
                    # Process event based on type
                    if event["type"] in self.message_handlers:
                        await self.message_handlers[event["type"]](event["data"])
                except asyncio.QueueEmpty:
                    continue
                except Exception as e:
                    logger.error(f"Error processing event from {queue_name}: {e}")
                    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from all components"""
        metrics = {}
        for name, component in self.components.items():
            if hasattr(component, 'get_metrics'):
                metrics[name] = component.get_metrics()
        return metrics 
 