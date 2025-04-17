#!/usr/bin/env python3
"""
V5 Lumina Integration

This module serves as the integration point between the V5 Visualization System,
the Language Memory System, and the Frontend components. It provides the necessary
bridges, adapters, and initialization logic to create a cohesive system.
"""

import os
import sys
import json
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("v5_integration.log")
    ]
)
logger = logging.getLogger("v5-lumina-integration")

# Add project root to path if needed
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


class V5LuminaIntegration:
    """
    Main integration class for the V5 Lumina system.
    
    This class coordinates the initialization and connection of all system components:
    - V5 Visualization System
    - Language Memory System
    - Frontend components
    - Neural Linguistic Processor
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the V5 Lumina Integration system.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.mock_mode = self.config.get("mock_mode", False)
        self.components = {}
        
        logger.info(f"Initializing V5 Lumina Integration (mock_mode={self.mock_mode})")
        
    def initialize(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            Success status
        """
        logger.info("Starting V5 Lumina component initialization")
        
        # Initialize components in the correct order
        initialization_order = [
            self._initialize_language_memory,
            self._initialize_neural_linguistic_processor,
            self._initialize_visualization_bridge,
            self._initialize_frontend_components
        ]
        
        for init_func in initialization_order:
            success = init_func()
            if not success and not self.mock_mode:
                logger.error(f"Failed to initialize component: {init_func.__name__}")
                return False
        
        # Connect components
        self._connect_components()
        
        logger.info("V5 Lumina system initialization complete")
        return True
    
    def _initialize_language_memory(self) -> bool:
        """
        Initialize the Language Memory System.
        
        Returns:
            Success status
        """
        logger.info("Initializing Language Memory System")
        
        try:
            # Try to import the language memory integration module
            from language_memory_synthesis_integration import LanguageMemorySynthesisIntegration
            
            # Initialize the integration system
            memory_synthesis = LanguageMemorySynthesisIntegration()
            self.components["language_memory_synthesis"] = memory_synthesis
            
            # Try to initialize the Memory API
            try:
                from memory_api import MemoryAPI
                memory_api = MemoryAPI()
                self.components["memory_api"] = memory_api
                logger.info("Successfully initialized Memory API")
            except ImportError as e:
                logger.warning(f"Memory API not available: {e}")
                if not self.mock_mode:
                    logger.warning("Switching to mock mode for Memory API")
            
            logger.info("Language Memory System initialization complete")
            return True
            
        except ImportError as e:
            logger.warning(f"Language Memory System not available: {e}")
            if not self.mock_mode:
                logger.warning("Switching to mock mode for Language Memory System")
            return False
    
    def _initialize_neural_linguistic_processor(self) -> bool:
        """
        Initialize the Neural Linguistic Processor.
        
        Returns:
            Success status
        """
        logger.info("Initializing Neural Linguistic Processor")
        
        try:
            # Try to import the neural linguistic processor
            from neural_linguistic_processor import get_linguistic_processor
            
            # Initialize the processor
            processor = get_linguistic_processor(config={"mock_mode": self.mock_mode})
            self.components["neural_linguistic_processor"] = processor
            
            # Start the processor
            processor.start_processor()
            
            logger.info("Neural Linguistic Processor initialization complete")
            return True
            
        except ImportError as e:
            logger.warning(f"Neural Linguistic Processor not available: {e}")
            if not self.mock_mode:
                logger.warning("Switching to mock mode for Neural Linguistic Processor")
            return False
    
    def _initialize_visualization_bridge(self) -> bool:
        """
        Initialize the V5 Visualization Bridge.
        
        Returns:
            Success status
        """
        logger.info("Initializing V5 Visualization Bridge")
        
        try:
            # Try to import the visualization bridge
            from v5_integration.visualization_bridge import get_visualization_bridge
            
            # Initialize the bridge
            bridge = get_visualization_bridge(mock_mode=self.mock_mode)
            self.components["visualization_bridge"] = bridge
            
            logger.info("V5 Visualization Bridge initialization complete")
            return True
            
        except ImportError as e:
            logger.warning(f"V5 Visualization Bridge not available: {e}")
            if not self.mock_mode:
                logger.warning("Switching to mock mode for V5 Visualization")
            return False
    
    def _initialize_frontend_components(self) -> bool:
        """
        Initialize the Frontend components.
        
        Returns:
            Success status
        """
        logger.info("Initializing Frontend Components")
        
        try:
            # Try to initialize socket providers for frontend integration
            from memory_api_socket import MemoryAPISocketProvider
            
            # Initialize the Memory API socket provider
            memory_socket = MemoryAPISocketProvider(plugin_id="memory_api_socket")
            self.components["memory_api_socket"] = memory_socket
            
            # Try to initialize frontend socket manager
            try:
                # Import the frontend socket manager if available
                from v5.frontend_socket_manager import FrontendSocketManager
                socket_manager = FrontendSocketManager()
                self.components["frontend_socket_manager"] = socket_manager
                
                # Register socket providers with manager
                socket_manager.register_plugin(memory_socket)
                logger.info("Registered Memory API Socket with Frontend Socket Manager")
            except ImportError as e:
                logger.warning(f"Frontend Socket Manager not available: {e}")
            
            logger.info("Frontend Components initialization complete")
            return True
            
        except ImportError as e:
            logger.warning(f"Frontend Components not available: {e}")
            if not self.mock_mode:
                logger.warning("Switching to mock mode for Frontend Components")
            return False
    
    def _connect_components(self) -> None:
        """
        Connect all initialized components.
        """
        logger.info("Connecting V5 Lumina components")
        
        # Connect Neural Linguistic Processor to Language Memory System
        if "neural_linguistic_processor" in self.components and "language_memory_synthesis" in self.components:
            logger.info("Connecting Neural Linguistic Processor to Language Memory System")
            
            # Get references to components
            processor = self.components["neural_linguistic_processor"]
            memory_synthesis = self.components["language_memory_synthesis"]
            
            # Set up direct connection
            try:
                # Register processor as a component in memory synthesis
                if hasattr(memory_synthesis, "register_component"):
                    memory_synthesis.register_component("neural_linguistic_processor", processor)
                    logger.info("Neural Linguistic Processor registered with Language Memory System")
                
                # Set memory synthesis reference in processor
                if hasattr(processor, "set_memory_synthesis"):
                    processor.set_memory_synthesis(memory_synthesis)
                    logger.info("Memory Synthesis reference set in Neural Linguistic Processor")
            except Exception as e:
                logger.error(f"Error connecting Neural Linguistic Processor to Language Memory System: {e}")
        
        # Connect V5 Visualization to Language Memory System
        if "visualization_bridge" in self.components and "language_memory_synthesis" in self.components:
            logger.info("Connecting V5 Visualization to Language Memory System")
            
            # Get references to components
            bridge = self.components["visualization_bridge"]
            memory_synthesis = self.components["language_memory_synthesis"]
            
            # Set up direct connection
            try:
                # Set memory synthesis reference in visualization bridge
                if hasattr(bridge, "set_memory_system"):
                    bridge.set_memory_system(memory_synthesis)
                    logger.info("Memory Synthesis reference set in Visualization Bridge")
                
                # Connect Language Memory System to V5 visualization plugins if possible
                if "frontend_socket_manager" in self.components:
                    socket_manager = self.components["frontend_socket_manager"]
                    
                    # Import and register Language Memory Integration Plugin
                    try:
                        from v5.language_memory_integration import LanguageMemoryIntegrationPlugin
                        language_plugin = LanguageMemoryIntegrationPlugin()
                        
                        # Set language memory synthesis reference in plugin
                        if hasattr(language_plugin, "set_memory_system"):
                            language_plugin.set_memory_system(memory_synthesis)
                        
                        # Register plugin with socket manager
                        socket_manager.register_plugin(language_plugin)
                        logger.info("Language Memory Integration Plugin registered with Frontend Socket Manager")
                    except ImportError as e:
                        logger.warning(f"Language Memory Integration Plugin not available: {e}")
            except Exception as e:
                logger.error(f"Error connecting V5 Visualization to Language Memory System: {e}")
        
        # Connect Neural Linguistic Processor to V5 Visualization
        if "neural_linguistic_processor" in self.components and "visualization_bridge" in self.components:
            logger.info("Connecting Neural Linguistic Processor to V5 Visualization")
            
            # Get references to components
            processor = self.components["neural_linguistic_processor"]
            bridge = self.components["visualization_bridge"]
            
            # Set up direct connection
            try:
                # Register processor with visualization bridge
                if hasattr(bridge, "register_processor"):
                    bridge.register_processor(processor)
                    logger.info("Neural Linguistic Processor registered with Visualization Bridge")
            except Exception as e:
                logger.error(f"Error connecting Neural Linguistic Processor to V5 Visualization: {e}")
        
        # Connect v1-v2 interfaces to V5 system
        logger.info("Connecting v1-v2 interfaces to V5 system")
        try:
            # Import v1-v2 connector
            from interface_connector import InterfaceConnector
            
            # Create connector instance
            v1v2_connector = InterfaceConnector()
            self.components["v1v2_connector"] = v1v2_connector
            
            # Connect Language Memory System to v1-v2 interfaces
            if "language_memory_synthesis" in self.components:
                memory_synthesis = self.components["language_memory_synthesis"]
                v1v2_connector.connect_memory_system(memory_synthesis)
                logger.info("Connected Language Memory System to v1-v2 interfaces")
            
            # Connect v1-v2 interfaces to Frontend Socket Manager
            if "frontend_socket_manager" in self.components:
                socket_manager = self.components["frontend_socket_manager"]
                
                # Create bidirectional connection
                v1v2_connector.connect_to_socket_manager(socket_manager)
                logger.info("Connected v1-v2 interfaces to Frontend Socket Manager")
                
            # Initialize NodeSocket for v1-v2 communication
            from v5.node_socket import NodeSocket
            v1v2_socket = NodeSocket("v1v2_interface", "bridge")
            self.components["v1v2_socket"] = v1v2_socket
            
            # Connect v1v2 socket to central node
            if "frontend_socket_manager" in self.components:
                socket_manager = self.components["frontend_socket_manager"]
                v1v2_socket.connect_to(socket_manager.manager_socket)
                logger.info("Connected v1-v2 socket to Frontend Socket Manager")
        except ImportError as e:
            logger.warning(f"v1-v2 interface connector not available: {e}")
            logger.warning("v1-v2 interface integration skipped")
        except Exception as e:
            logger.error(f"Error connecting v1-v2 interfaces: {e}")
        
        # Connect v3-v4 components to V5 system
        logger.info("Connecting v3-v4 components to V5 system")
        try:
            # Import v3-v4 connector
            from breath_interface_connector import BreathInterfaceConnector
            
            # Create connector instance
            v3v4_connector = BreathInterfaceConnector()
            self.components["v3v4_connector"] = v3v4_connector
            
            # Connect v3-v4 components to Language Memory System
            if "language_memory_synthesis" in self.components:
                memory_synthesis = self.components["language_memory_synthesis"]
                v3v4_connector.connect_memory_system(memory_synthesis)
                logger.info("Connected v3-v4 components to Language Memory System")
            
            # Connect v3-v4 components to V5 Visualization
            if "visualization_bridge" in self.components:
                bridge = self.components["visualization_bridge"]
                v3v4_connector.connect_visualization(bridge)
                logger.info("Connected v3-v4 components to V5 Visualization")
            
            # Initialize NodeSocket for v3-v4 communication
            from v5.node_socket import NodeSocket
            v3v4_socket = NodeSocket("v3v4_interface", "bridge")
            self.components["v3v4_socket"] = v3v4_socket
            
            # Connect v3v4 socket to central node
            if "frontend_socket_manager" in self.components:
                socket_manager = self.components["frontend_socket_manager"]
                v3v4_socket.connect_to(socket_manager.manager_socket)
                logger.info("Connected v3-v4 socket to Frontend Socket Manager")
            
            # Register v3-v4 message handlers
            v3v4_socket.register_message_handler("breath_state_update", v3v4_connector.handle_breath_state)
            v3v4_socket.register_message_handler("v3_glyph_update", v3v4_connector.handle_glyph_update)
            logger.info("Registered v3-v4 message handlers")
        except ImportError as e:
            logger.warning(f"v3-v4 interface connector not available: {e}")
            logger.warning("v3-v4 interface integration skipped")
        except Exception as e:
            logger.error(f"Error connecting v3-v4 components: {e}")
        
        # Connect to Central Node if available
        logger.info("Connecting to Central Node")
        try:
            # Import central node
            from central_node import CentralNode
            
            # Try to get existing instance
            central_node = CentralNode()
            self.components["central_node"] = central_node
            
            # Register V5 components with central node
            for name, component in self.components.items():
                if name != "central_node":  # Avoid circular reference
                    central_node._register_component(f"v5_{name}", component)
            
            logger.info("Connected all V5 components to Central Node")
        except ImportError as e:
            logger.warning(f"Central Node not available: {e}")
            logger.warning("Central Node integration skipped")
        except Exception as e:
            logger.error(f"Error connecting to Central Node: {e}")
        
        logger.info("Component connections complete")
    
    def start_system(self) -> bool:
        """
        Start the V5 Lumina system.
        
        Returns:
            Success status
        """
        logger.info("Starting V5 Lumina system")
        
        # Start components in the correct order
        start_order = [
            "language_memory_synthesis",
            "memory_api",
            "neural_linguistic_processor",
            "visualization_bridge",
            "memory_api_socket",
            "frontend_socket_manager"
        ]
        
        for component_name in start_order:
            if component_name in self.components:
                component = self.components[component_name]
                logger.info(f"Starting component: {component_name}")
                
                # Call the start method if it exists
                if hasattr(component, "start") and callable(getattr(component, "start")):
                    try:
                        getattr(component, "start")()
                        logger.info(f"Component started: {component_name}")
                    except Exception as e:
                        logger.error(f"Failed to start component {component_name}: {e}")
                        if not self.mock_mode:
                            return False
        
        logger.info("V5 Lumina system started")
        return True
    
    def stop_system(self) -> None:
        """
        Stop the V5 Lumina system.
        """
        logger.info("Stopping V5 Lumina system")
        
        # Stop components in reverse order
        stop_order = [
            "frontend_socket_manager",
            "memory_api_socket",
            "visualization_bridge",
            "neural_linguistic_processor",
            "memory_api",
            "language_memory_synthesis"
        ]
        
        for component_name in stop_order:
            if component_name in self.components:
                component = self.components[component_name]
                logger.info(f"Stopping component: {component_name}")
                
                # Call the stop method if it exists
                if hasattr(component, "stop") and callable(getattr(component, "stop")):
                    try:
                        getattr(component, "stop")()
                        logger.info(f"Component stopped: {component_name}")
                    except Exception as e:
                        logger.error(f"Error stopping component {component_name}: {e}")
        
        logger.info("V5 Lumina system stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the V5 Lumina system.
        
        Returns:
            Status dictionary
        """
        status = {
            "system": "V5 Lumina Integration",
            "version": "1.0.0",
            "status": "online",
            "mock_mode": self.mock_mode,
            "initialized_components": list(self.components.keys()),
            "timestamp": time.time(),
            "components": {}
        }
        
        # Get status for each component
        for component_name, component in self.components.items():
            if hasattr(component, "get_status") and callable(getattr(component, "get_status")):
                try:
                    component_status = getattr(component, "get_status")()
                    status["components"][component_name] = component_status
                except Exception as e:
                    status["components"][component_name] = {"error": str(e)}
            else:
                status["components"][component_name] = {"status": "unknown"}
        
        return status


def get_integration_system(config: Dict[str, Any] = None) -> V5LuminaIntegration:
    """
    Get or create a V5LuminaIntegration instance.
    
    Args:
        config: Optional configuration
    
    Returns:
        Integration system instance
    """
    global _integration_instance
    if '_integration_instance' not in globals() or _integration_instance is None:
        _integration_instance = V5LuminaIntegration(config)
    return _integration_instance


def main() -> None:
    """
    Main entry point for the V5 Lumina Integration system.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="V5 Lumina Integration")
    
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--no-start", action="store_true", help="Initialize but don't start")
    parser.add_argument("--status", action="store_true", help="Print system status and exit")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        "mock_mode": args.mock,
        "test_mode": args.test
    }
    
    # Get the integration system
    integration = get_integration_system(config)
    
    # Initialize the system
    if not integration.initialize():
        logger.error("Failed to initialize V5 Lumina system")
        sys.exit(1)
    
    # Print status if requested
    if args.status:
        status = integration.get_status()
        print(json.dumps(status, indent=2))
        sys.exit(0)
    
    # Start the system unless --no-start is specified
    if not args.no_start:
        if not integration.start_system():
            logger.error("Failed to start V5 Lumina system")
            sys.exit(1)
    
    # Run test sequence if test mode is enabled
    if args.test:
        run_test_sequence(integration)
        integration.stop_system()
        sys.exit(0)
    
    # Main loop
    try:
        logger.info("V5 Lumina Integration running. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal. Stopping system...")
        integration.stop_system()


def run_test_sequence(integration: V5LuminaIntegration) -> None:
    """
    Run a test sequence to verify system functionality.
    
    Args:
        integration: The integration system
    """
    logger.info("Running test sequence")
    
    # Get system status
    status = integration.get_status()
    logger.info(f"System status: {status['status']}")
    
    # Get processor
    if "neural_linguistic_processor" in integration.components:
        processor = integration.components["neural_linguistic_processor"]
        
        # Process test text
        test_text = "The neural networks create fractal patterns that represent consciousness and language understanding."
        result = processor.process_text(test_text)
        
        logger.info(f"Processed text: {test_text}")
        logger.info(f"Analysis features: {result['analysis']['features']}")
        logger.info(f"Generated pattern with {len(result['pattern']['nodes'])} nodes")
    
    # Get visualization bridge
    if "visualization_bridge" in integration.components:
        bridge = integration.components["visualization_bridge"]
        
        # Visualize a topic
        if hasattr(bridge, "visualize_topic") and callable(getattr(bridge, "visualize_topic")):
            try:
                visualization_data = bridge.visualize_topic("neural networks", depth=3)
                logger.info(f"Visualized topic with {len(visualization_data.get('nodes', []))} nodes")
            except Exception as e:
                logger.error(f"Failed to visualize topic: {e}")
    
    # Get memory synthesis
    if "language_memory_synthesis" in integration.components:
        memory_synthesis = integration.components["language_memory_synthesis"]
        
        # Synthesize a topic
        if hasattr(memory_synthesis, "synthesize_topic") and callable(getattr(memory_synthesis, "synthesize_topic")):
            try:
                results = memory_synthesis.synthesize_topic("consciousness")
                logger.info(f"Synthesized topic with {len(results.get('related_topics', []))} related topics")
            except Exception as e:
                logger.error(f"Failed to synthesize topic: {e}")
    
    logger.info("Test sequence complete")


if __name__ == "__main__":
    main() 