#!/usr/bin/env python3
"""
V5 Visualization Bridge

This module provides a bridge between the Language Memory System and
the V5 Fractal Echo Visualization system.
"""

import os
import sys
import logging
from pathlib import Path
import json
from typing import Dict, Any, Optional, List, Union

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("v5_visualization_bridge")

# Try to import required components
try:
    from src.language_memory_synthesis_integration import LanguageMemorySynthesisIntegration
    logger.info("Successfully imported LanguageMemorySynthesisIntegration")
except ImportError as e:
    logger.error(f"Failed to import LanguageMemorySynthesisIntegration: {str(e)}")

try:
    from src.v5.frontend_socket_manager import FrontendSocketManager
    from src.v5.language_memory_integration import LanguageMemoryIntegrationPlugin
    V5_AVAILABLE = True
    logger.info("Successfully imported V5 components")
except ImportError as e:
    V5_AVAILABLE = False
    logger.error(f"Failed to import V5 components: {str(e)}")


class VisualizationBridge:
    """
    Bridge between Language Memory System and V5 Visualization
    
    This class provides:
    1. Initialization of all necessary components
    2. Simplified API for visualization integration
    3. Fallback mechanisms when components are missing
    4. Data transformation between systems
    """
    
    def __init__(self):
        """Initialize the visualization bridge"""
        self.memory_system = None
        self.socket_manager = None
        self.language_integration_plugin = None
        self.v5_visualization_available = False
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all required components"""
        # Try to initialize language memory synthesis
        try:
            self.memory_system = LanguageMemorySynthesisIntegration()
            logger.info("Memory system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {str(e)}")
            return
        
        # Check if V5 components are available
        if not V5_AVAILABLE:
            logger.warning("V5 visualization components not available")
            return
        
        # Try to initialize V5 components
        try:
            # Create socket manager
            self.socket_manager = FrontendSocketManager()
            
            # Create language memory integration plugin
            self.language_integration_plugin = LanguageMemoryIntegrationPlugin(
                plugin_id="language_memory_integration"
            )
            
            # Add reference to memory system
            if hasattr(self.language_integration_plugin, "language_memory_synthesis"):
                self.language_integration_plugin.language_memory_synthesis = self.memory_system
                self.language_integration_plugin.mock_mode = False
                logger.info("Connected language memory synthesis to integration plugin")
            
            # Register plugin with socket manager
            self.socket_manager.register_plugin(self.language_integration_plugin)
            
            self.v5_visualization_available = True
            logger.info("V5 visualization components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize V5 visualization components: {str(e)}")
    
    def is_visualization_available(self) -> bool:
        """Check if V5 visualization is available"""
        return self.v5_visualization_available
    
    def get_socket_manager(self) -> Optional[FrontendSocketManager]:
        """Get the socket manager for UI integration"""
        return self.socket_manager
    
    def get_memory_system(self) -> Optional[LanguageMemorySynthesisIntegration]:
        """Get the memory system"""
        return self.memory_system
    
    def visualize_topic(self, topic: str, depth: int = 3) -> Dict[str, Any]:
        """
        Visualize a topic using V5 visualization system
        
        Args:
            topic: The topic to visualize
            depth: The search depth (1-5)
            
        Returns:
            Visualization data or error information
        """
        if not self.memory_system:
            return {"error": "Memory system not available"}
        
        if not self.v5_visualization_available:
            return {"error": "V5 visualization not available"}
        
        try:
            # Process the data through the language integration plugin
            visualization_data = self.language_integration_plugin.process_language_data(topic, depth)
            return visualization_data
        except Exception as e:
            error_msg = f"Error visualizing topic: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_available_visualization_components(self) -> List[str]:
        """Get a list of available visualization components"""
        if not self.v5_visualization_available:
            return []
        
        components = []
        
        # Check if fractal pattern panel is available
        try:
            from src.v5.ui.panels.fractal_pattern_panel import FractalPatternPanel
            components.append("fractal_pattern_panel")
        except ImportError:
            pass
        
        # Check if node consciousness panel is available
        try:
            from src.v5.ui.panels.node_consciousness_panel import NodeConsciousnessPanel
            components.append("node_consciousness_panel")
        except ImportError:
            pass
        
        # Check if network visualization panel is available
        try:
            from src.v5.ui.panels.network_visualization_panel import NetworkVisualizationPanel
            components.append("network_visualization_panel")
        except ImportError:
            pass
        
        # Check if memory synthesis panel is available
        try:
            from src.v5.ui.panels.memory_synthesis_panel import MemorySynthesisPanel
            components.append("memory_synthesis_panel")
        except ImportError:
            pass
        
        return components
    
    def create_visualization_panel(self, panel_type: str) -> Optional[Any]:
        """
        Create a visualization panel of the specified type
        
        Args:
            panel_type: Type of panel to create
            
        Returns:
            The created panel or None if not available
        """
        if not self.v5_visualization_available:
            return None
        
        if not self.socket_manager:
            return None
        
        try:
            if panel_type == "fractal_pattern_panel":
                from src.v5.ui.panels.fractal_pattern_panel import FractalPatternPanel
                return FractalPatternPanel(self.socket_manager)
            
            elif panel_type == "node_consciousness_panel":
                from src.v5.ui.panels.node_consciousness_panel import NodeConsciousnessPanel
                return NodeConsciousnessPanel(self.socket_manager)
            
            elif panel_type == "network_visualization_panel":
                from src.v5.ui.panels.network_visualization_panel import NetworkVisualizationPanel
                return NetworkVisualizationPanel(self.socket_manager)
            
            elif panel_type == "memory_synthesis_panel":
                from src.v5.ui.panels.memory_synthesis_panel import MemorySynthesisPanel
                return MemorySynthesisPanel(self.socket_manager)
            
            return None
        except Exception as e:
            logger.error(f"Error creating visualization panel: {str(e)}")
            return None


# Singleton instance
_bridge_instance = None


def get_visualization_bridge() -> VisualizationBridge:
    """Get the visualization bridge singleton instance"""
    global _bridge_instance
    
    if _bridge_instance is None:
        _bridge_instance = VisualizationBridge()
    
    return _bridge_instance


if __name__ == "__main__":
    # Simple test to verify bridge functionality
    bridge = get_visualization_bridge()
    
    # Check if visualization is available
    if bridge.is_visualization_available():
        print("V5 visualization is available")
        
        # Get available visualization components
        components = bridge.get_available_visualization_components()
        print(f"Available components: {components}")
        
        # Visualize a topic
        visualization_data = bridge.visualize_topic("neural networks", 3)
        
        if "error" in visualization_data:
            print(f"Error: {visualization_data['error']}")
        else:
            print("Visualization data generated successfully")
    else:
        print("V5 visualization is not available") 