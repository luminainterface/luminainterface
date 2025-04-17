#!/usr/bin/env python3
"""
Language Memory V5 Bridge

This module provides a bridge between the Language Memory System and
the V5 Fractal Echo Visualization system.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/language_memory_v5_bridge.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("language_memory_v5_bridge")

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Import required components
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


class LanguageMemoryV5Bridge:
    """
    Bridge between Language Memory System and V5 Visualization System
    
    This class provides:
    1. Initialization of all necessary components
    2. Simplified API for visualization integration
    3. Connection between language memory and visualization system
    4. Data transformation between systems
    """
    
    def __init__(self, mock_mode: bool = False):
        """
        Initialize the Language Memory V5 Bridge
        
        Args:
            mock_mode: Use mock data instead of actual language memory system
        """
        logger.info("Initializing Language Memory V5 Bridge")
        
        self.memory_system = None
        self.socket_manager = None
        self.language_integration_plugin = None
        self.mock_mode = mock_mode
        self.v5_visualization_available = False
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all required components"""
        # Initialize language memory synthesis
        try:
            if self.mock_mode:
                logger.info("Using mock mode for language memory synthesis")
                self.memory_system = self._create_mock_memory_system()
            else:
                self.memory_system = LanguageMemorySynthesisIntegration()
            
            logger.info("Language memory system initialized successfully")
            except Exception as e:
            logger.error(f"Failed to initialize language memory system: {str(e)}")
            # Create mock system if real one fails
            logger.info("Falling back to mock memory system")
            self.memory_system = self._create_mock_memory_system()
            self.mock_mode = True
        
        # Check if V5 components are available
        if not V5_AVAILABLE:
            logger.warning("V5 visualization components not available")
            return
        
        # Initialize V5 components
        try:
            # Create socket manager
            self.socket_manager = FrontendSocketManager()
            
            # Create language memory integration plugin
            self.language_integration_plugin = LanguageMemoryIntegrationPlugin()
            
            # Set memory system in the plugin if possible
            if hasattr(self.language_integration_plugin, 'set_memory_system'):
                self.language_integration_plugin.set_memory_system(self.memory_system)
            
            # Register plugin with socket manager
            self.socket_manager.register_plugin(self.language_integration_plugin)
            
            self.v5_visualization_available = True
            logger.info("V5 visualization components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize V5 visualization components: {str(e)}")
    
    def _create_mock_memory_system(self):
        """Create a mock memory system for testing"""
        logger.info("Creating mock memory system")
        
        class MockMemorySystem:
            """Mock implementation of LanguageMemorySynthesisIntegration"""
            
            def __init__(self):
                self.topics = ["neural networks", "consciousness", "language", "memory", "visualization"]
                self.synthesis_count = 0
            
            def synthesize_topic(self, topic, depth=3):
                """Mock topic synthesis"""
                self.synthesis_count += 1
                
                # Create a mock synthesis result
                return {
                    "status": "success",
                    "topic": topic,
                    "depth": depth,
                    "synthesis_results": {
                        "synthesized_memory": {
                            "topics": [topic],
                            "core_understanding": f"This is a mock understanding of {topic}.",
                            "novel_insights": [
                                f"Mock insight 1 about {topic}",
                                f"Mock insight 2 about {topic}",
                                f"Mock insight 3 about {topic}"
                            ]
                        },
                        "related_topics": [
                            {"topic": f"Related to {topic} 1", "relevance": 0.9},
                            {"topic": f"Related to {topic} 2", "relevance": 0.7},
                            {"topic": f"Related to {topic} 3", "relevance": 0.5}
                        ]
                    },
                    "component_contributions": {
                        "mock_component": {"contribution": "Mock data"}
                    }
                }
            
            def get_stats(self):
                """Mock statistics"""
                return {
                    "synthesis_stats": {
                        "synthesis_count": self.synthesis_count,
                        "topics_synthesized": set(self.topics),
                        "last_synthesis_timestamp": time.time()
                    },
                    "performance_metrics": {
                        "avg_synthesis_time": 0.1,
                        "cache_hits": 5,
                        "cache_misses": 3
                    }
                }
            
            def register_component(self, name, component):
                """Mock component registration"""
                logger.info(f"Mock registered component: {name}")
                return True
        
        return MockMemorySystem()
    
    def connect(self):
        """Connect the bridge components"""
        if not self.socket_manager or not self.language_integration_plugin:
            logger.error("Cannot connect bridge: components not initialized")
            return False
        
        # This method can be extended to establish additional connections
        # or restart connections if they were broken
        
        logger.info("Bridge components connected")
        return True
    
    def synthesize_topic(self, topic: str, depth: int = 3) -> Dict[str, Any]:
        """
        Synthesize a topic using the language memory system
        
        Args:
            topic: The topic to synthesize
            depth: The search depth (1-5)
            
        Returns:
            Synthesis results
        """
        if not self.memory_system:
            logger.error("Language memory system not available")
            return {"error": "Language memory system not available"}
        
        try:
            # Synthesize topic using the language memory system
            logger.info(f"Synthesizing topic: {topic}, depth: {depth}")
            synthesis_results = self.memory_system.synthesize_topic(topic, depth)
            
            # Process with V5 visualization if available
            if self.v5_visualization_available and self.language_integration_plugin:
                try:
                    logger.info("Processing with V5 visualization")
                    visualization_data = self.language_integration_plugin.process_language_data(topic, depth)
                    logger.info("V5 visualization processing complete")
                except Exception as e:
                    logger.error(f"Error processing with V5 visualization: {str(e)}")
            
            return synthesis_results
        except Exception as e:
            logger.error(f"Error synthesizing topic: {str(e)}")
            return {"error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the language memory system
        
        Returns:
            Statistics from the language memory system
        """
        if not self.memory_system:
            logger.error("Language memory system not available")
            return {"error": "Language memory system not available"}
        
        try:
            # Get stats from the language memory system
            logger.info("Getting language memory statistics")
            stats = self.memory_system.get_stats()
            return stats
        except Exception as e:
            logger.error(f"Error getting language memory statistics: {str(e)}")
            return {"error": str(e)}
    
    def get_visualization_plugin(self):
        """
        Get the language memory integration plugin
            
        Returns:
            The language memory integration plugin
        """
        return self.language_integration_plugin
    
    def get_socket_manager(self):
        """
        Get the socket manager
        
        Returns:
            The socket manager
        """
        return self.socket_manager
    
    def is_visualization_available(self) -> bool:
        """
        Check if V5 visualization is available
        
        Returns:
            True if V5 visualization is available, False otherwise
        """
        return self.v5_visualization_available
    
    def is_using_mock_data(self) -> bool:
        """
        Check if the bridge is using mock data
            
        Returns:
            True if mock data is being used, False otherwise
        """
        return self.mock_mode
    
    def set_mock_mode(self, mock_mode: bool):
        """
        Set the mock mode
        
        Args:
            mock_mode: Use mock data instead of actual language memory system
        """
        if self.mock_mode == mock_mode:
            return
        
        self.mock_mode = mock_mode
        
        # Reinitialize components with new mock mode
        self._initialize_components()


# Create a singleton instance
_bridge_instance = None


def get_language_memory_v5_bridge(mock_mode: bool = False) -> LanguageMemoryV5Bridge:
    """
    Get the language memory V5 bridge singleton instance
    
    Args:
        mock_mode: Use mock data instead of actual language memory system
        
    Returns:
        The language memory V5 bridge instance
    """
    global _bridge_instance
    
    if _bridge_instance is None:
        _bridge_instance = LanguageMemoryV5Bridge(mock_mode=mock_mode)
    elif _bridge_instance.is_using_mock_data() != mock_mode:
        _bridge_instance.set_mock_mode(mock_mode)
    
    return _bridge_instance


if __name__ == "__main__":
    # Simple test to verify bridge functionality
    bridge = get_language_memory_v5_bridge()
    
    print("\n" + "="*80)
    print("LANGUAGE MEMORY V5 BRIDGE TEST")
    print("="*80 + "\n")
    
    print(f"Using mock data: {bridge.is_using_mock_data()}")
    print(f"V5 visualization available: {bridge.is_visualization_available()}")
    
    # Test topic synthesis
    print("\nSynthesizing topic: 'neural networks'...")
    result = bridge.synthesize_topic("neural networks", 3)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("Synthesis successful!")
        
        # Print some results
        if "synthesis_results" in result and "synthesized_memory" in result["synthesis_results"]:
            memory = result["synthesis_results"]["synthesized_memory"]
            print(f"\nCore understanding: {memory.get('core_understanding', '')[:100]}...")
            
            print("\nInsights:")
            for i, insight in enumerate(memory.get("novel_insights", [])[:3]):
                print(f"  - {insight[:100]}...")
    
    # Get stats
    print("\nGetting language memory statistics...")
    stats = bridge.get_stats()
    
    if "error" in stats:
        print(f"Error: {stats['error']}")
    else:
        print("Statistics retrieved successfully!")
        
        # Print some stats
        if "synthesis_stats" in stats:
            print(f"\nSynthesis count: {stats['synthesis_stats'].get('synthesis_count', 0)}")
            print(f"Topics synthesized: {len(stats['synthesis_stats'].get('topics_synthesized', []))}")
        
        if "performance_metrics" in stats:
            print(f"\nAverage synthesis time: {stats['performance_metrics'].get('avg_synthesis_time', 0):.3f} seconds")
            print(f"Cache hits: {stats['performance_metrics'].get('cache_hits', 0)}")
            print(f"Cache misses: {stats['performance_metrics'].get('cache_misses', 0)}")
    
    print("\nBridge test complete!") 