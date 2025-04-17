"""
Language Memory Integration Plugin for V5 Visualization System

This module provides the LanguageMemoryIntegrationPlugin that connects to 
the language memory synthesis system and prepares data for visualization.
"""

import time
import logging
import uuid
import random
import threading
from collections import defaultdict

from .node_socket import NodeSocket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import the connection discovery service
try:
    from connection_discovery import ConnectionDiscovery, register_node
    HAS_DISCOVERY = True
except ImportError:
    logger.warning("ConnectionDiscovery not available. Limited plugin discovery will be used.")
    HAS_DISCOVERY = False
    # Define a dummy register_node function for when discovery is not available
    def register_node(node, **kwargs):
        logger.warning(f"Cannot register node: {node.node_id}")
        return None


class LanguageMemoryIntegrationPlugin:
    """Plugin for integrating V5 visualization with Language Memory System"""
    
    def __init__(self, plugin_id="language_memory_integration", language_memory_synthesis=None):
        """
        Initialize the language memory integration plugin
        
        Args:
            plugin_id: Unique identifier for this plugin
            language_memory_synthesis: Optional LanguageMemorySynthesisIntegration instance
        """
        self.plugin_id = plugin_id
        self.socket = NodeSocket(plugin_id, "integration")
        self.mock_mode = False
        self.language_memory_synthesis = language_memory_synthesis
        self.cache = {}
        
        # Set up message handlers
        self.socket.register_message_handler("process_topic", self._handle_topic_request)
        
        # Register with discovery service if available
        if HAS_DISCOVERY:
            self.client = register_node(self)
            logger.info(f"Registered plugin with discovery service: {plugin_id}")
        else:
            self.client = None
            logger.warning(f"Running without discovery service: {plugin_id}")
        
        # Standard plugin interface attributes
        self.plugin_type = "v5_plugin"
        self.plugin_subtype = "integration"
        self.api_version = "v5.0"
        self.ui_requirements = ["language_stats", "memory_visualization", "pattern_view"]
        
        # Initialize language memory synthesis integration if not provided
        if self.language_memory_synthesis is None:
            try:
                from src.language_memory_synthesis_integration import LanguageMemorySynthesisIntegration
                self.language_memory_synthesis = LanguageMemorySynthesisIntegration()
                self.mock_mode = False
                logger.info("Successfully connected to Language Memory Synthesis Integration")
            except ImportError as e:
                logger.error(f"Failed to import Language Memory Synthesis Integration: {str(e)}")
                self.mock_mode = True
                logger.warning("Running in mock mode")
            
        logger.info(f"Language Memory Integration Plugin initialized: {plugin_id}")
    
    def _handle_topic_request(self, message):
        """
        Handle a request to process a language topic
        
        Args:
            message: Message containing the request
        """
        try:
            content = message.get("content", {})
            topic = content.get("topic")
            depth = content.get("depth", 3)
            request_id = message.get("request_id")
            
            if not topic:
                logger.warning("Received topic request without topic")
                if request_id:
                    self.socket.send_message({
                        "type": "language_memory_update",
                        "request_id": request_id,
                        "error": "No topic provided"
                    })
                return
                
            # Process the topic
            result = self.process_language_data(topic, depth)
            
            # Send the result
            if request_id:
                result["request_id"] = request_id
                self.socket.send_message({
                    "type": "language_memory_update",
                    "request_id": request_id,
                    "data": result
                })
                
            logger.debug(f"Processed language data for topic: {topic}, depth: {depth}")
            
        except Exception as e:
            logger.error(f"Error handling topic request: {str(e)}")
            if message.get("request_id"):
                self.socket.send_message({
                    "type": "language_memory_update",
                    "request_id": message.get("request_id"),
                    "error": str(e)
                })
    
    def process_language_data(self, topic, depth=3):
        """
        Process language data for a specific topic and prepare for visualization
        
        Args:
            topic: The topic to synthesize
            depth: How deep to search for related memories (1-5)
            
        Returns:
            Visualization-ready data structure
        """
        # Check cache first
        cache_key = f"{topic}_{depth}"
        if cache_key in self.cache:
            logger.info(f"Using cached data for topic: {topic}, depth: {depth}")
            return self.cache[cache_key]
            
        try:
            start_time = time.time()
            
            if self.mock_mode or not self.language_memory_synthesis:
                # Generate mock data in mock mode
                result = self._generate_mock_data(topic, depth)
                processing_time = time.time() - start_time
                
                # Add processing time to the mock data
                if "metrics" in result:
                    result["metrics"]["avg_synthesis_time"] = round(processing_time, 3)
                
                logger.info(f"Generated mock data for topic: {topic}, depth: {depth} in {processing_time:.3f}s")
                
                # Cache the result
                self.cache[cache_key] = result
                return result
            
            # Synthesize topic using the language memory system
            synthesis_response = self.language_memory_synthesis.synthesize_topic(topic, depth)
            
            # Extract performance metrics
            stats = self.language_memory_synthesis.get_stats()
            
            # Transform to visualization-friendly format
            visualization_data = self._prepare_visualization_data(
                synthesis_response, 
                stats
            )
            
            # Extract standard statistics for logging
            processing_time = time.time() - start_time
            logger.info(f"Topics Synthesized: {stats['synthesis_stats']['topics_synthesized']}")
            logger.info(f"Average Synthesis Time: {stats['performance_metrics']['avg_synthesis_time']} seconds")
            logger.info(f"Total processing time: {processing_time:.3f} seconds")
            
            # Add request processing time
            if "metrics" in visualization_data:
                visualization_data["metrics"]["request_time"] = round(processing_time, 3)
            
            # Cache the result
            self.cache[cache_key] = visualization_data
            return visualization_data
            
        except Exception as e:
            logger.error(f"Error processing language data: {str(e)}")
            # Return minimal error data
            return {
                "error": str(e),
                "topic": topic,
                "depth": depth,
                "timestamp": time.time()
            }
    
    def _prepare_visualization_data(self, synthesis_response, stats):
        """
        Transform synthesis results into visualization-friendly format
        
        Args:
            synthesis_response: Response from the language memory synthesis system
            stats: Stats from the language memory synthesis system
            
        Returns:
            Visualization-ready data structure
        """
        try:
            # Extract synthesis results
            synthesis_results = synthesis_response.get("synthesis_results", {})
            synthesized_memory = synthesis_results.get("synthesized_memory", {})
            related_topics = synthesis_results.get("related_topics", [])
            
            # Component contributions
            component_contributions = synthesis_response.get("component_contributions", {})
            
            # Prepare network visualization data
            nodes = []
            edges = []
            
            # Create central node for the main topic
            main_topic = synthesis_results.get("topic", "unknown")
            if isinstance(main_topic, list) and main_topic:
                main_topic = main_topic[0]
                
            if not isinstance(main_topic, str):
                main_topic = str(main_topic)
                
            nodes.append({
                "id": main_topic,
                "label": main_topic,
                "group": "topic",
                "size": 30
            })
            
            # Add related topics as nodes
            for i, topic in enumerate(related_topics):
                if isinstance(topic, dict):
                    topic_name = topic.get("topic", f"related_{i}")
                    relevance = topic.get("relevance", 0.5)
                else:
                    topic_name = f"related_{i}"
                    relevance = 0.5
                
                nodes.append({
                    "id": topic_name,
                    "label": topic_name,
                    "group": "related_topic",
                    "size": 15 + (relevance * 10)
                })
                
                # Add edge connecting to main topic
                edges.append({
                    "from": main_topic,
                    "to": topic_name,
                    "value": relevance,
                    "title": f"Relevance: {relevance:.2f}"
                })
            
            # Add component contributions as nodes
            for component_name, data in component_contributions.items():
                if isinstance(data, dict) and "error" in data:
                    continue
                    
                # Add component node
                nodes.append({
                    "id": component_name,
                    "label": component_name,
                    "group": "component",
                    "size": 20
                })
                
                # Connect to main topic
                edges.append({
                    "from": main_topic,
                    "to": component_name,
                    "value": 1.0,
                    "title": "Contributes to"
                })
            
            # Create fractal visualization data
            fractal_data = {
                "pattern_style": "neural",
                "fractal_depth": 4,
                "metrics": {
                    "fractal_dimension": 1.6 + (len(nodes) / 50),  # Approximate based on network complexity
                    "complexity_index": min(98, len(nodes) * 5),
                    "pattern_coherence": max(50, 100 - (len(nodes) * 2))
                }
            }
            
            # Extract memory insights
            memory_insights = []
            if isinstance(synthesized_memory, dict) and "core_insights" in synthesized_memory:
                memory_insights = synthesized_memory["core_insights"]
            
            # Get metrics from stats
            synthesis_stats = stats.get('synthesis_stats', {})
            performance_metrics = stats.get('performance_metrics', {})
            
            # Final visualization data
            visualization_data = {
                "network": {
                    "nodes": nodes,
                    "edges": edges
                },
                "metrics": {
                    "topics_synthesized": len(synthesis_stats.get('topics_synthesized', [])),
                    "avg_synthesis_time": performance_metrics.get('avg_synthesis_time', 0),
                    "cache_hits": performance_metrics.get('cache_hits', 0),
                    "cache_misses": performance_metrics.get('cache_misses', 0)
                },
                "fractal_data": fractal_data,
                "memory_insights": memory_insights
            }
            
            return visualization_data
            
        except Exception as e:
            logger.error(f"Error preparing visualization data: {str(e)}")
            # Return minimal data
            return {
                "network": {
                    "nodes": [],
                    "edges": []
                },
                "metrics": {},
                "fractal_data": {},
                "memory_insights": [],
                "error": str(e)
            }
    
    def _generate_mock_data(self, topic, depth):
        """
        Generate mock language memory data for testing
        
        Args:
            topic: The topic to generate data for
            depth: The depth of the search
            
        Returns:
            Mock visualization data
        """
        # Create mock nodes and edges
        nodes = []
        edges = []
        
        # Main topic node
        nodes.append({
            "id": topic,
            "label": topic,
            "group": "topic",
            "size": 30
        })
        
        # Generate related topics
        related_count = 5 + (depth * 2)
        related_topics = []
        
        for i in range(related_count):
            relevance = random.uniform(0.3, 0.9)
            topic_name = f"related_{topic}_{i}"
            
            nodes.append({
                "id": topic_name,
                "label": topic_name,
                "group": "related_topic",
                "size": 15 + (relevance * 10)
            })
            
            edges.append({
                "from": topic,
                "to": topic_name,
                "value": relevance,
                "title": f"Relevance: {relevance:.2f}"
            })
            
            related_topics.append({
                "topic": topic_name,
                "relevance": relevance
            })
        
        # Add some component nodes
        components = ["memory_system", "language_processor", "consciousness_module"]
        
        for component in components:
            nodes.append({
                "id": component,
                "label": component,
                "group": "component",
                "size": 20
            })
            
            edges.append({
                "from": topic,
                "to": component,
                "value": 1.0,
                "title": "Contributes to"
            })
        
        # Generate some insights
        insights = [
            f"The concept of {topic} involves pattern recognition",
            f"{topic} demonstrates emergent properties",
            f"Information about {topic} is stored across multiple memory components",
            f"Neural patterns related to {topic} show high coherence",
            f"Processing {topic} activates both concrete and abstract reasoning pathways"
        ]
        
        # Mock fractal data
        fractal_data = {
            "pattern_style": "neural",
            "fractal_depth": 4,
            "metrics": {
                "fractal_dimension": 1.68,
                "complexity_index": 78,
                "pattern_coherence": 85
            }
        }
        
        # Create mock metrics
        metrics = {
            "topics_synthesized": random.randint(10, 50),
            "avg_synthesis_time": round(random.uniform(0.1, 1.5), 3),
            "cache_hits": random.randint(0, 20),
            "cache_misses": random.randint(1, 10)
        }
        
        # Final mock data
        mock_data = {
            "network": {
                "nodes": nodes,
                "edges": edges
            },
            "metrics": metrics,
            "fractal_data": fractal_data,
            "memory_insights": insights,
            "mock": True
        }
        
        return mock_data
    
    def get_stats(self):
        """
        Get statistics about the language memory system
        
        Returns:
            Dictionary containing statistics
        """
        if self.mock_mode or not self.language_memory_synthesis:
            # Generate mock stats in mock mode
            return {
                "synthesis_stats": {
                    "topics_synthesized": random.randint(10, 50),
                    "patterns_identified": random.randint(20, 100),
                    "memory_connections": random.randint(50, 200)
                },
                "performance_metrics": {
                    "avg_synthesis_time": round(random.uniform(0.1, 1.5), 3),
                    "cache_hits": random.randint(0, 20),
                    "cache_misses": random.randint(1, 10),
                    "total_requests": random.randint(10, 50)
                }
            }
            
        try:
            # Get real stats from the language memory synthesis system
            return self.language_memory_synthesis.get_stats()
        except Exception as e:
            logger.error(f"Error getting language memory stats: {str(e)}")
            # Return minimal stats on error
            return {
                "synthesis_stats": {},
                "performance_metrics": {},
                "error": str(e)
            }
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()
        logger.info("Cleared language memory integration cache")
    
    def get_socket_descriptor(self):
        """
        Return socket descriptor for frontend integration
        
        Returns:
            Socket descriptor dictionary
        """
        return {
            "plugin_id": self.plugin_id,
            "message_types": ["language_memory_update", "process_topic"],
            "data_format": "json",
            "subscription_mode": "push",
            "ui_components": self.ui_requirements
        } 