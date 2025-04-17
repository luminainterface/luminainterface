#!/usr/bin/env python3
"""
Memory API Socket Provider

This module provides a socket-based interface for the Memory API, 
allowing it to be used by the V5 Fractal Echo Visualization system.
"""

import os
import sys
import json
import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path
from queue import Queue

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/memory_api_socket.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("memory_api_socket")

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Import required components
try:
    from src.v5.node_socket import NodeSocket
    from src.memory_api import MemoryAPI
except ImportError as e:
    logger.error(f"Failed to import required components: {str(e)}")
    raise

# Try to import connection discovery service
try:
    from connection_discovery import register_node
    HAS_DISCOVERY = True
    logger.info("Successfully imported connection discovery service")
except ImportError:
    HAS_DISCOVERY = False
    logger.warning("Connection discovery service not available, using limited discovery")
    
    # Define a dummy register_node function
    def register_node(node, **kwargs):
        logger.warning(f"Cannot register node: {node.plugin_id}")
        return None


class MemoryAPISocketProvider:
    """
    Socket provider for Memory API integration with V5 visualization system
    
    This class connects the Memory API to the V5 socket system, allowing
    memory operations to be performed through the socket interface.
    """
    
    def __init__(self, plugin_id="memory_api_socket"):
        """
        Initialize the memory API socket provider
        
        Args:
            plugin_id: Unique identifier for this plugin
        """
        self.plugin_id = plugin_id
        self.socket = NodeSocket(plugin_id, "service")
        
        # Initialize Memory API
        try:
            self.api = MemoryAPI()
            logger.info("Successfully initialized Memory API")
        except Exception as e:
            logger.error(f"Failed to initialize Memory API: {str(e)}")
            self.api = None
        
        # Register message handlers
        self.socket.register_message_handler("synthesize_topic", self._handle_synthesize_request)
        self.socket.register_message_handler("get_stats", self._handle_stats_request)
        self.socket.register_message_handler("store_memory", self._handle_store_request)
        self.socket.register_message_handler("retrieve_memories", self._handle_retrieve_request)
        self.socket.register_message_handler("get_topics", self._handle_topics_request)
        
        logger.info(f"Registered message handlers for {plugin_id}")
        
        # Register with discovery service if available
        if HAS_DISCOVERY:
            self.client = register_node(self)
            logger.info(f"Registered with discovery service: {plugin_id}")
        else:
            self.client = None
            logger.warning(f"Running without discovery service: {plugin_id}")
        
        # Standard plugin interface attributes
        self.plugin_type = "v5_plugin"
        self.plugin_subtype = "memory_api"
        self.api_version = "v5.0"
        self.ui_requirements = ["memory_synthesis", "topic_browser"]
    
    def _handle_synthesize_request(self, message):
        """
        Handle request to synthesize a topic
        
        Args:
            message: Message containing the request
        """
        if not self.api:
            self._send_error_response(message, "Memory API not initialized")
            return
        
        try:
            content = message.get("content", {})
            topic = content.get("topic")
            depth = content.get("depth", 3)
            
            if not topic:
                self._send_error_response(message, "Missing topic parameter")
                return
            
            logger.info(f"Synthesizing topic: {topic}, depth: {depth}")
            
            # Synthesize topic using the memory API
            result = self.api.synthesize_topic(topic, depth)
            
            # Format the result for visualization
            viz_data = self._prepare_visualization_data(result)
            
            # Send the response
            self._send_response(message, viz_data)
        except Exception as e:
            logger.error(f"Error synthesizing topic: {str(e)}")
            self._send_error_response(message, str(e))
    
    def _handle_stats_request(self, message):
        """
        Handle request to get memory statistics
        
        Args:
            message: Message containing the request
        """
        if not self.api:
            self._send_error_response(message, "Memory API not initialized")
            return
        
        try:
            logger.info("Getting memory statistics")
            
            # Get stats using the memory API
            result = self.api.get_memory_stats()
            
            # Send the response
            self._send_response(message, result)
        except Exception as e:
            logger.error(f"Error getting memory statistics: {str(e)}")
            self._send_error_response(message, str(e))
    
    def _handle_store_request(self, message):
        """
        Handle request to store a memory
        
        Args:
            message: Message containing the request
        """
        if not self.api:
            self._send_error_response(message, "Memory API not initialized")
            return
        
        try:
            content = message.get("content", {})
            memory_content = content.get("content")
            metadata = content.get("metadata", {})
            
            if not memory_content:
                self._send_error_response(message, "Missing content parameter")
                return
            
            logger.info(f"Storing memory: {memory_content[:50]}...")
            
            # Store memory using the memory API
            result = self.api.store_external_memory(memory_content, metadata)
            
            # Send the response
            self._send_response(message, result)
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            self._send_error_response(message, str(e))
    
    def _handle_retrieve_request(self, message):
        """
        Handle request to retrieve memories
        
        Args:
            message: Message containing the request
        """
        if not self.api:
            self._send_error_response(message, "Memory API not initialized")
            return
        
        try:
            content = message.get("content", {})
            query = content.get("query")
            max_results = content.get("max_results", 5)
            
            if not query:
                self._send_error_response(message, "Missing query parameter")
                return
            
            logger.info(f"Retrieving memories for query: {query}, max results: {max_results}")
            
            # Retrieve memories using the memory API
            result = self.api.retrieve_relevant_memories(query, max_results)
            
            # Send the response
            self._send_response(message, result)
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}")
            self._send_error_response(message, str(e))
    
    def _handle_topics_request(self, message):
        """
        Handle request to get available topics
        
        Args:
            message: Message containing the request
        """
        if not self.api:
            self._send_error_response(message, "Memory API not initialized")
            return
        
        try:
            content = message.get("content", {})
            limit = content.get("limit", 20)
            
            logger.info(f"Getting topics, limit: {limit}")
            
            # Get topics using the memory API
            result = self.api.get_topics(limit)
            
            # Send the response
            self._send_response(message, result)
        except Exception as e:
            logger.error(f"Error getting topics: {str(e)}")
            self._send_error_response(message, str(e))
    
    def _prepare_visualization_data(self, synthesis_result):
        """
        Transform synthesis result to visualization-friendly format
        
        Args:
            synthesis_result: Result from memory API
            
        Returns:
            Visualization-ready data
        """
        if synthesis_result.get("status") != "success":
            return synthesis_result
        
        # Extract relevant data for visualization
        synthesis = synthesis_result.get("synthesis_results", {})
        memory = synthesis.get("synthesized_memory", {})
        
        # Create visualization-ready format
        visualization_data = {
            "topic": memory.get("topics", ["unknown"])[0],
            "core_understanding": memory.get("core_understanding", ""),
            "insights": memory.get("novel_insights", []),
            "related_topics": synthesis.get("related_topics", []),
            
            # Network visualization data
            "network": {
                "nodes": [
                    # Main topic node
                    {"id": "main_topic", "label": memory.get("topics", ["unknown"])[0], "group": "topic", "size": 30}
                ],
                "edges": []
            },
            
            # Fractal visualization data
            "fractal_data": {
                "pattern_style": "neural",
                "fractal_depth": 4,
                "metrics": {
                    "fractal_dimension": 1.62,
                    "complexity_index": 85,
                    "pattern_coherence": 92
                }
            }
        }
        
        # Add related topics as nodes
        for i, topic in enumerate(synthesis.get("related_topics", [])):
            topic_name = topic.get("topic", f"related_{i}")
            relevance = topic.get("relevance", 0.5)
            
            # Add node
            visualization_data["network"]["nodes"].append({
                "id": f"topic_{i}",
                "label": topic_name,
                "group": "related_topic",
                "size": 15 + (relevance * 10)
            })
            
            # Add edge connecting to main topic
            visualization_data["network"]["edges"].append({
                "from": "main_topic",
                "to": f"topic_{i}",
                "value": relevance,
                "title": f"Relevance: {relevance:.2f}"
            })
        
        # Add memory insights as nodes
        for i, insight in enumerate(memory.get("novel_insights", [])):
            # Add node for insight
            insight_id = f"insight_{i}"
            visualization_data["network"]["nodes"].append({
                "id": insight_id,
                "label": f"Insight {i+1}",
                "group": "memory",
                "size": 12
            })
            
            # Add edge connecting insight to main topic
            visualization_data["network"]["edges"].append({
                "from": "main_topic",
                "to": insight_id,
                "value": 0.7,
                "title": f"Insight {i+1}"
            })
        
        return visualization_data
    
    def _send_response(self, request_message, result):
        """
        Send response to a request
        
        Args:
            request_message: Original request message
            result: Result data
        """
        response = {
            "type": "api_response",
            "request_id": request_message.get("request_id"),
            "data": result
        }
        
        self.socket.send_message(response)
    
    def _send_error_response(self, request_message, error):
        """
        Send error response
        
        Args:
            request_message: Original request message
            error: Error message
        """
        response = {
            "type": "api_response",
            "request_id": request_message.get("request_id"),
            "status": "error",
            "error": error
        }
        
        self.socket.send_message(response)
    
    def get_socket_descriptor(self):
        """
        Return socket descriptor for frontend integration
        
        Returns:
            Socket descriptor dictionary
        """
        return {
            "plugin_id": self.plugin_id,
            "message_types": [
                "synthesize_topic", 
                "get_stats", 
                "store_memory",
                "retrieve_memories", 
                "get_topics",
                "api_response"
            ],
            "data_format": "json",
            "subscription_mode": "dual",  # Both push and request-response
            "ui_components": self.ui_requirements
        }


def main():
    """Main function for standalone operation"""
    provider = MemoryAPISocketProvider()
    
    # Simple test to verify functionality
    if provider.api:
        print(f"Memory API Socket Provider initialized: {provider.plugin_id}")
        print("Plugin ready to handle socket messages")
        
        # Keep running to handle messages
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                print("Shutting down...")
                break
    else:
        print("Memory API initialization failed")


if __name__ == "__main__":
    main() 