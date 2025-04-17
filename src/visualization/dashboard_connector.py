#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Connector for LUMINA V7 Dashboard
==========================================

Integrates connection recovery with dashboard panels.
"""

import os
import sys
import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable

# Import connection recovery manager
from src.visualization.connection_recovery_manager import ConnectionRecoveryManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DashboardConnector")

class DashboardConnector:
    """
    Manages connections between dashboard panels and LUMINA V7 system
    
    This class:
    - Maintains a single connection to the LUMINA V7 system
    - Provides connection sharing across multiple panels
    - Handles connection recovery automatically
    - Manages mock mode fallback when connection fails
    - Provides centralized request handling
    """
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls, host="localhost", port=5678, **kwargs):
        """Get or create the singleton instance"""
        if cls._instance is None:
            cls._instance = cls(host, port, **kwargs)
        return cls._instance
    
    def __init__(self, host="localhost", port=5678, 
                auto_reconnect=True, mock_on_failure=True,
                metrics_db_path="data/neural_metrics.db"):
        """
        Initialize the dashboard connector
        
        Args:
            host: Hostname of the LUMINA V7 system
            port: Port number of the LUMINA V7 system
            auto_reconnect: Whether to automatically reconnect when connection is lost
            mock_on_failure: Whether to fall back to mock mode when connection fails
            metrics_db_path: Path to the metrics database
        """
        # Check for singleton pattern
        if DashboardConnector._instance is not None:
            raise RuntimeError("DashboardConnector is a singleton - use get_instance() instead")
        
        # Connection parameters
        self.host = host
        self.port = port
        self.mock_on_failure = mock_on_failure
        self.metrics_db_path = metrics_db_path
        
        # Create connection recovery manager
        self.connection_manager = ConnectionRecoveryManager(
            host=host,
            port=port,
            auto_reconnect=auto_reconnect,
            fallback_to_mock=mock_on_failure
        )
        
        # Register callbacks
        self.connection_manager.set_callbacks(
            on_connected=self._on_connected,
            on_disconnected=self._on_disconnected,
            on_connection_failed=self._on_connection_failed,
            on_mock_mode=self._on_mock_mode
        )
        
        # Mock mode state
        self.mock_mode = False
        
        # Registered panels
        self.panels = {}
        
        # Initialize connection
        if not self.connection_manager.connect():
            logger.warning("Initial connection failed")
        
        logger.info(f"DashboardConnector initialized: {host}:{port}")
    
    def register_panel(self, panel_id, panel_instance):
        """
        Register a panel with the connector
        
        Args:
            panel_id: Unique identifier for the panel
            panel_instance: Reference to the panel instance
        """
        if panel_id in self.panels:
            logger.warning(f"Panel '{panel_id}' already registered - updating reference")
        
        self.panels[panel_id] = panel_instance
        
        # Set mock mode if already in mock mode
        if self.mock_mode and hasattr(panel_instance, 'set_mock_mode'):
            panel_instance.set_mock_mode(True)
        
        logger.debug(f"Registered panel: {panel_id}")
    
    def unregister_panel(self, panel_id):
        """
        Unregister a panel from the connector
        
        Args:
            panel_id: Unique identifier for the panel
        """
        if panel_id in self.panels:
            del self.panels[panel_id]
            logger.debug(f"Unregistered panel: {panel_id}")
    
    def fetch_metrics(self, metric_type, params=None):
        """
        Fetch metrics from the LUMINA V7 system
        
        Args:
            metric_type: Type of metrics to fetch ('neural', 'language', 'system', etc.)
            params: Additional parameters for the request
            
        Returns:
            Tuple of (success, metrics_data or error_message)
        """
        # Build request
        request = {
            "type": "metrics_request",
            "metric_type": metric_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add additional parameters if provided
        if params:
            request.update(params)
        
        # Send request via connection manager
        success, response = self.connection_manager.send_request(request)
        
        # If in mock mode, use mock data
        if not success or self.mock_mode:
            if self.mock_mode:
                logger.debug(f"Generating mock metrics for '{metric_type}'")
            else:
                logger.warning(f"Failed to fetch metrics for '{metric_type}', using mock data: {response}")
                
            return False, self._generate_mock_metrics(metric_type, params)
        
        return success, response
    
    def send_command(self, command, params=None):
        """
        Send command to the LUMINA V7 system
        
        Args:
            command: Command to send
            params: Additional parameters for the command
            
        Returns:
            Tuple of (success, response_data or error_message)
        """
        # Build request
        request = {
            "type": "command",
            "command": command,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add additional parameters if provided
        if params:
            request.update(params)
        
        # Send request via connection manager
        success, response = self.connection_manager.send_request(request)
        
        # If in mock mode, simulate command response
        if not success or self.mock_mode:
            if self.mock_mode:
                logger.debug(f"Simulating command response for '{command}'")
            else:
                logger.warning(f"Failed to send command '{command}', simulating response: {response}")
                
            return False, self._simulate_command_response(command, params)
        
        return success, response
    
    def get_connection_status(self):
        """
        Get current connection status
        
        Returns:
            Dictionary with connection status information
        """
        status = self.connection_manager.get_status()
        status["mock_mode"] = self.mock_mode
        status["registered_panels"] = list(self.panels.keys())
        
        return status
    
    def set_mock_mode(self, enabled=True):
        """
        Manually set mock mode
        
        Args:
            enabled: Whether mock mode should be enabled
        """
        # Update state
        self.mock_mode = enabled
        
        # Update all registered panels
        for panel_id, panel in self.panels.items():
            if hasattr(panel, 'set_mock_mode'):
                panel.set_mock_mode(enabled)
        
        logger.info(f"Mock mode {'enabled' if enabled else 'disabled'}")
    
    def reconnect(self):
        """Attempt to reconnect to the LUMINA V7 system"""
        if self.connection_manager.connect():
            logger.info("Reconnected successfully")
            
            # Disable mock mode if successful
            if self.mock_mode:
                self.set_mock_mode(False)
                
            return True
        else:
            logger.warning("Reconnection failed")
            return False
    
    def _on_connected(self):
        """Callback when connection is established"""
        logger.info("Connected to LUMINA V7 system")
        
        # Disable mock mode if it was enabled due to connection failure
        if self.mock_mode:
            self.set_mock_mode(False)
    
    def _on_disconnected(self):
        """Callback when connection is lost"""
        logger.warning("Disconnected from LUMINA V7 system")
    
    def _on_connection_failed(self, error):
        """Callback when connection fails"""
        logger.error(f"Connection failed: {error}")
    
    def _on_mock_mode(self):
        """Callback when entering mock mode"""
        logger.warning("Entering mock mode due to connection failure")
        self.set_mock_mode(True)
    
    def _generate_mock_metrics(self, metric_type, params=None):
        """
        Generate mock metrics data for testing
        
        Args:
            metric_type: Type of metrics to generate
            params: Additional parameters
            
        Returns:
            Mock metrics data
        """
        import random
        
        # Common fields
        mock_data = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "mock": True,
            "metric_type": metric_type
        }
        
        # Generate metrics based on type
        if metric_type == "neural":
            mock_data.update({
                "consciousness_level": random.uniform(0.3, 0.9),
                "learning_rate": random.uniform(0.01, 0.1),
                "connection_strength": random.uniform(0.4, 0.95),
                "pattern_formation": random.uniform(0.2, 0.8),
                "node_count": random.randint(1000, 5000),
                "active_nodes": random.randint(500, 2000)
            })
            
        elif metric_type == "language":
            mock_data.update({
                "semantic_understanding": random.uniform(0.5, 0.9),
                "vocabulary_count": random.randint(5000, 20000),
                "conversation_quality": random.uniform(0.6, 0.95),
                "response_time_ms": random.randint(50, 500),
                "memory_utilization": random.uniform(0.3, 0.8),
                "llm_integration_level": random.uniform(0.7, 0.95)
            })
            
        elif metric_type == "system":
            mock_data.update({
                "cpu_usage": random.uniform(10, 80),
                "memory_usage": random.uniform(20, 70),
                "disk_usage": random.uniform(30, 60),
                "gpu_usage": random.uniform(0, 50) if random.random() > 0.3 else 0,
                "process_count": random.randint(10, 30),
                "thread_count": random.randint(50, 200),
                "uptime_seconds": random.randint(60, 100000)
            })
            
        elif metric_type == "dream":
            # Generate a sine wave pattern with some randomness
            import numpy as np
            x = np.linspace(0, 10, 100)
            pattern = np.sin(x) * 0.5 + 0.5
            
            # Add complexity
            for i in range(1, int(5 * random.random()) + 1):
                pattern += np.sin(x * (i+1) * random.uniform(0.8, 1.2)) * (0.5 / (i+1)) * random.uniform(0.8, 1.2)
            
            # Normalize to 0-1 range
            pattern = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern))
            
            mock_data.update({
                "dream_active": random.random() > 0.3,
                "creativity_index": random.uniform(0.2, 0.9),
                "pattern_count": random.randint(1, 100),
                "current_pattern": pattern.tolist()
            })
            
        else:
            # Generic metrics
            mock_data.update({
                "value": random.random(),
                "count": random.randint(1, 100),
                "status": random.choice(["active", "idle", "learning", "processing"])
            })
        
        return mock_data
    
    def _simulate_command_response(self, command, params=None):
        """
        Simulate command response in mock mode
        
        Args:
            command: The command to simulate
            params: Command parameters
            
        Returns:
            Simulated command response
        """
        # Generic success response
        response = {
            "success": True,
            "command": command,
            "timestamp": datetime.now().isoformat(),
            "mock": True
        }
        
        # Handle specific commands
        if command == "start_dream_mode":
            response["status"] = "dream_mode_started"
            response["message"] = "Dream mode activated successfully"
            
        elif command == "stop_dream_mode":
            response["status"] = "dream_mode_stopped"
            response["message"] = "Dream mode deactivated successfully"
            
        elif command == "set_weights":
            nn_weight = params.get("nn_weight", 0.5) if params else 0.5
            llm_weight = params.get("llm_weight", 0.5) if params else 0.5
            
            response["status"] = "weights_updated"
            response["message"] = f"Weights updated: NN={nn_weight}, LLM={llm_weight}"
            response["nn_weight"] = nn_weight
            response["llm_weight"] = llm_weight
            
        else:
            # Generic command response
            response["status"] = "executed"
            response["message"] = f"Command '{command}' simulated successfully"
        
        return response
    
    def cleanup(self):
        """Clean up resources when connector is no longer needed"""
        # Clean up connection manager
        self.connection_manager.cleanup()
        
        # Clear panel references
        self.panels.clear()
        
        logger.info("DashboardConnector cleaned up")


# For testing the connector
if __name__ == "__main__":
    # Create connector
    connector = DashboardConnector.get_instance(host="localhost", port=5678)
    
    # Test connection
    print(f"Connection status: {connector.get_connection_status()}")
    
    # Test fetching metrics
    success, metrics = connector.fetch_metrics("system")
    print(f"Metrics fetch success: {success}")
    if success:
        print(f"System metrics: {metrics}")
    
    # Test sending command
    success, response = connector.send_command("set_weights", {"nn_weight": 0.7, "llm_weight": 0.3})
    print(f"Command response: {response}")
    
    # Clean up
    connector.cleanup() 