#!/usr/bin/env python3
"""
Dashboard Metrics Plugin for Lumina V7
======================================

Provides system metrics to the Dashboard V7 Bridge via socket connection.
Acts as a server that responds to metrics requests from the dashboard.
"""

import os
import sys
import time
import json
import socket
import logging
import threading
import random
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/dashboard_metrics_plugin_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DashboardMetricsPlugin")

class DashboardMetricsPlugin:
    """
    Plugin that provides system metrics to the dashboard via socket connection.
    """
    
    def __init__(self, port=5678, consciousness_module=None, language_module=None):
        """
        Initialize the metrics plugin
        
        Args:
            port: Port to listen on
            consciousness_module: Reference to consciousness module for metrics
            language_module: Reference to language module for metrics
        """
        self.port = port
        self.consciousness_module = consciousness_module
        self.language_module = language_module
        self.running = False
        self.server_socket = None
        self.connections = []
        self.lock = threading.Lock()
        
        # Set up server
        self.setup_server()
    
    def setup_server(self):
        """Set up the metrics server socket"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(("localhost", self.port))
            self.server_socket.listen(5)
            logger.info(f"Dashboard metrics server listening on port {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to set up metrics server: {e}")
            return False
    
    def get_consciousness_metrics(self):
        """Get metrics from the consciousness module"""
        try:
            if self.consciousness_module and hasattr(self.consciousness_module, 'get_metrics'):
                return self.consciousness_module.get_metrics()
            
            # Return mock data if module not available
            return {
                "consciousness_level": random.uniform(0.3, 0.9),
                "self_awareness": random.uniform(0.2, 0.8),
                "integration_level": random.uniform(0.4, 0.7)
            }
        except Exception as e:
            logger.error(f"Error getting consciousness metrics: {e}")
            return {"consciousness_level": 0.5}
    
    def get_language_metrics(self):
        """Get metrics from the language module"""
        try:
            if self.language_module and hasattr(self.language_module, 'get_metrics'):
                return self.language_module.get_metrics()
            
            # Return mock data if module not available
            return {
                "mistral_activity": random.uniform(0.4, 0.9),
                "token_usage": random.randint(50, 500),
                "response_time": random.uniform(0.1, 2.0)
            }
        except Exception as e:
            logger.error(f"Error getting language metrics: {e}")
            return {"mistral_activity": 0.5}
    
    def get_system_metrics(self):
        """Get system metrics"""
        try:
            import psutil
            
            # Get actual system metrics
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            return {
                "system_usage": cpu_percent / 100.0,
                "memory_usage": memory_percent / 100.0,
                "disk_usage": disk_percent / 100.0
            }
        except Exception as e:
            logger.warning(f"Error getting system metrics: {e}")
            
            # Return mock data on failure
            return {
                "system_usage": random.uniform(0.2, 0.7),
                "memory_usage": random.uniform(0.3, 0.8),
                "disk_usage": random.uniform(0.1, 0.6)
            }
    
    def get_learning_metrics(self):
        """Get learning metrics"""
        try:
            # Try to access learning metrics if available
            # In a real implementation, this would access the actual learning system
            if hasattr(self, 'get_learning_data'):
                return self.get_learning_data()
            
            # Return mock data for now
            return {
                "learning_rate": random.uniform(0.01, 0.3),
                "new_patterns": random.randint(0, 10),
                "recall_accuracy": random.uniform(0.7, 0.95)
            }
        except Exception as e:
            logger.error(f"Error getting learning metrics: {e}")
            return {"learning_rate": 0.1}
    
    def get_all_metrics(self):
        """Get all metrics from different modules"""
        metrics = {}
        descriptions = {}
        
        # Consciousness metrics
        consciousness_metrics = self.get_consciousness_metrics()
        metrics.update(consciousness_metrics)
        descriptions["consciousness_level"] = "Current consciousness integration level"
        
        # Language metrics
        language_metrics = self.get_language_metrics()
        metrics.update(language_metrics)
        descriptions["mistral_activity"] = "Mistral LLM activity level"
        
        # System metrics
        system_metrics = self.get_system_metrics()
        metrics.update(system_metrics)
        descriptions["system_usage"] = f"CPU usage: {metrics.get('system_usage', 0) * 100:.1f}%"
        
        # Learning metrics
        learning_metrics = self.get_learning_metrics()
        metrics.update(learning_metrics)
        descriptions["learning_rate"] = f"Learning rate: {metrics.get('learning_rate', 0):.3f}"
        
        return {
            "metrics": metrics,
            "descriptions": descriptions,
            "timestamp": datetime.now().isoformat()
        }
    
    def handle_client(self, client_socket, address):
        """Handle client connection"""
        logger.info(f"New dashboard connection from {address}")
        
        try:
            while self.running:
                try:
                    # Receive request
                    data = client_socket.recv(1024)
                    if not data:
                        break
                    
                    # Parse request
                    request = json.loads(data.decode('utf-8'))
                    
                    # Handle metrics request
                    if request.get("type") == "metrics_request":
                        metrics_data = self.get_all_metrics()
                        client_socket.sendall(json.dumps(metrics_data).encode('utf-8'))
                    
                except Exception as e:
                    logger.error(f"Error handling client request: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            with self.lock:
                if client_socket in self.connections:
                    self.connections.remove(client_socket)
            client_socket.close()
            logger.info(f"Connection from {address} closed")
    
    def start(self):
        """Start the metrics server"""
        if self.running:
            logger.warning("Metrics server already running")
            return
            
        self.running = True
        
        # Start server thread
        server_thread = threading.Thread(target=self.run_server)
        server_thread.daemon = True
        server_thread.start()
        
        logger.info("Dashboard metrics plugin started")
    
    def run_server(self):
        """Run the server to accept connections"""
        if not self.server_socket:
            if not self.setup_server():
                logger.error("Failed to run server: socket not set up")
                return
        
        logger.info("Metrics server running, waiting for connections")
        
        try:
            while self.running:
                try:
                    # Accept connection with timeout
                    self.server_socket.settimeout(1.0)
                    client_socket, address = self.server_socket.accept()
                    
                    # Add to connections list
                    with self.lock:
                        self.connections.append(client_socket)
                    
                    # Handle in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client, 
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except socket.timeout:
                    # This is expected, just continue
                    continue
                except Exception as e:
                    logger.error(f"Error accepting connection: {e}")
                    time.sleep(1)
        except Exception as e:
            logger.error(f"Error in server thread: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the metrics server"""
        self.running = False
        
        # Close all connections
        with self.lock:
            for conn in self.connections:
                try:
                    conn.close()
                except:
                    pass
            self.connections = []
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
            self.server_socket = None
            
        logger.info("Dashboard metrics plugin stopped")

# Plugin registration for Lumina V7
def register_plugin(plugin_manager, consciousness_module=None, language_module=None):
    """Register the plugin with the Lumina V7 system"""
    plugin = DashboardMetricsPlugin(
        consciousness_module=consciousness_module,
        language_module=language_module
    )
    
    # Register with plugin manager
    plugin_manager.register_plugin(
        "dashboard_metrics",
        plugin,
        {
            "name": "Dashboard Metrics Provider",
            "description": "Provides system metrics to the PyQt5 Dashboard",
            "version": "1.0.0",
            "author": "Lumina Team"
        }
    )
    
    # Start the plugin
    plugin.start()
    
    return plugin 