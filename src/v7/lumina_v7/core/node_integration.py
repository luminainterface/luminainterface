"""
LUMINA V7 Node Integration System

This module integrates the database and learning mechanisms with the node consciousness system.
It provides automatic registration, connection, and learning for consciousness nodes.
"""

import os
import logging
import time
import threading
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from pathlib import Path

from .node_consciousness_manager import NodeConsciousnessManager, BaseConsciousnessNode
from .database_manager import DatabaseManager
from .system_integrator import SystemIntegrator
from .database_integration import DatabaseIntegration

# Configure logging
logger = logging.getLogger("lumina_v7.node_integration")

class NodeIntegrationSystem:
    """
    Node Integration System for LUMINA V7 that connects nodes, 
    database, and learning mechanisms.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the node integration system.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        
        # Initialize components
        self.node_manager = NodeConsciousnessManager()
        self.db_manager = DatabaseManager.get_instance()
        self.system_integrator = SystemIntegrator(self.node_manager)
        self.db_integration = DatabaseIntegration(self.node_manager)
        
        # Node registry
        self.registered_nodes = {}
        self.node_types = set()
        
        # System metrics
        self.metrics = {
            "nodes": {
                "active": 0,
                "total": 0,
                "by_type": {}
            },
            "connections": {
                "active": 0,
                "total": 0,
                "message_throughput": 0
            },
            "system": {
                "uptime": 0,
                "cpu_usage": 0,
                "memory_usage": 0
            },
            "learning": {
                "patterns": 0,
                "samples": 0,
                "clusters": 0
            },
            "history": {
                "1m": {},
                "5m": {},
                "15m": {},
                "1h": {}
            }
        }
        
        # Start time for uptime calculation
        self.start_time = time.time()
        
        # Metrics collection
        self.metrics_thread = threading.Thread(target=self._metrics_collector, daemon=True)
        self.metrics_thread.start()
        
        logger.info("Node Integration System initialized")
    
    def register_node(self, node: BaseConsciousnessNode, node_type: Optional[str] = None) -> bool:
        """
        Register a node with the integration system.
        
        Args:
            node: The node to register
            node_type: Type of node (optional, will use node.node_type if not specified)
            
        Returns:
            bool: True if registration was successful
        """
        # Register with node manager
        if not self.system_integrator.register_node(node, node_type):
            return False
        
        # Register with database integration if it's a memory or learning node
        actual_type = node_type or node.node_type
        if actual_type in ['memory', 'auto_learning']:
            self.db_integration.register_memory_node(node)
        
        # Track in local registry
        self.registered_nodes[node.node_id] = {
            "node": node,
            "type": actual_type,
            "registered_at": time.time()
        }
        self.node_types.add(actual_type)
        
        # Store registration in database for persistence
        node_data = {
            "node_id": node.node_id,
            "node_type": actual_type,
            "status": "registered",
            "registration_time": time.time()
        }
        
        # Persist state to database
        self._store_node_data(node.node_id, "registration", node_data)
        
        # Update metrics
        self.metrics["nodes"]["total"] = len(self.registered_nodes)
        if actual_type not in self.metrics["nodes"]["by_type"]:
            self.metrics["nodes"]["by_type"][actual_type] = 0
        self.metrics["nodes"]["by_type"][actual_type] += 1
        
        logger.info(f"Node registered with integration system: {node.node_id} (type: {actual_type})")
        return True
    
    def apply_integration_strategy(self, strategy_name: str = "balanced") -> bool:
        """
        Apply an integration strategy to connect and organize nodes.
        
        Args:
            strategy_name: Name of the strategy to apply
            
        Returns:
            bool: True if strategy was applied successfully
        """
        # Apply strategy with system integrator
        success = self.system_integrator.apply_integration_strategy(strategy_name)
        
        if success:
            # Get integration status
            status = self.system_integrator.get_integration_status()
            self.metrics["connections"]["total"] = status["connection_count"]
            
            # Persist configuration to database
            self._store_system_data("integration_strategy", {
                "name": strategy_name,
                "status": status,
                "applied_at": time.time()
            })
            
            logger.info(f"Applied integration strategy: {strategy_name}")
            
        return success
    
    def _store_node_data(self, node_id: str, key: str, data: Any):
        """Store node data in database."""
        try:
            # Convert data to JSON
            if isinstance(data, dict):
                data_blob = json.dumps(data).encode()
            else:
                data_blob = str(data).encode()
            
            # Store in database
            self.db_manager.execute_query('''
            INSERT OR REPLACE INTO node_data (node_id, key, value, data_type)
            VALUES (?, ?, ?, ?)
            ''', (node_id, key, data_blob, type(data).__name__), fetch_mode='none')
        except Exception as e:
            logger.error(f"Error storing node data: {str(e)}")
    
    def _store_system_data(self, key: str, data: Any):
        """Store system data in database."""
        self._store_node_data("system", key, data)
    
    def _metrics_collector(self):
        """Background thread for collecting system metrics."""
        last_minute = time.time()
        last_five_minute = time.time()
        last_hour = time.time()
        
        while True:
            try:
                # Sleep to avoid excessive processing
                time.sleep(10)
                current_time = time.time()
                
                # Update basic metrics
                self.metrics["system"]["uptime"] = current_time - self.start_time
                
                # Update node metrics
                active_nodes = 0
                for node_id, info in self.registered_nodes.items():
                    if self.node_manager.is_node_active(node_id):
                        active_nodes += 1
                
                self.metrics["nodes"]["active"] = active_nodes
                
                # Update connection metrics
                status = self.system_integrator.get_integration_status()
                self.metrics["connections"]["active"] = status["connection_count"]
                
                # Update learning metrics
                learning_stats = self.db_integration.get_learning_statistics()
                self.metrics["learning"]["clusters"] = learning_stats.get("clusters", {}).get("count", 0)
                
                pattern_count = 0
                for pattern_type, stats in learning_stats.get("patterns", {}).items():
                    pattern_count += stats.get("count", 0)
                self.metrics["learning"]["patterns"] = pattern_count
                self.metrics["learning"]["samples"] = learning_stats.get("total_samples", 0)
                
                # Calculate time period metrics
                if current_time - last_minute >= 60:
                    self._calculate_time_period_metrics("1m")
                    last_minute = current_time
                
                if current_time - last_five_minute >= 300:
                    self._calculate_time_period_metrics("5m")
                    self._calculate_time_period_metrics("15m")
                    last_five_minute = current_time
                
                if current_time - last_hour >= 3600:
                    self._calculate_time_period_metrics("1h")
                    last_hour = current_time
                
                # Persist metrics to database periodically
                self._store_system_data("metrics", self.metrics)
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {str(e)}")
    
    def _calculate_time_period_metrics(self, period: str):
        """Calculate metrics for a specific time period."""
        # Take snapshots of current metrics for trend analysis
        self.metrics["history"][period] = {
            "timestamp": time.time(),
            "nodes_active": self.metrics["nodes"]["active"],
            "connections_active": self.metrics["connections"]["active"],
            "learning_patterns": self.metrics["learning"]["patterns"],
            "learning_samples": self.metrics["learning"]["samples"]
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            Dictionary with system status
        """
        # Get database statistics
        db_stats = self.db_manager.get_system_statistics()
        
        # Get node manager status
        node_status = self.node_manager.get_system_status()
        
        # Get integration status
        integration_status = self.system_integrator.get_integration_status()
        
        # Combine into comprehensive status
        status = {
            "system_info": {
                "uptime": self.metrics["system"]["uptime"],
                "uptime_hours": round(self.metrics["system"]["uptime"] / 3600, 2),
                "database_size_mb": db_stats.get("database_size_mb", 0),
                "node_types": list(self.node_types)
            },
            "nodes": {
                "total": self.metrics["nodes"]["total"],
                "active": self.metrics["nodes"]["active"],
                "by_type": self.metrics["nodes"]["by_type"],
                "details": node_status["node_statuses"]
            },
            "connections": {
                "total": integration_status["connection_count"],
                "integration_level": integration_status["integration_level"]
            },
            "learning": {
                "patterns": self.metrics["learning"]["patterns"],
                "samples": self.metrics["learning"]["samples"],
                "clusters": self.metrics["learning"]["clusters"]
            },
            "trends": {
                # Include delta calculations for showing trends
                "5m_delta": {
                    "nodes_active": self.metrics["nodes"]["active"] - 
                        self.metrics["history"].get("5m", {}).get("nodes_active", 0),
                    "patterns": self.metrics["learning"]["patterns"] - 
                        self.metrics["history"].get("5m", {}).get("learning_patterns", 0)
                },
                "1h_delta": {
                    "nodes_active": self.metrics["nodes"]["active"] - 
                        self.metrics["history"].get("1h", {}).get("nodes_active", 0),
                    "patterns": self.metrics["learning"]["patterns"] - 
                        self.metrics["history"].get("1h", {}).get("learning_patterns", 0)
                }
            }
        }
        
        return status
    
    def optimize_integration(self) -> Dict[str, Any]:
        """
        Optimize integration based on current system state.
        
        Returns:
            Dictionary with optimization results
        """
        # Get current status
        status = self.get_system_status()
        
        # Determine optimal strategy based on active nodes
        node_count = status["nodes"]["active"]
        node_types = status["system_info"]["node_types"]
        
        if "monday" in node_types and node_count >= 5:
            strategy = "monday_centered"
        elif "language" in node_types and "memory" in node_types:
            strategy = "language_focused"
        elif node_count <= 3:
            strategy = "simplified"
        else:
            strategy = "balanced"
        
        # Apply the determined strategy
        success = self.apply_integration_strategy(strategy)
        
        result = {
            "strategy_applied": strategy,
            "success": success,
            "timestamp": time.time(),
            "node_count": node_count,
            "active_types": node_types
        }
        
        # Store optimization result
        self._store_system_data("optimization_result", result)
        
        return result
    
    def shutdown(self):
        """Shutdown the integration system."""
        try:
            # Close database integration
            self.db_integration.close()
            
            # Stop system integrator
            self.system_integrator.shutdown()
            
            # Stop node manager
            self.node_manager.stop()
            
            # Update final metrics and persist
            final_metrics = self.metrics
            final_metrics["system"]["shutdown_time"] = time.time()
            final_metrics["system"]["total_uptime"] = \
                final_metrics["system"]["shutdown_time"] - self.start_time
            
            self._store_system_data("final_metrics", final_metrics)
            
            # Close database manager
            self.db_manager.close()
            
            logger.info("Node Integration System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            
    def __del__(self):
        """Ensure shutdown on destruction."""
        self.shutdown() 