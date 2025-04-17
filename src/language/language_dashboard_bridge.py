#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Language Dashboard Bridge
========================

This module connects the Enhanced Language System with the LUMINA V7 Dashboard panels.
It provides bidirectional communication between language components and dashboard visualization.
"""

import os
import sys
import time
import logging
import threading
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("language-dashboard-bridge")

class LanguageDashboardBridge:
    """
    Bridge between the Enhanced Language System and LUMINA V7 Dashboard
    
    This class provides:
    1. Real-time data flow from language components to dashboard panels
    2. Configuration updates from dashboard to language components
    3. Synchronized metrics collection and visualization
    4. Weight parameter synchronization across components
    """
    
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls, config=None):
        """Get singleton instance of the bridge"""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance
    
    def __init__(self, config=None):
        """
        Initialize the Language Dashboard Bridge
        
        Args:
            config: Configuration dictionary
        """
        if LanguageDashboardBridge._instance is not None:
            raise RuntimeError("Use get_instance() to get the singleton instance")
            
        self.config = config or {}
        self.db_path = self.config.get("db_path", "data/neural_metrics.db")
        self.mock_mode = self.config.get("mock_mode", False)
        self.initialized = False
        self.running = False
        
        # Track component references
        self.central_language_node = None
        self.language_memory = None
        self.neural_linguistic_processor = None
        self.conscious_mirror = None
        self.pattern_analyzer = None
        self.neural_flex_bridge = None
        
        # Track dashboard panel references
        self.language_panel = None
        self.neural_panel = None
        
        # Data queues for async communication
        self.language_data_queue = queue.Queue()
        self.neural_data_queue = queue.Queue()
        self.command_queue = queue.Queue()
        
        # Metrics storage
        self.metrics = {
            "language_metrics": [],
            "neural_metrics": [],
            "integration_metrics": []
        }
        
        # Initialize database
        self._initialize_database()
        
        # Set as initialized
        self.initialized = True
        logger.info("Language Dashboard Bridge initialized")
        
    def _initialize_database(self):
        """Initialize the metrics database"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS language_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                value REAL,
                description TEXT
            )
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS neural_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                value REAL,
                description TEXT
            )
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS integration_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT,
                metric_value REAL,
                details TEXT
            )
            """)
            
            conn.commit()
            conn.close()
            logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def connect_language_components(self, components=None):
        """
        Connect to language system components
        
        Args:
            components: Dictionary of component instances
        """
        if not components:
            # Try to import and get components
            try:
                from language.central_language_node import CentralLanguageNode
                
                # Create or get central node
                self.central_language_node = components.get("central_language_node") if components else None
                
                if self.central_language_node is None:
                    # Create a new central node instance
                    self.central_language_node = CentralLanguageNode(
                        data_dir=self.config.get("data_dir", "data"),
                        llm_weight=self.config.get("llm_weight", 0.5),
                        nn_weight=self.config.get("nn_weight", 0.5)
                    )
                    logger.info("Created new CentralLanguageNode instance")
                    
                # Get references to individual components
                if hasattr(self.central_language_node, "language_memory"):
                    self.language_memory = self.central_language_node.language_memory
                
                if hasattr(self.central_language_node, "neural_processor"):
                    self.neural_linguistic_processor = self.central_language_node.neural_processor
                
                if hasattr(self.central_language_node, "conscious_mirror"):
                    self.conscious_mirror = self.central_language_node.conscious_mirror
                
                if hasattr(self.central_language_node, "pattern_analyzer"):
                    self.pattern_analyzer = self.central_language_node.pattern_analyzer
                
                if hasattr(self.central_language_node, "neural_flex_bridge"):
                    self.neural_flex_bridge = self.central_language_node.neural_flex_bridge
                
                logger.info("Connected to language components through CentralLanguageNode")
                return True
                
            except ImportError as e:
                logger.warning(f"Could not import language components: {e}")
                if not self.mock_mode:
                    logger.warning("Enabling mock mode due to import failure")
                    self.mock_mode = True
        else:
            # Use provided component instances
            self.central_language_node = components.get("central_language_node")
            self.language_memory = components.get("language_memory")
            self.neural_linguistic_processor = components.get("neural_linguistic_processor")
            self.conscious_mirror = components.get("conscious_mirror")
            self.pattern_analyzer = components.get("pattern_analyzer")
            self.neural_flex_bridge = components.get("neural_flex_bridge")
            
            logger.info("Connected to provided language component instances")
            return True
            
        return False
    
    def connect_dashboard_panels(self, panels=None):
        """
        Connect to dashboard panels
        
        Args:
            panels: Dictionary of panel instances
        """
        if panels:
            self.language_panel = panels.get("language_processing_panel")
            self.neural_panel = panels.get("neural_activity_panel")
            
            logger.info("Connected to dashboard panel instances")
            return True
        else:
            logger.warning("No dashboard panels provided")
            return False
    
    def start(self):
        """Start the bridge processing"""
        if self.running:
            logger.info("Bridge is already running")
            return True
            
        logger.info("Starting Language Dashboard Bridge")
        
        # Start background threads for data processing
        self.running = True
        
        # Start collector thread
        self.collector_thread = threading.Thread(
            target=self._metrics_collector_loop,
            daemon=True,
            name="LanguageBridgeCollector"
        )
        self.collector_thread.start()
        
        # Start database writer thread
        self.writer_thread = threading.Thread(
            target=self._database_writer_loop,
            daemon=True,
            name="LanguageBridgeWriter"
        )
        self.writer_thread.start()
        
        # Start command processor thread
        self.command_thread = threading.Thread(
            target=self._command_processor_loop,
            daemon=True,
            name="LanguageBridgeCommander"
        )
        self.command_thread.start()
        
        logger.info("Language Dashboard Bridge started")
        return True
    
    def stop(self):
        """Stop the bridge processing"""
        if not self.running:
            return True
            
        logger.info("Stopping Language Dashboard Bridge")
        
        # Stop processing threads
        self.running = False
        
        # Wait for threads to finish
        if hasattr(self, 'collector_thread') and self.collector_thread.is_alive():
            self.collector_thread.join(timeout=2.0)
            
        if hasattr(self, 'writer_thread') and self.writer_thread.is_alive():
            self.writer_thread.join(timeout=2.0)
            
        if hasattr(self, 'command_thread') and self.command_thread.is_alive():
            self.command_thread.join(timeout=2.0)
            
        logger.info("Language Dashboard Bridge stopped")
        return True
    
    def _metrics_collector_loop(self):
        """Background thread for collecting metrics from language components"""
        while self.running:
            try:
                # Skip processing if in mock mode
                if self.mock_mode:
                    time.sleep(1.0)
                    continue
                    
                # Collect metrics from central node
                if self.central_language_node:
                    # Get system status
                    status = self.central_language_node.get_system_status()
                    
                    # Extract key metrics
                    llm_weight = status.get("llm_weight", 0.5)
                    nn_weight = status.get("nn_weight", 0.5)
                    
                    # Create language metric
                    language_metric = {
                        "timestamp": datetime.now(),
                        "value": 0.7 + 0.3 * llm_weight,  # Example calculation
                        "description": json.dumps({
                            "llm_weight": llm_weight,
                            "integration_level": 0.8 * llm_weight + 0.2 * nn_weight
                        })
                    }
                    
                    # Create neural metric
                    neural_metric = {
                        "timestamp": datetime.now(),
                        "value": 0.6 + 0.4 * nn_weight,  # Example calculation
                        "description": json.dumps({
                            "nn_weight": nn_weight,
                            "processor_activity": 0.75 * nn_weight
                        })
                    }
                    
                    # Add to metrics queues
                    self.language_data_queue.put(language_metric)
                    self.neural_data_queue.put(neural_metric)
                    
                # Process component-specific metrics if available
                self._collect_component_metrics()
                
                # Sleep to prevent high CPU usage
                time.sleep(1.0)
                    
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                time.sleep(5.0)  # Longer sleep on error
    
    def _collect_component_metrics(self):
        """Collect metrics from individual components"""
        try:
            # Collect from language memory
            if self.language_memory and hasattr(self.language_memory, 'get_status'):
                memory_status = self.language_memory.get_status()
                
                # Parse metrics and add to integration metrics
                if memory_status:
                    self.metrics["integration_metrics"].append({
                        "timestamp": datetime.now(),
                        "metric_name": "language_memory_associations",
                        "metric_value": memory_status.get("association_count", 0),
                        "details": json.dumps(memory_status)
                    })
            
            # Collect from neural linguistic processor
            if self.neural_linguistic_processor and hasattr(self.neural_linguistic_processor, 'get_status'):
                processor_status = self.neural_linguistic_processor.get_status()
                
                if processor_status:
                    self.metrics["integration_metrics"].append({
                        "timestamp": datetime.now(),
                        "metric_name": "neural_linguistic_score",
                        "metric_value": processor_status.get("activity_level", 0.5),
                        "details": json.dumps(processor_status)
                    })
                    
            # Collect from conscious mirror
            if self.conscious_mirror and hasattr(self.conscious_mirror, 'get_status'):
                mirror_status = self.conscious_mirror.get_status()
                
                if mirror_status:
                    self.metrics["integration_metrics"].append({
                        "timestamp": datetime.now(),
                        "metric_name": "consciousness_level",
                        "metric_value": mirror_status.get("consciousness_level", 0.5),
                        "details": json.dumps(mirror_status)
                    })
            
            # Collect from neural flex bridge
            if self.neural_flex_bridge and hasattr(self.neural_flex_bridge, 'get_status'):
                bridge_status = self.neural_flex_bridge.get_status()
                
                if bridge_status:
                    self.metrics["integration_metrics"].append({
                        "timestamp": datetime.now(),
                        "metric_name": "neural_flex_bridge_activity",
                        "metric_value": bridge_status.get("activity_level", 0.5),
                        "details": json.dumps(bridge_status)
                    })
                    
        except Exception as e:
            logger.error(f"Error collecting component metrics: {e}")
    
    def _database_writer_loop(self):
        """Background thread for writing metrics to database"""
        while self.running:
            try:
                # Process language metrics
                if not self.language_data_queue.empty():
                    metric = self.language_data_queue.get(block=False)
                    self._write_metric_to_database("language_metrics", metric)
                    self.language_data_queue.task_done()
                
                # Process neural metrics
                if not self.neural_data_queue.empty():
                    metric = self.neural_data_queue.get(block=False)
                    self._write_metric_to_database("neural_metrics", metric)
                    self.neural_data_queue.task_done()
                
                # Process integration metrics
                if self.metrics["integration_metrics"]:
                    metric = self.metrics["integration_metrics"].pop(0)
                    self._write_integration_metric_to_database(metric)
                
                # Sleep briefly to prevent CPU spinning
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in database writer: {e}")
                time.sleep(1.0)
    
    def _write_metric_to_database(self, table, metric):
        """
        Write a metric to the database
        
        Args:
            table: Table name
            metric: Metric dictionary
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Format timestamp
            timestamp_str = metric["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            
            # Insert metric
            cursor.execute(f"""
            INSERT INTO {table} (timestamp, value, description)
            VALUES (?, ?, ?)
            """, (timestamp_str, metric["value"], metric["description"]))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error writing metric to database: {e}")
    
    def _write_integration_metric_to_database(self, metric):
        """
        Write an integration metric to the database
        
        Args:
            metric: Integration metric dictionary
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Format timestamp
            timestamp_str = metric["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            
            # Insert metric
            cursor.execute("""
            INSERT INTO integration_metrics (timestamp, metric_name, metric_value, details)
            VALUES (?, ?, ?, ?)
            """, (timestamp_str, metric["metric_name"], metric["metric_value"], metric["details"]))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error writing integration metric to database: {e}")
    
    def _command_processor_loop(self):
        """Background thread for processing commands from dashboard to language components"""
        while self.running:
            try:
                if self.command_queue.empty():
                    time.sleep(0.1)  # Prevent CPU spinning
                    continue
                
                # Get command from queue
                command = self.command_queue.get(block=False)
                command_type = command.get("type")
                
                # Process different command types
                if command_type == "set_llm_weight":
                    self._handle_set_llm_weight(command)
                elif command_type == "set_nn_weight":
                    self._handle_set_nn_weight(command)
                elif command_type == "process_text":
                    self._handle_process_text(command)
                elif command_type == "refresh_stats":
                    self._handle_refresh_stats(command)
                else:
                    logger.warning(f"Unknown command type: {command_type}")
                
                self.command_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in command processor: {e}")
                time.sleep(1.0)
    
    def _handle_set_llm_weight(self, command):
        """Handle command to set LLM weight"""
        weight = command.get("weight")
        if weight is None:
            logger.warning("No weight specified in set_llm_weight command")
            return
            
        logger.info(f"Setting LLM weight to {weight}")
        
        if self.central_language_node:
            self.central_language_node.set_llm_weight(weight)
    
    def _handle_set_nn_weight(self, command):
        """Handle command to set neural network weight"""
        weight = command.get("weight")
        if weight is None:
            logger.warning("No weight specified in set_nn_weight command")
            return
            
        logger.info(f"Setting NN weight to {weight}")
        
        if self.central_language_node:
            self.central_language_node.set_nn_weight(weight)
    
    def _handle_process_text(self, command):
        """Handle command to process text"""
        text = command.get("text")
        callback = command.get("callback")
        
        if not text:
            logger.warning("No text specified in process_text command")
            return
            
        logger.info(f"Processing text: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        if self.central_language_node:
            result = self.central_language_node.process_text(text)
            
            # Call callback if provided
            if callback and callable(callback):
                callback(result)
                
            # Record metrics from result
            self._record_text_processing_metrics(result)
    
    def _handle_refresh_stats(self, command):
        """Handle command to refresh statistics"""
        if self.central_language_node:
            status = self.central_language_node.get_system_status()
            
            # Record as metrics
            self.metrics["integration_metrics"].append({
                "timestamp": datetime.now(),
                "metric_name": "system_status_refresh",
                "metric_value": 1.0,
                "details": json.dumps(status)
            })
    
    def _record_text_processing_metrics(self, result):
        """
        Record metrics from text processing result
        
        Args:
            result: Processing result dictionary
        """
        try:
            # Extract key metrics
            consciousness_level = result.get("consciousness_level", 0.5)
            neural_score = result.get("neural_linguistic_score", 0.6)
            final_score = result.get("final_score", 0.7)
            llm_weight_used = result.get("llm_weight_used", 0.5)
            nn_weight_used = result.get("nn_weight_used", 0.5)
            
            # Record as language metric
            self.language_data_queue.put({
                "timestamp": datetime.now(),
                "value": consciousness_level,
                "description": json.dumps({
                    "final_score": final_score,
                    "llm_weight": llm_weight_used
                })
            })
            
            # Record as neural metric
            self.neural_data_queue.put({
                "timestamp": datetime.now(),
                "value": neural_score,
                "description": json.dumps({
                    "final_score": final_score,
                    "nn_weight": nn_weight_used
                })
            })
            
        except Exception as e:
            logger.error(f"Error recording text processing metrics: {e}")
    
    def enqueue_command(self, command):
        """
        Enqueue a command for processing
        
        Args:
            command: Command dictionary
        """
        self.command_queue.put(command)
    
    def set_llm_weight(self, weight):
        """
        Set the LLM weight
        
        Args:
            weight: New weight value (0.0-1.0)
        """
        self.enqueue_command({
            "type": "set_llm_weight",
            "weight": weight
        })
    
    def set_nn_weight(self, weight):
        """
        Set the neural network weight
        
        Args:
            weight: New weight value (0.0-1.0)
        """
        self.enqueue_command({
            "type": "set_nn_weight",
            "weight": weight
        })
    
    def process_text(self, text, callback=None):
        """
        Process text through the language system
        
        Args:
            text: Text to process
            callback: Optional callback function for result
        """
        self.enqueue_command({
            "type": "process_text",
            "text": text,
            "callback": callback
        })
    
    def refresh_stats(self):
        """Refresh system statistics"""
        self.enqueue_command({
            "type": "refresh_stats"
        })
    
    def get_status(self):
        """
        Get bridge status
        
        Returns:
            Status dictionary
        """
        return {
            "initialized": self.initialized,
            "running": self.running,
            "mock_mode": self.mock_mode,
            "db_path": self.db_path,
            "components_connected": {
                "central_language_node": self.central_language_node is not None,
                "language_memory": self.language_memory is not None,
                "neural_linguistic_processor": self.neural_linguistic_processor is not None,
                "conscious_mirror": self.conscious_mirror is not None,
                "pattern_analyzer": self.pattern_analyzer is not None,
                "neural_flex_bridge": self.neural_flex_bridge is not None
            },
            "panels_connected": {
                "language_panel": self.language_panel is not None,
                "neural_panel": self.neural_panel is not None
            },
            "queue_status": {
                "language_data_queue": self.language_data_queue.qsize(),
                "neural_data_queue": self.neural_data_queue.qsize(),
                "command_queue": self.command_queue.qsize()
            }
        }

# Convenience function to get the bridge instance
def get_language_dashboard_bridge(config=None):
    """
    Get the singleton instance of the Language Dashboard Bridge
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LanguageDashboardBridge instance
    """
    return LanguageDashboardBridge.get_instance(config) 