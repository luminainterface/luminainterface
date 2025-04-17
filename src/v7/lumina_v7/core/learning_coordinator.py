"""
Learning Coordinator for LUMINA V7

This module coordinates learning processes across consciousness nodes,
managing knowledge transfer, learning optimization, and pattern recognition.
"""

import os
import logging
import time
import threading
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from pathlib import Path
from collections import defaultdict

from .node_consciousness_manager import NodeConsciousnessManager
from .database_manager import DatabaseManager
from .database_integration import DatabaseIntegration

# Configure logging
logger = logging.getLogger("lumina_v7.learning")

class LearningCoordinator:
    """
    Coordinates learning processes across consciousness nodes.
    Manages knowledge transfer, learning optimization, and pattern recognition.
    """
    
    def __init__(self, node_manager: NodeConsciousnessManager,
                 db_manager: DatabaseManager,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the learning coordinator.
        
        Args:
            node_manager: Node consciousness manager instance
            db_manager: Database manager instance
            config: Configuration options
        """
        self.node_manager = node_manager
        self.db_manager = db_manager
        self.config = config or {}
        
        # Learning state
        self.learning_nodes = set()
        self.active_patterns = {}
        self.knowledge_transfers = {}
        
        # Performance metrics
        self.metrics = {
            "learning_cycles": 0,
            "patterns_learned": 0,
            "knowledge_transfers": 0,
            "optimization_events": 0,
            "performance": {
                "avg_learning_time": 0,
                "avg_transfer_time": 0,
                "success_rate": 0
            }
        }
        
        # Learning configuration
        self.learning_config = {
            "max_concurrent_learning": 3,
            "min_confidence_threshold": 0.7,
            "max_pattern_age": 3600,  # 1 hour
            "optimization_interval": 300,  # 5 minutes
            "transfer_batch_size": 10
        }
        
        # Update config with any provided values
        self.learning_config.update(self.config.get("learning", {}))
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("Learning Coordinator initialized")
    
    def register_learning_node(self, node_id: str) -> bool:
        """
        Register a node for learning coordination.
        
        Args:
            node_id: ID of the node to register
            
        Returns:
            bool: True if registration was successful
        """
        if node_id in self.learning_nodes:
            logger.warning(f"Node {node_id} already registered for learning")
            return False
            
        # Verify node exists and is active
        if not self.node_manager.is_node_active(node_id):
            logger.error(f"Cannot register inactive node: {node_id}")
            return False
            
        # Add to learning nodes
        self.learning_nodes.add(node_id)
        
        # Initialize learning state for node
        self.active_patterns[node_id] = []
        self.knowledge_transfers[node_id] = {
            "pending": [],
            "completed": [],
            "failed": []
        }
        
        logger.info(f"Registered node for learning: {node_id}")
        return True
    
    def unregister_learning_node(self, node_id: str) -> bool:
        """
        Unregister a node from learning coordination.
        
        Args:
            node_id: ID of the node to unregister
            
        Returns:
            bool: True if unregistration was successful
        """
        if node_id not in self.learning_nodes:
            logger.warning(f"Node {node_id} not registered for learning")
            return False
            
        # Remove from learning nodes
        self.learning_nodes.remove(node_id)
        
        # Clean up learning state
        if node_id in self.active_patterns:
            del self.active_patterns[node_id]
        if node_id in self.knowledge_transfers:
            del self.knowledge_transfers[node_id]
            
        logger.info(f"Unregistered node from learning: {node_id}")
        return True
    
    def start_learning_cycle(self, node_id: str, 
                           learning_data: Dict[str, Any]) -> bool:
        """
        Start a learning cycle for a node.
        
        Args:
            node_id: ID of the node to start learning
            learning_data: Data to use for learning
            
        Returns:
            bool: True if learning cycle started successfully
        """
        if node_id not in self.learning_nodes:
            logger.error(f"Node {node_id} not registered for learning")
            return False
            
        # Check if node is already learning
        if len(self.active_patterns[node_id]) >= self.learning_config["max_concurrent_learning"]:
            logger.warning(f"Node {node_id} has too many active learning patterns")
            return False
            
        try:
            # Start learning process
            pattern_id = f"pattern_{int(time.time())}_{len(self.active_patterns[node_id])}"
            self.active_patterns[node_id].append({
                "id": pattern_id,
                "start_time": time.time(),
                "data": learning_data,
                "status": "learning"
            })
            
            # Store learning data
            self.db_manager.store_learning_data(
                node_id,
                "pattern",
                {
                    "pattern_id": pattern_id,
                    "data": learning_data,
                    "status": "learning"
                }
            )
            
            logger.info(f"Started learning cycle for node {node_id}: {pattern_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting learning cycle: {e}")
            return False
    
    def complete_learning_cycle(self, node_id: str, pattern_id: str,
                              results: Dict[str, Any]) -> bool:
        """
        Complete a learning cycle and store results.
        
        Args:
            node_id: ID of the node that completed learning
            pattern_id: ID of the pattern that was learned
            results: Learning results including confidence and patterns
            
        Returns:
            bool: True if learning cycle was completed successfully
        """
        if node_id not in self.learning_nodes:
            logger.error(f"Node {node_id} not registered for learning")
            return False
            
        # Find and update the pattern
        pattern_found = False
        for pattern in self.active_patterns[node_id]:
            if pattern["id"] == pattern_id:
                pattern["end_time"] = time.time()
                pattern["results"] = results
                pattern["status"] = "completed"
                pattern_found = True
                break
                
        if not pattern_found:
            logger.error(f"Pattern {pattern_id} not found for node {node_id}")
            return False
            
        # Store results
        try:
            self.db_manager.store_learning_data(
                node_id,
                "pattern_results",
                {
                    "pattern_id": pattern_id,
                    "results": results,
                    "learning_time": time.time() - pattern["start_time"]
                }
            )
            
            # Update metrics
            self.metrics["patterns_learned"] += 1
            self.metrics["learning_cycles"] += 1
            
            # Calculate performance metrics
            learning_time = time.time() - pattern["start_time"]
            self.metrics["performance"]["avg_learning_time"] = (
                (self.metrics["performance"]["avg_learning_time"] * 
                 (self.metrics["learning_cycles"] - 1) + learning_time) /
                self.metrics["learning_cycles"]
            )
            
            logger.info(f"Completed learning cycle for node {node_id}: {pattern_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error completing learning cycle: {e}")
            return False
    
    def schedule_knowledge_transfer(self, source_node_id: str,
                                  target_node_id: str,
                                  knowledge_data: Dict[str, Any]) -> bool:
        """
        Schedule a knowledge transfer between nodes.
        
        Args:
            source_node_id: ID of the source node
            target_node_id: ID of the target node
            knowledge_data: Data to transfer
            
        Returns:
            bool: True if transfer was scheduled successfully
        """
        if source_node_id not in self.learning_nodes:
            logger.error(f"Source node {source_node_id} not registered for learning")
            return False
            
        if target_node_id not in self.learning_nodes:
            logger.error(f"Target node {target_node_id} not registered for learning")
            return False
            
        try:
            # Create transfer record
            transfer_id = f"transfer_{int(time.time())}"
            transfer = {
                "id": transfer_id,
                "source": source_node_id,
                "target": target_node_id,
                "data": knowledge_data,
                "status": "pending",
                "created_at": time.time()
            }
            
            # Add to pending transfers
            self.knowledge_transfers[source_node_id]["pending"].append(transfer)
            
            # Store transfer record
            self.db_manager.store_learning_data(
                source_node_id,
                "knowledge_transfer",
                transfer
            )
            
            logger.info(f"Scheduled knowledge transfer from {source_node_id} to {target_node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error scheduling knowledge transfer: {e}")
            return False
    
    def _start_background_tasks(self) -> None:
        """Start background tasks for learning coordination."""
        # Start optimization thread
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True
        )
        self.optimization_thread.start()
        
        # Start transfer processing thread
        self.transfer_thread = threading.Thread(
            target=self._transfer_processing_loop,
            daemon=True
        )
        self.transfer_thread.start()
        
        logger.info("Started background learning coordination tasks")
    
    def _optimization_loop(self) -> None:
        """Background loop for learning optimization."""
        while True:
            try:
                # Sleep for optimization interval
                time.sleep(self.learning_config["optimization_interval"])
                
                # Perform optimization
                self._optimize_learning()
                
                # Clean up old patterns
                self._cleanup_old_patterns()
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
    
    def _transfer_processing_loop(self) -> None:
        """Background loop for processing knowledge transfers."""
        while True:
            try:
                # Process transfers for each node
                for node_id in self.learning_nodes:
                    self._process_transfers(node_id)
                    
                # Sleep briefly to avoid excessive CPU usage
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in transfer processing loop: {e}")
    
    def _optimize_learning(self) -> None:
        """Optimize learning processes based on current state."""
        try:
            # Get current learning statistics
            stats = self._get_learning_statistics()
            
            # Adjust learning parameters based on performance
            if stats["success_rate"] < 0.8:
                # Increase confidence threshold if success rate is low
                self.learning_config["min_confidence_threshold"] = min(
                    0.9,
                    self.learning_config["min_confidence_threshold"] + 0.05
                )
                
            # Update metrics
            self.metrics["optimization_events"] += 1
            
            logger.info("Completed learning optimization cycle")
            
        except Exception as e:
            logger.error(f"Error optimizing learning: {e}")
    
    def _cleanup_old_patterns(self) -> None:
        """Clean up old learning patterns."""
        current_time = time.time()
        max_age = self.learning_config["max_pattern_age"]
        
        for node_id in self.learning_nodes:
            # Remove old patterns
            self.active_patterns[node_id] = [
                p for p in self.active_patterns[node_id]
                if current_time - p["start_time"] < max_age
            ]
    
    def _process_transfers(self, node_id: str) -> None:
        """Process pending knowledge transfers for a node."""
        try:
            transfers = self.knowledge_transfers[node_id]["pending"]
            if not transfers:
                return
                
            # Process transfers in batches
            batch_size = self.learning_config["transfer_batch_size"]
            for i in range(0, len(transfers), batch_size):
                batch = transfers[i:i + batch_size]
                
                # Process each transfer in the batch
                for transfer in batch:
                    if self._execute_transfer(transfer):
                        # Move to completed
                        self.knowledge_transfers[node_id]["pending"].remove(transfer)
                        self.knowledge_transfers[node_id]["completed"].append(transfer)
                    else:
                        # Move to failed
                        self.knowledge_transfers[node_id]["pending"].remove(transfer)
                        self.knowledge_transfers[node_id]["failed"].append(transfer)
                        
                # Update metrics
                self.metrics["knowledge_transfers"] += len(batch)
                
        except Exception as e:
            logger.error(f"Error processing transfers for node {node_id}: {e}")
    
    def _execute_transfer(self, transfer: Dict[str, Any]) -> bool:
        """
        Execute a knowledge transfer.
        
        Args:
            transfer: Transfer record to execute
            
        Returns:
            bool: True if transfer was successful
        """
        try:
            # Get source and target nodes
            source_id = transfer["source"]
            target_id = transfer["target"]
            
            # Verify nodes are active
            if not (self.node_manager.is_node_active(source_id) and 
                   self.node_manager.is_node_active(target_id)):
                return False
                
            # Execute transfer through node manager
            success = self.node_manager.transfer_knowledge(
                source_id,
                target_id,
                transfer["data"]
            )
            
            # Update transfer status
            transfer["status"] = "completed" if success else "failed"
            transfer["completed_at"] = time.time()
            
            # Store transfer result
            self.db_manager.store_learning_data(
                source_id,
                "transfer_result",
                {
                    "transfer_id": transfer["id"],
                    "success": success,
                    "completion_time": transfer["completed_at"]
                }
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing transfer {transfer['id']}: {e}")
            return False
    
    def _get_learning_statistics(self) -> Dict[str, Any]:
        """
        Get current learning statistics.
        
        Returns:
            Dictionary with learning statistics
        """
        stats = {
            "active_nodes": len(self.learning_nodes),
            "active_patterns": sum(len(p) for p in self.active_patterns.values()),
            "pending_transfers": sum(
                len(t["pending"]) for t in self.knowledge_transfers.values()
            ),
            "success_rate": 0,
            "avg_learning_time": self.metrics["performance"]["avg_learning_time"],
            "avg_transfer_time": self.metrics["performance"]["avg_transfer_time"]
        }
        
        # Calculate success rate
        total_cycles = self.metrics["learning_cycles"]
        if total_cycles > 0:
            stats["success_rate"] = (
                self.metrics["patterns_learned"] / total_cycles
            )
            
        return stats
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """
        Get current coordination status.
        
        Returns:
            Dictionary with coordination status
        """
        return {
            "learning_nodes": list(self.learning_nodes),
            "active_patterns": {
                node_id: len(patterns)
                for node_id, patterns in self.active_patterns.items()
            },
            "knowledge_transfers": {
                node_id: {
                    "pending": len(transfers["pending"]),
                    "completed": len(transfers["completed"]),
                    "failed": len(transfers["failed"])
                }
                for node_id, transfers in self.knowledge_transfers.items()
            },
            "metrics": self.metrics,
            "config": self.learning_config
        }
    
    def shutdown(self) -> None:
        """Shutdown the learning coordinator."""
        try:
            # Stop background threads
            if hasattr(self, 'optimization_thread'):
                self.optimization_thread.join(timeout=1)
            if hasattr(self, 'transfer_thread'):
                self.transfer_thread.join(timeout=1)
                
            # Store final metrics
            self.db_manager.store_learning_data(
                "system",
                "learning_coordinator_metrics",
                {
                    "final_metrics": self.metrics,
                    "shutdown_time": time.time()
                }
            )
            
            logger.info("Learning Coordinator shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# For standalone testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test instances
    node_manager = NodeConsciousnessManager()
    db_manager = DatabaseManager()
    
    # Create coordinator
    coordinator = LearningCoordinator(node_manager, db_manager)
    
    # Test registration
    test_node = "test_node_1"
    coordinator.register_learning_node(test_node)
    
    # Test learning cycle
    coordinator.start_learning_cycle(
        test_node,
        {"data": "test_data", "type": "test_pattern"}
    )
    
    # Test knowledge transfer
    coordinator.schedule_knowledge_transfer(
        test_node,
        "test_node_2",
        {"knowledge": "test_knowledge"}
    )
    
    # Get status
    status = coordinator.get_coordination_status()
    print(f"Coordination status: {status}")
    
    # Shutdown
    coordinator.shutdown() 