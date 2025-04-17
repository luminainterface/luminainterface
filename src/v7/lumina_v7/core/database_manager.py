"""
LUMINA V7 Database Manager

This module provides a robust database management system for LUMINA V7,
featuring automatic sorting and learning mechanisms.
"""

import os
import json
import sqlite3
import logging
import time
import threading
import numpy as np
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from datetime import datetime
from pathlib import Path
import hashlib
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lumina_v7.database")

# Import the database integration module
from src.v7.lumina_v7.core.database_integration import DatabaseIntegration

class DatabaseManager:
    """
    High-level database manager for the LUMINA V7 node consciousness system.
    Provides connection pooling, transaction management, and query optimization.
    """
    
    def __init__(self, db_path: str = None, pool_size: int = 3, 
                create_tables: bool = True):
        """
        Initialize the database manager with connection pooling.
        
        Args:
            db_path: Path to the SQLite database file
            pool_size: Size of the connection pool
            create_tables: Whether to create tables if they don't exist
        """
        self.logger = logging.getLogger("V7.DatabaseManager")
        self.db_path = db_path
        self.pool_size = max(1, min(pool_size, 10))  # Between 1 and 10
        
        # Connection pool
        self.connection_pool = []
        self.pool_lock = threading.RLock()
        self.pool_semaphore = threading.Semaphore(self.pool_size)
        
        # Create initial connections in the pool
        self._initialize_pool(create_tables)
        
        # Stats
        self.stats = {
            "total_queries": 0,
            "failed_queries": 0,
            "transactions": 0,
            "peak_connections": 0,
            "connection_wait_time": 0,
        }
        
        self.logger.info(f"Database manager initialized with pool size: {self.pool_size}")
    
    def _initialize_pool(self, create_tables: bool) -> None:
        """Initialize the connection pool with database connections."""
        self.logger.debug(f"Initializing connection pool with {self.pool_size} connections")
        
        # Create the first connection with table creation if needed
        first_connection = DatabaseIntegration(self.db_path, create_tables)
        self.connection_pool.append({"connection": first_connection, "in_use": False})
        
        # Create additional connections without table creation (tables already exist)
        for _ in range(self.pool_size - 1):
            connection = DatabaseIntegration(self.db_path, create_tables=False)
            self.connection_pool.append({"connection": connection, "in_use": False})
    
    def _get_connection(self) -> Tuple[DatabaseIntegration, int]:
        """
        Get a database connection from the pool.
        
        Returns:
            Tuple of (connection, connection_index)
        """
        start_time = time.time()
        
        # Acquire the semaphore to ensure we don't exceed pool size
        acquired = self.pool_semaphore.acquire(timeout=30)  # 30-second timeout
        if not acquired:
            self.logger.error("Timed out waiting for database connection")
            raise TimeoutError("Timed out waiting for database connection")
        
        try:
            with self.pool_lock:
                # Look for an available connection
                for idx, conn_data in enumerate(self.connection_pool):
                    if not conn_data["in_use"]:
                        conn_data["in_use"] = True
                        wait_time = time.time() - start_time
                        self.stats["connection_wait_time"] += wait_time
                        return conn_data["connection"], idx
                
                # No connection available (should not happen with semaphore)
                self.logger.error("No available connection in pool despite semaphore")
                raise RuntimeError("No available database connection")
        except Exception as e:
            # Release the semaphore if we couldn't get a connection
            self.pool_semaphore.release()
            raise e
    
    def _release_connection(self, connection_index: int) -> None:
        """
        Release a connection back to the pool.
        
        Args:
            connection_index: Index of the connection in the pool
        """
        with self.pool_lock:
            if 0 <= connection_index < len(self.connection_pool):
                self.connection_pool[connection_index]["in_use"] = False
            else:
                self.logger.error(f"Invalid connection index: {connection_index}")
        
        # Release the semaphore
        self.pool_semaphore.release()
    
    def execute_operation(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute a database operation using a connection from the pool.
        
        Args:
            operation: Function that takes a DatabaseIntegration as first arg
            *args: Arguments to pass to the operation
            **kwargs: Keyword arguments to pass to the operation
            
        Returns:
            Result of the operation
        """
        connection = None
        connection_index = -1
        
        try:
            # Get a connection from the pool
            connection, connection_index = self._get_connection()
            
            # Execute the operation and update stats
            self.stats["total_queries"] += 1
            result = operation(connection, *args, **kwargs)
            return result
            
        except Exception as e:
            self.stats["failed_queries"] += 1
            self.logger.error(f"Error executing database operation: {e}")
            raise
        finally:
            # Release the connection back to the pool
            if connection_index >= 0:
                self._release_connection(connection_index)
    
    # Node operations
    def store_node(self, node_id: str, node_type: str, capabilities: Dict, 
                 status: str, metadata: Optional[Dict] = None) -> bool:
        """
        Store node data in the database.
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type of the node
            capabilities: Dictionary of node capabilities
            status: Current status of the node
            metadata: Additional metadata for the node
            
        Returns:
            Success status of the operation
        """
        return self.execute_operation(
            lambda conn, *args, **kwargs: conn.store_node_data(*args, **kwargs),
            node_id, node_type, capabilities, status, metadata
        )
    
    def get_nodes(self, node_id: str = None, node_type: str = None) -> List[Dict]:
        """
        Retrieve node data from the database.
        
        Args:
            node_id: Optional ID to filter by specific node
            node_type: Optional type to filter by node type
            
        Returns:
            List of node data dictionaries
        """
        return self.execute_operation(
            lambda conn, *args, **kwargs: conn.get_node_data(*args, **kwargs),
            node_id, node_type
        )
    
    # Connection operations
    def store_connection(self, connection_id: str, source_node_id: str, 
                       target_node_id: str, connection_type: str, 
                       status: str = "active", metadata: Optional[Dict] = None) -> bool:
        """
        Store connection data in the database.
        
        Args:
            connection_id: Unique identifier for the connection
            source_node_id: ID of the source node
            target_node_id: ID of the target node
            connection_type: Type of connection
            status: Status of the connection
            metadata: Additional metadata
            
        Returns:
            Success status of the operation
        """
        return self.execute_operation(
            lambda conn, *args, **kwargs: conn.store_connection(*args, **kwargs),
            connection_id, source_node_id, target_node_id, connection_type, 
            status, metadata
        )
    
    def get_connections(self, node_id: str = None, 
                      connection_type: str = None) -> List[Dict]:
        """
        Retrieve connection data from the database.
        
        Args:
            node_id: Optional ID to filter connections by node
            connection_type: Optional type to filter by connection type
            
        Returns:
            List of connection data dictionaries
        """
        return self.execute_operation(
            lambda conn, *args, **kwargs: conn.get_connections(*args, **kwargs),
            node_id, connection_type
        )
    
    # Metrics operations
    def store_metrics(self, metric_type: str, metric_data: Dict) -> bool:
        """
        Store system metrics in the database.
        
        Args:
            metric_type: Type of metrics being stored
            metric_data: Dictionary containing the metrics
            
        Returns:
            Success status of the operation
        """
        return self.execute_operation(
            lambda conn, *args, **kwargs: conn.store_metrics(*args, **kwargs),
            metric_type, metric_data
        )
    
    def get_metrics(self, metric_type: str = None, 
                  start_time: str = None, end_time: str = None, 
                  limit: int = 100) -> List[Dict]:
        """
        Retrieve metrics from the database with optional filtering.
        
        Args:
            metric_type: Optional type to filter metrics
            start_time: Optional ISO format start time for filtering
            end_time: Optional ISO format end time for filtering
            limit: Maximum number of metrics to return
            
        Returns:
            List of metric data dictionaries
        """
        return self.execute_operation(
            lambda conn, *args, **kwargs: conn.get_metrics(*args, **kwargs),
            metric_type, start_time, end_time, limit
        )
    
    # Learning data operations
    def store_learning_data(self, node_id: str, data_type: str, data: Dict) -> bool:
        """
        Store learning data for a specific node.
        
        Args:
            node_id: ID of the node
            data_type: Type of learning data
            data: The learning data to store
            
        Returns:
            Success status of the operation
        """
        return self.execute_operation(
            lambda conn, *args, **kwargs: conn.store_learning_data(*args, **kwargs),
            node_id, data_type, data
        )
    
    def get_learning_data(self, node_id: str = None, data_type: str = None,
                        limit: int = 100) -> List[Dict]:
        """
        Retrieve learning data from the database.
        
        Args:
            node_id: Optional ID to filter by specific node
            data_type: Optional type to filter by data type
            limit: Maximum number of records to return
            
        Returns:
            List of learning data dictionaries
        """
        return self.execute_operation(
            lambda conn, *args, **kwargs: conn.get_learning_data(*args, **kwargs),
            node_id, data_type, limit
        )
    
    # Transaction management
    def transaction(self, operations: List[Dict]) -> Dict:
        """
        Execute multiple operations as a single transaction.
        
        Args:
            operations: List of operation dictionaries with keys:
                - 'type': Operation type (store_node, store_connection, etc.)
                - 'args': Dict of arguments for the operation
            
        Returns:
            Dictionary with results for each operation
        """
        self.stats["transactions"] += 1
        results = {"success": True, "operations": []}
        
        connection = None
        connection_index = -1
        
        try:
            # Get a connection from the pool
            connection, connection_index = self._get_connection()
            
            # Execute each operation
            for op in operations:
                op_type = op.get('type')
                op_args = op.get('args', {})
                
                try:
                    # Map operation type to method
                    if op_type == 'store_node':
                        result = connection.store_node_data(**op_args)
                    elif op_type == 'store_connection':
                        result = connection.store_connection(**op_args)
                    elif op_type == 'store_metrics':
                        result = connection.store_metrics(**op_args)
                    elif op_type == 'store_learning_data':
                        result = connection.store_learning_data(**op_args)
                    else:
                        result = False
                        self.logger.error(f"Unknown operation type: {op_type}")
                    
                    results["operations"].append({
                        "type": op_type,
                        "success": result
                    })
                    
                    # Update query stats
                    self.stats["total_queries"] += 1
                    if not result:
                        self.stats["failed_queries"] += 1
                        
                except Exception as e:
                    self.logger.error(f"Error in transaction operation {op_type}: {e}")
                    results["operations"].append({
                        "type": op_type,
                        "success": False,
                        "error": str(e)
                    })
                    results["success"] = False
                    self.stats["failed_queries"] += 1
            
            return results
            
        except Exception as e:
            self.stats["failed_queries"] += len(operations)
            self.logger.error(f"Transaction failed: {e}")
            raise
        finally:
            # Release the connection back to the pool
            if connection_index >= 0:
                self._release_connection(connection_index)
    
    def backup_database(self, backup_path: str = None) -> bool:
        """
        Create a backup of the current database.
        
        Args:
            backup_path: Path where the backup should be stored
            
        Returns:
            Success status of the operation
        """
        return self.execute_operation(
            lambda conn, *args, **kwargs: conn.backup_database(*args, **kwargs),
            backup_path
        )
    
    def get_stats(self) -> Dict:
        """
        Get database manager statistics.
        
        Returns:
            Dictionary of statistics
        """
        # Count current connections in use
        connections_in_use = 0
        with self.pool_lock:
            for conn_data in self.connection_pool:
                if conn_data["in_use"]:
                    connections_in_use += 1
        
        # Update peak connections if needed
        if connections_in_use > self.stats["peak_connections"]:
            self.stats["peak_connections"] = connections_in_use
        
        # Create a copy of the stats dictionary with current connections
        stats = dict(self.stats)
        stats["current_connections"] = connections_in_use
        stats["total_connections"] = len(self.connection_pool)
        
        if stats["total_queries"] > 0:
            stats["failure_rate"] = stats["failed_queries"] / stats["total_queries"]
        else:
            stats["failure_rate"] = 0
            
        return stats
    
    def close(self) -> None:
        """Close all database connections in the pool."""
        self.logger.info("Closing all database connections in the pool")
        
        with self.pool_lock:
            for conn_data in self.connection_pool:
                try:
                    conn_data["connection"].close()
                except Exception as e:
                    self.logger.error(f"Error closing database connection: {e}")
            
            # Clear the connection pool
            self.connection_pool.clear()


# For standalone testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create an instance and test functionality
    db_manager = DatabaseManager(db_path="test_lumina_v7.db", pool_size=2)
    
    # Test storing node data
    success = db_manager.store_node(
        "test_node_2",
        "language",
        {"can_process": True, "can_generate": True},
        "active",
        {"description": "Test language node"}
    )
    print(f"Store node result: {success}")
    
    # Test transaction with multiple operations
    transaction_results = db_manager.transaction([
        {
            "type": "store_node",
            "args": {
                "node_id": "test_node_3",
                "node_type": "attention",
                "capabilities": {"can_focus": True},
                "status": "inactive"
            }
        },
        {
            "type": "store_connection",
            "args": {
                "connection_id": "conn_2",
                "source_node_id": "test_node_2",
                "target_node_id": "test_node_3",
                "connection_type": "pipeline"
            }
        }
    ])
    print(f"Transaction results: {transaction_results}")
    
    # Test getting nodes
    nodes = db_manager.get_nodes()
    print(f"Retrieved {len(nodes)} nodes")
    
    # Test stats
    stats = db_manager.get_stats()
    print(f"Database manager stats: {stats}")
    
    # Close the manager
    db_manager.close() 