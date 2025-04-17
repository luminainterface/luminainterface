"""
Database Integration Module for LUMINA V7.

This module provides the database integration component that connects 
the database system with the node consciousness system, facilitating
data storage, retrieval, and analysis for consciousness nodes.
"""

import os
import logging
import json
import sqlite3
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

class DatabaseIntegration:
    """
    Manages database operations for the LUMINA V7 node consciousness system.
    Provides interfaces for storing and retrieving node data, system metrics,
    and learning information.
    """
    
    def __init__(self, db_path: str = None, create_tables: bool = True):
        """
        Initialize the database integration component.
        
        Args:
            db_path: Path to the SQLite database file
            create_tables: Whether to create tables if they don't exist
        """
        self.logger = logging.getLogger("V7.DatabaseIntegration")
        
        # Set default database path if not provided
        if not db_path:
            # Create data directory if it doesn't exist
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            db_path = data_dir / "lumina_v7.db"
            
        self.db_path = str(db_path)
        self.logger.info(f"Initializing database integration with DB at: {self.db_path}")
        
        self.connection = None
        self._connect()
        
        if create_tables:
            self._create_tables()
            
        self.logger.info("Database integration initialized successfully")
    
    def _connect(self) -> None:
        """Establish connection to the database."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            self.logger.debug("Database connection established")
        except sqlite3.Error as e:
            self.logger.error(f"Database connection error: {e}")
            raise
    
    def _create_tables(self) -> None:
        """Create necessary database tables if they don't exist."""
        tables = {
            "nodes": """
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    capabilities TEXT NOT NULL,
                    status TEXT NOT NULL,
                    last_active TEXT NOT NULL,
                    creation_time TEXT NOT NULL,
                    metadata TEXT
                )
            """,
            "connections": """
                CREATE TABLE IF NOT EXISTS connections (
                    connection_id TEXT PRIMARY KEY,
                    source_node_id TEXT NOT NULL,
                    target_node_id TEXT NOT NULL,
                    connection_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (source_node_id) REFERENCES nodes (node_id),
                    FOREIGN KEY (target_node_id) REFERENCES nodes (node_id)
                )
            """,
            "metrics": """
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    metric_data TEXT NOT NULL
                )
            """,
            "learning_data": """
                CREATE TABLE IF NOT EXISTS learning_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    FOREIGN KEY (node_id) REFERENCES nodes (node_id)
                )
            """
        }
        
        cursor = self.connection.cursor()
        for table_name, create_sql in tables.items():
            try:
                cursor.execute(create_sql)
                self.logger.debug(f"Created or verified table: {table_name}")
            except sqlite3.Error as e:
                self.logger.error(f"Error creating table {table_name}: {e}")
        
        self.connection.commit()
    
    def store_node_data(self, node_id: str, node_type: str, capabilities: Dict, 
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
        try:
            now = datetime.datetime.now().isoformat()
            cursor = self.connection.cursor()
            
            # Check if the node already exists
            cursor.execute("SELECT node_id FROM nodes WHERE node_id = ?", (node_id,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing node
                cursor.execute("""
                    UPDATE nodes 
                    SET node_type = ?, capabilities = ?, status = ?, 
                        last_active = ?, metadata = ?
                    WHERE node_id = ?
                """, (
                    node_type, 
                    json.dumps(capabilities), 
                    status,
                    now,
                    json.dumps(metadata) if metadata else None,
                    node_id
                ))
            else:
                # Insert new node
                cursor.execute("""
                    INSERT INTO nodes 
                    (node_id, node_type, capabilities, status, last_active, creation_time, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    node_id,
                    node_type,
                    json.dumps(capabilities),
                    status,
                    now,
                    now,
                    json.dumps(metadata) if metadata else None
                ))
                
            self.connection.commit()
            self.logger.debug(f"Stored data for node: {node_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error storing node data: {e}")
            return False
    
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
        try:
            now = datetime.datetime.now().isoformat()
            cursor = self.connection.cursor()
            
            # Check if the connection already exists
            cursor.execute("SELECT connection_id FROM connections WHERE connection_id = ?", 
                         (connection_id,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing connection
                cursor.execute("""
                    UPDATE connections 
                    SET source_node_id = ?, target_node_id = ?, connection_type = ?, 
                        status = ?, metadata = ?
                    WHERE connection_id = ?
                """, (
                    source_node_id,
                    target_node_id,
                    connection_type,
                    status,
                    json.dumps(metadata) if metadata else None,
                    connection_id
                ))
            else:
                # Insert new connection
                cursor.execute("""
                    INSERT INTO connections 
                    (connection_id, source_node_id, target_node_id, connection_type, 
                     status, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    connection_id,
                    source_node_id,
                    target_node_id,
                    connection_type,
                    status,
                    now,
                    json.dumps(metadata) if metadata else None
                ))
                
            self.connection.commit()
            self.logger.debug(f"Stored connection: {connection_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error storing connection data: {e}")
            return False
    
    def store_metrics(self, metric_type: str, metric_data: Dict) -> bool:
        """
        Store system metrics in the database.
        
        Args:
            metric_type: Type of metrics being stored
            metric_data: Dictionary containing the metrics
            
        Returns:
            Success status of the operation
        """
        try:
            now = datetime.datetime.now().isoformat()
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT INTO metrics (timestamp, metric_type, metric_data)
                VALUES (?, ?, ?)
            """, (
                now,
                metric_type,
                json.dumps(metric_data)
            ))
                
            self.connection.commit()
            self.logger.debug(f"Stored metrics of type: {metric_type}")
            return True
        except Exception as e:
            self.logger.error(f"Error storing metrics: {e}")
            return False
    
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
        try:
            now = datetime.datetime.now().isoformat()
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT INTO learning_data (node_id, timestamp, data_type, data)
                VALUES (?, ?, ?, ?)
            """, (
                node_id,
                now,
                data_type,
                json.dumps(data)
            ))
                
            self.connection.commit()
            self.logger.debug(f"Stored learning data for node: {node_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error storing learning data: {e}")
            return False
    
    def get_node_data(self, node_id: str = None, node_type: str = None) -> List[Dict]:
        """
        Retrieve node data from the database.
        
        Args:
            node_id: Optional ID to filter by specific node
            node_type: Optional type to filter by node type
            
        Returns:
            List of node data dictionaries
        """
        try:
            cursor = self.connection.cursor()
            query = "SELECT * FROM nodes"
            params = []
            
            if node_id or node_type:
                query += " WHERE"
                
                if node_id:
                    query += " node_id = ?"
                    params.append(node_id)
                    
                if node_type:
                    if node_id:
                        query += " AND"
                    query += " node_type = ?"
                    params.append(node_type)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries with parsed JSON fields
            result = []
            for row in rows:
                node_dict = dict(row)
                node_dict['capabilities'] = json.loads(node_dict['capabilities'])
                if node_dict['metadata']:
                    node_dict['metadata'] = json.loads(node_dict['metadata'])
                result.append(node_dict)
                
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving node data: {e}")
            return []
    
    def get_connections(self, node_id: str = None, connection_type: str = None) -> List[Dict]:
        """
        Retrieve connection data from the database.
        
        Args:
            node_id: Optional ID to filter connections by node
            connection_type: Optional type to filter by connection type
            
        Returns:
            List of connection data dictionaries
        """
        try:
            cursor = self.connection.cursor()
            query = "SELECT * FROM connections"
            params = []
            
            if node_id or connection_type:
                query += " WHERE"
                
                if node_id:
                    query += " (source_node_id = ? OR target_node_id = ?)"
                    params.extend([node_id, node_id])
                    
                if connection_type:
                    if node_id:
                        query += " AND"
                    query += " connection_type = ?"
                    params.append(connection_type)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries with parsed JSON fields
            result = []
            for row in rows:
                conn_dict = dict(row)
                if conn_dict['metadata']:
                    conn_dict['metadata'] = json.loads(conn_dict['metadata'])
                result.append(conn_dict)
                
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving connection data: {e}")
            return []
    
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
        try:
            cursor = self.connection.cursor()
            query = "SELECT * FROM metrics"
            params = []
            
            where_clauses = []
            if metric_type:
                where_clauses.append("metric_type = ?")
                params.append(metric_type)
                
            if start_time:
                where_clauses.append("timestamp >= ?")
                params.append(start_time)
                
            if end_time:
                where_clauses.append("timestamp <= ?")
                params.append(end_time)
                
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
                
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries with parsed JSON fields
            result = []
            for row in rows:
                metric_dict = dict(row)
                metric_dict['metric_data'] = json.loads(metric_dict['metric_data'])
                result.append(metric_dict)
                
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving metrics: {e}")
            return []
    
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
        try:
            cursor = self.connection.cursor()
            query = "SELECT * FROM learning_data"
            params = []
            
            where_clauses = []
            if node_id:
                where_clauses.append("node_id = ?")
                params.append(node_id)
                
            if data_type:
                where_clauses.append("data_type = ?")
                params.append(data_type)
                
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
                
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries with parsed JSON fields
            result = []
            for row in rows:
                data_dict = dict(row)
                data_dict['data'] = json.loads(data_dict['data'])
                result.append(data_dict)
                
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving learning data: {e}")
            return []
    
    def backup_database(self, backup_path: str = None) -> bool:
        """
        Create a backup of the current database.
        
        Args:
            backup_path: Path where the backup should be stored
            
        Returns:
            Success status of the operation
        """
        if not backup_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"data/lumina_v7_backup_{timestamp}.db"
            
        try:
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Create backup connection
            backup_conn = sqlite3.connect(backup_path)
            
            # Copy database to backup
            with backup_conn:
                self.connection.backup(backup_conn)
                
            backup_conn.close()
            self.logger.info(f"Database backup created at: {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error creating database backup: {e}")
            return False
    
    def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed")


# For standalone testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create an instance and test functionality
    db = DatabaseIntegration(db_path="test_lumina_v7.db")
    
    # Test storing and retrieving node data
    db.store_node_data(
        "test_node_1",
        "memory",
        {"can_learn": True, "can_recall": True},
        "active",
        {"description": "Test memory node"}
    )
    
    # Test storing a connection
    db.store_connection(
        "conn_1",
        "test_node_1",
        "test_node_2",
        "mirror",
        "active",
        {"strength": 0.9}
    )
    
    # Test storing metrics
    db.store_metrics(
        "system_status",
        {
            "active_nodes": 2,
            "total_nodes": 5,
            "connections": 3,
            "cpu_usage": 0.4,
            "memory_usage": 0.3
        }
    )
    
    # Test retrieving data
    nodes = db.get_node_data()
    print(f"Retrieved {len(nodes)} nodes")
    
    connections = db.get_connections()
    print(f"Retrieved {len(connections)} connections")
    
    metrics = db.get_metrics(limit=10)
    print(f"Retrieved {len(metrics)} metrics")
    
    # Close the connection
    db.close() 