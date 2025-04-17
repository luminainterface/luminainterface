"""
Database manager for the V1 Spiderweb Bridge.
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from .schema import SCHEMA

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations for the V1 Spiderweb Bridge."""
    
    def __init__(self, db_path: str = "v1_spiderweb.db"):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_exists()
        
    def _ensure_db_exists(self):
        """Ensure database exists and has correct schema."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables from schema
            for table_name, create_sql in SCHEMA.items():
                cursor.execute(create_sql)
            
            conn.commit()
            conn.close()
            logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
            
    def add_node(self, node_data: Dict[str, Any]) -> bool:
        """
        Add a new node to the database.
        
        Args:
            node_data: Dictionary containing node information
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO nodes (
                    node_id, name, type, status, version, 
                    config, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                node_data['node_id'],
                node_data['name'],
                node_data['type'],
                node_data['status'],
                node_data['version'],
                json.dumps(node_data.get('config', {})),
                json.dumps(node_data.get('metadata', {}))
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Added node: {node_data['node_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding node: {str(e)}")
            return False
            
    def add_connection(self, connection_data: Dict[str, Any]) -> bool:
        """
        Add a new connection between nodes.
        
        Args:
            connection_data: Dictionary containing connection information
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO connections (
                    source_id, target_id, connection_type,
                    strength, status, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                connection_data['source_id'],
                connection_data['target_id'],
                connection_data['connection_type'],
                connection_data.get('strength', 1.0),
                connection_data['status'],
                json.dumps(connection_data.get('metadata', {}))
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Added connection: {connection_data['source_id']} -> {connection_data['target_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding connection: {str(e)}")
            return False
            
    def add_metric(self, metric_data: Dict[str, Any]) -> bool:
        """
        Add a new metric measurement.
        
        Args:
            metric_data: Dictionary containing metric information
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO metrics (
                    metric_type, value, node_id,
                    connection_id, metadata
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                metric_data['metric_type'],
                metric_data['value'],
                metric_data.get('node_id'),
                metric_data.get('connection_id'),
                json.dumps(metric_data.get('metadata', {}))
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Added metric: {metric_data['metric_type']}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding metric: {str(e)}")
            return False
            
    def add_sync_event(self, event_data: Dict[str, Any]) -> bool:
        """
        Add a new synchronization event.
        
        Args:
            event_data: Dictionary containing event information
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sync_events (
                    event_type, status, source_version,
                    target_version, details, error_message
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                event_data['event_type'],
                event_data['status'],
                event_data['source_version'],
                event_data['target_version'],
                json.dumps(event_data.get('details', {})),
                event_data.get('error_message')
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Added sync event: {event_data['event_type']}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding sync event: {str(e)}")
            return False
            
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get node information by ID.
        
        Args:
            node_id: The ID of the node to retrieve
        
        Returns:
            Optional[Dict]: Node information or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM nodes WHERE node_id = ?', (node_id,))
            row = cursor.fetchone()
            
            if row:
                columns = [desc[0] for desc in cursor.description]
                node_data = dict(zip(columns, row))
                node_data['config'] = json.loads(node_data['config'])
                node_data['metadata'] = json.loads(node_data['metadata'])
                
                conn.close()
                return node_data
                
            conn.close()
            return None
            
        except Exception as e:
            logger.error(f"Error getting node: {str(e)}")
            return None
            
    def get_node_connections(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get all connections for a node.
        
        Args:
            node_id: The ID of the node
        
        Returns:
            List[Dict]: List of connection information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM connections 
                WHERE source_id = ? OR target_id = ?
            ''', (node_id, node_id))
            
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            connections = []
            for row in rows:
                conn_data = dict(zip(columns, row))
                conn_data['metadata'] = json.loads(conn_data['metadata'])
                connections.append(conn_data)
                
            conn.close()
            return connections
            
        except Exception as e:
            logger.error(f"Error getting node connections: {str(e)}")
            return []
            
    def get_metrics(self, 
                   metric_type: Optional[str] = None,
                   node_id: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get metrics based on filters.
        
        Args:
            metric_type: Optional type of metric to filter by
            node_id: Optional node ID to filter by
            start_time: Optional start time for time range
            end_time: Optional end time for time range
        
        Returns:
            List[Dict]: List of metric information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = 'SELECT * FROM metrics WHERE 1=1'
            params = []
            
            if metric_type:
                query += ' AND metric_type = ?'
                params.append(metric_type)
            
            if node_id:
                query += ' AND node_id = ?'
                params.append(node_id)
            
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time.isoformat())
            
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time.isoformat())
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            metrics = []
            for row in rows:
                metric_data = dict(zip(columns, row))
                metric_data['metadata'] = json.loads(metric_data['metadata'])
                metrics.append(metric_data)
                
            conn.close()
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return []
            
    def get_sync_events(self,
                       event_type: Optional[str] = None,
                       status: Optional[str] = None,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get sync events based on filters.
        
        Args:
            event_type: Optional type of event to filter by
            status: Optional status to filter by
            start_time: Optional start time for time range
            end_time: Optional end time for time range
        
        Returns:
            List[Dict]: List of sync event information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = 'SELECT * FROM sync_events WHERE 1=1'
            params = []
            
            if event_type:
                query += ' AND event_type = ?'
                params.append(event_type)
            
            if status:
                query += ' AND status = ?'
                params.append(status)
            
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time.isoformat())
            
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time.isoformat())
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            events = []
            for row in rows:
                event_data = dict(zip(columns, row))
                event_data['details'] = json.loads(event_data['details'])
                events.append(event_data)
                
            conn.close()
            return events
            
        except Exception as e:
            logger.error(f"Error getting sync events: {str(e)}")
            return []
            
    def update_node_status(self, node_id: str, status: str) -> bool:
        """
        Update a node's status.
        
        Args:
            node_id: The ID of the node to update
            status: The new status
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE nodes 
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE node_id = ?
            ''', (status, node_id))
            
            conn.commit()
            conn.close()
            logger.info(f"Updated node status: {node_id} -> {status}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating node status: {str(e)}")
            return False
            
    def update_connection_strength(self, connection_id: int, strength: float) -> bool:
        """
        Update a connection's strength.
        
        Args:
            connection_id: The ID of the connection to update
            strength: The new strength value
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE connections 
                SET strength = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (strength, connection_id))
            
            conn.commit()
            conn.close()
            logger.info(f"Updated connection strength: {connection_id} -> {strength}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating connection strength: {str(e)}")
            return False 