"""
Bridge connector for integrating the V1 database with the Spiderweb Bridge.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .database.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class BridgeConnector:
    """Connects the V1 database to the Spiderweb Bridge system."""
    
    def __init__(self, db_path: str = "v1_spiderweb.db"):
        """
        Initialize the bridge connector.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db = DatabaseManager(db_path)
        self.active_nodes = {}
        self.active_connections = {}
        
    def handle_node_creation(self, node_data: Dict[str, Any]) -> bool:
        """
        Handle node creation event from Spiderweb Bridge.
        
        Args:
            node_data: Dictionary containing node information
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Add node to database
            if self.db.add_node(node_data):
                # Track active node
                self.active_nodes[node_data['node_id']] = node_data
                
                # Log sync event
                self.db.add_sync_event({
                    'event_type': 'node_creation',
                    'status': 'success',
                    'source_version': node_data['version'],
                    'target_version': 'v1',
                    'details': {'node_id': node_data['node_id']}
                })
                
                logger.info(f"Node creation handled: {node_data['node_id']}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error handling node creation: {str(e)}")
            return False
            
    def handle_connection_creation(self, connection_data: Dict[str, Any]) -> bool:
        """
        Handle connection creation event from Spiderweb Bridge.
        
        Args:
            connection_data: Dictionary containing connection information
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Add connection to database
            if self.db.add_connection(connection_data):
                # Track active connection
                conn_key = f"{connection_data['source_id']}->{connection_data['target_id']}"
                self.active_connections[conn_key] = connection_data
                
                # Log sync event
                self.db.add_sync_event({
                    'event_type': 'connection_creation',
                    'status': 'success',
                    'source_version': 'v1',
                    'target_version': 'v1',
                    'details': {
                        'source_id': connection_data['source_id'],
                        'target_id': connection_data['target_id']
                    }
                })
                
                logger.info(f"Connection creation handled: {conn_key}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error handling connection creation: {str(e)}")
            return False
            
    def handle_metric_update(self, metric_data: Dict[str, Any]) -> bool:
        """
        Handle metric update event from Spiderweb Bridge.
        
        Args:
            metric_data: Dictionary containing metric information
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Add metric to database
            if self.db.add_metric(metric_data):
                # Log sync event if it's a significant metric
                if metric_data.get('significant', False):
                    self.db.add_sync_event({
                        'event_type': 'metric_update',
                        'status': 'success',
                        'source_version': 'v1',
                        'target_version': 'v1',
                        'details': {
                            'metric_type': metric_data['metric_type'],
                            'value': metric_data['value']
                        }
                    })
                
                logger.info(f"Metric update handled: {metric_data['metric_type']}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error handling metric update: {str(e)}")
            return False
            
    def handle_node_status_change(self, node_id: str, status: str) -> bool:
        """
        Handle node status change event from Spiderweb Bridge.
        
        Args:
            node_id: The ID of the node
            status: The new status
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Update node status in database
            if self.db.update_node_status(node_id, status):
                # Update active nodes tracking
                if node_id in self.active_nodes:
                    self.active_nodes[node_id]['status'] = status
                
                # Log sync event
                self.db.add_sync_event({
                    'event_type': 'node_status_change',
                    'status': 'success',
                    'source_version': 'v1',
                    'target_version': 'v1',
                    'details': {
                        'node_id': node_id,
                        'new_status': status
                    }
                })
                
                logger.info(f"Node status change handled: {node_id} -> {status}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error handling node status change: {str(e)}")
            return False
            
    def handle_connection_strength_update(self, connection_id: int, strength: float) -> bool:
        """
        Handle connection strength update event from Spiderweb Bridge.
        
        Args:
            connection_id: The ID of the connection
            strength: The new strength value
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Update connection strength in database
            if self.db.update_connection_strength(connection_id, strength):
                # Log sync event
                self.db.add_sync_event({
                    'event_type': 'connection_strength_update',
                    'status': 'success',
                    'source_version': 'v1',
                    'target_version': 'v1',
                    'details': {
                        'connection_id': connection_id,
                        'new_strength': strength
                    }
                })
                
                logger.info(f"Connection strength update handled: {connection_id} -> {strength}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error handling connection strength update: {str(e)}")
            return False
            
    def get_node_metrics(self, node_id: str, 
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get metrics for a specific node.
        
        Args:
            node_id: The ID of the node
            start_time: Optional start time for time range
            end_time: Optional end time for time range
        
        Returns:
            List[Dict]: List of metric information
        """
        return self.db.get_metrics(node_id=node_id, start_time=start_time, end_time=end_time)
        
    def get_sync_history(self, 
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get synchronization event history.
        
        Args:
            start_time: Optional start time for time range
            end_time: Optional end time for time range
        
        Returns:
            List[Dict]: List of sync event information
        """
        return self.db.get_sync_events(start_time=start_time, end_time=end_time)
        
    def get_active_nodes(self) -> Dict[str, Dict[str, Any]]:
        """
        Get currently active nodes.
        
        Returns:
            Dict[str, Dict]: Dictionary of active node information
        """
        return self.active_nodes
        
    def get_active_connections(self) -> Dict[str, Dict[str, Any]]:
        """
        Get currently active connections.
        
        Returns:
            Dict[str, Dict]: Dictionary of active connection information
        """
        return self.active_connections 