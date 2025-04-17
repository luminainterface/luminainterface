"""
Enhanced database manager for the V2 Spiderweb Bridge.
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import numpy as np

from .schema import SCHEMA

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations for the V2 Spiderweb Bridge."""
    
    def __init__(self, db_path: str = "v2_spiderweb.db"):
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
                    config, metadata, consciousness_level,
                    energy_level, stability_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                node_data['node_id'],
                node_data['name'],
                node_data['type'],
                node_data['status'],
                node_data['version'],
                json.dumps(node_data.get('config', {})),
                json.dumps(node_data.get('metadata', {})),
                node_data.get('consciousness_level', 0.0),
                node_data.get('energy_level', 1.0),
                node_data.get('stability_score', 1.0)
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Added node: {node_data['node_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding node: {str(e)}")
            return False
            
    def update_node_quantum_state(self, node_id: str, quantum_data: Dict[str, Any]) -> bool:
        """
        Update a node's quantum state.
        
        Args:
            node_id: The ID of the node
            quantum_data: Dictionary containing quantum state information
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO quantum_states (
                    node_id, state_vector, entanglement_map,
                    coherence_level, decoherence_rate,
                    measurement_basis, collapse_probability
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                node_id,
                json.dumps(quantum_data['state_vector']),
                json.dumps(quantum_data.get('entanglement_map', {})),
                quantum_data.get('coherence_level', 1.0),
                quantum_data.get('decoherence_rate', 0.0),
                quantum_data.get('measurement_basis', 'computational'),
                quantum_data.get('collapse_probability', 0.0)
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Updated quantum state for node: {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating quantum state: {str(e)}")
            return False
            
    def update_node_cosmic_state(self, node_id: str, cosmic_data: Dict[str, Any]) -> bool:
        """
        Update a node's cosmic state.
        
        Args:
            node_id: The ID of the node
            cosmic_data: Dictionary containing cosmic state information
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO cosmic_states (
                    node_id, dimensional_signature, resonance_pattern,
                    universal_phase, cosmic_frequency,
                    stability_matrix, harmonic_index
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                node_id,
                json.dumps(cosmic_data['dimensional_signature']),
                json.dumps(cosmic_data.get('resonance_pattern', {})),
                cosmic_data.get('universal_phase', 0.0),
                cosmic_data.get('cosmic_frequency', 0.0),
                json.dumps(cosmic_data.get('stability_matrix', {})),
                cosmic_data.get('harmonic_index', 1.0)
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Updated cosmic state for node: {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating cosmic state: {str(e)}")
            return False
            
    def add_node_relationship(self, relationship_data: Dict[str, Any]) -> bool:
        """
        Add a relationship between nodes.
        
        Args:
            relationship_data: Dictionary containing relationship information
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO node_relationships (
                    source_node_id, target_node_id, relationship_type,
                    strength, metadata, sync_frequency,
                    mutual_influence_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                relationship_data['source_node_id'],
                relationship_data['target_node_id'],
                relationship_data['relationship_type'],
                relationship_data.get('strength', 1.0),
                json.dumps(relationship_data.get('metadata', {})),
                relationship_data.get('sync_frequency', 1.0),
                relationship_data.get('mutual_influence_score', 0.0)
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Added relationship: {relationship_data['source_node_id']} -> {relationship_data['target_node_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding relationship: {str(e)}")
            return False
            
    def add_performance_metric(self, metric_data: Dict[str, Any]) -> bool:
        """
        Add a performance metric.
        
        Args:
            metric_data: Dictionary containing metric information
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics (
                    metric_name, value, component_id,
                    component_type, aggregation_window,
                    threshold_value, alert_level, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metric_data['metric_name'],
                metric_data['value'],
                metric_data.get('component_id'),
                metric_data.get('component_type'),
                metric_data.get('aggregation_window'),
                metric_data.get('threshold_value'),
                metric_data.get('alert_level', 'normal'),
                json.dumps(metric_data.get('metadata', {}))
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Added performance metric: {metric_data['metric_name']}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding performance metric: {str(e)}")
            return False
            
    def get_node_quantum_state(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest quantum state for a node.
        
        Args:
            node_id: The ID of the node
        
        Returns:
            Optional[Dict]: Quantum state information or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM quantum_states
                WHERE node_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (node_id,))
            
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                state_data = dict(zip(columns, row))
                state_data['state_vector'] = json.loads(state_data['state_vector'])
                state_data['entanglement_map'] = json.loads(state_data['entanglement_map'])
                
                conn.close()
                return state_data
                
            conn.close()
            return None
            
        except Exception as e:
            logger.error(f"Error getting quantum state: {str(e)}")
            return None
            
    def get_node_cosmic_state(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest cosmic state for a node.
        
        Args:
            node_id: The ID of the node
        
        Returns:
            Optional[Dict]: Cosmic state information or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM cosmic_states
                WHERE node_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (node_id,))
            
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                state_data = dict(zip(columns, row))
                state_data['dimensional_signature'] = json.loads(state_data['dimensional_signature'])
                state_data['resonance_pattern'] = json.loads(state_data['resonance_pattern'])
                state_data['stability_matrix'] = json.loads(state_data['stability_matrix'])
                
                conn.close()
                return state_data
                
            conn.close()
            return None
            
        except Exception as e:
            logger.error(f"Error getting cosmic state: {str(e)}")
            return None
            
    def get_node_relationships(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get all relationships for a node.
        
        Args:
            node_id: The ID of the node
        
        Returns:
            List[Dict]: List of relationship information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM node_relationships
                WHERE source_node_id = ? OR target_node_id = ?
            ''', (node_id, node_id))
            
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            relationships = []
            for row in rows:
                rel_data = dict(zip(columns, row))
                rel_data['metadata'] = json.loads(rel_data['metadata'])
                relationships.append(rel_data)
                
            conn.close()
            return relationships
            
        except Exception as e:
            logger.error(f"Error getting node relationships: {str(e)}")
            return []
            
    def get_performance_metrics(self,
                              component_id: Optional[str] = None,
                              metric_name: Optional[str] = None,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get performance metrics based on filters.
        
        Args:
            component_id: Optional component ID to filter by
            metric_name: Optional metric name to filter by
            start_time: Optional start time for time range
            end_time: Optional end time for time range
        
        Returns:
            List[Dict]: List of performance metric information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = 'SELECT * FROM performance_metrics WHERE 1=1'
            params = []
            
            if component_id:
                query += ' AND component_id = ?'
                params.append(component_id)
            
            if metric_name:
                query += ' AND metric_name = ?'
                params.append(metric_name)
            
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
            logger.error(f"Error getting performance metrics: {str(e)}")
            return [] 