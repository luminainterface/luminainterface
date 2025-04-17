#!/usr/bin/env python3
"""
Knowledge Database System for v8

This module implements a database system for storing, retrieving, and managing
knowledge within the v8 knowledge CI/CD pipeline. It provides:

1. Database schema creation and management
2. Query interfaces for knowledge retrieval
3. Transaction management
4. Backup and recovery mechanisms
5. Integration with the CI/CD pipeline stages
"""

import os
import sys
import time
import json
import sqlite3
import logging
import threading
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Union

# Add parent directory to path for imports
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/knowledge_db_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("v8.knowledge_database")

class KnowledgeDatabase:
    """
    Main database interface for the Knowledge CI/CD system.
    Manages connections, transactions, and provides query methods.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the knowledge database system"""
        self.config = {
            "main_db": {
                "type": "sqlite",
                "path": "data/knowledge_db.sqlite",
                "backup_path": "data/backups/knowledge_db_backup.sqlite",
                "backup_frequency": 24,
                "vacuum_frequency": 168
            },
            "metrics_db": {
                "type": "sqlite",
                "path": "data/metrics_db.sqlite",
                "backup_frequency": 48
            }
        }
        
        # Load config if provided
        if config_path:
            self._load_config(config_path)
        
        # Initialize connections
        self.main_conn = None
        self.metrics_conn = None
        self.lock = threading.RLock()
        
        # Track if tables have been created
        self.tables_created = False
        
        # Connect to databases
        self._connect()
    
    def _load_config(self, config_path: str):
        """Load database configuration from file"""
        try:
            with open(config_path, 'r') as f:
                full_config = json.load(f)
                if 'database' in full_config:
                    self.config.update(full_config['database'])
                    logger.info(f"Loaded database configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    def _connect(self):
        """Connect to the databases"""
        try:
            # Connect to main knowledge database
            self.main_conn = sqlite3.connect(
                self.config["main_db"]["path"], 
                check_same_thread=False
            )
            self.main_conn.row_factory = sqlite3.Row
            
            # Connect to metrics database
            self.metrics_conn = sqlite3.connect(
                self.config["metrics_db"]["path"],
                check_same_thread=False
            )
            self.metrics_conn.row_factory = sqlite3.Row
            
            logger.info("Connected to knowledge and metrics databases")
            
            # Create tables if they don't exist
            if not self.tables_created:
                self._create_tables()
                self.tables_created = True
            
            return True
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return False
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        try:
            # Create tables in main knowledge database
            main_cursor = self.main_conn.cursor()
            
            # Concepts table - stores core knowledge entities
            main_cursor.execute('''
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                weight REAL DEFAULT 0.5,
                creation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source TEXT,
                verified INTEGER DEFAULT 0
            )
            ''')
            
            # Connections table - links between concepts
            main_cursor.execute('''
            CREATE TABLE IF NOT EXISTS connections (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                weight REAL DEFAULT 0.5,
                connection_type TEXT NOT NULL,
                creation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                bidirectional INTEGER DEFAULT 0,
                FOREIGN KEY (source_id) REFERENCES concepts(id),
                FOREIGN KEY (target_id) REFERENCES concepts(id)
            )
            ''')
            
            # Knowledge sources table
            main_cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_sources (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                url TEXT,
                description TEXT,
                discovery_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP,
                reliability_score REAL DEFAULT 0.5
            )
            ''')
            
            # Attachments table - links concepts to sources
            main_cursor.execute('''
            CREATE TABLE IF NOT EXISTS attachments (
                id TEXT PRIMARY KEY,
                concept_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                similarity_score REAL,
                attachment_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (concept_id) REFERENCES concepts(id),
                FOREIGN KEY (source_id) REFERENCES knowledge_sources(id)
            )
            ''')
            
            # Spatial nodes table - for organizing concepts in spatial temple
            main_cursor.execute('''
            CREATE TABLE IF NOT EXISTS spatial_nodes (
                id TEXT PRIMARY KEY,
                concept_id TEXT NOT NULL,
                zone_id TEXT,
                x_coord REAL,
                y_coord REAL,
                z_coord REAL,
                node_type TEXT,
                creation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (concept_id) REFERENCES concepts(id)
            )
            ''')
            
            # Create indexes
            main_cursor.execute('CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts(name)')
            main_cursor.execute('CREATE INDEX IF NOT EXISTS idx_connections_source ON connections(source_id)')
            main_cursor.execute('CREATE INDEX IF NOT EXISTS idx_connections_target ON connections(target_id)')
            main_cursor.execute('CREATE INDEX IF NOT EXISTS idx_spatial_zone ON spatial_nodes(zone_id)')
            
            # Create tables in metrics database
            metrics_cursor = self.metrics_conn.cursor()
            
            # Pipeline metrics table
            metrics_cursor.execute('''
            CREATE TABLE IF NOT EXISTS pipeline_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                discoveries INTEGER DEFAULT 0,
                attachments INTEGER DEFAULT 0,
                growth_points INTEGER DEFAULT 0,
                bidirectional_links INTEGER DEFAULT 0,
                deployments INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0,
                status TEXT
            )
            ''')
            
            # Stage metrics table
            metrics_cursor.execute('''
            CREATE TABLE IF NOT EXISTS stage_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                stage_name TEXT NOT NULL,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                items_processed INTEGER DEFAULT 0,
                items_succeeded INTEGER DEFAULT 0,
                items_failed INTEGER DEFAULT 0,
                status TEXT
            )
            ''')
            
            # Knowledge growth metrics
            metrics_cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_concepts INTEGER DEFAULT 0,
                total_connections INTEGER DEFAULT 0,
                total_sources INTEGER DEFAULT 0,
                zones_count INTEGER DEFAULT 0,
                average_connections_per_concept REAL DEFAULT 0,
                top_zone_id TEXT,
                top_zone_size INTEGER DEFAULT 0
            )
            ''')
            
            # Commit changes
            self.main_conn.commit()
            self.metrics_conn.commit()
            
            logger.info("Database tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            return False
    
    def backup_database(self, force: bool = False):
        """Create a backup of the knowledge database"""
        try:
            # Get path information
            db_path = self.config["main_db"]["path"]
            backup_path = self.config["main_db"]["backup_path"]
            
            # Check if backup is needed based on frequency setting
            if not force:
                try:
                    last_backup_time = os.path.getmtime(backup_path)
                    backup_age_hours = (time.time() - last_backup_time) / 3600
                    
                    if backup_age_hours < self.config["main_db"]["backup_frequency"]:
                        logger.info(f"Skipping backup, last backup is {backup_age_hours:.1f} hours old")
                        return True
                except FileNotFoundError:
                    # No previous backup exists, continue with backup
                    pass
            
            # Ensure database is in consistent state
            with self.lock:
                self.main_conn.commit()
                # Create backup directory if it doesn't exist
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                # Copy database file to backup location
                shutil.copy2(db_path, backup_path)
            
            logger.info(f"Database backup created at {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Database backup error: {e}")
            return False
    
    def vacuum_database(self):
        """Perform maintenance on the database to optimize performance"""
        try:
            with self.lock:
                # Vacuum the main database
                self.main_conn.execute("VACUUM")
                # Vacuum the metrics database
                self.metrics_conn.execute("VACUUM")
                
            logger.info("Database vacuum completed successfully")
            return True
        except Exception as e:
            logger.error(f"Database vacuum error: {e}")
            return False
    
    def begin_transaction(self):
        """Begin a new transaction for atomic operations"""
        self.main_conn.execute("BEGIN TRANSACTION")
    
    def commit_transaction(self):
        """Commit the current transaction"""
        self.main_conn.commit()
    
    def rollback_transaction(self):
        """Rollback the current transaction in case of errors"""
        self.main_conn.rollback()
    
    def _ensure_connected(self) -> bool:
        """Ensure the database connections are alive, reconnect if needed"""
        with self.lock:
            try:
                # Check main connection
                if not self.main_conn:
                    logger.warning("Main database connection is missing, reconnecting...")
                    reconnected = self._connect()
                    return reconnected
                
                # Test main connection
                try:
                    self.main_conn.execute("SELECT 1")
                except Exception as e:
                    logger.warning(f"Main database connection lost: {e}, reconnecting...")
                    self.close()
                    reconnected = self._connect()
                    return reconnected
                
                # Check metrics connection
                if not self.metrics_conn:
                    logger.warning("Metrics database connection is missing, reconnecting...")
                    reconnected = self._connect()
                    return reconnected
                
                # Test metrics connection
                try:
                    self.metrics_conn.execute("SELECT 1")
                except Exception as e:
                    logger.warning(f"Metrics database connection lost: {e}, reconnecting...")
                    self.close()
                    reconnected = self._connect()
                    return reconnected
                
                return True
            except Exception as e:
                logger.error(f"Error checking database connections: {e}")
                try:
                    self.close()
                    reconnected = self._connect()
                    return reconnected
                except Exception as reconnect_error:
                    logger.error(f"Failed to reconnect to databases: {reconnect_error}")
                    return False
    
    def add_concept(self, concept_id: str, name: str, description: str = "", 
                   weight: float = 0.5, source: str = None) -> bool:
        """Add a concept to the knowledge database"""
        if not self._ensure_connected():
            return False
            
        try:
            with self.lock:
                # Begin transaction
                self.main_conn.execute("BEGIN TRANSACTION")
                
                cursor = self.main_conn.cursor()
                cursor.execute('''
                INSERT INTO concepts (id, name, description, weight, source)
                VALUES (?, ?, ?, ?, ?)
                ''', (concept_id, name, description, weight, source))
                
                # Commit transaction
                self.main_conn.commit()
                
            return True
        except Exception as e:
            logger.error(f"Error adding concept: {e}")
            
            # Rollback transaction
            try:
                self.main_conn.rollback()
            except Exception as rollback_error:
                logger.error(f"Error during rollback: {rollback_error}")
                
            return False
    
    def add_connection(self, connection_id: str, source_id: str, target_id: str,
                      weight: float = 0.5, connection_type: str = "related",
                      bidirectional: bool = False) -> bool:
        """Add a new connection between concepts"""
        try:
            with self.lock:
                cursor = self.main_conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO connections
                (id, source_id, target_id, weight, connection_type, creation_time, 
                last_updated, bidirectional)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                ''', (connection_id, source_id, target_id, weight, connection_type, 
                     1 if bidirectional else 0))
                self.main_conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding connection: {e}")
            return False
    
    def add_knowledge_source(self, source_id: str, name: str, source_type: str,
                           url: str = None, description: str = None,
                           reliability_score: float = 0.5) -> bool:
        """Add a new knowledge source to the database"""
        try:
            with self.lock:
                cursor = self.main_conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO knowledge_sources
                (id, name, source_type, url, description, discovery_time, reliability_score)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                ''', (source_id, name, source_type, url, description, reliability_score))
                self.main_conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding knowledge source: {e}")
            return False
    
    def add_attachment(self, attachment_id: str, concept_id: str, source_id: str,
                     similarity_score: float = None) -> bool:
        """Add an attachment between a concept and a knowledge source"""
        try:
            with self.lock:
                cursor = self.main_conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO attachments
                (id, concept_id, source_id, similarity_score, attachment_time)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (attachment_id, concept_id, source_id, similarity_score))
                self.main_conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding attachment: {e}")
            return False
    
    def add_spatial_node(self, node_id: str, concept_id: str, zone_id: str = None,
                        x_coord: float = 0.0, y_coord: float = 0.0, z_coord: float = 0.0,
                        node_type: str = "concept") -> bool:
        """Add a spatial node for temple visualization"""
        try:
            with self.lock:
                cursor = self.main_conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO spatial_nodes
                (id, concept_id, zone_id, x_coord, y_coord, z_coord, node_type,
                creation_time, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ''', (node_id, concept_id, zone_id, x_coord, y_coord, z_coord, node_type))
                self.main_conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding spatial node: {e}")
            return False
    
    def get_concept(self, concept_id: str) -> Dict:
        """Retrieve a concept by ID"""
        try:
            cursor = self.main_conn.cursor()
            cursor.execute("SELECT * FROM concepts WHERE id = ?", (concept_id,))
            result = cursor.fetchone()
            if result:
                return dict(result)
            return None
        except Exception as e:
            logger.error(f"Error retrieving concept: {e}")
            return None
    
    def get_concepts_by_name(self, name_pattern: str) -> List[Dict]:
        """Search for concepts by name pattern"""
        try:
            cursor = self.main_conn.cursor()
            cursor.execute("SELECT * FROM concepts WHERE name LIKE ?", (f"%{name_pattern}%",))
            results = cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error searching concepts: {e}")
            return []
    
    def get_connections(self, concept_id: str, direction: str = "outgoing") -> List[Dict]:
        """Get connections for a concept (outgoing, incoming, or both)"""
        try:
            cursor = self.main_conn.cursor()
            if direction == "outgoing":
                cursor.execute('''
                SELECT c.*, target.name as target_name 
                FROM connections c
                JOIN concepts target ON c.target_id = target.id
                WHERE c.source_id = ?
                ''', (concept_id,))
            elif direction == "incoming":
                cursor.execute('''
                SELECT c.*, source.name as source_name 
                FROM connections c
                JOIN concepts source ON c.source_id = source.id
                WHERE c.target_id = ?
                ''', (concept_id,))
            else:  # both
                cursor.execute('''
                SELECT c.*, 
                       s.name as source_name,
                       t.name as target_name
                FROM connections c
                JOIN concepts s ON c.source_id = s.id
                JOIN concepts t ON c.target_id = t.id
                WHERE c.source_id = ? OR c.target_id = ?
                ''', (concept_id, concept_id))
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error retrieving connections: {e}")
            return []
    
    def get_knowledge_sources(self, concept_id: str = None) -> List[Dict]:
        """Get knowledge sources, optionally filtered by attached concept"""
        try:
            cursor = self.main_conn.cursor()
            if concept_id:
                cursor.execute('''
                SELECT ks.*, a.similarity_score
                FROM knowledge_sources ks
                JOIN attachments a ON ks.id = a.source_id
                WHERE a.concept_id = ?
                ''', (concept_id,))
            else:
                cursor.execute("SELECT * FROM knowledge_sources")
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error retrieving knowledge sources: {e}")
            return []
    
    def record_pipeline_run(self, run_id: str, metrics: Dict[str, Any]) -> bool:
        """Record metrics for a complete pipeline run"""
        try:
            with self.lock:
                cursor = self.metrics_conn.cursor()
                cursor.execute('''
                INSERT INTO pipeline_metrics
                (run_id, start_time, end_time, discoveries, attachments, 
                growth_points, bidirectional_links, deployments, errors, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    run_id,
                    metrics.get("start_time"),
                    metrics.get("end_time"),
                    metrics.get("discoveries", 0),
                    metrics.get("attachments", 0),
                    metrics.get("growth_points", 0),
                    metrics.get("bidirectional_links", 0),
                    metrics.get("deployments", 0),
                    metrics.get("errors", 0),
                    metrics.get("status", "completed")
                ))
                self.metrics_conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error recording pipeline metrics: {e}")
            return False
    
    def record_stage_metrics(self, run_id: str, stage_name: str, metrics: Dict[str, Any]) -> bool:
        """Record metrics for a pipeline stage run"""
        if not self.metrics_conn:
            logger.warning("Cannot record stage metrics: No metrics database connection")
            return False
            
        # Validate run_id
        if not run_id:
            logger.warning("Cannot record stage metrics: No run_id provided")
            run_id = f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Created fallback run_id: {run_id}")
            
        try:
            metrics_cursor = self.metrics_conn.cursor()
            
            with self.lock:
                # Begin transaction
                self.metrics_conn.execute("BEGIN TRANSACTION")
                
                # Insert stage metrics
                metrics_cursor.execute('''
                INSERT INTO stage_metrics (
                    run_id, stage_name, start_time, end_time, 
                    items_processed, items_succeeded, items_failed, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    run_id,
                    stage_name,
                    metrics.get("start_time", datetime.now().isoformat()),
                    metrics.get("end_time", datetime.now().isoformat()),
                    metrics.get("items_processed", 0),
                    metrics.get("items_succeeded", 0),
                    metrics.get("items_failed", 0),
                    metrics.get("status", "unknown")
                ))
                
                # Commit transaction
                self.metrics_conn.commit()
                
            return True
        except Exception as e:
            logger.error(f"Error recording stage metrics: {e}")
            
            # Rollback transaction
            try:
                self.metrics_conn.rollback()
            except Exception as rollback_error:
                logger.error(f"Error during rollback: {rollback_error}")
                
            return False
    
    def update_knowledge_metrics(self) -> bool:
        """Calculate and record current knowledge metrics"""
        try:
            with self.lock:
                # Get counts from main database
                main_cursor = self.main_conn.cursor()
                
                main_cursor.execute("SELECT COUNT(*) FROM concepts")
                total_concepts = main_cursor.fetchone()[0]
                
                main_cursor.execute("SELECT COUNT(*) FROM connections")
                total_connections = main_cursor.fetchone()[0]
                
                main_cursor.execute("SELECT COUNT(*) FROM knowledge_sources")
                total_sources = main_cursor.fetchone()[0]
                
                # Calculate average connections per concept
                avg_connections = 0
                if total_concepts > 0:
                    avg_connections = total_connections / total_concepts
                
                # Get zone information
                main_cursor.execute("""
                SELECT zone_id, COUNT(*) as zone_size
                FROM spatial_nodes
                WHERE zone_id IS NOT NULL
                GROUP BY zone_id
                ORDER BY zone_size DESC
                LIMIT 1
                """)
                zone_result = main_cursor.fetchone()
                top_zone_id = None
                top_zone_size = 0
                
                if zone_result:
                    top_zone_id = zone_result[0]
                    top_zone_size = zone_result[1]
                
                # Count total zones
                main_cursor.execute("""
                SELECT COUNT(DISTINCT zone_id)
                FROM spatial_nodes
                WHERE zone_id IS NOT NULL
                """)
                zones_count = main_cursor.fetchone()[0]
                
                # Record metrics
                metrics_cursor = self.metrics_conn.cursor()
                metrics_cursor.execute('''
                INSERT INTO knowledge_metrics
                (timestamp, total_concepts, total_connections, total_sources,
                zones_count, average_connections_per_concept, top_zone_id, top_zone_size)
                VALUES (CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    total_concepts,
                    total_connections,
                    total_sources,
                    zones_count,
                    avg_connections,
                    top_zone_id,
                    top_zone_size
                ))
                
                self.metrics_conn.commit()
                
                return True
        except Exception as e:
            logger.error(f"Error updating knowledge metrics: {e}")
            return False
    
    def get_recent_metrics(self, limit: int = 30) -> List[Dict]:
        """Get recent knowledge metrics"""
        try:
            cursor = self.metrics_conn.cursor()
            cursor.execute("""
            SELECT * FROM knowledge_metrics
            ORDER BY timestamp DESC
            LIMIT ?
            """, (limit,))
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error retrieving recent metrics: {e}")
            return []
    
    def close(self):
        """Close database connections"""
        if self.main_conn:
            self.main_conn.close()
        if self.metrics_conn:
            self.metrics_conn.close()
        logger.info("Database connections closed")


# Testing code
if __name__ == "__main__":
    # Initialize database
    config_path = os.path.join(project_root, "config", "knowledge_ci_cd.json")
    db = KnowledgeDatabase(config_path)
    
    # Test backup
    db.backup_database(force=True)
    
    # Test database operations
    concept_id = f"concept_{int(time.time())}"
    db.add_concept(concept_id, "Test Concept", "A test concept for database validation")
    
    # Verify concept was added
    concept = db.get_concept(concept_id)
    print(f"Added concept: {concept}")
    
    # Add a connection
    connection_id = f"conn_{int(time.time())}"
    db.add_connection(connection_id, concept_id, concept_id, connection_type="self-reference")
    
    # Update knowledge metrics
    db.update_knowledge_metrics()
    
    # Get recent metrics
    metrics = db.get_recent_metrics(1)
    print(f"Current metrics: {metrics}")
    
    # Clean up
    db.vacuum_database()
    db.close() 