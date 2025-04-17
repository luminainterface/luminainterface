"""
Database Manager for V5 Visualization System

This module provides database persistence for the V5 system,
allowing components to save their state and retrieve it later.
"""

import os
import json
import sqlite3
import logging
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from .config import MEMORY_CONFIG, V5_DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations for V5 system persistence"""
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance of DatabaseManager"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = DatabaseManager()
            return cls._instance
    
    def __init__(self):
        """Initialize the database manager"""
        # Database file path
        self.db_dir = Path(MEMORY_CONFIG.get("storage_path", str(V5_DATA_DIR / "memory")))
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_file = self.db_dir / "v5_system.db"
        
        # Connection and cursor
        self.conn = None
        self.cursor = None
        
        # Auto-save settings
        self.auto_save_interval = MEMORY_CONFIG.get("auto_save_interval_sec", 300)
        self.auto_save_thread = None
        self.running = False
        
        # Connect to database
        self._connect()
        
        # Initialize database schema
        self._init_schema()
        
        # Start auto-save if enabled
        if self.auto_save_interval > 0:
            self.start_auto_save()
    
    def _connect(self):
        """Connect to the SQLite database"""
        try:
            self.conn = sqlite3.connect(str(self.db_file), check_same_thread=False)
            # Enable foreign keys
            self.conn.execute("PRAGMA foreign_keys = ON")
            # Use Row factory for easier column access
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database: {self.db_file}")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise
    
    def _init_schema(self):
        """Initialize database schema if not exists"""
        try:
            # Create neural_states table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS neural_states (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    state_type TEXT,
                    data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create pattern_data table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_data (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    pattern_style TEXT,
                    fractal_depth INTEGER,
                    data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create consciousness_data table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS consciousness_data (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    metrics TEXT,
                    nodes TEXT,
                    connections TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create language_memory table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS language_memory (
                    id TEXT PRIMARY KEY,
                    topic TEXT,
                    timestamp REAL,
                    content TEXT,
                    related_topics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create system_settings table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create visualization_state table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS visualization_state (
                    id TEXT PRIMARY KEY,
                    component TEXT,
                    state TEXT,
                    timestamp REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Commit schema changes
            self.conn.commit()
            logger.info("Database schema initialized")
        except sqlite3.Error as e:
            logger.error(f"Error initializing database schema: {str(e)}")
            self.conn.rollback()
            raise
    
    def start_auto_save(self):
        """Start auto-save thread"""
        if self.auto_save_thread is not None and self.auto_save_thread.is_alive():
            logger.warning("Auto-save thread already running")
            return
        
        self.running = True
        
        def auto_save_worker():
            """Worker function for auto-save thread"""
            while self.running:
                try:
                    # Commit any pending transactions
                    if self.conn:
                        self.conn.commit()
                        logger.debug(f"Auto-saved database at {datetime.now()}")
                except Exception as e:
                    logger.error(f"Error in auto-save: {str(e)}")
                finally:
                    # Sleep until next save interval
                    time.sleep(self.auto_save_interval)
        
        self.auto_save_thread = threading.Thread(target=auto_save_worker, daemon=True)
        self.auto_save_thread.start()
        logger.info(f"Started auto-save with interval {self.auto_save_interval} seconds")
    
    def stop_auto_save(self):
        """Stop auto-save thread"""
        self.running = False
        if self.auto_save_thread:
            self.auto_save_thread.join(timeout=2)
            self.auto_save_thread = None
        logger.info("Stopped auto-save")
    
    def save_neural_state(self, state_data: Dict[str, Any]) -> bool:
        """
        Save neural state data to database
        
        Args:
            state_data: Neural state data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            state_id = state_data.get("id", str(time.time()))
            timestamp = state_data.get("timestamp", time.time())
            state_type = state_data.get("state_type", "unknown")
            
            # Convert data to JSON string
            data_json = json.dumps(state_data)
            
            # Insert or replace existing record
            self.cursor.execute('''
                INSERT OR REPLACE INTO neural_states (id, timestamp, state_type, data)
                VALUES (?, ?, ?, ?)
            ''', (state_id, timestamp, state_type, data_json))
            
            # Commit changes
            self.conn.commit()
            
            return True
        except Exception as e:
            logger.error(f"Error saving neural state: {str(e)}")
            self.conn.rollback()
            return False
    
    def get_neural_state(self, state_id: str) -> Optional[Dict[str, Any]]:
        """
        Get neural state by ID
        
        Args:
            state_id: ID of the neural state
            
        Returns:
            Neural state dictionary or None if not found
        """
        try:
            self.cursor.execute('''
                SELECT * FROM neural_states WHERE id = ?
            ''', (state_id,))
            
            row = self.cursor.fetchone()
            if row:
                # Parse JSON data
                state_data = json.loads(row["data"])
                return state_data
            
            return None
        except Exception as e:
            logger.error(f"Error getting neural state: {str(e)}")
            return None
    
    def get_latest_neural_states(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get latest neural states
        
        Args:
            limit: Maximum number of states to return
            
        Returns:
            List of neural state dictionaries
        """
        try:
            self.cursor.execute('''
                SELECT * FROM neural_states
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            states = []
            for row in self.cursor.fetchall():
                # Parse JSON data
                state_data = json.loads(row["data"])
                states.append(state_data)
            
            return states
        except Exception as e:
            logger.error(f"Error getting latest neural states: {str(e)}")
            return []
    
    def save_pattern_data(self, pattern_data: Dict[str, Any]) -> bool:
        """
        Save pattern data to database
        
        Args:
            pattern_data: Pattern data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pattern_id = pattern_data.get("id", str(time.time()))
            timestamp = pattern_data.get("timestamp", time.time())
            pattern_style = pattern_data.get("pattern_style", "unknown")
            fractal_depth = pattern_data.get("fractal_depth", 0)
            
            # Convert data to JSON string
            data_json = json.dumps(pattern_data)
            
            # Insert or replace existing record
            self.cursor.execute('''
                INSERT OR REPLACE INTO pattern_data (id, timestamp, pattern_style, fractal_depth, data)
                VALUES (?, ?, ?, ?, ?)
            ''', (pattern_id, timestamp, pattern_style, fractal_depth, data_json))
            
            # Commit changes
            self.conn.commit()
            
            return True
        except Exception as e:
            logger.error(f"Error saving pattern data: {str(e)}")
            self.conn.rollback()
            return False
    
    def get_pattern_data(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Get pattern data by ID
        
        Args:
            pattern_id: ID of the pattern data
            
        Returns:
            Pattern data dictionary or None if not found
        """
        try:
            self.cursor.execute('''
                SELECT * FROM pattern_data WHERE id = ?
            ''', (pattern_id,))
            
            row = self.cursor.fetchone()
            if row:
                # Parse JSON data
                pattern_data = json.loads(row["data"])
                return pattern_data
            
            return None
        except Exception as e:
            logger.error(f"Error getting pattern data: {str(e)}")
            return None
    
    def get_latest_pattern_by_style(self, style: str) -> Optional[Dict[str, Any]]:
        """
        Get latest pattern data by style
        
        Args:
            style: Pattern style to filter by
            
        Returns:
            Pattern data dictionary or None if not found
        """
        try:
            self.cursor.execute('''
                SELECT * FROM pattern_data
                WHERE pattern_style = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (style,))
            
            row = self.cursor.fetchone()
            if row:
                # Parse JSON data
                pattern_data = json.loads(row["data"])
                return pattern_data
            
            return None
        except Exception as e:
            logger.error(f"Error getting latest pattern by style: {str(e)}")
            return None
    
    def save_consciousness_data(self, consciousness_data: Dict[str, Any]) -> bool:
        """
        Save consciousness data to database
        
        Args:
            consciousness_data: Consciousness data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data_id = consciousness_data.get("id", str(time.time()))
            timestamp = consciousness_data.get("timestamp", time.time())
            
            # Extract components
            metrics = json.dumps(consciousness_data.get("global_metrics", {}))
            nodes = json.dumps(consciousness_data.get("nodes", []))
            connections = json.dumps(consciousness_data.get("connections", []))
            
            # Insert or replace existing record
            self.cursor.execute('''
                INSERT OR REPLACE INTO consciousness_data (id, timestamp, metrics, nodes, connections)
                VALUES (?, ?, ?, ?, ?)
            ''', (data_id, timestamp, metrics, nodes, connections))
            
            # Commit changes
            self.conn.commit()
            
            return True
        except Exception as e:
            logger.error(f"Error saving consciousness data: {str(e)}")
            self.conn.rollback()
            return False
    
    def get_latest_consciousness_data(self) -> Optional[Dict[str, Any]]:
        """
        Get latest consciousness data
        
        Returns:
            Consciousness data dictionary or None if not found
        """
        try:
            self.cursor.execute('''
                SELECT * FROM consciousness_data
                ORDER BY timestamp DESC
                LIMIT 1
            ''')
            
            row = self.cursor.fetchone()
            if row:
                # Construct consciousness data
                return {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "global_metrics": json.loads(row["metrics"]),
                    "nodes": json.loads(row["nodes"]),
                    "connections": json.loads(row["connections"])
                }
            
            return None
        except Exception as e:
            logger.error(f"Error getting latest consciousness data: {str(e)}")
            return None
    
    def save_language_memory(self, memory_data: Dict[str, Any]) -> bool:
        """
        Save language memory to database
        
        Args:
            memory_data: Language memory data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            memory_id = memory_data.get("id", str(time.time()))
            topic = memory_data.get("topic", "unknown")
            timestamp = memory_data.get("timestamp", time.time())
            content = json.dumps(memory_data.get("content", {}))
            related_topics = json.dumps(memory_data.get("related_topics", []))
            
            # Insert or replace existing record
            self.cursor.execute('''
                INSERT OR REPLACE INTO language_memory (id, topic, timestamp, content, related_topics)
                VALUES (?, ?, ?, ?, ?)
            ''', (memory_id, topic, timestamp, content, related_topics))
            
            # Commit changes
            self.conn.commit()
            
            return True
        except Exception as e:
            logger.error(f"Error saving language memory: {str(e)}")
            self.conn.rollback()
            return False
    
    def get_language_memory_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """
        Get language memory by topic
        
        Args:
            topic: Topic to filter by
            
        Returns:
            List of language memory dictionaries
        """
        try:
            self.cursor.execute('''
                SELECT * FROM language_memory
                WHERE topic = ?
                ORDER BY timestamp DESC
            ''', (topic,))
            
            memories = []
            for row in self.cursor.fetchall():
                memory = {
                    "id": row["id"],
                    "topic": row["topic"],
                    "timestamp": row["timestamp"],
                    "content": json.loads(row["content"]),
                    "related_topics": json.loads(row["related_topics"])
                }
                memories.append(memory)
            
            return memories
        except Exception as e:
            logger.error(f"Error getting language memory by topic: {str(e)}")
            return []
    
    def save_visualization_state(self, component: str, state: Dict[str, Any]) -> bool:
        """
        Save visualization state for a component
        
        Args:
            component: Name of the UI component
            state: State data to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            state_id = f"{component}_{int(time.time())}"
            timestamp = time.time()
            state_json = json.dumps(state)
            
            # Insert or replace existing record
            self.cursor.execute('''
                INSERT OR REPLACE INTO visualization_state (id, component, state, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (state_id, component, state_json, timestamp))
            
            # Commit changes
            self.conn.commit()
            
            return True
        except Exception as e:
            logger.error(f"Error saving visualization state: {str(e)}")
            self.conn.rollback()
            return False
    
    def get_latest_visualization_state(self, component: str) -> Optional[Dict[str, Any]]:
        """
        Get latest visualization state for a component
        
        Args:
            component: Name of the UI component
            
        Returns:
            State data dictionary or None if not found
        """
        try:
            self.cursor.execute('''
                SELECT * FROM visualization_state
                WHERE component = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (component,))
            
            row = self.cursor.fetchone()
            if row:
                # Parse JSON data
                state_data = json.loads(row["state"])
                return state_data
            
            return None
        except Exception as e:
            logger.error(f"Error getting visualization state: {str(e)}")
            return None
    
    def save_system_setting(self, key: str, value: Any) -> bool:
        """
        Save system setting
        
        Args:
            key: Setting key
            value: Setting value (will be JSON serialized)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            value_json = json.dumps(value)
            
            # Insert or replace existing setting
            self.cursor.execute('''
                INSERT OR REPLACE INTO system_settings (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (key, value_json))
            
            # Commit changes
            self.conn.commit()
            
            return True
        except Exception as e:
            logger.error(f"Error saving system setting: {str(e)}")
            self.conn.rollback()
            return False
    
    def get_system_setting(self, key: str, default: Any = None) -> Any:
        """
        Get system setting
        
        Args:
            key: Setting key
            default: Default value if setting not found
            
        Returns:
            Setting value or default if not found
        """
        try:
            self.cursor.execute('''
                SELECT value FROM system_settings
                WHERE key = ?
            ''', (key,))
            
            row = self.cursor.fetchone()
            if row:
                # Parse JSON value
                return json.loads(row["value"])
            
            return default
        except Exception as e:
            logger.error(f"Error getting system setting: {str(e)}")
            return default
    
    def vacuum_database(self) -> bool:
        """
        Optimize database by removing unused space
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.cursor.execute("VACUUM")
            logger.info("Database optimized (VACUUM completed)")
            return True
        except Exception as e:
            logger.error(f"Error optimizing database: {str(e)}")
            return False
    
    def close(self):
        """Close database connection"""
        try:
            # Stop auto-save thread
            self.stop_auto_save()
            
            # Commit any pending changes
            if self.conn:
                self.conn.commit()
                
            # Close cursor and connection
            if self.cursor:
                self.cursor.close()
                
            if self.conn:
                self.conn.close()
                
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure database is closed"""
        self.close() 