import sqlite3
import json
import logging
from pathlib import Path
import time
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DatabaseManager:
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
        # Database file path - use absolute path for consistency
        self.db_path = Path("data/node_zero.db").resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        self.conn = None
        self.cursor = None
        
        # Connect to database
        self._connect()
        
        # Initialize database schema
        self._init_db()
        
        logger.info(f"Database initialized at {self.db_path}")
        
    def _connect(self):
        """Connect to the SQLite database"""
        try:
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            # Enable foreign keys
            self.conn.execute("PRAGMA foreign_keys = ON")
            # Use Row factory for easier column access
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise
            
    def _init_db(self):
        """Initialize the database schema"""
        try:
            # Create tables if they don't exist
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS neural_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    state_data TEXT NOT NULL,
                    version TEXT,
                    metadata TEXT
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS network_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    metric_type TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metadata TEXT
                )
            ''')
            
            # Create indices for better performance
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_neural_states_timestamp ON neural_states(timestamp)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_network_metrics_timestamp ON network_metrics(timestamp)')
            
            self.conn.commit()
            logger.info("Database schema initialized successfully")
        except sqlite3.Error as e:
            logger.error(f"Error initializing database schema: {e}")
            raise
            
    def save_neural_state(self, state_data: dict) -> bool:
        """Save neural state to database"""
        try:
            with self.conn:
                self.cursor.execute("""
                    INSERT OR REPLACE INTO neural_states 
                    (id, timestamp, state_type, metrics, version_data)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    state_data['id'],
                    state_data['timestamp'],
                    state_data['state_type'],
                    json.dumps(state_data.get('metrics', {})),
                    json.dumps(state_data.get('version_data', {}))
                ))
                
                logger.info(f"Saved neural state: {state_data['id']}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving neural state: {str(e)}")
            return False
            
    def get_neural_state(self, state_id: str) -> Optional[Dict]:
        """Retrieve neural state from database"""
        try:
            with self.conn:
                self.cursor.execute("""
                    SELECT * FROM neural_states WHERE id = ?
                """, (state_id,))
                
                row = self.cursor.fetchone()
                if row:
                    return {
                        'id': row['id'],
                        'timestamp': row['timestamp'],
                        'state_type': row['state_type'],
                        'metrics': json.loads(row['metrics']),
                        'version_data': json.loads(row['version_data'])
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving neural state: {str(e)}")
            return None
            
    def delete_neural_state(self, state_id: str) -> bool:
        """Delete neural state from database"""
        try:
            with self.conn:
                self.cursor.execute("""
                    DELETE FROM neural_states WHERE id = ?
                """, (state_id,))
                
                logger.info(f"Deleted neural state: {state_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting neural state: {str(e)}")
            return False
            
    def get_all_states(self) -> list:
        """Get all neural states"""
        try:
            with self.conn:
                self.cursor.execute("SELECT * FROM neural_states")
                
                states = []
                for row in self.cursor.fetchall():
                    states.append({
                        'id': row['id'],
                        'timestamp': row['timestamp'],
                        'state_type': row['state_type'],
                        'metrics': json.loads(row['metrics']),
                        'version_data': json.loads(row['version_data'])
                    })
                return states
                
        except Exception as e:
            logger.error(f"Error retrieving all states: {str(e)}")
            return []
            
    def get_system_setting(self, key: str, default: Any = None) -> Any:
        """Get system setting"""
        try:
            with self.conn:
                self.cursor.execute("""
                    SELECT value FROM system_settings
                    WHERE key = ?
                """, (key,))
                
                row = self.cursor.fetchone()
                if row:
                    return json.loads(row['value'])
                
                return default
        except Exception as e:
            logger.error(f"Error getting system setting: {str(e)}")
            return default
            
    def set_system_setting(self, key: str, value: Any) -> bool:
        """Set system setting"""
        try:
            with self.conn:
                self.cursor.execute("""
                    INSERT OR REPLACE INTO system_settings
                    (key, value) VALUES (?, ?)
                """, (key, json.dumps(value)))
                
                logger.info(f"Set system setting: {key}")
                return True
        except Exception as e:
            logger.error(f"Error setting system setting: {str(e)}")
            return False
            
    def close(self):
        """Close database connection"""
        try:
            if self.cursor:
                self.cursor.close()
                
            if self.conn:
                self.conn.close()
                
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database: {str(e)}")
            
    def __del__(self):
        """Ensure database connection is closed on deletion"""
        self.close() 