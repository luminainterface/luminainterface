"""
Spiderweb V3 Database Manager
Handles advanced state management, temporal tracking, and optimization features.
"""

import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpiderwebDBV3:
    def __init__(self, db_path: str = "spiderweb_v3.db"):
        """Initialize the V3 database manager."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._initialize_db()

    def _initialize_db(self):
        """Initialize database connection and create tables if they don't exist."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            self._create_tables()
            logger.info("Database initialized successfully")
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def _create_tables(self):
        """Create necessary database tables."""
        try:
            # Temporal States Table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS temporal_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    state_hash TEXT NOT NULL,
                    previous_state_hash TEXT,
                    state_type TEXT NOT NULL,
                    state_data TEXT NOT NULL,
                    version INTEGER DEFAULT 3,
                    FOREIGN KEY (node_id) REFERENCES nodes(node_id)
                )
            """)

            # State Transitions Table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS state_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_state_id INTEGER NOT NULL,
                    target_state_id INTEGER NOT NULL,
                    transition_type TEXT NOT NULL,
                    probability REAL DEFAULT 1.0,
                    energy_delta REAL DEFAULT 0.0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_state_id) REFERENCES temporal_states(id),
                    FOREIGN KEY (target_state_id) REFERENCES temporal_states(id)
                )
            """)

            # Cache Management Table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_hash TEXT NOT NULL UNIQUE,
                    data TEXT NOT NULL,
                    access_count INTEGER DEFAULT 1,
                    last_access DATETIME DEFAULT CURRENT_TIMESTAMP,
                    priority INTEGER DEFAULT 0,
                    size_bytes INTEGER NOT NULL,
                    expiry DATETIME
                )
            """)

            # Optimization Metrics Table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimization_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    context TEXT,
                    node_id TEXT,
                    FOREIGN KEY (node_id) REFERENCES nodes(node_id)
                )
            """)

            # Create indices for performance
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_temporal_states_node_id ON temporal_states(node_id)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_entries_key_hash ON cache_entries(key_hash)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_optimization_metrics_type ON optimization_metrics(metric_type)")

            self.conn.commit()
            logger.info("Tables created successfully")
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")
            raise

    def store_temporal_state(self, node_id: str, state_type: str, state_data: Dict) -> int:
        """Store a new temporal state."""
        try:
            state_json = json.dumps(state_data)
            state_hash = hashlib.sha256(state_json.encode()).hexdigest()

            # Get previous state hash
            self.cursor.execute("""
                SELECT state_hash FROM temporal_states 
                WHERE node_id = ? ORDER BY timestamp DESC LIMIT 1
            """, (node_id,))
            result = self.cursor.fetchone()
            previous_hash = result[0] if result else None

            # Insert new state
            self.cursor.execute("""
                INSERT INTO temporal_states 
                (node_id, state_hash, previous_state_hash, state_type, state_data)
                VALUES (?, ?, ?, ?, ?)
            """, (node_id, state_hash, previous_hash, state_type, state_json))
            
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Error storing temporal state: {e}")
            self.conn.rollback()
            raise

    def record_state_transition(self, source_id: int, target_id: int, 
                              transition_type: str, probability: float = 1.0,
                              energy_delta: float = 0.0) -> int:
        """Record a state transition between two temporal states."""
        try:
            self.cursor.execute("""
                INSERT INTO state_transitions 
                (source_state_id, target_state_id, transition_type, probability, energy_delta)
                VALUES (?, ?, ?, ?, ?)
            """, (source_id, target_id, transition_type, probability, energy_delta))
            
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Error recording state transition: {e}")
            self.conn.rollback()
            raise

    def manage_cache(self, key: str, data: str, priority: int = 0, 
                    expiry_hours: int = 24) -> bool:
        """Manage cache entries with priority and expiration."""
        try:
            key_hash = hashlib.sha256(key.encode()).hexdigest()
            size_bytes = len(data.encode())
            expiry = datetime.now() + timedelta(hours=expiry_hours)

            self.cursor.execute("""
                INSERT OR REPLACE INTO cache_entries 
                (key_hash, data, priority, size_bytes, expiry)
                VALUES (?, ?, ?, ?, ?)
            """, (key_hash, data, priority, size_bytes, expiry))
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error managing cache: {e}")
            self.conn.rollback()
            return False

    def record_optimization_metric(self, metric_type: str, value: float, 
                                context: Optional[str] = None,
                                node_id: Optional[str] = None) -> int:
        """Record an optimization metric."""
        try:
            self.cursor.execute("""
                INSERT INTO optimization_metrics 
                (metric_type, value, context, node_id)
                VALUES (?, ?, ?, ?)
            """, (metric_type, value, context, node_id))
            
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Error recording optimization metric: {e}")
            self.conn.rollback()
            raise

    def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries."""
        try:
            self.cursor.execute("""
                DELETE FROM cache_entries 
                WHERE expiry < datetime('now')
            """)
            deleted_count = self.cursor.rowcount
            self.conn.commit()
            return deleted_count
        except sqlite3.Error as e:
            logger.error(f"Error cleaning up cache: {e}")
            self.conn.rollback()
            return 0

    def get_state_history(self, node_id: str, limit: int = 10) -> List[Dict]:
        """Retrieve state history for a node."""
        try:
            self.cursor.execute("""
                SELECT * FROM temporal_states 
                WHERE node_id = ? 
                ORDER BY timestamp DESC LIMIT ?
            """, (node_id, limit))
            
            columns = [description[0] for description in self.cursor.description]
            states = []
            for row in self.cursor.fetchall():
                state_dict = dict(zip(columns, row))
                state_dict['state_data'] = json.loads(state_dict['state_data'])
                states.append(state_dict)
            
            return states
        except sqlite3.Error as e:
            logger.error(f"Error retrieving state history: {e}")
            return []

    def get_optimization_metrics(self, metric_type: Optional[str] = None,
                               node_id: Optional[str] = None,
                               limit: int = 100) -> List[Dict]:
        """Retrieve optimization metrics with optional filtering."""
        try:
            query = "SELECT * FROM optimization_metrics WHERE 1=1"
            params = []
            
            if metric_type:
                query += " AND metric_type = ?"
                params.append(metric_type)
            if node_id:
                query += " AND node_id = ?"
                params.append(node_id)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            self.cursor.execute(query, params)
            
            columns = [description[0] for description in self.cursor.description]
            metrics = []
            for row in self.cursor.fetchall():
                metrics.append(dict(zip(columns, row)))
            
            return metrics
        except sqlite3.Error as e:
            logger.error(f"Error retrieving optimization metrics: {e}")
            return []

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close() 