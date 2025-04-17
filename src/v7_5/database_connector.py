#!/usr/bin/env python
"""
LUMINA V7.5 Database Connector
Provides connectivity to the neural metrics database with v7.5 extensions
"""

import os
import sys
import logging
import sqlite3
import datetime
import argparse
import json
import time
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'db', 'database_connector.log'), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DatabaseConnector")

# Default database path
DEFAULT_DB_PATH = os.environ.get('NEURAL_METRICS_DB', 'data/neural_metrics.db')

class DatabaseConnector:
    """
    Database connector for the LUMINA V7.5 system.
    Provides unified access to various databases used by the system.
    """
    
    def __init__(self, data_dir=None, mock_mode=False):
        """
        Initialize the database connector.
        
        Args:
            data_dir (str): Directory where databases are stored
            mock_mode (bool): Whether to run in mock mode
        """
        # Configure logging
        self.logger = logging.getLogger("lumina.database")
        self.logger.setLevel(logging.INFO)
        
        log_dir = os.path.join("logs", "database")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, "database.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Initialize paths and settings
        self.data_dir = data_dir or os.environ.get("LUMINA_DATA_DIR", "data")
        self.mock_mode = mock_mode
        self.connections = {}
        self.running = True
        
        self.logger.info(f"Initializing DatabaseConnector")
        self.logger.info(f"Data directory: {self.data_dir}")
        self.logger.info(f"Mock mode: {self.mock_mode}")
        
        # Update component status
        self._update_component_status("online")
        
        # Create directory structure if needed
        self._initialize_directories()
        
        # Initialize the databases
        self._initialize_databases()
        
        # Start periodic check thread
        self.maintenance_thread = threading.Thread(target=self._maintenance_thread)
        self.maintenance_thread.daemon = True
        self.maintenance_thread.start()
        
    def _update_component_status(self, status):
        """Update component status in the status file"""
        status_file = os.path.join("data", "component_status.json")
        
        try:
            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
            else:
                status_data = {
                    "Neural Seed": "offline",
                    "Consciousness": "offline",
                    "Holographic UI": "offline",
                    "Chat Interface": "offline",
                    "Database Connector": "offline",
                    "Autowiki": "offline"
                }
            
            status_data["Database Connector"] = status
            
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
                
            self.logger.info(f"Updated Database Connector status to: {status}")
        except Exception as e:
            self.logger.error(f"Failed to update component status: {e}")
    
    def _initialize_directories(self):
        """Create necessary data directories"""
        # Main data directory
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            self.logger.info(f"Created main data directory: {self.data_dir}")
        
        # Database-specific directories
        db_dirs = [
            os.path.join(self.data_dir, "neural"),
            os.path.join(self.data_dir, "memory"),
            os.path.join(self.data_dir, "consciousness"),
            os.path.join(self.data_dir, "knowledge")
        ]
        
        for directory in db_dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)
                self.logger.info(f"Created database directory: {directory}")
    
    def _initialize_databases(self):
        """Initialize database connections and schemas"""
        try:
            # Define database paths
            self.db_paths = {
                "neural": os.path.join(self.data_dir, "neural", "neural_metrics.db"),
                "memory": os.path.join(self.data_dir, "memory", "memory.db"),
                "consciousness": os.path.join(self.data_dir, "consciousness", "consciousness.db"),
                "knowledge": os.path.join(self.data_dir, "knowledge", "knowledge.db")
            }
            
            # Initialize each database
            for db_name, db_path in self.db_paths.items():
                self._init_database(db_name, db_path)
                
            self.logger.info("All databases initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize databases: {e}")
    
    def _init_database(self, db_name, db_path):
        """Initialize a specific database with its schema"""
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            # Connect to database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create tables based on database type
            if db_name == "neural":
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS neural_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    node_id TEXT,
                    additional_data TEXT
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS neural_nodes (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    properties TEXT
                )
                ''')
                
            elif db_name == "memory":
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT,
                    importance REAL DEFAULT 0.5,
                    tags TEXT
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_associations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    target_id INTEGER NOT NULL,
                    strength REAL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES memory_entries (id),
                    FOREIGN KEY (target_id) REFERENCES memory_entries (id)
                )
                ''')
                
            elif db_name == "consciousness":
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS consciousness_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    state_type TEXT NOT NULL,
                    state_value TEXT NOT NULL,
                    duration REAL,
                    notes TEXT
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS consciousness_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    component TEXT,
                    notes TEXT
                )
                ''')
                
            elif db_name == "knowledge":
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT,
                    source TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_connections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    target_id INTEGER NOT NULL,
                    relationship_type TEXT NOT NULL,
                    strength REAL DEFAULT 0.5,
                    FOREIGN KEY (source_id) REFERENCES knowledge_entries (id),
                    FOREIGN KEY (target_id) REFERENCES knowledge_entries (id)
                )
                ''')
            
            # Commit changes and close
            conn.commit()
            conn.close()
            
            self.logger.info(f"Initialized {db_name} database at {db_path}")
            
            # Insert mock data if in mock mode
            if self.mock_mode:
                self._insert_mock_data(db_name)
                
        except Exception as e:
            self.logger.error(f"Failed to initialize {db_name} database: {e}")
    
    def _insert_mock_data(self, db_name):
        """Insert mock data into specified database for testing"""
        try:
            now = datetime.datetime.now().isoformat()
            
            if db_name == "neural":
                conn = self.get_connection("neural")
                cursor = conn.cursor()
                
                # Insert mock neural metrics
                for i in range(10):
                    cursor.execute(
                        "INSERT INTO neural_metrics (timestamp, metric_type, value, node_id) VALUES (?, ?, ?, ?)",
                        (now, f"activity_{i % 3}", 0.1 * i, f"node_{i}")
                    )
                
                # Insert mock neural nodes
                for i in range(5):
                    node_type = ["sensory", "processing", "memory", "output", "control"][i]
                    cursor.execute(
                        "INSERT INTO neural_nodes (id, type, created_at, status, properties) VALUES (?, ?, ?, ?, ?)",
                        (f"node_{i}", node_type, now, "active", json.dumps({"level": i}))
                    )
                
                conn.commit()
                
            elif db_name == "memory":
                conn = self.get_connection("memory")
                cursor = conn.cursor()
                
                # Insert mock memory entries
                categories = ["conversation", "system", "learning", "experience", "observation"]
                for i in range(5):
                    cursor.execute(
                        "INSERT INTO memory_entries (timestamp, category, content, source, importance, tags) VALUES (?, ?, ?, ?, ?, ?)",
                        (now, categories[i], f"Mock memory content {i}", "mock_system", 0.5 + (i * 0.1), json.dumps(["mock", "test"]))
                    )
                
                conn.commit()
                
            elif db_name == "consciousness":
                conn = self.get_connection("consciousness")
                cursor = conn.cursor()
                
                # Insert mock consciousness states
                states = ["focused", "exploratory", "contemplative", "analytical", "creative"]
                for i in range(5):
                    cursor.execute(
                        "INSERT INTO consciousness_states (timestamp, state_type, state_value, duration, notes) VALUES (?, ?, ?, ?, ?)",
                        (now, "cognition", states[i], 60.0, f"Mock state {i}")
                    )
                
                conn.commit()
                
            elif db_name == "knowledge":
                conn = self.get_connection("knowledge")
                cursor = conn.cursor()
                
                # Insert mock knowledge entries
                topics = ["Neural Networks", "Consciousness", "Memory Systems", "Learning Algorithms", "LUMINA Architecture"]
                for i in range(5):
                    cursor.execute(
                        "INSERT INTO knowledge_entries (title, content, category, source, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (topics[i], f"Mock content about {topics[i]}", "core_concepts", "system_init", now, now)
                    )
                
                conn.commit()
            
            self.logger.info(f"Inserted mock data into {db_name} database")
            
        except Exception as e:
            self.logger.error(f"Failed to insert mock data into {db_name} database: {e}")
    
    def get_connection(self, db_name):
        """
        Get a connection to the specified database
        
        Args:
            db_name (str): Name of the database to connect to
        
        Returns:
            sqlite3.Connection: Database connection
        """
        if db_name not in self.db_paths:
            raise ValueError(f"Unknown database: {db_name}")
        
        # Check if connection exists and is valid
        if db_name in self.connections:
            try:
                # Test the connection with a simple query
                self.connections[db_name].execute("SELECT 1")
                return self.connections[db_name]
            except sqlite3.Error:
                # Connection is not valid, remove it
                self.logger.warning(f"Existing connection to {db_name} is invalid, recreating...")
                del self.connections[db_name]
        
        # Create a new connection
        try:
            conn = sqlite3.connect(self.db_paths[db_name])
            conn.row_factory = sqlite3.Row  # Enable row name access
            self.connections[db_name] = conn
            return conn
        except sqlite3.Error as e:
            self.logger.error(f"Failed to connect to {db_name} database: {e}")
            raise
    
    def execute_query(self, db_name, query, params=None):
        """
        Execute a query on the specified database
        
        Args:
            db_name (str): Name of the database
            query (str): SQL query to execute
            params (tuple, optional): Parameters for the query
        
        Returns:
            list: Query results
        """
        try:
            conn = self.get_connection(db_name)
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if query.strip().upper().startswith(('SELECT', 'PRAGMA')):
                return [dict(row) for row in cursor.fetchall()]
            else:
                conn.commit()
                return cursor.rowcount
                
        except Exception as e:
            self.logger.error(f"Query execution failed on {db_name}: {e}")
            self.logger.error(f"Query: {query}")
            self.logger.error(f"Params: {params}")
            raise
    
    def insert_data(self, db_name, table, data):
        """
        Insert data into a table
        
        Args:
            db_name (str): Name of the database
            table (str): Table name
            data (dict): Data to insert
        
        Returns:
            int: Last row ID
        """
        try:
            conn = self.get_connection(db_name)
            cursor = conn.cursor()
            
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?'] * len(data))
            values = tuple(data.values())
            
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            cursor.execute(query, values)
            conn.commit()
            
            return cursor.lastrowid
        except Exception as e:
            self.logger.error(f"Failed to insert data into {db_name}.{table}: {e}")
            raise
    
    def get_database_stats(self):
        """
        Get statistics for all databases
        
        Returns:
            dict: Database statistics
        """
        stats = {}
        
        try:
            for db_name in self.db_paths:
                db_stats = {"tables": {}, "total_rows": 0, "size_bytes": 0}
                
                # Get table statistics
                conn = self.get_connection(db_name)
                
                # Get list of tables
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row['name'] for row in cursor.fetchall()]
                
                # Get row count for each table
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                    row_count = cursor.fetchone()['count']
                    db_stats["tables"][table] = row_count
                    db_stats["total_rows"] += row_count
                
                # Get database file size
                if os.path.exists(self.db_paths[db_name]):
                    db_stats["size_bytes"] = os.path.getsize(self.db_paths[db_name])
                    db_stats["size_mb"] = round(db_stats["size_bytes"] / (1024 * 1024), 2)
                
                stats[db_name] = db_stats
            
            return stats
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}
    
    def _maintenance_thread(self):
        """Background thread for database maintenance tasks"""
        self.logger.info("Database maintenance thread started")
        
        try:
            while self.running:
                # Perform periodic maintenance
                for db_name in list(self.connections.keys()):
                    try:
                        # Execute PRAGMA optimize on each database
                        conn = self.get_connection(db_name)
                        conn.execute("PRAGMA optimize")
                        conn.commit()
                        self.logger.debug(f"Optimized {db_name} database")
                    except Exception as e:
                        self.logger.error(f"Error optimizing {db_name} database: {e}")
                
                # Sleep for 30 minutes
                for _ in range(30 * 60):
                    if not self.running:
                        break
                    time.sleep(1)
                    
        except Exception as e:
            self.logger.error(f"Error in maintenance thread: {e}")
        
        self.logger.info("Database maintenance thread stopped")
    
    def close_connections(self):
        """Close all database connections"""
        for db_name, conn in list(self.connections.items()):
            try:
                conn.close()
                self.logger.debug(f"Closed connection to {db_name}")
            except Exception as e:
                self.logger.error(f"Error closing {db_name} connection: {e}")
            
        self.connections.clear()
        self.logger.info("All database connections closed")
    
    def shutdown(self):
        """Shutdown the database connector"""
        self.logger.info("Shutting down DatabaseConnector")
        
        # Stop the maintenance thread
        self.running = False
        if self.maintenance_thread and self.maintenance_thread.is_alive():
            self.maintenance_thread.join(timeout=5.0)
        
        # Close all connections
        self.close_connections()
        
        # Update component status
        self._update_component_status("offline")

def main():
    """Main entry point for the database connector"""
    parser = argparse.ArgumentParser(description="LUMINA Database Connector")
    parser.add_argument("--mock", action="store_true", help="Run with mock data")
    parser.add_argument("--data-dir", help="Data directory path")
    
    args = parser.parse_args()
    
    try:
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger("lumina.database.main")
        
        logger.info("Starting Database Connector")
        
        # Initialize database connector
        db_connector = DatabaseConnector(
            data_dir=args.data_dir,
            mock_mode=args.mock
        )
        
        # Display database statistics
        stats = db_connector.get_database_stats()
        logger.info(f"Database statistics: {json.dumps(stats, indent=2)}")
        
        # Keep running until interrupted
        try:
            logger.info("Database Connector is running. Press Ctrl+C to exit.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            db_connector.shutdown()
            
    except Exception as e:
        logger.error(f"Error in Database Connector main: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 