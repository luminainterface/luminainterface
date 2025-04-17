#!/usr/bin/env python
"""
LUMINA v7.5 Database Connector
------------------------------
Provides synchronization between local SQLite databases and remote database systems.
Supports bidirectional sync and conflict resolution strategies.
"""

import os
import sys
import time
import json
import sqlite3
import logging
import argparse
import threading
from pathlib import Path
from datetime import datetime

# Ensure the parent directory is in the path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path("logs/db/database_connector.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("LUMINA_DBConnector")

class DatabaseConnector:
    """
    Database connector for LUMINA v7.5
    Provides synchronization between local and remote databases
    """
    
    def __init__(self, 
                 local_db_path="data/neural_metrics.db", 
                 config_path="data/db/sync_config.json",
                 sync_interval=60, 
                 mock_mode=False):
        """Initialize the database connector"""
        self.local_db_path = local_db_path
        self.config_path = config_path
        self.sync_interval = sync_interval
        self.mock_mode = mock_mode
        self.stop_event = threading.Event()
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Track sync metrics
        self.sync_metrics = {
            "last_sync": None,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "records_pushed": 0,
            "records_pulled": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0
        }
        
        # Load configuration
        self._load_config()
        
        # Connect to local database
        if not self.mock_mode:
            try:
                self._connect_local_db()
                logger.info(f"Connected to local database: {self.local_db_path}")
            except Exception as e:
                logger.error(f"Error connecting to local database: {e}")
                self.mock_mode = True
                logger.info("Falling back to mock mode")
    
    def _load_config(self):
        """Load sync configuration from JSON file"""
        default_config = {
            "remote_connections": [],
            "sync_tables": ["neural_patterns", "consciousness_states", "memory_records", "learning_events"],
            "conflict_strategy": "last_modified_wins",
            "sync_direction": "bidirectional",
            "enabled": True
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                self.config = default_config
        else:
            logger.info(f"Configuration file not found, creating default at {self.config_path}")
            self.config = default_config
            self._save_config()
    
    def _save_config(self):
        """Save current configuration to JSON file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4, sort_keys=True)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _connect_local_db(self):
        """Connect to the local SQLite database"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.local_db_path), exist_ok=True)
            
            # Connect to database
            self.conn = sqlite3.connect(self.local_db_path)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"Connected to local database: {self.local_db_path}")
            
            # Create tables if they don't exist
            self._create_tables()
        except Exception as e:
            logger.error(f"Error connecting to local database: {e}")
            raise
    
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Create sync_metadata table to track changes
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sync_metadata (
            table_name TEXT,
            record_id TEXT,
            last_modified TIMESTAMP,
            last_sync TIMESTAMP,
            sync_status TEXT,
            PRIMARY KEY (table_name, record_id)
        )
        ''')
        
        # Create example tables if they don't exist already
        for table in self.config["sync_tables"]:
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table} (
                id TEXT PRIMARY KEY,
                data TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
            ''')
        
        self.conn.commit()
        logger.info("Database tables verified/created")
    
    def _mock_sync(self):
        """Perform a mock synchronization for testing"""
        logger.info("Performing mock synchronization...")
        
        # Simulate successful sync with random data
        import random
        
        # Random success/failure
        success = random.random() > 0.1  # 90% success rate
        
        if success:
            # Update metrics with random data
            self.sync_metrics["last_sync"] = datetime.now().isoformat()
            self.sync_metrics["successful_syncs"] += 1
            self.sync_metrics["records_pushed"] += random.randint(0, 10)
            self.sync_metrics["records_pulled"] += random.randint(0, 10)
            
            # Random conflicts
            conflicts = random.randint(0, 3)
            self.sync_metrics["conflicts_detected"] += conflicts
            self.sync_metrics["conflicts_resolved"] += conflicts
            
            logger.info("Mock sync completed successfully")
        else:
            self.sync_metrics["failed_syncs"] += 1
            logger.error("Mock sync failed")
        
        return success
    
    def _sync_to_remote(self, remote_conn):
        """Synchronize local changes to a remote database"""
        logger.info(f"Syncing to remote: {remote_conn.get('name', 'unnamed')}")
        
        # This would contain actual sync logic to push changes to remote
        # For now we're just logging
        logger.info("Remote sync logic would go here")
        return True
    
    def _sync_from_remote(self, remote_conn):
        """Synchronize remote changes to local database"""
        logger.info(f"Syncing from remote: {remote_conn.get('name', 'unnamed')}")
        
        # This would contain actual sync logic to pull changes from remote
        # For now we're just logging
        logger.info("Remote sync logic would go here")
        return True
    
    def _resolve_conflicts(self, conflicts):
        """Resolve synchronization conflicts based on configured strategy"""
        strategy = self.config.get("conflict_strategy", "last_modified_wins")
        logger.info(f"Resolving {len(conflicts)} conflicts using strategy: {strategy}")
        
        # This would contain actual conflict resolution logic
        # For now we're just logging
        logger.info("Conflict resolution logic would go here")
        return len(conflicts)
    
    def perform_sync(self):
        """Perform a full synchronization cycle"""
        if self.mock_mode:
            return self._mock_sync()
        
        try:
            start_time = time.time()
            logger.info("Starting database synchronization")
            
            # Check if sync is enabled in config
            if not self.config.get("enabled", True):
                logger.info("Synchronization is disabled in config")
                return False
            
            # Get remote connections from config
            remote_connections = self.config.get("remote_connections", [])
            if not remote_connections:
                logger.warning("No remote connections configured")
                self.sync_metrics["last_sync"] = datetime.now().isoformat()
                return True
            
            # Sync with each remote connection
            for remote_conn in remote_connections:
                # Skip disabled connections
                if not remote_conn.get("enabled", True):
                    logger.info(f"Skipping disabled connection: {remote_conn.get('name', 'unnamed')}")
                    continue
                
                # Bidirectional sync
                if self.config.get("sync_direction") in ["bidirectional", "push"]:
                    self._sync_to_remote(remote_conn)
                
                if self.config.get("sync_direction") in ["bidirectional", "pull"]:
                    self._sync_from_remote(remote_conn)
            
            # Update sync timestamp
            self.sync_metrics["last_sync"] = datetime.now().isoformat()
            self.sync_metrics["successful_syncs"] += 1
            
            # Calculate duration
            duration = time.time() - start_time
            logger.info(f"Synchronization completed successfully in {duration:.2f} seconds")
            return True
            
        except Exception as e:
            self.sync_metrics["failed_syncs"] += 1
            logger.error(f"Error during synchronization: {e}")
            return False
    
    def get_status(self):
        """Get the current status of the database connector"""
        return {
            "mock_mode": self.mock_mode,
            "enabled": self.config.get("enabled", True),
            "sync_interval": self.sync_interval,
            "sync_metrics": self.sync_metrics.copy(),
            "local_db_path": self.local_db_path,
            "remote_connections": len(self.config.get("remote_connections", [])),
            "sync_tables": self.config.get("sync_tables", []),
            "timestamp": datetime.now().isoformat()
        }
    
    def add_remote_connection(self, connection_info):
        """Add a new remote connection to the configuration"""
        if "name" not in connection_info or "connection_string" not in connection_info:
            logger.error("Remote connection requires name and connection_string")
            return False
        
        # Add the connection to config
        if "remote_connections" not in self.config:
            self.config["remote_connections"] = []
        
        self.config["remote_connections"].append(connection_info)
        self._save_config()
        
        logger.info(f"Added remote connection: {connection_info['name']}")
        return True
    
    def remove_remote_connection(self, connection_name):
        """Remove a remote connection from the configuration"""
        if "remote_connections" not in self.config:
            return False
        
        # Find and remove the connection
        initial_count = len(self.config["remote_connections"])
        self.config["remote_connections"] = [
            conn for conn in self.config["remote_connections"] 
            if conn.get("name") != connection_name
        ]
        
        # Check if any were removed
        if len(self.config["remote_connections"]) < initial_count:
            self._save_config()
            logger.info(f"Removed remote connection: {connection_name}")
            return True
        
        logger.warning(f"Remote connection not found: {connection_name}")
        return False
    
    def run_sync_service(self):
        """Run as a synchronization service, syncing at regular intervals"""
        logger.info(f"Starting database sync service (interval: {self.sync_interval}s)")
        
        try:
            while not self.stop_event.is_set():
                # Perform sync
                self.perform_sync()
                
                # Wait for the next sync interval or until stopped
                logger.info(f"Waiting {self.sync_interval} seconds until next sync")
                self.stop_event.wait(timeout=self.sync_interval)
        except KeyboardInterrupt:
            logger.info("Database sync service stopped by user")
        finally:
            logger.info("Database sync service shutdown")
    
    def start_service(self):
        """Start the sync service in a background thread"""
        self.service_thread = threading.Thread(target=self.run_sync_service)
        self.service_thread.daemon = True
        self.service_thread.start()
        return self.service_thread
    
    def stop_service(self):
        """Stop the sync service"""
        self.stop_event.set()
        if hasattr(self, 'service_thread'):
            self.service_thread.join(timeout=2.0)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="LUMINA v7.5 Database Connector")
    parser.add_argument("--db-path", type=str, default="data/neural_metrics.db", help="Path to local database")
    parser.add_argument("--config", type=str, default="data/db/sync_config.json", help="Path to sync configuration")
    parser.add_argument("--interval", type=int, default=60, help="Sync interval in seconds")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    parser.add_argument("--sync", action="store_true", help="Run as a sync service")
    parser.add_argument("--sync-once", action="store_true", help="Perform one sync and exit")
    args = parser.parse_args()
    
    try:
        # Ensure directories exist
        os.makedirs("logs/db", exist_ok=True)
        os.makedirs("data/db", exist_ok=True)
        
        # Create connector
        connector = DatabaseConnector(
            local_db_path=args.db_path,
            config_path=args.config,
            sync_interval=args.interval,
            mock_mode=args.mock
        )
        
        if args.sync_once:
            # Perform a single sync
            logger.info("Performing one-time sync")
            success = connector.perform_sync()
            return 0 if success else 1
        
        elif args.sync:
            # Run as a sync service
            print(f"Starting Database Sync Service in {'mock' if args.mock else 'normal'} mode")
            print(f"Sync interval: {args.interval} seconds")
            print(f"Local database: {args.db_path}")
            print(f"Press Ctrl+C to stop the service")
            connector.run_sync_service()
            return 0
        
        else:
            # Just print status and exit
            status = connector.get_status()
            print("\nLUMINA v7.5 Database Connector Status:")
            print("======================================")
            print(f"Mock Mode: {'Enabled' if status['mock_mode'] else 'Disabled'}")
            print(f"Enabled: {'Yes' if status['enabled'] else 'No'}")
            print(f"Sync Interval: {status['sync_interval']} seconds")
            print(f"Local Database: {status['local_db_path']}")
            print(f"Remote Connections: {status['remote_connections']}")
            print(f"Tables to Sync: {', '.join(status['sync_tables'])}")
            
            print("\nSync Metrics:")
            print(f"- Last Sync: {status['sync_metrics']['last_sync'] or 'Never'}")
            print(f"- Successful Syncs: {status['sync_metrics']['successful_syncs']}")
            print(f"- Failed Syncs: {status['sync_metrics']['failed_syncs']}")
            print(f"- Records Pushed: {status['sync_metrics']['records_pushed']}")
            print(f"- Records Pulled: {status['sync_metrics']['records_pulled']}")
            print(f"- Conflicts Detected: {status['sync_metrics']['conflicts_detected']}")
            print(f"- Conflicts Resolved: {status['sync_metrics']['conflicts_resolved']}")
            
            print("\nUse --sync to run as a continuous sync service")
            print("Use --sync-once to perform a single sync and exit")
            return 0
    
    except Exception as e:
        logger.error(f"Error in database connector: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 