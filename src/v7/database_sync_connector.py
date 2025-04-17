#!/usr/bin/env python
"""
V7 Database Synchronization Connector

This module provides comprehensive bidirectional synchronization between database systems,
with transaction management, conflict resolution, and monitoring capabilities.

Key features:
- Bidirectional synchronization between language and central databases
- Transaction management for data integrity
- Conflict detection and resolution
- Change tracking for efficient synchronization
- Monitoring and metrics collection
"""

import os
import sys
import time
import threading
import logging
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from queue import Queue
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("database-sync-connector")

class DatabaseInfo:
    """Information about a database connection"""
    
    def __init__(self, name: str, path: str, tables: Optional[List[str]] = None,
                 priority: int = 1, metadata: Optional[Dict] = None):
        self.name = name
        self.path = path
        self.tables = tables or []
        self.priority = priority  # Higher priority wins in conflicts
        self.metadata = metadata or {}
        self.last_sync = None
        self.connection = None
    
    def get_connection(self) -> sqlite3.Connection:
        """Get SQLite connection to the database"""
        if self.connection is None:
            try:
                self.connection = sqlite3.connect(self.path)
                self.connection.row_factory = sqlite3.Row
            except sqlite3.Error as e:
                logger.error(f"Error connecting to database {self.name}: {e}")
                raise
        return self.connection
    
    def close_connection(self):
        """Close the database connection"""
        if self.connection:
            try:
                self.connection.close()
                self.connection = None
            except sqlite3.Error as e:
                logger.error(f"Error closing database {self.name}: {e}")
    
    def get_tables(self) -> List[str]:
        """Get list of tables in the database"""
        if not self.tables:
            try:
                conn = self.get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                self.tables = [row[0] for row in cursor.fetchall()]
            except sqlite3.Error as e:
                logger.error(f"Error getting tables for {self.name}: {e}")
                return []
        return self.tables
    
    def to_dict(self) -> Dict:
        """Convert database info to dictionary"""
        return {
            "name": self.name,
            "path": self.path,
            "tables": self.tables,
            "priority": self.priority,
            "metadata": self.metadata,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DatabaseInfo':
        """Create DatabaseInfo from dictionary"""
        db_info = cls(
            name=data["name"],
            path=data["path"],
            tables=data.get("tables", []),
            priority=data.get("priority", 1),
            metadata=data.get("metadata", {})
        )
        if data.get("last_sync"):
            db_info.last_sync = datetime.fromisoformat(data["last_sync"])
        return db_info


class SyncTransaction:
    """Represents a synchronization transaction between databases"""
    
    def __init__(self, source_db: str, target_db: str,
                 tables: List[str], transaction_id: Optional[str] = None):
        self.transaction_id = transaction_id or f"sync_{int(time.time())}_{hash(source_db + target_db)}"
        self.source_db = source_db
        self.target_db = target_db
        self.tables = tables
        self.start_time = datetime.now()
        self.end_time = None
        self.status = "created"  # created, in_progress, completed, failed, rolled_back
        self.operations = []
        self.conflicts = []
        self.error = None
    
    def start(self):
        """Start the transaction"""
        self.status = "in_progress"
        logger.info(f"Starting transaction {self.transaction_id}: {self.source_db} → {self.target_db}")
        return self
    
    def add_operation(self, operation: Dict):
        """Add an operation to the transaction"""
        self.operations.append(operation)
    
    def add_conflict(self, conflict: Dict):
        """Add a conflict to the transaction"""
        self.conflicts.append(conflict)
    
    def complete(self):
        """Mark the transaction as completed"""
        self.status = "completed"
        self.end_time = datetime.now()
        logger.info(f"Completed transaction {self.transaction_id} in {self.duration():.2f} seconds")
        return self
    
    def fail(self, error: str):
        """Mark the transaction as failed"""
        self.status = "failed"
        self.error = error
        self.end_time = datetime.now()
        logger.error(f"Failed transaction {self.transaction_id}: {error}")
        return self
    
    def rollback(self):
        """Mark the transaction as rolled back"""
        self.status = "rolled_back"
        self.end_time = datetime.now()
        logger.warning(f"Rolled back transaction {self.transaction_id}")
        return self
    
    def duration(self) -> float:
        """Get the duration of the transaction in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict:
        """Convert transaction to dictionary"""
        return {
            "transaction_id": self.transaction_id,
            "source_db": self.source_db,
            "target_db": self.target_db,
            "tables": self.tables,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "operations": self.operations,
            "conflicts": self.conflicts,
            "error": self.error,
            "duration": self.duration()
        }


class ConflictResolution:
    """
    Handles conflict detection and resolution for database synchronization
    
    Strategies:
    - TIMESTAMP: Use the most recently updated record
    - PRIORITY: Use the record from the higher priority database
    - MERGE: Attempt to merge the records (field by field)
    - SOURCE: Always use the source record
    - TARGET: Always use the target record
    """
    
    STRATEGIES = {
        "TIMESTAMP": 1,
        "PRIORITY": 2,
        "MERGE": 3,
        "SOURCE": 4,
        "TARGET": 5
    }
    
    def __init__(self, default_strategy: str = "TIMESTAMP"):
        self.default_strategy = default_strategy
        self.table_strategies = {}  # Map of table name to resolution strategy
        self.field_strategies = {}  # Map of table.field to resolution strategy
        self.resolved_conflicts = []
        
    def set_table_strategy(self, table: str, strategy: str):
        """Set resolution strategy for a table"""
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Invalid resolution strategy: {strategy}")
        self.table_strategies[table] = strategy
    
    def set_field_strategy(self, table: str, field: str, strategy: str):
        """Set resolution strategy for a specific field"""
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Invalid resolution strategy: {strategy}")
        self.field_strategies[f"{table}.{field}"] = strategy
    
    def get_strategy(self, table: str, field: Optional[str] = None) -> str:
        """Get the resolution strategy for a table or field"""
        if field and f"{table}.{field}" in self.field_strategies:
            return self.field_strategies[f"{table}.{field}"]
        if table in self.table_strategies:
            return self.table_strategies[table]
        return self.default_strategy
    
    def resolve_conflict(self, table: str, source_record: Dict, target_record: Dict,
                        source_db: DatabaseInfo, target_db: DatabaseInfo) -> Dict:
        """Resolve conflict between two records"""
        resolution = {
            "table": table,
            "source_record": source_record,
            "target_record": target_record,
            "resolved_fields": {},
            "strategy_used": {},
            "final_record": {}
        }
        
        # Get primary key fields
        primary_key = source_record.get("id", None)
        if primary_key is None:
            # Try to find a field ending with _id
            for field in source_record:
                if field.endswith("_id"):
                    primary_key = field
                    break
        
        # If still no primary key, use first field
        if primary_key is None and source_record:
            primary_key = list(source_record.keys())[0]
        
        resolution["primary_key"] = {primary_key: source_record.get(primary_key)}
        
        # Combine all fields from both records
        all_fields = set(source_record.keys()) | set(target_record.keys())
        
        # Resolve each field
        for field in all_fields:
            source_value = source_record.get(field)
            target_value = target_record.get(field)
            
            # If values are the same, no conflict
            if source_value == target_value:
                resolution["final_record"][field] = source_value
                continue
            
            # If field only exists in one record, use that value
            if field not in source_record:
                resolution["final_record"][field] = target_value
                resolution["strategy_used"][field] = "TARGET_ONLY"
                continue
            if field not in target_record:
                resolution["final_record"][field] = source_value
                resolution["strategy_used"][field] = "SOURCE_ONLY"
                continue
            
            # Get strategy for this field
            strategy = self.get_strategy(table, field)
            resolution["strategy_used"][field] = strategy
            
            # Apply resolution strategy
            if strategy == "SOURCE":
                resolution["final_record"][field] = source_value
            elif strategy == "TARGET":
                resolution["final_record"][field] = target_value
            elif strategy == "TIMESTAMP":
                # Use timestamp fields if available
                source_updated = source_record.get("updated_at", source_record.get("timestamp", 0))
                target_updated = target_record.get("updated_at", target_record.get("timestamp", 0))
                
                if source_updated and target_updated:
                    if source_updated > target_updated:
                        resolution["final_record"][field] = source_value
                    else:
                        resolution["final_record"][field] = target_value
                else:
                    # Default to source if timestamps not available
                    resolution["final_record"][field] = source_value
            elif strategy == "PRIORITY":
                if source_db.priority >= target_db.priority:
                    resolution["final_record"][field] = source_value
                else:
                    resolution["final_record"][field] = target_value
            elif strategy == "MERGE":
                # For merge, we need type-specific logic
                if isinstance(source_value, dict) and isinstance(target_value, dict):
                    # Merge dictionaries
                    merged = {**target_value, **source_value}
                    resolution["final_record"][field] = merged
                elif isinstance(source_value, list) and isinstance(target_value, list):
                    # Combine lists with deduplication
                    merged = list(set(target_value + source_value))
                    resolution["final_record"][field] = merged
                else:
                    # For other types, use source (default)
                    resolution["final_record"][field] = source_value
            else:
                # Unknown strategy, default to source
                resolution["final_record"][field] = source_value
            
            # Record the resolution
            resolution["resolved_fields"][field] = {
                "source_value": source_value,
                "target_value": target_value,
                "resolved_value": resolution["final_record"][field],
                "strategy": strategy
            }
        
        # Add to resolved conflicts list
        self.resolved_conflicts.append(resolution)
        
        return resolution["final_record"]


class ChangeTracker:
    """
    Tracks changes in databases for efficient synchronization
    
    Uses multiple strategies for change detection:
    - Timestamp-based tracking
    - Hash-based tracking
    - Change log tracking
    """
    
    def __init__(self, storage_path: str = "data/sync_metadata"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # Last sync timestamps by table
        self.last_sync = {}  # {db_name: {table: timestamp}}
        
        # Record hashes for change detection
        self.record_hashes = {}  # {db_name: {table: {id: hash}}}
        
        # Load existing data
        self._load_metadata()
    
    def _load_metadata(self):
        """Load change tracking metadata from disk"""
        try:
            last_sync_path = os.path.join(self.storage_path, "last_sync.json")
            hashes_path = os.path.join(self.storage_path, "record_hashes.json")
            
            if os.path.exists(last_sync_path):
                with open(last_sync_path, 'r') as f:
                    self.last_sync = json.load(f)
            
            if os.path.exists(hashes_path):
                with open(hashes_path, 'r') as f:
                    self.record_hashes = json.load(f)
        except Exception as e:
            logger.error(f"Error loading change tracking metadata: {e}")
    
    def _save_metadata(self):
        """Save change tracking metadata to disk"""
        try:
            last_sync_path = os.path.join(self.storage_path, "last_sync.json")
            hashes_path = os.path.join(self.storage_path, "record_hashes.json")
            
            with open(last_sync_path, 'w') as f:
                json.dump(self.last_sync, f, indent=2)
            
            with open(hashes_path, 'w') as f:
                json.dump(self.record_hashes, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving change tracking metadata: {e}")
    
    def get_last_sync(self, db_name: str, table: str) -> Optional[float]:
        """Get the timestamp of the last sync for a table"""
        return self.last_sync.get(db_name, {}).get(table)
    
    def update_last_sync(self, db_name: str, table: str, timestamp: Optional[float] = None):
        """Update the last sync timestamp for a table"""
        if db_name not in self.last_sync:
            self.last_sync[db_name] = {}
        
        self.last_sync[db_name][table] = timestamp or time.time()
        self._save_metadata()
    
    def compute_record_hash(self, record: Dict) -> str:
        """Compute a hash for a record to detect changes"""
        # Create a sorted representation of the record for consistent hashing
        record_str = json.dumps(record, sort_keys=True)
        return hashlib.md5(record_str.encode()).hexdigest()
    
    def has_changed(self, db_name: str, table: str, record_id: Any, record: Dict) -> bool:
        """Check if a record has changed since the last sync"""
        new_hash = self.compute_record_hash(record)
        old_hash = self.record_hashes.get(db_name, {}).get(table, {}).get(str(record_id))
        
        if old_hash is None:
            # No previous hash, consider it changed
            self.update_record_hash(db_name, table, record_id, new_hash)
            return True
        
        # Update hash and return change status
        has_changed = new_hash != old_hash
        if has_changed:
            self.update_record_hash(db_name, table, record_id, new_hash)
        
        return has_changed
    
    def update_record_hash(self, db_name: str, table: str, record_id: Any, record_hash: Optional[str] = None):
        """Update the hash for a record"""
        if db_name not in self.record_hashes:
            self.record_hashes[db_name] = {}
        
        if table not in self.record_hashes[db_name]:
            self.record_hashes[db_name][table] = {}
        
        self.record_hashes[db_name][table][str(record_id)] = record_hash
        # Not saving after every update for performance, call _save_metadata() periodically
    
    def get_record_hash(self, db_name: str, table: str, record_id: Any) -> Optional[str]:
        """Get the stored hash for a record"""
        return self.record_hashes.get(db_name, {}).get(table, {}).get(str(record_id))
    
    def clear_table_hashes(self, db_name: str, table: str):
        """Clear all record hashes for a table"""
        if db_name in self.record_hashes and table in self.record_hashes[db_name]:
            self.record_hashes[db_name][table] = {}
            self._save_metadata()


class DatabaseSyncMetrics:
    """Collects and reports metrics for database synchronization"""
    
    def __init__(self):
        self.sync_count = 0
        self.record_count = 0
        self.conflict_count = 0
        self.error_count = 0
        self.start_time = time.time()
        self.last_sync_time = None
        self.sync_durations = []
        self.table_metrics = {}  # {table: {records: int, conflicts: int, errors: int}}
        self.transactions = []  # Recent transactions
    
    def start_sync(self):
        """Record the start of a sync operation"""
        self.sync_count += 1
        self.last_sync_time = time.time()
        return self.last_sync_time
    
    def end_sync(self, start_time: float):
        """Record the end of a sync operation"""
        duration = time.time() - start_time
        self.sync_durations.append(duration)
        # Keep only the last 100 durations
        if len(self.sync_durations) > 100:
            self.sync_durations.pop(0)
        return duration
    
    def add_transaction(self, transaction: Dict):
        """Add a transaction to the metrics"""
        self.transactions.append(transaction)
        # Keep only the last 100 transactions
        if len(self.transactions) > 100:
            self.transactions.pop(0)
    
    def record_table_sync(self, table: str, records: int, conflicts: int, errors: int):
        """Record metrics for a table sync"""
        if table not in self.table_metrics:
            self.table_metrics[table] = {"records": 0, "conflicts": 0, "errors": 0}
        
        self.table_metrics[table]["records"] += records
        self.table_metrics[table]["conflicts"] += conflicts
        self.table_metrics[table]["errors"] += errors
        
        self.record_count += records
        self.conflict_count += conflicts
        self.error_count += errors
    
    def get_metrics(self) -> Dict:
        """Get all metrics as a dictionary"""
        avg_duration = sum(self.sync_durations) / len(self.sync_durations) if self.sync_durations else 0
        
        return {
            "sync_count": self.sync_count,
            "record_count": self.record_count,
            "conflict_count": self.conflict_count,
            "error_count": self.error_count,
            "uptime": time.time() - self.start_time,
            "last_sync_time": self.last_sync_time,
            "average_sync_duration": avg_duration,
            "table_metrics": self.table_metrics,
            "recent_transactions": self.transactions[-10:],  # Last 10 transactions
            "timestamp": datetime.now().isoformat()
        }


class DatabaseSyncConnector:
    """
    Provides bidirectional synchronization between database systems
    
    This connector handles data synchronization between language and central databases,
    with support for transaction management, conflict resolution, and change tracking.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the database sync connector"""
        self.config = {
            "sync_interval": 300,  # seconds
            "batch_size": 100,
            "max_retries": 3,
            "retry_delay": 5,
            "conflict_resolution": "TIMESTAMP",
            "save_conflicts": True,
            "log_level": "INFO",
            "storage_path": "data/database_sync",
            "enable_change_tracking": True,
            "monitor_interval": 60,  # seconds
            "enable_metrics": True
        }
        if config:
            self.config.update(config)
        
        # Set log level
        logger.setLevel(getattr(logging, self.config["log_level"]))
        
        # Initialize storage directories
        os.makedirs(self.config["storage_path"], exist_ok=True)
        os.makedirs(os.path.join(self.config["storage_path"], "conflicts"), exist_ok=True)
        os.makedirs(os.path.join(self.config["storage_path"], "transactions"), exist_ok=True)
        
        # Initialize components
        self.databases = {}  # {name: DatabaseInfo}
        self.conflict_resolver = ConflictResolution(self.config["conflict_resolution"])
        self.change_tracker = ChangeTracker(os.path.join(self.config["storage_path"], "change_tracking"))
        self.metrics = DatabaseSyncMetrics()
        
        # Synchronization state
        self.sync_thread = None
        self.monitor_thread = None
        self.running = False
        self.sync_queue = Queue()
        
        logger.info(f"Database Sync Connector initialized with config: {self.config}")
    
    def register_database(self, db_info: DatabaseInfo) -> bool:
        """Register a database for synchronization"""
        try:
            # Test connection
            connection = db_info.get_connection()
            
            # Store database info
            self.databases[db_info.name] = db_info
            
            # Close test connection
            db_info.close_connection()
            
            logger.info(f"Registered database: {db_info.name} ({db_info.path})")
            return True
        except Exception as e:
            logger.error(f"Failed to register database {db_info.name}: {e}")
            return False
    
    def get_database(self, name: str) -> Optional[DatabaseInfo]:
        """Get database info by name"""
        return self.databases.get(name)
    
    def start(self):
        """Start the synchronization process"""
        if self.running:
            logger.warning("Sync connector is already running")
            return
        
        self.running = True
        
        # Start the sync thread
        self.sync_thread = threading.Thread(
            target=self._sync_worker,
            daemon=True,
            name="SyncWorkerThread"
        )
        self.sync_thread.start()
        
        # Start the monitor thread
        if self.config["monitor_interval"] > 0:
            self.monitor_thread = threading.Thread(
                target=self._monitor_worker,
                daemon=True,
                name="MonitorThread"
            )
            self.monitor_thread.start()
        
        logger.info("Database Sync Connector started")
    
    def stop(self):
        """Stop the synchronization process"""
        self.running = False
        
        # Wait for threads to finish
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=5.0)
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        # Close all database connections
        for db_name, db_info in self.databases.items():
            db_info.close_connection()
        
        logger.info("Database Sync Connector stopped")
    
    def sync_databases(self, source_db: str, target_db: str, tables: Optional[List[str]] = None,
                     force: bool = False) -> Dict:
        """
        Synchronize data between two databases
        
        Args:
            source_db: Name of the source database
            target_db: Name of the target database
            tables: List of tables to synchronize (None for all)
            force: Force synchronization even if no changes detected
            
        Returns:
            Dict with synchronization results
        """
        # Check if databases exist
        source_info = self.get_database(source_db)
        target_info = self.get_database(target_db)
        
        if not source_info:
            logger.error(f"Source database not found: {source_db}")
            return {"status": "error", "message": f"Source database not found: {source_db}"}
        
        if not target_info:
            logger.error(f"Target database not found: {target_db}")
            return {"status": "error", "message": f"Target database not found: {target_db}"}
        
        # Get list of tables if not provided
        if tables is None:
            tables = source_info.get_tables()
        
        # Create transaction
        transaction = SyncTransaction(source_db, target_db, tables)
        transaction.start()
        
        # Start metrics tracking
        sync_start = self.metrics.start_sync()
        
        try:
            # Process each table
            for table in tables:
                table_results = self._sync_table(
                    source_info, 
                    target_info, 
                    table, 
                    transaction,
                    force=force
                )
                
                transaction.add_operation({
                    "table": table,
                    "records_synced": table_results["records_synced"],
                    "conflicts": table_results["conflicts"],
                    "errors": table_results["errors"]
                })
                
                # Update metrics
                self.metrics.record_table_sync(
                    table,
                    table_results["records_synced"],
                    table_results["conflicts"],
                    table_results["errors"]
                )
            
            # Complete transaction
            transaction.complete()
            
            # Update last sync time for databases
            source_info.last_sync = datetime.now()
            target_info.last_sync = datetime.now()
            
            # Save transaction
            self._save_transaction(transaction)
            
            # Update metrics
            sync_duration = self.metrics.end_sync(sync_start)
            self.metrics.add_transaction(transaction.to_dict())
            
            logger.info(f"Synchronization completed in {sync_duration:.2f} seconds")
            
            return {
                "status": "success",
                "transaction_id": transaction.transaction_id,
                "duration": sync_duration,
                "tables_processed": len(tables),
                "conflicts": len(transaction.conflicts),
                "operations": transaction.operations
            }
            
        except Exception as e:
            # Handle failure
            error_msg = str(e)
            logger.error(f"Synchronization failed: {error_msg}")
            
            # Fail transaction
            transaction.fail(error_msg)
            
            # Save failed transaction
            self._save_transaction(transaction)
            
            # Update metrics
            sync_duration = self.metrics.end_sync(sync_start)
            self.metrics.add_transaction(transaction.to_dict())
            self.metrics.error_count += 1
            
            return {
                "status": "error",
                "message": error_msg,
                "transaction_id": transaction.transaction_id,
                "duration": sync_duration
            }
    
    def _sync_table(self, source_db: DatabaseInfo, target_db: DatabaseInfo,
                   table: str, transaction: SyncTransaction, force: bool = False) -> Dict:
        """
        Synchronize a single table between two databases
        
        Args:
            source_db: Source database info
            target_db: Target database info
            table: Table name to synchronize
            transaction: Current sync transaction
            force: Force synchronization even if no changes detected
            
        Returns:
            Dict with table synchronization results
        """
        logger.info(f"Synchronizing table {table} from {source_db.name} to {target_db.name}")
        
        records_synced = 0
        conflicts = 0
        errors = 0
        
        try:
            # Get database connections
            source_conn = source_db.get_connection()
            target_conn = target_db.get_connection()
            
            # Check if table exists in both databases
            source_cursor = source_conn.cursor()
            target_cursor = target_conn.cursor()
            
            # Check if table exists in source
            source_cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}';")
            if not source_cursor.fetchone():
                logger.warning(f"Table {table} does not exist in source database {source_db.name}")
                return {"records_synced": 0, "conflicts": 0, "errors": 1}
            
            # Check if table exists in target
            target_cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}';")
            if not target_cursor.fetchone():
                logger.warning(f"Table {table} does not exist in target database {target_db.name}")
                return {"records_synced": 0, "conflicts": 0, "errors": 1}
            
            # Get the last sync time for this table
            last_sync = None
            if not force and self.config["enable_change_tracking"]:
                last_sync = self.change_tracker.get_last_sync(source_db.name, table)
            
            # Build query to get records from source
            query = f"SELECT * FROM {table}"
            params = []
            
            # Add timestamp filter if available
            if last_sync and not force:
                # Try different timestamp field names
                timestamp_fields = ["updated_at", "timestamp", "modified_at", "updated", "created_at"]
                
                # Check which timestamp field exists in the table
                source_cursor.execute(f"PRAGMA table_info({table})")
                columns = [row[1] for row in source_cursor.fetchall()]
                
                timestamp_field = None
                for field in timestamp_fields:
                    if field in columns:
                        timestamp_field = field
                        break
                
                if timestamp_field:
                    query += f" WHERE {timestamp_field} > ?"
                    # Convert timestamp to appropriate format
                    params.append(last_sync)
            
            # Execute query to get records from source
            source_cursor.execute(query, params)
            source_records = source_cursor.fetchall()
            
            # Get primary key for the table
            source_cursor.execute(f"PRAGMA table_info({table})")
            columns = source_cursor.fetchall()
            primary_key = None
            
            for column in columns:
                if column[5] == 1:  # pk column in sqlite
                    primary_key = column[1]
                    break
            
            # If no primary key defined, use first column or id
            if not primary_key:
                if "id" in [column[1] for column in columns]:
                    primary_key = "id"
                else:
                    primary_key = columns[0][1]
            
            # Process each record
            for source_record in source_records:
                try:
                    # Convert to dict for easier handling
                    source_record_dict = dict(source_record)
                    
                    # Get primary key value
                    pk_value = source_record_dict[primary_key]
                    
                    # Check if record exists in target
                    target_cursor.execute(f"SELECT * FROM {table} WHERE {primary_key} = ?", (pk_value,))
                    target_record = target_cursor.fetchone()
                    
                    if target_record:
                        # Record exists in target - check for changes
                        target_record_dict = dict(target_record)
                        
                        # Skip if record hasn't changed (using hash comparison)
                        if self.config["enable_change_tracking"] and not force:
                            if not self.change_tracker.has_changed(source_db.name, table, pk_value, source_record_dict):
                                continue
                        
                        # Check for conflicts
                        if self._has_conflicts(source_record_dict, target_record_dict):
                            # Resolve conflict
                            resolved_record = self.conflict_resolver.resolve_conflict(
                                table, source_record_dict, target_record_dict, source_db, target_db
                            )
                            
                            # Log conflict
                            conflict_data = {
                                "table": table,
                                "primary_key": {primary_key: pk_value},
                                "source_record": source_record_dict,
                                "target_record": target_record_dict,
                                "resolved_record": resolved_record,
                                "timestamp": datetime.now().isoformat()
                            }
                            transaction.add_conflict(conflict_data)
                            conflicts += 1
                            
                            # Save conflict if configured
                            if self.config["save_conflicts"]:
                                self._save_conflict(conflict_data)
                            
                            # Update target with resolved record
                            self._update_record(target_conn, table, primary_key, pk_value, resolved_record)
                            
                        else:
                            # No conflicts - update target with source record
                            self._update_record(target_conn, table, primary_key, pk_value, source_record_dict)
                        
                    else:
                        # Record doesn't exist in target - insert it
                        self._insert_record(target_conn, table, source_record_dict)
                    
                    records_synced += 1
                    
                except Exception as e:
                    logger.error(f"Error processing record in table {table}: {e}")
                    errors += 1
                    continue
            
            # Commit changes to target database
            target_conn.commit()
            
            # Update change tracking for this table
            if self.config["enable_change_tracking"]:
                self.change_tracker.update_last_sync(source_db.name, table)
                # Save on each table to avoid losing data in case of crash
                self.change_tracker._save_metadata()
            
            return {
                "records_synced": records_synced,
                "conflicts": conflicts,
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"Error synchronizing table {table}: {e}")
            return {
                "records_synced": records_synced,
                "conflicts": conflicts,
                "errors": errors + 1
            }
    
    def _has_conflicts(self, source_record: Dict, target_record: Dict) -> bool:
        """Check if there are conflicts between source and target records"""
        # Compare records field by field
        for field, source_value in source_record.items():
            if field in target_record:
                target_value = target_record[field]
                if source_value != target_value:
                    return True
        
        return False
    
    def _insert_record(self, conn: sqlite3.Connection, table: str, record: Dict):
        """Insert a record into a table"""
        # Build SQL statement
        fields = list(record.keys())
        placeholders = ",".join(["?" for _ in fields])
        field_list = ",".join(fields)
        
        # Prepare values
        values = [record[field] for field in fields]
        
        # Execute insert
        conn.execute(f"INSERT INTO {table} ({field_list}) VALUES ({placeholders})", values)
    
    def _update_record(self, conn: sqlite3.Connection, table: str, pk_field: str, pk_value: Any, record: Dict):
        """Update a record in a table"""
        # Build SQL statement
        set_clauses = []
        values = []
        
        for field, value in record.items():
            if field != pk_field:  # Skip primary key
                set_clauses.append(f"{field} = ?")
                values.append(value)
        
        # Add primary key value for WHERE clause
        values.append(pk_value)
        
        # Execute update
        if set_clauses:
            set_clause = ", ".join(set_clauses)
            conn.execute(f"UPDATE {table} SET {set_clause} WHERE {pk_field} = ?", values)
    
    def _save_transaction(self, transaction: SyncTransaction):
        """Save transaction details to disk"""
        try:
            # Create filename with timestamp and status
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}_{transaction.status}_{transaction.transaction_id}.json"
            filepath = os.path.join(self.config["storage_path"], "transactions", filename)
            
            # Save transaction data
            with open(filepath, 'w') as f:
                json.dump(transaction.to_dict(), f, indent=2)
                
            logger.debug(f"Saved transaction to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save transaction: {e}")
    
    def _save_conflict(self, conflict_data: Dict):
        """Save conflict details to disk"""
        try:
            # Create filename with timestamp and table
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            table = conflict_data["table"]
            pk_field = list(conflict_data["primary_key"].keys())[0]
            pk_value = conflict_data["primary_key"][pk_field]
            
            filename = f"{timestamp}_{table}_{pk_field}_{pk_value}.json"
            filepath = os.path.join(self.config["storage_path"], "conflicts", filename)
            
            # Save conflict data
            with open(filepath, 'w') as f:
                json.dump(conflict_data, f, indent=2)
                
            logger.debug(f"Saved conflict to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save conflict: {e}")
    
    def _sync_worker(self):
        """Background worker for scheduled synchronization"""
        logger.info("Sync worker thread started")
        
        last_full_sync = 0
        
        while self.running:
            try:
                # Check for manual sync requests in queue
                try:
                    request = self.sync_queue.get(block=False)
                    # Process manual sync request
                    self.sync_databases(
                        request["source_db"],
                        request["target_db"],
                        request.get("tables"),
                        request.get("force", False)
                    )
                    self.sync_queue.task_done()
                    continue
                except Exception:
                    # No manual requests, continue with scheduled sync
                    pass
                
                # Check if it's time for a full sync
                current_time = time.time()
                if current_time - last_full_sync >= self.config["sync_interval"]:
                    # Time for full sync
                    logger.info("Starting scheduled full synchronization")
                    
                    # Get all database pairs
                    db_names = list(self.databases.keys())
                    
                    # Sync each pair of databases
                    for source_name in db_names:
                        for target_name in db_names:
                            if source_name != target_name:
                                try:
                                    self.sync_databases(source_name, target_name)
                                except Exception as e:
                                    logger.error(f"Error in scheduled sync {source_name} → {target_name}: {e}")
                    
                    last_full_sync = current_time
                
                # Sleep for a while
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in sync worker: {e}")
                time.sleep(5.0)  # Sleep longer on error
    
    def _monitor_worker(self):
        """Background worker for monitoring system health"""
        logger.info("Monitor thread started")
        
        while self.running:
            try:
                # Check database connections
                for db_name, db_info in self.databases.items():
                    try:
                        conn = db_info.get_connection()
                        cursor = conn.cursor()
                        cursor.execute("SELECT 1")
                        cursor.fetchone()
                    except Exception as e:
                        logger.error(f"Database {db_name} connection error: {e}")
                
                # Save change tracking metadata periodically
                if self.config["enable_change_tracking"]:
                    self.change_tracker._save_metadata()
                
                # Sleep until next check
                time.sleep(self.config["monitor_interval"])
                
            except Exception as e:
                logger.error(f"Error in monitor worker: {e}")
                time.sleep(5.0)  # Sleep longer on error
    
    def schedule_sync(self, source_db: str, target_db: str, tables: Optional[List[str]] = None,
                      force: bool = False):
        """Schedule a synchronization to run in the background"""
        if not self.running:
            self.start()
        
        # Add sync request to queue
        self.sync_queue.put({
            "source_db": source_db,
            "target_db": target_db,
            "tables": tables,
            "force": force
        })
        
        logger.info(f"Scheduled sync from {source_db} to {target_db}")
        return True
    
    def get_status(self) -> Dict:
        """Get current status of the connector"""
        # Count active syncs
        active_syncs = 0
        if self.sync_thread and self.sync_thread.is_alive():
            active_syncs = self.sync_queue.qsize() + (1 if not self.sync_queue.empty() else 0)
        
        return {
            "running": self.running,
            "databases": {name: db.to_dict() for name, db in self.databases.items()},
            "active_syncs": active_syncs,
            "sync_interval": self.config["sync_interval"],
            "change_tracking_enabled": self.config["enable_change_tracking"],
            "metrics": self.metrics.get_metrics() if self.config["enable_metrics"] else None,
            "threads": {
                "sync_thread": self.sync_thread and self.sync_thread.is_alive(),
                "monitor_thread": self.monitor_thread and self.monitor_thread.is_alive()
            },
            "timestamp": datetime.now().isoformat()
        }


# Global instance for module-level access
_connector_instance = None

def get_database_sync_connector(config: Optional[Dict] = None) -> DatabaseSyncConnector:
    """Get the global Database Sync Connector instance"""
    global _connector_instance
    
    if _connector_instance is None:
        _connector_instance = DatabaseSyncConnector(config)
    elif config:
        logger.warning("Connector already initialized, config update ignored")
        
    return _connector_instance


# Example usage when run directly
if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create connector
    connector = get_database_sync_connector({
        "sync_interval": 60,  # 1 minute for testing
        "batch_size": 50,
        "save_conflicts": True,
        "enable_change_tracking": True
    })
    
    # Register example databases
    language_db = DatabaseInfo(
        name="language_db",
        path="data/language/language.db",
        priority=2
    )
    
    central_db = DatabaseInfo(
        name="central_db",
        path="data/neural_metrics.db",
        priority=1
    )
    
    # Register databases
    connector.register_database(language_db)
    connector.register_database(central_db)
    
    # Start connector
    connector.start()
    
    # Schedule a sync
    connector.schedule_sync("language_db", "central_db")
    
    try:
        # Keep running for a while
        for i in range(10):
            print(f"Running... {i+1}/10")
            time.sleep(10)
            
            # Print status every 10 seconds
            status = connector.get_status()
            print(f"Databases: {list(status['databases'].keys())}")
            print(f"Active syncs: {status['active_syncs']}")
            
            if 'metrics' in status and status['metrics']:
                print(f"Records synced: {status['metrics']['record_count']}")
                print(f"Conflicts: {status['metrics']['conflict_count']}")
    finally:
        # Stop connector
        connector.stop()
        print("Database Sync Connector stopped") 