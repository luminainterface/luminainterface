#!/usr/bin/env python3
"""
Persistence Manager

This module implements the persistence manager that handles auto-saving and loading
of system state across all components.
"""

import os
import json
import asyncio
import logging
import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass

from .collectors import create_collector, CollectorConfig

logger = logging.getLogger("persistence_manager")

@dataclass
class PersistenceConfig:
    """Persistence manager configuration"""
    auto_save_interval: int = 300  # seconds
    max_versions: int = 100
    storage_path: str = "storage"
    db_path: str = "storage/persistence.db"
    collector_config: Optional[Dict[str, Any]] = None

class PersistenceManager:
    """Manages system-wide persistence"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = PersistenceConfig(**config if config else {})
        self.collectors = self._initialize_collectors()
        self._setup_storage()
        self._setup_database()
        self._auto_save_task = None
        
    def _initialize_collectors(self) -> Dict[str, Any]:
        """Initialize component collectors"""
        return {
            component_type: create_collector(component_type, self.config.collector_config)
            for component_type in ['bridge', 'neural_seed', 'autowiki', 'spiderweb']
        }
        
    def _setup_storage(self) -> None:
        """Setup storage directories"""
        os.makedirs(self.config.storage_path, exist_ok=True)
        for component in self.collectors.keys():
            os.makedirs(os.path.join(self.config.storage_path, component), exist_ok=True)
            
    def _setup_database(self) -> None:
        """Setup SQLite database"""
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                version INTEGER NOT NULL,
                state_path TEXT NOT NULL,
                checksum TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS component_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                system_state_id INTEGER,
                component_type TEXT NOT NULL,
                state_path TEXT NOT NULL,
                checksum TEXT NOT NULL,
                FOREIGN KEY (system_state_id) REFERENCES system_states(id)
            )
        """)
        
        conn.commit()
        conn.close()
        
    async def start(self) -> None:
        """Start the persistence manager"""
        logger.info("Starting persistence manager")
        self._auto_save_task = asyncio.create_task(self._auto_save_loop())
        
    async def stop(self) -> None:
        """Stop the persistence manager"""
        logger.info("Stopping persistence manager")
        if self._auto_save_task:
            self._auto_save_task.cancel()
            try:
                await self._auto_save_task
            except asyncio.CancelledError:
                pass
            
    async def _auto_save_loop(self) -> None:
        """Auto-save loop"""
        while True:
            try:
                await self.save_all()
                await asyncio.sleep(self.config.auto_save_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-save loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
                
    async def save_all(self) -> Dict[str, Any]:
        """Save all component states"""
        logger.info("Saving all component states")
        timestamp = datetime.now()
        version = await self._get_next_version()
        
        # Collect states from all components
        states = {}
        for component_type, collector in self.collectors.items():
            try:
                state = await collector.collect()
                states[component_type] = state
            except Exception as e:
                logger.error(f"Error collecting state for {component_type}: {e}")
                
        # Save system state
        system_state = {
            'version': version,
            'timestamp': timestamp.isoformat(),
            'components': list(states.keys()),
            'states': states
        }
        
        # Save to storage
        system_state_path = os.path.join(
            self.config.storage_path,
            f"system_state_v{version}.json"
        )
        
        with open(system_state_path, 'w') as f:
            json.dump(system_state, f, indent=2)
            
        # Update database
        await self._update_database(version, timestamp, system_state_path, states)
        
        # Prune old versions
        await self._prune_old_versions()
        
        return system_state
        
    async def load_latest(self) -> Dict[str, Any]:
        """Load the latest system state"""
        version = await self._get_latest_version()
        return await self.load_version(version)
        
    async def load_version(self, version: int) -> Dict[str, Any]:
        """Load a specific system state version"""
        logger.info(f"Loading system state version {version}")
        
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        
        # Get system state
        cursor.execute("""
            SELECT state_path, checksum
            FROM system_states
            WHERE version = ?
        """, (version,))
        result = cursor.fetchone()
        
        if not result:
            raise ValueError(f"Version {version} not found")
            
        state_path, checksum = result
        
        # Verify file exists
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"State file not found: {state_path}")
            
        # Load and verify state
        with open(state_path, 'r') as f:
            state = json.load(f)
            
        # Verify checksum
        if self._calculate_checksum(state) != checksum:
            raise ValueError("State file checksum mismatch")
            
        conn.close()
        return state
        
    async def _get_next_version(self) -> int:
        """Get the next version number"""
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT MAX(version) FROM system_states")
        result = cursor.fetchone()[0]
        
        conn.close()
        return (result or 0) + 1
        
    async def _get_latest_version(self) -> int:
        """Get the latest version number"""
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT MAX(version) FROM system_states")
        result = cursor.fetchone()[0]
        
        conn.close()
        if not result:
            raise ValueError("No versions found")
        return result
        
    async def _update_database(
        self,
        version: int,
        timestamp: datetime,
        state_path: str,
        states: Dict[str, Any]
    ) -> None:
        """Update database with new state information"""
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert system state
            cursor.execute("""
                INSERT INTO system_states (version, timestamp, state_path, checksum)
                VALUES (?, ?, ?, ?)
            """, (
                version,
                timestamp.isoformat(),
                state_path,
                self._calculate_checksum(states)
            ))
            
            system_state_id = cursor.lastrowid
            
            # Insert component states
            for component_type, state in states.items():
                component_path = os.path.join(
                    self.config.storage_path,
                    component_type,
                    f"state_v{version}.json"
                )
                
                with open(component_path, 'w') as f:
                    json.dump(state, f, indent=2)
                    
                cursor.execute("""
                    INSERT INTO component_states
                    (system_state_id, component_type, state_path, checksum)
                    VALUES (?, ?, ?, ?)
                """, (
                    system_state_id,
                    component_type,
                    component_path,
                    self._calculate_checksum(state)
                ))
                
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise e
            
        finally:
            conn.close()
            
    async def _prune_old_versions(self) -> None:
        """Remove old versions exceeding max_versions"""
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        
        try:
            # Get versions to remove
            cursor.execute("""
                SELECT id, state_path
                FROM system_states
                ORDER BY version DESC
                LIMIT -1 OFFSET ?
            """, (self.config.max_versions,))
            
            to_remove = cursor.fetchall()
            
            for state_id, state_path in to_remove:
                # Remove component states
                cursor.execute("""
                    SELECT state_path
                    FROM component_states
                    WHERE system_state_id = ?
                """, (state_id,))
                
                component_paths = cursor.fetchall()
                
                # Remove files
                if os.path.exists(state_path):
                    os.remove(state_path)
                    
                for (component_path,) in component_paths:
                    if os.path.exists(component_path):
                        os.remove(component_path)
                        
                # Remove database entries
                cursor.execute("""
                    DELETE FROM component_states
                    WHERE system_state_id = ?
                """, (state_id,))
                
                cursor.execute("""
                    DELETE FROM system_states
                    WHERE id = ?
                """, (state_id,))
                
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise e
            
        finally:
            conn.close()
            
    def _calculate_checksum(self, data: Any) -> str:
        """Calculate checksum for data"""
        import hashlib
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
        
async def create_persistence_manager(config: Optional[Dict[str, Any]] = None) -> PersistenceManager:
    """Create and initialize a persistence manager"""
    manager = PersistenceManager(config)
    await manager.start()
    return manager 