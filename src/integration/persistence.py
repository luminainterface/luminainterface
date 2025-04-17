#!/usr/bin/env python3
"""
Persistence System

This module implements a comprehensive persistence system that handles auto-saving,
saving, and loading across all system components, including bridges, connections,
databases, and neural networks.
"""

import os
import json
import sqlite3
import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("persistence_system")

@dataclass
class PersistenceConfig:
    """Persistence system configuration"""
    auto_save_interval: int = 300  # seconds
    max_versions: int = 5
    compression_enabled: bool = True
    encryption_enabled: bool = True
    sync_timeout: int = 30  # seconds
    base_path: str = "data/persistence"

@dataclass
class SystemState:
    """System state information"""
    timestamp: datetime
    version: str
    components: Dict[str, Any]
    bridges: Dict[str, Any]
    connections: Dict[str, Any]
    metrics: Dict[str, Any]

class PersistenceSystem:
    """
    Comprehensive persistence system for all components.
    
    Features:
    1. Auto-save with configurable intervals
    2. Version control for saved states
    3. Component-specific persistence
    4. Cross-system synchronization
    5. Data integrity verification
    6. Recovery mechanisms
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = PersistenceConfig(**config if config else {})
        self.state_history: List[SystemState] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        self.last_save = None
        self._setup_storage()
        
    def _setup_storage(self):
        """Set up storage directories and databases"""
        # Create base directory
        os.makedirs(self.config.base_path, exist_ok=True)
        
        # Create component directories
        components = ['bridges', 'neural_seed', 'autowiki', 'spiderweb']
        for component in components:
            os.makedirs(os.path.join(self.config.base_path, component), exist_ok=True)
            
        # Initialize SQLite database
        self._init_database()
        
    def _init_database(self):
        """Initialize the SQLite database"""
        db_path = os.path.join(self.config.base_path, 'persistence.db')
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Create state history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS state_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    version TEXT NOT NULL,
                    state_data TEXT NOT NULL,
                    checksum TEXT NOT NULL
                )
            """)
            
            # Create component state table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS component_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component_id TEXT NOT NULL,
                    component_type TEXT NOT NULL,
                    state_data TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    version TEXT NOT NULL,
                    UNIQUE(component_id, version)
                )
            """)
            
            # Create metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component_id TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp DATETIME NOT NULL
                )
            """)
            
            conn.commit()
            
    async def start(self):
        """Start the persistence system"""
        self.running = True
        asyncio.create_task(self._auto_save_loop())
        logger.info("Persistence system started")
        
    async def stop(self):
        """Stop the persistence system"""
        self.running = False
        await self.save_all()  # Final save before stopping
        logger.info("Persistence system stopped")
        
    async def _auto_save_loop(self):
        """Auto-save loop"""
        while self.running:
            try:
                await self.save_all()
                await asyncio.sleep(self.config.auto_save_interval)
            except Exception as e:
                logger.error(f"Auto-save error: {e}")
                await asyncio.sleep(10)  # Short delay before retry
                
    async def save_all(self):
        """Save all system components"""
        try:
            state = await self._collect_system_state()
            await self._save_state(state)
            await self._save_components(state)
            await self._save_metrics(state)
            
            self.last_save = datetime.now()
            self._prune_old_states()
            
            logger.info("System state saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving system state: {e}")
            return False
            
    async def _collect_system_state(self) -> SystemState:
        """Collect current system state"""
        # Collect state from all components concurrently
        tasks = [
            self._collect_bridge_states(),
            self._collect_neural_seed_state(),
            self._collect_autowiki_state(),
            self._collect_spiderweb_state()
        ]
        
        results = await asyncio.gather(*tasks)
        
        return SystemState(
            timestamp=datetime.now(),
            version=self._generate_version(),
            components={
                'neural_seed': results[1],
                'autowiki': results[2],
                'spiderweb': results[3]
            },
            bridges=results[0],
            connections=await self._collect_connections(),
            metrics=await self._collect_metrics()
        )
        
    async def _save_state(self, state: SystemState):
        """Save system state to database"""
        state_data = self._serialize_state(state)
        checksum = self._calculate_checksum(state_data)
        
        db_path = os.path.join(self.config.base_path, 'persistence.db')
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                INSERT INTO state_history (timestamp, version, state_data, checksum)
                VALUES (?, ?, ?, ?)
            """, (state.timestamp, state.version, state_data, checksum))
            await db.commit()
            
    async def _save_components(self, state: SystemState):
        """Save individual component states"""
        tasks = []
        for component_id, component_state in state.components.items():
            tasks.append(self._save_component_state(
                component_id,
                component_state,
                state.version
            ))
            
        await asyncio.gather(*tasks)
        
    async def _save_component_state(self, component_id: str, state: Dict[str, Any], version: str):
        """Save individual component state"""
        state_data = json.dumps(state)
        
        db_path = os.path.join(self.config.base_path, 'persistence.db')
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO component_states 
                (component_id, component_type, state_data, timestamp, version)
                VALUES (?, ?, ?, ?, ?)
            """, (component_id, state.get('type', 'unknown'), state_data, 
                  datetime.now(), version))
            await db.commit()
            
    async def _save_metrics(self, state: SystemState):
        """Save system metrics"""
        tasks = []
        for component_id, metrics in state.metrics.items():
            for metric_type, value in metrics.items():
                tasks.append(self._save_metric(
                    component_id,
                    metric_type,
                    value
                ))
                
        await asyncio.gather(*tasks)
        
    async def _save_metric(self, component_id: str, metric_type: str, value: float):
        """Save individual metric"""
        db_path = os.path.join(self.config.base_path, 'persistence.db')
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                INSERT INTO metrics (component_id, metric_type, value, timestamp)
                VALUES (?, ?, ?, ?)
            """, (component_id, metric_type, value, datetime.now()))
            await db.commit()
            
    async def load_latest(self) -> Optional[SystemState]:
        """Load latest system state"""
        try:
            db_path = os.path.join(self.config.base_path, 'persistence.db')
            async with aiosqlite.connect(db_path) as db:
                async with db.execute("""
                    SELECT timestamp, version, state_data, checksum
                    FROM state_history
                    ORDER BY timestamp DESC
                    LIMIT 1
                """) as cursor:
                    row = await cursor.fetchone()
                    
                    if not row:
                        return None
                        
                    timestamp, version, state_data, checksum = row
                    
                    # Verify checksum
                    if not self._verify_checksum(state_data, checksum):
                        raise ValueError("State data corruption detected")
                        
                    return self._deserialize_state(state_data)
                    
        except Exception as e:
            logger.error(f"Error loading latest state: {e}")
            return None
            
    async def load_version(self, version: str) -> Optional[SystemState]:
        """Load specific version of system state"""
        try:
            db_path = os.path.join(self.config.base_path, 'persistence.db')
            async with aiosqlite.connect(db_path) as db:
                async with db.execute("""
                    SELECT timestamp, version, state_data, checksum
                    FROM state_history
                    WHERE version = ?
                """, (version,)) as cursor:
                    row = await cursor.fetchone()
                    
                    if not row:
                        return None
                        
                    timestamp, version, state_data, checksum = row
                    
                    # Verify checksum
                    if not self._verify_checksum(state_data, checksum):
                        raise ValueError("State data corruption detected")
                        
                    return self._deserialize_state(state_data)
                    
        except Exception as e:
            logger.error(f"Error loading version {version}: {e}")
            return None
            
    def _generate_version(self) -> str:
        """Generate version string"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v_{timestamp}"
        
    def _serialize_state(self, state: SystemState) -> str:
        """Serialize system state"""
        state_dict = {
            'timestamp': state.timestamp.isoformat(),
            'version': state.version,
            'components': state.components,
            'bridges': state.bridges,
            'connections': state.connections,
            'metrics': state.metrics
        }
        return json.dumps(state_dict)
        
    def _deserialize_state(self, state_data: str) -> SystemState:
        """Deserialize system state"""
        state_dict = json.loads(state_data)
        return SystemState(
            timestamp=datetime.fromisoformat(state_dict['timestamp']),
            version=state_dict['version'],
            components=state_dict['components'],
            bridges=state_dict['bridges'],
            connections=state_dict['connections'],
            metrics=state_dict['metrics']
        )
        
    def _calculate_checksum(self, data: str) -> str:
        """Calculate checksum for data integrity"""
        import hashlib
        return hashlib.sha256(data.encode()).hexdigest()
        
    def _verify_checksum(self, data: str, checksum: str) -> bool:
        """Verify data integrity using checksum"""
        return self._calculate_checksum(data) == checksum
        
    def _prune_old_states(self):
        """Remove old states exceeding max_versions"""
        try:
            db_path = os.path.join(self.config.base_path, 'persistence.db')
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Keep only max_versions number of most recent states
                cursor.execute("""
                    DELETE FROM state_history
                    WHERE id NOT IN (
                        SELECT id FROM state_history
                        ORDER BY timestamp DESC
                        LIMIT ?
                    )
                """, (self.config.max_versions,))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error pruning old states: {e}")
            
    async def _collect_bridge_states(self) -> Dict[str, Any]:
        """Collect states from all bridges"""
        # Implement bridge state collection
        return {}
        
    async def _collect_neural_seed_state(self) -> Dict[str, Any]:
        """Collect Neural Seed state"""
        # Implement Neural Seed state collection
        return {}
        
    async def _collect_autowiki_state(self) -> Dict[str, Any]:
        """Collect AutoWiki state"""
        # Implement AutoWiki state collection
        return {}
        
    async def _collect_spiderweb_state(self) -> Dict[str, Any]:
        """Collect Spiderweb state"""
        # Implement Spiderweb state collection
        return {}
        
    async def _collect_connections(self) -> Dict[str, Any]:
        """Collect connection states"""
        # Implement connection state collection
        return {}
        
    async def _collect_metrics(self) -> Dict[str, Any]:
        """Collect system metrics"""
        # Implement metrics collection
        return {} 