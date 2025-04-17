#!/usr/bin/env python3
"""
Tests for the persistence manager
"""

import os
import json
import pytest
import asyncio
import tempfile
from datetime import datetime
from pathlib import Path

from ..persistence_manager import PersistenceManager, PersistenceConfig
from ..collectors import CollectorConfig

@pytest.fixture
async def persistence_manager():
    """Create a persistence manager with temporary storage"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = PersistenceConfig(
            storage_path=os.path.join(temp_dir, "storage"),
            db_path=os.path.join(temp_dir, "storage", "test.db"),
            auto_save_interval=1,  # 1 second for testing
            max_versions=5,
            collector_config=CollectorConfig(
                max_history=10,
                include_metrics=True
            )
        )
        
        manager = PersistenceManager(config)
        await manager.start()
        yield manager
        await manager.stop()

@pytest.mark.asyncio
async def test_initialization(persistence_manager):
    """Test persistence manager initialization"""
    assert persistence_manager is not None
    assert os.path.exists(persistence_manager.config.storage_path)
    assert os.path.exists(persistence_manager.config.db_path)
    assert len(persistence_manager.collectors) == 4  # bridge, neural_seed, autowiki, spiderweb

@pytest.mark.asyncio
async def test_save_and_load(persistence_manager):
    """Test saving and loading system state"""
    # Save initial state
    initial_state = await persistence_manager.save_all()
    assert initial_state is not None
    assert 'version' in initial_state
    assert 'timestamp' in initial_state
    assert 'components' in initial_state
    assert 'states' in initial_state
    
    # Load latest state
    loaded_state = await persistence_manager.load_latest()
    assert loaded_state is not None
    assert loaded_state['version'] == initial_state['version']
    assert loaded_state['timestamp'] == initial_state['timestamp']
    assert loaded_state['components'] == initial_state['components']
    
    # Verify component states
    for component in initial_state['components']:
        assert component in loaded_state['states']
        assert 'state' in loaded_state['states'][component]
        assert 'metrics' in loaded_state['states'][component]
        assert 'history' in loaded_state['states'][component]

@pytest.mark.asyncio
async def test_version_management(persistence_manager):
    """Test version management and pruning"""
    # Save multiple versions
    versions = []
    for _ in range(10):
        state = await persistence_manager.save_all()
        versions.append(state['version'])
        await asyncio.sleep(0.1)  # Ensure different timestamps
        
    # Verify only max_versions are kept
    latest_state = await persistence_manager.load_latest()
    assert latest_state['version'] == versions[-1]
    
    # Try to load oldest version (should be pruned)
    with pytest.raises(ValueError):
        await persistence_manager.load_version(versions[0])

@pytest.mark.asyncio
async def test_auto_save(persistence_manager):
    """Test auto-save functionality"""
    # Wait for auto-save to trigger
    await asyncio.sleep(2)  # Wait for 2 auto-save intervals
    
    # Verify state was saved
    latest_state = await persistence_manager.load_latest()
    assert latest_state is not None
    assert latest_state['version'] > 1

@pytest.mark.asyncio
async def test_error_handling(persistence_manager):
    """Test error handling in persistence operations"""
    # Test loading non-existent version
    with pytest.raises(ValueError):
        await persistence_manager.load_version(999)
        
    # Test loading with corrupted file
    latest_state = await persistence_manager.load_latest()
    state_path = os.path.join(
        persistence_manager.config.storage_path,
        f"system_state_v{latest_state['version']}.json"
    )
    
    # Corrupt the file
    with open(state_path, 'w') as f:
        f.write("corrupted data")
        
    # Verify checksum validation
    with pytest.raises(ValueError):
        await persistence_manager.load_version(latest_state['version'])

@pytest.mark.asyncio
async def test_component_state_management(persistence_manager):
    """Test component-specific state management"""
    # Save state
    state = await persistence_manager.save_all()
    
    # Verify component states
    for component in state['components']:
        component_path = os.path.join(
            persistence_manager.config.storage_path,
            component,
            f"state_v{state['version']}.json"
        )
        assert os.path.exists(component_path)
        
        with open(component_path, 'r') as f:
            component_state = json.load(f)
            assert 'state' in component_state
            assert 'metrics' in component_state
            assert 'history' in component_state

@pytest.mark.asyncio
async def test_database_integrity(persistence_manager):
    """Test database integrity and relationships"""
    # Save state
    state = await persistence_manager.save_all()
    
    # Verify database entries
    import sqlite3
    conn = sqlite3.connect(persistence_manager.config.db_path)
    cursor = conn.cursor()
    
    # Check system state
    cursor.execute("""
        SELECT id, version, timestamp, state_path, checksum
        FROM system_states
        WHERE version = ?
    """, (state['version'],))
    system_state = cursor.fetchone()
    assert system_state is not None
    
    # Check component states
    cursor.execute("""
        SELECT component_type, state_path, checksum
        FROM component_states
        WHERE system_state_id = ?
    """, (system_state[0],))
    component_states = cursor.fetchall()
    assert len(component_states) == len(state['components'])
    
    conn.close()

if __name__ == '__main__':
    pytest.main([__file__]) 