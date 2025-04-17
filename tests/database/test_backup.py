import pytest
import shutil
import sqlite3
import json
from pathlib import Path
from database.database_manager import DatabaseManager

@pytest.fixture
def backup_db():
    """Create a database instance for backup testing"""
    db_path = Path("backup_node_zero.db")
    backup_path = Path("backup_node_zero.backup.db")
    
    if db_path.exists():
        db_path.unlink()
    if backup_path.exists():
        backup_path.unlink()
    
    db = DatabaseManager()
    db.db_path = db_path
    db._connect()
    db._init_db()
    
    yield db
    
    # Cleanup
    db.close()
    if db_path.exists():
        db_path.unlink()
    if backup_path.exists():
        backup_path.unlink()

def test_backup_creation(backup_db):
    """Test that backups can be created and verified"""
    # Add some test data
    for i in range(100):
        state = {
            'id': f'backup_test_{i}',
            'timestamp': 1234567890.0 + i,
            'state_type': 'backup_test',
            'metrics': {'iteration': i},
            'version_data': {'test': 'backup_verification'}
        }
        backup_db.save_neural_state(state)
    
    # Create backup
    shutil.copy2(backup_db.db_path, backup_db.db_path.with_suffix('.backup.db'))
    
    # Verify backup exists
    assert backup_db.db_path.with_suffix('.backup.db').exists()
    
    # Verify backup integrity
    with sqlite3.connect(str(backup_db.db_path.with_suffix('.backup.db'))) as conn:
        cursor = conn.cursor()
        
        # Check table structure
        cursor.execute("PRAGMA table_info(neural_states)")
        columns = cursor.fetchall()
        assert len(columns) == 5  # id, timestamp, state_type, metrics, version_data
        
        # Check data integrity
        cursor.execute("SELECT COUNT(*) FROM neural_states")
        count = cursor.fetchone()[0]
        assert count == 100
        
        # Verify sample data
        cursor.execute("SELECT * FROM neural_states WHERE id = 'backup_test_0'")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 'backup_test_0'
        assert json.loads(row[3]) == {'iteration': 0}

def test_backup_restore(backup_db):
    """Test that backups can be restored"""
    # Add initial data
    for i in range(50):
        state = {
            'id': f'initial_test_{i}',
            'timestamp': 1234567890.0 + i,
            'state_type': 'initial_test',
            'metrics': {'iteration': i},
            'version_data': {'test': 'initial_data'}
        }
        backup_db.save_neural_state(state)
    
    # Create backup
    shutil.copy2(backup_db.db_path, backup_db.db_path.with_suffix('.backup.db'))
    
    # Add more data
    for i in range(50, 100):
        state = {
            'id': f'new_test_{i}',
            'timestamp': 1234567890.0 + i,
            'state_type': 'new_test',
            'metrics': {'iteration': i},
            'version_data': {'test': 'new_data'}
        }
        backup_db.save_neural_state(state)
    
    # Restore from backup
    backup_db.close()
    shutil.copy2(backup_db.db_path.with_suffix('.backup.db'), backup_db.db_path)
    backup_db._connect()
    
    # Verify restored data
    states = backup_db.get_all_states()
    assert len(states) == 50  # Only initial data should be present
    
    # Verify specific data points
    initial_state = backup_db.get_neural_state('initial_test_0')
    assert initial_state is not None
    assert initial_state['state_type'] == 'initial_test'
    
    new_state = backup_db.get_neural_state('new_test_50')
    assert new_state is None  # New data should not be present

def test_backup_consistency(backup_db):
    """Test that backups maintain data consistency"""
    # Add test data with relationships
    for i in range(10):
        # Create a parent state
        parent_state = {
            'id': f'parent_{i}',
            'timestamp': 1234567890.0 + i,
            'state_type': 'parent',
            'metrics': {'parent_id': i},
            'version_data': {'test': 'parent_data'}
        }
        backup_db.save_neural_state(parent_state)
        
        # Create child states
        for j in range(5):
            child_state = {
                'id': f'child_{i}_{j}',
                'timestamp': 1234567890.0 + i + j,
                'state_type': 'child',
                'metrics': {'parent_id': i, 'child_id': j},
                'version_data': {'test': 'child_data'}
            }
            backup_db.save_neural_state(child_state)
    
    # Create backup
    shutil.copy2(backup_db.db_path, backup_db.db_path.with_suffix('.backup.db'))
    
    # Verify backup consistency
    with sqlite3.connect(str(backup_db.db_path.with_suffix('.backup.db'))) as conn:
        cursor = conn.cursor()
        
        # Check parent-child relationships
        for i in range(10):
            # Verify parent exists
            cursor.execute("SELECT * FROM neural_states WHERE id = ?", (f'parent_{i}',))
            parent = cursor.fetchone()
            assert parent is not None
            
            # Verify all children exist
            cursor.execute(
                "SELECT COUNT(*) FROM neural_states WHERE state_type = 'child' AND json_extract(metrics, '$.parent_id') = ?",
                (i,)
            )
            child_count = cursor.fetchone()[0]
            assert child_count == 5

def test_backup_encryption(backup_db):
    """Test that sensitive data is properly handled in backups"""
    # Add sensitive data
    sensitive_states = [
        {
            'id': 'sensitive_1',
            'timestamp': 1234567890.0,
            'state_type': 'sensitive',
            'metrics': {'api_key': 'secret_key_123', 'other_data': 'normal_data'},
            'version_data': {'test': 'sensitive_data'}
        },
        {
            'id': 'sensitive_2',
            'timestamp': 1234567890.0,
            'state_type': 'sensitive',
            'metrics': {'password': 'secret_password', 'other_data': 'normal_data'},
            'version_data': {'test': 'sensitive_data'}
        }
    ]
    
    for state in sensitive_states:
        backup_db.save_neural_state(state)
    
    # Create backup
    shutil.copy2(backup_db.db_path, backup_db.db_path.with_suffix('.backup.db'))
    
    # Verify sensitive data is not exposed in backup
    with sqlite3.connect(str(backup_db.db_path.with_suffix('.backup.db'))) as conn:
        cursor = conn.cursor()
        
        # Check that sensitive fields are not in plain text
        cursor.execute("SELECT metrics FROM neural_states WHERE state_type = 'sensitive'")
        for row in cursor.fetchall():
            metrics = json.loads(row[0])
            assert 'api_key' not in metrics
            assert 'password' not in metrics
            assert 'other_data' in metrics 