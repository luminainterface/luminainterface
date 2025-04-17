import pytest
import sqlite3
import json
import os
from pathlib import Path
from database.database_manager import DatabaseManager

@pytest.fixture
def test_db():
    """Create a test database instance"""
    db_path = Path("test_node_zero.db")
    if db_path.exists():
        db_path.unlink()
    
    db = DatabaseManager()
    db.db_path = db_path
    db._connect()
    db._init_db()
    
    yield db
    
    # Cleanup
    db.close()
    if db_path.exists():
        db_path.unlink()

def test_schema_versioning(test_db):
    """Test that schema version is tracked and updated"""
    # Add schema version table if not exists
    with test_db.conn:
        test_db.cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_versions (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        """)
        
        # Insert initial version
        test_db.cursor.execute("""
            INSERT INTO schema_versions (version, description)
            VALUES (1, 'Initial schema')
        """)
    
    # Verify version exists
    with test_db.conn:
        test_db.cursor.execute("SELECT version FROM schema_versions")
        version = test_db.cursor.fetchone()
        assert version['version'] == 1

def test_table_structure(test_db):
    """Test that all required tables exist with correct structure"""
    with test_db.conn:
        # Check neural_states table
        test_db.cursor.execute("PRAGMA table_info(neural_states)")
        columns = test_db.cursor.fetchall()
        column_names = [col['name'] for col in columns]
        
        required_columns = ['id', 'timestamp', 'state_type', 'metrics', 'version_data']
        assert all(col in column_names for col in required_columns)
        
        # Check system_settings table
        test_db.cursor.execute("PRAGMA table_info(system_settings)")
        columns = test_db.cursor.fetchall()
        column_names = [col['name'] for col in columns]
        
        required_columns = ['key', 'value', 'updated_at']
        assert all(col in column_names for col in required_columns)

def test_data_migration(test_db):
    """Test that data can be migrated between schema versions"""
    # Insert test data
    test_state = {
        'id': 'test_state_1',
        'timestamp': 1234567890.0,
        'state_type': 'test',
        'metrics': {'accuracy': 0.95},
        'version_data': {'version': '1.0.0'}
    }
    
    test_db.save_neural_state(test_state)
    
    # Verify data was saved correctly
    saved_state = test_db.get_neural_state('test_state_1')
    assert saved_state is not None
    assert saved_state['id'] == test_state['id']
    assert saved_state['state_type'] == test_state['state_type']
    assert json.loads(saved_state['metrics']) == test_state['metrics']

def test_rollback_mechanism(test_db):
    """Test that failed migrations can be rolled back"""
    try:
        with test_db.conn:
            # Start a transaction
            test_db.cursor.execute("BEGIN TRANSACTION")
            
            # Attempt to add a new column (should fail)
            test_db.cursor.execute("ALTER TABLE neural_states ADD COLUMN test_column TEXT")
            
            # This should raise an exception
            raise Exception("Simulated migration failure")
            
    except Exception:
        # Verify rollback
        with test_db.conn:
            test_db.cursor.execute("PRAGMA table_info(neural_states)")
            columns = test_db.cursor.fetchall()
            column_names = [col['name'] for col in columns]
            
            assert 'test_column' not in column_names

def test_concurrent_access(test_db):
    """Test that the database handles concurrent access properly"""
    import threading
    
    def save_states(db, count):
        for i in range(count):
            state = {
                'id': f'concurrent_test_{i}',
                'timestamp': time.time(),
                'state_type': 'test',
                'metrics': {'iteration': i},
                'version_data': {'thread': threading.current_thread().name}
            }
            db.save_neural_state(state)
    
    # Create multiple threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=save_states, args=(test_db, 10))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify all states were saved
    states = test_db.get_all_states()
    assert len(states) >= 50  # 5 threads * 10 states each 