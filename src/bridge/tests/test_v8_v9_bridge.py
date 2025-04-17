#!/usr/bin/env python3
"""
Unit tests for the v8-v9 Neural Network Bridge
"""

import os
import sys
import json
import unittest
import tempfile
import sqlite3
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from bridge.v8_v9_bridge import V8V9Bridge

class TestV8V9Bridge(unittest.TestCase):
    """Test cases for the V8V9Bridge class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directories for test data
        self.temp_dir = tempfile.mkdtemp()
        self.v8_data_path = os.path.join(self.temp_dir, "v8")
        self.v9_data_path = os.path.join(self.temp_dir, "v9")
        
        # Create directory structure
        os.makedirs(os.path.join(self.v8_data_path, "states"), exist_ok=True)
        os.makedirs(os.path.join(self.v9_data_path, "states"), exist_ok=True)
        
        # Initialize bridge with test paths
        self.bridge = V8V9Bridge(
            v8_data_path=self.v8_data_path,
            v9_data_path=self.v9_data_path,
            create_backups=False
        )
        
        # Create test data
        self._create_test_databases()
        self._create_test_state_files()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_databases(self):
        """Create test databases for v8 and v9"""
        # Create v8 database
        v8_db_path = os.path.join(self.v8_data_path, "neural_database.db")
        conn = sqlite3.connect(v8_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            duration INTEGER,
            play_type TEXT,
            intensity REAL,
            consciousness_peak REAL,
            patterns_detected INTEGER,
            total_activations INTEGER,
            metadata TEXT
        )
        ''')
        
        cursor.execute('''
        INSERT INTO sessions VALUES 
        ('session_1', '2023-01-01T12:00:00', 100, 'free', 0.7, 0.85, 10, 500, '{"notes": "v8 test session"}')
        ''')
        
        conn.commit()
        conn.close()
        
        # Create v9 database
        v9_db_path = os.path.join(self.v9_data_path, "neural_database.db")
        conn = sqlite3.connect(v9_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            duration INTEGER,
            play_type TEXT,
            intensity REAL,
            breathing_pattern TEXT,
            consciousness_peak REAL,
            patterns_detected INTEGER,
            total_activations INTEGER,
            metadata TEXT
        )
        ''')
        
        cursor.execute('''
        INSERT INTO sessions VALUES 
        ('session_2', '2023-02-01T12:00:00', 200, 'guided', 0.8, 'calm', 0.9, 20, 1000, '{"notes": "v9 test session"}')
        ''')
        
        conn.commit()
        conn.close()
    
    def _create_test_state_files(self):
        """Create test state files for v8 and v9"""
        # Create v8 state file
        v8_state = {
            "neurons": [{"id": 1, "connections": [2, 3]}, {"id": 2, "connections": [1]}, {"id": 3, "connections": [1]}],
            "activation_history": [0.1, 0.3, 0.5],
            "metadata": {"version": "8.0", "created": "2023-01-01T12:00:00"}
        }
        
        with open(os.path.join(self.v8_data_path, "states", "v8_state.json"), "w") as f:
            json.dump(v8_state, f)
        
        # Create v9 state file
        v9_state = {
            "neurons": [{"id": 1, "connections": [2, 3]}, {"id": 2, "connections": [1]}, {"id": 3, "connections": [1]}],
            "activation_history": [0.1, 0.3, 0.5],
            "breathing": {"pattern": "calm", "coherence": 0.8, "rate": 6.0},
            "growth": {"neurons_created_total": 5, "neurons_pruned_total": 2},
            "metadata": {"version": "9.0", "created": "2023-02-01T12:00:00"}
        }
        
        with open(os.path.join(self.v9_data_path, "states", "v9_state.json"), "w") as f:
            json.dump(v9_state, f)
    
    def test_create_backup(self):
        """Test the create_backup method"""
        # Enable backups for this test
        self.bridge.create_backups = True
        
        # Create a test file
        test_file = os.path.join(self.temp_dir, "test_file.txt")
        with open(test_file, "w") as f:
            f.write("test content")
        
        # Create backup
        backup_path = self.bridge.create_backup(Path(test_file), "test")
        
        # Check backup was created and contains correct content
        self.assertTrue(os.path.exists(backup_path))
        with open(backup_path, "r") as f:
            self.assertEqual(f.read(), "test content")
    
    def test_migrate_database_v8_to_v9(self):
        """Test database migration from v8 to v9"""
        # Run migration
        result = self.bridge.migrate_database(direction="v8_to_v9")
        
        # Check success
        self.assertTrue(result)
        
        # Verify migration results
        conn = sqlite3.connect(os.path.join(self.v9_data_path, "neural_database.db"))
        cursor = conn.cursor()
        
        # Check if v8 session was migrated
        cursor.execute("SELECT id, play_type, consciousness_peak, breathing_pattern FROM sessions WHERE id = 'session_1'")
        row = cursor.fetchone()
        
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "session_1")
        self.assertEqual(row[1], "free")
        self.assertEqual(row[2], 0.85)
        # Check that breathing_pattern was added with default value
        self.assertEqual(row[3], "unknown")
        
        conn.close()
    
    def test_migrate_database_v9_to_v8(self):
        """Test database migration from v9 to v8"""
        # Run migration
        result = self.bridge.migrate_database(direction="v9_to_v8")
        
        # Check success
        self.assertTrue(result)
        
        # Verify migration results
        conn = sqlite3.connect(os.path.join(self.v8_data_path, "neural_database.db"))
        cursor = conn.cursor()
        
        # Check if v9 session was migrated (without breathing_pattern)
        cursor.execute("SELECT id, play_type, consciousness_peak FROM sessions WHERE id = 'session_2'")
        row = cursor.fetchone()
        
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "session_2")
        self.assertEqual(row[1], "guided")
        self.assertEqual(row[2], 0.9)
        
        # Verify breathing_pattern column doesn't exist in v8
        cursor.execute("PRAGMA table_info(sessions)")
        columns = [row[1] for row in cursor.fetchall()]
        self.assertNotIn("breathing_pattern", columns)
        
        conn.close()
    
    def test_migrate_neural_states_v8_to_v9(self):
        """Test neural state migration from v8 to v9"""
        # Run migration
        result = self.bridge.migrate_neural_states(direction="v8_to_v9")
        
        # Check success
        self.assertTrue(result)
        
        # Verify migration results
        v9_state_path = os.path.join(self.v9_data_path, "states", "v8_state.json")
        self.assertTrue(os.path.exists(v9_state_path))
        
        with open(v9_state_path, "r") as f:
            migrated_state = json.load(f)
        
        # Check v9-specific fields were added
        self.assertIn("breathing", migrated_state)
        self.assertIn("growth", migrated_state)
        self.assertEqual(migrated_state["metadata"]["migrated_from"], "v8")
    
    def test_migrate_neural_states_v9_to_v8(self):
        """Test neural state migration from v9 to v8"""
        # Run migration
        result = self.bridge.migrate_neural_states(direction="v9_to_v8")
        
        # Check success
        self.assertTrue(result)
        
        # Verify migration results
        v8_state_path = os.path.join(self.v8_data_path, "states", "v9_state.json")
        self.assertTrue(os.path.exists(v8_state_path))
        
        with open(v8_state_path, "r") as f:
            migrated_state = json.load(f)
        
        # Check v9-specific fields were removed
        self.assertNotIn("breathing", migrated_state)
        self.assertNotIn("growth", migrated_state)
        self.assertEqual(migrated_state["metadata"]["migrated_from"], "v9")
    
    @patch("src.bridge.v8_v9_bridge.import_v9_components")
    @patch("src.bridge.v8_v9_bridge.import_v8_components")
    def test_sync_databases(self, mock_import_v8, mock_import_v9):
        """Test database synchronization"""
        # Mock components
        mock_import_v8.return_value = {"NeuralPlayground": MagicMock()}
        mock_import_v9.return_value = {"IntegratedNeuralPlayground": MagicMock()}
        
        # Test sync with v9 as primary
        result = self.bridge.sync_databases(primary="v9")
        self.assertTrue(result)
        
        # Test sync with v8 as primary
        result = self.bridge.sync_databases(primary="v8")
        self.assertTrue(result)
    
    def test_run_compatibility_test(self):
        """Test compatibility testing"""
        # Run compatibility test
        results = self.bridge.run_compatibility_test()
        
        # Check results structure
        self.assertIn("v8_db_exists", results)
        self.assertIn("v9_db_exists", results)
        self.assertIn("v8_states_dir_exists", results)
        self.assertIn("v9_states_dir_exists", results)
        self.assertIn("tests", results)
    
    def test_transform_state_functions(self):
        """Test state transformation functions"""
        # Test v8 to v9 transformation
        v8_state = {
            "neurons": [{"id": 1, "connections": [2]}],
            "metadata": {"version": "8.0"}
        }
        
        v9_state = self.bridge._transform_v8_to_v9_state(v8_state)
        
        # Check v9-specific fields
        self.assertIn("breathing", v9_state)
        self.assertIn("growth", v9_state)
        self.assertEqual(v9_state["metadata"]["version"], "9.0")
        
        # Test v9 to v8 transformation
        v9_state = {
            "neurons": [{"id": 1, "connections": [2]}],
            "breathing": {"pattern": "calm"},
            "growth": {"neurons_created_total": 5},
            "metadata": {"version": "9.0"}
        }
        
        v8_state = self.bridge._transform_v9_to_v8_state(v9_state)
        
        # Check v9-specific fields were removed
        self.assertNotIn("breathing", v8_state)
        self.assertNotIn("growth", v8_state)
        self.assertEqual(v8_state["metadata"]["version"], "8.0")

if __name__ == "__main__":
    unittest.main() 