#!/usr/bin/env python3
"""
v8-v9 Neural Network Bridge Module

This module provides bridging functionality between v8 and v9 of the Lumina Neural Network system.
It enables data transfer, compatibility layers, and migration tools to ensure seamless
integration between versions, with special focus on neural playground and breathing integration.

Key features:
- Neural state and configuration transfer between v8 and v9
- Database migration and synchronization
- Compatibility layers for API differences
- Session history preservation
- Neural structure mapping
"""

import os
import sys
import json
import logging
import shutil
import sqlite3
import importlib
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bridge.v8_v9")

# Import v9 components dynamically to avoid dependency issues
def import_v9_components():
    """Dynamically import v9 components to avoid import errors if not available"""
    components = {}
    try:
        from src.v9 import IntegratedNeuralPlayground, neural_database, BreathingPattern
        components["IntegratedNeuralPlayground"] = IntegratedNeuralPlayground
        components["neural_database"] = neural_database
        components["BreathingPattern"] = BreathingPattern
        logger.info("Successfully imported v9 components")
    except ImportError as e:
        logger.warning(f"Could not import v9 components: {e}")
    return components

# Import v8 components dynamically
def import_v8_components():
    """Dynamically import v8 components to avoid import errors if not available"""
    components = {}
    try:
        from src.v8 import NeuralPlayground, neural_database
        components["NeuralPlayground"] = NeuralPlayground
        components["neural_database"] = neural_database
        logger.info("Successfully imported v8 components")
    except ImportError as e:
        logger.warning(f"Could not import v8 components: {e}")
    return components

class V8V9Bridge:
    """
    Bridge class for v8-v9 integration
    
    This class handles the transfer of data, configuration, and neural structures
    between v8 and v9 versions of the Lumina Neural Network system.
    """
    
    def __init__(self, 
                 v8_data_path: str = "data/v8", 
                 v9_data_path: str = "data/v9",
                 create_backups: bool = True):
        """
        Initialize the bridge
        
        Args:
            v8_data_path: Path to v8 data directory
            v9_data_path: Path to v9 data directory
            create_backups: Whether to create backups before operations
        """
        self.v8_data_path = Path(v8_data_path)
        self.v9_data_path = Path(v9_data_path)
        self.create_backups = create_backups
        
        # Ensure data directories exist
        self.v8_data_path.mkdir(parents=True, exist_ok=True)
        self.v9_data_path.mkdir(parents=True, exist_ok=True)
        
        # Import components
        self.v8 = import_v8_components()
        self.v9 = import_v9_components()
        
        # Setup database paths
        self.v8_db_path = self.v8_data_path / "neural_database.db"
        self.v9_db_path = self.v9_data_path / "neural_database.db"
        
        logger.info(f"Bridge initialized with v8 path: {v8_data_path}, v9 path: {v9_data_path}")
    
    def create_backup(self, source_path: Path, label: str = "backup"):
        """Create a backup of a file or directory"""
        if not self.create_backups:
            return None
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if source_path.is_file():
            backup_path = source_path.with_suffix(f".{label}_{timestamp}{source_path.suffix}")
            shutil.copy2(source_path, backup_path)
        elif source_path.is_dir():
            backup_path = source_path.with_name(f"{source_path.name}_{label}_{timestamp}")
            shutil.copytree(source_path, backup_path)
        else:
            return None
            
        logger.info(f"Created backup at {backup_path}")
        return backup_path
    
    def migrate_database(self, direction: str = "v8_to_v9") -> bool:
        """
        Migrate database between v8 and v9
        
        Args:
            direction: Migration direction, either "v8_to_v9" or "v9_to_v8"
            
        Returns:
            Success status
        """
        if direction == "v8_to_v9":
            source_db = self.v8_db_path
            target_db = self.v9_db_path
            source_version = "v8"
            target_version = "v9"
        else:  # v9_to_v8
            source_db = self.v9_db_path
            target_db = self.v8_db_path
            source_version = "v9"
            target_version = "v8"
        
        # Check if source database exists
        if not source_db.exists():
            logger.error(f"Source database {source_db} does not exist")
            return False
        
        # Create backup of target database if it exists
        if target_db.exists():
            self.create_backup(target_db)
        
        try:
            # Connect to databases
            source_conn = sqlite3.connect(source_db)
            source_cursor = source_conn.cursor()
            
            # Get tables from source
            source_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in source_cursor.fetchall() 
                      if row[0] != 'sqlite_sequence']
            
            # Create target database if it doesn't exist
            target_conn = sqlite3.connect(target_db)
            target_cursor = target_conn.cursor()
            
            # Migrate each table
            for table in tables:
                # Get schema
                source_cursor.execute(f"SELECT sql FROM sqlite_master WHERE name='{table}';")
                create_sql = source_cursor.fetchone()[0]
                
                # Make any necessary schema adjustments for version compatibility
                if direction == "v8_to_v9" and table == "sessions":
                    # Add v9-specific columns if migrating from v8 to v9
                    if "breathing_pattern" not in create_sql:
                        create_sql = create_sql.replace(
                            "consciousness_peak REAL",
                            "consciousness_peak REAL, breathing_pattern TEXT"
                        )
                
                # Drop existing table in target
                target_cursor.execute(f"DROP TABLE IF EXISTS {table};")
                
                # Create table in target
                target_cursor.execute(create_sql)
                
                # Get data from source
                source_cursor.execute(f"SELECT * FROM {table};")
                rows = source_cursor.fetchall()
                
                if rows:
                    # Get column names
                    source_cursor.execute(f"PRAGMA table_info({table});")
                    source_columns = [row[1] for row in source_cursor.fetchall()]
                    
                    target_cursor.execute(f"PRAGMA table_info({table});")
                    target_columns = [row[1] for row in target_cursor.fetchall()]
                    
                    # Find common columns
                    common_columns = [col for col in source_columns if col in target_columns]
                    
                    # Create placeholders for SQL insert
                    placeholders = ", ".join(["?"] * len(common_columns))
                    columns_str = ", ".join(common_columns)
                    
                    # Insert data into target
                    for row in rows:
                        # Extract only common columns
                        values = [row[source_columns.index(col)] for col in common_columns]
                        
                        # Add missing default values for v9-specific columns if needed
                        if direction == "v8_to_v9" and table == "sessions":
                            if "breathing_pattern" in target_columns and "breathing_pattern" not in common_columns:
                                values.append("unknown")
                        
                        # Insert into target
                        target_cursor.execute(
                            f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})",
                            values
                        )
            
            # Commit changes
            target_conn.commit()
            
            # Close connections
            source_conn.close()
            target_conn.close()
            
            logger.info(f"Successfully migrated database from {source_version} to {target_version}")
            return True
        
        except Exception as e:
            logger.error(f"Database migration error: {e}")
            return False
    
    def migrate_neural_states(self, direction: str = "v8_to_v9") -> bool:
        """
        Migrate neural state files between v8 and v9
        
        Args:
            direction: Migration direction, either "v8_to_v9" or "v9_to_v8"
            
        Returns:
            Success status
        """
        if direction == "v8_to_v9":
            source_dir = self.v8_data_path / "states"
            target_dir = self.v9_data_path / "states"
        else:  # v9_to_v8
            source_dir = self.v9_data_path / "states"
            target_dir = self.v8_data_path / "states"
        
        # Check if source directory exists
        if not source_dir.exists():
            logger.error(f"Source states directory {source_dir} does not exist")
            return False
        
        # Create target directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get all state files
            state_files = list(source_dir.glob("*.json"))
            
            for state_file in state_files:
                # Read the state file
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                
                # Transform state data for compatibility
                if direction == "v8_to_v9":
                    transformed_data = self._transform_v8_to_v9_state(state_data)
                else:  # v9_to_v8
                    transformed_data = self._transform_v9_to_v8_state(state_data)
                
                # Write transformed data to target
                target_file = target_dir / state_file.name
                with open(target_file, 'w') as f:
                    json.dump(transformed_data, f, indent=2)
                
                logger.info(f"Migrated state file: {state_file.name}")
            
            logger.info(f"Successfully migrated {len(state_files)} state files")
            return True
        
        except Exception as e:
            logger.error(f"Neural state migration error: {e}")
            return False
    
    def _transform_v8_to_v9_state(self, state_data: Dict) -> Dict:
        """Transform v8 state format to v9 format"""
        # Create a deep copy to avoid modifying the original
        transformed = state_data.copy()
        
        # Add v9-specific structures if they don't exist
        if "breathing" not in transformed:
            transformed["breathing"] = {
                "pattern": "calm",
                "coherence": 0.5,
                "rate": 12.0
            }
        
        if "growth" not in transformed:
            transformed["growth"] = {
                "neurons_created_total": 0,
                "neurons_pruned_total": 0,
                "regions_formed_total": 0,
                "growth_cycles_total": 0,
                "neuron_growth_by_pattern": {
                    "calm": 0,
                    "focused": 0,
                    "meditative": 0,
                    "excited": 0
                }
            }
        
        # Update metadata
        if "metadata" not in transformed:
            transformed["metadata"] = {}
        
        transformed["metadata"]["version"] = "9.0"
        transformed["metadata"]["migrated_from"] = "v8"
        transformed["metadata"]["migration_time"] = time.time()
        
        return transformed
    
    def _transform_v9_to_v8_state(self, state_data: Dict) -> Dict:
        """Transform v9 state format to v8 format"""
        # Create a deep copy to avoid modifying the original
        transformed = state_data.copy()
        
        # Remove v9-specific structures
        if "breathing" in transformed:
            del transformed["breathing"]
        
        if "growth" in transformed:
            del transformed["growth"]
        
        # Update metadata
        if "metadata" not in transformed:
            transformed["metadata"] = {}
        
        transformed["metadata"]["version"] = "8.0"
        transformed["metadata"]["migrated_from"] = "v9"
        transformed["metadata"]["migration_time"] = time.time()
        
        return transformed
    
    def create_v9_instance_from_v8(self, v8_state_path: Optional[str] = None) -> Any:
        """
        Create a v9 IntegratedNeuralPlayground instance based on v8 data
        
        Args:
            v8_state_path: Optional path to v8 state file to load
            
        Returns:
            v9 IntegratedNeuralPlayground instance or None if creation fails
        """
        if "IntegratedNeuralPlayground" not in self.v9:
            logger.error("v9 IntegratedNeuralPlayground not available")
            return None
        
        try:
            # Create v9 instance
            v9_instance = self.v9["IntegratedNeuralPlayground"](
                size=100,  # Default size, will be overridden if state is loaded
                breathing_pattern=self.v9.get("BreathingPattern", {}).get("CALM", "calm"),
                growth_rate=0.05
            )
            
            # If state path provided, load the v8 state
            if v8_state_path:
                # First check if the file exists
                if not os.path.exists(v8_state_path):
                    logger.error(f"v8 state file {v8_state_path} does not exist")
                    return v9_instance
                
                # Read v8 state
                with open(v8_state_path, 'r') as f:
                    v8_state = json.load(f)
                
                # Transform to v9 format
                v9_state = self._transform_v8_to_v9_state(v8_state)
                
                # Create temporary v9 state file
                temp_v9_state_path = self.v9_data_path / f"temp_v9_state_{int(time.time())}.json"
                with open(temp_v9_state_path, 'w') as f:
                    json.dump(v9_state, f, indent=2)
                
                # Load the transformed state
                v9_instance.load_state(str(temp_v9_state_path))
                
                # Clean up temporary file
                os.remove(temp_v9_state_path)
                
                logger.info(f"Created v9 instance from v8 state: {v8_state_path}")
            
            return v9_instance
        
        except Exception as e:
            logger.error(f"Error creating v9 instance from v8: {e}")
            return None
    
    def create_v8_instance_from_v9(self, v9_state_path: Optional[str] = None) -> Any:
        """
        Create a v8 NeuralPlayground instance based on v9 data
        
        Args:
            v9_state_path: Optional path to v9 state file to load
            
        Returns:
            v8 NeuralPlayground instance or None if creation fails
        """
        if "NeuralPlayground" not in self.v8:
            logger.error("v8 NeuralPlayground not available")
            return None
        
        try:
            # Create v8 instance
            v8_instance = self.v8["NeuralPlayground"](
                neuron_count=100  # Default size, will be overridden if state is loaded
            )
            
            # If state path provided, load the v9 state
            if v9_state_path:
                # First check if the file exists
                if not os.path.exists(v9_state_path):
                    logger.error(f"v9 state file {v9_state_path} does not exist")
                    return v8_instance
                
                # Read v9 state
                with open(v9_state_path, 'r') as f:
                    v9_state = json.load(f)
                
                # Transform to v8 format
                v8_state = self._transform_v9_to_v8_state(v9_state)
                
                # Create temporary v8 state file
                temp_v8_state_path = self.v8_data_path / f"temp_v8_state_{int(time.time())}.json"
                with open(temp_v8_state_path, 'w') as f:
                    json.dump(v8_state, f, indent=2)
                
                # Load the transformed state
                v8_instance.load_state(str(temp_v8_state_path))
                
                # Clean up temporary file
                os.remove(temp_v8_state_path)
                
                logger.info(f"Created v8 instance from v9 state: {v9_state_path}")
            
            return v8_instance
        
        except Exception as e:
            logger.error(f"Error creating v8 instance from v9: {e}")
            return None

    def sync_databases(self, primary: str = "v9") -> bool:
        """
        Synchronize databases between v8 and v9
        
        Args:
            primary: Primary version to use as source ("v8" or "v9")
            
        Returns:
            Success status
        """
        if primary == "v9":
            return self.migrate_database(direction="v9_to_v8")
        else:
            return self.migrate_database(direction="v8_to_v9")
    
    def run_compatibility_test(self) -> Dict:
        """
        Run compatibility tests between v8 and v9
        
        Returns:
            Test results dictionary
        """
        results = {
            "v8_components_available": bool(self.v8),
            "v9_components_available": bool(self.v9),
            "v8_db_exists": self.v8_db_path.exists(),
            "v9_db_exists": self.v9_db_path.exists(),
            "v8_states_dir_exists": (self.v8_data_path / "states").exists(),
            "v9_states_dir_exists": (self.v9_data_path / "states").exists(),
            "tests": {}
        }
        
        # Test database migration if databases exist
        if results["v8_db_exists"] and self.v8 and self.v9:
            # Create backup of v9 database
            self.create_backup(self.v9_db_path, "compatibility_test")
            
            # Test v8 to v9 migration
            v8_to_v9_result = self.migrate_database(direction="v8_to_v9")
            results["tests"]["v8_to_v9_db_migration"] = v8_to_v9_result
        
        if results["v9_db_exists"] and self.v8 and self.v9:
            # Create backup of v8 database
            self.create_backup(self.v8_db_path, "compatibility_test")
            
            # Test v9 to v8 migration
            v9_to_v8_result = self.migrate_database(direction="v9_to_v8")
            results["tests"]["v9_to_v8_db_migration"] = v9_to_v8_result
        
        # Test neural state migration if state directories exist
        if results["v8_states_dir_exists"] and self.v8 and self.v9:
            v8_to_v9_state_result = self.migrate_neural_states(direction="v8_to_v9")
            results["tests"]["v8_to_v9_state_migration"] = v8_to_v9_state_result
        
        if results["v9_states_dir_exists"] and self.v8 and self.v9:
            v9_to_v8_state_result = self.migrate_neural_states(direction="v9_to_v8")
            results["tests"]["v9_to_v8_state_migration"] = v9_to_v8_state_result
        
        # Overall result
        results["overall_compatibility"] = all(results["tests"].values()) if results["tests"] else False
        
        return results

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="v8-v9 Neural Network Bridge")
    parser.add_argument("--v8-path", type=str, default="data/v8", help="Path to v8 data directory")
    parser.add_argument("--v9-path", type=str, default="data/v9", help="Path to v9 data directory")
    parser.add_argument("--action", type=str, required=True, 
                      choices=["migrate-db", "migrate-states", "sync", "test"],
                      help="Action to perform")
    parser.add_argument("--direction", type=str, default="v8_to_v9",
                      choices=["v8_to_v9", "v9_to_v8"],
                      help="Migration direction")
    parser.add_argument("--primary", type=str, default="v9",
                      choices=["v8", "v9"],
                      help="Primary version for sync")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backups")
    
    args = parser.parse_args()
    
    # Create bridge
    bridge = V8V9Bridge(
        v8_data_path=args.v8_path,
        v9_data_path=args.v9_path,
        create_backups=not args.no_backup
    )
    
    # Perform action
    if args.action == "migrate-db":
        success = bridge.migrate_database(direction=args.direction)
        print(f"Database migration {'succeeded' if success else 'failed'}")
        sys.exit(0 if success else 1)
    
    elif args.action == "migrate-states":
        success = bridge.migrate_neural_states(direction=args.direction)
        print(f"Neural state migration {'succeeded' if success else 'failed'}")
        sys.exit(0 if success else 1)
    
    elif args.action == "sync":
        success = bridge.sync_databases(primary=args.primary)
        print(f"Database synchronization {'succeeded' if success else 'failed'}")
        sys.exit(0 if success else 1)
    
    elif args.action == "test":
        results = bridge.run_compatibility_test()
        print("Compatibility Test Results:")
        print(json.dumps(results, indent=2))
        sys.exit(0 if results["overall_compatibility"] else 1) 