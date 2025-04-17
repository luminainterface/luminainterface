"""
Migration Utilities for Version 2
This module provides utility functions to assist in migrating from v1 to v2.
"""

import os
import shutil
from pathlib import Path
import logging
from typing import List, Dict, Optional
import json
from .bridge import V1ToV2Bridge

logger = logging.getLogger(__name__)

class MigrationManager:
    def __init__(self, v1_root: str, v2_root: str):
        """
        Initialize the migration manager.
        
        Args:
            v1_root: Root directory of v1 codebase
            v2_root: Root directory of v2 codebase
        """
        self.v1_root = Path(v1_root)
        self.v2_root = Path(v2_root)
        self.migration_log: Dict[str, List[str]] = {
            'successful': [],
            'failed': [],
            'skipped': []
        }
        
        logger.info(f"Initialized MigrationManager: v1_root={v1_root}, v2_root={v2_root}")
    
    def scan_v1_modules(self) -> List[Path]:
        """
        Scan for v1 modules that need to be migrated.
        
        Returns:
            List of paths to v1 modules
        """
        modules = []
        for root, _, files in os.walk(self.v1_root):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    modules.append(Path(root) / file)
        
        logger.info(f"Found {len(modules)} v1 modules to scan")
        return modules
    
    def create_v2_structure(self) -> None:
        """
        Create the v2 directory structure.
        """
        try:
            # Create main directories
            for dir_name in ['core', 'nodes', 'utils', 'infection', 'autolearning', 'fractal', 'ui']:
                (self.v2_root / dir_name).mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py files
            for dir_path in self.v2_root.rglob('*'):
                if dir_path.is_dir():
                    (dir_path / '__init__.py').touch(exist_ok=True)
            
            logger.info("Created v2 directory structure")
        except Exception as e:
            logger.error(f"Failed to create v2 structure: {str(e)}")
            raise
    
    def migrate_module(self, v1_path: Path) -> bool:
        """
        Migrate a single v1 module to v2.
        
        Args:
            v1_path: Path to v1 module
            
        Returns:
            True if migration successful, False otherwise
        """
        try:
            # Create bridge
            bridge = V1ToV2Bridge(str(v1_path))
            
            # Check compatibility
            if not bridge.check_compatibility():
                logger.warning(f"Skipping incompatible module: {v1_path}")
                self.migration_log['skipped'].append(str(v1_path))
                return False
            
            # Determine v2 path
            rel_path = v1_path.relative_to(self.v1_root)
            v2_path = self.v2_root / rel_path
            
            # Create v2 directory if needed
            v2_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy and update file
            shutil.copy2(v1_path, v2_path)
            
            # Update imports and code
            self._update_module_code(v2_path)
            
            logger.info(f"Successfully migrated module: {v1_path}")
            self.migration_log['successful'].append(str(v1_path))
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate module {v1_path}: {str(e)}")
            self.migration_log['failed'].append(str(v1_path))
            return False
    
    def _update_module_code(self, v2_path: Path) -> None:
        """
        Update module code for v2 compatibility.
        
        Args:
            v2_path: Path to v2 module
        """
        with open(v2_path, 'r') as f:
            code = f.read()
        
        # Update imports
        code = code.replace('from v1.', 'from v2.')
        code = code.replace('import v1.', 'import v2.')
        
        # Update class references
        code = code.replace('NeuralNetwork', 'NeuralNetworkV2')
        
        with open(v2_path, 'w') as f:
            f.write(code)
    
    def migrate_config_files(self) -> None:
        """
        Migrate configuration files to v2 format.
        """
        try:
            for config_file in self.v1_root.rglob('*.json'):
                if 'config' in config_file.name.lower():
                    # Create bridge
                    bridge = V1ToV2Bridge(str(config_file))
                    
                    # Load v1 config
                    with open(config_file, 'r') as f:
                        v1_config = json.load(f)
                    
                    # Migrate config
                    v2_config = bridge.migrate_config(v1_config)
                    
                    # Save v2 config
                    v2_path = self.v2_root / config_file.relative_to(self.v1_root)
                    v2_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(v2_path, 'w') as f:
                        json.dump(v2_config, f, indent=4)
                    
                    logger.info(f"Migrated config file: {config_file}")
        except Exception as e:
            logger.error(f"Failed to migrate config files: {str(e)}")
    
    def get_migration_report(self) -> Dict:
        """
        Get a report of the migration process.
        
        Returns:
            Dictionary containing migration statistics
        """
        return {
            'total_modules': len(self.migration_log['successful']) + 
                            len(self.migration_log['failed']) + 
                            len(self.migration_log['skipped']),
            'successful': len(self.migration_log['successful']),
            'failed': len(self.migration_log['failed']),
            'skipped': len(self.migration_log['skipped']),
            'details': self.migration_log
        }

# Export functionality for node integration
functionality = {
    'classes': {
        'MigrationManager': MigrationManager
    },
    'description': 'Migration utilities for v1 to v2 transition'
} 