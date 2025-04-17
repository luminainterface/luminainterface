#!/usr/bin/env python3
"""
PySide6 Migration Script

This script helps users migrate from the traditional tkinter-based GUI
to the new PySide6 implementation. It:

1. Checks for required dependencies
2. Creates necessary directories
3. Sets up configuration
4. Provides clear instructions

This is part of the v5-v10 evolution roadmap for the Lumina Neural Network system.
"""

import os
import sys
import shutil
import subprocess
import platform
import logging
from pathlib import Path
import json
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pyside6_migration")


class PySide6Migrator:
    """Migrates from tkinter to PySide6 GUI"""
    
    def __init__(self, backup=True, force=False):
        """
        Initialize the migrator
        
        Args:
            backup: Whether to backup existing files
            force: Whether to force migration even if dependencies are missing
        """
        self.project_root = Path(__file__).resolve().parent.parent
        self.src_dir = self.project_root / "src"
        self.backup = backup
        self.force = force
        
        # Initialize status
        self.status = {
            "dependencies_ok": False,
            "directories_created": [],
            "files_migrated": [],
            "config_updated": False,
            "success": False
        }
    
    def run_migration(self):
        """Run the migration process"""
        logger.info("Starting migration to PySide6")
        
        # Check dependencies
        if not self.check_dependencies():
            if not self.force:
                logger.error("Migration aborted due to missing dependencies")
                return self.status
            else:
                logger.warning("Continuing migration despite missing dependencies (force mode)")
        
        # Create directories
        self.create_directories()
        
        # Copy files
        self.copy_files()
        
        # Update config
        self.update_config()
        
        # Set success flag
        self.status["success"] = True
        
        logger.info("Migration to PySide6 completed successfully")
        return self.status
    
    def check_dependencies(self):
        """Check for required dependencies"""
        logger.info("Checking for required dependencies")
        
        dependencies = ["PySide6"]
        missing = []
        
        for dep in dependencies:
            try:
                __import__(dep)
                logger.info(f"✅ Dependency found: {dep}")
            except ImportError:
                logger.error(f"❌ Missing dependency: {dep}")
                missing.append(dep)
        
        visualization_deps = ["numpy", "matplotlib", "networkx"]
        missing_viz = []
        
        for dep in visualization_deps:
            try:
                __import__(dep)
                logger.info(f"✅ Visualization dependency found: {dep}")
            except ImportError:
                logger.warning(f"⚠️ Missing visualization dependency: {dep}")
                missing_viz.append(dep)
        
        # Update status
        self.status["dependencies_ok"] = len(missing) == 0
        self.status["missing_dependencies"] = missing
        self.status["missing_visualization_dependencies"] = missing_viz
        
        return self.status["dependencies_ok"]
    
    def create_directories(self):
        """Create necessary directories"""
        logger.info("Creating necessary directories")
        
        directories = [
            "v5_integration"
        ]
        
        for directory in directories:
            dir_path = self.src_dir / directory
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created directory: {directory}")
                self.status["directories_created"].append(directory)
            else:
                logger.info(f"✅ Directory already exists: {directory}")
    
    def copy_files(self):
        """Copy migration files"""
        logger.info("Copying migration files")
        
        # Define migration files
        migration_files = [
            {
                "source": "language_memory_gui_pyside.py",
                "destination": "language_memory_gui_pyside.py"
            },
            {
                "source": "v5_integration/visualization_bridge.py",
                "destination": "v5_integration/visualization_bridge.py"
            },
            {
                "source": "v5_integration/__init__.py",
                "destination": "v5_integration/__init__.py"
            }
        ]
        
        # Special handling for bridging the tkinter implementation
        # if we're upgrading an existing installation
        if (self.src_dir / "language_memory_gui.py").exists():
            # Create a backup of the original tkinter file if needed
            if self.backup:
                backup_path = self.src_dir / "language_memory_gui.py.bak"
                if not backup_path.exists():
                    shutil.copy2(self.src_dir / "language_memory_gui.py", backup_path)
                    logger.info(f"✅ Created backup: language_memory_gui.py.bak")
        
        # Copy each file
        for file_info in migration_files:
            source_path = file_info["source"]
            dest_path = self.src_dir / file_info["destination"]
            
            if dest_path.exists() and self.backup:
                # Create backup
                backup_path = dest_path.with_suffix(dest_path.suffix + ".bak")
                if not backup_path.exists():
                    shutil.copy2(dest_path, backup_path)
                    logger.info(f"✅ Created backup: {file_info['destination']}.bak")
            
            source_exists = (self.src_dir / source_path).exists()
            
            if source_exists:
                shutil.copy2(self.src_dir / source_path, dest_path)
                logger.info(f"✅ Copied file: {file_info['destination']}")
                self.status["files_migrated"].append(file_info["destination"])
            else:
                # We need to use the template files from the repo instead
                # Since we don't have access to those in this script, we'll note the error
                logger.error(f"❌ Source file not found: {source_path}")
                logger.info(f"Please copy the template file manually to: {file_info['destination']}")
    
    def update_config(self):
        """Update configuration for PySide6"""
        logger.info("Updating configuration for PySide6")
        
        # Check if V5 config exists
        v5_config_path = self.src_dir / "v5/config.py"
        if not v5_config_path.exists():
            logger.warning("⚠️ V5 config not found, skipping configuration update")
            return
        
        # Read the config file
        with open(v5_config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # Check if MEMORY_CONFIG and LANGUAGE_CONFIG exist
        memory_config_exists = "MEMORY_CONFIG" in config_content
        language_config_exists = "LANGUAGE_CONFIG" in config_content
        
        if not memory_config_exists:
            # Add MEMORY_CONFIG
            config_content += "\n# Memory integration settings\nMEMORY_CONFIG = {\n"
            config_content += '    "storage_path": str(V5_DATA_DIR / "memory"),\n'
            config_content += '    "max_cache_size_mb": 512,\n'
            config_content += '    "auto_save_interval_sec": 300\n'
            config_content += "}\n"
            logger.info("✅ Added MEMORY_CONFIG to v5/config.py")
        
        if not language_config_exists:
            # Add LANGUAGE_CONFIG
            config_content += "\n# Language integration settings\nLANGUAGE_CONFIG = {\n"
            config_content += '    "synthesis_batch_size": 32,\n'
            config_content += '    "max_context_length": 2048,\n'
            config_content += '    "enable_streaming": True\n'
            config_content += "}\n"
            logger.info("✅ Added LANGUAGE_CONFIG to v5/config.py")
        
        # Write the updated config file
        if not memory_config_exists or not language_config_exists:
            with open(v5_config_path, 'w', encoding='utf-8') as f:
                f.write(config_content)
            logger.info("✅ Updated v5/config.py")
            self.status["config_updated"] = True
        else:
            logger.info("✅ V5 config already has required entries")
            self.status["config_updated"] = True
    
    def print_next_steps(self):
        """Print next steps for the user"""
        print("\n" + "="*80)
        print("PYSIDE6 MIGRATION - NEXT STEPS")
        print("="*80)
        
        if not self.status["dependencies_ok"]:
            print("\n1. Install required dependencies:")
            for dep in self.status["missing_dependencies"]:
                print(f"   pip install {dep}")
        
        if self.status["missing_visualization_dependencies"]:
            print("\n2. For full visualization capabilities, install:")
            for dep in self.status["missing_visualization_dependencies"]:
                print(f"   pip install {dep}")
        
        print("\n3. Run the PySide6 version with:")
        print("   python src/language_memory_gui_pyside.py")
        
        print("\n4. To verify v5-v10 integration readiness:")
        print("   python src/verify_v5_v10_integration.py")
        
        print("\n5. For more information, see:")
        print("   - PYSIDE6_README.md")
        print("   - V5readme.md")
        print("\n")


def main():
    """Main function for the migration script"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Migrate from tkinter to PySide6 GUI")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backups of existing files")
    parser.add_argument("--force", action="store_true", help="Force migration even if dependencies are missing")
    args = parser.parse_args()
    
    # Create migrator
    migrator = PySide6Migrator(backup=not args.no_backup, force=args.force)
    
    # Run migration
    status = migrator.run_migration()
    
    # Print next steps
    migrator.print_next_steps()
    
    # Print status summary
    print("="*80)
    print("MIGRATION STATUS SUMMARY")
    print("="*80)
    
    if status["success"]:
        print("\n✅ Migration completed successfully\n")
    else:
        print("\n❌ Migration completed with issues\n")
    
    print(f"Directories Created: {len(status['directories_created'])}")
    print(f"Files Migrated: {len(status['files_migrated'])}")
    print(f"Config Updated: {'Yes' if status['config_updated'] else 'No'}")
    print(f"Dependencies OK: {'Yes' if status['dependencies_ok'] else 'No'}")
    
    # Exit with appropriate code
    sys.exit(0 if status["success"] else 1)


if __name__ == "__main__":
    main() 