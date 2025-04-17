#!/usr/bin/env python3
"""
V5-V10 Integration Verification Script

This script verifies that all components of the PySide6 integration and Language Memory System
are properly set up for integration with the v5-v10 evolution roadmap.

It checks:
1. File existence
2. Import compatibility
3. Integration points
4. Configuration consistency
5. Version alignment
"""

import os
import sys
import importlib
import logging
from pathlib import Path
import json
from typing import Dict, List, Set, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("v5_v10_verification")

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))


class IntegrationVerifier:
    """Verifies v5-v10 integration readiness"""
    
    def __init__(self):
        """Initialize the verifier"""
        self.project_root = Path(__file__).resolve().parent.parent
        self.src_dir = self.project_root / "src"
        
        # Initialize verification results
        self.results = {
            "files_exist": {},
            "imports_ok": {},
            "integration_points": {},
            "config_consistency": {},
            "version_alignment": {}
        }
    
    def verify_all(self) -> Dict:
        """Run all verification checks"""
        logger.info("Starting v5-v10 integration verification")
        
        # Verify file existence
        self.verify_files_exist()
        
        # Verify imports
        self.verify_imports()
        
        # Verify integration points
        self.verify_integration_points()
        
        # Verify configuration consistency
        self.verify_config_consistency()
        
        # Verify version alignment
        self.verify_version_alignment()
        
        # Summarize results
        return self._summarize_results()
    
    def verify_files_exist(self):
        """Verify all required files exist"""
        logger.info("Verifying required files exist")
        
        # Define required files
        required_files = [
            # Core PySide6 implementation
            "language_memory_gui_pyside.py",
            
            # V5 integration components
            "v5_integration/visualization_bridge.py",
            "v5_integration/__init__.py",
            
            # Documentation
            "../PYSIDE6_README.md",
            "../V5readme.md",
            "../v10readme.md",
            "../MASTERreadme.md",
            "../languageReadme.md",
            "../pyside6_migration_plan.md",
            
            # Core system files
            "language_memory_synthesis_integration.py",
            "conversation_memory.py",
            "english_language_trainer.py",
            "memory_api.py",
            "memory_api_server.py",
            "memory_api_client.py",
            "enhance_llm_prompt.py",
            "launch_memory_system.py",
            
            # V5 visualization components
            "v5/language_memory_integration.py",
            "v5/frontend_socket_manager.py",
            "v5/config.py"
        ]
        
        # Check each file
        for file_path in required_files:
            full_path = self.src_dir / file_path
            exists = full_path.exists()
            self.results["files_exist"][file_path] = exists
            if exists:
                logger.info(f"✅ Found: {file_path}")
            else:
                logger.error(f"❌ Missing: {file_path}")
    
    def verify_imports(self):
        """Verify import compatibility"""
        logger.info("Verifying import compatibility")
        
        # Define import relationships to check
        import_checks = [
            {
                "file": "language_memory_gui_pyside.py",
                "imports": [
                    "PySide6.QtCore", 
                    "PySide6.QtGui", 
                    "PySide6.QtWidgets",
                    "language_memory_synthesis_integration.LanguageMemorySynthesisIntegration"
                ]
            },
            {
                "file": "v5_integration/visualization_bridge.py",
                "imports": [
                    "language_memory_synthesis_integration.LanguageMemorySynthesisIntegration",
                    "v5.frontend_socket_manager.FrontendSocketManager",
                    "v5.language_memory_integration.LanguageMemoryIntegrationPlugin"
                ]
            }
        ]
        
        # Check each import relationship
        for check in import_checks:
            file_path = self.src_dir / check["file"]
            if not file_path.exists():
                logger.warning(f"Cannot verify imports for non-existent file: {check['file']}")
                continue
            
            # Try to import the file as a module
            file_dir = file_path.parent
            file_name = file_path.stem
            
            # Add to python path temporarily
            sys.path.insert(0, str(file_dir))
            
            # Track import results
            imports_ok = {}
            
            for import_path in check["imports"]:
                try:
                    parts = import_path.split(".")
                    if len(parts) > 1:
                        module_path = ".".join(parts[:-1])
                        attribute = parts[-1]
                        module = importlib.import_module(module_path)
                        getattr(module, attribute)
                    else:
                        importlib.import_module(import_path)
                    imports_ok[import_path] = True
                    logger.info(f"✅ Import OK: {import_path} in {check['file']}")
                except (ImportError, AttributeError) as e:
                    imports_ok[import_path] = False
                    logger.error(f"❌ Import failed: {import_path} in {check['file']}: {str(e)}")
            
            # Restore python path
            sys.path.pop(0)
            
            self.results["imports_ok"][check["file"]] = imports_ok
    
    def verify_integration_points(self):
        """Verify integration points between components"""
        logger.info("Verifying integration points")
        
        # Define integration points to check
        integration_points = [
            {
                "name": "language_gui_pyside_v5_integration",
                "source_file": "language_memory_gui_pyside.py",
                "target_file": "v5/language_memory_integration.py",
                "integration_marker": "v5_integration"
            },
            {
                "name": "visualization_bridge_language_memory",
                "source_file": "v5_integration/visualization_bridge.py",
                "target_file": "language_memory_synthesis_integration.py",
                "integration_marker": "LanguageMemorySynthesisIntegration"
            },
            {
                "name": "v5_language_memory_plugin",
                "source_file": "v5/language_memory_integration.py",
                "target_file": "language_memory_synthesis_integration.py",
                "integration_marker": "language_memory_synthesis"
            }
        ]
        
        # Check each integration point
        for point in integration_points:
            source_path = self.src_dir / point["source_file"]
            target_path = self.src_dir / point["target_file"]
            
            if not source_path.exists() or not target_path.exists():
                logger.warning(f"Cannot verify integration for non-existent files: {point['source_file']} -> {point['target_file']}")
                self.results["integration_points"][point["name"]] = False
                continue
            
            # Check for integration marker in source file
            with open(source_path, 'r', encoding='utf-8') as f:
                source_content = f.read()
                
                if point["integration_marker"] in source_content:
                    logger.info(f"✅ Integration point found: {point['name']}")
                    self.results["integration_points"][point["name"]] = True
                else:
                    logger.error(f"❌ Integration point missing: {point['name']}")
                    self.results["integration_points"][point["name"]] = False
    
    def verify_config_consistency(self):
        """Verify configuration consistency across components"""
        logger.info("Verifying configuration consistency")
        
        # Define configuration files to check
        config_files = [
            "v5/config.py"
        ]
        
        # Define expected configuration values
        expected_config = {
            "v5/config.py": [
                "MEMORY_CONFIG",
                "LANGUAGE_CONFIG"
            ]
        }
        
        # Check each configuration file
        for config_file in config_files:
            file_path = self.src_dir / config_file
            
            if not file_path.exists():
                logger.warning(f"Cannot verify config for non-existent file: {config_file}")
                self.results["config_consistency"][config_file] = False
                continue
            
            # Check for expected configuration values
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Track found configuration values
                found_values = {}
                
                for expected_value in expected_config.get(config_file, []):
                    found = expected_value in content
                    found_values[expected_value] = found
                    
                    if found:
                        logger.info(f"✅ Config value found: {expected_value} in {config_file}")
                    else:
                        logger.error(f"❌ Config value missing: {expected_value} in {config_file}")
                
                self.results["config_consistency"][config_file] = found_values
    
    def verify_version_alignment(self):
        """Verify version alignment with v5-v10 roadmap"""
        logger.info("Verifying version alignment")
        
        # Define version references to check
        version_references = [
            {
                "file": "../v10readme.md",
                "marker": "PySide6 Integration Milestone",
                "version": "v5-v6"
            },
            {
                "file": "../v10readme.md",
                "marker": "PYSIDE6_README.md",
                "version": "v5-v10"
            },
            {
                "file": "../MASTERreadme.md",
                "marker": "PYSIDE6_README.md",
                "version": "PySide6 Integration"
            },
            {
                "file": "../V5readme.md",
                "marker": "PySide6 Integration Guide",
                "version": "v5"
            }
        ]
        
        # Check each version reference
        for ref in version_references:
            file_path = self.project_root / ref["file"].lstrip("../")
            
            if not file_path.exists():
                logger.warning(f"Cannot verify version alignment for non-existent file: {ref['file']}")
                self.results["version_alignment"][f"{ref['file']}-{ref['marker']}"] = False
                continue
            
            # Check for version reference
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                if ref["marker"] in content and ref["version"] in content:
                    logger.info(f"✅ Version alignment found: {ref['marker']} -> {ref['version']} in {ref['file']}")
                    self.results["version_alignment"][f"{ref['file']}-{ref['marker']}"] = True
                else:
                    logger.error(f"❌ Version alignment missing: {ref['marker']} -> {ref['version']} in {ref['file']}")
                    self.results["version_alignment"][f"{ref['file']}-{ref['marker']}"] = False
    
    def _summarize_results(self) -> Dict:
        """Summarize verification results"""
        summary = {
            "files_missing": [f for f, exists in self.results["files_exist"].items() if not exists],
            "imports_failing": [],
            "integration_points_missing": [name for name, exists in self.results["integration_points"].items() if not exists],
            "config_inconsistencies": [],
            "version_misalignments": [ref for ref, aligned in self.results["version_alignment"].items() if not aligned]
        }
        
        # Collect failing imports
        for file, imports in self.results["imports_ok"].items():
            for import_path, ok in imports.items():
                if not ok:
                    summary["imports_failing"].append(f"{file}: {import_path}")
        
        # Collect config inconsistencies
        for file, values in self.results["config_consistency"].items():
            if isinstance(values, dict):
                for value, found in values.items():
                    if not found:
                        summary["config_inconsistencies"].append(f"{file}: {value}")
            elif not values:
                summary["config_inconsistencies"].append(file)
        
        # Calculate overall status
        summary["ready_for_integration"] = (
            len(summary["files_missing"]) == 0 and
            len(summary["imports_failing"]) == 0 and
            len(summary["integration_points_missing"]) == 0 and
            len(summary["config_inconsistencies"]) == 0 and
            len(summary["version_misalignments"]) == 0
        )
        
        return summary


def main():
    """Main function for the verification script"""
    # Create verifier
    verifier = IntegrationVerifier()
    
    # Run verification
    results = verifier.verify_all()
    
    # Print summary
    print("\n" + "="*80)
    print("V5-V10 INTEGRATION VERIFICATION SUMMARY")
    print("="*80)
    
    if results["ready_for_integration"]:
        print("\n✅ SYSTEM READY FOR V5-V10 INTEGRATION\n")
    else:
        print("\n❌ SYSTEM REQUIRES FIXES BEFORE V5-V10 INTEGRATION\n")
    
    # Print detailed issues
    if results["files_missing"]:
        print("Missing Files:")
        for file in results["files_missing"]:
            print(f"  - {file}")
        print()
    
    if results["imports_failing"]:
        print("Import Issues:")
        for import_issue in results["imports_failing"]:
            print(f"  - {import_issue}")
        print()
    
    if results["integration_points_missing"]:
        print("Missing Integration Points:")
        for point in results["integration_points_missing"]:
            print(f"  - {point}")
        print()
    
    if results["config_inconsistencies"]:
        print("Configuration Inconsistencies:")
        for inconsistency in results["config_inconsistencies"]:
            print(f"  - {inconsistency}")
        print()
    
    if results["version_misalignments"]:
        print("Version Misalignments:")
        for misalignment in results["version_misalignments"]:
            print(f"  - {misalignment}")
        print()
    
    # Exit with appropriate code
    sys.exit(0 if results["ready_for_integration"] else 1)


if __name__ == "__main__":
    main() 