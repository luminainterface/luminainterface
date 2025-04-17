#!/usr/bin/env python
"""
Debug script for V6 Language Module

This script tests importing each component individually to identify issues.
"""

import os
import sys
import logging
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DebugLanguage")

def test_import(module_path, class_name=None):
    """Test importing a module and optionally a class from it"""
    try:
        print(f"Trying to import {module_path}...", end=" ")
        module = __import__(module_path, fromlist=["*"])
        print("SUCCESS")
        
        if class_name:
            print(f"  Trying to access {class_name}...", end=" ")
            try:
                cls = getattr(module, class_name)
                print("SUCCESS")
                return cls
            except AttributeError:
                print("FAILED")
                print(f"  Available attributes in {module_path}:")
                for attr in dir(module):
                    if not attr.startswith("__"):
                        print(f"    - {attr}")
                return None
        return module
    except ImportError as e:
        print(f"FAILED: {e}")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return None

def test_directory(dir_path):
    """Test if a directory exists and create it if it doesn't"""
    path = Path(dir_path)
    print(f"Checking directory {dir_path}...", end=" ")
    if path.exists():
        print("EXISTS")
    else:
        try:
            path.mkdir(parents=True, exist_ok=True)
            print("CREATED")
        except Exception as e:
            print(f"ERROR: {e}")

def main():
    """Main debug function"""
    print("=" * 60)
    print("V6 LANGUAGE MODULE DEBUG SCRIPT")
    print("=" * 60)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check directories
    print("\n-- Checking data directories --")
    test_directory("data/memory/language_memory")
    test_directory("data/neural_linguistic")
    test_directory("data/v10")
    test_directory("data/central_language")
    
    # Test PySide6
    print("\n-- Testing PySide6 --")
    pyside6 = test_import("PySide6")
    
    # Test core modules
    print("\n-- Testing V6 socket manager --")
    socket_manager = test_import("src.v6.socket_manager", "V6SocketManager")
    
    print("\n-- Testing V6 symbolic state manager --")
    symbolics = test_import("src.v6.symbolic_state_manager")
    
    print("\n-- Testing V6 bridge manager --")
    bridge_manager = test_import("src.v6.version_bridge_manager", "VersionBridgeManager")
    
    # Test panel base
    print("\n-- Testing V6 panel base --")
    panel_base = test_import("src.v6.ui.panel_base", "V6PanelBase")
    
    # Test language module imports
    print("\n-- Testing language module adapters --")
    pyside6_adapter = test_import("src.language.pyside6_adapter")
    
    print("\n-- Testing language module classes --")
    language_memory = test_import("src.language.language_memory", "LanguageMemory")
    neural_processor = test_import("src.language.neural_linguistic_processor", "NeuralLinguisticProcessor")
    conscious_mirror = test_import("src.language.conscious_mirror_language", "ConsciousMirrorLanguage")
    central_node = test_import("src.language.central_language_node", "CentralLanguageNode")
    
    # Test language module panel
    print("\n-- Testing language module panel --")
    language_panel = test_import("src.v6.ui.panels.language_module_panel", "LanguageModulePanel")
    
    # Summarize results
    print("\n=" * 60)
    print("DEBUG SUMMARY")
    print("=" * 60)
    
    modules = {
        "PySide6": pyside6,
        "Socket Manager": socket_manager,
        "Symbolic State Manager": symbolics,
        "Bridge Manager": bridge_manager,
        "Panel Base": panel_base,
        "PySide6 Adapter": pyside6_adapter,
        "Language Memory": language_memory,
        "Neural Processor": neural_processor,
        "Conscious Mirror": conscious_mirror,
        "Central Node": central_node,
        "Language Panel": language_panel,
    }
    
    for name, module in modules.items():
        status = "SUCCESS" if module else "FAILED"
        print(f"{name}: {status}")
    
    print("\nDebug complete.")

if __name__ == "__main__":
    main() 