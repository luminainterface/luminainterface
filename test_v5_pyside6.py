#!/usr/bin/env python3
"""
Test V5 PySide6 Integration

This script tests the V5 Fractal Echo Visualization system with PySide6 integration
and helps diagnose issues with the PySide6 migration.
"""

import os
import sys
import traceback
from pathlib import Path

# Ensure the V5_QT_FRAMEWORK is set to PySide6
os.environ["V5_QT_FRAMEWORK"] = "PySide6"

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

def check_imports():
    """Check if all required packages are available"""
    print("\n=== Checking imports ===")
    
    # Essential packages
    try:
        from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
        from PySide6.QtCore import Qt
        print("✓ PySide6 core modules imported successfully")
    except ImportError as e:
        print(f"✗ Error importing PySide6: {e}")
        print("  • Try installing it with: pip install PySide6")
        return False
    
    # Check for compatibility layer
    try:
        from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui, QtCompat
        print(f"✓ Qt compatibility layer imported successfully")
        print(f"  • Active framework: {QtCompat.get_framework_name()}")
    except ImportError as e:
        print(f"✗ Error importing qt_compat: {e}")
        return False
    
    # Check for visualization bridge
    try:
        from src.v5_integration.visualization_bridge import get_visualization_bridge
        print("✓ Visualization bridge imported successfully")
    except ImportError as e:
        print(f"✗ Error importing visualization bridge: {e}")
        print("  • This is needed for V5 visualization integration")
    
    return True

def check_panels():
    """Check if panel components can be imported"""
    print("\n=== Checking panel components ===")
    
    from src.v5.ui.qt_compat import QtCompat
    
    panel_paths = [
        "src.v5.ui.panels.fractal_pattern_panel",
        "src.v5.ui.panels.node_consciousness_panel",
        "src.v5.ui.panels.memory_synthesis_panel",
        "src.v5.ui.panels.network_visualization_panel"
    ]
    
    panel_classes = [
        "FractalPatternPanel",
        "NodeConsciousnessPanel",
        "MemorySynthesisPanel",
        "NetworkVisualizationPanel"
    ]
    
    success = True
    for path, cls in zip(panel_paths, panel_classes):
        try:
            module = __import__(path, fromlist=[cls])
            panel_class = getattr(module, cls)
            print(f"✓ Successfully imported {cls}")
        except ImportError as e:
            print(f"✗ Error importing {cls}: {e}")
            success = False
        except AttributeError as e:
            print(f"✗ Error finding {cls} in {path}: {e}")
            success = False
    
    # Check if the panel factory works
    try:
        from src.v5.ui.panels import get_panel
        print("✓ Panel factory imported successfully")
    except (ImportError, AttributeError) as e:
        print(f"✗ Error importing panel factory: {e}")
        success = False
    
    return success

def check_visualization_bridge():
    """Check if the visualization bridge works"""
    print("\n=== Checking visualization bridge ===")
    
    try:
        from src.v5_integration.visualization_bridge import get_visualization_bridge
        bridge = get_visualization_bridge()
        
        print(f"• Visualization available: {bridge.is_visualization_available()}")
        
        if bridge.is_visualization_available():
            components = bridge.get_available_visualization_components()
            print(f"• Available components: {components}")
            
            if components:
                print("✓ Visualization bridge is working")
                return True
            else:
                print("✗ No visualization components found")
                return False
        else:
            print("✗ Visualization is not available")
            print("  • Check that all V5 components are properly installed")
            return False
    except Exception as e:
        print(f"✗ Error initializing visualization bridge: {e}")
        traceback.print_exc()
        return False

def check_memory_integration():
    """Check if language memory integration works"""
    print("\n=== Checking language memory integration ===")
    
    try:
        from src.language_memory_synthesis_integration import LanguageMemorySynthesisIntegration
        print("✓ LanguageMemorySynthesisIntegration imported successfully")
        
        memory_system = LanguageMemorySynthesisIntegration()
        print("✓ Memory system initialized successfully")
        
        from src.v5.language_memory_integration import LanguageMemoryIntegrationPlugin
        print("✓ LanguageMemoryIntegrationPlugin imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Error importing memory integration: {e}")
        return False
    except Exception as e:
        print(f"✗ Error initializing memory system: {e}")
        return False

def check_gui_app():
    """Check if the GUI application runs"""
    print("\n=== Checking GUI application ===")
    
    try:
        from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui
        
        # Create a basic application
        app = QtWidgets.QApplication([])
        
        # Create a simple window
        window = QtWidgets.QMainWindow()
        window.setWindowTitle("PySide6 Test")
        window.resize(800, 600)
        
        # Create a central widget
        central_widget = QtWidgets.QWidget()
        window.setCentralWidget(central_widget)
        
        # Create a layout
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Add a label
        label = QtWidgets.QLabel("PySide6 is working correctly!")
        label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(label)
        
        # Try to create a visualization component
        try:
            from src.v5_integration.visualization_bridge import get_visualization_bridge
            bridge = get_visualization_bridge()
            
            if bridge.is_visualization_available():
                components = bridge.get_available_visualization_components()
                
                if "fractal_pattern_panel" in components:
                    panel = bridge.create_visualization_panel("fractal_pattern_panel")
                    if panel:
                        layout.addWidget(panel)
                        label.setText("PySide6 is working with V5 visualization!")
        except Exception as e:
            print(f"Note: Could not create visualization component: {e}")
        
        # Show the window briefly to test rendering
        window.show()
        QtCore.QTimer.singleShot(500, app.quit)  # Quit after 500ms
        
        app.exec()
        print("✓ GUI application runs successfully")
        return True
    except Exception as e:
        print(f"✗ Error running GUI application: {e}")
        traceback.print_exc()
        return False

def check_lumina_run():
    """Diagnose issues with running the main Lumina GUI"""
    print("\n=== Checking lumina_gui_next_run.py ===")
    
    # Check MainController
    try:
        from src.ui.MainController import MainController
        print("✓ MainController imported successfully")
    except ImportError as e:
        print(f"✗ Error importing MainController: {e}")
        print("  • This is needed for the Lumina GUI to run")
        print("  • Check that the file exists at src/ui/MainController.py")
        return False
    
    # Check FrontendSocketManager
    try:
        from src.v5.frontend_socket_manager import FrontendSocketManager
        print("✓ FrontendSocketManager imported successfully")
    except ImportError as e:
        print(f"✗ Error importing FrontendSocketManager: {e}")
        
    # Check component import function
    try:
        from src.ui.MainController import import_component
        print("✓ import_component function is available")
    except ImportError as e:
        print(f"✗ Error importing import_component: {e}")
    
    print("\nTo run the Lumina GUI, use one of these commands:")
    print("  • python lumina_gui_next_run.py")
    print("  • python -m src.ui.MainController")
    
    return True

def main():
    """Run all tests"""
    print("V5 PySide6 Integration Test")
    print("===========================")
    
    if not check_imports():
        print("\n❌ Basic imports failed. Please fix these issues first.")
        return
    
    check_panels()
    check_visualization_bridge()
    check_memory_integration()
    check_gui_app()
    check_lumina_run()
    
    print("\n=== Summary ===")
    print("If all tests passed, you should be able to run:")
    print("  • python lumina_gui_next_run.py")
    print("  • python test_v5_pyside6.py")
    print("\nIf there were errors, try:")
    print("  1. Installing missing packages: pip install PySide6")
    print("  2. Using the migration script: python migrate_to_qt_compat.py src/v5/ui/panels")
    print("  3. Setting environment variable: export V5_QT_FRAMEWORK=PySide6 (Linux/Mac) or set V5_QT_FRAMEWORK=PySide6 (Windows)")
    print("  4. Falling back to PyQt5: export V5_QT_FRAMEWORK=PyQt5 (Linux/Mac) or set V5_QT_FRAMEWORK=PyQt5 (Windows)")

if __name__ == "__main__":
    main() 