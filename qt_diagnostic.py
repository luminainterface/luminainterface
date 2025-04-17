import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QLibraryInfo
import os

def check_qt_setup():
    print("Python version:", sys.version)
    print("\nQt Library Paths:")
    paths = QLibraryInfo.location(QLibraryInfo.LibraryPath.PluginsPath)
    print(f"Plugins path: {paths}")
    
    print("\nEnvironment variables:")
    qt_vars = [var for var in os.environ if 'QT' in var or 'PYTHON' in var]
    for var in qt_vars:
        print(f"{var}: {os.environ[var]}")
    
    print("\nCreating QApplication...")
    app = QApplication(sys.argv)
    print("QApplication created successfully")
    
    print("\nAvailable styles:", QApplication.styles())
    
    return app.exec()

if __name__ == "__main__":
    print("Starting Qt diagnostics...")
    sys.exit(check_qt_setup()) 