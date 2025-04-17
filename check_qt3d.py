#!/usr/bin/env python3
"""Check if Qt3D modules are available"""

import sys
import platform

print(f"Python version: {sys.version}")
print(f"Platform: {platform.system()} {platform.release()}")

try:
    from PySide6 import QtCore
    print("PySide6 is available")
    
    # Try importing Qt3D modules
    try:
        from PySide6 import Qt3DCore
        print("Qt3DCore is available")
    except ImportError as e:
        print(f"Qt3DCore is NOT available: {e}")
    
    try:
        from PySide6 import Qt3DRender
        print("Qt3DRender is available")
    except ImportError as e:
        print(f"Qt3DRender is NOT available: {e}")
    
    try:
        from PySide6 import Qt3DExtras
        print("Qt3DExtras is available")
    except ImportError as e:
        print(f"Qt3DExtras is NOT available: {e}")
    
    try:
        from PySide6 import Qt3DInput
        print("Qt3DInput is available")
    except ImportError as e:
        print(f"Qt3DInput is NOT available: {e}")
    
    # Check for OpenGL support
    try:
        from PySide6.QtGui import QOpenGLContext
        context = QOpenGLContext()
        print(f"OpenGL context creation successful: {context.create()}")
        if context.isValid():
            print(f"OpenGL version: {context.format().majorVersion()}.{context.format().minorVersion()}")
        else:
            print("OpenGL context is not valid")
    except Exception as e:
        print(f"OpenGL error: {e}")
        
except ImportError as e:
    print(f"PySide6 is NOT available: {e}")

# Check if running in a virtual environment
try:
    import os
    venv = os.environ.get('VIRTUAL_ENV')
    if venv:
        print(f"Running in virtual environment: {venv}")
    else:
        print("Not running in a virtual environment")
except Exception as e:
    print(f"Error checking virtual environment: {e}")

print("\nSystem PATH:")
for path in sys.path:
    print(f"  {path}")

print("\nDone!") 