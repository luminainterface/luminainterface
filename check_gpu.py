#!/usr/bin/env python3
"""Check GPU and OpenGL information"""

import sys
import platform
import subprocess
import os

print(f"Python version: {sys.version}")
print(f"Platform: {platform.system()} {platform.release()}")

# Try to get GPU information on Windows
def get_gpu_info_windows():
    try:
        result = subprocess.run(
            ['wmic', 'path', 'win32_VideoController', 'get', 'name,AdapterDACType,AdapterRAM,DriverVersion'],
            capture_output=True, 
            text=True, 
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error getting GPU info: {result.stderr}"
    except Exception as e:
        return f"Error running GPU check: {e}"

# Check if Qt and Qt3D modules are available
try:
    from PySide6 import QtCore
    print(f"PySide6 version: {QtCore.__version__}")
    
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
    
    # Check for OpenGL support with PySide6
    print("\nPySide6 OpenGL Information:")
    try:
        from PySide6.QtGui import QOpenGLContext, QSurfaceFormat
        
        # Get default format
        default_format = QSurfaceFormat.defaultFormat()
        print(f"Default OpenGL Format:")
        print(f"  Profile: {default_format.profile()}")
        print(f"  Version: {default_format.majorVersion()}.{default_format.minorVersion()}")
        print(f"  Samples: {default_format.samples()}")
        
        # Try to create a context
        context = QOpenGLContext()
        if context.create():
            print("OpenGL context creation: SUCCESS")
            surface_format = context.format()
            print(f"Created context version: {surface_format.majorVersion()}.{surface_format.minorVersion()}")
            print(f"Created context profile: {surface_format.profile()}")
            
            if context.isValid():
                functions = context.functions()
                if hasattr(functions, 'glGetString'):
                    try:
                        from PySide6.QtGui import QOffscreenSurface
                        # Create an offscreen surface for the context to make it current
                        surface = QOffscreenSurface()
                        surface.create()
                        
                        if context.makeCurrent(surface):
                            vendor = functions.glGetString(functions.GL_VENDOR)
                            renderer = functions.glGetString(functions.GL_RENDERER)
                            version = functions.glGetString(functions.GL_VERSION)
                            
                            print(f"OpenGL Vendor: {vendor}")
                            print(f"OpenGL Renderer: {renderer}")
                            print(f"OpenGL Version: {version}")
                            
                            context.doneCurrent()
                        else:
                            print("Could not make OpenGL context current")
                    except Exception as e:
                        print(f"Error getting OpenGL info: {e}")
            else:
                print("OpenGL context is not valid")
        else:
            print("OpenGL context creation: FAILED")
            
    except Exception as e:
        print(f"OpenGL error: {e}")
        
except ImportError as e:
    print(f"PySide6 is NOT available: {e}")

# Get GPU information
if platform.system() == 'Windows':
    print("\nGPU Information:")
    gpu_info = get_gpu_info_windows()
    print(gpu_info)

# Try to use PyOpenGL to get more info
try:
    print("\nTrying PyOpenGL:")
    try:
        import OpenGL
        print(f"PyOpenGL version: {OpenGL.__version__}")
        
        from OpenGL.GL import *
        
        # Need a context to use PyOpenGL
        print("PyOpenGL imported successfully, but needs a valid OpenGL context")
        
    except ImportError as e:
        print(f"PyOpenGL is not installed: {e}")
except Exception as e:
    print(f"Error with PyOpenGL: {e}")

print("\nDone!") 