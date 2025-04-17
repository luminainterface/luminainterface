"""
Qt Framework Compatibility Layer for V5 System

This module provides a consistent interface for Qt functionality regardless of whether
PyQt5 or PySide6 is used as the underlying implementation. It helps ensure seamless
transition between frameworks.
"""

import os
import importlib
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Framework selection
class QtFramework(Enum):
    PYSIDE6 = "PySide6"
    PYQT5 = "PyQt5"

# Allow environment variable override
QT_FRAMEWORK = os.environ.get("V5_QT_FRAMEWORK", QtFramework.PYSIDE6.value)

class QtCompat:
    """
    Compatibility layer for Qt frameworks to ensure seamless transitions
    between PyQt5 and PySide6.
    """
    
    @classmethod
    def init(cls):
        """
        Initialize the compatibility layer.
        
        Detects available frameworks and sets up the appropriate bindings.
        """
        # Try to import the framework specified in the environment variable
        if QT_FRAMEWORK == QtFramework.PYSIDE6.value:
            try:
                import PySide6
                cls.framework = QtFramework.PYSIDE6
                logger.info("Using PySide6 framework")
                return
            except ImportError:
                logger.warning("PySide6 not available, falling back to PyQt5")
        
        # Fall back to PyQt5
        try:
            import PyQt5
            cls.framework = QtFramework.PYQT5
            logger.info("Using PyQt5 framework")
        except ImportError:
            logger.error("Neither PySide6 nor PyQt5 is available!")
            raise ImportError("No Qt framework available! Please install PySide6 or PyQt5.")

    @classmethod
    def is_pyside6(cls):
        """Check if PySide6 is being used."""
        return cls.framework == QtFramework.PYSIDE6
        
    @classmethod
    def is_pyqt5(cls):
        """Check if PyQt5 is being used."""
        return cls.framework == QtFramework.PYQT5
    
    @classmethod
    def get_framework_name(cls):
        """Get the name of the active framework."""
        return cls.framework.value
        
    @classmethod
    def import_module(cls, submodule):
        """
        Import a Qt submodule from the active framework.
        
        Args:
            submodule: The Qt submodule to import (e.g., 'QtWidgets')
            
        Returns:
            The imported module
        """
        full_module = f"{cls.framework.value}.{submodule}"
        return importlib.import_module(full_module)
    
    @classmethod
    def get_application(cls):
        """
        Get the QApplication instance or create one if it doesn't exist.
        
        Returns:
            QApplication instance
        """
        if cls.framework == QtFramework.PYSIDE6:
            from PySide6.QtWidgets import QApplication
        else:
            from PyQt5.QtWidgets import QApplication
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app
    
    @classmethod
    def get_signal_class(cls):
        """
        Get the appropriate Signal class for the active framework.
        
        Returns:
            Signal class
        """
        if cls.framework == QtFramework.PYSIDE6:
            from PySide6.QtCore import Signal
            return Signal
        else:
            from PyQt5.QtCore import pyqtSignal
            return pyqtSignal
    
    @classmethod
    def get_slot_decorator(cls):
        """
        Get the appropriate Slot decorator for the active framework.
        
        Returns:
            Slot decorator
        """
        if cls.framework == QtFramework.PYSIDE6:
            from PySide6.QtCore import Slot
            return Slot
        else:
            from PyQt5.QtCore import pyqtSlot
            return pyqtSlot
    
    @classmethod
    def map_qt_constants(cls):
        """
        Map Qt constants to the appropriate values for the active framework.
        
        Returns:
            Dictionary of mapped constants
        """
        # Import Qt constants from the active framework
        if cls.framework == QtFramework.PYSIDE6:
            from PySide6.QtCore import Qt
        else:
            from PyQt5.QtCore import Qt
        
        # Return common constants that might differ between frameworks
        return {
            "AlignCenter": Qt.AlignmentFlag.AlignCenter if cls.is_pyside6() else Qt.AlignCenter,
            "AlignLeft": Qt.AlignmentFlag.AlignLeft if cls.is_pyside6() else Qt.AlignLeft,
            "AlignRight": Qt.AlignmentFlag.AlignRight if cls.is_pyside6() else Qt.AlignRight,
            "AlignTop": Qt.AlignmentFlag.AlignTop if cls.is_pyside6() else Qt.AlignTop,
            "AlignBottom": Qt.AlignmentFlag.AlignBottom if cls.is_pyside6() else Qt.AlignBottom,
        }

# Initialize the compatibility layer
QtCompat.init()

# Export convenience functions
def get_widgets():
    """Get the QtWidgets module"""
    if QtCompat.is_pyside6():
        from PySide6 import QtWidgets
        return QtWidgets
    else:
        from PyQt5 import QtWidgets
        return QtWidgets

def get_core():
    """Get the QtCore module"""
    if QtCompat.is_pyside6():
        from PySide6 import QtCore
        return QtCore
    else:
        from PyQt5 import QtCore
        return QtCore

def get_gui():
    """Get the QtGui module"""
    if QtCompat.is_pyside6():
        from PySide6 import QtGui
        return QtGui
    else:
        from PyQt5 import QtGui
        return QtGui

# Signal and Slot convenience functions
def Signal(*args, **kwargs):
    """Create a framework-appropriate Signal"""
    signal_class = QtCompat.get_signal_class()
    return signal_class(*args, **kwargs)

def Slot(*args, **kwargs):
    """Get a framework-appropriate Slot decorator"""
    slot_decorator = QtCompat.get_slot_decorator()
    return slot_decorator(*args, **kwargs)

# Constants
Qt = get_core().Qt
QtWidgets = get_widgets()
QtCore = get_core()
QtGui = get_gui() 