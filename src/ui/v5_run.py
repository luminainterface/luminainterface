#!/usr/bin/env python
"""
V5 Visualization System Launcher

This script launches the V5 Fractal Echo Visualization system.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/v5_visualization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("v5_launcher")

# Try using PySide6, fallback to PyQt5 if needed
try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from v5_controller import V5Controller
    logger.info("Using PySide6 for V5 visualization")
    USING_PYSIDE6 = True
except ImportError:
    logger.warning("PySide6 not found, falling back to PyQt5")
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    from v5_controller import V5Controller
    logger.info("Using PyQt5 for V5 visualization")
    USING_PYSIDE6 = False

def main():
    """Main entry point for the V5 Visualization application"""
    # Ensure directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("assets/icons", exist_ok=True)
    
    # Set application attributes
    if USING_PYSIDE6:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    else:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Lumina Neural Network - V5 Visualization")
    app.setApplicationVersion("5.0")
    
    # Set style
    app.setStyle("Fusion")
    
    # Create and show the main controller
    controller = V5Controller()
    controller.show()
    
    logger.info("V5 Visualization System started")
    
    # Execute application
    return app.exec() if USING_PYSIDE6 else app.exec_()

if __name__ == "__main__":
    sys.exit(main()) 