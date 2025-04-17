#!/usr/bin/env python3
"""
Minimal Qt test script to verify GUI functionality
"""

import sys
import logging
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestWindow(QMainWindow):
    def __init__(self):
        logger.debug("Initializing TestWindow")
        super().__init__()
        self.setWindowTitle("Qt Test Window")
        self.resize(400, 300)
        
        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Add some test widgets
        label = QLabel("Test Window")
        layout.addWidget(label)
        
        button = QPushButton("Test Button")
        button.clicked.connect(lambda: logger.debug("Button clicked!"))
        layout.addWidget(button)
        
        logger.debug("TestWindow setup complete")

def main():
    try:
        logger.debug("Starting test application")
        app = QApplication(sys.argv)
        
        window = TestWindow()
        window.show()
        
        logger.debug("Running application")
        return app.exec()
        
    except Exception as e:
        logger.error(f"Error in test application: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 