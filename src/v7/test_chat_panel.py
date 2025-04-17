#!/usr/bin/env python3
"""
Test script for the V7 Language Chat Panel with streaming responses
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine which Qt library to use
try:
    from PySide6.QtWidgets import QApplication, QMainWindow
    logger.info("Using PySide6 for UI")
    USING_PYSIDE6 = True
except ImportError:
    try:
        from PyQt5.QtWidgets import QApplication, QMainWindow
        logger.info("Using PyQt5 for UI")
        USING_PYSIDE6 = False
    except ImportError:
        logger.error("Neither PySide6 nor PyQt5 is available. UI cannot be initialized.")
        sys.exit(1)

# Import our panel
try:
    from src.v7.ui.panels.language_chat_panel import LanguageChatPanel
except ImportError as e:
    logger.error(f"Failed to import LanguageChatPanel: {e}")
    sys.exit(1)

class TestWindow(QMainWindow):
    """Simple test window to host the panel"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("V7 Language Chat Panel Test")
        self.setMinimumSize(800, 600)
        
        # Create language chat panel
        self.chat_panel = LanguageChatPanel()
        self.setCentralWidget(self.chat_panel)

def main():
    """Main entry point"""
    # Create application
    app = QApplication(sys.argv)
    
    # Create main window
    window = TestWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec() if USING_PYSIDE6 else app.exec_())

if __name__ == "__main__":
    main() 