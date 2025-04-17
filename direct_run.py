#!/usr/bin/env python3
"""
Direct Run Script for V5 Visualization

This script directly creates a PySide6 application and window without relying
on imports from other modules, to isolate and resolve initialization issues.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("direct_run")

def main():
    """Main entry point"""
    logger.info("Starting direct run script")
    
    # Ensure QApplication is created before any QWidget
    try:
        # Import PySide6
        from PySide6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, 
            QLabel, QPushButton, QTabWidget
        )
        from PySide6.QtCore import Qt, QTimer
        
        logger.info("Successfully imported PySide6")
        
        # Check if QApplication already exists
        app = QApplication.instance()
        if app is None:
            logger.info("Creating new QApplication")
            app = QApplication(sys.argv)
        else:
            logger.info("Using existing QApplication")
        
        # Create main window
        logger.info("Creating main window")
        window = QMainWindow()
        window.setWindowTitle("V5 Fractal Echo Visualization (Direct Run)")
        window.resize(1000, 700)
        
        # Create central widget
        central = QWidget()
        window.setCentralWidget(central)
        
        # Create layout
        layout = QVBoxLayout(central)
        
        # Add header
        header = QLabel("V5 Fractal Echo Visualization System")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #4B6EAF;")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Add tabs
        tabs = QTabWidget()
        
        # Add placeholder tabs
        pattern_tab = QWidget()
        pattern_layout = QVBoxLayout(pattern_tab)
        pattern_layout.addWidget(QLabel("Fractal Pattern Visualization"))
        pattern_layout.addWidget(QLabel("This tab would show fractal pattern visualizations"))
        tabs.addTab(pattern_tab, "Fractal Patterns")
        
        memory_tab = QWidget()
        memory_layout = QVBoxLayout(memory_tab)
        memory_layout.addWidget(QLabel("Memory Synthesis"))
        memory_layout.addWidget(QLabel("This tab would show memory synthesis visualizations"))
        tabs.addTab(memory_tab, "Memory Synthesis")
        
        conversation_tab = QWidget()
        conversation_layout = QVBoxLayout(conversation_tab)
        conversation_layout.addWidget(QLabel("Conversation"))
        conversation_layout.addWidget(QLabel("This tab would provide a conversation interface"))
        tabs.addTab(conversation_tab, "Conversation")
        
        layout.addWidget(tabs)
        
        # Status message
        status = QLabel("Running in direct mode")
        status.setStyleSheet("color: #666;")
        layout.addWidget(status)
        
        # Show window
        window.show()
        
        # Execute application
        logger.info("Starting application event loop")
        return app.exec()
        
    except ImportError as e:
        logger.error(f"Error importing PySide6: {e}")
        print(f"Error: PySide6 is required but not installed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 