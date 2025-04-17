#!/usr/bin/env python3
"""
Lumina GUI Next Launcher - A launcher script for the next generation Lumina GUI

This launcher script provides a simple way to start the upgraded Lumina GUI system.
"""

import os
import sys
import argparse
import traceback
from pathlib import Path
import logging

# Create required directories if they don't exist
os.makedirs("logs", exist_ok=True)
os.makedirs("assets/icons", exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/lumina_gui.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Decide which Qt binding to use
try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QCoreApplication, Qt
    logger.info("Successfully imported PySide6 modules")
    QT_BINDING = "PySide6"
except ImportError:
    logger.warning("PySide6 not found, falling back to PyQt5")
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QCoreApplication, Qt
        logger.info("Successfully imported PyQt5 modules")
        QT_BINDING = "PyQt5"
    except ImportError:
        logger.error("Neither PySide6 nor PyQt5 are installed. Please install one of these packages.")
        sys.exit(1)

# Add src directory to path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / "src"
sys.path.append(str(current_dir))

def create_placeholder_icons():
    """Create placeholder icons if they don't exist"""
    try:
        if QT_BINDING == "PySide6":
            from PySide6.QtGui import QPixmap, QPainter, QColor, QFont, QBrush, QPen
            from PySide6.QtCore import Qt
        else:
            from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QBrush, QPen
            from PyQt5.QtCore import Qt
        
        icons = {
            "user": "#5C6BC0",
            "star": "#FFB300",
            "path": "#26A69A",
            "bulb": "#FFA726",
            "lotus": "#AB47BC",
            "memory": "#66BB6A",
            "network": "#42A5F5",
            "brain": "#EC407A",
            "database": "#7E57C2",
            "settings": "#78909C"
        }
        
        for name, color in icons.items():
            icon_path = f"assets/icons/{name}.png"
            
            # Skip if icon already exists
            if os.path.exists(icon_path):
                continue
                
            # Create a simple colored square with text as placeholder
            pixmap = QPixmap(64, 64)
            pixmap.fill(Qt.transparent)
            
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Draw colored background
            painter.setBrush(QBrush(QColor(color)))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(4, 4, 56, 56, 10, 10)
            
            # Draw text
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.setFont(QFont("Arial", 12, QFont.Bold))
            painter.drawText(pixmap.rect(), Qt.AlignCenter, name[0].upper())
            
            painter.end()
            
            # Save the pixmap
            pixmap.save(icon_path)
            
        logger.info("Created placeholder icons")
    except Exception as e:
        logger.error(f"Error creating placeholder icons: {e}")


def main():
    """Main application entry point"""
    logger.info("Starting Lumina GUI Next V3 application")
    
    # Set application attributes
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Lumina Neural Network")
    app.setApplicationVersion("3.0")
    
    # Set style
    app.setStyle("Fusion")
    
    # Create placeholder icons AFTER QApplication is created
    create_placeholder_icons()
    
    # Try to import and use our new MainController
    try:
        logger.info("Attempting to import new MainController...")
        from src.ui.MainController import MainController
        logger.info("New MainController imported successfully")
        
        main_window = MainController()
        logger.info("New MainController instance created")
        main_window.show()
        logger.info("New MainController window shown")
        
        # Run application
        return app.exec()
    except Exception as e:
        logger.error(f"Error using new MainController: {e}")
        logger.warning("Falling back to original Lumina GUI implementation")
        
        # Try importing original LuminaGUINext implementation
        try:
            logger.info("Importing original LuminaGUINext...")
            from lumina_gui_next import LuminaGUINext, create_central_node
            
            central_node = create_central_node()
            main_window = LuminaGUINext(central_node)
            main_window.show()
            logger.info("Original LuminaGUINext window shown")
            
            # Run application
            return app.exec()
        except Exception as e:
            logger.error(f"Error starting original application: {e}")
            
            # Create a simple fallback window
            if QT_BINDING == "PySide6":
                from PySide6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget
            else:
                from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget
            
            class FallbackMainWindow(QMainWindow):
                def __init__(self):
                    super().__init__()
                    self.setWindowTitle("Lumina GUI - Fallback Mode")
                    self.setMinimumSize(800, 600)
                    
                    central_widget = QWidget()
                    layout = QVBoxLayout(central_widget)
                    
                    error_label = QLabel("Failed to load full application. Running in fallback mode.")
                    error_label.setStyleSheet("font-size: 18px; color: red;")
                    layout.addWidget(error_label)
                    
                    details_label = QLabel(f"Error: {str(e)}")
                    layout.addWidget(details_label)
                    
                    self.setCentralWidget(central_widget)
            
            fallback_window = FallbackMainWindow()
            fallback_window.show()
            logger.warning("Using minimal fallback window due to errors")
            
            # Run application
            return app.exec()


if __name__ == "__main__":
    sys.exit(main()) 