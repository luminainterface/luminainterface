#!/usr/bin/env python3
import sys
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Decide which Qt binding to use
try:
    from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton
    from PySide6.QtCore import Qt
    logger.info("Successfully imported PySide6 modules")
    QT_BINDING = "PySide6"
except ImportError:
    logger.warning("PySide6 not found, falling back to PyQt5")
    try:
        from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton
        from PyQt5.QtCore import Qt
        logger.info("Successfully imported PyQt5 modules")
        QT_BINDING = "PyQt5"
    except ImportError:
        logger.error("Neither PySide6 nor PyQt5 are installed. Please install one of these packages.")
        sys.exit(1)


class TestMainWindow(QMainWindow):
    """Simple test main window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lumina GUI - Test Window")
        self.setMinimumSize(800, 600)
        
        # Create central widget
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        
        # Add title
        title = QLabel("Lumina GUI Test Window")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #333333;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Add description
        description = QLabel("This is a test window to verify that the Qt setup is working correctly.")
        description.setStyleSheet("font-size: 16px; color: #666666;")
        description.setAlignment(Qt.AlignCenter)
        layout.addWidget(description)
        
        # Add button
        button = QPushButton("Click Me")
        button.setStyleSheet("""
            QPushButton {
                background-color: #4B6EAF;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 16px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #5680C0;
            }
            QPushButton:pressed {
                background-color: #3A5A8C;
            }
        """)
        button.clicked.connect(self.on_button_clicked)
        layout.addWidget(button, 0, Qt.AlignCenter)
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
        logger.info("Test window initialized")
    
    def on_button_clicked(self):
        """Handle button click event"""
        logger.info("Button clicked")
        
        # Add a label to the window
        layout = self.centralWidget().layout()
        result_label = QLabel("Button was clicked!")
        result_label.setStyleSheet("font-size: 18px; color: #008800; margin-top: 20px;")
        result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(result_label)


def main():
    """Main application entry point"""
    logger.info("Starting test application")
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Lumina GUI Test")
    app.setStyle("Fusion")
    
    # Create and show main window
    main_window = TestMainWindow()
    main_window.show()
    
    logger.info("Test window shown")
    
    # Run application
    return app.exec()


if __name__ == "__main__":
    sys.exit(main()) 