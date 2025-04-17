"""
Theme Manager for V5 PySide6 Client

Manages application themes and styles.
"""

import os
import logging
from pathlib import Path

# Try to import from PySide6
try:
    from PySide6.QtCore import QObject, Qt
    from PySide6.QtGui import QPalette, QColor
    from PySide6.QtWidgets import QApplication, QStyleFactory
except ImportError:
    from PyQt5.QtCore import QObject, Qt
    from PyQt5.QtGui import QPalette, QColor
    from PyQt5.QtWidgets import QApplication, QStyleFactory

logger = logging.getLogger(__name__)

class ThemeManager(QObject):
    """Theme management for the application"""
    
    def __init__(self):
        """Initialize the theme manager"""
        super().__init__()
        self.current_theme = "system"
        self.available_styles = QStyleFactory.keys()
        logger.info(f"Available styles: {', '.join(self.available_styles)}")
    
    def apply_theme(self, theme_name):
        """
        Apply a theme to the application
        
        Args:
            theme_name: Name of the theme to apply ('light', 'dark', 'system')
        """
        self.current_theme = theme_name
        
        app = QApplication.instance()
        if app is None:
            logger.warning("No QApplication instance found")
            return
        
        if theme_name == "light":
            self._apply_light_theme(app)
        elif theme_name == "dark":
            self._apply_dark_theme(app)
        else:  # system
            self._apply_system_theme(app)
            
        logger.info(f"Applied {theme_name} theme")
    
    def _apply_light_theme(self, app):
        """Apply light theme to application"""
        # Set Fusion style for consistent appearance
        if "Fusion" in self.available_styles:
            app.setStyle("Fusion")
        
        # Create light palette
        palette = QPalette()
        
        # Set light colors
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, QColor(10, 10, 10))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(233, 231, 245))
        palette.setColor(QPalette.Text, QColor(10, 10, 10))
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, QColor(10, 10, 10))
        palette.setColor(QPalette.Link, QColor(66, 134, 244))
        palette.setColor(QPalette.Highlight, QColor(66, 134, 244))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        
        # Set the palette
        app.setPalette(palette)
        
        # Set stylesheet for additional customization
        app.setStyleSheet("""
            QToolTip { 
                color: #0A0A0A; 
                background-color: #FFFEF0; 
                border: 1px solid #C0C0C0; 
            }
            
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            QTabWidget::pane {
                border: 1px solid #C0C0C0;
                border-radius: 3px;
            }
            
            QTabBar::tab {
                background: #E8E8E8;
                border: 1px solid #C0C0C0;
                border-bottom-color: #C0C0C0;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 5px 10px;
            }
            
            QTabBar::tab:selected {
                background: #FFFFFF;
                border-bottom-color: #FFFFFF;
            }
            
            QPushButton {
                background-color: #F0F0F0;
                border: 1px solid #C0C0C0;
                border-radius: 4px;
                padding: 5px 10px;
            }
            
            QPushButton:hover {
                background-color: #E0E0E0;
            }
            
            QPushButton:pressed {
                background-color: #D0D0D0;
            }
            
            QGroupBox {
                border: 1px solid #C0C0C0;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                color: #505050;
            }
        """)
    
    def _apply_dark_theme(self, app):
        """Apply dark theme to application"""
        # Set Fusion style for consistent appearance
        if "Fusion" in self.available_styles:
            app.setStyle("Fusion")
        
        # Create dark palette
        palette = QPalette()
        
        # Set dark colors
        palette.setColor(QPalette.Window, QColor(45, 45, 45))
        palette.setColor(QPalette.WindowText, QColor(230, 230, 230))
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
        palette.setColor(QPalette.Text, QColor(230, 230, 230))
        palette.setColor(QPalette.Button, QColor(45, 45, 45))
        palette.setColor(QPalette.ButtonText, QColor(230, 230, 230))
        palette.setColor(QPalette.Link, QColor(66, 134, 244))
        palette.setColor(QPalette.Highlight, QColor(66, 134, 244))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        
        # Set the palette
        app.setPalette(palette)
        
        # Set stylesheet for additional customization
        app.setStyleSheet("""
            QToolTip { 
                color: #E6E6E6; 
                background-color: #2A2A2A; 
                border: 1px solid #3A3A3A; 
            }
            
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            QTabWidget::pane {
                border: 1px solid #3A3A3A;
                border-radius: 3px;
            }
            
            QTabBar::tab {
                background: #2A2A2A;
                border: 1px solid #3A3A3A;
                border-bottom-color: #3A3A3A;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 5px 10px;
            }
            
            QTabBar::tab:selected {
                background: #323232;
                border-bottom-color: #323232;
            }
            
            QPushButton {
                background-color: #424242;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                padding: 5px 10px;
                color: #E6E6E6;
            }
            
            QPushButton:hover {
                background-color: #505050;
            }
            
            QPushButton:pressed {
                background-color: #606060;
            }
            
            QGroupBox {
                border: 1px solid #3A3A3A;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                color: #AAAAAA;
            }
            
            QLineEdit, QTextEdit, QPlainTextEdit, QComboBox {
                background-color: #323232;
                border: 1px solid #3A3A3A;
                border-radius: 3px;
                padding: 2px;
                color: #E6E6E6;
            }
        """)
    
    def _apply_system_theme(self, app):
        """Apply system-native theme to application"""
        # Reset to system style and palette
        app.setStyle("")  # Reset to system style
        app.setPalette(QPalette())  # Reset to default palette
        app.setStyleSheet("")  # Clear stylesheets
        
        # Find the best available style for the platform
        preferred_styles = ["windowsvista", "Fusion", "Windows"]
        
        for style in preferred_styles:
            if style in self.available_styles:
                app.setStyle(style)
                logger.info(f"Using {style} style for system theme")
                break