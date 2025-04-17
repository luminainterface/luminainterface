"""
Theme configuration for LUMINA v7.5
"""

from PySide6.QtGui import QColor, QPalette, QFont
from PySide6.QtCore import Qt

class LuminaTheme:
    """Theme configuration for LUMINA"""
    
    # Color definitions
    PRIMARY = QColor("#D4AF37")  # Gold
    SECONDARY = QColor("#1A1A1A")  # Dark gray
    BACKGROUND = QColor("#1A1A1A")  # Dark gray
    TEXT = QColor("#FFFFFF")  # White
    ACCENT = QColor("#FFD700")  # Bright gold
    ERROR = QColor("#FF4444")  # Red
    SUCCESS = QColor("#44FF44")  # Green
    WARNING = QColor("#FFAA44")  # Orange
    
    # Font definitions
    FONT_FAMILY = "Consolas"
    FONT_SIZE = 12
    FONT_WEIGHT = QFont.Weight.Normal
    
    @staticmethod
    def apply_theme(app):
        """Apply the theme to the application"""
        # Set the application style
        app.setStyle("Fusion")
        
        # Create and set the palette
        palette = QPalette()
        
        # Base colors
        palette.setColor(QPalette.Window, LuminaTheme.BACKGROUND)
        palette.setColor(QPalette.WindowText, LuminaTheme.TEXT)
        palette.setColor(QPalette.Base, LuminaTheme.SECONDARY)
        palette.setColor(QPalette.AlternateBase, LuminaTheme.BACKGROUND)
        palette.setColor(QPalette.ToolTipBase, LuminaTheme.BACKGROUND)
        palette.setColor(QPalette.ToolTipText, LuminaTheme.TEXT)
        palette.setColor(QPalette.Text, LuminaTheme.TEXT)
        palette.setColor(QPalette.Button, LuminaTheme.SECONDARY)
        palette.setColor(QPalette.ButtonText, LuminaTheme.TEXT)
        palette.setColor(QPalette.BrightText, LuminaTheme.ACCENT)
        palette.setColor(QPalette.Link, LuminaTheme.PRIMARY)
        palette.setColor(QPalette.Highlight, LuminaTheme.PRIMARY)
        palette.setColor(QPalette.HighlightedText, LuminaTheme.BACKGROUND)
        
        # Disabled colors
        palette.setColor(QPalette.Disabled, QPalette.WindowText, LuminaTheme.TEXT.darker())
        palette.setColor(QPalette.Disabled, QPalette.Text, LuminaTheme.TEXT.darker())
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, LuminaTheme.TEXT.darker())
        palette.setColor(QPalette.Disabled, QPalette.Highlight, LuminaTheme.PRIMARY.darker())
        palette.setColor(QPalette.Disabled, QPalette.HighlightedText, LuminaTheme.BACKGROUND.darker())
        
        # Set the application palette
        app.setPalette(palette)
        
        # Set the application font
        font = QFont(LuminaTheme.FONT_FAMILY, LuminaTheme.FONT_SIZE, LuminaTheme.FONT_WEIGHT)
        app.setFont(font)
        
    @staticmethod
    def get_stylesheet():
        """Get the application stylesheet"""
        return f"""
            QMainWindow, QWidget {{
                background-color: {LuminaTheme.BACKGROUND.name()};
                color: {LuminaTheme.TEXT.name()};
            }}
            
            QGroupBox {{
                background-color: {LuminaTheme.SECONDARY.name()};
                border: 1px solid {LuminaTheme.PRIMARY.name()};
                padding: 15px;
                margin-top: 1.5em;
            }}
            
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {LuminaTheme.PRIMARY.name()};
                font-family: '{LuminaTheme.FONT_FAMILY}';
                font-weight: bold;
                font-size: 13px;
            }}
            
            QComboBox, QSpinBox {{
                background-color: {LuminaTheme.SECONDARY.name()};
                border: 1px solid {LuminaTheme.PRIMARY.name()};
                color: {LuminaTheme.TEXT.name()};
                padding: 5px;
                font-family: '{LuminaTheme.FONT_FAMILY}';
            }}
            
            QSlider::handle {{
                background: {LuminaTheme.PRIMARY.name()};
                border: 1px solid {LuminaTheme.PRIMARY.name()};
            }}
            
            QSlider::groove:horizontal {{
                border: 1px solid {LuminaTheme.PRIMARY.name()};
                height: 4px;
                background: {LuminaTheme.SECONDARY.name()};
            }}
            
            QPushButton {{
                background-color: {LuminaTheme.SECONDARY.name()};
                border: 1px solid {LuminaTheme.PRIMARY.name()};
                color: {LuminaTheme.PRIMARY.name()};
                padding: 5px 10px;
                font-family: '{LuminaTheme.FONT_FAMILY}';
                font-size: 12px;
            }}
            
            QPushButton:hover {{
                background-color: {LuminaTheme.SECONDARY.name()};
                border-color: {LuminaTheme.ACCENT.name()};
            }}
            
            QLabel {{
                font-family: '{LuminaTheme.FONT_FAMILY}';
                font-size: 12px;
                color: {LuminaTheme.TEXT.name()};
            }}
            
            QTextEdit {{
                background-color: {LuminaTheme.SECONDARY.name()};
                border: 1px solid {LuminaTheme.PRIMARY.name()};
                color: {LuminaTheme.TEXT.name()};
                font-family: '{LuminaTheme.FONT_FAMILY}';
                font-size: 12px;
                padding: 5px;
            }}
            
            QScrollBar:vertical {{
                background-color: {LuminaTheme.SECONDARY.name()};
                width: 12px;
                margin: 0px;
            }}
            
            QScrollBar::handle:vertical {{
                background-color: {LuminaTheme.PRIMARY.name()};
                min-height: 20px;
                border-radius: 6px;
            }}
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """ 