import logging
from PySide6.QtGui import QFont, QPalette, QColor, QFontDatabase
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

class LuminaTheme:
    """LUMINA design system theme specification"""
    
    # Color palette
    COLORS = {
        'primary': '#000000',      # Black
        'accent': '#C6A962',       # Gold
        'background': '#F5F5F2',   # Off-white
        'card': '#FFFFFF',         # White
        'text': '#1A1A1A',         # Dark gray
        'success': '#4A5D4F',      # Muted green
        'warning': '#8B7355',      # Muted brown
        'error': '#8B4545',        # Muted red
        'border': '#E0E0E0',       # Light gray for borders
        'disabled': '#CCCCCC'      # Disabled state
    }
    
    # Typography
    FONTS = {}  # Will be filled during setup
    
    # Sizes
    SIZES = {
        'padding': 15,
        'spacing': 20,
        'border_radius': 8,
        'card_shadow': 15,
        'icon_size_small': 16,
        'icon_size_medium': 24,
        'icon_size_large': 32
    }
    
    # Animation durations
    ANIMATION = {
        'fast': 150,   # ms
        'normal': 300, # ms
        'slow': 500    # ms
    }
    
    @classmethod
    def setup_fonts(cls):
        """Initialize font configuration"""
        try:
            # Use system fonts
            title_font = QFont("Segoe UI")
            title_font.setPointSize(24)
            title_font.setBold(True)
            
            heading_font = QFont("Segoe UI")
            heading_font.setPointSize(16)
            heading_font.setBold(True)
            
            body_font = QFont("Segoe UI")
            body_font.setPointSize(12)
            
            small_font = QFont("Segoe UI") 
            small_font.setPointSize(10)
            
            cls.FONTS = {
                'title': title_font,
                'heading': heading_font,
                'body': body_font,
                'small': small_font
            }
            
            logging.info("LuminaTheme fonts initialized successfully")
        except Exception as e:
            logging.error(f"Error setting up fonts: {str(e)}")
            # Fallback to basic system font if there's an error
            cls.FONTS = {
                'title': QFont("Segoe UI", 24, QFont.Bold),
                'heading': QFont("Segoe UI", 16, QFont.Bold),
                'body': QFont("Segoe UI", 12),
                'small': QFont("Segoe UI", 10)
            }
    
    @classmethod
    def setup_application_theme(cls, app: QApplication):
        """Apply LUMINA theme to entire application"""
        if not cls.FONTS:
            cls.setup_fonts()
            
        # Set application font
        app.setFont(cls.FONTS['body'])
        
        # Create and configure palette
        palette = QPalette()
        
        # Set color roles
        palette.setColor(QPalette.Window, QColor(cls.COLORS['background']))
        palette.setColor(QPalette.WindowText, QColor(cls.COLORS['text']))
        palette.setColor(QPalette.Base, QColor(cls.COLORS['card']))
        palette.setColor(QPalette.AlternateBase, QColor(cls.COLORS['background']))
        palette.setColor(QPalette.Text, QColor(cls.COLORS['text']))
        palette.setColor(QPalette.Button, QColor(cls.COLORS['background']))
        palette.setColor(QPalette.ButtonText, QColor(cls.COLORS['text']))
        palette.setColor(QPalette.BrightText, QColor(cls.COLORS['accent']))
        palette.setColor(QPalette.Link, QColor(cls.COLORS['accent']))
        palette.setColor(QPalette.Highlight, QColor(cls.COLORS['accent']))
        palette.setColor(QPalette.HighlightedText, QColor(cls.COLORS['card']))
        
        # Set application palette
        app.setPalette(palette)
        
        # Set stylesheet for global styling
        app.setStyleSheet(f"""
            QToolTip {{
                background-color: {cls.COLORS['card']};
                color: {cls.COLORS['text']};
                border: 1px solid {cls.COLORS['border']};
                border-radius: {cls.SIZES['border_radius']}px;
                padding: 5px;
            }}
            
            QScrollBar:vertical {{
                border: none;
                background-color: {cls.COLORS['background']};
                width: 10px;
                margin: 0px;
            }}
            
            QScrollBar::handle:vertical {{
                background-color: {cls.COLORS['border']};
                border-radius: 5px;
            }}
            
            QScrollBar::handle:vertical:hover {{
                background-color: {cls.COLORS['accent']};
            }}
            
            QScrollBar:horizontal {{
                border: none;
                background-color: {cls.COLORS['background']};
                height: 10px;
                margin: 0px;
            }}
            
            QScrollBar::handle:horizontal {{
                background-color: {cls.COLORS['border']};
                border-radius: 5px;
            }}
            
            QScrollBar::handle:horizontal:hover {{
                background-color: {cls.COLORS['accent']};
            }}
        """)
        
        logging.info("LuminaTheme applied to application")
 