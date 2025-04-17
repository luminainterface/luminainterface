from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QWidget, QHBoxLayout, QGridLayout
from PySide6.QtGui import QColor
from ..theme import LuminaTheme

class ModernCard(QFrame):
    """Modern card component following Lumina design system specifications"""
    def __init__(self, title: str = None, parent=None):
        super().__init__(parent)
        self.setObjectName("ModernCard")
        
        # Apply card styling with box-shadow
        self.setStyleSheet(f"""
            QFrame#ModernCard {{
                background-color: {LuminaTheme.COLORS['card']};
                border-radius: {LuminaTheme.SIZES['border_radius']}px;
                border: 1px solid {LuminaTheme.COLORS['border']};
                margin: 2px;  /* Needed for box-shadow to be visible */
                padding: 1px;
            }}
            QFrame#ModernCard:enabled {{
                background-color: {LuminaTheme.COLORS['card']};
                border: 1px solid {LuminaTheme.COLORS['border']};
            }}
            QFrame#ModernCard:disabled {{
                background-color: {LuminaTheme.COLORS['disabled']};
                border: 1px solid {LuminaTheme.COLORS['border']};
            }}
        """)
        
        # Setup layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(
            LuminaTheme.SIZES['padding'],
            LuminaTheme.SIZES['padding'],
            LuminaTheme.SIZES['padding'],
            LuminaTheme.SIZES['padding']
        )
        self.layout.setSpacing(LuminaTheme.SIZES['spacing'])
        
        # Add title if provided
        if title:
            title_label = QLabel(title)
            title_label.setFont(LuminaTheme.FONTS['heading'])
            title_label.setStyleSheet(f"color: {LuminaTheme.COLORS['primary']};")
            self.layout.addWidget(title_label)
            
    def _check_layout(self, widget: QWidget) -> bool:
        """Check if a widget already has a layout"""
        return widget.layout() is not None
            
    def add_section(self, title: str = None) -> QWidget:
        """Add a new section to the card with optional title"""
        section = QWidget(self)
        if not self._check_layout(section):
            section_layout = QVBoxLayout(section)
            section_layout.setContentsMargins(0, 0, 0, 0)
            section_layout.setSpacing(LuminaTheme.SIZES['spacing'])
            
            if title:
                title_label = QLabel(title)
                title_label.setFont(LuminaTheme.FONTS['body'])
                title_label.setStyleSheet(f"color: {LuminaTheme.COLORS['text']};")
                section_layout.addWidget(title_label)
                
        self.layout.addWidget(section)
        return section
        
    def add_grid_section(self, title: str = None, columns: int = 2) -> QWidget:
        """Add a new grid section to the card"""
        section = QWidget(self)
        if not self._check_layout(section):
            section_layout = QGridLayout(section)
            section_layout.setContentsMargins(0, 0, 0, 0)
            section_layout.setSpacing(LuminaTheme.SIZES['spacing'])
            
            if title:
                title_label = QLabel(title)
                title_label.setFont(LuminaTheme.FONTS['body'])
                title_label.setStyleSheet(f"color: {LuminaTheme.COLORS['text']};")
                section_layout.addWidget(title_label, 0, 0, 1, columns)
                
        self.layout.addWidget(section)
        return section
        
    def add_footer(self) -> QWidget:
        """Add a footer section to the card"""
        footer = QWidget(self)
        if not self._check_layout(footer):
            footer_layout = QHBoxLayout(footer)
            footer_layout.setContentsMargins(0, LuminaTheme.SIZES['padding'], 0, 0)
            footer_layout.setSpacing(LuminaTheme.SIZES['spacing'])
            
        self.layout.addWidget(footer)
        return footer
        
    def set_accent(self, color: str = None):
        """Set card accent color"""
        if not color:
            color = LuminaTheme.COLORS['accent']
            
        self.setStyleSheet(f"""
            QFrame#ModernCard {{
                background-color: {LuminaTheme.COLORS['card']};
                border-radius: {LuminaTheme.SIZES['border_radius']}px;
                border: 2px solid {color};
                margin: 2px;
                padding: 1px;
            }}
        """)
        
    def set_loading(self, loading: bool = True):
        """Set card loading state"""
        self.setEnabled(not loading)
        opacity = "0.7" if loading else "1.0"
        self.setStyleSheet(f"""
            QFrame#ModernCard {{
                background-color: {LuminaTheme.COLORS['card']};
                border-radius: {LuminaTheme.SIZES['border_radius']}px;
                border: 1px solid {LuminaTheme.COLORS['border']};
                margin: 2px;
                padding: 1px;
                opacity: {opacity};
            }}
        """) 