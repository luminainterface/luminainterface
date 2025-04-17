"""
LUMINA UI Components
Custom UI components that inherit from PySide6 widgets and apply the Lumina theme
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame, QScrollArea, QGroupBox,
                             QTextEdit, QComboBox, QSpinBox, QSlider)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QIcon, QColor

from ui.theme import LuminaTheme
from .components.modern_card import ModernCard
from .components.modern_progress_circle import ModernProgressCircle
from .components.modern_button import ModernButton
from .components.modern_metrics_card import ModernMetricsCard
from .components.modern_log_viewer import ModernLogViewer

__all__ = [
    'LuminaButton', 'LuminaLabel', 'LuminaTextEdit', 'LuminaComboBox',
    'LuminaSpinBox', 'LuminaSlider', 'LuminaGroupBox', 'LuminaScrollArea',
    'LuminaFrame', 'ModernCard', 'ModernProgressCircle', 'ModernButton',
    'ModernMetricsCard', 'ModernLogViewer'
]

class LuminaButton(QPushButton):
    """Custom button with Lumina styling"""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(LuminaTheme.get_stylesheet())
        self.setMinimumHeight(30)
        self.setFont(QFont(LuminaTheme.FONT_FAMILY, LuminaTheme.FONT_SIZE))

class LuminaLabel(QLabel):
    """Custom label with Lumina styling"""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(LuminaTheme.get_stylesheet())
        self.setFont(QFont(LuminaTheme.FONT_FAMILY, LuminaTheme.FONT_SIZE))

class LuminaTextEdit(QTextEdit):
    """Custom text edit with Lumina styling"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(LuminaTheme.get_stylesheet())
        self.setFont(QFont(LuminaTheme.FONT_FAMILY, LuminaTheme.FONT_SIZE))

class LuminaComboBox(QComboBox):
    """Custom combo box with Lumina styling"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(LuminaTheme.get_stylesheet())
        self.setFont(QFont(LuminaTheme.FONT_FAMILY, LuminaTheme.FONT_SIZE))

class LuminaSpinBox(QSpinBox):
    """Custom spin box with Lumina styling"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(LuminaTheme.get_stylesheet())
        self.setFont(QFont(LuminaTheme.FONT_FAMILY, LuminaTheme.FONT_SIZE))

class LuminaSlider(QSlider):
    """Custom slider with Lumina styling"""
    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setStyleSheet(LuminaTheme.get_stylesheet())

class LuminaGroupBox(QGroupBox):
    """Custom group box with Lumina styling"""
    def __init__(self, title="", parent=None):
        super().__init__(title, parent)
        self.setStyleSheet(LuminaTheme.get_stylesheet())
        self.setFont(QFont(LuminaTheme.FONT_FAMILY, LuminaTheme.FONT_SIZE))

class LuminaScrollArea(QScrollArea):
    """Custom scroll area with Lumina styling"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(LuminaTheme.get_stylesheet())
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)

class LuminaFrame(QFrame):
    """A styled frame for visual separation."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(LuminaTheme.get_stylesheet())
        self.setFont(LuminaTheme.get_font())

class ModernCard(QFrame):
    """A modern card component with shadow and rounded corners."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            ModernCard {
                background-color: #ffffff;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
                padding: 16px;
                margin: 8px;
            }
            ModernCard:hover {
                background-color: #f5f5f5;
            }
        """) 