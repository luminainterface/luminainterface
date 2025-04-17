"""
V6 Panel Base Class

This module provides the base panel implementation for V6 UI components,
with a modern holographic appearance and improved sizing.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    # Import Qt compatibility layer from V5
    from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui, Qt, Signal, Slot
    from src.v5.ui.qt_compat import get_widgets, get_gui, get_core
except ImportError:
    logging.warning("V5 Qt compatibility layer not found. Using direct PySide6 imports.")
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        from PySide6.QtCore import Qt, Signal, Slot
        from PySide6.QtWidgets import QGraphicsDropShadowEffect  # Fix for PySide6
        
        # Simple compatibility functions
        def get_widgets():
            return QtWidgets
            
        def get_gui():
            return QtGui
            
        def get_core():
            return QtCore
    except ImportError:
        logging.error("PySide6 not found. Please install PySide6 or configure the V5 Qt compatibility layer.")
        sys.exit(1)

# Get required Qt classes
QSplitter = get_widgets().QSplitter
QPainter = get_gui().QPainter
QLinearGradient = get_gui().QLinearGradient
QRadialGradient = get_gui().QRadialGradient
QColor = get_gui().QColor
QFont = get_gui().QFont
QPen = get_gui().QPen
QTabWidget = get_widgets().QTabWidget

# In some Qt versions, QGraphicsDropShadowEffect is in QtWidgets not QtGui
try:
    QGraphicsDropShadowEffect = get_gui().QGraphicsDropShadowEffect
except AttributeError:
    QGraphicsDropShadowEffect = get_widgets().QGraphicsDropShadowEffect

QBrush = get_gui().QBrush

# Set up logging
logger = logging.getLogger(__name__)

class V6PanelBase(QtWidgets.QWidget):
    """
    Base class for all V6 panels with a modern holographic appearance.
    
    This class provides common functionality for all V6 panels, including:
    - Responsive sizing with minimum dimensions
    - Semi-transparent backgrounds with subtle gradients
    - Holographic-style glowing edges
    - Consistent styling across all panels
    """
    
    def __init__(self, parent=None):
        """Initialize the V6 panel base"""
        super().__init__(parent)
        self.setMinimumSize(400, 300)  # Minimum size
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, 
            QtWidgets.QSizePolicy.Expanding
        )
        self._init_styling()
        
        # For the holographic effect
        self.glow_opacity = 0.0
        self.glow_timer = QtCore.QTimer(self)
        self.glow_timer.timeout.connect(self._update_glow)
        self.glow_timer.start(50)  # 50ms update
        self.glow_direction = 1  # 1 for increasing, -1 for decreasing
    
    def _init_styling(self):
        """Initialize panel styling"""
        self.setStyleSheet("""
            V6PanelBase {
                border: none;
                color: #ECF0F1;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
            }
        """)
        
        # Add glow effect
        self.glow_effect = QGraphicsDropShadowEffect()
        self.glow_effect.setBlurRadius(15)
        self.glow_effect.setColor(QColor(52, 152, 219, 150))  # Blue glow
        self.glow_effect.setOffset(0, 0)
        self.setGraphicsEffect(self.glow_effect)
    
    def _update_glow(self):
        """Update the glow animation"""
        # Update glow opacity for pulsing effect
        self.glow_opacity += 0.02 * self.glow_direction
        if self.glow_opacity >= 1.0:
            self.glow_opacity = 1.0
            self.glow_direction = -1
        elif self.glow_opacity <= 0.5:
            self.glow_opacity = 0.5
            self.glow_direction = 1
            
        # Update the glow effect
        color = QColor(52, 152, 219, int(100 * self.glow_opacity))
        self.glow_effect.setColor(color)
    
    def paintEvent(self, event):
        """Custom paint event for holographic background"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Create a semi-transparent background
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(32, 60, 86, 180))  # Semi-transparent dark blue at top
        gradient.setColorAt(1, QColor(16, 30, 43, 150))  # More transparent darker blue at bottom
        
        # Fill the background
        painter.fillRect(self.rect(), gradient)
        
        # Add a subtle grid pattern for holographic effect
        self._draw_holographic_grid(painter)
        
        # Draw a glowing border
        self._draw_glowing_border(painter)
    
    def _draw_holographic_grid(self, painter):
        """Draw a subtle grid pattern for holographic effect"""
        # Set up the pen for grid lines
        pen = QPen(QColor(52, 152, 219, 30))  # Very transparent blue
        pen.setWidth(1)
        painter.setPen(pen)
        
        # Draw horizontal lines
        spacing = 20
        for y in range(0, self.height(), spacing):
            painter.drawLine(0, y, self.width(), y)
        
        # Draw vertical lines
        for x in range(0, self.width(), spacing):
            painter.drawLine(x, 0, x, self.height())
    
    def _draw_glowing_border(self, painter):
        """Draw a glowing border around the panel"""
        # Set up the pen for the border
        pen = QPen(QColor(52, 152, 219, int(150 * self.glow_opacity)))
        pen.setWidth(2)
        painter.setPen(pen)
        
        # Draw the border
        painter.drawRoundedRect(1, 1, self.width() - 2, self.height() - 2, 6, 6)

class V6PanelContainer(QtWidgets.QWidget):
    """
    A container for V6 panels with enhanced styling and tab controls.
    This container supports multiple tabbed panels with a holographic appearance.
    """
    
    def __init__(self, title, panels=None, parent=None):
        """
        Initialize a panel container with tabs
        
        Args:
            title (str): The title of the panel container
            panels (dict): Dictionary of {tab_name: panel_widget} pairs
            parent (QWidget, optional): Parent widget
        """
        super().__init__(parent)
        self.title = title
        self.panels = panels or {}
        self.initUI()
        
        # For the holographic effect
        self.glow_opacity = 0.0
        self.glow_timer = QtCore.QTimer(self)
        self.glow_timer.timeout.connect(self._update_glow)
        self.glow_timer.start(80)  # 80ms update
        self.glow_direction = 1  # 1 for increasing, -1 for decreasing
    
    def initUI(self):
        """Initialize the user interface"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create title bar with holographic appearance
        title_bar = QtWidgets.QWidget()
        title_bar.setFixedHeight(40)
        title_bar.setStyleSheet("""
            background-color: rgba(26, 38, 52, 200);
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
        """)
        
        title_layout = QtWidgets.QHBoxLayout(title_bar)
        title_layout.setContentsMargins(15, 0, 15, 0)
        
        # Title with glow effect
        title_label = QtWidgets.QLabel(self.title)
        title_label.setStyleSheet("""
            color: #3498DB;
            font-weight: bold;
            font-size: 16px;
        """)
        
        # Add glow effect to title
        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(10)
        glow.setColor(QColor(52, 152, 219, 150))
        glow.setOffset(0, 0)
        title_label.setGraphicsEffect(glow)
        
        # Add to layout
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        # Add title bar to main layout
        layout.addWidget(title_bar)
        
        # Create a container for the panels with proper styling
        panel_container = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(panel_container)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        panel_layout.setSpacing(0)
        
        # Create tab widget with holographic styling
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background-color: transparent;
            }
            
            QTabBar::tab {
                background-color: rgba(44, 62, 80, 150);
                color: #ECF0F1;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            
            QTabBar::tab:selected {
                background-color: rgba(52, 152, 219, 180);
                color: white;
                font-weight: bold;
            }
            
            QTabBar::tab:hover:!selected {
                background-color: rgba(52, 73, 94, 180);
            }
        """)
        
        # Add panels to tabs if provided
        if self.panels:
            for tab_name, panel in self.panels.items():
                self.tab_widget.addTab(panel, tab_name)
        
        panel_layout.addWidget(self.tab_widget)
        layout.addWidget(panel_container, 1)
        
        # Apply holographic effect to the container
        self.setStyleSheet("""
            background-color: transparent;
        """)
    
    def _update_glow(self):
        """Update the glow animation for title"""
        # Update glow opacity for pulsing effect
        self.glow_opacity += 0.03 * self.glow_direction
        if self.glow_opacity >= 1.0:
            self.glow_opacity = 1.0
            self.glow_direction = -1
        elif self.glow_opacity <= 0.6:
            self.glow_opacity = 0.6
            self.glow_direction = 1
        
        # Find the title label and update its glow
        for i in range(self.layout().count()):
            widget = self.layout().itemAt(i).widget()
            if isinstance(widget, QtWidgets.QWidget):
                for child in widget.findChildren(QtWidgets.QLabel):
                    if child.text() == self.title:
                        effect = child.graphicsEffect()
                        if effect:
                            color = QColor(52, 152, 219, int(150 * self.glow_opacity))
                            effect.setColor(color)
                            break
    
    def addPanel(self, name, panel):
        """
        Add a new panel to the container
        
        Args:
            name (str): The tab name
            panel (QWidget): The panel widget to add
        """
        self.panels[name] = panel
        self.tab_widget.addTab(panel, name)
    
    def removePanel(self, name):
        """
        Remove a panel from the container
        
        Args:
            name (str): The tab name to remove
        """
        if name in self.panels:
            for i in range(self.tab_widget.count()):
                if self.tab_widget.tabText(i) == name:
                    self.tab_widget.removeTab(i)
                    del self.panels[name]
                    break
    
    def paintEvent(self, event):
        """Custom paint event for holographic container background"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw a glowing border
        pen = QPen(QColor(52, 152, 219, int(150 * self.glow_opacity)))
        pen.setWidth(2)
        painter.setPen(pen)
        
        # Draw the border
        painter.drawRoundedRect(1, 1, self.width() - 2, self.height() - 2, 6, 6) 