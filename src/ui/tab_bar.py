import logging
from PySide6.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QHBoxLayout, 
                              QFrame, QSizePolicy, QLabel)
from PySide6.QtCore import (Qt, QSize, QPropertyAnimation, QEasingCurve, 
                           QPoint, Signal)
from PySide6.QtGui import (QPixmap, QFont, QCursor)

from ui.theme import LuminaTheme

class ModernTabBar(QWidget):
    """Modern tab widget with gold underline and animated transitions"""
    tabChanged = Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Tab header
        self.header = QWidget()
        self.header_layout = QHBoxLayout(self.header)
        self.header_layout.setContentsMargins(0, 0, 0, 0)
        self.header_layout.setSpacing(0)
        
        # Tab content
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, LuminaTheme.SIZES['padding'], 0, 0)
        
        # Add to main layout
        self.layout.addWidget(self.header)
        self.layout.addWidget(self.content)
        
        # Tab properties
        self.tabs = []
        self.tab_buttons = []
        self.current_index = -1
        self.indicator = QFrame(self.header)
        self.indicator.setFrameShape(QFrame.HLine)
        self.indicator.setFixedHeight(2)
        self.indicator.setStyleSheet(f"background-color: {LuminaTheme.COLORS['accent']};")
        self.indicator.raise_()
        self.indicator.hide()
        
        # Indicator animation
        self.indicator_animation = QPropertyAnimation(self.indicator, b"pos")
        self.indicator_animation.setDuration(LuminaTheme.ANIMATION['normal'])
        self.indicator_animation.setEasingCurve(QEasingCurve.OutCubic)
        
    def addTab(self, widget: QWidget, title: str) -> int:
        """Add a new tab"""
        # Create tab button
        tab_button = QPushButton(title)
        tab_button.setCheckable(True)
        tab_button.setCursor(QCursor(Qt.PointingHandCursor))
        tab_button.setFont(LuminaTheme.FONTS['body'])
        tab_button.setFixedHeight(40)
        tab_button.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                border-bottom: 1px solid {LuminaTheme.COLORS['border']};
                padding: 10px 20px;
                color: {LuminaTheme.COLORS['text']};
                text-align: center;
            }}
            
            QPushButton:checked {{
                color: {LuminaTheme.COLORS['accent']};
                border-bottom: 1px solid {LuminaTheme.COLORS['accent']};
                font-weight: bold;
            }}
            
            QPushButton:hover:!checked {{
                color: {LuminaTheme.COLORS['accent']};
            }}
        """)
        
        # Connect button to tab selection
        index = len(self.tabs)
        tab_button.clicked.connect(lambda: self.setCurrentIndex(index))
        
        # Add to layout
        self.header_layout.addWidget(tab_button)
        self.tab_buttons.append(tab_button)
        
        # Add widget to content
        widget.setVisible(False)
        self.content_layout.addWidget(widget)
        self.tabs.append(widget)
        
        # Select first tab automatically
        if len(self.tabs) == 1:
            self.setCurrentIndex(0)
            
        # Add stretch at the end to left-align tabs
        if len(self.tabs) == 1:
            self.header_layout.addStretch()
            
        return index
        
    def setCurrentIndex(self, index: int):
        """Set the current tab index"""
        if index < 0 or index >= len(self.tabs) or index == self.current_index:
            return
            
        # Update tab buttons
        for i, button in enumerate(self.tab_buttons):
            button.setChecked(i == index)
            
        # Hide previous tab
        if self.current_index >= 0:
            self.tabs[self.current_index].setVisible(False)
            
        # Show current tab
        self.tabs[index].setVisible(True)
        self.current_index = index
        
        # Animate indicator
        self._update_indicator()
        
        # Emit signal
        self.tabChanged.emit(index)
        
    def _update_indicator(self):
        """Update indicator position with animation"""
        if self.current_index < 0:
            self.indicator.hide()
            return
            
        button = self.tab_buttons[self.current_index]
        target_pos = button.pos() + QPoint(0, button.height() - 2)
        indicator_width = button.width()
        
        # Set indicator width
        self.indicator.setFixedWidth(indicator_width)
        
        # If initial position, just place it without animation
        if self.indicator.isHidden():
            self.indicator.move(target_pos)
            self.indicator.show()
        else:
            # Animate movement
            self.indicator_animation.setStartValue(self.indicator.pos())
            self.indicator_animation.setEndValue(target_pos)
            self.indicator_animation.start()
            
    def setTabEnabled(self, index: int, enabled: bool):
        """Enable or disable a tab"""
        if 0 <= index < len(self.tab_buttons):
            self.tab_buttons[index].setEnabled(enabled)
            
    def setTabText(self, index: int, text: str):
        """Change tab text"""
        if 0 <= index < len(self.tab_buttons):
            self.tab_buttons[index].setText(text)
            
    def setTabIcon(self, index: int, icon_path: str):
        """Set tab icon"""
        if 0 <= index < len(self.tab_buttons):
            pixmap = QPixmap(icon_path)
            if not pixmap.isNull():
                self.tab_buttons[index].setIcon(pixmap)
                self.tab_buttons[index].setIconSize(QSize(16, 16))
                
    def tabText(self, index: int) -> str:
        """Get tab text"""
        if 0 <= index < len(self.tab_buttons):
            return self.tab_buttons[index].text()
        return ""
        
    def count(self) -> int:
        """Get number of tabs"""
        return len(self.tabs)
        
    def currentIndex(self) -> int:
        """Get current tab index"""
        return self.current_index
        
    def currentWidget(self) -> QWidget:
        """Get current widget"""
        if 0 <= self.current_index < len(self.tabs):
            return self.tabs[self.current_index]
        return None
        
    def add_tab_with_close(self, widget: QWidget, title: str) -> int:
        """Add a tab with close button"""
        index = self.addTab(widget, title)
        
        # Add close button to this tab
        button = self.tab_buttons[index]
        close_label = QLabel("✕")
        close_label.setStyleSheet(f"""
            QLabel {{
                color: {LuminaTheme.COLORS['text']};
                margin-left: 5px;
                font-size: 12px;
            }}
            QLabel:hover {{
                color: {LuminaTheme.COLORS['error']};
            }}
        """)
        close_label.setCursor(QCursor(Qt.PointingHandCursor))
        # Add close functionality in future enhancement
        
        return index 