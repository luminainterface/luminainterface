import logging
import math
from typing import List, Dict, Optional, Tuple, Any, Callable

from PySide6.QtWidgets import (QWidget, QFrame, QLabel, QPushButton, QVBoxLayout, 
                              QHBoxLayout, QGridLayout, QGraphicsDropShadowEffect,
                              QSizePolicy, QSpacerItem)
from PySide6.QtCore import (Qt, QSize, QPropertyAnimation, QEasingCurve, 
                           QPoint, QTimer, Signal, Property, QPointF, QRectF)
from PySide6.QtGui import (QPainter, QPen, QColor, QBrush, QLinearGradient, 
                          QRadialGradient, QConicalGradient, QPainterPath, 
                          QFont, QPalette, QPixmap, QCursor)

from ui.theme import LuminaTheme

class ModernCard(QFrame):
    """Modern card component following Lumina design system specifications"""
    def __init__(self, title: str = None, parent=None):
        super().__init__(parent)
        self.setObjectName("ModernCard")
        self._accent_color = None
        self._loading = False
        
        # Apply card styling
        self.setStyleSheet(f"""
            QFrame#ModernCard {{
                background-color: {LuminaTheme.COLORS['card']};
                border-radius: {LuminaTheme.SIZES['border_radius']}px;
                border: 1px solid {LuminaTheme.COLORS['border']};
            }}
        """)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(LuminaTheme.SIZES['card_shadow'])
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
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
            title_widget = QWidget()
            title_layout = QHBoxLayout(title_widget)
            title_layout.setContentsMargins(0, 0, 0, 8)
            title_layout.addWidget(title_label)
            title_layout.addStretch()
            
            # Add gold accent line
            accent_line = QFrame()
            accent_line.setFrameShape(QFrame.HLine)
            accent_line.setFrameShadow(QFrame.Sunken)
            accent_line.setStyleSheet(f"background-color: {LuminaTheme.COLORS['accent']}; min-height: 2px;")
            
            self.layout.addWidget(title_widget)
            self.layout.addWidget(accent_line)
            
    def add_section(self, title: str = None) -> QWidget:
        """Add a new section to the card with optional title"""
        section = QWidget(self)
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
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, LuminaTheme.SIZES['padding'], 0, 0)
        footer_layout.setSpacing(LuminaTheme.SIZES['spacing'])
        
        self.layout.addWidget(footer)
        return footer
        
    def set_accent(self, color: str = None):
        """Set card accent color"""
        if not color:
            color = LuminaTheme.COLORS['accent']
            
        self._accent_color = color
        self.setStyleSheet(f"""
            QFrame#ModernCard {{
                background-color: {LuminaTheme.COLORS['card']};
                border-radius: {LuminaTheme.SIZES['border_radius']}px;
                border: 2px solid {color};
            }}
        """)
        
    def set_loading(self, loading: bool = True):
        """Set card loading state"""
        self._loading = loading
        self.setEnabled(not loading)
        self.setStyleSheet(f"""
            QFrame#ModernCard {{
                background-color: {LuminaTheme.COLORS['card']};
                border-radius: {LuminaTheme.SIZES['border_radius']}px;
                border: {f"2px solid {self._accent_color}" if self._accent_color else f"1px solid {LuminaTheme.COLORS['border']}"};
                opacity: {0.7 if loading else 1.0};
            }}
        """)
        
        # Apply loading overlay if needed
        if loading:
            # Future enhancement: add loading animation overlay
            pass
    
    def add_item(self, label: str, value: str = None, icon_path: str = None) -> QWidget:
        """Add a labeled item to the card with optional icon and value"""
        item = QWidget()
        item_layout = QHBoxLayout(item)
        item_layout.setContentsMargins(0, 0, 0, 0)
        item_layout.setSpacing(10)
        
        # Add icon if provided
        if icon_path:
            icon = QLabel()
            pixmap = QPixmap(icon_path)
            icon.setPixmap(pixmap.scaled(20, 20, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            item_layout.addWidget(icon)
        
        # Add label
        label_widget = QLabel(label)
        label_widget.setFont(LuminaTheme.FONTS['body'])
        item_layout.addWidget(label_widget)
        
        # Add value if provided
        if value:
            item_layout.addStretch()
            value_widget = QLabel(value)
            value_widget.setStyleSheet(f"color: {LuminaTheme.COLORS['accent']};")
            item_layout.addWidget(value_widget)
        
        self.layout.addWidget(item)
        return item
    
    def add_button(self, text: str, callback: Callable = None, 
                 color: str = None, icon_path: str = None) -> QPushButton:
        """Add a modern styled button to the card"""
        button = ModernButton(text, icon_path, color)
        if callback:
            button.clicked.connect(callback)
        self.layout.addWidget(button)
        return button


class ModernProgressCircle(QWidget):
    """Circular progress indicator with modern styling"""
    valueChanged = Signal(float)
    
    def __init__(self, parent=None, size: int = 100, stroke_width: int = None, 
               start_color: str = None, end_color: str = None):
        super().__init__(parent)
        self.min_size = size
        self.stroke_width = stroke_width or int(size * 0.1)  # 10% of size by default
        self.animation = None
        self._value = 0
        self.max_value = 100
        self.animation_duration = LuminaTheme.ANIMATION['normal']
        self.setMinimumSize(self.min_size, self.min_size)
        
        # Colors
        self.start_color = start_color or LuminaTheme.COLORS['accent']
        self.end_color = end_color or LuminaTheme.COLORS['success']
        
        # Label settings
        self.show_text = True
        self.text_format = "{value}%"
        self.text_color = LuminaTheme.COLORS['text']
        
        # Progress animation
        self._pulse_effect = False
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._update_pulse)
        self._pulse_angle = 0
        
        # Setup animation
        self._setup_animation()
        
    def _setup_animation(self):
        """Setup value animation"""
        self.animation = QPropertyAnimation(self, b"value")
        self.animation.setDuration(self.animation_duration)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        
    @Property(float)
    def value(self):
        return self._value
        
    @value.setter
    def value(self, val):
        val = max(0, min(val, self.max_value))
        if val != self._value:
            self._value = val
            self.valueChanged.emit(val)
            self.update()
        
    def setValue(self, value: float, animate: bool = True):
        """Set progress value with optional animation"""
        value = max(0, min(value, self.max_value))
        
        if animate and hasattr(self, 'animation'):
            self.animation.stop()
            self.animation.setStartValue(self._value)
            self.animation.setEndValue(value)
            self.animation.start()
        else:
            self.value = value
            
    def setMaxValue(self, max_value: float):
        """Set maximum value"""
        self.max_value = max_value
        self._value = min(self._value, max_value)
        self.update()
        
    def set_pulse_effect(self, enabled: bool):
        """Enable/disable pulse effect animation"""
        self._pulse_effect = enabled
        if enabled:
            self._pulse_timer.start(50)  # Update every 50ms
        else:
            self._pulse_timer.stop()
            
    def _update_pulse(self):
        """Update pulse animation angle"""
        self._pulse_angle = (self._pulse_angle + 5) % 360
        self.update()
        
    def paintEvent(self, event):
        """Paint the progress circle"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate sizes
        width = self.width()
        height = self.height()
        size = min(width, height)
        pen_width = self.stroke_width
        margin = pen_width / 2
        
        # Create rect for circle
        rect = QRectF(
            margin,
            margin, 
            size - pen_width,
            size - pen_width
        )
        
        # Draw background circle
        background_pen = QPen(QColor(LuminaTheme.COLORS['border']))
        background_pen.setWidth(pen_width)
        background_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(background_pen)
        painter.drawArc(rect, 0, 360 * 16)
        
        # Create gradient for progress
        gradient = QConicalGradient(rect.center(), 90)
        gradient.setColorAt(0, QColor(self.start_color))
        gradient.setColorAt(1, QColor(self.end_color))
        
        # Draw progress arc with gradient
        progress_pen = QPen()
        progress_pen.setWidth(pen_width)
        progress_pen.setColor(QColor(self.start_color))
        progress_pen.setCapStyle(Qt.RoundCap)
        
        if self._pulse_effect:
            # Draw pulsing progress
            painter.setPen(progress_pen)
            pulse_angle = self._pulse_angle
            painter.drawArc(rect, pulse_angle * 16, 60 * 16)  # 60° arc that rotates
        else:
            # Draw normal progress
            progress_brush = QBrush(gradient)
            progress_pen.setBrush(progress_brush)
            painter.setPen(progress_pen)
            span = -360 * 16 * (self._value / self.max_value)
            painter.drawArc(rect, 90 * 16, span)
        
        # Draw text if enabled
        if self.show_text:
            painter.setPen(QColor(self.text_color))
            painter.setFont(LuminaTheme.FONTS['heading'])
            
            text = self.text_format.format(value=int(self._value))
            painter.drawText(rect, Qt.AlignCenter, text)
            
    def sizeHint(self):
        return QSize(self.min_size, self.min_size)
        
    def minimumSizeHint(self):
        return QSize(self.min_size, self.min_size)


class ModernButton(QPushButton):
    """Modern button with Lumina styling and animations"""
    def __init__(self, text: str, icon_path: str = None, 
               color: str = None, parent=None):
        super().__init__(text, parent)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        
        # Set properties
        self.color = color or LuminaTheme.COLORS['accent']
        self.hover_animation = QPropertyAnimation(self, b"_hover_value")
        self.hover_animation.setDuration(LuminaTheme.ANIMATION['fast'])
        self.hover_animation.setStartValue(0.0)
        self.hover_animation.setEndValue(1.0)
        self._hover_value = 0.0
        
        # Set icon if provided
        if icon_path:
            self.setIcon(QPixmap(icon_path))
            self.setIconSize(QSize(20, 20))
        
        # Set style
        self.setFont(LuminaTheme.FONTS['body'])
        self.update_style()
        
    def update_style(self):
        """Update button styling"""
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {LuminaTheme.COLORS['background']};
                border: 2px solid {self.color};
                border-radius: {LuminaTheme.SIZES['border_radius']}px;
                padding: 8px 16px;
                color: {self.color};
                font-weight: bold;
            }}
            
            QPushButton:hover {{
                background-color: {self.color};
                color: {LuminaTheme.COLORS['background']};
            }}
            
            QPushButton:pressed {{
                background-color: {LuminaTheme.COLORS['primary']};
                border-color: {LuminaTheme.COLORS['primary']};
                color: {LuminaTheme.COLORS['background']};
            }}
            
            QPushButton:disabled {{
                background-color: {LuminaTheme.COLORS['background']};
                border-color: {LuminaTheme.COLORS['disabled']};
                color: {LuminaTheme.COLORS['disabled']};
            }}
        """)
        
    def enterEvent(self, event):
        """Handle mouse enter event with animation"""
        self.hover_animation.setDirection(QPropertyAnimation.Forward)
        self.hover_animation.start()
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        """Handle mouse leave event with animation"""
        self.hover_animation.setDirection(QPropertyAnimation.Backward)
        self.hover_animation.start()
        super().leaveEvent(event)
        
    @Property(float)
    def _hover_value(self):
        return self.__hover_value
        
    @_hover_value.setter
    def _hover_value(self, value):
        self.__hover_value = value
        self.update()
        
    def set_color(self, color: str):
        """Change button color"""
        self.color = color
        self.update_style()


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


class ModernMetricsCard(ModernCard):
    """Modern card for displaying metrics with charts and indicators"""
    def __init__(self, title: str = None, parent=None):
        super().__init__(title, parent)
        self.metrics = {}
        self.charts = {}
        self.indicators = {}
        self.update_interval = 1000  # Default update interval (ms)
        
        # Setup metrics container
        self.metrics_container = QWidget()
        self.metrics_layout = QGridLayout(self.metrics_container)
        self.metrics_layout.setContentsMargins(0, 0, 0, 0)
        self.metrics_layout.setSpacing(LuminaTheme.SIZES['spacing'])
        self.layout.addWidget(self.metrics_container)
        
    def add_metric(self, name: str, label: str, unit: str = "", 
                 min_value: float = 0, max_value: float = 100, 
                 color: str = None, chart_type: str = "line") -> dict:
        """Add a new metric to the card"""
        metric = {
            'name': name,
            'label': label,
            'unit': unit,
            'min_value': min_value,
            'max_value': max_value,
            'color': color or LuminaTheme.COLORS['accent'],
            'chart_type': chart_type,
            'data': [],
            'max_data_points': 100
        }
        
        # Create metric widget
        metric_widget = QWidget()
        metric_layout = QVBoxLayout(metric_widget)
        metric_layout.setContentsMargins(0, 0, 0, 0)
        metric_layout.setSpacing(5)
        
        # Header with label and value
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(10)
        
        # Label
        label_widget = QLabel(label)
        label_widget.setFont(LuminaTheme.FONTS['body'])
        header_layout.addWidget(label_widget)
        
        # Spacer
        header_layout.addStretch()
        
        # Value
        value_widget = QLabel(f"0 {unit}")
        value_widget.setStyleSheet(f"color: {metric['color']};")
        value_widget.setFont(LuminaTheme.FONTS['body'])
        header_layout.addWidget(value_widget)
        
        metric_layout.addWidget(header)
        
        # Chart placeholder (will be implemented in derived classes)
        chart_widget = QFrame()
        chart_widget.setMinimumHeight(80)
        chart_widget.setStyleSheet(f"""
            background-color: {LuminaTheme.COLORS['background']};
            border-radius: {LuminaTheme.SIZES['border_radius']}px;
        """)
        
        metric_layout.addWidget(chart_widget)
        
        # Store references
        self.metrics[name] = metric
        self.charts[name] = chart_widget
        self.indicators[name] = {'label': label_widget, 'value': value_widget}
        
        # Add to layout
        row = len(self.metrics) // 2
        col = len(self.metrics) % 2
        self.metrics_layout.addWidget(metric_widget, row, col)
        
        return metric
        
    def update_metric(self, name: str, value: float):
        """Update a metric with a new value"""
        if name not in self.metrics:
            return
            
        metric = self.metrics[name]
        
        # Update data
        metric['data'].append(value)
        if len(metric['data']) > metric['max_data_points']:
            metric['data'].pop(0)
            
        # Update value indicator
        if name in self.indicators:
            if metric['unit'] == '%':
                self.indicators[name]['value'].setText(f"{value:.1f}{metric['unit']}")
            elif metric['unit'] == 'ms':
                self.indicators[name]['value'].setText(f"{value:.1f} {metric['unit']}")
            else:
                self.indicators[name]['value'].setText(f"{value:.2f} {metric['unit']}")
                
        # Future enhancement: update chart
        
    def set_update_interval(self, interval: int):
        """Set the update interval for metrics in milliseconds"""
        self.update_interval = interval


class ModernLogViewer(QFrame):
    """Modern log viewer with syntax highlighting and filtering"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ModernLogViewer")
        self.setStyleSheet(f"""
            QFrame#ModernLogViewer {{
                background-color: {LuminaTheme.COLORS['card']};
                border-radius: {LuminaTheme.SIZES['border_radius']}px;
                border: 1px solid {LuminaTheme.COLORS['border']};
            }}
        """)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(LuminaTheme.SIZES['card_shadow'])
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        # Setup layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(
            LuminaTheme.SIZES['padding'],
            LuminaTheme.SIZES['padding'],
            LuminaTheme.SIZES['padding'],
            LuminaTheme.SIZES['padding']
        )
        self.layout.setSpacing(LuminaTheme.SIZES['spacing'])
        
        # Header with controls
        self.header = QWidget()
        self.header_layout = QHBoxLayout(self.header)
        self.header_layout.setContentsMargins(0, 0, 0, 0)
        self.header_layout.setSpacing(10)
        
        # Title
        self.title = QLabel("System Logs")
        self.title.setFont(LuminaTheme.FONTS['heading'])
        self.header_layout.addWidget(self.title)
        
        # Search field (placeholder for future enhancement)
        self.header_layout.addStretch()
        
        # Log level filter (placeholder for future enhancement)
        
        # Clear button
        self.clear_button = ModernButton("Clear", color=LuminaTheme.COLORS['warning'])
        self.clear_button.clicked.connect(self.clear_logs)
        self.header_layout.addWidget(self.clear_button)
        
        self.layout.addWidget(self.header)
        
        # Log text display (placeholder)
        self.log_display = QLabel("Log output will appear here...")
        self.log_display.setWordWrap(True)
        self.log_display.setFont(QFont("Consolas", 10))
        self.log_display.setStyleSheet(f"""
            background-color: {LuminaTheme.COLORS['background']};
            color: {LuminaTheme.COLORS['text']};
            border-radius: {LuminaTheme.SIZES['border_radius']}px;
            padding: 10px;
        """)
        
        self.layout.addWidget(self.log_display)
        
    def add_log(self, message: str, level: str = "INFO"):
        """Add a log entry"""
        # Future enhancement: implement proper log display with parsing, 
        # formatting and scrolling
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Colorize based on level
        level_color = {
            "DEBUG": LuminaTheme.COLORS['success'],
            "INFO": LuminaTheme.COLORS['accent'],
            "WARNING": LuminaTheme.COLORS['warning'],
            "ERROR": LuminaTheme.COLORS['error']
        }.get(level, LuminaTheme.COLORS['text'])
        
        # Update display (this is a placeholder implementation)
        current_text = self.log_display.text()
        if current_text == "Log output will appear here...":
            current_text = ""
            
        new_entry = f"<span style='color:{level_color}'>{timestamp} - {level}</span> {message}<br>"
        self.log_display.setText(current_text + new_entry)
        
    def clear_logs(self):
        """Clear all logs"""
        self.log_display.setText("Log output will appear here...") 