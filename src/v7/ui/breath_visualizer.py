#!/usr/bin/env python
"""
Breath Visualizer Component for V7 Node Consciousness Visualization

This module provides a visualization component for displaying breath patterns
and their relationship to the V7 Node Consciousness system.
"""

import logging
import time
import math
from typing import Dict, Any, Optional, List, Tuple

try:
    from PySide6.QtCore import Qt, QTimer, Signal, Slot, QRectF, QPointF
    from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
    PYSIDE6_AVAILABLE = True
except ImportError:
    try:
        from PyQt5.QtCore import Qt, QTimer, pyqtSignal as Signal, pyqtSlot as Slot, QRectF, QPointF
        from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath
        from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
        PYSIDE6_AVAILABLE = False
    except ImportError:
        logging.error("Neither PySide6 nor PyQt5 is available - breath visualization will be disabled")
        PYSIDE6_AVAILABLE = False

from src.v7.ui.base_visualizer import BaseVisualizer

# Configure logging
logger = logging.getLogger(__name__)

class BreathCanvas(QWidget):
    """Canvas widget that displays the breath visualization."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 200)
        self.breath_data = []
        self.max_points = 100
        self.amplitude = 0.0
        self.phase = 0.0
        self.frequency = 1.0
        self.is_inhale = False
        self.is_active = False
        self.colors = {
            'background': '#1E1E1E',
            'foreground': '#FFFFFF',
            'breath_line': '#3498DB',
            'inhale': '#2ECC71',
            'exhale': '#E74C3C',
            'inactive': '#7F8C8D'
        }
        
    def set_colors(self, colors: Dict[str, str]) -> None:
        """Update the color palette."""
        self.colors.update(colors)
        self.update()
        
    def update_breath(self, data: Dict[str, Any]) -> None:
        """Update the visualization with new breath data."""
        if 'amplitude' in data:
            self.amplitude = data['amplitude']
        if 'phase' in data:
            self.phase = data['phase']
        if 'frequency' in data:
            self.frequency = data['frequency']
        if 'is_inhale' in data:
            self.is_inhale = data['is_inhale']
        if 'is_active' in data:
            self.is_active = data['is_active']
            
        # Add a new data point based on the current state
        if len(self.breath_data) >= self.max_points:
            self.breath_data.pop(0)
        
        # Calculate the y value using a sine wave modulated by amplitude and phase
        t = len(self.breath_data) / 10.0
        value = self.amplitude * math.sin(2 * math.pi * self.frequency * t + self.phase)
        self.breath_data.append(value)
        
        self.update()
        
    def clear_data(self) -> None:
        """Clear all breath data points."""
        self.breath_data.clear()
        self.update()
        
    def paintEvent(self, event) -> None:
        """Render the breath visualization."""
        if not self.breath_data:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set background
        painter.fillRect(self.rect(), QColor(self.colors['background']))
        
        # If not active, draw a simple inactive indicator
        if not self.is_active:
            painter.setPen(QPen(QColor(self.colors['inactive']), 2))
            painter.drawText(self.rect(), Qt.AlignCenter, "Breath Detection Inactive")
            return
        
        # Calculate scaling factors
        width = self.width()
        height = self.height()
        mid_y = height / 2
        
        # Draw the center line
        painter.setPen(QPen(QColor(self.colors['foreground']).lighter(150), 1, Qt.DashLine))
        painter.drawLine(0, mid_y, width, mid_y)
        
        # Create path for the breath line
        path = QPainterPath()
        x_step = width / (len(self.breath_data) - 1) if len(self.breath_data) > 1 else width
        
        # Start the path at the first point
        path.moveTo(0, mid_y - self.breath_data[0] * (height / 3))
        
        # Add points to the path
        for i, value in enumerate(self.breath_data):
            x = i * x_step
            y = mid_y - value * (height / 3)  # Scale to fit in the widget
            path.lineTo(x, y)
            
        # Draw the breath line with a gradient based on inhale/exhale
        if self.is_inhale:
            pen_color = QColor(self.colors['inhale'])
        else:
            pen_color = QColor(self.colors['exhale'])
            
        painter.setPen(QPen(pen_color, 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPath(path)
        
        # Add a gradient fill under the path
        gradient_path = QPainterPath(path)
        gradient_path.lineTo(width, mid_y)
        gradient_path.lineTo(0, mid_y)
        gradient_path.closeSubpath()
        
        fill_color = pen_color.lighter(150)
        fill_color.setAlpha(50)  # Make it semi-transparent
        painter.fillPath(gradient_path, fill_color)
        
        # Draw the current breath state indicator
        state_text = "Inhaling" if self.is_inhale else "Exhaling"
        painter.setPen(QPen(pen_color, 2))
        painter.drawText(10, 20, state_text)
        
        # Draw amplitude indicator
        amplitude_text = f"Amplitude: {self.amplitude:.2f}"
        painter.drawText(10, height - 10, amplitude_text)
        
        painter.end()


class BreathVisualizer(BaseVisualizer):
    """
    Visualizer for breath patterns in the V7 Node Consciousness system.
    
    This visualizer displays real-time breath data, showing inhalation and
    exhalation patterns and their relationship to consciousness states.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the breath visualizer with configuration settings."""
        super().__init__(config)
        
        self.widget = None
        self.canvas = None
        self.update_timer = None
        
        # Default breath visualization configuration
        default_config = {
            'name': 'Breath Visualizer',
            'description': 'Displays real-time breath patterns and their relationship to consciousness',
            'update_interval': 50,  # ms
            'max_points': 100,
            'show_indicators': True,
            'auto_clear_after': 300  # seconds
        }
        
        # Override defaults with provided config
        if config:
            default_config.update(config)
        self.config = default_config
        
        # Special color palette for breath visualization
        breath_colors = {
            'breath_line': '#3498DB',
            'inhale': '#2ECC71',
            'exhale': '#E74C3C'
        }
        self.colors.update(breath_colors)
        
        # Initialize canvas if Qt is available
        if PYSIDE6_AVAILABLE:
            self._initialize_widget()
        else:
            logger.warning("Qt libraries not available - BreathVisualizer will run in headless mode")
    
    def _initialize_widget(self):
        """Initialize the visualization widget and canvas."""
        if not PYSIDE6_AVAILABLE:
            return
            
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        
        # Title label
        title_label = QLabel(self.get_name())
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"color: {self.colors['foreground']}; font-weight: bold;")
        
        # Create canvas
        self.canvas = BreathCanvas()
        self.canvas.set_colors(self.colors)
        self.canvas.max_points = self.config.get('max_points', 100)
        
        # Add widgets to layout
        layout.addWidget(title_label)
        layout.addWidget(self.canvas)
        
        # Configure widget styling
        self.widget.setStyleSheet(f"background-color: {self.colors['background']}; color: {self.colors['foreground']};")
        
        # Create update timer
        self.update_timer = QTimer(self.widget)
        self.update_timer.timeout.connect(self._animate)
        
    def _animate(self):
        """Update animation based on current data."""
        if not self.active or not self.canvas:
            return
            
        # Get current time for phase calculation
        t = time.time()
        
        # Create simulated breath data when no real data is available
        # This is just for demonstration when no breath sensor is connected
        if not self.data.get('real_data', False):
            phase = (t % 5) / 5 * 2 * math.pi
            is_inhale = phase < math.pi
            amplitude = 0.8 + 0.2 * math.sin(t / 10)
            
            self.data.update({
                'amplitude': amplitude,
                'phase': phase,
                'frequency': 0.2,
                'is_inhale': is_inhale,
                'is_active': self.active
            })
        
        if self.canvas:
            self.canvas.update_breath(self.data)
    
    def update(self, data: Dict[str, Any]) -> None:
        """
        Update the visualizer with new breath data.
        
        Args:
            data: Dictionary containing the breath data to visualize
        """
        self.data.update(data)
        self.last_update_time = time.time()
        
        # Force immediate update if we have a canvas
        if self.canvas:
            self.canvas.update_breath(self.data)
    
    def render(self) -> None:
        """Render the current visualization state."""
        if self.canvas:
            self.canvas.update()
    
    def create_widget(self) -> QWidget:
        """
        Create and return a widget that can be added to a Qt layout.
        
        Returns:
            A Qt widget that displays the breath visualization
        """
        if not PYSIDE6_AVAILABLE:
            logger.warning("Cannot create widget - Qt libraries not available")
            return None
            
        if not self.widget:
            self._initialize_widget()
            
        return self.widget
    
    def start(self) -> None:
        """Start the breath visualizer."""
        super().start()
        if self.update_timer:
            self.update_timer.start(self.config.get('update_interval', 50))
            
        if self.canvas:
            self.canvas.is_active = True
            self.canvas.update()
    
    def stop(self) -> None:
        """Stop the breath visualizer."""
        super().stop()
        if self.update_timer:
            self.update_timer.stop()
            
        if self.canvas:
            self.canvas.is_active = False
            self.canvas.update()
    
    def resize(self, width: int, height: int) -> None:
        """Handle resize events for the visualization."""
        if self.widget:
            self.widget.resize(width, height)
    
    def set_color_palette(self, colors: Dict[str, str]) -> None:
        """Set a custom color palette for the visualizer."""
        super().set_color_palette(colors)
        if self.canvas:
            self.canvas.set_colors(self.colors)
    
    def clear_data(self) -> None:
        """Clear all visualization data."""
        if self.canvas:
            self.canvas.clear_data()
            
        self.data = {}
        
    def handle_breath_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle a breath event from the breath detection system.
        
        Args:
            event_data: Dictionary containing breath event data
        """
        # Mark this as real data (not simulated)
        event_data['real_data'] = True
        
        # Update the visualization
        self.update(event_data) 