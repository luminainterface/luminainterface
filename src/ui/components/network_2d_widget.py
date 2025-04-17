#!/usr/bin/env python3
"""
Network2DWidget

This module implements the 2D network visualization widget.
"""

import logging
from typing import Dict, Any, List, Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush

logger = logging.getLogger(__name__)

class Network2DWidget(QWidget):
    """2D network visualization widget."""
    
    # Signals
    node_clicked = Signal(dict)
    connection_clicked = Signal(dict)
    growth_updated = Signal(str)
    
    def __init__(self, parent=None):
        """Initialize the widget."""
        super().__init__(parent)
        self._config = {}
        self._nodes = []
        self._connections = []
        self._signals = []
        self._growth_stage = "SEED"
        self._animation_timer = QTimer()
        self._animation_timer.timeout.connect(self._update_animation)
        self._animation_timer.start(16)  # ~60 FPS
        
        # Set up widget
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        
    def set_config(self, config: Dict[str, Any]):
        """Set widget configuration."""
        self._config = config
        
    def update_data(self, nodes: List[Dict], connections: List[Dict], signals: List[Dict]):
        """Update visualization data."""
        self._nodes = nodes
        self._connections = connections
        self._signals = signals
        self.update()
        
    def update_growth(self, stage: str, nodes: List[Dict], connections: List[Dict]):
        """Update growth visualization."""
        self._growth_stage = stage
        self._nodes = nodes
        self._connections = connections
        self.growth_updated.emit(stage)
        self.update()
        
    def _update_animation(self):
        """Update animation state."""
        self.update()
        
    def paintEvent(self, event):
        """Paint the widget."""
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Draw background
            self._draw_background(painter)
            
            # Draw connections
            self._draw_connections(painter)
            
            # Draw nodes
            self._draw_nodes(painter)
            
            # Draw signals
            self._draw_signals(painter)
            
        except Exception as e:
            logger.error(f"Error painting widget: {e}")
            
    def _draw_background(self, painter: QPainter):
        """Draw widget background."""
        try:
            # Get background color from config
            bg_color = QColor(self._config.get('background_color', '#1E1E1E'))
            painter.fillRect(self.rect(), bg_color)
            
            # Draw grid if enabled
            if self._config.get('grid_enabled', True):
                self._draw_grid(painter)
                
        except Exception as e:
            logger.error(f"Error drawing background: {e}")
            
    def _draw_grid(self, painter: QPainter):
        """Draw background grid."""
        try:
            grid_color = QColor(50, 50, 50, 50)
            painter.setPen(QPen(grid_color, 1, Qt.DotLine))
            
            # Draw vertical lines
            for x in range(0, self.width(), 20):
                painter.drawLine(x, 0, x, self.height())
                
            # Draw horizontal lines
            for y in range(0, self.height(), 20):
                painter.drawLine(0, y, self.width(), y)
                
        except Exception as e:
            logger.error(f"Error drawing grid: {e}")
            
    def _draw_nodes(self, painter: QPainter):
        """Draw network nodes."""
        try:
            for node in self._nodes:
                # Get node position and size
                x = node.get('x', 0)
                y = node.get('y', 0)
                size = node.get('size', 30)
                
                # Get node color based on type
                node_type = node.get('type', 'normal')
                color = QColor(self._config.get('node_colors', {}).get(node_type, '#3498db'))
                
                # Draw node
                painter.setPen(QPen(color.darker(), 2))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(x - size/2, y - size/2, size, size)
                
                # Draw node label
                if 'label' in node:
                    painter.setPen(QPen(Qt.white))
                    painter.drawText(x - size/2, y - size/2 - 5, size, size,
                                   Qt.AlignCenter, node['label'])
                    
        except Exception as e:
            logger.error(f"Error drawing nodes: {e}")
            
    def _draw_connections(self, painter: QPainter):
        """Draw network connections."""
        try:
            for conn in self._connections:
                # Get connection endpoints
                from_node = next((n for n in self._nodes if n['id'] == conn['from']), None)
                to_node = next((n for n in self._nodes if n['id'] == conn['to']), None)
                
                if from_node and to_node:
                    # Get connection color based on type
                    conn_type = conn.get('type', 'literal')
                    color = QColor(self._config.get('connection_colors', {}).get(conn_type, '#3498db'))
                    
                    # Get connection width
                    width = conn.get('width', 2)
                    
                    # Draw connection
                    painter.setPen(QPen(color, width))
                    painter.drawLine(
                        from_node['x'], from_node['y'],
                        to_node['x'], to_node['y']
                    )
                    
        except Exception as e:
            logger.error(f"Error drawing connections: {e}")
            
    def _draw_signals(self, painter: QPainter):
        """Draw signal animations."""
        try:
            for signal in self._signals:
                # Get signal position and size
                x = signal.get('x', 0)
                y = signal.get('y', 0)
                size = signal.get('size', 10)
                
                # Get signal color
                color = QColor(signal.get('color', '#ffffff'))
                
                # Draw signal
                painter.setPen(QPen(color, 1))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(x - size/2, y - size/2, size, size)
                
        except Exception as e:
            logger.error(f"Error drawing signals: {e}")
            
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        try:
            # Check for node clicks
            for node in self._nodes:
                x = node.get('x', 0)
                y = node.get('y', 0)
                size = node.get('size', 30)
                
                if (abs(event.x() - x) <= size/2 and
                    abs(event.y() - y) <= size/2):
                    self.node_clicked.emit(node)
                    return
                    
            # Check for connection clicks
            for conn in self._connections:
                from_node = next((n for n in self._nodes if n['id'] == conn['from']), None)
                to_node = next((n for n in self._nodes if n['id'] == conn['to']), None)
                
                if from_node and to_node:
                    # Calculate distance from point to line
                    # (This is a simplified version - you might want to use a more
                    # accurate method for line-point distance)
                    if self._is_point_near_line(
                        event.x(), event.y(),
                        from_node['x'], from_node['y'],
                        to_node['x'], to_node['y']
                    ):
                        self.connection_clicked.emit(conn)
                        return
                        
        except Exception as e:
            logger.error(f"Error handling mouse press: {e}")
            
    def _is_point_near_line(self, px: float, py: float, x1: float, y1: float,
                          x2: float, y2: float, threshold: float = 5.0) -> bool:
        """Check if a point is near a line segment."""
        try:
            # Calculate line length
            line_length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            
            # Calculate distance from point to line
            if line_length == 0:
                return False
                
            # Calculate the projection of the point onto the line
            t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_length**2)
            
            # Clamp t to the line segment
            t = max(0, min(1, t))
            
            # Calculate the closest point on the line
            closest_x = x1 + t * (x2 - x1)
            closest_y = y1 + t * (y2 - y1)
            
            # Calculate distance from point to closest point
            distance = ((px - closest_x)**2 + (py - closest_y)**2)**0.5
            
            return distance <= threshold
            
        except Exception as e:
            logger.error(f"Error checking point-line distance: {e}")
            return False
            
    def cleanup(self):
        """Clean up resources."""
        try:
            self._animation_timer.stop()
            self._animation_timer.deleteLater()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}") 