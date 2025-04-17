"""
Visualization System

This module provides a comprehensive visualization system that integrates with
the PySide6 integration layer to provide advanced visualization capabilities.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
import math

from .pyside6_integration import pyside6_integration, PYSIDE6_AVAILABLE
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QRectF, QPointF, QSize
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QPainterPath
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel

# Configure logging
logger = logging.getLogger(__name__)

class VisualizationSystem:
    """Main visualization system class"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._visualizations = {}
        self._active_visualization = None
        self._update_timer = QTimer()
        self._update_timer.setTimerType(Qt.PreciseTimer)
        self._update_timer.timeout.connect(self._update_visualizations)
        self._last_update_time = 0
        self._frame_time = 16  # ~60fps
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize visualization components"""
        if not PYSIDE6_AVAILABLE:
            return
        
        # Create visualization components
        self._visualizations = {
            'network': NetworkVisualization(),
            'growth': GrowthVisualization(),
            'metrics': MetricsVisualization(),
            'system': SystemVisualization()
        }
        
        # Start update timer
        self._update_timer.start(self._frame_time)
    
    def _update_visualizations(self):
        """Update all active visualizations"""
        if not PYSIDE6_AVAILABLE:
            return
        
        current_time = QTimer.elapsed()
        if current_time - self._last_update_time >= self._frame_time:
            self._last_update_time = current_time
            
            for visualization in self._visualizations.values():
                if visualization.is_active():
                    visualization.update()
    
    def get_visualization(self, name: str) -> Optional[QWidget]:
        """Get a visualization by name"""
        return self._visualizations.get(name)
    
    def set_active_visualization(self, name: str):
        """Set the active visualization"""
        if name in self._visualizations:
            if self._active_visualization:
                self._active_visualization.set_active(False)
            self._active_visualization = self._visualizations[name]
            self._active_visualization.set_active(True)
            self._last_update_time = QTimer.elapsed()
    
    def get_active_visualization(self) -> Optional[QWidget]:
        """Get the currently active visualization"""
        return self._active_visualization
    
    def cleanup(self):
        """Clean up resources"""
        self._update_timer.stop()
        for visualization in self._visualizations.values():
            visualization.set_active(False)

class BaseVisualization(QWidget):
    """Base class for all visualizations"""
    
    def __init__(self, title: str = ""):
        super().__init__()
        self._title = title
        self._active = False
        self._data = {}
        self._frame_time = 16  # ~60fps
        self._last_frame_time = 0
        self._initialize_ui()
    
    def _initialize_ui(self):
        """Initialize the UI components"""
        if not PYSIDE6_AVAILABLE:
            return
        
        # Set widget attributes for transparency
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        
        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)
        
        # Create title label
        if self._title:
            title_label = pyside6_integration.get_widget_factory().create_label(
                self._title, style="title"
            )
            layout.addWidget(title_label)
    
    def set_active(self, active: bool):
        """Set the visualization as active/inactive"""
        self._active = active
        if active:
            self.show()
            self._last_frame_time = 0
        else:
            self.hide()
    
    def is_active(self) -> bool:
        """Check if the visualization is active"""
        return self._active
    
    def update(self):
        """Update the visualization"""
        if not self._active:
            return
        
        current_time = QTimer.elapsed()
        if current_time - self._last_frame_time >= self._frame_time:
            self._last_frame_time = current_time
            self._update_data()
            self.repaint()
    
    def _update_data(self):
        """Update visualization data"""
        # To be implemented by subclasses
    
    def paintEvent(self, event):
        """Handle paint events"""
        if not PYSIDE6_AVAILABLE:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.HighQualityAntialiasing)
        
        # Clear with transparent background
        painter.fillRect(self.rect(), Qt.transparent)
        
        # Paint the visualization
        self._paint_visualization(painter)
    
    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        self._handle_resize(event.size())
    
    def _handle_resize(self, size: QSize):
        """Handle resize events"""
        # To be implemented by subclasses
    
    def _paint_visualization(self, painter: QPainter):
        """Paint the visualization"""
        # To be implemented by subclasses

class NetworkVisualization(BaseVisualization):
    """Network visualization component"""
    
    def __init__(self):
        super().__init__("Network Visualization")
        self._nodes = []
        self._connections = []
        self._hovered_node = None
        self._selected_node = None
        self._node_radius = 20
        self._connection_width = 2
        self._initialize_network()
    
    def _initialize_network(self):
        """Initialize the network visualization"""
        self.setMouseTracking(True)
        self._layout_nodes()
    
    def _layout_nodes(self):
        """Layout nodes in a grid pattern"""
        if not self._nodes:
            return
            
        width = self.width()
        height = self.height()
        node_count = len(self._nodes)
        
        # Calculate grid dimensions
        cols = int(math.ceil(math.sqrt(node_count)))
        rows = int(math.ceil(node_count / cols))
        
        # Calculate spacing
        x_spacing = width / (cols + 1)
        y_spacing = height / (rows + 1)
        
        # Position nodes
        for i, node in enumerate(self._nodes):
            col = i % cols
            row = i // cols
            node['position'] = QPointF(
                (col + 1) * x_spacing,
                (row + 1) * y_spacing
            )
    
    def _update_data(self):
        """Update network data"""
        # Update node states
        for node in self._nodes:
            if 'activation' in node:
                node['color'] = self._get_node_color(node['activation'])
            
        # Update connection states
        for conn in self._connections:
            if 'weight' in conn:
                conn['color'] = self._get_connection_color(conn['weight'])
    
    def _get_node_color(self, activation: float) -> QColor:
        """Get color based on node activation"""
        intensity = int(255 * activation)
        return QColor(intensity, intensity, intensity)
    
    def _get_connection_color(self, weight: float) -> QColor:
        """Get color based on connection weight"""
        intensity = int(255 * abs(weight))
        if weight > 0:
            return QColor(0, intensity, 0)  # Green for positive
        else:
            return QColor(intensity, 0, 0)  # Red for negative
    
    def _paint_visualization(self, painter: QPainter):
        """Paint the network visualization"""
        # Draw connections
        for conn in self._connections:
            self._draw_connection(painter, conn)
        
        # Draw nodes
        for node in self._nodes:
            self._draw_node(painter, node)
    
    def _draw_node(self, painter: QPainter, node: dict):
        """Draw a single node"""
        pos = node['position']
        radius = self._node_radius
        
        # Draw node circle
        painter.setPen(QPen(Qt.black, 2))
        painter.setBrush(QBrush(node.get('color', Qt.white)))
        painter.drawEllipse(pos, radius, radius)
        
        # Draw node label if exists
        if 'label' in node:
            painter.drawText(
                QRectF(pos.x() - radius, pos.y() - radius,
                      radius * 2, radius * 2),
                Qt.AlignCenter,
                node['label']
            )
    
    def _draw_connection(self, painter: QPainter, conn: dict):
        """Draw a single connection"""
        start = conn['source']['position']
        end = conn['target']['position']
        
        # Draw connection line
        painter.setPen(QPen(
            conn.get('color', Qt.gray),
            self._connection_width
        ))
        painter.drawLine(start, end)
        
        # Draw weight label if exists
        if 'weight' in conn:
            mid_point = (start + end) / 2
            painter.drawText(
                mid_point,
                f"{conn['weight']:.2f}"
            )
    
    def _handle_resize(self, size: QSize):
        """Handle resize events"""
        self._layout_nodes()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        pos = event.position()
        self._hovered_node = self._find_node_at(pos)
        self.update()
    
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.LeftButton:
            pos = event.position()
            self._selected_node = self._find_node_at(pos)
            self.update()
    
    def _find_node_at(self, pos: QPointF) -> Optional[dict]:
        """Find node at given position"""
        for node in self._nodes:
            if (node['position'] - pos).manhattanLength() <= self._node_radius:
                return node
        return None
    
    def add_node(self, node_id: str, activation: float = 0.0, label: str = ""):
        """Add a new node to the visualization"""
        self._nodes.append({
            'id': node_id,
            'activation': activation,
            'label': label,
            'position': QPointF(0, 0),
            'color': self._get_node_color(activation)
        })
        self._layout_nodes()
    
    def add_connection(self, source_id: str, target_id: str, weight: float = 0.0):
        """Add a new connection to the visualization"""
        source = next((n for n in self._nodes if n['id'] == source_id), None)
        target = next((n for n in self._nodes if n['id'] == target_id), None)
        
        if source and target:
            self._connections.append({
                'source': source,
                'target': target,
                'weight': weight,
                'color': self._get_connection_color(weight)
            })
    
    def update_node(self, node_id: str, activation: float):
        """Update node activation"""
        node = next((n for n in self._nodes if n['id'] == node_id), None)
        if node:
            node['activation'] = activation
            node['color'] = self._get_node_color(activation)
    
    def update_connection(self, source_id: str, target_id: str, weight: float):
        """Update connection weight"""
        conn = next((c for c in self._connections 
                    if c['source']['id'] == source_id 
                    and c['target']['id'] == target_id), None)
        if conn:
            conn['weight'] = weight
            conn['color'] = self._get_connection_color(weight)
    
    def clear(self):
        """Clear all nodes and connections"""
        self._nodes.clear()
        self._connections.clear()
        self._hovered_node = None
        self._selected_node = None
        self.update()

class GrowthVisualization(BaseVisualization):
    """Growth visualization component"""
    
    def __init__(self):
        super().__init__("Growth Visualization")
        self._growth_stages = []
        self._current_stage = 0
        self._stage_progress = 0.0
        self._animation_time = 0.0
        self._initialize_growth()
    
    def _initialize_growth(self):
        """Initialize the growth visualization"""
        self._growth_stages = [
            {'name': 'Seed', 'color': QColor(139, 69, 19), 'duration': 2.0},
            {'name': 'Sprout', 'color': QColor(34, 139, 34), 'duration': 3.0},
            {'name': 'Branch', 'color': QColor(85, 107, 47), 'duration': 4.0},
            {'name': 'Leaf', 'color': QColor(50, 205, 50), 'duration': 3.0},
            {'name': 'Flower', 'color': QColor(255, 105, 180), 'duration': 2.0},
            {'name': 'Fruit', 'color': QColor(255, 165, 0), 'duration': 3.0}
        ]
    
    def _update_data(self):
        """Update growth data"""
        if not self._growth_stages:
            return
            
        # Update animation time
        self._animation_time += self._frame_time / 1000.0  # Convert to seconds
        
        # Update stage progress
        current_stage = self._growth_stages[self._current_stage]
        self._stage_progress = min(1.0, self._animation_time / current_stage['duration'])
        
        # Check for stage completion
        if self._stage_progress >= 1.0:
            if self._current_stage < len(self._growth_stages) - 1:
                self._current_stage += 1
                self._animation_time = 0.0
                self._stage_progress = 0.0
    
    def _paint_visualization(self, painter: QPainter):
        """Paint the growth visualization"""
        if not self._growth_stages:
            return
            
        # Get current stage
        current_stage = self._growth_stages[self._current_stage]
        
        # Draw growth visualization based on current stage
        if current_stage['name'] == 'Seed':
            self._draw_seed(painter)
        elif current_stage['name'] == 'Sprout':
            self._draw_sprout(painter)
        elif current_stage['name'] == 'Branch':
            self._draw_branch(painter)
        elif current_stage['name'] == 'Leaf':
            self._draw_leaf(painter)
        elif current_stage['name'] == 'Flower':
            self._draw_flower(painter)
        elif current_stage['name'] == 'Fruit':
            self._draw_fruit(painter)
    
    def _draw_seed(self, painter: QPainter):
        """Draw seed stage"""
        center = QPointF(self.width() / 2, self.height() / 2)
        radius = 20 * self._stage_progress
        
        painter.setPen(QPen(Qt.black, 2))
        painter.setBrush(QBrush(self._growth_stages[0]['color']))
        painter.drawEllipse(center, radius, radius)
    
    def _draw_sprout(self, painter: QPainter):
        """Draw sprout stage"""
        center = QPointF(self.width() / 2, self.height() / 2)
        height = 100 * self._stage_progress
        
        # Draw stem
        painter.setPen(QPen(Qt.black, 2))
        painter.setBrush(QBrush(self._growth_stages[1]['color']))
        painter.drawLine(
            center,
            QPointF(center.x(), center.y() - height)
        )
        
        # Draw leaves
        if self._stage_progress > 0.5:
            leaf_size = 20 * (self._stage_progress - 0.5) * 2
            painter.drawEllipse(
                QPointF(center.x() - 30, center.y() - height + 20),
                leaf_size, leaf_size
            )
            painter.drawEllipse(
                QPointF(center.x() + 30, center.y() - height + 20),
                leaf_size, leaf_size
            )
    
    def _draw_branch(self, painter: QPainter):
        """Draw branch stage"""
        center = QPointF(self.width() / 2, self.height() / 2)
        height = 150 * self._stage_progress
        
        # Draw main stem
        painter.setPen(QPen(Qt.black, 3))
        painter.setBrush(QBrush(self._growth_stages[2]['color']))
        painter.drawLine(
            center,
            QPointF(center.x(), center.y() - height)
        )
        
        # Draw branches
        if self._stage_progress > 0.3:
            branch_length = 50 * (self._stage_progress - 0.3) * (10/7)
            for angle in [-30, 30]:
                rad = math.radians(angle)
                end_x = center.x() + branch_length * math.sin(rad)
                end_y = center.y() - height + branch_length * math.cos(rad)
                painter.drawLine(
                    QPointF(center.x(), center.y() - height/2),
                    QPointF(end_x, end_y)
                )
    
    def _draw_leaf(self, painter: QPainter):
        """Draw leaf stage"""
        center = QPointF(self.width() / 2, self.height() / 2)
        height = 200 * self._stage_progress
        
        # Draw stem and branches
        self._draw_branch(painter)
        
        # Draw leaves
        if self._stage_progress > 0.4:
            leaf_size = 30 * (self._stage_progress - 0.4) * (10/6)
            for angle in [-45, -15, 15, 45]:
                rad = math.radians(angle)
                pos_x = center.x() + 100 * math.sin(rad)
                pos_y = center.y() - height + 100 * math.cos(rad)
                painter.setBrush(QBrush(self._growth_stages[3]['color']))
                painter.drawEllipse(
                    QPointF(pos_x, pos_y),
                    leaf_size, leaf_size
                )
    
    def _draw_flower(self, painter: QPainter):
        """Draw flower stage"""
        center = QPointF(self.width() / 2, self.height() / 2)
        height = 250 * self._stage_progress
        
        # Draw stem and leaves
        self._draw_leaf(painter)
        
        # Draw flower
        if self._stage_progress > 0.5:
            flower_size = 40 * (self._stage_progress - 0.5) * 2
            painter.setBrush(QBrush(self._growth_stages[4]['color']))
            for angle in range(0, 360, 45):
                rad = math.radians(angle)
                pos_x = center.x() + flower_size * math.sin(rad)
                pos_y = center.y() - height + flower_size * math.cos(rad)
                painter.drawEllipse(
                    QPointF(pos_x, pos_y),
                    flower_size/2, flower_size/2
                )
    
    def _draw_fruit(self, painter: QPainter):
        """Draw fruit stage"""
        center = QPointF(self.width() / 2, self.height() / 2)
        height = 300 * self._stage_progress
        
        # Draw stem, leaves, and flower
        self._draw_flower(painter)
        
        # Draw fruit
        if self._stage_progress > 0.6:
            fruit_size = 50 * (self._stage_progress - 0.6) * (10/4)
            painter.setBrush(QBrush(self._growth_stages[5]['color']))
            for angle in [-30, 0, 30]:
                rad = math.radians(angle)
                pos_x = center.x() + 150 * math.sin(rad)
                pos_y = center.y() - height + 150 * math.cos(rad)
                painter.drawEllipse(
                    QPointF(pos_x, pos_y),
                    fruit_size, fruit_size
                )
    
    def reset_growth(self):
        """Reset growth to initial stage"""
        self._current_stage = 0
        self._stage_progress = 0.0
        self._animation_time = 0.0
        self.update()
    
    def get_current_stage(self) -> str:
        """Get current growth stage name"""
        if self._growth_stages:
            return self._growth_stages[self._current_stage]['name']
        return ""
    
    def get_stage_progress(self) -> float:
        """Get current stage progress (0.0 to 1.0)"""
        return self._stage_progress

class MetricsVisualization(BaseVisualization):
    """Metrics visualization component"""
    
    def __init__(self):
        super().__init__("Metrics Visualization")
        self._metrics = {}
        self._history = {}
        self._max_history = 100
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize the metrics visualization"""
        # Initialize metric history
        self._history = {
            'health': [],
            'stability': [],
            'energy': [],
            'consciousness': []
        }
        
        # Initialize current metrics
        self._metrics = {
            'health': 0.0,
            'stability': 0.0,
            'energy': 0.0,
            'consciousness': 0.0,
            'gate_states': {},
            'system_status': 'Initializing'
        }
    
    def _update_data(self):
        """Update metrics data"""
        # Update history
        for metric in ['health', 'stability', 'energy', 'consciousness']:
            self._history[metric].append(self._metrics[metric])
            if len(self._history[metric]) > self._max_history:
                self._history[metric].pop(0)
    
    def _paint_visualization(self, painter: QPainter):
        """Paint the metrics visualization"""
        # Draw background
        painter.fillRect(self.rect(), QColor(240, 240, 240, 200))
        
        # Draw metrics
        self._draw_metric_charts(painter)
        self._draw_current_metrics(painter)
        self._draw_gate_states(painter)
        self._draw_system_status(painter)
    
    def _draw_metric_charts(self, painter: QPainter):
        """Draw metric history charts"""
        width = self.width()
        height = self.height() // 2
        margin = 20
        
        # Draw chart area
        chart_rect = QRectF(margin, margin, width - 2*margin, height - margin)
        painter.setPen(QPen(Qt.black, 1))
        painter.drawRect(chart_rect)
        
        # Draw metrics
        metrics = ['health', 'stability', 'energy', 'consciousness']
        colors = [Qt.red, Qt.blue, Qt.green, Qt.magenta]
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            history = self._history[metric]
            if not history:
                continue
                
            # Draw line
            painter.setPen(QPen(color, 2))
            path = QPainterPath()
            
            for j, value in enumerate(history):
                x = chart_rect.left() + (j / len(history)) * chart_rect.width()
                y = chart_rect.bottom() - value * chart_rect.height()
                
                if j == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            
            painter.drawPath(path)
            
            # Draw label
            painter.setPen(QPen(color, 1))
            painter.drawText(
                QRectF(chart_rect.left(), chart_rect.top() + i*20,
                      chart_rect.width(), 20),
                Qt.AlignLeft,
                f"{metric}: {self._metrics[metric]:.2f}"
            )
    
    def _draw_current_metrics(self, painter: QPainter):
        """Draw current metric values"""
        width = self.width()
        height = self.height() // 2
        margin = 20
        
        # Draw metric values
        metrics = ['health', 'stability', 'energy', 'consciousness']
        colors = [Qt.red, Qt.blue, Qt.green, Qt.magenta]
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            value = self._metrics[metric]
            
            # Draw bar
            bar_width = width - 2*margin
            bar_height = 20
            bar_x = margin
            bar_y = height + margin + i*(bar_height + 10)
            
            painter.setPen(QPen(Qt.black, 1))
            painter.setBrush(QBrush(color))
            painter.drawRect(
                bar_x, bar_y,
                value * bar_width, bar_height
            )
            
            # Draw label
            painter.setPen(QPen(Qt.black, 1))
            painter.drawText(
                QRectF(bar_x, bar_y, bar_width, bar_height),
                Qt.AlignLeft | Qt.AlignVCenter,
                f"{metric}: {value:.2f}"
            )
    
    def _draw_gate_states(self, painter: QPainter):
        """Draw gate states"""
        width = self.width()
        height = self.height()
        margin = 20
        
        # Draw gate states
        gate_states = self._metrics.get('gate_states', {})
        if not gate_states:
            return
            
        # Calculate layout
        gate_count = len(gate_states)
        gate_width = (width - 2*margin) / gate_count
        gate_height = 30
        
        for i, (gate_id, state) in enumerate(gate_states.items()):
            x = margin + i*gate_width
            y = height - margin - gate_height
            
            # Draw gate state
            color = Qt.green if state else Qt.red
            painter.setPen(QPen(Qt.black, 1))
            painter.setBrush(QBrush(color))
            painter.drawRect(x, y, gate_width - 5, gate_height)
            
            # Draw label
            painter.setPen(QPen(Qt.black, 1))
            painter.drawText(
                QRectF(x, y, gate_width - 5, gate_height),
                Qt.AlignCenter,
                gate_id
            )
    
    def _draw_system_status(self, painter: QPainter):
        """Draw system status"""
        status = self._metrics.get('system_status', 'Unknown')
        
        # Draw status
        painter.setPen(QPen(Qt.black, 1))
        painter.drawText(
            QRectF(0, 0, self.width(), 20),
            Qt.AlignCenter,
            f"System Status: {status}"
        )
    
    def update_metric(self, metric: str, value: float):
        """Update a metric value"""
        if metric in self._metrics:
            self._metrics[metric] = value
            self.update()
    
    def update_gate_state(self, gate_id: str, state: bool):
        """Update a gate state"""
        self._metrics['gate_states'][gate_id] = state
        self.update()
    
    def update_system_status(self, status: str):
        """Update system status"""
        self._metrics['system_status'] = status
        self.update()
    
    def get_metric(self, metric: str) -> float:
        """Get a metric value"""
        return self._metrics.get(metric, 0.0)
    
    def get_gate_state(self, gate_id: str) -> bool:
        """Get a gate state"""
        return self._metrics['gate_states'].get(gate_id, False)
    
    def get_system_status(self) -> str:
        """Get system status"""
        return self._metrics.get('system_status', 'Unknown')

class SystemVisualization(BaseVisualization):
    """System visualization component"""
    
    def __init__(self):
        super().__init__("System Visualization")
        self._system_state = {}
        self._components = {}
        self._connections = []
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the system visualization"""
        # Initialize system state
        self._system_state = {
            'status': 'Initializing',
            'health': 0.0,
            'stability': 0.0,
            'energy': 0.0,
            'consciousness': 0.0,
            'components': {},
            'connections': []
        }
        
        # Initialize components
        self._components = {
            'backend': {'type': 'Backend', 'status': 'Offline'},
            'network': {'type': 'Network', 'status': 'Offline'},
            'growth': {'type': 'Growth', 'status': 'Offline'},
            'metrics': {'type': 'Metrics', 'status': 'Offline'}
        }
    
    def _update_data(self):
        """Update system data"""
        # Update component states
        for component_id, component in self._components.items():
            if component_id in self._system_state['components']:
                component['status'] = self._system_state['components'][component_id]
    
    def _paint_visualization(self, painter: QPainter):
        """Paint the system visualization"""
        # Draw background
        painter.fillRect(self.rect(), QColor(240, 240, 240, 200))
        
        # Draw system overview
        self._draw_system_overview(painter)
        
        # Draw components
        self._draw_components(painter)
        
        # Draw connections
        self._draw_connections(painter)
    
    def _draw_system_overview(self, painter: QPainter):
        """Draw system overview"""
        width = self.width()
        height = self.height()
        margin = 20
        
        # Draw system status
        status = self._system_state['status']
        painter.setPen(QPen(Qt.black, 2))
        painter.drawText(
            QRectF(margin, margin, width - 2*margin, 30),
            Qt.AlignCenter,
            f"System Status: {status}"
        )
        
        # Draw system metrics
        metrics = ['health', 'stability', 'energy', 'consciousness']
        colors = [Qt.red, Qt.blue, Qt.green, Qt.magenta]
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            value = self._system_state[metric]
            
            # Draw metric bar
            bar_width = width - 2*margin
            bar_height = 20
            bar_x = margin
            bar_y = margin + 40 + i*(bar_height + 10)
            
            painter.setPen(QPen(Qt.black, 1))
            painter.setBrush(QBrush(color))
            painter.drawRect(
                bar_x, bar_y,
                value * bar_width, bar_height
            )
            
            # Draw label
            painter.setPen(QPen(Qt.black, 1))
            painter.drawText(
                QRectF(bar_x, bar_y, bar_width, bar_height),
                Qt.AlignLeft | Qt.AlignVCenter,
                f"{metric}: {value:.2f}"
            )
    
    def _draw_components(self, painter: QPainter):
        """Draw system components"""
        width = self.width()
        height = self.height()
        margin = 20
        
        # Calculate component positions
        component_width = 100
        component_height = 60
        spacing = 50
        
        # Draw components
        for i, (component_id, component) in enumerate(self._components.items()):
            x = margin + i*(component_width + spacing)
            y = height - margin - component_height
            
            # Draw component box
            status_color = Qt.green if component['status'] == 'Online' else Qt.red
            painter.setPen(QPen(Qt.black, 2))
            painter.setBrush(QBrush(status_color))
            painter.drawRect(x, y, component_width, component_height)
            
            # Draw component label
            painter.setPen(QPen(Qt.black, 1))
            painter.drawText(
                QRectF(x, y, component_width, component_height),
                Qt.AlignCenter,
                f"{component['type']}\n{component['status']}"
            )
    
    def _draw_connections(self, painter: QPainter):
        """Draw component connections"""
        width = self.width()
        height = self.height()
        margin = 20
        
        # Draw connections
        for connection in self._connections:
            source = connection['source']
            target = connection['target']
            status = connection['status']
            
            # Get component positions
            source_x = margin + list(self._components.keys()).index(source) * (100 + 50) + 50
            source_y = height - margin - 30
            target_x = margin + list(self._components.keys()).index(target) * (100 + 50) + 50
            target_y = height - margin - 30
            
            # Draw connection line
            color = Qt.green if status == 'Active' else Qt.red
            painter.setPen(QPen(color, 2))
            painter.drawLine(source_x, source_y, target_x, target_y)
    
    def update_component_status(self, component_id: str, status: str):
        """Update component status"""
        if component_id in self._components:
            self._components[component_id]['status'] = status
            self._system_state['components'][component_id] = status
            self.update()
    
    def update_connection_status(self, source: str, target: str, status: str):
        """Update connection status"""
        connection = next((c for c in self._connections 
                         if c['source'] == source and c['target'] == target), None)
        
        if connection:
            connection['status'] = status
        else:
            self._connections.append({
                'source': source,
                'target': target,
                'status': status
            })
        
        self.update()
    
    def update_system_status(self, status: str):
        """Update system status"""
        self._system_state['status'] = status
        self.update()
    
    def update_system_metric(self, metric: str, value: float):
        """Update system metric"""
        if metric in self._system_state:
            self._system_state[metric] = value
            self.update()
    
    def get_component_status(self, component_id: str) -> str:
        """Get component status"""
        return self._components.get(component_id, {}).get('status', 'Unknown')
    
    def get_connection_status(self, source: str, target: str) -> str:
        """Get connection status"""
        connection = next((c for c in self._connections 
                         if c['source'] == source and c['target'] == target), None)
        return connection['status'] if connection else 'Unknown'
    
    def get_system_status(self) -> str:
        """Get system status"""
        return self._system_state.get('status', 'Unknown')
    
    def get_system_metric(self, metric: str) -> float:
        """Get system metric"""
        return self._system_state.get(metric, 0.0)

# Create singleton instance
visualization_system = VisualizationSystem() 