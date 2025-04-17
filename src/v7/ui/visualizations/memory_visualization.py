#!/usr/bin/env python3
"""
Memory visualization component for the V7 system.
This visualization displays memory operations, strengths, and statistics.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

try:
    # Try to import from PySide6 first
    from PySide6.QtCore import Qt, QTimer, Signal, Slot, QRectF, QSize, QPointF, Property
    from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QPainterPath, QFont, QFontMetrics, QLinearGradient
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGraphicsView, QGraphicsScene, QGraphicsItem
    from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
    PYSIDE6_AVAILABLE = True
except ImportError:
    try:
        # Fall back to PyQt5
        from PyQt5.QtCore import Qt, QTimer, pyqtSignal as Signal, pyqtSlot as Slot, QRectF, QSize, QPointF
        from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QPainterPath, QFont, QFontMetrics, QLinearGradient
        from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGraphicsView, QGraphicsScene, QGraphicsItem
        from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis
        PYSIDE6_AVAILABLE = False
    except ImportError:
        # If neither is available, create stub classes for type hints
        class QWidget:
            pass
        class QGraphicsItem:
            pass
        PYSIDE6_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

class MemoryNode(QGraphicsItem):
    """Represents a single memory node in the visualization."""
    
    def __init__(self, memory_data: Dict[str, Any], parent=None):
        """Initialize a memory node visualization item."""
        super().__init__(parent)
        self.memory_data = memory_data
        self.radius = 30
        self.pulse_animation = 0
        self.pulse_direction = 1
        self.pulse_speed = 0.05
        self.hover = False
        self.setAcceptHoverEvents(True)
        
        # Set position based on memory type and timestamp
        memory_type = memory_data.get('memory_type', 'generic')
        timestamp = memory_data.get('timestamp', time.time())
        self.setPos(((hash(memory_type) % 100) * 10), (timestamp % 100) * 10)
        
        # Set Z value based on strength for proper layering
        strength = memory_data.get('strength', 0.5)
        self.setZValue(strength * 10)
        
        # Start animation based on operation type
        self.animation_type = memory_data.get('animation', 'none')
        self.animation_progress = 0
        self.animation_duration = memory_data.get('duration', 1000) / 1000  # Convert to seconds
        
        # Start a timer for animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)  # ~60fps
        
    def boundingRect(self) -> QRectF:
        """Return the bounding rectangle of this item."""
        extra_space = 20 if self.hover else 10
        return QRectF(-self.radius - extra_space, -self.radius - extra_space, 
                      (self.radius + extra_space) * 2, (self.radius + extra_space) * 2)
    
    def shape(self) -> QPainterPath:
        """Return the shape of this item for collision detection."""
        path = QPainterPath()
        path.addEllipse(-self.radius, -self.radius, self.radius * 2, self.radius * 2)
        return path
    
    def paint(self, painter: QPainter, option, widget=None):
        """Paint the memory node."""
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get colors from memory data
        primary_color = QColor(self.memory_data.get('primary_color', '#607D8B'))
        secondary_color = QColor(self.memory_data.get('secondary_color', '#455A64'))
        text_color = QColor(self.memory_data.get('text_color', '#FFFFFF'))
        
        # Adjust opacity based on strength and animation
        strength = self.memory_data.get('strength', 0.5)
        painter.setOpacity(min(1.0, strength + 0.2))
        
        # Draw the pulse effect if pulsing
        if self.animation_type == 'pulse' or self.hover:
            pulse_radius = self.radius + (self.pulse_animation * 15)
            pulse_pen = QPen(primary_color)
            pulse_pen.setWidth(2)
            pulse_pen.setStyle(Qt.DashLine)
            painter.setPen(pulse_pen)
            painter.setOpacity(0.3 - (self.pulse_animation * 0.3))
            painter.drawEllipse(-pulse_radius, -pulse_radius, pulse_radius * 2, pulse_radius * 2)
            painter.setOpacity(min(1.0, strength + 0.2))
        
        # Draw the main circle with gradient
        gradient = QLinearGradient(0, -self.radius, 0, self.radius)
        gradient.setColorAt(0, primary_color)
        gradient.setColorAt(1, secondary_color)
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        
        # Adjust radius based on animation
        current_radius = self.radius
        if self.animation_type == 'grow':
            current_radius = self.radius * (0.2 + (self.animation_progress * 0.8))
        elif self.animation_type == 'fade':
            painter.setOpacity(min(1.0, strength) * (1 - self.animation_progress))
        
        painter.drawEllipse(-current_radius, -current_radius, current_radius * 2, current_radius * 2)
        
        # Draw memory type indicator
        painter.setPen(QPen(text_color))
        painter.setFont(QFont("Arial", 8))
        
        memory_type = self.memory_data.get('memory_type', 'generic')
        type_initial = memory_type[0].upper() if memory_type else '?'
        
        metrics = QFontMetrics(painter.font())
        text_width = metrics.horizontalAdvance(type_initial)
        text_height = metrics.height()
        
        painter.drawText(-text_width/2, text_height/4, type_initial)
        
        # Draw memory strength indicator as a small circle
        painter.setBrush(QBrush(text_color))
        strength_radius = current_radius * 0.15
        strength_x = current_radius * 0.6
        strength_y = 0
        painter.drawEllipse(QRectF(strength_x - strength_radius, 
                                   strength_y - strength_radius,
                                   strength_radius * 2, 
                                   strength_radius * 2))
    
    def update_animation(self):
        """Update the animation state."""
        # Update pulse animation
        self.pulse_animation += self.pulse_direction * self.pulse_speed
        if self.pulse_animation >= 1.0:
            self.pulse_animation = 1.0
            self.pulse_direction = -1
        elif self.pulse_animation <= 0.0:
            self.pulse_animation = 0.0
            self.pulse_direction = 1
        
        # Update operation-specific animation
        if self.animation_type != 'none':
            self.animation_progress += 0.016 / self.animation_duration  # 16ms / duration
            if self.animation_progress >= 1.0:
                self.animation_progress = 1.0
                if self.animation_type == 'fade':
                    # Remove self when fade animation completes
                    scene = self.scene()
                    if scene:
                        scene.removeItem(self)
                        self.timer.stop()
                        return
                # Stop animation
                self.animation_type = 'none'
        
        self.update()
    
    def hoverEnterEvent(self, event):
        """Handle hover enter events."""
        self.hover = True
        self.update()
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        """Handle hover leave events."""
        self.hover = False
        self.update()
        super().hoverLeaveEvent(event)

class MemoryStrengthChart(QChartView):
    """Chart showing memory strength over time."""
    
    def __init__(self, parent=None):
        """Initialize the memory strength chart."""
        super().__init__(parent)
        self.chart = QChart()
        self.chart.setTitle("Memory Strength")
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        self.chart.legend().setVisible(True)
        
        # Create series for different memory types
        self.series = {}
        for memory_type, color in [
            ('experience', '#8BC34A'),
            ('conversation', '#2196F3'),
            ('emotional', '#FF9800'),
            ('contradiction', '#F44336'),
            ('insight', '#9C27B0'),
            ('generic', '#607D8B')
        ]:
            series = QLineSeries()
            series.setName(memory_type.capitalize())
            series.setPen(QPen(QColor(color), 2))
            self.chart.addSeries(series)
            self.series[memory_type] = series
        
        # Create axes
        self.axisX = QValueAxis()
        self.axisX.setTitleText("Time (s)")
        self.axisX.setRange(0, 60)
        
        self.axisY = QValueAxis()
        self.axisY.setTitleText("Strength")
        self.axisY.setRange(0, 1)
        
        self.chart.addAxis(self.axisX, Qt.AlignBottom)
        self.chart.addAxis(self.axisY, Qt.AlignLeft)
        
        for series in self.series.values():
            series.attachAxis(self.axisX)
            series.attachAxis(self.axisY)
        
        self.setChart(self.chart)
        self.setRenderHint(QPainter.Antialiasing)
        
        # Track start time for relative timestamps
        self.start_time = time.time()
        self.max_points = 100  # Maximum points to show
        
        # Last points for each series
        self.last_points = {memory_type: {} for memory_type in self.series.keys()}
    
    def add_memory_data(self, memory_data: Dict[str, Any]):
        """Add new memory data to the chart."""
        memory_type = memory_data.get('memory_type', 'generic')
        strength = memory_data.get('strength', 0.5)
        memory_id = memory_data.get('memory_id', 'unknown')
        
        # Calculate X value (time since start)
        current_time = time.time()
        x_value = current_time - self.start_time
        
        # If this type isn't tracked, use generic
        if memory_type not in self.series:
            memory_type = 'generic'
        
        # Add the point to the series
        self.series[memory_type].append(x_value, strength)
        self.last_points[memory_type][memory_id] = (x_value, strength)
        
        # Remove old points if we have too many
        if self.series[memory_type].count() > self.max_points:
            # Create a new series and copy the recent points
            new_series = QLineSeries()
            new_series.setName(self.series[memory_type].name())
            new_series.setPen(self.series[memory_type].pen())
            
            # Get the most recent points
            points = []
            for i in range(max(0, self.series[memory_type].count() - self.max_points), 
                         self.series[memory_type].count()):
                point = self.series[memory_type].at(i)
                points.append((point.x(), point.y()))
            
            # Add points to the new series
            for x, y in points:
                new_series.append(x, y)
            
            # Replace the old series
            self.chart.removeSeries(self.series[memory_type])
            self.chart.addSeries(new_series)
            new_series.attachAxis(self.axisX)
            new_series.attachAxis(self.axisY)
            self.series[memory_type] = new_series
        
        # Adjust X axis if needed
        if x_value > self.axisX.max():
            self.axisX.setRange(max(0, x_value - 60), x_value + 5)

class MemoryStatsPanel(QWidget):
    """Panel displaying memory statistics."""
    
    def __init__(self, parent=None):
        """Initialize the memory stats panel."""
        super().__init__(parent)
        
        self.layout = QVBoxLayout(self)
        
        # Title
        self.title_label = QLabel("Memory Statistics")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.layout.addWidget(self.title_label)
        
        # Stats layout
        self.stats_layout = QVBoxLayout()
        
        # Create labels for various stats
        self.stats_labels = {
            'total_memories': QLabel("Total Memories: 0"),
            'active_memories': QLabel("Active Memories: 0"),
            'decay_enabled': QLabel("Memory Decay: Disabled"),
            'store_type': QLabel("Storage Type: Unknown"),
            'avg_strength': QLabel("Average Strength: 0.0"),
            'recent_ops': QLabel("Recent Operations: 0"),
        }
        
        for label in self.stats_labels.values():
            self.stats_layout.addWidget(label)
        
        self.layout.addLayout(self.stats_layout)
        self.layout.addStretch(1)
        
        # Operation counters
        self.operation_counts = {
            'store': 0,
            'retrieve': 0,
            'update': 0,
            'decay': 0,
            'delete': 0
        }
        
        # Operation timestamps for calculating recent operations
        self.operation_timestamps = []
    
    def update_stats(self, stats_data: Dict[str, Any]):
        """Update the displayed statistics."""
        # Update basic stats
        if 'total_memories' in stats_data:
            self.stats_labels['total_memories'].setText(f"Total Memories: {stats_data['total_memories']}")
        
        if 'memory_stats' in stats_data:
            memory_stats = stats_data['memory_stats']
            
            if 'memories_stored' in memory_stats:
                self.stats_labels['total_memories'].setText(f"Total Memories: {memory_stats['memories_stored']}")
            
            if 'active_memories' in memory_stats:
                self.stats_labels['active_memories'].setText(f"Active Memories: {memory_stats['active_memories']}")
            
            if 'average_strength' in memory_stats:
                avg_strength = memory_stats['average_strength']
                self.stats_labels['avg_strength'].setText(f"Average Strength: {avg_strength:.2f}")
        
        # Update decay status
        if 'decay_enabled' in stats_data:
            status = "Enabled" if stats_data['decay_enabled'] else "Disabled"
            self.stats_labels['decay_enabled'].setText(f"Memory Decay: {status}")
        
        # Update storage type
        if 'store_type' in stats_data:
            self.stats_labels['store_type'].setText(f"Storage Type: {stats_data['store_type']}")
    
    def record_operation(self, operation: str):
        """Record a memory operation and update stats."""
        if operation in self.operation_counts:
            self.operation_counts[operation] += 1
        
        # Record timestamp for recent operations calculation
        current_time = time.time()
        self.operation_timestamps.append(current_time)
        
        # Remove timestamps older than 60 seconds
        self.operation_timestamps = [ts for ts in self.operation_timestamps 
                                    if current_time - ts <= 60]
        
        # Update the recent operations count
        recent_count = len(self.operation_timestamps)
        self.stats_labels['recent_ops'].setText(f"Recent Operations: {recent_count}/min")
        
        # Create summary text of operations
        ops_text = ", ".join([f"{op}: {count}" for op, count in self.operation_counts.items()])
        tooltip = f"Operation counts: {ops_text}"
        self.stats_labels['recent_ops'].setToolTip(tooltip)

class MemoryVisualization(QWidget):
    """Main memory visualization widget."""
    
    def __init__(self, parent=None, visualization_connector=None):
        """Initialize the memory visualization widget."""
        super().__init__(parent)
        
        # Set up the layout
        self.layout = QVBoxLayout(self)
        
        # Title and info
        self.title_label = QLabel("V7 Memory Visualization")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.layout.addWidget(self.title_label)
        
        # Main content area with memory nodes visualization and stats
        self.content_layout = QHBoxLayout()
        
        # Left side - memory nodes visualization
        self.memory_scene = QGraphicsScene(self)
        self.memory_view = QGraphicsView(self.memory_scene)
        self.memory_view.setRenderHint(QPainter.Antialiasing)
        self.memory_view.setMinimumHeight(300)
        self.memory_view.setStyleSheet("background-color: #2d2d2d;")
        
        # Right side - memory stats
        self.stats_panel = MemoryStatsPanel()
        self.stats_panel.setMaximumWidth(250)
        
        self.content_layout.addWidget(self.memory_view, 7)
        self.content_layout.addWidget(self.stats_panel, 3)
        self.layout.addLayout(self.content_layout)
        
        # Bottom chart
        self.strength_chart = MemoryStrengthChart()
        self.layout.addWidget(self.strength_chart)
        
        # Set up connection to visualization connector
        self.visualization_connector = visualization_connector
        if self.visualization_connector:
            self.register_with_connector()
        
        # Maximum number of nodes to display
        self.max_nodes = 50
        self.memory_nodes = {}
    
    def register_with_connector(self):
        """Register this widget with the visualization connector."""
        if not self.visualization_connector:
            logger.warning("No visualization connector provided")
            return
        
        logger.info("Registering memory visualization with connector")
        self.visualization_connector.register_memory_visualizer(self.handle_memory_event)
    
    def handle_memory_event(self, event_data: Dict[str, Any]):
        """Handle memory events from the visualization connector."""
        event_type = event_data.get('type', 'unknown')
        
        if event_type == 'memory':
            # Handle individual memory event
            self.handle_memory_operation(event_data)
        elif event_type == 'memory_summary':
            # Handle memory statistics update
            self.stats_panel.update_stats(event_data)
    
    def handle_memory_operation(self, event_data: Dict[str, Any]):
        """Handle a memory operation event."""
        operation = event_data.get('operation', 'unknown')
        memory_id = event_data.get('memory_id', 'unknown')
        
        # Record the operation in stats
        self.stats_panel.record_operation(operation)
        
        # Update the strength chart
        self.strength_chart.add_memory_data(event_data)
        
        # Handle based on operation type
        if operation == 'store':
            # Create a new memory node
            memory_node = MemoryNode(event_data)
            self.memory_scene.addItem(memory_node)
            self.memory_nodes[memory_id] = memory_node
            
            # Remove old nodes if we have too many
            if len(self.memory_nodes) > self.max_nodes:
                oldest_id = next(iter(self.memory_nodes))
                oldest_node = self.memory_nodes.pop(oldest_id)
                self.memory_scene.removeItem(oldest_node)
                
        elif operation == 'retrieve':
            # Highlight the memory node if it exists
            if memory_id in self.memory_nodes:
                node = self.memory_nodes[memory_id]
                # Update the node data and trigger animation
                node.memory_data = event_data
                node.animation_type = 'pulse'
                node.animation_progress = 0
                
        elif operation == 'decay':
            # Start fade animation on the node if it exists
            if memory_id in self.memory_nodes:
                node = self.memory_nodes[memory_id]
                # Update strength and trigger fade animation
                node.memory_data = event_data
                node.animation_type = 'fade'
                node.animation_progress = 0
                
        elif operation == 'delete':
            # Remove the node if it exists
            if memory_id in self.memory_nodes:
                node = self.memory_nodes.pop(memory_id)
                self.memory_scene.removeItem(node)

def create_memory_visualization(parent=None, visualization_connector=None):
    """
    Create a memory visualization widget.
    
    Args:
        parent: Parent widget
        visualization_connector: V7VisualizationConnector instance
        
    Returns:
        MemoryVisualization widget if Qt is available, otherwise None
    """
    if not PYSIDE6_AVAILABLE:
        logger.warning("Cannot create memory visualization: Qt not available")
        return None
    
    try:
        return MemoryVisualization(parent, visualization_connector)
    except Exception as e:
        logger.error(f"Error creating memory visualization: {e}")
        return None

if __name__ == "__main__":
    # Simple test code to show the visualization
    import sys
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create test visualization
    viz = create_memory_visualization()
    
    # Add some test data
    for i in range(10):
        memory_type = ['experience', 'conversation', 'emotional', 'insight'][i % 4]
        viz.handle_memory_event({
            'type': 'memory',
            'operation': 'store',
            'memory_id': f'mem_{i}',
            'memory_type': memory_type,
            'strength': 0.5 + (i * 0.05),
            'created_at': time.time() - (i * 10),
            'last_accessed': time.time(),
            'tags': [f'tag_{i % 3}', f'tag_{(i+1) % 3}'],
            'animation': 'grow'
        })
    
    # Add test stats
    viz.handle_memory_event({
        'type': 'memory_summary',
        'memory_stats': {
            'memories_stored': 120,
            'active_memories': 87,
            'average_strength': 0.67
        },
        'store_type': 'sqlite',
        'decay_enabled': True
    })
    
    viz.show()
    sys.exit(app.exec()) 