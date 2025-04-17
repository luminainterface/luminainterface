"""
Fractal Pattern Panel for V5 PySide6 Client

This panel visualizes fractal patterns from neural network activity.
"""

import os
import sys
import time
import random
import math
import logging
from pathlib import Path

# Add parent directory to path if needed
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

logger = logging.getLogger(__name__)

# Try to import PySide6
try:
    from PySide6.QtCore import Qt, QTimer, Signal, Slot, QPointF, QRectF
    from PySide6.QtGui import QPainter, QPainterPath, QPen, QBrush, QColor, QLinearGradient
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, 
        QComboBox, QSpinBox, QSlider, QGroupBox, QSplitter, QFrame
    )
    USING_PYSIDE6 = True
except ImportError:
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal as Signal, pyqtSlot as Slot, QPointF, QRectF
    from PyQt5.QtGui import QPainter, QPainterPath, QPen, QBrush, QColor, QLinearGradient
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, 
        QComboBox, QSpinBox, QSlider, QGroupBox, QSplitter, QFrame
    )
    USING_PYSIDE6 = False

class FractalPatternPanel(QWidget):
    """Panel for visualizing fractal patterns from neural network activity"""
    
    # Class variables
    PANEL_NAME = "Fractal Pattern Visualization"
    PANEL_DESCRIPTION = "Visualizes neural patterns as fractal structures"
    
    @classmethod
    def get_panel_name(cls):
        """Get the name of the panel"""
        return cls.PANEL_NAME
    
    @classmethod
    def get_panel_description(cls):
        """Get the description of the panel"""
        return cls.PANEL_DESCRIPTION
    
    def __init__(self, socket_manager=None):
        """
        Initialize the Fractal Pattern Panel
        
        Args:
            socket_manager: Socket manager for communication with backend
        """
        super().__init__()
        
        self.socket_manager = socket_manager
        self.mock_mode = socket_manager is None or getattr(socket_manager, 'mock_mode', False)
        
        # State variables
        self.current_pattern = None
        self.pattern_style = "neural"
        self.fractal_depth = 5
        self.animation_speed = 1.0
        self.animation_phase = 0.0
        self.node_positions = []
        self.neural_weight = 0.5
        self.metrics = {}
        self.insights = {}
        
        # Set up UI
        self.init_ui()
        
        # Set up animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(50)  # 20 fps
        
        # Connect to socket manager
        self.connect_to_socket_manager()
        
        logger.info("Fractal Pattern Panel initialized")
    
    def init_ui(self):
        """Initialize the user interface"""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title = QLabel("Fractal Pattern Visualization")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #4B6EAF; margin: 5px;")
        layout.addWidget(title)
        
        # Create splitter for controls and visualization
        splitter = QSplitter(Qt.Horizontal)
        
        # Control panel
        self.control_panel = self.create_control_panel()
        splitter.addWidget(self.control_panel)
        
        # Visualization panel
        self.visualization_panel = FractalVisualizationPanel()
        splitter.addWidget(self.visualization_panel)
        
        # Add splitter to layout
        layout.addWidget(splitter)
        
        # Set initial splitter sizes (1:3 ratio)
        splitter.setSizes([200, 600])
        
        # Status bar
        self.status_bar = QFrame()
        status_layout = QHBoxLayout(self.status_bar)
        status_layout.setContentsMargins(5, 2, 5, 2)
        
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch(1)
        
        self.pattern_status = QLabel("No pattern loaded")
        status_layout.addWidget(self.pattern_status)
        
        layout.addWidget(self.status_bar)
    
    def create_control_panel(self):
        """Create the control panel with all settings"""
        # Control panel container
        control_panel = QWidget()
        control_panel.setMinimumWidth(200)
        control_panel.setMaximumWidth(300)
        
        # Control panel layout
        control_layout = QVBoxLayout(control_panel)
        
        # Pattern style group
        style_group = QGroupBox("Pattern Style")
        style_layout = QVBoxLayout(style_group)
        
        self.style_selector = QComboBox()
        self.style_selector.addItems(["Neural", "Mandelbrot", "Julia", "Tree"])
        self.style_selector.setCurrentText(self.pattern_style.capitalize())
        self.style_selector.currentTextChanged.connect(self.on_style_changed)
        
        style_layout.addWidget(self.style_selector)
        control_layout.addWidget(style_group)
        
        # Fractal depth group
        depth_group = QGroupBox("Fractal Depth")
        depth_layout = QVBoxLayout(depth_group)
        
        self.depth_spinner = QSpinBox()
        self.depth_spinner.setRange(1, 10)
        self.depth_spinner.setValue(self.fractal_depth)
        self.depth_spinner.valueChanged.connect(self.on_depth_changed)
        
        depth_layout.addWidget(self.depth_spinner)
        control_layout.addWidget(depth_group)
        
        # Animation speed group
        animation_group = QGroupBox("Animation Speed")
        animation_layout = QVBoxLayout(animation_group)
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 200)
        self.speed_slider.setValue(int(self.animation_speed * 100))
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        
        speed_label_layout = QHBoxLayout()
        speed_label_layout.addWidget(QLabel("Slow"))
        speed_label_layout.addStretch(1)
        speed_label_layout.addWidget(QLabel("Fast"))
        
        animation_layout.addWidget(self.speed_slider)
        animation_layout.addLayout(speed_label_layout)
        
        control_layout.addWidget(animation_group)
        
        # Metrics group
        metrics_group = QGroupBox("Pattern Metrics")
        metrics_layout = QFormLayout(metrics_group)
        
        self.dimension_label = QLabel("1.68")
        self.complexity_label = QLabel("78")
        self.coherence_label = QLabel("92")
        self.entropy_label = QLabel("Medium")
        
        metrics_layout.addRow("Fractal Dimension:", self.dimension_label)
        metrics_layout.addRow("Complexity Index:", self.complexity_label)
        metrics_layout.addRow("Pattern Coherence:", self.coherence_label)
        metrics_layout.addRow("Entropy Level:", self.entropy_label)
        
        control_layout.addWidget(metrics_group)
        
        # Insights group
        insights_group = QGroupBox("Pattern Insights")
        insights_layout = QVBoxLayout(insights_group)
        
        self.insights_label = QLabel("No patterns detected")
        self.insights_label.setWordWrap(True)
        
        insights_layout.addWidget(self.insights_label)
        control_layout.addWidget(insights_group)
        
        # Add stretch to push everything to the top
        control_layout.addStretch(1)
        
        return control_panel
    
    def connect_to_socket_manager(self):
        """Connect to the socket manager for data exchange"""
        if not self.socket_manager:
            logger.warning("No socket manager available")
            self.set_status("No socket manager available")
            return
        
        try:
            # Register message handlers
            self.socket_manager.register_message_handler(
                "pattern_data_updated", 
                self.handle_pattern_data_update
            )
            
            # Request initial pattern data
            self.request_pattern_data()
            
            self.set_status("Connected to socket manager")
            
        except Exception as e:
            logger.error(f"Error connecting to socket manager: {e}")
            self.set_status(f"Connection error: {str(e)}")
    
    def request_pattern_data(self):
        """Request pattern data from the server"""
        if not self.socket_manager:
            logger.warning("Cannot request pattern data: No socket manager available")
            return
        
        try:
            # Create request message
            message = {
                "type": "request_pattern_data",
                "request_id": f"pattern_request_{int(time.time())}",
                "content": {
                    "pattern_style": self.pattern_style.lower(),
                    "fractal_depth": self.fractal_depth,
                    "neural_weight": self.neural_weight
                }
            }
            
            # Send message
            self.socket_manager.send_message(message)
            
            self.set_status("Requesting pattern data...")
            
        except Exception as e:
            logger.error(f"Error requesting pattern data: {e}")
            self.set_status(f"Request error: {str(e)}")
    
    def handle_pattern_data_update(self, message):
        """
        Handle pattern data update message
        
        Args:
            message: Message data
        """
        try:
            data = message.get("data", {})
            
            # Check for errors
            if "error" in data:
                error_msg = data.get("error", "Unknown error")
                self.set_status(f"Error: {error_msg}")
                return
            
            # Update pattern data
            self.current_pattern = data
            self.pattern_style = data.get("pattern_style", self.pattern_style)
            self.fractal_depth = data.get("fractal_depth", self.fractal_depth)
            
            # Update UI to match data
            self.style_selector.setCurrentText(self.pattern_style.capitalize())
            self.depth_spinner.setValue(self.fractal_depth)
            
            # Update metrics
            self.metrics = data.get("metrics", {})
            self.update_metrics_display()
            
            # Update insights
            self.insights = data.get("insights", {})
            self.update_insights_display()
            
            # Update nodes
            self.node_positions = data.get("nodes", [])
            
            # Update visualization
            self.visualization_panel.set_pattern_data(
                self.pattern_style,
                self.fractal_depth,
                self.node_positions
            )
            
            # Update status
            self.set_pattern_status(f"{self.pattern_style.capitalize()} pattern (depth: {self.fractal_depth})")
            self.set_status("Pattern data updated")
            
        except Exception as e:
            logger.error(f"Error handling pattern data update: {e}")
            self.set_status(f"Error updating pattern: {str(e)}")
    
    def update_metrics_display(self):
        """Update the metrics display with current data"""
        if not self.metrics:
            return
        
        self.dimension_label.setText(str(self.metrics.get("fractal_dimension", "1.68")))
        self.complexity_label.setText(str(self.metrics.get("complexity_index", "78")))
        self.coherence_label.setText(str(self.metrics.get("pattern_coherence", "92")))
        self.entropy_label.setText(str(self.metrics.get("entropy_level", "Medium")))
    
    def update_insights_display(self):
        """Update the insights display with current data"""
        if not self.insights:
            return
        
        detected_patterns = self.insights.get("detected_patterns", [])
        if detected_patterns:
            self.insights_label.setText("\n".join(f"• {pattern}" for pattern in detected_patterns))
        else:
            self.insights_label.setText("No patterns detected")
    
    @Slot(str)
    def on_style_changed(self, style):
        """
        Handle pattern style change
        
        Args:
            style: New pattern style
        """
        self.pattern_style = style.lower()
        self.request_pattern_data()
    
    @Slot(int)
    def on_depth_changed(self, depth):
        """
        Handle fractal depth change
        
        Args:
            depth: New fractal depth
        """
        self.fractal_depth = depth
        self.request_pattern_data()
    
    @Slot(int)
    def on_speed_changed(self, value):
        """
        Handle animation speed change
        
        Args:
            value: New speed slider value
        """
        self.animation_speed = value / 100.0
        self.visualization_panel.set_animation_speed(self.animation_speed)
    
    def update_animation(self):
        """Update animation state"""
        self.animation_phase += 0.05 * self.animation_speed
        if self.animation_phase > 2 * math.pi:
            self.animation_phase -= 2 * math.pi
        
        self.visualization_panel.set_animation_phase(self.animation_phase)
    
    def set_neural_weight(self, weight):
        """
        Set the neural network weight
        
        Args:
            weight: Neural network weight (0.0-1.0)
        """
        if self.neural_weight != weight:
            self.neural_weight = weight
            self.request_pattern_data()
    
    def set_status(self, message):
        """
        Set the status message
        
        Args:
            message: Status message
        """
        self.status_label.setText(message)
    
    def set_pattern_status(self, message):
        """
        Set the pattern status message
        
        Args:
            message: Pattern status message
        """
        self.pattern_status.setText(message)


class FractalVisualizationPanel(QWidget):
    """Panel for visualizing fractal patterns"""
    
    def __init__(self):
        """Initialize the visualization panel"""
        super().__init__()
        
        # State variables
        self.pattern_style = "neural"
        self.fractal_depth = 5
        self.nodes = []
        self.animation_phase = 0.0
        self.animation_speed = 1.0
        
        # Set minimum size
        self.setMinimumSize(400, 300)
        
        # Set focus policy to accept keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)
    
    def set_pattern_data(self, style, depth, nodes):
        """
        Set the pattern data
        
        Args:
            style: Pattern style
            depth: Fractal depth
            nodes: Node positions
        """
        self.pattern_style = style
        self.fractal_depth = depth
        self.nodes = nodes
        self.update()
    
    def set_animation_phase(self, phase):
        """
        Set the animation phase
        
        Args:
            phase: Animation phase (0.0-2π)
        """
        self.animation_phase = phase
        self.update()
    
    def set_animation_speed(self, speed):
        """
        Set the animation speed
        
        Args:
            speed: Animation speed
        """
        self.animation_speed = speed
    
    def paintEvent(self, event):
        """Paint the visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # Fill background with gradient
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor(20, 20, 40))
        gradient.setColorAt(1, QColor(10, 10, 30))
        painter.fillRect(self.rect(), gradient)
        
        # Draw pattern based on style
        if self.pattern_style == "neural":
            self.draw_neural_pattern(painter)
        elif self.pattern_style == "mandelbrot":
            self.draw_mandelbrot_pattern(painter)
        elif self.pattern_style == "julia":
            self.draw_julia_pattern(painter)
        elif self.pattern_style == "tree":
            self.draw_tree_pattern(painter)
        else:
            self.draw_neural_pattern(painter)  # Default
    
    def draw_neural_pattern(self, painter):
        """
        Draw neural network pattern
        
        Args:
            painter: QPainter to use
        """
        if not self.nodes:
            # Draw placeholder
            painter.setPen(QColor(100, 100, 150))
            painter.drawText(self.rect(), Qt.AlignCenter, "Neural Pattern Visualization\nWaiting for data...")
            return
        
        # Draw connections
        for node in self.nodes:
            x1 = node["x"] * self.width()
            y1 = node["y"] * self.height()
            
            # Get connections
            connections = node.get("connections", [])
            
            for conn_id in connections:
                # Find the connected node
                conn_node = next((n for n in self.nodes if n.get("id") == conn_id), None)
                if not conn_node:
                    continue
                
                x2 = conn_node["x"] * self.width()
                y2 = conn_node["y"] * self.height()
                
                # Set up pen
                color = QColor(60, 80, 120, 100)
                pen = QPen(color, 1)
                painter.setPen(pen)
                
                # Draw connection
                painter.drawLine(x1, y1, x2, y2)
        
        # Draw nodes
        for node in self.nodes:
            x = node["x"] * self.width()
            y = node["y"] * self.height()
            
            # Get color and size
            color_vals = node.get("color", [100, 150, 255])
            size = node.get("size", 0.1) * min(self.width(), self.height()) * 0.1
            
            # Apply animation
            size *= 1.0 + 0.2 * math.sin(self.animation_phase + x * 0.01 + y * 0.01)
            
            # Create a gradient for the node
            color = QColor(color_vals[0], color_vals[1], color_vals[2])
            brush = QBrush(color)
            
            # Draw node
            painter.setBrush(brush)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(x, y), size, size)
            
            # Draw highlight
            highlight = QColor(255, 255, 255, 100)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(highlight, 1))
            painter.drawEllipse(QPointF(x, y), size * 0.8, size * 0.8)
    
    def draw_mandelbrot_pattern(self, painter):
        """
        Draw Mandelbrot fractal pattern
        
        Args:
            painter: QPainter to use
        """
        # This is a simplified version - a real implementation would compute the actual fractal
        center_x = self.width() / 2
        center_y = self.height() / 2
        size = min(self.width(), self.height()) * 0.4
        
        # Draw main shape
        path = QPainterPath()
        
        path.moveTo(center_x - size, center_y)
        
        # Left bulb
        path.arcTo(QRectF(center_x - size * 2, center_y - size, size * 2, size * 2), 0, 180)
        
        # Right bulb
        path.arcTo(QRectF(center_x - size, center_y - size / 2, size, size), 90, 180)
        path.arcTo(QRectF(center_x, center_y - size / 2, size, size), 270, 180)
        
        path.closeSubpath()
        
        # Create gradient fill
        gradient = QLinearGradient(center_x - size, center_y - size, center_x + size, center_y + size)
        gradient.setColorAt(0, QColor(30, 0, 100))
        gradient.setColorAt(0.5, QColor(80, 0, 120))
        gradient.setColorAt(1, QColor(50, 0, 80))
        
        # Draw pattern
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawPath(path)
        
        # Draw details
        for i in range(self.fractal_depth):
            scale = 1.0 - (i / self.fractal_depth) * 0.8
            offset_x = math.sin(self.animation_phase + i * 0.5) * size * 0.05
            offset_y = math.cos(self.animation_phase + i * 0.5) * size * 0.05
            
            # Draw detail circle
            painter.setPen(QPen(QColor(100, 50, 200, 100 - i * 10), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(center_x + offset_x, center_y + offset_y), 
                             size * scale, size * scale)
    
    def draw_julia_pattern(self, painter):
        """
        Draw Julia fractal pattern
        
        Args:
            painter: QPainter to use
        """
        # This is a simplified version - a real implementation would compute the actual fractal
        center_x = self.width() / 2
        center_y = self.height() / 2
        size = min(self.width(), self.height()) * 0.3
        
        # Parameters (simplified)
        c_real = 0.285 * math.sin(self.animation_phase * 0.2)
        c_imag = 0.01 * math.cos(self.animation_phase * 0.2)
        
        # Draw main shape
        for i in range(4):
            angle = i * math.pi / 2 + self.animation_phase * 0.1
            
            # Create a path for each lobe
            path = QPainterPath()
            
            # Starting point
            x = center_x + math.cos(angle) * size * 0.2
            y = center_y + math.sin(angle) * size * 0.2
            path.moveTo(x, y)
            
            # Create lobe shape
            for j in range(6):
                factor = 1.0 - j * 0.15
                angle_offset = j * 0.1 + self.animation_phase * 0.05
                
                x = center_x + math.cos(angle + angle_offset) * size * factor
                y = center_y + math.sin(angle + angle_offset) * size * factor
                
                path.lineTo(x, y)
            
            # Close the path
            path.closeSubpath()
            
            # Set color based on lobe
            hue = (i * 60 + int(self.animation_phase * 20)) % 360
            color = QColor()
            color.setHsv(hue, 200, 200, 200)
            
            # Draw lobe
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(255, 255, 255, 30), 1))
            painter.drawPath(path)
        
        # Draw detail circles
        for i in range(self.fractal_depth * 2):
            angle = i * math.pi / (self.fractal_depth) + self.animation_phase
            dist = 0.3 + i * 0.1
            
            x = center_x + math.cos(angle) * size * dist
            y = center_y + math.sin(angle) * size * dist
            
            radius = size * (0.1 - i * 0.01)
            
            # Create gradient
            gradient = QRadialGradient(x, y, radius)
            
            hue = (i * 40 + int(self.animation_phase * 30)) % 360
            color = QColor()
            color.setHsv(hue, 220, 220, 150)
            
            gradient.setColorAt(0, color)
            gradient.setColorAt(1, QColor(0, 0, 0, 0))
            
            # Draw circle
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(x, y), radius, radius)
    
    def draw_tree_pattern(self, painter):
        """
        Draw fractal tree pattern
        
        Args:
            painter: QPainter to use
        """
        # Starting point at bottom center
        start_x = self.width() / 2
        start_y = self.height() * 0.95
        
        # Initial branch length
        length = self.height() * 0.3
        
        # Initial angle (straight up)
        angle = -math.pi / 2
        
        # Animation angle offset
        angle_offset = math.sin(self.animation_phase) * 0.05
        
        # Draw the tree
        painter.setPen(QPen(QColor(80, 200, 120), 2))
        self._draw_branch(painter, start_x, start_y, length, angle, self.fractal_depth, angle_offset)
    
    def _draw_branch(self, painter, x, y, length, angle, depth, angle_offset):
        """
        Recursively draw tree branches
        
        Args:
            painter: QPainter to use
            x, y: Starting coordinates
            length: Branch length
            angle: Branch angle
            depth: Recursion depth
            angle_offset: Animation angle offset
        """
        if depth <= 0:
            return
        
        # Calculate end point
        end_x = x + math.cos(angle) * length
        end_y = y + math.sin(angle) * length
        
        # Set color based on depth
        if depth > 2:
            # Brown for trunk and main branches
            color = QColor(100 + depth * 20, 60 + depth * 5, 20)
            width = depth * 0.5
        else:
            # Green for leaves
            color = QColor(20, 150 + depth * 40, 50 + depth * 20)
            width = depth * 0.3
        
        # Draw branch
        painter.setPen(QPen(color, width))
        painter.drawLine(x, y, end_x, end_y)
        
        # Recursion factor
        factor = 0.65
        
        # Animation offsets
        branch_angle = 0.15 * (4 - depth) + angle_offset
        
        # Draw right branch
        self._draw_branch(
            painter, end_x, end_y, 
            length * factor, 
            angle - branch_angle, 
            depth - 1,
            angle_offset
        )
        
        # Draw left branch
        self._draw_branch(
            painter, end_x, end_y, 
            length * factor, 
            angle + branch_angle, 
            depth - 1,
            angle_offset
        )
        
        # Add a middle branch for more complex trees (for higher depths)
        if depth > 3:
            self._draw_branch(
                painter, end_x, end_y, 
                length * factor, 
                angle, 
                depth - 2,
                angle_offset
            ) 