"""
Fractal Pattern Panel for V5 Visualization System

This panel visualizes neural patterns as fractal structures,
showing their complexity, coherence, and other metrics.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to Python path if needed
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import Qt compatibility layer
from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui, Qt, Signal, Slot
# Import specific Qt classes needed
from src.v5.ui.qt_compat import get_widgets, get_gui, get_core, QtCompat

# Get required Qt classes
QSplitter = get_widgets().QSplitter
QFormLayout = get_widgets().QFormLayout
QTimer = get_core().QTimer
QPainter = get_gui().QPainter
QLinearGradient = get_gui().QLinearGradient
QColor = get_gui().QColor
QFont = get_gui().QFont
QPainterPath = get_gui().QPainterPath
QPen = get_gui().QPen
QBrush = get_gui().QBrush
QRadialGradient = get_gui().QRadialGradient
QPointF = QtCore.QPointF

import json
import logging
import math
import random
import time
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FractalPatternPanel(QtWidgets.QWidget):
    """Panel for visualizing fractal patterns from neural network activity"""
    
    # Signals
    pattern_selected = Signal(dict)  # Emitted when a pattern is selected
    
    def __init__(self, socket_manager=None):
        """
        Initialize the Fractal Pattern Panel
        
        Args:
            socket_manager: Optional socket manager for plugin communication
        """
        super().__init__()
        
        # Component name for state persistence
        self.component_name = "fractal_pattern_panel"
        
        # Initialize data
        self.socket_manager = socket_manager
        self.current_pattern = {}
        self.pattern_style = "neural"
        self.fractal_depth = 5
        self.animation_speed = 1.0
        self.animation_timer = None
        self.animation_phase = 0.0
        self.node_positions = []
        self.metrics = {
            "fractal_dimension": 0.0,
            "complexity_index": 0,
            "pattern_coherence": 0,
            "entropy_level": "Unknown"
        }
        self.insights = {}
        
        # Set up UI
        self.initUI()
        
        # Connect to socket manager if available
        if socket_manager:
            self.connect_to_socket_manager()
        
        # Start animation
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(50)  # 20 fps
        
    def initUI(self):
        """Initialize the user interface"""
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QtWidgets.QLabel("Fractal Pattern Visualization")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #4B6EAF;")
        layout.addWidget(title)
        
        # Controls and visualization splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Control panel on the left
        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_layout.setContentsMargins(0, 0, 10, 0)
        
        # Pattern style selector
        style_group = QtWidgets.QGroupBox("Pattern Style")
        style_layout = QtWidgets.QVBoxLayout(style_group)
        self.style_selector = QtWidgets.QComboBox()
        self.style_selector.addItems(["Neural", "Mandelbrot", "Julia", "Tree"])
        self.style_selector.setCurrentText(self.pattern_style.capitalize())
        self.style_selector.currentTextChanged.connect(self.on_style_changed)
        style_layout.addWidget(self.style_selector)
        control_layout.addWidget(style_group)
        
        # Depth control
        depth_group = QtWidgets.QGroupBox("Fractal Depth")
        depth_layout = QtWidgets.QVBoxLayout(depth_group)
        self.depth_spinner = QtWidgets.QSpinBox()
        self.depth_spinner.setRange(1, 10)
        self.depth_spinner.setValue(self.fractal_depth)
        self.depth_spinner.valueChanged.connect(self.on_depth_changed)
        depth_layout.addWidget(self.depth_spinner)
        control_layout.addWidget(depth_group)
        
        # Animation speed
        animation_group = QtWidgets.QGroupBox("Animation Speed")
        animation_layout = QtWidgets.QVBoxLayout(animation_group)
        self.speed_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 200)
        self.speed_slider.setValue(int(self.animation_speed * 100))
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        animation_layout.addWidget(self.speed_slider)
        control_layout.addWidget(animation_group)
        
        # Metrics display
        metrics_group = QtWidgets.QGroupBox("Metrics")
        metrics_layout = QFormLayout(metrics_group)
        
        self.dimension_label = QtWidgets.QLabel("1.68")
        self.complexity_label = QtWidgets.QLabel("78")
        self.coherence_label = QtWidgets.QLabel("92")
        self.entropy_label = QtWidgets.QLabel("Medium")
        
        metrics_layout.addRow("Fractal Dimension:", self.dimension_label)
        metrics_layout.addRow("Complexity Index:", self.complexity_label)
        metrics_layout.addRow("Pattern Coherence:", self.coherence_label)
        metrics_layout.addRow("Entropy Level:", self.entropy_label)
        
        control_layout.addWidget(metrics_group)
        
        # Insights
        insights_group = QtWidgets.QGroupBox("Pattern Insights")
        insights_layout = QtWidgets.QVBoxLayout(insights_group)
        self.insights_label = QtWidgets.QLabel("No patterns detected")
        self.insights_label.setWordWrap(True)
        insights_layout.addWidget(self.insights_label)
        control_layout.addWidget(insights_group)
        
        # Add stretch to fill remaining space
        control_layout.addStretch(1)
        
        # Fractal visualization panel on the right
        self.fractal_view = FractalPatternView()
        
        # Add panels to splitter
        splitter.addWidget(control_panel)
        splitter.addWidget(self.fractal_view)
        
        # Set initial sizes (1:3 ratio)
        splitter.setSizes([int(self.width() * 0.25), int(self.width() * 0.75)])
        
        layout.addWidget(splitter)
        
        # Status bar
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet("color: #666;")
        layout.addWidget(self.status_label)
    
    def connect_to_socket_manager(self):
        """Connect to the socket manager"""
        try:
            # Get direct access to the socket manager
            plugin = self.socket_manager.get_plugin("pattern_processor_1")
            
            if plugin:
                logger.info(f"Found socket manager directly: {plugin.node_id}")
                
                # Connect to socket
                plugin.socket.connect_to(self.socket_manager.manager_socket)
                
                # Register for pattern update messages
                self.socket_manager.register_message_handler(
                    "pattern_data_updated", 
                    self.handle_pattern_update
                )
                
                # Request initial pattern data
                self.request_pattern_data()
                
                self.status_label.setText("Connected to Socket Manager")
                return
                
            # Try to directly connect to the socket manager by ID
            connected = self.socket_manager.establish_direct_connection(
                self, 
                "pattern_processor_1", 
                "fractal_view"
            )
            
            if connected:
                # Register for pattern update messages if not already registered
                self.socket_manager.register_message_handler(
                    "pattern_data_updated", 
                    self.handle_pattern_update
                )
                
                # Request initial pattern data
                self.request_pattern_data()
                
                self.status_label.setText("Connected to Socket Manager")
                return
                
            # Fallback to traditional component provider approach
            providers = self.socket_manager.get_ui_component_providers("fractal_view")
            if providers:
                plugin_id = providers[0]["plugin"].node_id
                logger.info(f"Connecting to socket manager: {plugin_id}")
                
                # Connect socket
                self.socket_manager.connect_ui_to_plugin(self, plugin_id)
                
                # Register for pattern update messages
                self.socket_manager.register_message_handler(
                    "pattern_data_updated", 
                    self.handle_pattern_update
                )
                
                # Request initial pattern data
                self.request_pattern_data()
                
                self.status_label.setText("Connected to Socket Manager")
            else:
                logger.warning("No socket manager found")
                self.status_label.setText("No Socket Manager found")
        except Exception as e:
            logger.error(f"Error connecting to socket manager: {str(e)}")
            self.status_label.setText(f"Connection error: {str(e)}")
    
    def request_pattern_data(self):
        """Request pattern data from socket manager"""
        request_id = f"pattern_request_{int(time.time())}"
        message = {
            "type": "request_pattern_data",
            "request_id": request_id,
            "plugin_id": "pattern_processor_1",  # Explicitly target the socket manager
            "content": {
                "pattern_style": self.pattern_style.lower(),
                "fractal_depth": self.fractal_depth
            }
        }
        logger.info(f"Sending pattern data request with ID: {request_id}")
        self.socket_manager.send_message(message)
    
    def handle_pattern_update(self, message):
        """Handle pattern data update messages"""
        try:
            data = message.get("data", {})
            if "error" in data:
                error_msg = data.get("error", "Unknown error")
                self.status_label.setText(f"Error: {error_msg}")
                return
            
            # Update pattern data
            self.current_pattern = data
            received_style = data.get("pattern_style", self.pattern_style).lower()
            self.pattern_style = received_style
            self.fractal_depth = data.get("fractal_depth", self.fractal_depth)
            
            # Ensure style selector matches the current pattern
            current_text = self.style_selector.currentText().lower()
            if current_text != self.pattern_style:
                # Update the style selector without triggering signals
                self.style_selector.blockSignals(True)
                self.style_selector.setCurrentText(self.pattern_style.capitalize())
                self.style_selector.blockSignals(False)
            
            # Update the spinner without triggering signals
            self.depth_spinner.blockSignals(True)
            self.depth_spinner.setValue(self.fractal_depth)
            self.depth_spinner.blockSignals(False)
            
            # Update metrics
            self.metrics = data.get("metrics", {})
            self.update_metrics_display()
            
            # Update insights
            self.insights = data.get("insights", {})
            self.update_insights_display()
            
            # Update visualization
            self.node_positions = data.get("nodes", [])
            self.fractal_view.update_pattern(
                self.pattern_style, 
                self.fractal_depth,
                self.node_positions
            )
            
            # Signal pattern selection for other components
            self.pattern_selected.emit(data)
            
            self.status_label.setText(f"Pattern updated: {self.pattern_style} (depth {self.fractal_depth})")
        except Exception as e:
            logger.error(f"Error handling pattern update: {str(e)}")
            self.status_label.setText(f"Error processing pattern: {str(e)}")
    
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
            self.insights_label.setText("\n".join([f"• {pattern}" for pattern in detected_patterns]))
        else:
            self.insights_label.setText("No patterns detected")
    
    @Slot(str)
    def on_style_changed(self, style):
        """Handle pattern style change"""
        new_style = style.lower()
        if new_style != self.pattern_style:
            logger.info(f"Pattern style changed from {self.pattern_style} to {new_style}")
            self.pattern_style = new_style
            
            # Update the fractal view directly to give immediate feedback
            self.fractal_view.pattern_style = new_style
            self.fractal_view.update()
            
            # Then request updated data from the socket manager
            self.request_pattern_data()
    
    @Slot(int)
    def on_depth_changed(self, depth):
        """Handle fractal depth change"""
        self.fractal_depth = depth
        self.request_pattern_data()
    
    @Slot(int)
    def on_speed_changed(self, value):
        """Handle animation speed change"""
        self.animation_speed = value / 100.0
    
    def update_animation(self):
        """Update animation state"""
        self.animation_phase += 0.05 * self.animation_speed
        if self.animation_phase > 2 * math.pi:
            self.animation_phase -= 2 * math.pi
        
        self.fractal_view.set_animation_phase(self.animation_phase)
        self.fractal_view.update()
    
    def update_visualization(self):
        """Update the visualization with latest data"""
        # Request latest pattern data
        self.request_pattern_data()
    
    def cleanup(self):
        """Clean up resources before closing"""
        # Stop animation timer
        if self.animation_timer and self.animation_timer.isActive():
            self.animation_timer.stop()
            
        # Deregister message handlers
        self.socket_manager.deregister_message_handler("pattern_data_updated")

    def set_neural_weight(self, weight):
        """
        Set the neural network weight for pattern generation
        
        Args:
            weight: Neural network weight (0.0-1.0)
        """
        # Store the weight
        self.neural_weight = weight
        
        # Update fractal parameters based on the weight
        if hasattr(self, 'fractal_params'):
            # Adjust complexity based on neural weight (higher weight = more complex)
            self.fractal_params["complexity"] = 0.4 + (weight * 0.6)
            
            # Adjust recursion depth based on neural weight
            self.fractal_params["recursion_depth"] = max(2, min(6, int(2 + (weight * 4))))
            
            # Adjust other parameters
            self.fractal_params["resonance_factor"] = 0.4 + (weight * 0.6)
            
            # Log the change
            import logging
            logging.getLogger(__name__).info(
                f"Updated fractal parameters: complexity={self.fractal_params['complexity']:.2f}, "
                f"recursion_depth={self.fractal_params['recursion_depth']}"
            )
            
            # Update the visualization if auto-update is enabled
            if hasattr(self, 'auto_update') and self.auto_update:
                self.update_visualization()

    def restore_state(self, state):
        """
        Restore panel state from saved data
        
        Args:
            state: State data dictionary
        """
        try:
            # Update pattern style and depth if available
            if "pattern_style" in state:
                self.pattern_style = state["pattern_style"]
                self.style_selector.setCurrentText(self.pattern_style.capitalize())
                
            if "fractal_depth" in state:
                self.fractal_depth = state["fractal_depth"]
                self.depth_spinner.setValue(self.fractal_depth)
                
            # Update metrics if available
            if "metrics" in state:
                self.metrics = state["metrics"]
                self.update_metrics_display()
                
            # Update insights if available
            if "insights" in state:
                self.insights = state["insights"]
                self.update_insights_display()
                
            # Update visualization if nodes are available
            if "nodes" in state:
                self.node_positions = state["nodes"]
                self.fractal_view.update_pattern(
                    self.pattern_style,
                    self.fractal_depth,
                    self.node_positions
                )
                
            logger.info(f"Restored state for {self.component_name}")
        except Exception as e:
            logger.error(f"Error restoring state: {str(e)}")


class FractalPatternView(QtWidgets.QWidget):
    """Custom widget for visualizing fractal patterns"""
    
    def __init__(self):
        super().__init__()
        self.pattern_style = "neural"
        self.fractal_depth = 5
        self.nodes = []
        self.animation_phase = 0.0
        
        # Set up widget
        self.setMinimumSize(400, 300)
    
    def update_pattern(self, style, depth, nodes):
        """Update the pattern visualization"""
        self.pattern_style = style
        self.fractal_depth = depth
        self.nodes = nodes
        self.update()
    
    def set_animation_phase(self, phase):
        """Set the animation phase"""
        self.animation_phase = phase
    
    def paintEvent(self, event):
        """Paint the fractal pattern"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor(20, 20, 40))
        gradient.setColorAt(1, QColor(10, 10, 30))
        painter.fillRect(self.rect(), gradient)
        
        # Select drawing method based on pattern style
        if self.pattern_style == "neural":
            self.draw_neural_pattern(painter)
        elif self.pattern_style == "mandelbrot":
            self.draw_mandelbrot_pattern(painter)
        elif self.pattern_style == "julia":
            self.draw_julia_pattern(painter)
        elif self.pattern_style == "tree":
            self.draw_tree_pattern(painter)
        else:
            # Draw fallback with style name
            painter.setPen(QColor(150, 150, 150))
            painter.setFont(QFont("Arial", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, f"Unknown pattern style: {self.pattern_style}")
            logger.warning(f"Unknown pattern style selected: {self.pattern_style}")
    
    def draw_neural_pattern(self, painter):
        """Draw neural network pattern"""
        if not self.nodes:
            # Draw placeholder
            painter.setPen(QColor(100, 100, 150))
            painter.setFont(QFont("Arial", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "Neural Pattern Visualization")
            return
        
        # Scale coordinates to fit view
        min_x = min((node.get("x", 0) for node in self.nodes), default=0)
        max_x = max((node.get("x", 100) for node in self.nodes), default=100)
        min_y = min((node.get("y", 0) for node in self.nodes), default=0)
        max_y = max((node.get("y", 100) for node in self.nodes), default=100)
        
        width = max(1, max_x - min_x)
        height = max(1, max_y - min_y)
        
        scale_x = (self.width() - 40) / width
        scale_y = (self.height() - 40) / height
        scale = min(scale_x, scale_y)
        
        offset_x = (self.width() - width * scale) / 2
        offset_y = (self.height() - height * scale) / 2
        
        # Draw connections
        painter.setPen(QPen(QColor(60, 90, 180, 80), 1))
        
        for node in self.nodes:
            x1 = offset_x + (node.get("x", 0) - min_x) * scale
            y1 = offset_y + (node.get("y", 0) - min_y) * scale
            
            # Draw connections to other nodes
            for conn in node.get("connections", []):
                # Find target node
                target = next((n for n in self.nodes if n.get("id") == conn), None)
                if target:
                    x2 = offset_x + (target.get("x", 0) - min_x) * scale
                    y2 = offset_y + (target.get("y", 0) - min_y) * scale
                    
                    # Draw animated connection
                    path = QPainterPath()
                    path.moveTo(x1, y1)
                    
                    # Curved path with animation
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    # Add slight curve
                    angle = math.atan2(y2 - y1, x2 - x1) + math.pi/2
                    curve_distance = min(30, ((x2-x1)**2 + (y2-y1)**2)**0.5 / 4)
                    
                    # Animation effect
                    curve_amplitude = curve_distance * (0.5 + 0.5 * math.sin(self.animation_phase))
                    
                    cx += curve_amplitude * math.cos(angle)
                    cy += curve_amplitude * math.sin(angle)
                    
                    path.quadTo(cx, cy, x2, y2)
                    painter.drawPath(path)
        
        # Draw nodes
        for node in self.nodes:
            x = offset_x + (node.get("x", 0) - min_x) * scale
            y = offset_y + (node.get("y", 0) - min_y) * scale
            
            # Node size based on depth
            depth = node.get("depth", 0)
            base_size = 16 - depth * 2
            size = max(4, base_size)
            
            # Animation effect - pulsing
            size_mod = 1.0 + 0.2 * math.sin(self.animation_phase + depth)
            size *= size_mod
            
            # Node color based on depth
            hue = (depth * 20) % 360
            color = QColor()
            color.setHsv(hue, 200, 255, 200)
            
            # Draw node
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QPointF(x, y), size/2, size/2)
            
            # Draw glow effect
            glow = QPainterPath()
            glow.addEllipse(QPointF(x, y), size, size)
            gradient = QRadialGradient(x, y, size)
            gradient.setColorAt(0, QColor(color.red(), color.green(), color.blue(), 50))
            gradient.setColorAt(1, QColor(color.red(), color.green(), color.blue(), 0))
            painter.fillPath(glow, gradient)
    
    def draw_mandelbrot_pattern(self, painter):
        """Draw Mandelbrot-inspired pattern"""
        width = self.width()
        height = self.height()
        
        # Set up coordinate mapping for Mandelbrot set
        x_min, x_max = -2.5, 1.0
        y_min, y_max = -1.5, 1.5
        
        # Add slight animation to the coordinates
        animation_offset = 0.05 * math.sin(self.animation_phase)
        x_min += animation_offset
        x_max += animation_offset
        
        # Resolution for performance - reduce for better visibility
        resolution = 2  # Lower for better quality, higher for better performance
        
        # Maximum iterations
        max_iterations = 100
        
        # Draw pattern
        for x in range(0, width, resolution):
            for y in range(0, height, resolution):
                # Map screen coordinates to Mandelbrot coordinates
                c_real = x_min + (x_max - x_min) * x / width
                c_imag = y_min + (y_max - y_min) * y / height
                
                # Mandelbrot calculation
                z_real, z_imag = 0, 0
                iteration = 0
                
                # Main Mandelbrot iteration
                while iteration < max_iterations and z_real*z_real + z_imag*z_imag < 4:
                    # Calculate z = z² + c
                    temp = z_real*z_real - z_imag*z_imag + c_real
                    z_imag = 2*z_real*z_imag + c_imag
                    z_real = temp
                    iteration += 1
                
                # Color mapping based on iteration count
                if iteration < max_iterations:
                    # Smooth coloring algorithm
                    if z_real*z_real + z_imag*z_imag > 1e-10:  # Prevent log(0)
                        log_zn = math.log(z_real*z_real + z_imag*z_imag) / 2
                        nu = math.log(log_zn / math.log(2)) / math.log(2)
                        iteration = iteration + 1 - nu
                    
                    # Color mapping
                    color1 = QColor(0, 7, 100)
                    color2 = QColor(32, 107, 203) 
                    color3 = QColor(237, 255, 255)
                    
                    # Use a better color scheme for visibility
                    if iteration < max_iterations * 0.16:
                        ratio = iteration / (max_iterations * 0.16)
                        color = self._interpolate_color(color1, color2, ratio)
                    else:
                        ratio = (iteration - max_iterations * 0.16) / (max_iterations * 0.84)
                        color = self._interpolate_color(color2, color3, ratio)
                else:
                    # Inside the set - black
                    color = QColor(0, 0, 0)
                
                # Draw pixel block
                painter.fillRect(x, y, resolution, resolution, color)
    
    def draw_julia_pattern(self, painter):
        """Draw Julia set pattern with spiral features"""
        width = self.width()
        height = self.height()
        
        # Animation for Julia parameters - use values that produce spiral patterns
        # These c values are known to produce interesting spiral patterns for Julia sets
        time_factor = self.animation_phase / 4
        t = time_factor % 1.0
        
        # Oscillate between different spiral-producing constants
        if t < 0.25:
            # First spiral pattern
            c_real = -0.7269 + 0.05 * math.cos(self.animation_phase)
            c_imag = 0.1889 + 0.05 * math.sin(self.animation_phase)
        elif t < 0.5:
            # Second spiral pattern (Douady's rabbit)
            c_real = -0.123 + 0.03 * math.cos(self.animation_phase)
            c_imag = 0.745 + 0.03 * math.sin(self.animation_phase)
        elif t < 0.75:
            # Third spiral pattern (Siegel disk)
            c_real = -0.391 + 0.04 * math.cos(self.animation_phase)
            c_imag = -0.587 + 0.04 * math.sin(self.animation_phase)
        else:
            # Fourth spiral pattern (San Marco)
            c_real = -0.75 + 0.05 * math.cos(self.animation_phase)
            c_imag = 0.1 + 0.05 * math.sin(self.animation_phase)
        
        # Set up coordinate mapping - use a smaller window to zoom in on patterns
        x_min, x_max = -1.5, 1.5
        y_min, y_max = -1.2, 1.2
        
        # Resolution for performance
        resolution = 2  # Lower resolution = higher quality
        
        # Maximum iterations
        max_iterations = 120
        
        # Draw pattern
        for x in range(0, width, resolution):
            for y in range(0, height, resolution):
                # Map screen coordinates to complex plane
                z_real = x_min + (x_max - x_min) * x / width
                z_imag = y_min + (y_max - y_min) * y / height
                
                # Julia set iteration
                iteration = 0
                
                # Main Julia iteration
                while iteration < max_iterations and z_real*z_real + z_imag*z_imag < 4:
                    # Calculate z = z² + c
                    temp = z_real*z_real - z_imag*z_imag + c_real
                    z_imag = 2*z_real*z_imag + c_imag
                    z_real = temp
                    iteration += 1
                
                # Color mapping based on iteration count
                if iteration < max_iterations:
                    # Create a smooth gradient
                    # Calculate smooth iteration count
                    if z_real*z_real + z_imag*z_imag > 1e-10:  # Prevent log(0)
                        log_zn = math.log(z_real*z_real + z_imag*z_imag) / 2
                        nu = math.log(log_zn / math.log(2)) / math.log(2)
                        iteration = iteration + 1 - nu
                    
                    # Use a vibrant color palette for spirals
                    color1 = QColor(20, 0, 80)    # Deep purple
                    color2 = QColor(120, 0, 170)  # Violet 
                    color3 = QColor(220, 60, 180) # Pink
                    color4 = QColor(255, 180, 60) # Orange-gold
                    
                    # Multi-point gradient for more interesting visuals
                    if iteration < max_iterations * 0.3:
                        ratio = iteration / (max_iterations * 0.3)
                        color = self._interpolate_color(color1, color2, ratio)
                    elif iteration < max_iterations * 0.6:
                        ratio = (iteration - max_iterations * 0.3) / (max_iterations * 0.3)
                        color = self._interpolate_color(color2, color3, ratio)
                    else:
                        ratio = (iteration - max_iterations * 0.6) / (max_iterations * 0.4)
                        color = self._interpolate_color(color3, color4, ratio)
                else:
                    # Inside the set - darker color
                    color = QColor(5, 0, 20)
                
                # Draw pixel block with full opacity
                painter.fillRect(x, y, resolution, resolution, color)
    
    def _interpolate_color(self, color1, color2, ratio):
        """Interpolate between two colors"""
        r = int(color1.red() + (color2.red() - color1.red()) * ratio)
        g = int(color1.green() + (color2.green() - color1.green()) * ratio)
        b = int(color1.blue() + (color2.blue() - color1.blue()) * ratio)
        
        return QColor(r, g, b)
    
    def draw_tree_pattern(self, painter):
        """Draw recursive tree pattern"""
        width = self.width()
        height = self.height()
        
        # Start drawing from bottom center
        start_x = width / 2
        start_y = height * 0.9
        
        # Initial branch length
        length = height * 0.3
        
        # Animation parameters
        angle_offset = 0.1 * math.sin(self.animation_phase)
        
        # Set up painter
        painter.setPen(QPen(QColor(150, 255, 150, 100), 1))
        
        # Draw the tree
        self._draw_branch(painter, start_x, start_y, length, -math.pi/2, self.fractal_depth, angle_offset)
    
    def _draw_branch(self, painter, x, y, length, angle, depth, angle_offset):
        """Recursively draw tree branches"""
        if depth <= 0:
            return
            
        # Calculate endpoint
        end_x = x + length * math.cos(angle)
        end_y = y + length * math.sin(angle)
        
        # Branch width based on depth
        width = max(1, depth * 1.5)
        
        # Branch color based on depth
        hue = (100 + depth * 15) % 360
        saturation = 150 + depth * 10
        value = 150 + depth * 10
        alpha = 50 + depth * 20
        
        color = QColor()
        color.setHsv(int(hue), int(saturation), int(value), int(alpha))
        
        painter.setPen(QPen(color, width))
        
        # Draw branch
        painter.drawLine(int(x), int(y), int(end_x), int(end_y))
        
        # Reduce length for next branches
        new_length = length * 0.7
        
        # Branch angles with animation
        left_angle = angle - math.pi/5 - angle_offset
        right_angle = angle + math.pi/5 + angle_offset
        
        # Draw branches
        self._draw_branch(painter, end_x, end_y, new_length, left_angle, depth-1, angle_offset)
        self._draw_branch(painter, end_x, end_y, new_length, right_angle, depth-1, angle_offset)
        # Add middle branch occasionally
        if depth > 2 and random.random() < 0.4:
            middle_angle = angle + angle_offset * 2
            self._draw_branch(painter, end_x, end_y, new_length * 0.9, middle_angle, depth-2, angle_offset) 