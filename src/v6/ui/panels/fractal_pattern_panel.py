"""
Fractal Pattern Visualization Panel

Provides a high-density fractal pattern visualization based on the V5 Fractal Echo Visualization.
"""

import math
import random
import logging
import time
from pathlib import Path

# Add project root to path if needed
import sys
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
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

# Import the panel base
from ..panel_base import V6PanelBase

# Set up logging
logger = logging.getLogger(__name__)

class FractalPatternPanel(V6PanelBase):
    """Fractal pattern visualization panel with controls and real-time updates"""
    
    # Signal emitted when pattern changes
    patternChanged = Signal(dict)
    
    def __init__(self, socket_manager=None, parent=None):
        super().__init__(parent)
        self.socket_manager = socket_manager
        
        # Pattern state
        self.pattern_type = "julia"  # julia, mandelbrot
        self.fractal_depth = 5
        self.animation_speed = 50  # 0-100
        self.animation_timer = None
        self.animation_time = 0
        self.pattern_metrics = {
            "dimension": 1.79,
            "complexity": 80,
            "coherence": 88,
            "entropy": "Medium-High"
        }
        
        # Julia set parameters
        self.julia_c_real = -0.7
        self.julia_c_imag = 0.27
        
        # Mandelbrot parameters
        self.mandelbrot_center_x = 0.0
        self.mandelbrot_center_y = 0.0
        self.mandelbrot_zoom = 1.0
        
        # Common parameters
        self.max_iterations = 300
        self.color_scheme = 0  # 0-5 different color schemes
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        
        # Cached rendering
        self.fractal_image = None
        self.last_render_time = 0
        self.needs_redraw = True
        
        # Mouse interaction
        self.last_mouse_pos = None
        self.is_dragging = False
        
        # Initialize UI
        self.init_ui()
        
        # Set up socket manager event handling if available
        if self.socket_manager:
            self.setup_socket_events()
        
        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)
        
        # Start animation
        self.start_animation()
    
    def init_ui(self):
        """Initialize the user interface components"""
        # Create main layout with sidebar and visualization area
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create left sidebar for controls
        self.sidebar = QtWidgets.QWidget()
        self.sidebar.setFixedWidth(180)
        self.sidebar.setStyleSheet("""
            background-color: rgba(26, 38, 52, 220);
            border-right: 1px solid rgba(52, 73, 94, 150);
        """)
        
        sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(5)
        
        # Pattern Style Selector
        sidebar_layout.addWidget(self.create_heading("Pattern Style"))
        
        self.pattern_combo = QtWidgets.QComboBox()
        self.pattern_combo.addItems(["Julia", "Mandelbrot"])
        self.pattern_combo.setStyleSheet("""
            background-color: rgba(44, 62, 80, 180);
            color: white;
            padding: 5px;
            border: 1px solid rgba(52, 73, 94, 120);
            border-radius: 4px;
            selection-background-color: rgba(52, 152, 219, 150);
        """)
        self.pattern_combo.currentTextChanged.connect(self.change_pattern_type)
        sidebar_layout.addWidget(self.pattern_combo)
        
        # Fractal Depth Slider
        sidebar_layout.addWidget(self.create_heading("Fractal Depth"))
        
        depth_layout = QtWidgets.QHBoxLayout()
        
        self.depth_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.depth_slider.setRange(1, 10)
        self.depth_slider.setValue(self.fractal_depth)
        self.depth_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: rgba(52, 73, 94, 150);
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: rgba(52, 152, 219, 200);
                border: 1px solid rgba(41, 128, 185, 255);
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
        """)
        self.depth_slider.valueChanged.connect(self.change_fractal_depth)
        
        self.depth_value = QtWidgets.QLabel(str(self.fractal_depth))
        self.depth_value.setStyleSheet("color: white; min-width: 20px; text-align: right;")
        self.depth_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        depth_layout.addWidget(self.depth_slider)
        depth_layout.addWidget(self.depth_value)
        sidebar_layout.addLayout(depth_layout)
        
        # Animation Speed Slider
        sidebar_layout.addWidget(self.create_heading("Animation Speed"))
        
        self.speed_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 100)
        self.speed_slider.setValue(self.animation_speed)
        self.speed_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: rgba(52, 73, 94, 150);
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: rgba(52, 152, 219, 200);
                border: 1px solid rgba(41, 128, 185, 255);
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
        """)
        self.speed_slider.valueChanged.connect(self.change_animation_speed)
        sidebar_layout.addWidget(self.speed_slider)
        
        # Metrics Section
        sidebar_layout.addWidget(self.create_heading("Metrics"))
        
        metrics_widget = QtWidgets.QWidget()
        metrics_layout = QtWidgets.QFormLayout(metrics_widget)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setSpacing(3)
        metrics_layout.setLabelAlignment(Qt.AlignLeft)
        metrics_layout.setFormAlignment(Qt.AlignLeft)
        metrics_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        
        # Add metrics labels
        self.fractal_dimension = QtWidgets.QLabel(str(self.pattern_metrics["dimension"]))
        self.complexity_index = QtWidgets.QLabel(str(self.pattern_metrics["complexity"]))
        self.pattern_coherence = QtWidgets.QLabel(str(self.pattern_metrics["coherence"]))
        self.entropy_level = QtWidgets.QLabel(self.pattern_metrics["entropy"])
        
        # Set styles
        for label in [self.fractal_dimension, self.complexity_index, self.pattern_coherence, self.entropy_level]:
            label.setStyleSheet("color: #3498DB; font-weight: bold;")
        
        # Add to layout
        metrics_layout.addRow("Fractal Dimension:", self.fractal_dimension)
        metrics_layout.addRow("Complexity Index:", self.complexity_index)
        metrics_layout.addRow("Pattern Coherence:", self.pattern_coherence)
        metrics_layout.addRow("Entropy Level:", self.entropy_level)
        
        sidebar_layout.addWidget(metrics_widget)
        
        # Pattern Insights
        sidebar_layout.addWidget(self.create_heading("Pattern Insights"))
        
        self.insights_text = QtWidgets.QLabel("• Julia set connectivity patterns")
        self.insights_text.setWordWrap(True)
        self.insights_text.setStyleSheet("color: #ECF0F1; font-size: 12px;")
        sidebar_layout.addWidget(self.insights_text)
        
        # Status label
        self.status_label = QtWidgets.QLabel("Pattern updated: julia (depth 5)")
        self.status_label.setStyleSheet("color: rgba(127, 140, 141, 220); font-size: 11px;")
        sidebar_layout.addWidget(self.status_label)
        
        # Add stretch at the bottom
        sidebar_layout.addStretch(1)
        
        # Add sidebar to main layout
        layout.addWidget(self.sidebar)
        
        # Create visualization area
        self.viz_area = QtWidgets.QWidget()
        self.viz_area.setStyleSheet("""
            background-color: rgba(16, 26, 40, 180);
        """)
        self.viz_area.setMouseTracking(True)
        
        # Add visualization area to main layout
        layout.addWidget(self.viz_area, 1)  # 1 = stretch factor
    
    def create_heading(self, text):
        """Create a section heading label"""
        label = QtWidgets.QLabel(text)
        label.setStyleSheet("""
            color: #3498DB;
            font-weight: bold;
            font-size: 13px;
            border-bottom: 1px solid rgba(52, 152, 219, 150);
            padding-bottom: 3px;
            margin-top: 5px;
        """)
        return label
    
    def setup_socket_events(self):
        """Set up event handlers for socket manager events"""
        if not self.socket_manager:
            return
            
        # Register for pattern update events
        self.socket_manager.register_handler("fractal_pattern_update", self.handle_pattern_update)
        
        # Request initial pattern data
        self.socket_manager.emit("request_fractal_pattern", {
            "type": self.pattern_type,
            "depth": self.fractal_depth
        })
    
    def handle_pattern_update(self, data):
        """Handle pattern update events from socket manager"""
        if "metrics" in data:
            self.pattern_metrics = data["metrics"]
            self.update_metrics_display()
        
        if "insights" in data:
            self.insights_text.setText(data["insights"])
        
        # If pattern parameters are included, update them
        if "parameters" in data:
            params = data["parameters"]
            if self.pattern_type == "julia" and "julia_c" in params:
                self.julia_c_real = params["julia_c"][0]
                self.julia_c_imag = params["julia_c"][1]
            elif self.pattern_type == "mandelbrot" and "center" in params:
                self.mandelbrot_center_x = params["center"][0]
                self.mandelbrot_center_y = params["center"][1]
                if "zoom" in params:
                    self.mandelbrot_zoom = params["zoom"]
        
        # Force redraw
        self.needs_redraw = True
        self.update()
    
    def update_metrics_display(self):
        """Update the metrics display with current values"""
        self.fractal_dimension.setText(str(self.pattern_metrics["dimension"]))
        self.complexity_index.setText(str(self.pattern_metrics["complexity"]))
        self.pattern_coherence.setText(str(self.pattern_metrics["coherence"]))
        self.entropy_level.setText(self.pattern_metrics["entropy"])
    
    def change_pattern_type(self, pattern_type):
        """Change the fractal pattern type"""
        self.pattern_type = pattern_type.lower()
        logger.info(f"Changed pattern type to: {self.pattern_type}")
        
        # Update status
        self.status_label.setText(f"Pattern updated: {self.pattern_type} (depth {self.fractal_depth})")
        
        # Reset zoom and pan
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        
        # Force redraw
        self.needs_redraw = True
        self.update()
        
        # Send update via socket manager if available
        if self.socket_manager:
            self.socket_manager.emit("change_fractal_pattern", {
                "type": self.pattern_type,
                "depth": self.fractal_depth
            })
        
        # Emit signal
        self.patternChanged.emit({
            "type": self.pattern_type,
            "depth": self.fractal_depth
        })
    
    def change_fractal_depth(self, depth):
        """Change the fractal rendering depth"""
        self.fractal_depth = depth
        self.depth_value.setText(str(depth))
        logger.info(f"Changed fractal depth to: {depth}")
        
        # Update status
        self.status_label.setText(f"Pattern updated: {self.pattern_type} (depth {self.fractal_depth})")
        
        # Force redraw
        self.needs_redraw = True
        self.update()
        
        # Send update via socket manager if available
        if self.socket_manager:
            self.socket_manager.emit("change_fractal_depth", {
                "depth": self.fractal_depth
            })
        
        # Emit signal
        self.patternChanged.emit({
            "type": self.pattern_type,
            "depth": self.fractal_depth
        })
    
    def change_animation_speed(self, speed):
        """Change the animation speed"""
        self.animation_speed = speed
        logger.info(f"Changed animation speed to: {speed}")
        
        # Adjust animation timer if active
        if self.animation_timer and self.animation_timer.isActive():
            # If speed is 0, pause animation
            if speed == 0:
                self.animation_timer.stop()
            else:
                # Calculate interval based on speed (100 = fastest = 10ms, 1 = slowest = 100ms)
                interval = max(10, int(110 - speed))
                self.animation_timer.setInterval(interval)
                
                # Restart if stopped
                if not self.animation_timer.isActive():
                    self.animation_timer.start()
    
    def start_animation(self):
        """Start the fractal animation"""
        if not self.animation_timer:
            self.animation_timer = QtCore.QTimer(self)
            self.animation_timer.timeout.connect(self.animate_frame)
            
            # Calculate interval based on speed (100 = fastest = 10ms, 1 = slowest = 100ms)
            interval = max(10, int(110 - self.animation_speed))
            self.animation_timer.setInterval(interval)
            
            # Only start if speed > 0
            if self.animation_speed > 0:
                self.animation_timer.start()
    
    def stop_animation(self):
        """Stop the fractal animation"""
        if self.animation_timer and self.animation_timer.isActive():
            self.animation_timer.stop()
    
    def animate_frame(self):
        """Update animation for one frame"""
        # Update animation time
        self.animation_time += 0.05
        
        if self.pattern_type == "julia":
            # Animate Julia set by changing c parameter
            angle = self.animation_time
            radius = 0.3 + 0.1 * math.sin(self.animation_time * 0.2)
            self.julia_c_real = radius * math.cos(angle)
            self.julia_c_imag = radius * math.sin(angle)
        else:
            # For Mandelbrot, we could animate zoom or position
            zoom_factor = 1.0 + 0.1 * math.sin(self.animation_time * 0.3)
            self.mandelbrot_zoom = zoom_factor
        
        # Force redraw
        self.needs_redraw = True
        self.update()
    
    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        # Force redraw when size changes
        self.needs_redraw = True
    
    def mousePressEvent(self, event):
        """Handle mouse press events for interaction"""
        if event.button() == Qt.LeftButton and self.viz_area.geometry().contains(event.pos()):
            self.is_dragging = True
            self.last_mouse_pos = event.pos()
            # Pause animation while dragging
            if self.animation_timer and self.animation_timer.isActive():
                self.animation_timer.stop()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for interaction"""
        if self.is_dragging and self.last_mouse_pos:
            # Calculate drag delta in widget coordinates
            delta_x = event.pos().x() - self.last_mouse_pos.x()
            delta_y = event.pos().y() - self.last_mouse_pos.y()
            
            # Convert to fractal coordinates based on zoom
            scale = 2.0 / (min(self.viz_area.width(), self.viz_area.height()) * self.zoom)
            fractal_dx = delta_x * scale
            fractal_dy = delta_y * scale
            
            # Update pan values
            self.pan_x += fractal_dx
            self.pan_y += fractal_dy
            
            # Update mouse position
            self.last_mouse_pos = event.pos()
            
            # Force redraw
            self.needs_redraw = True
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if event.button() == Qt.LeftButton:
            self.is_dragging = False
            
            # Resume animation if speed > 0
            if self.animation_speed > 0 and self.animation_timer and not self.animation_timer.isActive():
                self.animation_timer.start()
    
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming"""
        # Get mouse position in widget coordinates
        mouse_pos = event.position()  # For Qt6
        if not hasattr(event, "position"):  # For Qt5 compatibility
            mouse_pos = event.posF()
        
        # Only process if over visualization area
        if self.viz_area.geometry().contains(mouse_pos.toPoint()):
            # Calculate zoom factor
            zoom_factor = 1.1 if event.angleDelta().y() > 0 else 0.9
            
            # Get mouse position relative to viz area
            viz_x = mouse_pos.x() - self.viz_area.x()
            viz_y = mouse_pos.y() - self.viz_area.y()
            
            # Convert to normalized coordinates (-1 to 1)
            norm_x = 2.0 * viz_x / self.viz_area.width() - 1.0
            norm_y = 2.0 * viz_y / self.viz_area.height() - 1.0
            
            # Convert to fractal coordinates
            fractal_x = norm_x / self.zoom + self.pan_x
            fractal_y = norm_y / self.zoom + self.pan_y
            
            # Update zoom
            self.zoom *= zoom_factor
            
            # Adjust pan to keep mouse point fixed
            self.pan_x = fractal_x - norm_x / self.zoom
            self.pan_y = fractal_y - norm_y / self.zoom
            
            # Force redraw
            self.needs_redraw = True
            self.update()
            
            # Accept the event
            event.accept()
    
    def paintEvent(self, event):
        """Custom paint event to render the fractal pattern"""
        super().paintEvent(event)
        
        # Get the visualization area geometry
        viz_rect = self.viz_area.geometry()
        
        # Skip if area is too small
        if viz_rect.width() < 10 or viz_rect.height() < 10:
            return
        
        # Check if we need to redraw the fractal
        current_time = time.time()
        if self.needs_redraw or not self.fractal_image or current_time - self.last_render_time > 0.5:
            # Create the fractal image
            self.fractal_image = self.render_fractal(viz_rect.width(), viz_rect.height())
            self.last_render_time = current_time
            self.needs_redraw = False
        
        # Paint the fractal image
        if self.fractal_image:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            
            # Draw the fractal image in the visualization area
            painter.drawImage(viz_rect, self.fractal_image)
            
            # Draw border around visualization area
            painter.setPen(QtGui.QPen(QtGui.QColor(52, 73, 94, 150), 1))
            painter.drawRect(viz_rect)
    
    def render_fractal(self, width, height):
        """Render the fractal pattern to an image"""
        if self.pattern_type == "julia":
            return self.render_julia_set(width, height)
        else:
            return self.render_mandelbrot_set(width, height)
    
    def render_julia_set(self, width, height):
        """Render a Julia set fractal"""
        # Create an image to draw on
        image = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)
        
        # Determine drawing dimensions
        size = min(width, height)
        center_x = width / 2
        center_y = height / 2
        scale = 1.5 / (size / 2) / self.zoom
        
        # Julia set parameters
        c_real = self.julia_c_real
        c_imag = self.julia_c_imag
        
        # Draw each pixel
        for y in range(height):
            for x in range(width):
                # Convert pixel coordinates to complex plane
                zx = (x - center_x) * scale + self.pan_x
                zy = (y - center_y) * scale + self.pan_y
                
                # Julia set iteration
                i = 0
                while zx*zx + zy*zy < 4 and i < self.max_iterations:
                    temp = zx*zx - zy*zy + c_real
                    zy = 2*zx*zy + c_imag
                    zx = temp
                    i += 1
                
                # Color mapping
                if i == self.max_iterations:
                    # Point is in set - black
                    color = QtGui.qRgb(0, 0, 0)
                else:
                    # Point is outside set - color based on iterations and scheme
                    color = self.map_color(i, self.max_iterations)
                
                # Set pixel color
                image.setPixel(x, y, color)
        
        return image
    
    def render_mandelbrot_set(self, width, height):
        """Render a Mandelbrot set fractal"""
        # Create an image to draw on
        image = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)
        
        # Determine drawing dimensions
        size = min(width, height)
        center_x = width / 2
        center_y = height / 2
        scale = 2.5 / (size / 2) / self.zoom
        
        # Mandelbrot set parameters
        center_real = self.mandelbrot_center_x
        center_imag = self.mandelbrot_center_y
        
        # Draw each pixel
        for y in range(height):
            for x in range(width):
                # Convert pixel coordinates to complex plane
                c_real = (x - center_x) * scale + center_real + self.pan_x
                c_imag = (y - center_y) * scale + center_imag + self.pan_y
                
                # Initial values
                zx = 0
                zy = 0
                
                # Mandelbrot set iteration
                i = 0
                while zx*zx + zy*zy < 4 and i < self.max_iterations:
                    temp = zx*zx - zy*zy + c_real
                    zy = 2*zx*zy + c_imag
                    zx = temp
                    i += 1
                
                # Color mapping
                if i == self.max_iterations:
                    # Point is in set - black
                    color = QtGui.qRgb(0, 0, 0)
                else:
                    # Point is outside set - color based on iterations and scheme
                    color = self.map_color(i, self.max_iterations)
                
                # Set pixel color
                image.setPixel(x, y, color)
        
        return image
    
    def map_color(self, iterations, max_iterations):
        """Map iteration count to RGB color"""
        if self.pattern_type == "julia":
            # Julia color scheme
            norm = iterations / max_iterations
            
            # Create a more psychedelic color scheme
            r = int(255 * (0.5 + 0.5 * math.sin(3.0 * norm * math.pi)))
            g = int(255 * (0.5 + 0.5 * math.sin(3.0 * norm * math.pi + 2 * math.pi / 3)))
            b = int(255 * (0.5 + 0.5 * math.sin(3.0 * norm * math.pi + 4 * math.pi / 3)))
            
            return QtGui.qRgb(r, g, b)
        else:
            # Mandelbrot color scheme
            norm = iterations / max_iterations
            
            # Use a different color mapping
            hue = 240 * norm
            sat = 70 + 30 * norm
            val = 80 + 20 * norm if norm > 0.2 else 50 * norm
            
            # Convert HSV to RGB
            color = QtGui.QColor()
            color.setHsv(int(hue), int(sat), int(val))
            return color.rgb()
    
    def update_random_metrics(self):
        """Update metrics with random values for demo purposes"""
        # In a real implementation, these would be calculated based on the fractal properties
        self.pattern_metrics["dimension"] = round(1.5 + random.random() * 0.5, 2)
        self.pattern_metrics["complexity"] = int(70 + random.random() * 30)
        self.pattern_metrics["coherence"] = int(70 + random.random() * 30)
        
        entropy_values = ["Low", "Medium-Low", "Medium", "Medium-High", "High"]
        self.pattern_metrics["entropy"] = random.choice(entropy_values)
        
        # Update display
        self.update_metrics_display()
        
        # Update insights
        if self.pattern_type == "julia":
            self.insights_text.setText("• Julia set connectivity patterns")
        else:
            self.insights_text.setText("• Mandelbrot boundary complexity")
    
    def closeEvent(self, event):
        """Clean up when panel is closed"""
        # Stop animation timer
        if self.animation_timer and self.animation_timer.isActive():
            self.animation_timer.stop()
        
        super().closeEvent(event) 