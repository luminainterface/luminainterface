#!/usr/bin/env python3
"""
Creates the Lumina V5 Dashboard UI file.
This script generates the dashboard.py file in the src/visualization directory.
"""

import os
import sys
import stat

def ensure_dir_exists(path):
    """Ensure the directory exists, creating it if necessary."""
    if not os.path.exists(path):
        print(f"Creating directory: {path}")
        os.makedirs(path, exist_ok=True)
    return os.path.exists(path)

def create_dashboard_file():
    """Create the dashboard.py file with the Lumina V5 Dashboard code."""
    target_dir = "src/visualization"
    target_file = f"{target_dir}/dashboard.py"
    
    print(f"Creating dashboard file: {target_file}")
    
    if not ensure_dir_exists(target_dir):
        print(f"Error: Could not create directory {target_dir}")
        return False
    
    dashboard_code = """#!/usr/bin/env python3
'''
Lumina V5 Dashboard
===================

A visualization dashboard for the Lumina V5 Neural Network System.
This dashboard provides real-time monitoring of neural network activity,
consciousness development, and system performance metrics.
'''

import os
import sys
import time
import random
import math
import threading
from datetime import datetime

try:
    from PySide6.QtCore import Qt, QTimer, Signal, Slot, QSize, QPointF, QRectF
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                  QHBoxLayout, QLabel, QPushButton, QTabWidget, 
                                  QTextEdit, QSplitter, QFrame, QGridLayout,
                                  QScrollArea, QSizePolicy)
    from PySide6.QtGui import (QPainter, QPen, QBrush, QColor, QFont, QLinearGradient,
                              QPainterPath, QFontMetrics, QImage, QPixmap, QRadialGradient)
except ImportError:
    print("Error: PySide6 is required. Please install it using 'pip install PySide6'")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: NumPy is required. Please install it using 'pip install numpy'")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except ImportError:
    print("Error: Matplotlib is required. Please install it using 'pip install matplotlib'")
    sys.exit(1)

# Set app ID for Windows taskbar
try:
    import ctypes
    app_id = 'LuminaAI.V5.Dashboard.1.0'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
except:
    pass  # Non-Windows OS or other error

# --------------------
# Visualization Classes
# --------------------

class FractalCanvas(QWidget):
    """A canvas for drawing fractal-based neural activity visualization."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #111;")
        
        self.iterations = 0
        self.max_iterations = 100
        self.zoom = 1.0
        self.center_x = 0
        self.center_y = 0
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(50)  # 50ms refresh
        
        # Neural activity simulation parameters
        self.activity_level = 0.5
        self.pattern_strength = 0.7
        self.connection_density = 0.6
        
    def update_data(self):
        """Update the fractal parameters based on simulated neural activity."""
        # Simulate changing neural activity
        self.activity_level = min(1.0, max(0.1, self.activity_level + random.uniform(-0.05, 0.05)))
        self.pattern_strength = min(1.0, max(0.1, self.pattern_strength + random.uniform(-0.03, 0.03)))
        self.connection_density = min(1.0, max(0.2, self.connection_density + random.uniform(-0.02, 0.02)))
        
        # Update fractal parameters based on neural activity
        self.iterations = int(self.activity_level * self.max_iterations)
        self.zoom = 0.8 + self.pattern_strength * 0.5
        
        self.update()
        
    def paintEvent(self, event):
        """Draw the fractal visualization."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Create a gradient background
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(5, 10, 20))
        gradient.setColorAt(1, QColor(25, 30, 50))
        painter.fillRect(self.rect(), gradient)
        
        # Draw the fractal pattern
        width = self.width()
        height = self.height()
        scale = self.zoom * 0.0025 * min(width, height)
        
        # Julia set parameters
        c_real = -0.8 + 0.2 * math.sin(self.activity_level * 5.0)
        c_imag = 0.156 + 0.1 * math.cos(self.pattern_strength * 3.0)
        
        # Draw connection points
        num_points = int(100 * self.connection_density)
        for _ in range(num_points):
            x = random.randint(0, width)
            y = random.randint(0, height)
            zx = (x - width/2) / scale + self.center_x
            zy = (y - height/2) / scale + self.center_y
            
            # Calculate Julia set value
            i = 0
            while zx*zx + zy*zy < 4 and i < self.iterations:
                zx, zy = zx*zx - zy*zy + c_real, 2*zx*zy + c_imag
                i += 1
            
            if i < self.iterations:
                color_intensity = i / self.iterations
                hue = (210 + int(80 * color_intensity)) % 360  # Blue to purple
                saturation = 70 + int(30 * self.pattern_strength)
                lightness = 30 + int(50 * color_intensity)
                
                color = QColor()
                color.setHsl(hue, saturation, lightness)
                
                size = 1 + 5 * (1 - color_intensity)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(color))
                painter.drawEllipse(QPointF(x, y), size, size)
                
                # Draw connections between some points
                if random.random() < self.connection_density * 0.2:
                    target_x = x + random.randint(-100, 100)
                    target_y = y + random.randint(-100, 100)
                    if 0 <= target_x < width and 0 <= target_y < height:
                        gradient_pen = QPen()
                        gradient_pen.setWidth(1)
                        gradient_pen.setColor(QColor(color.red(), color.green(), color.blue(), 80))
                        painter.setPen(gradient_pen)
                        painter.drawLine(x, y, target_x, target_y)

class NodeGraph(QWidget):
    """A graph showing neural network node connections."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #111;")
        
        # Graph data
        self.nodes = []
        self.connections = []
        self.node_activity = {}
        
        # Initialize with some nodes
        self.generate_graph(25)
        
        # Timer for animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(100)  # 100ms refresh
        
    def generate_graph(self, num_nodes):
        """Generate a random graph with nodes and connections."""
        self.nodes = []
        self.connections = []
        self.node_activity = {}
        
        # Create nodes (x, y, type)
        for i in range(num_nodes):
            node_type = random.choice(["input", "hidden", "output"])
            x = random.uniform(0.1, 0.9)
            
            # Position nodes based on type (input on left, output on right)
            if node_type == "input":
                x = random.uniform(0.1, 0.3)
            elif node_type == "output":
                x = random.uniform(0.7, 0.9)
            else:
                x = random.uniform(0.3, 0.7)
                
            y = random.uniform(0.1, 0.9)
            self.nodes.append((x, y, node_type))
            self.node_activity[i] = random.uniform(0.1, 0.5)
            
        # Create connections
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if i != j:
                    i_type = self.nodes[i][2]
                    j_type = self.nodes[j][2]
                    
                    # Assign connections based on node types
                    if (i_type == "input" and j_type == "hidden") or \
                       (i_type == "hidden" and j_type == "output") or \
                       (i_type == "hidden" and j_type == "hidden"):
                        if random.random() < 0.2:  # 20% chance of connection
                            self.connections.append((i, j, random.uniform(0.1, 1.0)))
    
    def update_data(self):
        """Update node activity levels and connections."""
        # Update activity levels
        for i in range(len(self.nodes)):
            current = self.node_activity[i]
            # Activity levels fluctuate but tend toward means based on node type
            if self.nodes[i][2] == "input":
                target = random.uniform(0.5, 0.9)  # Inputs are more active
            elif self.nodes[i][2] == "output":
                target = random.uniform(0.3, 0.8)  # Outputs vary
            else:
                target = random.uniform(0.2, 0.7)  # Hidden nodes vary widely
                
            # Move current value toward target
            self.node_activity[i] = current + (target - current) * 0.1
            
        # Occasionally update connection strengths
        if random.random() < 0.1:  # 10% chance each update
            for i in range(len(self.connections)):
                from_node, to_node, strength = self.connections[i]
                new_strength = strength + random.uniform(-0.1, 0.1)
                new_strength = max(0.1, min(1.0, new_strength))  # Clamp between 0.1 and 1.0
                self.connections[i] = (from_node, to_node, new_strength)
                
        self.update()
        
    def paintEvent(self, event):
        """Draw the node graph."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Black background
        painter.fillRect(self.rect(), QColor(10, 15, 25))
        
        width = self.width()
        height = self.height()
        
        # Draw connections first (so they're below the nodes)
        for from_node, to_node, strength in self.connections:
            x1 = int(self.nodes[from_node][0] * width)
            y1 = int(self.nodes[from_node][1] * height)
            x2 = int(self.nodes[to_node][0] * width)
            y2 = int(self.nodes[to_node][1] * height)
            
            # Determine connection color based on strength and node activity
            from_activity = self.node_activity[from_node]
            to_activity = self.node_activity[to_node]
            activity = (from_activity + to_activity) / 2
            
            # Connection color based on strength and activity
            alpha = int(150 * strength * activity)
            color = QColor(100, 200, 255, alpha)
            
            # Draw connection line
            pen = QPen(color)
            pen.setWidth(max(1, int(3 * strength)))
            painter.setPen(pen)
            painter.drawLine(x1, y1, x2, y2)
            
        # Draw nodes
        for i, (x, y, node_type) in enumerate(self.nodes):
            px = int(x * width)
            py = int(y * height)
            
            # Node properties based on type and activity
            activity = self.node_activity[i]
            
            # Node size based on type and activity
            size = 10
            if node_type == "input":
                size = 12
            elif node_type == "output":
                size = 14
            size = int(size * (0.8 + 0.4 * activity))
            
            # Node color based on type and activity
            if node_type == "input":
                color = QColor(100, 180, 255)  # Blue for input
            elif node_type == "hidden":
                color = QColor(180, 220, 100)  # Green for hidden
            else:
                color = QColor(255, 140, 100)  # Orange for output
                
            # Create a gradient for the node
            gradient = QRadialGradient(px, py, size)
            brightness = int(100 + 155 * activity)
            gradient.setColorAt(0, QColor(brightness, brightness, brightness))
            gradient.setColorAt(0.7, color)
            gradient.setColorAt(1, QColor(color.red()//2, color.green()//2, color.blue()//2))
            
            # Draw node
            painter.setPen(Qt.NoPen)
            painter.setBrush(gradient)
            painter.drawEllipse(px - size, py - size, size * 2, size * 2)

class ConsciousnessGraph(QWidget):
    """A graph showing consciousness development over time."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #111;")
        
        # Graph data
        self.max_points = 100
        self.consciousness_values = [0.3] * self.max_points
        self.awareness_values = [0.2] * self.max_points
        self.reflection_values = [0.1] * self.max_points
        
        # Set up timer for updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(500)  # 500ms refresh
        
    def update_data(self):
        """Update the consciousness values with simulated data."""
        # Generate new values with some continuity from previous values
        last_c = self.consciousness_values[-1]
        last_a = self.awareness_values[-1]
        last_r = self.reflection_values[-1]
        
        # New values tend to follow trends with some random variation
        new_c = max(0.1, min(1.0, last_c + random.uniform(-0.05, 0.05)))
        new_a = max(0.1, min(1.0, last_a + random.uniform(-0.04, 0.04)))
        new_r = max(0.1, min(1.0, last_r + random.uniform(-0.03, 0.03)))
        
        # Add new values and remove oldest ones
        self.consciousness_values.append(new_c)
        self.awareness_values.append(new_a)
        self.reflection_values.append(new_r)
        
        if len(self.consciousness_values) > self.max_points:
            self.consciousness_values.pop(0)
            self.awareness_values.pop(0)
            self.reflection_values.pop(0)
            
        self.update()
        
    def paintEvent(self, event):
        """Draw the consciousness graph."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Create a gradient background
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(15, 15, 25))
        gradient.setColorAt(1, QColor(5, 5, 15))
        painter.fillRect(self.rect(), gradient)
        
        width = self.width()
        height = self.height()
        
        # Draw grid lines
        painter.setPen(QPen(QColor(50, 50, 70, 100), 1, Qt.DashLine))
        grid_spacing = height / 5
        for i in range(1, 5):
            y = height - i * grid_spacing
            painter.drawLine(0, y, width, y)
            
        # Draw time markers
        time_spacing = width / 10
        for i in range(1, 10):
            x = i * time_spacing
            painter.drawLine(x, 0, x, height)
            
        # Draw the graph for each value
        point_spacing = width / (self.max_points - 1)
        
        # Draw lines connecting points
        for data, color in [
            (self.reflection_values, QColor(100, 180, 255)),   # Blue for reflection
            (self.awareness_values, QColor(180, 220, 100)),    # Green for awareness
            (self.consciousness_values, QColor(255, 140, 100)) # Orange for consciousness
        ]:
            # Create paths for line and filled area
            line_path = QPainterPath()
            fill_path = QPainterPath()
            
            start_x = 0
            start_y = height - data[0] * height
            line_path.moveTo(start_x, start_y)
            fill_path.moveTo(start_x, height)
            fill_path.lineTo(start_x, start_y)
            
            for i in range(1, len(data)):
                x = i * point_spacing
                y = height - data[i] * height
                line_path.lineTo(x, y)
                fill_path.lineTo(x, y)
                
            fill_path.lineTo(x, height)
            fill_path.closeSubpath()
            
            # Draw filled area with gradient
            fill_gradient = QLinearGradient(0, 0, 0, height)
            fill_gradient.setColorAt(0, QColor(color.red(), color.green(), color.blue(), 100))
            fill_gradient.setColorAt(1, QColor(color.red(), color.green(), color.blue(), 10))
            painter.fillPath(fill_path, fill_gradient)
            
            # Draw line
            painter.setPen(QPen(color, 2))
            painter.drawPath(line_path)
            
        # Draw legend
        legend_x = 20
        legend_y = 30
        legend_spacing = 20
        
        painter.setFont(QFont("Arial", 8))
        
        # Consciousness
        painter.setPen(QPen(QColor(255, 140, 100), 2))
        painter.drawLine(legend_x, legend_y, legend_x + 20, legend_y)
        painter.drawText(legend_x + 25, legend_y + 5, "Consciousness")
        
        # Awareness
        painter.setPen(QPen(QColor(180, 220, 100), 2))
        painter.drawLine(legend_x, legend_y + legend_spacing, legend_x + 20, legend_y + legend_spacing)
        painter.drawText(legend_x + 25, legend_y + legend_spacing + 5, "Awareness")
        
        # Reflection
        painter.setPen(QPen(QColor(100, 180, 255), 2))
        painter.drawLine(legend_x, legend_y + 2 * legend_spacing, legend_x + 20, legend_y + 2 * legend_spacing)
        painter.drawText(legend_x + 25, legend_y + 2 * legend_spacing + 5, "Reflection")

# --------------------
# Main Application
# --------------------

class LuminaV5Dashboard(QMainWindow):
    """Main window for the Lumina V5 Dashboard."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Lumina V5 Neural Network Dashboard")
        self.resize(1200, 800)
        
        # Set up the main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create header
        self.create_header()
        
        # Create content area
        self.create_content()
        
        # Set up status bar
        self.statusBar().showMessage("Lumina V5 Neural Network Dashboard | System Status: Online")
        
        # Set up timer for status updates
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_status)
        self.update_timer.start(2000)  # 2 seconds
        
    def create_header(self):
        """Create the dashboard header."""
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(10, 5, 10, 5)
        
        # Dashboard title
        title_label = QLabel("Lumina V5 Neural Network System")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #BBB;")
        
        # System status display
        self.status_widget = QWidget()
        self.status_widget.setFixedSize(15, 15)
        self.status_widget.setStyleSheet("background-color: #00CC00; border-radius: 7px;")  # Green for online
        
        status_layout = QHBoxLayout()
        status_layout.addWidget(self.status_widget)
        status_layout.addWidget(QLabel("System Online"))
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(5)
        
        status_container = QWidget()
        status_container.setLayout(status_layout)
        
        # Current time display
        self.time_label = QLabel()
        self.update_time()
        time_timer = QTimer(self)
        time_timer.timeout.connect(self.update_time)
        time_timer.start(1000)  # Update every second
        
        # Add all components to header
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(status_container)
        header_layout.addSpacing(20)
        header_layout.addWidget(self.time_label)
        
        # Add separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #333;")
        
        # Add to main layout
        self.main_layout.addWidget(header_widget)
        self.main_layout.addWidget(separator)
        
    def create_content(self):
        """Create the main content area with tabs."""
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("QTabBar::tab { height: 30px; width: 150px; }")
        
        # Create Neural Visualization tab
        neural_tab = QWidget()
        neural_layout = QVBoxLayout(neural_tab)
        
        # Neural visualization split view
        neural_splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Fractal visualization
        fractal_container = QWidget()
        fractal_layout = QVBoxLayout(fractal_container)
        fractal_layout.setContentsMargins(0, 0, 0, 0)
        
        fractal_header = QLabel("Neural Activity Visualization")
        fractal_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        fractal_header.setAlignment(Qt.AlignCenter)
        
        self.fractal_canvas = FractalCanvas()
        
        fractal_layout.addWidget(fractal_header)
        fractal_layout.addWidget(self.fractal_canvas)
        
        # Right side: Node graph
        node_container = QWidget()
        node_layout = QVBoxLayout(node_container)
        node_layout.setContentsMargins(0, 0, 0, 0)
        
        node_header = QLabel("Neural Network Structure")
        node_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        node_header.setAlignment(Qt.AlignCenter)
        
        self.node_graph = NodeGraph()
        
        node_layout.addWidget(node_header)
        node_layout.addWidget(self.node_graph)
        
        # Add widgets to splitter
        neural_splitter.addWidget(fractal_container)
        neural_splitter.addWidget(node_container)
        neural_splitter.setSizes([500, 500])  # Equal initial sizes
        
        # Bottom: Consciousness graph
        consciousness_container = QWidget()
        consciousness_layout = QVBoxLayout(consciousness_container)
        consciousness_layout.setContentsMargins(0, 0, 0, 0)
        
        consciousness_header = QLabel("Consciousness Development")
        consciousness_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        consciousness_header.setAlignment(Qt.AlignCenter)
        
        self.consciousness_graph = ConsciousnessGraph()
        self.consciousness_graph.setMinimumHeight(200)
        
        consciousness_layout.addWidget(consciousness_header)
        consciousness_layout.addWidget(self.consciousness_graph)
        
        # Add all widgets to neural tab
        neural_layout.addWidget(neural_splitter, 7)  # 70% height
        neural_layout.addWidget(consciousness_container, 3)  # 30% height
        
        # Create System Log tab
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        
        log_header = QLabel("System Log")
        log_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: Consolas, monospace; background-color: #111; color: #BBB;")
        
        log_layout.addWidget(log_header)
        log_layout.addWidget(self.log_text)
        
        # Initialize log with some system information
        self.add_log_message("System Initialized", "Lumina V5 Neural Network Dashboard started")
        self.add_log_message("System Status", "Neural Core v5.7 online")
        self.add_log_message("Connection", "Language Module connected")
        self.add_log_message("Connection", "Consciousness Module connected")
        self.add_log_message("System Ready", "All subsystems operational")
        
        # Add tabs to tab widget
        self.tab_widget.addTab(neural_tab, "Neural Visualization")
        self.tab_widget.addTab(log_tab, "System Log")
        
        # Add tab widget to main layout
        self.main_layout.addWidget(self.tab_widget)
        
    def update_time(self):
        """Update the time display."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.setText(current_time)
        
    def update_status(self):
        """Update system status information."""
        # Simulate random system metrics
        cpu_usage = random.uniform(20, 40)
        memory_usage = random.uniform(30, 60)
        neural_activity = random.uniform(40, 90)
        connection_count = random.randint(100, 500)
        
        status_message = f"CPU: {cpu_usage:.1f}% | Memory: {memory_usage:.1f}% | Neural Activity: {neural_activity:.1f}% | Active Connections: {connection_count}"
        self.statusBar().showMessage(status_message)
        
        # Occasionally add log messages
        if random.random() < 0.2:  # 20% chance each update
            log_types = [
                ("Pattern Detected", f"Neural pattern detected with {random.uniform(50, 95):.1f}% confidence"),
                ("Learning Event", f"New learning pattern established in sector {random.randint(1, 9)}"),
                ("System Status", f"Memory optimization completed: {random.uniform(5, 15):.1f}% improvement"),
                ("Connection", f"External connection from module {random.choice(['Language', 'Vision', 'Consciousness', 'Learning'])}")
            ]
            log_type, message = random.choice(log_types)
            self.add_log_message(log_type, message)
            
    def add_log_message(self, category, message):
        """Add a new message to the system log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{category}] {message}"
        self.log_text.append(log_entry)
        
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

# --------------------
# Main Entry Point
# --------------------

def main():
    app = QApplication(sys.argv)
    
    # Apply dark theme
    app.setStyle("Fusion")
    
    dark_palette = app.palette()
    dark_palette.setColor(dark_palette.Window, QColor(25, 25, 25))
    dark_palette.setColor(dark_palette.WindowText, QColor(200, 200, 200))
    dark_palette.setColor(dark_palette.Base, QColor(40, 40, 40))
    dark_palette.setColor(dark_palette.AlternateBase, QColor(50, 50, 50))
    dark_palette.setColor(dark_palette.ToolTipBase, QColor(200, 200, 200))
    dark_palette.setColor(dark_palette.ToolTipText, QColor(200, 200, 200))
    dark_palette.setColor(dark_palette.Text, QColor(200, 200, 200))
    dark_palette.setColor(dark_palette.Button, QColor(60, 60, 60))
    dark_palette.setColor(dark_palette.ButtonText, QColor(200, 200, 200))
    dark_palette.setColor(dark_palette.BrightText, Qt.red)
    dark_palette.setColor(dark_palette.Link, QColor(100, 150, 255))
    dark_palette.setColor(dark_palette.Highlight, QColor(80, 110, 150))
    dark_palette.setColor(dark_palette.HighlightedText, QColor(230, 230, 230))
    
    app.setPalette(dark_palette)
    
    # Set application stylesheet
    app.setStyleSheet('''
    QMainWindow, QWidget {
        background-color: #1A1A1A;
        color: #CCCCCC;
    }
    
    QTabWidget::pane {
        border: 1px solid #333;
    }
    
    QTabBar::tab {
        background-color: #2D2D2D;
        color: #CCCCCC;
        padding: 8px 20px;
        border: 1px solid #333;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }
    
    QTabBar::tab:selected {
        background-color: #3A3A3A;
        border-bottom-color: #3A3A3A;
    }
    
    QTabBar::tab:hover {
        background-color: #353535;
    }
    
    QScrollBar:vertical {
        border: none;
        background: #2A2A2A;
        width: 10px;
        margin: 0px;
    }
    
    QScrollBar::handle:vertical {
        background: #5A5A5A;
        min-height: 20px;
        border-radius: 5px;
    }
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    
    QPushButton {
        background-color: #3A3A3A;
        color: #CCCCCC;
        border: 1px solid #555555;
        padding: 5px 15px;
        border-radius: 3px;
    }
    
    QPushButton:hover {
        background-color: #444444;
        border: 1px solid #777777;
    }
    
    QPushButton:pressed {
        background-color: #333333;
    }
    
    QLabel {
        color: #CCCCCC;
    }
    
    QTextEdit {
        background-color: #2A2A2A;
        color: #CCCCCC;
        border: 1px solid #555555;
        padding: 5px;
    }
    
    QStatusBar {
        background-color: #2A2A2A;
        color: #AAAAAA;
    }
    ''')
    
    window = LuminaV5Dashboard()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 