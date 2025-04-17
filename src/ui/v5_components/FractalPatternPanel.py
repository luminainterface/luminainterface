from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QComboBox, QSlider, QFrame,
                             QScrollArea, QSplitter, QTabWidget, QGroupBox)
from PySide6.QtCore import Qt, Signal, QPointF, QTimer, QRectF
from PySide6.QtGui import QPainter, QBrush, QPen, QColor, QLinearGradient, QPainterPath, QFont

import math
import random

class FractalNodeWidget(QWidget):
    """Widget for visualizing and interacting with fractal nodes"""
    
    node_activated = Signal(dict)
    
    def __init__(self, node_data=None, parent=None):
        super().__init__(parent)
        self.node_data = node_data or {"id": 0, "depth": 3, "complexity": 0.7, "name": "Fractal Node"}
        self.setMinimumSize(120, 120)
        self.setMaximumSize(200, 200)
        self.is_hovering = False
        self.animation_phase = 0
        self.is_selected = False
        self.setMouseTracking(True)
        
        # Start animation timer
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.update_animation)
        self.anim_timer.start(50)
        
    def update_animation(self):
        """Update animation state"""
        self.animation_phase += 0.03
        if self.animation_phase > 2 * math.pi:
            self.animation_phase -= 2 * math.pi
        self.update()
    
    def enterEvent(self, event):
        """Handle mouse enter event"""
        self.is_hovering = True
        self.update()
    
    def leaveEvent(self, event):
        """Handle mouse leave event"""
        self.is_hovering = False
        self.update()
    
    def mousePressEvent(self, event):
        """Handle mouse press event"""
        self.is_selected = not self.is_selected
        self.node_activated.emit(self.node_data)
        self.update()
    
    def paintEvent(self, event):
        """Paint the fractal node"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        center_x = width / 2
        center_y = height / 2
        
        # Draw background
        if self.is_selected:
            bg_color = QColor(60, 80, 120)
        elif self.is_hovering:
            bg_color = QColor(50, 60, 90)
        else:
            bg_color = QColor(40, 45, 70)
        
        painter.fillRect(0, 0, width, height, bg_color)
        
        # Draw fractal pattern
        depth = self.node_data.get("depth", 3)
        complexity = self.node_data.get("complexity", 0.7)
        
        # Base pattern color
        pattern_color = QColor(100, 220, 255, 150)
        highlight_color = QColor(150, 250, 255, 200)
        
        # Recursive function to draw fractal pattern
        def draw_fractal(x, y, size, depth, angle_offset):
            if depth <= 0:
                return
            
            # Size of this level
            current_size = size * (0.8 + 0.2 * math.sin(self.animation_phase))
            
            # Draw current level
            points = []
            sides = 5  # Pentagon shape
            for i in range(sides):
                angle = angle_offset + i * 2 * math.pi / sides
                px = x + current_size * math.cos(angle)
                py = y + current_size * math.sin(angle)
                points.append(QPointF(px, py))
            
            # Create path for shape
            path = QPainterPath()
            path.moveTo(points[0])
            for point in points[1:]:
                path.lineTo(point)
            path.closeSubpath()
            
            # Set color based on depth
            alpha = 100 + 155 * (depth / 3)
            current_color = QColor(
                pattern_color.red(),
                pattern_color.green(),
                pattern_color.blue(),
                int(alpha)
            )
            
            # Fill shape
            painter.setBrush(QBrush(current_color))
            painter.setPen(QPen(highlight_color, 1))
            painter.drawPath(path)
            
            # Draw recursive fractals
            new_size = current_size * 0.4
            for i in range(sides):
                if random.random() < complexity:  # Skip some branches for variety
                    angle = angle_offset + i * 2 * math.pi / sides
                    new_x = x + current_size * 0.7 * math.cos(angle)
                    new_y = y + current_size * 0.7 * math.sin(angle)
                    new_angle = angle_offset + self.animation_phase * 0.2
                    draw_fractal(new_x, new_y, new_size, depth - 1, new_angle)
        
        # Draw the fractal
        draw_fractal(center_x, center_y, min(width, height) * 0.35, depth, self.animation_phase)
        
        # Draw node name
        name = self.node_data.get("name", "Fractal Node")
        painter.setPen(Qt.white)
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        text_rect = QRectF(10, height - 30, width - 20, 20)
        painter.drawText(text_rect, Qt.AlignCenter, name)


class ParadoxVisualizerWidget(QWidget):
    """Widget for visualizing paradox handling capabilities"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.animation_phase = 0
        self.paradox_level = 0.5  # 0.0 to 1.0
        
        # Animation timer
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.update_animation)
        self.anim_timer.start(50)
        
        # Sample paradox states
        self.paradox_states = [
            {"name": "Quantum Paradox", "resolution": 0.7, "complexity": 0.8},
            {"name": "Logical Contradiction", "resolution": 0.4, "complexity": 0.6},
            {"name": "Temporal Anomaly", "resolution": 0.9, "complexity": 0.9},
            {"name": "Perceptual Illusion", "resolution": 0.6, "complexity": 0.5}
        ]
        
        self.current_paradox = self.paradox_states[0]
    
    def set_paradox(self, paradox_data):
        """Set current paradox data"""
        self.current_paradox = paradox_data
        self.update()
    
    def update_animation(self):
        """Update animation state"""
        self.animation_phase += 0.02
        if self.animation_phase > 2 * math.pi:
            self.animation_phase -= 2 * math.pi
            
        # Occasionally change paradox level
        if random.random() < 0.01:
            self.paradox_level = min(1.0, max(0.1, self.paradox_level + random.uniform(-0.2, 0.2)))
            
        self.update()
    
    def paintEvent(self, event):
        """Paint paradox visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Draw background
        bg_gradient = QLinearGradient(0, 0, width, height)
        bg_gradient.setColorAt(0, QColor(30, 30, 50))
        bg_gradient.setColorAt(1, QColor(20, 20, 40))
        painter.fillRect(0, 0, width, height, bg_gradient)
        
        # Draw title
        title = self.current_paradox.get("name", "Paradox Visualization")
        painter.setPen(Qt.white)
        font = painter.font()
        font.setBold(True)
        font.setPointSize(12)
        painter.setFont(font)
        painter.drawText(20, 30, title)
        
        # Draw resolution status
        resolution = self.current_paradox.get("resolution", 0.5)
        resolution_text = f"Resolution: {int(resolution * 100)}%"
        painter.setPen(QColor(200, 200, 255))
        font.setPointSize(10)
        font.setBold(False)
        painter.setFont(font)
        painter.drawText(20, 50, resolution_text)
        
        # Draw paradox visualization
        center_x = width / 2
        center_y = height / 2 + 20
        
        # Draw dual circles representing the paradox
        radius = min(width, height) * 0.25
        complexity = self.current_paradox.get("complexity", 0.5)
        
        # First circle (reality)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(100, 200, 255, 150), 2))
        painter.drawEllipse(
            center_x - radius + radius * 0.3 * math.sin(self.animation_phase), 
            center_y - radius * 0.5, 
            radius * 2, 
            radius
        )
        
        # Second circle (alternative)
        painter.setPen(QPen(QColor(255, 150, 200, 150), 2))
        painter.drawEllipse(
            center_x - radius - radius * 0.3 * math.sin(self.animation_phase), 
            center_y - radius * 0.5, 
            radius * 2, 
            radius
        )
        
        # Draw intersection patterns
        num_lines = int(10 + 20 * complexity)
        for i in range(num_lines):
            angle = i * 2 * math.pi / num_lines + self.animation_phase
            offset1 = 0.3 * math.sin(self.animation_phase)
            offset2 = -0.3 * math.sin(self.animation_phase)
            
            x1 = center_x + offset1 * radius + radius * math.cos(angle)
            y1 = center_y + radius * 0.5 * math.sin(angle)
            
            x2 = center_x + offset2 * radius + radius * math.cos(angle + math.pi)
            y2 = center_y + radius * 0.5 * math.sin(angle + math.pi)
            
            # Color based on resolution
            color = QColor(
                int(255 * (1 - resolution * 0.5)),
                int(150 + 100 * resolution),
                255,
                100
            )
            
            painter.setPen(QPen(color, 1, Qt.DashLine))
            painter.drawLine(x1, y1, x2, y2)
        
        # Draw resolution bar
        bar_width = width - 40
        bar_height = 10
        painter.setBrush(QColor(50, 50, 70))
        painter.setPen(Qt.NoPen)
        painter.drawRect(20, height - 30, bar_width, bar_height)
        
        # Fill based on resolution
        resolution_gradient = QLinearGradient(0, 0, bar_width, 0)
        resolution_gradient.setColorAt(0, QColor(100, 100, 255))
        resolution_gradient.setColorAt(0.5, QColor(150, 100, 255))
        resolution_gradient.setColorAt(1, QColor(200, 100, 255))
        painter.setBrush(resolution_gradient)
        painter.drawRect(20, height - 30, bar_width * resolution, bar_height)
        
        # Resolution caption
        painter.setPen(Qt.white)
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(20, height - 10, "Paradox Resolution Status")


class FractalCanvas(QWidget):
    """Canvas for visualizing fractal patterns"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set up canvas properties
        self.setMinimumSize(500, 400)
        self.fractal_depth = 5
        self.iteration_factor = 0.5
        self.animation_speed = 15
        self.pattern_style = "mandelbrot"
        self.color_scheme = "spectral"
        
        # Animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_counter = 0
        self.animation_running = False
        
        # Fractal data
        self.fractal_nodes = []
        self.generate_fractal_nodes()
        
        # Start animation
        self.start_animation()
    
    def paintEvent(self, event):
        """Paint the fractal visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill background with dark gradient
        bg_gradient = QLinearGradient(0, 0, 0, self.height())
        bg_gradient.setColorAt(0, QColor(25, 25, 45))
        bg_gradient.setColorAt(1, QColor(10, 10, 20))
        painter.fillRect(self.rect(), QBrush(bg_gradient))
        
        # Draw fractal pattern based on selected style
        if self.pattern_style == "mandelbrot":
            self.draw_mandelbrot_style(painter)
        elif self.pattern_style == "julia":
            self.draw_julia_style(painter)
        elif self.pattern_style == "neural":
            self.draw_neural_style(painter)
        else:
            self.draw_tree_style(painter)
    
    def draw_mandelbrot_style(self, painter):
        """Draw a mandelbrot-like pattern visualization"""
        center_x = self.width() / 2
        center_y = self.height() / 2
        scale = min(self.width(), self.height()) * 0.4
        
        # Draw orbit traps with pulsing effect
        pulse = 0.5 + 0.5 * math.sin(self.animation_counter / 20)
        
        for i in range(self.fractal_depth):
            radius = scale * (0.3 + 0.7 * (i / self.fractal_depth))
            hue = (i * 40 + self.animation_counter) % 360
            
            for j in range(3 + i * 2):
                angle = 2 * math.pi * j / (3 + i * 2) + i * 0.2 + self.animation_counter / 50
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                
                # Draw orbit
                orbit_pen = QPen(QColor.fromHsv(hue, 200, 250, 100))
                orbit_pen.setWidth(2)
                painter.setPen(orbit_pen)
                
                orbit_radius = 30 + 15 * i + 10 * pulse
                painter.drawEllipse(QPointF(x, y), orbit_radius, orbit_radius)
                
                # Draw node
                node_color = QColor.fromHsv(hue, 255, 255)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(node_color))
                node_size = 8 + 4 * pulse
                painter.drawEllipse(QPointF(x, y), node_size, node_size)
    
    def draw_julia_style(self, painter):
        """Draw a julia-like pattern visualization"""
        center_x = self.width() / 2
        center_y = self.height() / 2
        scale = min(self.width(), self.height()) * 0.45
        
        # Draw spiraling paths
        path = QPainterPath()
        
        for i in range(8):
            angle_offset = 2 * math.pi * i / 8 + self.animation_counter / 100
            
            for j in range(100):
                t = j / 100
                spiral_factor = 1 + 3 * t
                angle = angle_offset + spiral_factor * 2 * math.pi * t
                radius = scale * t
                
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                
                if j == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            
            # Draw the path with gradient pen
            gradient = QLinearGradient(center_x, center_y, center_x + scale, center_y + scale)
            base_hue = (i * 45 + self.animation_counter) % 360
            gradient.setColorAt(0, QColor.fromHsv(base_hue, 255, 255, 200))
            gradient.setColorAt(1, QColor.fromHsv((base_hue + 100) % 360, 255, 255, 50))
            
            pen = QPen(QBrush(gradient), 2)
            painter.setPen(pen)
            painter.drawPath(path)
            path = QPainterPath()  # Reset path for next spiral
        
        # Draw center node
        painter.setPen(Qt.NoPen)
        pulse = 0.5 + 0.5 * math.sin(self.animation_counter / 15)
        center_size = 20 + 10 * pulse
        center_color = QColor.fromHsv((self.animation_counter * 2) % 360, 200, 255)
        painter.setBrush(QBrush(center_color))
        painter.drawEllipse(QPointF(center_x, center_y), center_size, center_size)
    
    def draw_neural_style(self, painter):
        """Draw a neural network-like fractal pattern"""
        # Draw connections between nodes
        for i, node in enumerate(self.fractal_nodes):
            # Determine number of connections based on depth
            num_connections = 2 + node["depth"]
            
            for j in range(num_connections):
                # Find a target node, preferably at a different depth
                target_idx = (i + 1 + j * 3) % len(self.fractal_nodes)
                target = self.fractal_nodes[target_idx]
                
                # Calculate connection strength based on pattern metrics
                connection_strength = 0.3 + 0.7 * abs(math.sin((node["depth"] + target["depth"]) * 0.3 + self.animation_counter / 30))
                
                # Set connection color based on strength and depth difference
                if self.color_scheme == "spectral":
                    hue = (node["depth"] * 30 + self.animation_counter) % 360
                    connection_color = QColor.fromHsv(hue, 200, 255, int(200 * connection_strength))
                else:
                    connection_color = QColor(100, 180, 255, int(200 * connection_strength))
                
                # Draw the connection
                pen = QPen(connection_color)
                pen.setWidth(1 + int(3 * connection_strength))
                painter.setPen(pen)
                
                # Draw curved lines for more interesting visuals
                path = QPainterPath()
                path.moveTo(node["x"], node["y"])
                
                # Control points for curve
                mid_x = (node["x"] + target["x"]) / 2
                mid_y = (node["y"] + target["y"]) / 2
                offset = 30 * (1 + (i % 3)) * math.sin(self.animation_counter / 50 + i * 0.1)
                
                if i % 2 == 0:
                    control1 = QPointF(mid_x - offset, mid_y + offset)
                    control2 = QPointF(mid_x + offset, mid_y - offset)
                else:
                    control1 = QPointF(mid_x + offset, mid_y + offset)
                    control2 = QPointF(mid_x - offset, mid_y - offset)
                
                path.cubicTo(control1, control2, QPointF(target["x"], target["y"]))
                painter.drawPath(path)
            
        # Draw nodes
        for node in self.fractal_nodes:
            pulse = 0.5 + 0.5 * math.sin(self.animation_counter / 20 + node["depth"] * 0.5)
            
            # Node color based on depth and color scheme
            if self.color_scheme == "spectral":
                hue = (node["depth"] * 40 + self.animation_counter) % 360
                node_color = QColor.fromHsv(hue, 220, 255)
            else:
                intensity = 150 + 105 * node["depth"] / self.fractal_depth
                node_color = QColor(100, intensity, 255)
            
            # Node size based on depth and pulse
            size = 4 + 3 * (self.fractal_depth - node["depth"]) + 2 * pulse
            
            # Draw node glow first (larger, semi-transparent circle)
            glow_size = size * 2.5
            glow_color = QColor(node_color)
            glow_color.setAlpha(50)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(glow_color))
            painter.drawEllipse(QPointF(node["x"], node["y"]), glow_size, glow_size)
            
            # Draw the node itself
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(node_color))
            painter.drawEllipse(QPointF(node["x"], node["y"]), size, size)
    
    def draw_tree_style(self, painter):
        """Draw a recursive tree-like pattern"""
        painter.setPen(Qt.NoPen)
        
        # Start from bottom center
        start_x = self.width() / 2
        start_y = self.height() * 0.9
        length = self.height() * 0.4
        angle = -math.pi / 2  # Straight up
        depth = 0
        
        # Draw the fractal tree
        self.draw_branch(painter, start_x, start_y, length, angle, depth)
    
    def draw_branch(self, painter, x, y, length, angle, depth):
        """Recursively draw branches of the tree"""
        if depth >= self.fractal_depth:
            return
        
        # Calculate end point
        end_x = x + length * math.cos(angle)
        end_y = y + length * math.sin(angle)
        
        # Determine branch color based on depth and scheme
        if self.color_scheme == "spectral":
            hue = (depth * 30 + self.animation_counter) % 360
            branch_color = QColor.fromHsv(hue, 200, 255)
        else:
            # Blue-white gradient
            intensity = 100 + 155 * depth / self.fractal_depth
            branch_color = QColor(100, intensity, 255)
        
        # Draw the branch
        pen = QPen(branch_color)
        pulse = 0.5 + 0.5 * math.sin(self.animation_counter / 20 + depth)
        pen.setWidth(int(8 - depth * 1.5 + pulse))
        painter.setPen(pen)
        painter.drawLine(int(x), int(y), int(end_x), int(end_y))
        
        # Draw a node at the end
        node_size = 5 - depth * 0.8 + pulse
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(branch_color))
        painter.drawEllipse(QPointF(end_x, end_y), node_size, node_size)
        
        # Recursively draw branches
        new_length = length * (0.7 - 0.05 * depth)
        
        # Variation based on animation counter for breathing effect
        angle_variation = 0.1 * math.sin(self.animation_counter / 30)
        
        # Left branch
        left_angle = angle - (0.4 + angle_variation + 0.05 * depth)
        self.draw_branch(painter, end_x, end_y, new_length, left_angle, depth + 1)
        
        # Right branch
        right_angle = angle + (0.4 - angle_variation + 0.05 * depth)
        self.draw_branch(painter, end_x, end_y, new_length, right_angle, depth + 1)
        
        # Middle branch (adding occasionally)
        if depth < 2 or random.random() < 0.7:
            middle_angle = angle + (angle_variation * 0.5)
            self.draw_branch(painter, end_x, end_y, new_length, middle_angle, depth + 1)
    
    def generate_fractal_nodes(self):
        """Generate nodes for fractal visualization"""
        self.fractal_nodes = []
        
        # Create nodes at different depths
        for depth in range(self.fractal_depth):
            # Number of nodes increases with depth
            num_nodes = 5 + depth * 4
            
            for i in range(num_nodes):
                # Position nodes in a spiral pattern
                angle = 2 * math.pi * i / num_nodes + depth * 0.5
                radius = 50 + depth * 40
                
                # Add some randomness to positions
                radius_jitter = random.uniform(0.8, 1.2) * radius
                angle_jitter = angle + random.uniform(-0.1, 0.1)
                
                center_x = self.width() / 2
                center_y = self.height() / 2
                
                x = center_x + radius_jitter * math.cos(angle_jitter)
                y = center_y + radius_jitter * math.sin(angle_jitter)
                
                # Add node to the list
                self.fractal_nodes.append({
                    "x": x,
                    "y": y,
                    "depth": depth,
                    "angle": angle,
                    "radius": radius
                })
    
    def update_animation(self):
        """Update animation counters and repaint"""
        self.animation_counter += self.animation_speed / 10
        
        # Occasionally update node positions slightly
        if self.animation_counter % 30 < 1:
            self.update_node_positions()
            
        self.update()
    
    def update_node_positions(self):
        """Slightly update node positions for organic movement"""
        for node in self.fractal_nodes:
            # Add small jitter to positions
            node["x"] += random.uniform(-2, 2)
            node["y"] += random.uniform(-2, 2)
            
            # Keep within bounds
            margin = 20
            node["x"] = max(margin, min(self.width() - margin, node["x"]))
            node["y"] = max(margin, min(self.height() - margin, node["y"]))
    
    def set_pattern_style(self, style):
        """Set the pattern style"""
        self.pattern_style = style
        self.update()
    
    def set_color_scheme(self, scheme):
        """Set the color scheme"""
        self.color_scheme = scheme
        self.update()
    
    def set_fractal_depth(self, depth):
        """Set the fractal depth"""
        self.fractal_depth = depth
        self.generate_fractal_nodes()
        self.update()
    
    def set_animation_speed(self, speed):
        """Set animation speed"""
        self.animation_speed = speed
        
        # Update timer interval if running
        if self.animation_running:
            self.animation_timer.setInterval(max(5, int(50 - speed * 3)))
    
    def start_animation(self):
        """Start the animation timer"""
        if not self.animation_running:
            self.animation_timer.start(max(5, int(50 - self.animation_speed * 3)))
            self.animation_running = True
    
    def stop_animation(self):
        """Stop the animation timer"""
        if self.animation_running:
            self.animation_timer.stop()
            self.animation_running = False
    
    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        self.generate_fractal_nodes()


class PatternMetricsWidget(QWidget):
    """Widget to display pattern metrics and insights"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
        # Update metrics periodically with slightly changing values
        self.metrics_timer = QTimer(self)
        self.metrics_timer.timeout.connect(self.update_metrics)
        self.metrics_timer.start(2000)  # Update every 2 seconds
        
        # Initial metrics update
        self.update_metrics()
    
    def initUI(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Pattern Metrics")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #FFFFFF;")
        layout.addWidget(title)
        
        # Create metrics groups
        self.create_complexity_metrics(layout)
        self.create_pattern_insights(layout)
        
        # Add stretch for proper layout
        layout.addStretch()
    
    def create_complexity_metrics(self, layout):
        """Create the complexity metrics group"""
        # Metrics group box
        metrics_group = QGroupBox("Complexity Analysis")
        metrics_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #3A3A5A;
                border-radius: 6px;
                margin-top: 12px;
                font-weight: bold;
                color: #BBBBBB;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
        """)
        
        metrics_layout = QVBoxLayout(metrics_group)
        metrics_layout.setSpacing(10)
        
        # Create metric labels
        fractal_dimension_layout, self.fractal_dimension_value = self.create_metric_item("Fractal Dimension", "1.68")
        complexity_index_layout, self.complexity_index_value = self.create_metric_item("Complexity Index", "78%")
        pattern_coherence_layout, self.pattern_coherence_value = self.create_metric_item("Pattern Coherence", "92%")
        entropy_level_layout, self.entropy_level_value = self.create_metric_item("Entropy Level", "Medium")
        
        # Add metrics to layout
        metrics_layout.addLayout(fractal_dimension_layout)
        metrics_layout.addLayout(complexity_index_layout)
        metrics_layout.addLayout(pattern_coherence_layout)
        metrics_layout.addLayout(entropy_level_layout)
        
        layout.addWidget(metrics_group)
    
    def create_pattern_insights(self, layout):
        """Create the pattern insights group"""
        # Insights group box
        insights_group = QGroupBox("Pattern Insights")
        insights_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #3A3A5A;
                border-radius: 6px;
                margin-top: 12px;
                font-weight: bold;
                color: #BBBBBB;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
        """)
        
        insights_layout = QVBoxLayout(insights_group)
        
        # Detected patterns section
        patterns_title = QLabel("Detected Patterns:")
        patterns_title.setStyleSheet("color: #CCCCCC;")
        insights_layout.addWidget(patterns_title)
        
        # Pattern list
        self.patterns_list = QLabel(
            "• Recursive symmetry at 3 levels\n"
            "• Bifurcation sequences detected\n"
            "• Self-similarity coefficient: High\n"
            "• Scale invariance observed"
        )
        self.patterns_list.setStyleSheet("color: #A0A0D0; padding-left: 10px;")
        self.patterns_list.setWordWrap(True)
        insights_layout.addWidget(self.patterns_list)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #3A3A5A;")
        insights_layout.addWidget(separator)
        
        # Recommendations
        recommendations_title = QLabel("Optimal Processing Parameters:")
        recommendations_title.setStyleSheet("color: #CCCCCC; margin-top: 5px;")
        insights_layout.addWidget(recommendations_title)
        
        self.recommendations = QLabel(
            "• Recursion depth: 6-8 levels\n"
            "• Integration threshold: 0.42\n"
            "• Pattern weight: Logarithmic\n"
            "• Neural binding: Moderate"
        )
        self.recommendations.setStyleSheet("color: #A0D0A0; padding-left: 10px;")
        self.recommendations.setWordWrap(True)
        insights_layout.addWidget(self.recommendations)
        
        layout.addWidget(insights_group)
    
    def create_metric_item(self, label_text, value_text):
        """Create a metric item with label and value"""
        layout = QHBoxLayout()
        
        label = QLabel(label_text)
        label.setStyleSheet("color: #CCCCCC;")
        
        value = QLabel(value_text)
        value.setStyleSheet("color: #77AAFF; font-weight: bold;")
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(value)
        
        return layout, value
    
    def update_metrics(self):
        """Update metrics with slightly varying values"""
        # Update fractal dimension (slight variations)
        dimension = 1.65 + random.uniform(-0.05, 0.05)
        self.fractal_dimension_value.setText(f"{dimension:.2f}")
        
        # Update complexity index with small changes
        complexity = 75 + random.randint(-3, 3)
        self.complexity_index_value.setText(f"{complexity}%")
        
        # Update pattern coherence
        coherence = 90 + random.randint(-2, 2)
        self.pattern_coherence_value.setText(f"{coherence}%")
        
        # Update entropy level
        entropy_values = ["Low", "Medium-Low", "Medium", "Medium-High", "High"]
        entropy_level = entropy_values[random.randint(1, 3)]  # Mostly stay in the middle range
        self.entropy_level_value.setText(entropy_level)


class FractalPatternPanel(QWidget):
    """Panel for visualizing and analyzing fractal patterns"""
    
    pattern_updated = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        """Initialize the user interface"""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        header = QWidget()
        header.setFixedHeight(60)
        header.setStyleSheet("background-color: #1A1A35;")
        
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 0, 20, 0)
        
        title = QLabel("Fractal Pattern Processing")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFFFFF;")
        
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        # Capture button
        capture_btn = QPushButton("Capture Pattern")
        capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #5050A0;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #6060C0;
            }
        """)
        header_layout.addWidget(capture_btn)
        
        main_layout.addWidget(header)
        
        # Content splitter
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setHandleWidth(1)
        content_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #3A3A5A;
            }
        """)
        
        # Left panel - Fractal canvas and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)
        left_layout.setSpacing(15)
        
        # Fractal canvas
        self.fractal_canvas = FractalCanvas()
        left_layout.addWidget(self.fractal_canvas, 1)
        
        # Controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(20)
        
        # Pattern style selector
        style_layout = QVBoxLayout()
        style_label = QLabel("Pattern Style:")
        style_label.setStyleSheet("color: #BBBBBB;")
        self.style_selector = QComboBox()
        self.style_selector.addItems(["mandelbrot", "julia", "neural", "tree"])
        self.style_selector.setStyleSheet("""
            QComboBox {
                background-color: #252545;
                border: 1px solid #3A3A5A;
                border-radius: 4px;
                color: #DDDDDD;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        self.style_selector.currentTextChanged.connect(self.update_pattern_style)
        style_layout.addWidget(style_label)
        style_layout.addWidget(self.style_selector)
        controls_layout.addLayout(style_layout)
        
        # Color scheme selector
        color_layout = QVBoxLayout()
        color_label = QLabel("Color Scheme:")
        color_label.setStyleSheet("color: #BBBBBB;")
        self.color_selector = QComboBox()
        self.color_selector.addItems(["spectral", "blue", "green", "purple"])
        self.color_selector.setStyleSheet("""
            QComboBox {
                background-color: #252545;
                border: 1px solid #3A3A5A;
                border-radius: 4px;
                color: #DDDDDD;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        self.color_selector.currentTextChanged.connect(self.update_color_scheme)
        color_layout.addWidget(color_label)
        color_layout.addWidget(self.color_selector)
        controls_layout.addLayout(color_layout)
        
        # Depth slider
        depth_layout = QVBoxLayout()
        depth_label = QLabel("Recursion Depth:")
        depth_label.setStyleSheet("color: #BBBBBB;")
        self.depth_slider = QSlider(Qt.Horizontal)
        self.depth_slider.setMinimum(3)
        self.depth_slider.setMaximum(8)
        self.depth_slider.setValue(5)
        self.depth_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #2A2A45;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #6070C0;
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
        """)
        self.depth_slider.valueChanged.connect(self.update_fractal_depth)
        depth_layout.addWidget(depth_label)
        depth_layout.addWidget(self.depth_slider)
        controls_layout.addLayout(depth_layout)
        
        # Animation speed slider
        speed_layout = QVBoxLayout()
        speed_label = QLabel("Animation Speed:")
        speed_label.setStyleSheet("color: #BBBBBB;")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(5)
        self.speed_slider.setMaximum(30)
        self.speed_slider.setValue(15)
        self.speed_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #2A2A45;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #6070C0;
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
        """)
        self.speed_slider.valueChanged.connect(self.update_animation_speed)
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_slider)
        controls_layout.addLayout(speed_layout)
        
        left_layout.addLayout(controls_layout)
        
        # Right panel - Metrics and status
        right_panel = QWidget()
        right_panel.setFixedWidth(300)
        right_panel.setStyleSheet("""
            background-color: #1D1D38;
            border-left: 1px solid #3A3A5A;
        """)
        
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        
        # Pattern metrics
        self.metrics_widget = PatternMetricsWidget()
        
        # Wrap in scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.metrics_widget)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background: #1D1D38;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #3A3A5A;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                background: none;
            }
        """)
        
        right_layout.addWidget(scroll_area)
        
        # Add panels to splitter
        content_splitter.addWidget(left_panel)
        content_splitter.addWidget(right_panel)
        content_splitter.setSizes([700, 300])
        
        main_layout.addWidget(content_splitter)
        
        # Connect signals
        capture_btn.clicked.connect(self.capture_pattern)
    
    def update_pattern_style(self, style):
        """Update the pattern style in the canvas"""
        self.fractal_canvas.set_pattern_style(style)
        self.pattern_updated.emit()
    
    def update_color_scheme(self, scheme):
        """Update the color scheme in the canvas"""
        self.fractal_canvas.set_color_scheme(scheme)
        self.pattern_updated.emit()
    
    def update_fractal_depth(self, depth):
        """Update the fractal depth in the canvas"""
        self.fractal_canvas.set_fractal_depth(depth)
        self.pattern_updated.emit()
    
    def update_animation_speed(self, speed):
        """Update the animation speed in the canvas"""
        self.fractal_canvas.set_animation_speed(speed)
    
    def capture_pattern(self):
        """Capture the current pattern (mock implementation)"""
        # In a real implementation, this would save the pattern or integrate with other systems
        print("Pattern captured for neural integration") 