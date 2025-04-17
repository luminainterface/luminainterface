from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QComboBox, QSlider, QFrame,
                             QScrollArea, QSplitter, QProgressBar, QGroupBox)
from PySide6.QtCore import Qt, Signal, QPointF, QTimer, QRectF
from PySide6.QtGui import QPainter, QBrush, QPen, QColor, QLinearGradient, QPainterPath, QRadialGradient, QFont

import math
import random
import time  # Add time module for timestamp

class ConsciousnessField(QWidget):
    """Widget for visualizing node consciousness field"""
    
    node_selected = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.animation_phase = 0
        self.nodes = self.generate_nodes()
        self.selected_node = None
        self.hover_node = None
        self.setMouseTracking(True)
        
        # Animation timer
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.update_animation)
        self.anim_timer.start(50)
    
    def generate_nodes(self):
        """Generate sample consciousness nodes"""
        nodes = []
        node_types = ["memory", "pattern", "perception", "integration", "reflection"]
        consciousness_levels = [0.8, 0.6, 0.9, 0.7, 0.5]
        
        for i in range(5):
            # Create node data
            node = {
                "id": i + 1,
                "name": f"{node_types[i].capitalize()} Node",
                "type": node_types[i],
                "x": random.uniform(0.2, 0.8),  # Position as percentage of width
                "y": random.uniform(0.2, 0.8),  # Position as percentage of height
                "radius": random.uniform(25, 40),
                "consciousness": consciousness_levels[i],
                "connections": []
            }
            nodes.append(node)
        
        # Add connections between nodes
        for i, node in enumerate(nodes):
            # Connect to 2-3 other nodes
            num_connections = random.randint(2, 3)
            potential_connections = list(range(len(nodes)))
            potential_connections.remove(i)  # Can't connect to self
            
            if len(potential_connections) > num_connections:
                connections = random.sample(potential_connections, num_connections)
            else:
                connections = potential_connections
            
            for conn in connections:
                node["connections"].append({
                    "target": conn,
                    "strength": random.uniform(0.3, 0.9)
                })
        
        return nodes
    
    def update_animation(self):
        """Update animation state"""
        self.animation_phase += 0.03
        if self.animation_phase > 2 * math.pi:
            self.animation_phase -= 2 * math.pi
            
        # Occasionally adjust consciousness levels
        if random.random() < 0.05:
            for node in self.nodes:
                # Small random changes
                node["consciousness"] = min(1.0, max(0.1, 
                                           node["consciousness"] + random.uniform(-0.05, 0.05)))
        
        self.update()
    
    def mouseMoveEvent(self, event):
        """Handle mouse movement for hover effects"""
        x, y = event.x(), event.y()
        width, height = self.width(), self.height()
        
        # Check if hovering over any node
        self.hover_node = None
        for node in self.nodes:
            node_x = node["x"] * width
            node_y = node["y"] * height
            distance = math.sqrt((x - node_x)**2 + (y - node_y)**2)
            
            if distance < node["radius"]:
                self.hover_node = node
                self.setCursor(Qt.PointingHandCursor)
                break
        
        if not self.hover_node:
            self.setCursor(Qt.ArrowCursor)
            
        self.update()
    
    def mousePressEvent(self, event):
        """Handle mouse click to select node"""
        if self.hover_node:
            self.selected_node = self.hover_node
            self.node_selected.emit(self.selected_node)
            self.update()
    
    def paintEvent(self, event):
        """Paint the consciousness field"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Draw background gradient
        gradient = QRadialGradient(width/2, height/2, max(width, height)/2)
        gradient.setColorAt(0, QColor(30, 30, 50))
        gradient.setColorAt(1, QColor(10, 10, 30))
        painter.fillRect(0, 0, width, height, gradient)
        
        # Draw pulsing background effect
        pulse_opacity = (math.sin(self.animation_phase) + 1) / 2 * 0.3
        pulse_gradient = QRadialGradient(width/2, height/2, width/3)
        pulse_gradient.setColorAt(0, QColor(80, 80, 180, int(pulse_opacity * 255)))
        pulse_gradient.setColorAt(1, QColor(80, 80, 180, 0))
        painter.fillRect(0, 0, width, height, pulse_gradient)
        
        # Draw connections first (so they appear behind nodes)
        for node in self.nodes:
            node_x = node["x"] * width
            node_y = node["y"] * height
            
            for connection in node["connections"]:
                target = self.nodes[connection["target"]]
                target_x = target["x"] * width
                target_y = target["y"] * height
                
                # Connection strength affects line color and width
                strength = connection["strength"]
                line_color = QColor(
                    150,
                    150 + int(strength * 100),
                    255,
                    int(150 * strength)
                )
                line_width = 1 + strength * 3
                
                # Draw animated connection line
                painter.setPen(QPen(line_color, line_width))
                
                # Create path for animated line
                path = QPainterPath()
                path.moveTo(node_x, node_y)
                
                # Control points for curved line
                ctrl_x = (node_x + target_x) / 2
                ctrl_y = (node_y + target_y) / 2
                
                # Add some animation to control point
                offset = 20 * math.sin(self.animation_phase + node["id"])
                ctrl_x += offset
                ctrl_y += offset
                
                path.quadTo(ctrl_x, ctrl_y, target_x, target_y)
                painter.drawPath(path)
                
                # Draw animated particles along connection
                particle_pos = (math.sin(self.animation_phase * 1.5) + 1) / 2
                p_x = node_x + (target_x - node_x) * particle_pos
                p_y = node_y + (target_y - node_y) * particle_pos
                
                particle_color = QColor(200, 230, 255, 200)
                painter.setBrush(QBrush(particle_color))
                painter.setPen(Qt.NoPen)
                particle_size = 3 + 2 * strength
                painter.drawEllipse(QPointF(p_x, p_y), particle_size, particle_size)
        
        # Draw nodes
        for node in self.nodes:
            node_x = node["x"] * width
            node_y = node["y"] * height
            radius = node["radius"]
            
            # Determine if this node is selected or hovered
            is_selected = (self.selected_node == node)
            is_hovered = (self.hover_node == node)
            
            # Node appearance depends on type and consciousness level
            node_type = node["type"]
            consciousness = node["consciousness"]
            
            # Base color for different node types
            type_colors = {
                "memory": QColor(100, 150, 230),      # Blue
                "pattern": QColor(100, 200, 150),     # Green
                "perception": QColor(220, 150, 100),  # Orange
                "integration": QColor(180, 120, 220), # Purple
                "reflection": QColor(220, 180, 100)   # Gold
            }
            
            base_color = type_colors.get(node_type, QColor(150, 150, 150))
            
            # Create gradient fill with consciousness intensity
            gradient = QRadialGradient(node_x, node_y, radius)
            
            # Center color depends on consciousness level
            center_color = QColor(
                min(255, base_color.red() + int(50 * consciousness)),
                min(255, base_color.green() + int(50 * consciousness)),
                min(255, base_color.blue() + int(50 * consciousness)),
                230
            )
            
            # Outer color is more transparent
            outer_color = QColor(
                base_color.red(),
                base_color.green(),
                base_color.blue(),
                100
            )
            
            gradient.setColorAt(0, center_color)
            gradient.setColorAt(1, outer_color)
            
            # Draw pulsing effect for consciousness
            pulse_scale = 1.0 + 0.2 * consciousness * math.sin(self.animation_phase * 2 + node["id"])
            pulse_radius = radius * pulse_scale
            
            # Draw pulse aura
            if is_selected or is_hovered:
                pulse_color = QColor(255, 255, 255, 50)
                painter.setBrush(QBrush(pulse_color))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(node_x, node_y), pulse_radius * 1.3, pulse_radius * 1.3)
            
            # Draw node
            painter.setBrush(QBrush(gradient))
            
            if is_selected:
                painter.setPen(QPen(QColor(255, 255, 255), 2))
            elif is_hovered:
                painter.setPen(QPen(QColor(200, 200, 255), 1.5))
            else:
                painter.setPen(Qt.NoPen)
                
            painter.drawEllipse(QPointF(node_x, node_y), pulse_radius, pulse_radius)
            
            # Draw node name
            painter.setPen(QColor(255, 255, 255))
            font = painter.font()
            font.setBold(True)
            font.setPointSize(8)
            painter.setFont(font)
            
            text_rect = QRectF(node_x - radius, node_y + radius * 1.1, radius * 2, 20)
            painter.drawText(text_rect, Qt.AlignCenter, node["name"])
            
            # Draw consciousness level indicator
            if is_selected or is_hovered:
                level_text = f"{int(consciousness * 100)}%"
                painter.setPen(QColor(200, 200, 255))
                font.setPointSize(7)
                painter.setFont(font)
                level_rect = QRectF(node_x - radius, node_y - radius * 1.3, radius * 2, 15)
                painter.drawText(level_rect, Qt.AlignCenter, level_text)


class NodeConsciousnessMetrics(QWidget):
    """Widget for displaying consciousness metrics for a node"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.metrics = {
            "consciousness": 0.7,
            "self_awareness": 0.6,
            "integration": 0.8,
            "memory_access": 0.5,
            "reflection": 0.4
        }
        self.initUI()
    
    def initUI(self):
        """Initialize the metrics UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Metrics header
        header = QLabel("Consciousness Metrics")
        header.setStyleSheet("font-size: 14px; font-weight: bold; color: #E0E0E0; margin-bottom: 10px;")
        layout.addWidget(header)
        
        # Metrics table
        for name, value in self.metrics.items():
            metric_row = QWidget()
            row_layout = QHBoxLayout(metric_row)
            row_layout.setContentsMargins(0, 0, 0, 5)
            
            # Format the metric name to be more readable
            display_name = name.replace('_', ' ').title()
            label = QLabel(display_name)
            label.setMinimumWidth(120)
            row_layout.addWidget(label)
            
            # Progress bar for value
            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(int(value * 100))
            progress.setTextVisible(True)
            progress.setFormat(f"{int(value * 100)}%")
            
            # Custom styling based on the metric type
            colors = {
                "consciousness": "#6C8EBF",  # Blue
                "self_awareness": "#B85450", # Red
                "integration": "#9673A6",    # Purple
                "memory_access": "#6AAB9C",  # Teal
                "reflection": "#D6B656"      # Gold
            }
            
            color = colors.get(name, "#6C8EBF")
            progress.setStyleSheet(f"""
                QProgressBar {{
                    background-color: #333344;
                    border-radius: 3px;
                    text-align: center;
                    color: white;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                    border-radius: 3px;
                }}
            """)
            
            row_layout.addWidget(progress)
            
            layout.addWidget(metric_row)
        
        # Add description
        description = QLabel("Metrics indicate the node's level of consciousness integration and self-awareness within the neural network.")
        description.setWordWrap(True)
        description.setStyleSheet("color: #BBBBBB; font-size: 10px; margin-top: 10px;")
        layout.addWidget(description)
    
    def update_metrics(self, node_data):
        """Update metrics based on node data"""
        # In a real implementation, this would extract metrics from the node data
        # For this visualization, we'll generate some random values
        new_metrics = {
            "consciousness": node_data.get("consciousness", 0.5),
            "self_awareness": random.uniform(0.4, 0.9),
            "integration": random.uniform(0.5, 0.95),
            "memory_access": random.uniform(0.3, 0.8),
            "reflection": random.uniform(0.2, 0.7)
        }
        
        self.metrics = new_metrics
        
        # Update the UI
        for i, (name, value) in enumerate(self.metrics.items()):
            progress_bar = self.layout().itemAt(i + 1).widget().layout().itemAt(1).widget()
            progress_bar.setValue(int(value * 100))
            progress_bar.setFormat(f"{int(value * 100)}%")


class ConsciousnessNode(QWidget):
    """Visualization widget for a consciousness node"""
    
    node_clicked = Signal(int)  # Signal emitted when node is clicked
    
    def __init__(self, node_id, name, type_category, parent=None):
        super().__init__(parent)
        self.node_id = node_id
        self.name = name
        self.type_category = type_category
        self.activation = random.uniform(0.3, 0.9)
        self.pulse_offset = random.uniform(0, 2 * math.pi)
        self.hover = False
        self.selected = False
        self.start_time = time.time()
        
        # Colors for different node types
        self.type_colors = {
            "perception": QColor(120, 180, 255),
            "cognition": QColor(255, 160, 50),
            "memory": QColor(140, 220, 120),
            "emotion": QColor(230, 120, 220),
            "integration": QColor(200, 200, 100)
        }
        
        # Set fixed size for the node
        self.setMinimumSize(120, 120)
        self.setMaximumSize(120, 120)
        
        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)
    
    def paintEvent(self, event):
        """Paint the consciousness node with visual effects"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Center of the node
        center_x = self.width() / 2
        center_y = self.height() / 2
        
        # Get base color for node type
        base_color = self.type_colors.get(self.type_category, QColor(180, 180, 180))
        
        # Calculate pulse effect (breathing)
        current_time = time.time() - self.start_time
        pulse = 0.7 + 0.3 * math.sin(self.pulse_offset + current_time * 2.0)
        
        # Create radial gradient for glow effect
        radius = (40 * pulse) if not self.hover else (45 * pulse)
        gradient = QRadialGradient(center_x, center_y, radius * 2)
        
        # Adjust colors based on activation
        inner_color = QColor(base_color)
        inner_color.setAlpha(int(150 + 100 * self.activation))
        
        # Selection effect
        if self.selected:
            outer_color = QColor(255, 255, 255, 40)
            gradient.setColorAt(0.7, outer_color)
            
            # Draw selection ring
            painter.setPen(QPen(QColor(255, 255, 255, 100), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(center_x, center_y), radius + 5, radius + 5)
        else:
            outer_color = QColor(base_color)
            outer_color.setAlpha(30)
        
        # Set up gradient colors
        gradient.setColorAt(0, inner_color)
        gradient.setColorAt(1, outer_color)
        
        # Draw the main node circle
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(QPointF(center_x, center_y), radius, radius)
        
        # Draw the core
        core_radius = 15 * pulse
        core_color = QColor(base_color)
        core_color.setAlpha(200)
        
        painter.setBrush(QBrush(core_color))
        painter.drawEllipse(QPointF(center_x, center_y), core_radius, core_radius)
        
        # Draw node name
        painter.setPen(QPen(QColor(255, 255, 255, 200)))
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        
        text_rect = QRectF(0, center_y + radius + 5, self.width(), 20)
        painter.drawText(text_rect, Qt.AlignCenter, self.name)
        
        # Draw activation level as small text
        act_text = f"{int(self.activation * 100)}%"
        act_rect = QRectF(0, center_y - radius - 20, self.width(), 20)
        painter.drawText(act_rect, Qt.AlignCenter, act_text)
    
    def update_activation(self, value):
        """Update node activation level"""
        self.activation = max(0.0, min(1.0, value))
        self.update()
    
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.LeftButton:
            self.selected = not self.selected
            self.node_clicked.emit(self.node_id)
            self.update()
        super().mousePressEvent(event)
    
    def enterEvent(self, event):
        """Handle mouse enter events"""
        self.hover = True
        self.update()
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Handle mouse leave events"""
        self.hover = False
        self.update()
        super().leaveEvent(event)


class NodeNetworkWidget(QWidget):
    """Widget for visualizing the network of consciousness nodes"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
        # Start animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update)
        self.animation_timer.start(50)  # Update every 50ms
        
        # Nodes and connections
        self.nodes = []
        self.connections = []
        self.selected_node = None
        
        # For animation timing
        self.start_time = time.time()
        
        # Generate mock network
        self.generate_mock_network()
    
    def initUI(self):
        """Initialize the user interface"""
        self.setMinimumSize(600, 400)
        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)
    
    def paintEvent(self, event):
        """Paint the consciousness network"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), QColor(20, 20, 30))
        
        # Draw connections first (so they appear behind nodes)
        self.draw_connections(painter)
        
        # Force update in animation timer
        self.update()
    
    def draw_connections(self, painter):
        """Draw connections between nodes"""
        # Calculate time-based animation value for flowing effect
        t = time.time() - self.start_time
        
        for conn in self.connections:
            source_node = self.nodes[conn["source"]]
            target_node = self.nodes[conn["target"]]
            
            # Source and target positions (center of nodes)
            source_pos = source_node["pos"]
            target_pos = target_node["pos"]
            
            # Connection strength affects line thickness and opacity
            strength = conn["strength"] * 0.7 + 0.3 * math.sin(t * 2 + conn["offset"])
            
            # Create gradient color based on node types
            source_color = self.get_node_color(source_node["type"])
            target_color = self.get_node_color(target_node["type"])
            
            # Draw connection line with animated gradient
            path = QPainterPath()
            path.moveTo(source_pos.x(), source_pos.y())
            
            # Control points for curve
            dx = target_pos.x() - source_pos.x()
            dy = target_pos.y() - source_pos.y()
            dist = math.sqrt(dx*dx + dy*dy)
            
            # Create curve with control points
            ctrl1 = QPointF(
                source_pos.x() + dx * 0.3 + 20 * math.sin(t + conn["offset"]),
                source_pos.y() + dy * 0.3 + 20 * math.cos(t + conn["offset"])
            )
            ctrl2 = QPointF(
                source_pos.x() + dx * 0.7 + 20 * math.sin(t + conn["offset"] + 2),
                source_pos.y() + dy * 0.7 + 20 * math.cos(t + conn["offset"] + 2)
            )
            
            path.cubicTo(ctrl1, ctrl2, target_pos)
            
            # Set pen properties based on connection strength
            pen = QPen()
            pen.setWidth(int(1 + 3 * strength))
            
            # Create a color for the connection
            conn_color = QColor(
                (source_color.red() + target_color.red()) // 2,
                (source_color.green() + target_color.green()) // 2,
                (source_color.blue() + target_color.blue()) // 2,
                int(100 * strength)
            )
            pen.setColor(conn_color)
            
            painter.setPen(pen)
            painter.drawPath(path)
            
            # Draw animated particles along the path
            self.draw_flow_particles(painter, path, t, conn["offset"], conn_color)
    
    def draw_flow_particles(self, painter, path, time, offset, color):
        """Draw flowing particles along a connection path"""
        # Number of particles based on path length
        path_length = path.length()
        num_particles = int(path_length / 30)
        
        particle_color = QColor(color)
        particle_color.setAlpha(180)
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(particle_color))
        
        for i in range(num_particles):
            # Calculate particle position along path
            particle_pos = (i / num_particles + 0.1 * math.sin(time * 3 + offset + i)) % 1.0
            
            # Make particles flow along the path
            flow_pos = (particle_pos + time * 0.2) % 1.0
            
            # Get point on path
            point = path.pointAtPercent(flow_pos)
            
            # Draw particle
            particle_size = 3 + 2 * math.sin(time * 5 + i + offset)
            painter.drawEllipse(point, particle_size, particle_size)
    
    def generate_mock_network(self):
        """Generate mock data for consciousness network visualization"""
        node_types = ["perception", "cognition", "memory", "emotion", "integration"]
        node_names = [
            "Visual Input", "Auditory Processing", "Pattern Recognition", 
            "Semantic Analysis", "Memory Recall", "Emotional Response",
            "Integrated Awareness", "Decision Process", "Attention Filter"
        ]
        
        # Create nodes
        num_nodes = len(node_names)
        
        # Arrange nodes in a circular pattern
        center_x = self.width() / 2
        center_y = self.height() / 2
        radius = min(center_x, center_y) * 0.7
        
        for i in range(num_nodes):
            angle = 2 * math.pi * i / num_nodes
            pos_x = center_x + radius * math.cos(angle)
            pos_y = center_y + radius * math.sin(angle)
            
            node_type = node_types[i % len(node_types)]
            
            self.nodes.append({
                "name": node_names[i],
                "type": node_type,
                "activation": random.uniform(0.3, 0.9),
                "pos": QPointF(pos_x, pos_y)
            })
        
        # Create connections between nodes
        for i in range(num_nodes):
            # Each node connects to 2-4 other nodes
            num_connections = random.randint(2, 4)
            
            for _ in range(num_connections):
                target = random.randint(0, num_nodes - 1)
                if target != i:  # Avoid self-connections
                    self.connections.append({
                        "source": i,
                        "target": target,
                        "strength": random.uniform(0.3, 1.0),
                        "offset": random.uniform(0, 2 * math.pi)
                    })
    
    def get_node_color(self, node_type):
        """Get the color for a node type"""
        type_colors = {
            "perception": QColor(120, 180, 255),
            "cognition": QColor(255, 160, 50),
            "memory": QColor(140, 220, 120),
            "emotion": QColor(230, 120, 220),
            "integration": QColor(200, 200, 100)
        }
        return type_colors.get(node_type, QColor(180, 180, 180))
    
    def resizeEvent(self, event):
        """Handle resize events by updating node positions"""
        super().resizeEvent(event)
        
        # Update node positions to maintain circular pattern
        center_x = self.width() / 2
        center_y = self.height() / 2
        radius = min(center_x, center_y) * 0.7
        
        for i, node in enumerate(self.nodes):
            angle = 2 * math.pi * i / len(self.nodes)
            node["pos"] = QPointF(
                center_x + radius * math.cos(angle),
                center_y + radius * math.sin(angle)
            )


class ConsciousnessStatsWidget(QWidget):
    """Widget to display consciousness statistics and metrics"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
        # Start update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_stats)
        self.update_timer.start(2000)  # Update every 2 seconds
        
        # Initial stats update
        self.update_stats()
    
    def initUI(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Node Consciousness Metrics")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #FFFFFF;")
        layout.addWidget(title)
        
        # Create stat groups
        self.create_consciousness_metrics(layout)
        self.create_network_insights(layout)
        
        # Add stretch to push everything to the top
        layout.addStretch()
    
    def create_consciousness_metrics(self, layout):
        """Create the consciousness metrics group"""
        metrics_group = QGroupBox("Consciousness Analysis")
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
        awareness_layout, self.awareness_value = self.create_metric_item("Awareness Level", "87%")
        integration_layout, self.integration_value = self.create_metric_item("Integration Index", "0.72")
        coherence_layout, self.coherence_value = self.create_metric_item("Neural Coherence", "High")
        responsiveness_layout, self.responsiveness_value = self.create_metric_item("Responsiveness", "94ms")
        
        # Add metrics to layout
        metrics_layout.addLayout(awareness_layout)
        metrics_layout.addLayout(integration_layout)
        metrics_layout.addLayout(coherence_layout)
        metrics_layout.addLayout(responsiveness_layout)
        
        layout.addWidget(metrics_group)
    
    def create_network_insights(self, layout):
        """Create the network insights group"""
        insights_group = QGroupBox("Network Dynamics")
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
        
        # Active nodes section
        nodes_title = QLabel("Active Node Types:")
        nodes_title.setStyleSheet("color: #CCCCCC;")
        insights_layout.addWidget(nodes_title)
        
        self.nodes_list = QLabel(
            "• Perception: 3 active nodes\n"
            "• Cognition: 2 active nodes\n"
            "• Memory: 1 active node\n"
            "• Emotion: 2 active nodes\n"
            "• Integration: 1 active node"
        )
        self.nodes_list.setStyleSheet("color: #A0A0D0; padding-left: 10px;")
        insights_layout.addWidget(self.nodes_list)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #3A3A5A;")
        insights_layout.addWidget(separator)
        
        # Current process
        process_title = QLabel("Current Process:")
        process_title.setStyleSheet("color: #CCCCCC; margin-top: 5px;")
        insights_layout.addWidget(process_title)
        
        self.current_process = QLabel(
            "Visual pattern recognition in progress\n"
            "Emotional response forming\n"
            "Memory association active"
        )
        self.current_process.setStyleSheet("color: #A0D0A0; padding-left: 10px;")
        self.current_process.setWordWrap(True)
        insights_layout.addWidget(self.current_process)
        
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
    
    def update_stats(self):
        """Update statistics with slightly varying values"""
        # Update awareness level
        awareness = 85 + random.randint(-3, 5)
        self.awareness_value.setText(f"{awareness}%")
        
        # Update integration index
        integration = 0.7 + random.uniform(-0.05, 0.07)
        self.integration_value.setText(f"{integration:.2f}")
        
        # Update coherence
        coherence_values = ["Medium", "Medium-High", "High", "Very High"]
        coherence = coherence_values[random.randint(1, 3)]
        self.coherence_value.setText(coherence)
        
        # Update responsiveness
        responsiveness = 90 + random.randint(-10, 15)
        self.responsiveness_value.setText(f"{responsiveness}ms")
        
        # Update active nodes
        perception_count = random.randint(2, 4)
        cognition_count = random.randint(1, 3)
        memory_count = random.randint(1, 2)
        emotion_count = random.randint(1, 3)
        integration_count = 1
        
        self.nodes_list.setText(
            f"• Perception: {perception_count} active nodes\n"
            f"• Cognition: {cognition_count} active nodes\n"
            f"• Memory: {memory_count} active node{'s' if memory_count > 1 else ''}\n"
            f"• Emotion: {emotion_count} active nodes\n"
            f"• Integration: {integration_count} active node"
        )
        
        # Update current process
        processes = [
            "Visual pattern recognition in progress",
            "Auditory processing active",
            "Semantic analysis running",
            "Emotional response forming",
            "Memory association active",
            "Decision matrix evaluating",
            "Attention filter engaged"
        ]
        
        # Select 2-3 random processes
        num_processes = random.randint(2, 3)
        selected_processes = random.sample(processes, num_processes)
        
        self.current_process.setText("\n".join(selected_processes))


class NodeConsciousnessPanel(QWidget):
    """Panel for visualizing node consciousness patterns and metrics"""
    
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
        
        title = QLabel("Node Consciousness Visualization")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFFFFF;")
        
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        # Sync button
        sync_btn = QPushButton("Sync Network")
        sync_btn.setStyleSheet("""
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
        header_layout.addWidget(sync_btn)
        
        main_layout.addWidget(header)
        
        # Content splitter
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setHandleWidth(1)
        content_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #3A3A5A;
            }
        """)
        
        # Left panel - Network visualization
        self.network_widget = NodeNetworkWidget()
        
        # Right panel - Node stats
        right_panel = QWidget()
        right_panel.setFixedWidth(300)
        right_panel.setStyleSheet("""
            background-color: #1D1D38;
            border-left: 1px solid #3A3A5A;
        """)
        
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        
        # Stats widget
        self.stats_widget = ConsciousnessStatsWidget()
        
        # Wrap in scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.stats_widget)
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
        content_splitter.addWidget(self.network_widget)
        content_splitter.addWidget(right_panel)
        content_splitter.setSizes([700, 300])
        
        main_layout.addWidget(content_splitter)
        
        # Connect signals and slots
        sync_btn.clicked.connect(self.sync_network)
    
    def sync_network(self):
        """Sync the network with the current state (mock implementation)"""
        # In a real implementation, this would update the network from a backend
        print("Network synchronization requested") 