"""
Consciousness Node for V7 Holographic Interface

Provides a visual representation of consciousness nodes and patterns
"""

import math
import random
from typing import Dict, List, Any

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QWidget, QVBoxLayout,
    QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsSimpleTextItem,
    QGraphicsPolygonItem, QGraphicsItem
)
from PySide6.QtCore import (
    Qt, QRectF, QPointF, QTimer, QPropertyAnimation,
    QEasingCurve, Slot, Property, QParallelAnimationGroup
)
from PySide6.QtGui import (
    QPainter, QColor, QBrush, QPen, QRadialGradient, QFont,
    QPainterPath, QTransform, QPolygonF
)

class ConsciousnessNode(QGraphicsItem):
    """
    Visual representation of a consciousness node in the V7 holographic interface
    """
    
    def __init__(self, node_id, node_type="standard", x=0, y=0, size=1.0, parent=None):
        super().__init__(parent)
        
        # Store properties
        self.node_id = node_id
        self.node_type = node_type
        self.setPos(x, y)
        self.size = size  # 0.0-1.0 relative size
        
        # Consciousness properties
        self.consciousness_level = 0.5  # 0.0-1.0
        self.neural_activity = 0.5  # 0.0-1.0
        self.insight_potential = 0.0  # 0.0-1.0
        
        # Animation properties
        self.pulse_phase = random.random() * math.pi * 2
        self.rotation_phase = random.random() * 360
        self.rotation_speed = (random.random() - 0.5) * 2.0
        
        # Set node color based on type
        if node_type == "insight":
            self.base_color = QColor(255, 200, 100)  # Gold
        elif node_type == "pattern":
            self.base_color = QColor(100, 200, 255)  # Blue
        elif node_type == "experience":
            self.base_color = QColor(150, 255, 150)  # Green
        else:
            self.base_color = QColor(200, 150, 255)  # Purple
        
        # Set interaction flags
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        
        # Create label
        self.label = QGraphicsSimpleTextItem(self)
        self.label.setBrush(QBrush(QColor(220, 255, 255)))
        self.label.setFont(QFont("Consolas", 8))
        self.label.setText(node_id)
        
        # Position the label below the node
        self._update_label_position()
    
    def _update_label_position(self):
        """Update the label position based on node size"""
        br = self.label.boundingRect()
        self.label.setPos(-br.width() / 2, self.base_radius() + 5)
    
    def base_radius(self):
        """Get the base radius based on size"""
        return 15 + 15 * self.size
    
    def set_consciousness_level(self, level):
        """Set consciousness level (0.0-1.0)"""
        self.consciousness_level = max(0.0, min(1.0, level))
        self.update()
    
    def set_neural_activity(self, level):
        """Set neural activity level (0.0-1.0)"""
        self.neural_activity = max(0.0, min(1.0, level))
        self.update()
    
    def set_insight_potential(self, level):
        """Set insight potential (0.0-1.0)"""
        self.insight_potential = max(0.0, min(1.0, level))
        self.update()
    
    def update_animation(self, dt=0.05):
        """Update animation state"""
        # Update pulse
        self.pulse_phase += dt
        if self.pulse_phase > math.pi * 2:
            self.pulse_phase -= math.pi * 2
        
        # Update rotation
        self.rotation_phase += self.rotation_speed * dt * 60
        
        # Only apply rotation if consciousness level is high enough
        if self.consciousness_level > 0.4:
            self.setRotation(self.rotation_phase * self.consciousness_level)
        
        # Create small random movement based on neural activity
        if self.neural_activity > 0.2:
            dx = math.sin(self.pulse_phase * 1.3) * 0.5 * self.neural_activity
            dy = math.cos(self.pulse_phase * 0.7) * 0.5 * self.neural_activity
            
            # Apply jitter only if not being manipulated
            if not self.isSelected():
                current_pos = self.pos()
                self.setPos(current_pos.x() + dx, current_pos.y() + dy)
        
        self.update()
    
    def boundingRect(self):
        """Define the bounding rectangle"""
        radius = self.base_radius() * 1.5  # Include glow effects
        return QRectF(-radius, -radius, radius * 2, radius * 2)
    
    def shape(self):
        """Define the shape for collision detection"""
        path = QPainterPath()
        path.addEllipse(QRectF(-self.base_radius(), -self.base_radius(), 
                             self.base_radius() * 2, self.base_radius() * 2))
        return path
    
    def hoverEnterEvent(self, event):
        """Handle hover enter events"""
        # Temporarily increase neural activity on hover
        self.prev_neural_activity = self.neural_activity
        self.neural_activity = min(1.0, self.neural_activity * 1.5)
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        """Handle hover leave events"""
        # Restore previous neural activity
        if hasattr(self, 'prev_neural_activity'):
            self.neural_activity = self.prev_neural_activity
        super().hoverLeaveEvent(event)
    
    def paint(self, painter, option, widget):
        """Custom painting with holographic effects"""
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # Base size
        radius = self.base_radius()
        
        # Calculate pulse effect (0.0-1.0)
        pulse = 0.5 + 0.5 * math.sin(self.pulse_phase)
        
        # Color intensity based on consciousness level
        intensity = self.consciousness_level
        
        # Modify base color based on node properties
        r, g, b = self.base_color.red(), self.base_color.green(), self.base_color.blue()
        
        # Adjust color based on insight potential (shift towards gold for high insight)
        if self.insight_potential > 0.5:
            r = min(255, r + int((255 - r) * (self.insight_potential - 0.5) * 2))
            g = min(255, g + int((200 - g) * (self.insight_potential - 0.5) * 2))
        
        # Final color with alpha based on consciousness
        main_color = QColor(r, g, b, 100 + int(155 * intensity))
        
        # Draw outer glow based on neural activity
        glow_radius = radius * (1.0 + 0.5 * self.neural_activity)
        gradient = QRadialGradient(0, 0, glow_radius)
        
        # Glow color with transparency
        glow_color = QColor(r, g, b, 40 + int(60 * pulse * self.neural_activity))
        
        gradient.setColorAt(0, glow_color)
        gradient.setColorAt(0.7, QColor(glow_color.red(), glow_color.green(), 
                                       glow_color.blue(), 20))
        gradient.setColorAt(1, QColor(0, 0, 0, 0))
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(QRectF(-glow_radius, -glow_radius, 
                                   glow_radius * 2, glow_radius * 2))
        
        # Draw core shape based on node type
        if self.node_type == "insight":
            # Star shape for insights
            points = 8
            inner_radius = radius * 0.6
            
            polygon = QPolygonF()
            for i in range(points * 2):
                angle = math.pi * i / points
                r = radius if i % 2 == 0 else inner_radius
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                polygon.append(QPointF(x, y))
            
            painter.setPen(QPen(main_color.lighter(150), 1.5))
            painter.setBrush(QBrush(main_color))
            painter.drawPolygon(polygon)
            
        elif self.node_type == "pattern":
            # Hexagon shape for patterns
            points = 6
            polygon = QPolygonF()
            for i in range(points):
                angle = 2 * math.pi * i / points
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                polygon.append(QPointF(x, y))
            
            painter.setPen(QPen(main_color.lighter(150), 1.5))
            painter.setBrush(QBrush(main_color))
            painter.drawPolygon(polygon)
            
        else:
            # Circle for standard nodes and experiences
            painter.setPen(QPen(main_color.lighter(150), 1.5))
            painter.setBrush(QBrush(main_color))
            painter.drawEllipse(QRectF(-radius, -radius, radius * 2, radius * 2))
        
        # Add internal details based on consciousness level
        if self.consciousness_level > 0.3:
            # Draw internal patterns
            internal_radius = radius * 0.7
            
            # Higher consciousness = more complex patterns
            if self.consciousness_level > 0.7:
                # Complex pattern
                path = QPainterPath()
                
                for i in range(5):
                    angle1 = 2 * math.pi * i / 5
                    angle2 = 2 * math.pi * ((i + 2) % 5) / 5
                    
                    x1 = internal_radius * math.cos(angle1)
                    y1 = internal_radius * math.sin(angle1)
                    x2 = internal_radius * math.cos(angle2)
                    y2 = internal_radius * math.sin(angle2)
                    
                    path.moveTo(x1, y1)
                    path.lineTo(x2, y2)
                
                painter.setPen(QPen(QColor(255, 255, 255, 80), 1))
                painter.drawPath(path)
                
            elif self.consciousness_level > 0.5:
                # Simpler pattern - inner circle
                painter.setPen(QPen(QColor(255, 255, 255, 60), 1))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QRectF(-internal_radius, -internal_radius, 
                                         internal_radius * 2, internal_radius * 2))
        
        # Add center highlight
        highlight_radius = radius * 0.3
        highlight_color = QColor(255, 255, 255, 80 + int(100 * pulse * intensity))
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(highlight_color))
        painter.drawEllipse(QRectF(-highlight_radius, -highlight_radius, 
                                  highlight_radius * 2, highlight_radius * 2))

class ConsciousnessConnection(QGraphicsPathItem):
    """
    Holographic connection between consciousness nodes
    """
    
    def __init__(self, source_node, target_node, strength=0.5, parent=None):
        super().__init__(parent)
        
        # Store properties
        self.source_node = source_node
        self.target_node = target_node
        self.strength = strength  # Connection strength (0.0-1.0)
        
        # Animation properties
        self.flow_phase = random.random()  # 0.0-1.0
        self.activation = 0.0  # Connection activation
        
        # Create the path
        self._update_path()
        
        # Set style
        self.setPen(Qt.NoPen)  # No standard pen - we'll custom draw
        self.setBrush(Qt.NoBrush)
        
        # Set z-value to be below nodes
        self.setZValue(-1)
    
    def _update_path(self):
        """Update connection path based on node positions"""
        if not self.source_node or not self.target_node:
            return
        
        # Get node positions
        source_pos = self.source_node.scenePos()
        target_pos = self.target_node.scenePos()
        
        # Create path
        path = QPainterPath()
        path.moveTo(source_pos)
        
        # Add curve - adjust control points based on distance
        dx = target_pos.x() - source_pos.x()
        dy = target_pos.y() - source_pos.y()
        dist = math.sqrt(dx * dx + dy * dy)
        
        # Calculate control points
        ctrl_scale = 0.3
        ctrl1_x = source_pos.x() + dx * 0.25 + dy * ctrl_scale
        ctrl1_y = source_pos.y() + dy * 0.25 - dx * ctrl_scale
        ctrl2_x = source_pos.x() + dx * 0.75 - dy * ctrl_scale
        ctrl2_y = source_pos.y() + dy * 0.75 + dx * ctrl_scale
        
        # Create the curve
        path.cubicTo(ctrl1_x, ctrl1_y, ctrl2_x, ctrl2_y, target_pos.x(), target_pos.y())
        
        # Set the path
        self.setPath(path)
    
    def set_activation(self, value):
        """Set connection activation level (0.0-1.0)"""
        self.activation = max(0.0, min(1.0, value))
        self.update()
    
    def update_animation(self, dt=0.05):
        """Update animation state"""
        # Update path if nodes have moved
        self._update_path()
        
        # Update flow animation
        flow_increment = 0.02 * dt * 10 * (0.2 + 0.8 * self.activation)
        self.flow_phase += flow_increment
        if self.flow_phase > 1.0:
            self.flow_phase -= 1.0
        
        self.update()
    
    def paint(self, painter, option, widget):
        """Custom painting with energy flow animation"""
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # Calculate visibility based on strength and activation
        visibility = self.strength * (0.3 + 0.7 * self.activation)
        
        if visibility < 0.05:
            return  # Too faint to see
        
        # Get path properties
        path = self.path()
        
        # Draw the base path
        base_alpha = int(50 * visibility)
        if base_alpha > 5:
            # Base color depends on connection strength
            if self.strength > 0.7:
                # Strong connections are more white
                base_color = QColor(180, 220, 255, base_alpha)
            elif self.strength > 0.4:
                # Medium connections are blue
                base_color = QColor(100, 180, 255, base_alpha)
            else:
                # Weak connections are darker blue
                base_color = QColor(70, 130, 200, base_alpha)
            
            painter.setPen(QPen(base_color, 1 + self.strength, Qt.DashLine))
            painter.drawPath(path)
        
        # If sufficiently active, draw energy particles
        if self.activation > 0.1:
            # Number of particles based on activation and strength
            num_particles = int(5 * visibility)
            
            # Set particle style
            painter.setPen(Qt.NoPen)
            
            for i in range(num_particles):
                # Calculate position along the path
                pos = (self.flow_phase + i / num_particles) % 1.0
                
                # Get point at this position
                point = path.pointAtPercent(pos)
                
                # Particle size and brightness based on position
                # Particles are brighter in the middle of their journey
                particle_brightness = 1.0 - abs(pos - 0.5) * 2.0
                particle_size = (2 + 3 * visibility) * particle_brightness
                
                # Particle color with varying alpha
                alpha = int(200 * particle_brightness * visibility)
                
                # Color depends on connection type and strength
                if self.strength > 0.7:
                    particle_color = QColor(220, 240, 255, alpha)  # Strong: white-blue
                elif self.strength > 0.4:
                    particle_color = QColor(100, 200, 255, alpha)  # Medium: cyan-blue
                else:
                    particle_color = QColor(80, 150, 230, alpha)   # Weak: darker blue
                
                # Draw particle glow
                gradient = QRadialGradient(point, particle_size * 2)
                gradient.setColorAt(0, particle_color)
                gradient.setColorAt(1, QColor(particle_color.red(), particle_color.green(), 
                                           particle_color.blue(), 0))
                
                painter.setBrush(QBrush(gradient))
                painter.drawEllipse(point, particle_size, particle_size)

class ConsciousnessNetworkVisualizer(QGraphicsView):
    """
    Holographic visualization of the consciousness network
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create scene
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(-300, -200, 600, 400)
        self.setScene(self.scene)
        
        # Configure view
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setInteractive(True)
        
        # Background
        self.setBackgroundBrush(QBrush(QColor(0, 10, 20)))
        
        # Store nodes and connections
        self.nodes = {}  # id -> ConsciousnessNode
        self.connections = []  # List of ConsciousnessConnection
        
        # Create backdrop grid
        self._create_grid()
        
        # Animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animate_step)
        self.animation_timer.start(30)  # ~33 fps
        
        # Default network
        self.generate_default_network()
    
    def _create_grid(self):
        """Create a holographic grid background"""
        # Horizontal lines
        for y in range(-200, 201, 40):
            line = QGraphicsPathItem()
            
            # Create a path
            path = QPainterPath()
            path.moveTo(-300, y)
            path.lineTo(300, y)
            
            line.setPath(path)
            line.setPen(QPen(QColor(0, 80, 120, 20), 1, Qt.DashLine))
            
            self.scene.addItem(line)
            line.setZValue(-10)  # Below everything
        
        # Vertical lines
        for x in range(-300, 301, 40):
            line = QGraphicsPathItem()
            
            # Create a path
            path = QPainterPath()
            path.moveTo(x, -200)
            path.lineTo(x, 200)
            
            line.setPath(path)
            line.setPen(QPen(QColor(0, 80, 120, 20), 1, Qt.DashLine))
            
            self.scene.addItem(line)
            line.setZValue(-10)  # Below everything
    
    def generate_default_network(self):
        """Generate a default consciousness network for initial display"""
        # Clear existing items
        for node_id in list(self.nodes.keys()):
            self.scene.removeItem(self.nodes[node_id])
        self.nodes.clear()
        
        for conn in self.connections:
            self.scene.removeItem(conn)
        self.connections.clear()
        
        # Create some sample nodes
        node_defs = [
            {"id": "central", "type": "pattern", "x": 0, "y": 0, "size": 1.0},
            {"id": "insight_1", "type": "insight", "x": -120, "y": -80, "size": 0.8},
            {"id": "pattern_1", "type": "pattern", "x": 120, "y": -80, "size": 0.7},
            {"id": "experience_1", "type": "experience", "x": -120, "y": 80, "size": 0.6},
            {"id": "standard_1", "type": "standard", "x": 120, "y": 80, "size": 0.5},
        ]
        
        # Add some random nodes
        for i in range(5):
            angle = random.random() * math.pi * 2
            distance = random.uniform(100, 200)
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)
            
            node_type = random.choice(["standard", "pattern", "experience"])
            if random.random() < 0.2:  # 20% chance for an insight
                node_type = "insight"
            
            node_defs.append({
                "id": f"node_{i}", 
                "type": node_type,
                "x": x, 
                "y": y,
                "size": random.uniform(0.4, 0.9)
            })
        
        # Create nodes
        for node_def in node_defs:
            node = ConsciousnessNode(
                node_def["id"],
                node_def["type"],
                node_def["x"],
                node_def["y"],
                node_def["size"]
            )
            
            self.scene.addItem(node)
            self.nodes[node_def["id"]] = node
            
            # Set random consciousness levels
            node.set_consciousness_level(random.uniform(0.4, 0.9))
            node.set_neural_activity(random.uniform(0.3, 0.8))
            
            # Insights have higher insight potential
            if node_def["type"] == "insight":
                node.set_insight_potential(random.uniform(0.7, 0.9))
            else:
                node.set_insight_potential(random.uniform(0.1, 0.4))
        
        # Create connections
        for node_id, node in self.nodes.items():
            # Connect to 2-3 other nodes
            connect_count = random.randint(2, 3)
            other_nodes = list(self.nodes.values())
            other_nodes.remove(node)
            
            if len(other_nodes) > connect_count:
                random.shuffle(other_nodes)
                connect_targets = other_nodes[:connect_count]
                
                for target in connect_targets:
                    # Create connection with random strength
                    strength = random.uniform(0.3, 0.9)
                    conn = ConsciousnessConnection(node, target, strength)
                    
                    self.scene.addItem(conn)
                    self.connections.append(conn)
                    
                    # Set random activation
                    conn.set_activation(random.uniform(0.3, 0.8))
    
    def update_network(self, network_data):
        """Update the network with new data"""
        # network_data should contain nodes and connections
        if "nodes" not in network_data or "connections" not in network_data:
            return
        
        # Update nodes
        for node_data in network_data["nodes"]:
            node_id = node_data.get("id")
            
            if node_id in self.nodes:
                # Update existing node
                node = self.nodes[node_id]
                node.set_consciousness_level(node_data.get("consciousness", 0.5))
                node.set_neural_activity(node_data.get("activity", 0.5))
                node.set_insight_potential(node_data.get("insight", 0.3))
            else:
                # Create new node
                x = node_data.get("x", random.uniform(-200, 200))
                y = node_data.get("y", random.uniform(-150, 150))
                node_type = node_data.get("type", "standard")
                size = node_data.get("size", 0.7)
                
                node = ConsciousnessNode(node_id, node_type, x, y, size)
                node.set_consciousness_level(node_data.get("consciousness", 0.5))
                node.set_neural_activity(node_data.get("activity", 0.5))
                node.set_insight_potential(node_data.get("insight", 0.3))
                
                self.scene.addItem(node)
                self.nodes[node_id] = node
        
        # Clean up removed nodes
        for node_id in list(self.nodes.keys()):
            if not any(n.get("id") == node_id for n in network_data["nodes"]):
                self.scene.removeItem(self.nodes[node_id])
                del self.nodes[node_id]
        
        # Remove all connections
        for conn in self.connections:
            self.scene.removeItem(conn)
        self.connections.clear()
        
        # Create new connections
        for conn_data in network_data["connections"]:
            source_id = conn_data.get("source")
            target_id = conn_data.get("target")
            
            if source_id in self.nodes and target_id in self.nodes:
                source_node = self.nodes[source_id]
                target_node = self.nodes[target_id]
                
                strength = conn_data.get("strength", 0.5)
                conn = ConsciousnessConnection(source_node, target_node, strength)
                conn.set_activation(conn_data.get("activation", 0.5))
                
                self.scene.addItem(conn)
                self.connections.append(conn)
    
    def animate_step(self):
        """Update animation state for all components"""
        # Update node animations
        for node in self.nodes.values():
            node.update_animation()
        
        # Update connection animations
        for conn in self.connections:
            conn.update_animation()
    
    def resizeEvent(self, event):
        """Handle resize events"""
        # Scale view to fit scene
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)
    
    def showEvent(self, event):
        """Handle show events"""
        # Scale view to fit scene when shown
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        super().showEvent(event) 