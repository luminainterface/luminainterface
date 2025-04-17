"""
Knowledge Graph Visualization for V7 Holographic Interface

Provides a futuristic visualization of knowledge connections
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

class KnowledgeNode(QGraphicsItem):
    """A holographic node representing a knowledge entity"""
    
    def __init__(self, node_id, label, x=0, y=0, size=1.0, parent=None):
        super().__init__(parent)
        
        # Store properties
        self.node_id = node_id
        self.label = label
        self.size = size  # 0.0-1.0 relative size
        self.highlight_level = 0.0  # 0.0-1.0 highlight intensity
        self.pulse_phase = random.random() * math.pi * 2
        
        # Animation properties
        self.rotation_phase = random.random() * 360
        self.rotation_speed = (random.random() - 0.5) * 1.0
        
        # Position
        self.setPos(x, y)
        
        # Interactive flags
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        
        # Create label text
        self.text = QGraphicsSimpleTextItem(self)
        self.text.setFont(QFont("Consolas", 8))
        self.text.setBrush(QBrush(QColor(220, 255, 255, 200)))
        self.text.setText(label)
        
        # Center the label
        br = self.text.boundingRect()
        self.text.setPos(-br.width() / 2, self.base_radius() + 5)
    
    def base_radius(self):
        """Get the base radius based on size"""
        return 15 + 15 * self.size
    
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
    
    def update_animation(self, dt=0.05):
        """Update animation state"""
        # Update pulse
        self.pulse_phase += dt
        if self.pulse_phase > math.pi * 2:
            self.pulse_phase -= math.pi * 2
        
        # Update rotation (if enabled)
        self.rotation_phase += self.rotation_speed * dt * 60
        self.setRotation(self.rotation_phase)
        
        # Create small random movement
        dx = math.sin(self.pulse_phase * 1.3) * 0.2
        dy = math.cos(self.pulse_phase * 0.7) * 0.2
        
        # Only apply jitter if not moving for other reasons
        if not self.isSelected():
            current_pos = self.pos()
            self.setPos(current_pos.x() + dx, current_pos.y() + dy)
        
        self.update()
    
    def set_highlight(self, level):
        """Set highlight level (0.0-1.0)"""
        self.highlight_level = max(0.0, min(1.0, level))
        self.update()
    
    def hover_highlight(self):
        """Highlight when hovered"""
        self.set_highlight(0.8)
    
    def reset_highlight(self):
        """Reset highlight when not hovered"""
        self.set_highlight(0.0)
    
    def hoverEnterEvent(self, event):
        """Handle hover enter events"""
        self.hover_highlight()
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        """Handle hover leave events"""
        self.reset_highlight()
        super().hoverLeaveEvent(event)
    
    def paint(self, painter, option, widget):
        """Custom painting with holographic effects"""
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # Base size
        radius = self.base_radius()
        
        # Calculate pulse effect (0.0-1.0)
        pulse = 0.5 + 0.5 * math.sin(self.pulse_phase)
        
        # Combine pulse with highlight for visual intensity
        intensity = max(pulse * 0.4, self.highlight_level)
        
        # Colors based on size and highlight
        base_hue = 180  # Cyan-blue
        hue_shift = int(self.size * 40)  # Shift towards green for larger nodes
        h = (base_hue + hue_shift) % 360
        
        # Create color with saturation and value based on highlight
        s = 80 + int(20 * intensity)
        v = 150 + int(105 * intensity)
        
        # Create QColor from HSV
        hsv_color = QColor()
        hsv_color.setHsv(h, s, v)
        
        # Draw outer glow
        glow_radius = radius * (1.0 + 0.5 * intensity)
        gradient = QRadialGradient(0, 0, glow_radius)
        
        # Outer glow color with transparency
        glow_color = QColor(hsv_color)
        glow_color.setAlpha(40 + int(60 * intensity))
        
        gradient.setColorAt(0, glow_color)
        gradient.setColorAt(0.7, QColor(glow_color.red(), glow_color.green(), 
                                       glow_color.blue(), 20))
        gradient.setColorAt(1, QColor(0, 0, 0, 0))
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(QRectF(-glow_radius, -glow_radius, 
                                   glow_radius * 2, glow_radius * 2))
        
        # Draw main circle
        main_color = QColor(hsv_color)
        main_color.setAlpha(100 + int(155 * intensity))
        
        painter.setPen(QPen(main_color.lighter(150), 1))
        
        # Create gradient for main circle
        main_gradient = QRadialGradient(0, 0, radius)
        main_gradient.setColorAt(0, main_color.lighter(150))
        main_gradient.setColorAt(1, QColor(main_color.red() * 0.7, 
                                         main_color.green() * 0.7, 
                                         main_color.blue() * 0.7, 
                                         main_color.alpha()))
        
        painter.setBrush(QBrush(main_gradient))
        painter.drawEllipse(QRectF(-radius, -radius, radius * 2, radius * 2))
        
        # Draw highlight/rim
        highlight_radius = radius * 0.9
        highlight_width = 2 + 2 * intensity
        
        painter.setPen(QPen(QColor(255, 255, 255, 50 + int(150 * intensity)), 
                           highlight_width))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QRectF(-highlight_radius, -highlight_radius, 
                                  highlight_radius * 2, highlight_radius * 2))
        
        # Draw center highlight
        center_radius = radius * 0.3
        center_gradient = QRadialGradient(0, 0, center_radius)
        center_gradient.setColorAt(0, QColor(255, 255, 255, 150 + int(105 * intensity)))
        center_gradient.setColorAt(1, QColor(200, 255, 255, 0))
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(center_gradient))
        painter.drawEllipse(QRectF(-center_radius, -center_radius, 
                                  center_radius * 2, center_radius * 2))

class KnowledgeEdge(QGraphicsPathItem):
    """A holographic edge connecting knowledge nodes"""
    
    def __init__(self, source_node, target_node, weight=0.5, parent=None):
        super().__init__(parent)
        
        # Store properties
        self.source_node = source_node
        self.target_node = target_node
        self.weight = weight  # 0.0-1.0 connection strength
        self.highlight_level = 0.0
        
        # Animation properties
        self.flow_phase = random.random()  # 0.0-1.0
        self.flow_speed = 0.01 + 0.02 * weight
        
        # Create path
        self._update_path()
        
        # Set styles based on weight
        alpha = 30 + int(100 * weight)
        self.setPen(QPen(QColor(0, 180, 255, alpha), 1 + weight))
        self.setBrush(Qt.NoBrush)
        
        # Set z-value to be below nodes
        self.setZValue(-1)
    
    def _update_path(self):
        """Update edge path based on node positions"""
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
        
        # Calculate control point displacement
        if dist < 50:
            # Short connections have small curves
            ctrl_scale = 0.1
        else:
            # Longer connections have more pronounced curves
            ctrl_scale = 0.25
        
        # Add random variation to control points
        ctrl_var_x = (random.random() - 0.5) * 20
        ctrl_var_y = (random.random() - 0.5) * 20
        
        # Calculate control point
        ctrl_x = source_pos.x() + dx * 0.5 + dy * ctrl_scale + ctrl_var_x
        ctrl_y = source_pos.y() + dy * 0.5 - dx * ctrl_scale + ctrl_var_y
        
        # Create the curve
        path.quadTo(ctrl_x, ctrl_y, target_pos.x(), target_pos.y())
        
        # Set the path
        self.setPath(path)
    
    def set_highlight(self, level):
        """Set highlight level (0.0-1.0)"""
        self.highlight_level = max(0.0, min(1.0, level))
        
        # Update visual based on highlight
        base_alpha = 30 + int(100 * self.weight)
        highlight_alpha = int(200 * self.highlight_level)
        alpha = max(base_alpha, highlight_alpha)
        
        width = 1 + self.weight + self.highlight_level
        
        self.setPen(QPen(QColor(0, 180, 255, alpha), width))
        self.update()
    
    def update_animation(self, dt=0.05):
        """Update animation state"""
        # Update flow animation
        flow_increment = self.flow_speed * dt * 10
        self.flow_phase += flow_increment
        if self.flow_phase > 1.0:
            self.flow_phase -= 1.0
        
        self.update()
    
    def paint(self, painter, option, widget):
        """Custom painting with animated particles"""
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # Draw the base path
        super().paint(painter, option, widget)
        
        # Calculate visibility factor (combination of weight and highlight)
        visibility = max(self.weight * 0.5, self.highlight_level)
        
        # If visible enough, draw animated particles
        if visibility > 0.1:
            # Number of particles based on visibility
            num_particles = 1 + int(5 * visibility)
            
            # Get path properties
            path = self.path()
            
            # Set particle style
            painter.setPen(Qt.NoPen)
            
            for i in range(num_particles):
                # Calculate position along the path
                pos = (self.flow_phase + i / num_particles) % 1.0
                
                # Get point at this position
                point = path.pointAtPercent(pos)
                
                # Particle size based on visibility and position along path
                particle_brightness = 1.0 - abs(pos - 0.5) * 2.0  # Brighter in middle
                particle_size = (2 + 3 * visibility) * particle_brightness
                
                # Particle color and alpha
                alpha = int(200 * particle_brightness * visibility)
                particle_color = QColor(100, 200, 255, alpha)
                
                # Draw particle glow
                gradient = QRadialGradient(point, particle_size * 2)
                gradient.setColorAt(0, particle_color)
                gradient.setColorAt(1, QColor(0, 100, 200, 0))
                
                painter.setBrush(QBrush(gradient))
                painter.drawEllipse(point, particle_size, particle_size)

class KnowledgeGraphWidget(QGraphicsView):
    """Holographic visualization of the knowledge graph"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create scene
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(-300, -200, 600, 400)
        self.setScene(self.scene)
        
        # Configure view
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.TextAntialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setOptimizationFlags(QGraphicsView.DontAdjustForAntialiasing | 
                                 QGraphicsView.DontSavePainterState)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setInteractive(True)
        
        # Background
        self.setBackgroundBrush(QBrush(QColor(0, 10, 20)))
        
        # Store nodes and edges
        self.nodes = {}  # id -> KnowledgeNode
        self.edges = []  # List of KnowledgeEdge objects
        
        # Create grid background
        self._create_grid()
        
        # Animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animate_step)
        self.animation_timer.start(30)  # ~33 fps
        
        # Default data
        self.generate_default_graph()
    
    def _create_grid(self):
        """Create a holographic grid background"""
        # Horizontal lines
        for y in range(-200, 201, 40):
            line = QGraphicsPathItem()
            
            # Create a path for a dashed line
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
            
            # Create a path for a dashed line
            path = QPainterPath()
            path.moveTo(x, -200)
            path.lineTo(x, 200)
            
            line.setPath(path)
            line.setPen(QPen(QColor(0, 80, 120, 20), 1, Qt.DashLine))
            
            self.scene.addItem(line)
            line.setZValue(-10)  # Below everything
    
    def generate_default_graph(self):
        """Generate a default knowledge graph for initial display"""
        # Sample topics
        topics = [
            "Neural Networks", "Consciousness", "Language Models",
            "Memory Systems", "Pattern Recognition", "Learning Algorithms"
        ]
        
        # Create nodes with random positions
        nodes = []
        for i, topic in enumerate(topics):
            # Calculate position in a circle
            angle = 2 * math.pi * i / len(topics)
            distance = 150  # Radius
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)
            
            # Create node data
            nodes.append({
                "id": f"topic_{i}",
                "label": topic,
                "size": random.uniform(0.5, 1.0)
            })
        
        # Create edges
        edges = []
        for i in range(len(nodes)):
            # Connect to 2-3 other nodes
            for _ in range(random.randint(2, 3)):
                target = random.randint(0, len(nodes) - 1)
                if target != i:
                    edges.append({
                        "source": nodes[i]["id"],
                        "target": nodes[target]["id"],
                        "weight": random.uniform(0.3, 1.0)
                    })
        
        # Update graph with generated data
        self.update_graph(nodes, edges)
    
    def update_graph(self, nodes, edges):
        """Update the graph with new data"""
        # Clear existing items
        for node_id in list(self.nodes.keys()):
            # Check if node still exists in new data
            if not any(n["id"] == node_id for n in nodes):
                # Remove node
                self.scene.removeItem(self.nodes[node_id])
                del self.nodes[node_id]
        
        # Remove all edges
        for edge in self.edges:
            self.scene.removeItem(edge)
        self.edges.clear()
        
        # Create/update nodes
        for node_data in nodes:
            node_id = node_data["id"]
            
            if node_id in self.nodes:
                # Update existing node
                # You could animate the transition to new properties here
                pass
            else:
                # Create new node
                
                # Determine position - if not provided, use random placement
                if "x" in node_data and "y" in node_data:
                    x, y = node_data["x"], node_data["y"]
                else:
                    # Random position in a circle
                    angle = random.random() * math.pi * 2
                    distance = random.uniform(50, 200)
                    x = distance * math.cos(angle)
                    y = distance * math.sin(angle)
                
                # Create node
                node = KnowledgeNode(
                    node_id, 
                    node_data["label"], 
                    x, y, 
                    node_data.get("size", 0.5)
                )
                
                self.scene.addItem(node)
                self.nodes[node_id] = node
        
        # Create edges
        for edge_data in edges:
            source_id = edge_data["source"]
            target_id = edge_data["target"]
            
            if source_id in self.nodes and target_id in self.nodes:
                source_node = self.nodes[source_id]
                target_node = self.nodes[target_id]
                
                edge = KnowledgeEdge(
                    source_node,
                    target_node,
                    edge_data.get("weight", 0.5)
                )
                
                self.scene.addItem(edge)
                self.edges.append(edge)
    
    def animate_step(self):
        """Update animation state for all components"""
        # Update node animations
        for node in self.nodes.values():
            node.update_animation()
        
        # Update edge animations
        for edge in self.edges:
            edge.update_animation()
    
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