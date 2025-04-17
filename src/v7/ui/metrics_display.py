"""
Holographic Metrics Display for V7 Interface

Provides a futuristic holographic display of consciousness metrics
"""

import math
import random
from typing import Dict, List, Any

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QWidget, QVBoxLayout,
    QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsSimpleTextItem,
    QGraphicsPolygonItem
)
from PySide6.QtCore import (
    Qt, QRectF, QPointF, QTimer, QPropertyAnimation,
    QEasingCurve, Slot, Property, QParallelAnimationGroup
)
from PySide6.QtGui import (
    QPainter, QColor, QBrush, QPen, QRadialGradient, QFont,
    QPainterPath, QTransform, QPolygonF
)

class MetricNode(QGraphicsPathItem):
    """
    A holographic node representing a consciousness metric
    """
    
    def __init__(self, x, y, radius, name, value=0.5, parent=None):
        super().__init__(parent)
        
        # Store properties
        self.center_x = x
        self.center_y = y
        self.radius = radius
        self.name = name
        self.value = value
        self.target_value = value
        self.setPos(x, y)
        
        # Animation properties
        self.animation_phase = random.random() * math.pi * 2
        self.highlight_intensity = 0.0
        
        # Create the path
        self._update_path()
        
        # Style
        self.setPen(QPen(QColor(0, 200, 255, 150), 1.5))
        self.setBrush(QBrush(QColor(0, 100, 200, 100)))
        
        # Create value text
        self.value_text = QGraphicsSimpleTextItem(self)
        self.value_text.setFont(QFont("Consolas", 9, QFont.Bold))
        self.value_text.setBrush(QBrush(QColor(220, 255, 255)))
        self._update_value_text()
        
        # Create label
        self.label = QGraphicsSimpleTextItem(self)
        self.label.setFont(QFont("Consolas", 8))
        self.label.setBrush(QBrush(QColor(180, 220, 255)))
        self.label.setText(name)
        
        # Position the label below the node
        br = self.label.boundingRect()
        self.label.setPos(-br.width() / 2, radius + 5)
    
    def _update_path(self):
        """Update the node shape based on current value"""
        # Create a hexagonal shape
        path = QPainterPath()
        
        # Size varies with value (0.5-1.5x)
        size = self.radius * (0.8 + 0.6 * self.value)
        
        # Number of points (more points for higher values)
        n_points = 6 + int(self.value * 4)
        
        # Create the polygon shape
        for i in range(n_points):
            angle = 2 * math.pi * i / n_points
            # Distorted circle
            dist = 1.0 + 0.2 * math.sin(3 * angle + self.animation_phase)
            x = size * dist * math.cos(angle)
            y = size * dist * math.sin(angle)
            
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        
        path.closeSubpath()
        self.setPath(path)
    
    def _update_value_text(self):
        """Update the value text display"""
        self.value_text.setText(f"{self.value:.2f}")
        br = self.value_text.boundingRect()
        self.value_text.setPos(-br.width() / 2, -br.height() / 2)
    
    def set_value(self, value):
        """Set the node value (0.0-1.0)"""
        self.target_value = max(0.0, min(1.0, value))
    
    def update_animation(self, dt=0.05):
        """Update animation state"""
        # Update phase
        self.animation_phase += dt
        if self.animation_phase > math.pi * 2:
            self.animation_phase -= math.pi * 2
        
        # Smooth value animation
        if abs(self.value - self.target_value) > 0.01:
            self.value += (self.target_value - self.value) * 0.1
            self._update_path()
            self._update_value_text()
        
        # Highlight pulsation
        self.highlight_intensity = 0.5 + 0.5 * math.sin(self.animation_phase)
        
        # Update colorization based on value and highlight
        highlight_factor = 100 + int(80 * self.highlight_intensity * self.value)
        value_factor = 50 + int(150 * self.value)  # Higher values are more colorful
        
        # Calculate color based on value
        # Lower values: blue, higher values: cyan-green
        r = min(255, int(0 + value_factor * 0.5))
        g = min(255, int(100 + value_factor * 0.6))
        b = min(255, int(200 + value_factor * 0.2))
        
        # Apply colors
        self.setPen(QPen(QColor(r, g, b, 150 + int(105 * self.highlight_intensity)), 
                         1.5 + self.value))
        self.setBrush(QBrush(QColor(r * 0.3, g * 0.3, b * 0.4, 
                                    70 + int(50 * self.value))))
    
    def paint(self, painter, option, widget):
        """Custom painting with extra effects"""
        # Enable antialiasing
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # Draw outer glow
        glow_path = self.path()
        painter.setPen(QPen(QColor(0, 200, 255, 50 + int(50 * self.highlight_intensity)), 
                           3 + 2 * self.value))
        painter.drawPath(glow_path)
        
        # Draw standard item
        super().paint(painter, option, widget)

class ConnectionLine(QGraphicsPathItem):
    """A holographic line connecting metric nodes"""
    
    def __init__(self, start_x, start_y, end_x, end_y, parent=None):
        super().__init__(parent)
        
        # Store properties
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
        
        # Animation properties
        self.animation_phase = random.random()  # 0.0-1.0
        self.particle_speed = 0.01 + random.random() * 0.02
        self.value = 0.5  # Line activation value
        
        # Create the path
        self._update_path()
        
        # Set style
        self.setPen(QPen(QColor(0, 180, 255, 60), 1, Qt.DashLine))
        self.setBrush(Qt.NoBrush)
    
    def _update_path(self):
        """Update the connection path"""
        # Create a curved path between points
        path = QPainterPath()
        path.moveTo(self.start_x, self.start_y)
        
        # Calculate control point
        mid_x = (self.start_x + self.end_x) / 2
        mid_y = (self.start_y + self.end_y) / 2
        
        # Add some random displacement
        disp_x = (random.random() - 0.5) * 30
        disp_y = (random.random() - 0.5) * 30
        
        # Create the curved path
        path.quadTo(mid_x + disp_x, mid_y + disp_y, self.end_x, self.end_y)
        
        self.setPath(path)
    
    def set_value(self, value):
        """Set connection activation value"""
        self.value = max(0.0, min(1.0, value))
        alpha = 30 + int(90 * self.value)
        self.setPen(QPen(QColor(0, 180, 255, alpha), 1, Qt.DashLine))
    
    def update_animation(self, dt=0.05):
        """Update animation state"""
        # Move particles along the line
        self.animation_phase += self.particle_speed * self.value * dt * 10
        if self.animation_phase > 1.0:
            self.animation_phase -= 1.0
        
        self.update()
    
    def paint(self, painter, option, widget):
        """Custom paint with animated particles"""
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # Draw the base path
        super().paint(painter, option, widget)
        
        # If sufficient value, draw particles
        if self.value > 0.2:
            # Number of particles scales with value
            num_particles = 1 + int(3 * self.value)
            particle_size = 3 * self.value
            
            path = self.path()
            path_length = path.length()
            
            # Set particle style
            painter.setPen(Qt.NoPen)
            
            for i in range(num_particles):
                # Calculate position along path
                pos = (self.animation_phase + i / num_particles) % 1.0
                
                # Get point at path percentage
                point = path.pointAtPercent(pos)
                
                # Particle glow effect - brighter in the middle of path
                particle_brightness = 1.0 - abs(pos - 0.5) * 2.0
                alpha = int(150 * particle_brightness * self.value)
                
                # Create gradient for particle
                gradient = QRadialGradient(point, particle_size * 2)
                gradient.setColorAt(0, QColor(100, 200, 255, alpha))
                gradient.setColorAt(1, QColor(0, 100, 200, 0))
                
                painter.setBrush(QBrush(gradient))
                painter.drawEllipse(point, particle_size, particle_size)

class CentralCore(QGraphicsEllipseItem):
    """Central holographic core of the consciousness system"""
    
    def __init__(self, x, y, radius, parent=None):
        super().__init__(-radius, -radius, radius * 2, radius * 2, parent)
        
        # Store properties
        self.setPos(x, y)
        self.radius = radius
        self.pulse_phase = 0.0
        self.rotation_phase = 0.0
        self.value = 0.5  # Core energy level
        
        # Style
        self.setPen(QPen(QColor(0, 200, 255, 150), 2))
        self.setBrush(QBrush(QColor(0, 50, 100, 100)))
        
        # Create the inner rotating parts
        self.create_inner_parts()
        
        # Create central value text
        self.value_text = QGraphicsSimpleTextItem(self)
        self.value_text.setFont(QFont("Consolas", 14, QFont.Bold))
        self.value_text.setBrush(QBrush(QColor(220, 255, 255)))
        self._update_value_text()
        
        # Create label
        self.label = QGraphicsSimpleTextItem(self)
        self.label.setFont(QFont("Consolas", 9))
        self.label.setBrush(QBrush(QColor(180, 220, 255)))
        self.label.setText("CONSCIOUSNESS")
        
        # Position the label in the center
        br = self.label.boundingRect()
        self.label.setPos(-br.width() / 2, radius / 2)
    
    def create_inner_parts(self):
        """Create the inner rotating elements"""
        # Inner ring
        self.inner_ring = QGraphicsEllipseItem(-self.radius * 0.7, -self.radius * 0.7, 
                                              self.radius * 1.4, self.radius * 1.4, self)
        self.inner_ring.setPen(QPen(QColor(0, 150, 255, 100), 1))
        self.inner_ring.setBrush(Qt.NoBrush)
        
        # Inner polygon
        polygon = QPolygonF()
        for i in range(6):
            angle = 2 * math.pi * i / 6
            x = self.radius * 0.5 * math.cos(angle)
            y = self.radius * 0.5 * math.sin(angle)
            polygon.append(QPointF(x, y))
        
        self.inner_polygon = QGraphicsPolygonItem(polygon, self)
        self.inner_polygon.setPen(QPen(QColor(100, 200, 255, 80), 1))
        self.inner_polygon.setBrush(QBrush(QColor(0, 100, 200, 40)))
    
    def _update_value_text(self):
        """Update the value text display"""
        self.value_text.setText(f"{self.value:.2f}")
        br = self.value_text.boundingRect()
        self.value_text.setPos(-br.width() / 2, -br.height() / 2 - self.radius / 4)
    
    def set_value(self, value):
        """Set the core value (0.0-1.0)"""
        self.value = max(0.0, min(1.0, value))
        self._update_value_text()
    
    def update_animation(self, dt=0.05):
        """Update animation state"""
        # Update pulse
        self.pulse_phase += dt
        if self.pulse_phase > math.pi * 2:
            self.pulse_phase -= math.pi * 2
        
        # Update rotation
        self.rotation_phase += dt * (0.5 + self.value * 0.5)
        
        # Rotate inner elements
        self.inner_polygon.setRotation(self.rotation_phase * 30)
        self.inner_ring.setRotation(-self.rotation_phase * 15)
        
        # Calculate pulse effect
        pulse = 0.5 + 0.5 * math.sin(self.pulse_phase)
        
        # Update core appearance based on value and pulse
        highlight_factor = 100 + int(80 * pulse * self.value)
        value_factor = 50 + int(150 * self.value)
        
        # Calculate color based on value
        r = min(255, int(0 + value_factor * 0.5))
        g = min(255, int(100 + value_factor * 0.6))
        b = min(255, int(200 + value_factor * 0.2))
        
        # Apply colors
        self.setPen(QPen(QColor(r, g, b, 150 + int(105 * pulse)), 
                         2 + self.value))
        self.setBrush(QBrush(QColor(r * 0.2, g * 0.2, b * 0.3, 
                                   80 + int(80 * self.value))))
        
        self.update()
    
    def paint(self, painter, option, widget):
        """Custom painting with extra effects"""
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # Pulse effect
        pulse = 0.5 + 0.5 * math.sin(self.pulse_phase)
        
        # Draw outer glow
        glow_radius = self.radius * (1.0 + 0.2 * pulse * self.value)
        gradient = QRadialGradient(0, 0, glow_radius)
        
        # Gradient colors based on value
        value_factor = 50 + int(150 * self.value)
        r = min(255, int(0 + value_factor * 0.5))
        g = min(255, int(100 + value_factor * 0.6))
        b = min(255, int(200 + value_factor * 0.2))
        
        gradient.setColorAt(0, QColor(r * 0.6, g * 0.6, b * 0.7, 
                                     100 + int(50 * pulse * self.value)))
        gradient.setColorAt(0.7, QColor(r * 0.3, g * 0.3, b * 0.5, 
                                      50 + int(30 * pulse * self.value)))
        gradient.setColorAt(1, QColor(0, 0, 0, 0))
        
        # Draw the glow
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(QRectF(-glow_radius, -glow_radius, 
                                   glow_radius * 2, glow_radius * 2))
        
        # Draw the standard item
        super().paint(painter, option, widget)

class MetricsHologramWidget(QGraphicsView):
    """Holographic visualization of consciousness metrics"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create scene
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(-200, -150, 400, 300)
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
        
        # Create metrics display
        self.central_core = None
        self.metric_nodes = {}
        self.connections = []
        self.metrics = {}
        
        self._create_metrics_display()
        
        # Animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animate_step)
        self.animation_timer.start(30)  # ~33 fps
    
    def _create_metrics_display(self):
        """Create the holographic metrics display components"""
        # Create central core
        self.central_core = CentralCore(0, 0, 40)
        self.scene.addItem(self.central_core)
        
        # Define metrics
        metric_defs = [
            {"name": "NEURAL", "x": -120, "y": -80, "radius": 25},
            {"name": "LINGUISTIC", "x": 120, "y": -80, "radius": 25},
            {"name": "MEMORY", "x": -120, "y": 80, "radius": 25},
            {"name": "INTEGRATION", "x": 120, "y": 80, "radius": 25},
            {"name": "AWARENESS", "x": 0, "y": -120, "radius": 25}
        ]
        
        # Create metric nodes
        for metric in metric_defs:
            node = MetricNode(
                metric["x"], metric["y"], metric["radius"], 
                metric["name"], random.uniform(0.5, 0.8)
            )
            self.scene.addItem(node)
            self.metric_nodes[metric["name"]] = node
            
            # Create connection to core
            conn = ConnectionLine(metric["x"], metric["y"], 0, 0)
            self.scene.addItem(conn)
            self.connections.append(conn)
            
            # Set initial connection value based on metric value
            conn.set_value(node.value)
    
    def update_metrics(self, metrics):
        """Update metrics values"""
        self.metrics = metrics
        
        # Map metrics to nodes
        mapping = {
            "neural_activity": "NEURAL",
            "linguistic_depth": "LINGUISTIC",
            "memory_coherence": "MEMORY",
            "system_integration": "INTEGRATION",
            "consciousness_level": "AWARENESS"
        }
        
        # Update nodes
        for metric_key, node_key in mapping.items():
            if metric_key in metrics and node_key in self.metric_nodes:
                value = metrics[metric_key]
                self.metric_nodes[node_key].set_value(value)
        
        # Update central core with consciousness level
        if "consciousness_level" in metrics:
            self.central_core.set_value(metrics["consciousness_level"])
    
    def animate_step(self):
        """Update animation state for all components"""
        # Update central core animation
        if self.central_core:
            self.central_core.update_animation()
        
        # Update metric nodes animation
        for node in self.metric_nodes.values():
            node.update_animation()
        
        # Update connection animations
        for conn in self.connections:
            conn.update_animation()
    
    def activate_animation(self):
        """Run activation animation sequence"""
        # Initial delay
        delay = 0
        
        # Activate central core
        QTimer.singleShot(delay, lambda: self.central_core.set_value(0.7))
        delay += 300
        
        # Activate nodes in sequence
        for node in self.metric_nodes.values():
            QTimer.singleShot(delay, lambda n=node: n.set_value(random.uniform(0.6, 0.9)))
            delay += 150
    
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