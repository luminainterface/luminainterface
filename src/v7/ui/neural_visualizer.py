"""
Neural Network Visualizer for V7 Holographic Interface

Provides a futuristic, holographic visualization of neural network activity
"""

import random
import math
from typing import Dict, List, Any

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QWidget, QVBoxLayout,
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsSimpleTextItem
)
from PySide6.QtCore import (
    Qt, QRectF, QLineF, QPointF, QTimer, QPropertyAnimation,
    QEasingCurve, Slot, Property
)
from PySide6.QtGui import (
    QPainter, QColor, QBrush, QPen, QRadialGradient, QFont,
    QPainterPath, QTransform
)

class NeuronItem(QGraphicsEllipseItem):
    """A neural network node with holographic glow effects"""
    
    def __init__(self, x, y, radius, neuron_type, parent=None):
        # Create a slightly larger rect for the full visual including glow
        visual_radius = radius * 1.5
        super().__init__(-visual_radius, -visual_radius, visual_radius * 2, visual_radius * 2, parent)
        
        # Store properties
        self.center_x = x
        self.center_y = y
        self.radius = radius
        self.visual_radius = visual_radius
        self.neuron_type = neuron_type
        self.activation = 0.0
        self.pulse_phase = random.random() * math.pi * 2  # Random starting phase
        
        # Set position
        self.setPos(x, y)
        
        # Set style based on neuron type
        if neuron_type == "input":
            self.base_color = QColor(0, 255, 200)  # Cyan
        elif neuron_type == "output":
            self.base_color = QColor(255, 100, 100)  # Red
        else:
            self.base_color = QColor(100, 180, 255)  # Blue
        
        # No need for standard brush - we'll custom paint
        self.setBrush(Qt.NoBrush)
        self.setPen(Qt.NoPen)
        
        # Enable custom painting
        self.setFlag(QGraphicsEllipseItem.ItemIsSelectable, True)
        
        # Create label
        self.label = QGraphicsSimpleTextItem(self)
        self.label.setFont(QFont("Consolas", 8))
        self.label.setBrush(QBrush(QColor(220, 220, 220)))
        self.label.setText(neuron_type[:3].upper())
        # Center the text
        br = self.label.boundingRect()
        self.label.setPos(-br.width() / 2, -br.height() / 2)
    
    def set_activation(self, value):
        """Set the neuron activation level (0.0-1.0)"""
        self.activation = max(0.0, min(1.0, value))
        self.update()
    
    def update_pulse(self, phase_increment=0.05):
        """Update the pulse animation phase"""
        self.pulse_phase += phase_increment
        if self.pulse_phase > math.pi * 2:
            self.pulse_phase -= math.pi * 2
        self.update()
    
    def paint(self, painter, option, widget):
        """Custom paint method for holographic effect"""
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # Calculate actual visual size based on activation
        visual_size = self.radius * (0.8 + 0.4 * self.activation)
        
        # Pulse effect modifier (0.0-1.0)
        pulse = 0.5 + 0.5 * math.sin(self.pulse_phase)
        
        # Create outer glow with gradient
        glow_radius = visual_size * (1.5 + 0.5 * pulse * self.activation)
        gradient = QRadialGradient(0, 0, glow_radius)
        
        # Color based on type and activation
        color = self.base_color.lighter(100 + int(50 * self.activation))
        
        # Set gradient colors
        gradient.setColorAt(0, QColor(color.red(), color.green(), color.blue(), 
                                     150 + int(105 * self.activation)))
        gradient.setColorAt(0.6, QColor(color.red(), color.green(), color.blue(), 
                                       80 + int(50 * self.activation)))
        gradient.setColorAt(1, QColor(color.red(), color.green(), color.blue(), 0))
        
        # Draw outer glow
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(QRectF(-glow_radius, -glow_radius, 
                                   glow_radius * 2, glow_radius * 2))
        
        # Draw core with solid color
        painter.setBrush(QBrush(color.lighter(100 + int(50 * pulse * self.activation))))
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
        painter.drawEllipse(QRectF(-visual_size, -visual_size, 
                                   visual_size * 2, visual_size * 2))
        
        # Draw center highlight
        highlight_size = visual_size * 0.4
        highlight_gradient = QRadialGradient(0, 0, highlight_size)
        highlight_gradient.setColorAt(0, QColor(255, 255, 255, 180))
        highlight_gradient.setColorAt(1, QColor(255, 255, 255, 0))
        painter.setBrush(QBrush(highlight_gradient))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QRectF(-highlight_size, -highlight_size, 
                                  highlight_size * 2, highlight_size * 2))

class ConnectionItem(QGraphicsLineItem):
    """A neural connection with energy flow animation"""
    
    def __init__(self, source_x, source_y, target_x, target_y, weight=0.5, parent=None):
        super().__init__(source_x, source_y, target_x, target_y, parent)
        
        # Store properties
        self.source_x = source_x
        self.source_y = source_y
        self.target_x = target_x
        self.target_y = target_y
        self.weight = weight
        self.activation = 0.0
        self.flow_phase = random.random()  # 0.0-1.0 for animation position
        
        # Set visual style
        self.base_color = QColor(100, 200, 255)  # Blue-cyan
        
        # No pen - we'll custom paint
        self.setPen(Qt.NoPen)
    
    def set_activation(self, value):
        """Set connection activation level (0.0-1.0)"""
        self.activation = max(0.0, min(1.0, value))
        self.update()
    
    def update_flow(self, phase_increment=0.02):
        """Update flow animation phase"""
        self.flow_phase += phase_increment * (0.5 + 0.8 * self.activation)
        if self.flow_phase > 1.0:
            self.flow_phase -= 1.0
        self.update()
    
    def paint(self, painter, option, widget):
        """Custom paint method for energy flow animation"""
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # Get line direction
        line = self.line()
        length = line.length()
        
        if length == 0:
            return
        
        # Line width based on weight and activation
        width = 1 + 3 * self.weight * (0.3 + 0.7 * self.activation)
        
        # Base line color with transparency based on activation
        line_alpha = 40 + int(60 * self.activation * self.weight)
        line_color = QColor(80, 180, 255, line_alpha)
        
        # Draw base line
        painter.setPen(QPen(line_color, width, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(line)
        
        # Draw animated flow particles if sufficiently activated
        if self.activation > 0.1:
            num_particles = int(5 * self.weight * self.activation)
            particle_spacing = 1.0 / max(1, num_particles)
            
            # Calculate unit vector for direction
            dx = line.dx() / length
            dy = line.dy() / length
            
            # Base particle color
            particle_color = self.base_color.lighter(100 + int(50 * self.activation))
            
            for i in range(num_particles):
                # Calculate position along the line (0.0-1.0)
                position = (self.flow_phase + i * particle_spacing) % 1.0
                
                # Particle visual parameters based on position and activation
                # Particles brighten in the middle of their journey
                brightness_factor = 1.0 - abs(position - 0.5) * 2
                particle_alpha = int(200 * brightness_factor * self.activation)
                particle_size = width * (1.0 + brightness_factor)
                
                # Calculate position coordinates
                px = line.x1() + position * line.dx()
                py = line.y1() + position * line.dy()
                
                # Draw particle
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(
                    particle_color.red(),
                    particle_color.green(),
                    particle_color.blue(),
                    particle_alpha
                )))
                painter.drawEllipse(QPointF(px, py), particle_size, particle_size)

class NeuralNetworkVisualizer(QGraphicsView):
    """Interactive visualization of neural network with holographic effects"""
    
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
        
        # Store neurons and connections
        self.neurons = {}  # id -> NeuronItem
        self.connections = {}  # id -> ConnectionItem
        
        # Animation properties
        self.animation_phase = 0.0
        
        # Create grid lines
        self._create_grid()
    
    def _create_grid(self):
        """Create holographic grid in the background"""
        # Horizontal lines
        for y in range(-200, 201, 40):
            line = QGraphicsLineItem(-300, y, 300, y)
            line.setPen(QPen(QColor(0, 100, 150, 30), 1, Qt.SolidLine))
            self.scene.addItem(line)
        
        # Vertical lines
        for x in range(-300, 301, 40):
            line = QGraphicsLineItem(x, -200, x, 200)
            line.setPen(QPen(QColor(0, 100, 150, 30), 1, Qt.SolidLine))
            self.scene.addItem(line)
    
    def update_network(self, neurons_data, connections_data, activation_levels=None):
        """Update the visualization with new network data"""
        # Remove existing items
        for neuron_id, item in self.neurons.items():
            if neuron_id not in neurons_data:
                self.scene.removeItem(item)
        
        for conn_id, item in self.connections.items():
            if conn_id not in connections_data:
                self.scene.removeItem(item)
        
        # Create/update neurons
        for neuron_id, data in neurons_data.items():
            # Scale position to scene coordinates
            x = (data.get("position", [0.5, 0.5])[0] - 0.5) * 500
            y = (data.get("position", [0.5, 0.5])[1] - 0.5) * 300
            neuron_type = data.get("type", "hidden")
            
            if neuron_id in self.neurons:
                # Update existing neuron
                self.neurons[neuron_id].setPos(x, y)
            else:
                # Create new neuron
                neuron = NeuronItem(x, y, 10, neuron_type)
                self.scene.addItem(neuron)
                self.neurons[neuron_id] = neuron
        
        # Create/update connections
        for conn_id, data in connections_data.items():
            source_id = data.get("source", "")
            target_id = data.get("target", "")
            weight = data.get("weight", 0.5)
            
            if source_id in self.neurons and target_id in self.neurons:
                source = self.neurons[source_id]
                target = self.neurons[target_id]
                
                if conn_id in self.connections:
                    # Update existing connection
                    conn = self.connections[conn_id]
                    conn.setLine(QLineF(source.x(), source.y(), target.x(), target.y()))
                else:
                    # Create new connection
                    conn = ConnectionItem(source.x(), source.y(), target.x(), target.y(), weight)
                    self.scene.addItem(conn)
                    self.connections[conn_id] = conn
        
        # Update activation levels
        if activation_levels:
            for neuron_id, activation in activation_levels.items():
                if neuron_id in self.neurons:
                    self.neurons[neuron_id].set_activation(activation)
            
            # For connections, use source neuron activation
            for conn_id, conn in self.connections.items():
                data = connections_data.get(conn_id, {})
                source_id = data.get("source", "")
                if source_id in activation_levels:
                    conn.set_activation(activation_levels[source_id])
    
    def animate_step(self):
        """Perform one step of the animation"""
        # Update neuron pulse effects
        for neuron in self.neurons.values():
            neuron.update_pulse()
        
        # Update connection flow animations
        for conn in self.connections.values():
            conn.update_flow()
        
        # Increment overall animation phase
        self.animation_phase += 0.02
        if self.animation_phase > math.pi * 2:
            self.animation_phase -= math.pi * 2
    
    def activate_animation(self):
        """Play an activation animation for the network"""
        # Activate neurons in sequence
        delay = 0
        for neuron_id, neuron in self.neurons.items():
            # Create animation to fade in neuron
            QTimer.singleShot(delay, lambda n=neuron: n.set_activation(0.8))
            delay += 100
        
        # Activate connections with slightly more delay
        delay = 300
        for conn_id, conn in self.connections.items():
            QTimer.singleShot(delay, lambda c=conn: c.set_activation(0.7))
            delay += 50
    
    def resizeEvent(self, event):
        """Handle resize event"""
        # Scale view to fit
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)
    
    def showEvent(self, event):
        """Handle show event"""
        # Scale view to fit when shown
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        super().showEvent(event) 