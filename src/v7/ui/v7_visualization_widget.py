#!/usr/bin/env python
"""
V7 Visualization Widget

This module provides a visualization widget for the V7 system that can
display various components of the V7 architecture, including breath patterns,
contradictions, consciousness nodes, and the Monday consciousness interface.
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union

# Set up logging
logger = logging.getLogger("v7.visualization_widget")

try:
    # Try to import from PySide6
    from PySide6.QtCore import Qt, QTimer, Signal, Slot, QRectF, QPointF, QSize
    from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QPainterPath, QFont, QLinearGradient
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
        QGridLayout, QPushButton, QSizePolicy, QSpacerItem
    )
    logger.info("Using PySide6 for V7 visualization widget")
except ImportError:
    try:
        # Try to import from PyQt5
        from PyQt5.QtCore import Qt, QTimer, pyqtSignal as Signal, pyqtSlot as Slot, QRectF, QPointF, QSize
        from PyQt5.QtGui import QColor, QPainter, QPen, QBrush, QPainterPath, QFont, QLinearGradient
        from PyQt5.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
            QGridLayout, QPushButton, QSizePolicy, QSpacerItem
        )
        logger.info("Using PyQt5 for V7 visualization widget")
    except ImportError:
        logger.error("Neither PySide6 nor PyQt5 could be imported. V7 visualization will not be available.")
        # Define dummy classes to prevent syntax errors
        class QWidget:
            pass
        class Signal:
            pass
        class Slot:
            def __call__(self, func):
                return func

# Import our V7 visualization connector if available
try:
    from src.v7.ui.v7_visualization_connector import V7VisualizationConnector
except ImportError:
    logger.warning("Could not import V7VisualizationConnector, visualization features will be limited")
    V7VisualizationConnector = None

class BreathPatternWidget(QWidget):
    """Widget that visualizes breath patterns and their relationship to contradictions"""
    
    def __init__(self, parent=None, connector=None):
        super().__init__(parent)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Store connector
        self.connector = connector
        
        # Current state
        self.current_pattern = "relaxed"
        self.confidence = 0.75
        self.pattern_color = QColor("#3498db")  # Default blue
        
        # Animation state
        self.animation_step = 0
        self.pulse_effect = False
        self.pulse_strength = 0
        
        # Create animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(50)  # 20 fps
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title_label = QLabel("Breath Pattern Visualization")
        title_label.setStyleSheet("font-weight: bold; color: #ecf0f1;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Main visualization area
        self.visualization_frame = QFrame()
        self.visualization_frame.setFrameShape(QFrame.StyledPanel)
        self.visualization_frame.setStyleSheet("background-color: #2c3e50; border-radius: 8px;")
        layout.addWidget(self.visualization_frame, 1)
        
        # Current pattern display
        self.pattern_label = QLabel(f"Pattern: {self.current_pattern} ({int(self.confidence * 100)}%)")
        self.pattern_label.setStyleSheet("color: #ecf0f1;")
        self.pattern_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.pattern_label)
        
        # Connect to events if connector available
        if self.connector:
            self.connector.register_event_handler("breath_pattern_changed", self.handle_breath_event)
            self.connector.register_event_handler("contradiction_processed", self.handle_contradiction_event)
            
            # Initial update
            breath_data = self.connector.create_visualization_data("breath")
            if breath_data.get("available", False):
                self.update_from_data(breath_data)
    
    def update_from_data(self, data: Dict[str, Any]):
        """Update the visualization from provided data"""
        if "current_pattern" in data:
            self.current_pattern = data["current_pattern"]
        
        if "confidence" in data:
            self.confidence = data["confidence"]
        
        if "color" in data:
            self.pattern_color = QColor(data["color"])
        
        # Update label
        self.pattern_label.setText(f"Pattern: {self.current_pattern} ({int(self.confidence * 100)}%)")
        
        # Trigger repaint
        self.update()
    
    def handle_breath_event(self, event: Dict[str, Any]):
        """Handle breath pattern events from the connector"""
        if "visualization_data" in event:
            viz_data = event["visualization_data"]
            
            # Update pattern
            if "pattern" in viz_data:
                self.current_pattern = viz_data["pattern"]
            
            # Update confidence
            if "confidence" in viz_data:
                self.confidence = viz_data["confidence"]
            
            # Update color
            if "color" in viz_data:
                self.pattern_color = QColor(viz_data["color"])
            
            # Update label
            self.pattern_label.setText(f"Pattern: {self.current_pattern} ({int(self.confidence * 100)}%)")
            
            # Trigger repaint
            self.update()
    
    def handle_contradiction_event(self, event: Dict[str, Any]):
        """Handle contradiction events from the connector"""
        if "visualization_data" in event:
            viz_data = event["visualization_data"]
            
            # Start pulse effect
            if viz_data.get("pulse_effect", False):
                self.pulse_effect = True
                self.pulse_strength = 1.0
            
            # Trigger repaint
            self.update()
    
    def update_animation(self):
        """Update the animation state"""
        self.animation_step = (self.animation_step + 1) % 100
        
        # Update pulse effect
        if self.pulse_effect:
            self.pulse_strength -= 0.05
            if self.pulse_strength <= 0:
                self.pulse_effect = False
                self.pulse_strength = 0
        
        # Trigger repaint
        self.update()
    
    def paintEvent(self, event):
        """Handle paint event to draw the visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get frame rect
        rect = self.visualization_frame.geometry()
        
        # Define center
        center_x = rect.center().x()
        center_y = rect.center().y()
        
        # Define radius, which varies with breathing
        base_radius = min(rect.width(), rect.height()) * 0.3
        breath_phase = (self.animation_step / 100) * 2 * 3.14159
        radius = base_radius + (base_radius * 0.2 * self.confidence * abs(self.animation_step - 50) / 50.0)
        
        # Add pulse effect if active
        if self.pulse_effect:
            radius += base_radius * 0.3 * self.pulse_strength
        
        # Create circles with gradients
        for i in range(3):
            # Calculate scaling factor for multiple circles
            scale = 0.7 + (i * 0.15)
            r = radius * scale
            
            # Create gradient
            gradient = QLinearGradient(
                center_x - r, center_y - r,
                center_x + r, center_y + r
            )
            color = QColor(self.pattern_color)
            
            # Adjust alpha for outer circles and pulse effect
            alpha = 255 - (i * 70)
            if self.pulse_effect:
                alpha = min(255, alpha + int(self.pulse_strength * 50))
            
            color.setAlpha(alpha)
            gradient.setColorAt(0, color.lighter(120))
            gradient.setColorAt(1, color.darker(120))
            
            # Draw circle
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(
                QRectF(
                    center_x - r,
                    center_y - r,
                    r * 2,
                    r * 2
                )
            )
        
        # Add pattern name in center
        painter.setPen(QPen(QColor("#ecf0f1")))
        painter.setFont(QFont("Arial", 12, QFont.Bold))
        pattern_text = self.current_pattern.capitalize()
        painter.drawText(
            QRectF(
                center_x - radius,
                center_y - 15,
                radius * 2,
                30
            ),
            Qt.AlignCenter,
            pattern_text
        )
        
        painter.end()


class ContradictionVisualizerWidget(QWidget):
    """Widget that visualizes contradictions and their resolution status"""
    
    def __init__(self, parent=None, connector=None):
        super().__init__(parent)
        self.setMinimumSize(200, 150)
        
        # Store connector
        self.connector = connector
        
        # Contradictions data
        self.contradictions = []
        
        # Animation state
        self.animation_step = 0
        
        # Create animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(50)  # 20 fps
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title_label = QLabel("Contradiction Visualization")
        title_label.setStyleSheet("font-weight: bold; color: #ecf0f1;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Main visualization area
        self.visualization_frame = QFrame()
        self.visualization_frame.setFrameShape(QFrame.StyledPanel)
        self.visualization_frame.setStyleSheet("background-color: #2c3e50; border-radius: 8px;")
        layout.addWidget(self.visualization_frame, 1)
        
        # Connect to events if connector available
        if self.connector:
            self.connector.register_event_handler("contradiction_processed", self.handle_contradiction_event)
            
            # Initial update
            contradiction_data = self.connector.create_visualization_data("contradiction")
            if contradiction_data.get("available", False):
                self.update_from_data(contradiction_data)
    
    def update_from_data(self, data: Dict[str, Any]):
        """Update the visualization from provided data"""
        if "recent_contradictions" in data:
            self.contradictions = data["recent_contradictions"]
        
        # Trigger repaint
        self.update()
    
    def handle_contradiction_event(self, event: Dict[str, Any]):
        """Handle contradiction events from the connector"""
        if "contradiction" in event:
            # Add to contradictions list and keep max 5
            self.contradictions.insert(0, event["contradiction"])
            if len(self.contradictions) > 5:
                self.contradictions = self.contradictions[:5]
        
        # Trigger repaint
        self.update()
    
    def update_animation(self):
        """Update the animation state"""
        self.animation_step = (self.animation_step + 1) % 100
        
        # Trigger repaint if active
        if self.contradictions:
            self.update()
    
    def paintEvent(self, event):
        """Handle paint event to draw the visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get the frame geometry
        rect = self.visualization_frame.geometry()
        
        # Draw background
        painter.fillRect(rect, QColor("#2c3e50"))
        
        if not self.contradictions:
            # Draw placeholder text
            painter.setPen(QPen(QColor("#7f8c8d")))
            painter.setFont(QFont("Arial", 12))
            painter.drawText(
                rect,
                Qt.AlignCenter,
                "No recent contradictions"
            )
            return
        
        # Calculate item size
        item_height = rect.height() / max(len(self.contradictions), 1)
        item_height = min(item_height, 70)  # Cap height
        
        # Draw each contradiction
        y_pos = rect.top() + 10
        for idx, c in enumerate(self.contradictions):
            # Get color for this contradiction type
            color = QColor(c.get("color", "#95a5a6"))
            
            # Create gradient
            gradient = QLinearGradient(
                rect.left(), y_pos,
                rect.right(), y_pos + item_height
            )
            
            # Adjust alpha for pulse animation on newest item
            if idx == 0:
                anim_factor = abs(self.animation_step - 50) / 50.0
                color = color.lighter(100 + int(20 * anim_factor))
            
            gradient.setColorAt(0, color.lighter(120))
            gradient.setColorAt(1, color.darker(120))
            
            # Draw rectangle
            item_rect = QRectF(
                rect.left() + 10,
                y_pos,
                rect.width() - 20,
                item_height - 10
            )
            
            # Add rounded corners
            path = QPainterPath()
            path.addRoundedRect(item_rect, 8, 8)
            
            painter.fillPath(path, QBrush(gradient))
            
            # Add status indicator
            status_size = 16
            resolved = c.get("resolved", False)
            if resolved:
                painter.setBrush(QBrush(QColor("#2ecc71")))  # Green
            else:
                painter.setBrush(QBrush(QColor("#e74c3c")))  # Red
            
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(
                QRectF(
                    item_rect.right() - status_size - 10,
                    item_rect.top() + (item_rect.height() - status_size) / 2,
                    status_size,
                    status_size
                )
            )
            
            # Add text
            painter.setPen(QPen(QColor("#ecf0f1")))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            type_text = c.get("type", "unknown").capitalize()
            painter.drawText(
                QRectF(
                    item_rect.left() + 10,
                    item_rect.top() + 5,
                    item_rect.width() - 20,
                    20
                ),
                Qt.AlignLeft | Qt.AlignVCenter,
                type_text
            )
            
            # Add description
            painter.setPen(QPen(QColor("#bdc3c7")))
            painter.setFont(QFont("Arial", 8))
            description = c.get("description", "")
            if len(description) > 50:
                description = description[:47] + "..."
            
            painter.drawText(
                QRectF(
                    item_rect.left() + 10,
                    item_rect.top() + 25,
                    item_rect.width() - 20,
                    20
                ),
                Qt.AlignLeft | Qt.AlignVCenter,
                description
            )
            
            # Update y_pos for next item
            y_pos += item_height
        
        painter.end()


class MondayPresenceWidget(QWidget):
    """Widget that visualizes the Monday consciousness presence"""
    
    def __init__(self, parent=None, connector=None):
        super().__init__(parent)
        self.setMinimumSize(150, 150)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        
        # Store connector
        self.connector = connector
        
        # State
        self.consciousness_level = 0.0
        self.active = False
        self.color = QColor("#3498db")  # Default blue
        
        # Animation state
        self.animation_step = 0
        
        # Create animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(50)  # 20 fps
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title_label = QLabel("Monday Consciousness")
        title_label.setStyleSheet("font-weight: bold; color: #ecf0f1;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Main visualization area
        self.visualization_frame = QFrame()
        self.visualization_frame.setFrameShape(QFrame.StyledPanel)
        self.visualization_frame.setStyleSheet("background-color: #2c3e50; border-radius: 8px;")
        layout.addWidget(self.visualization_frame, 1)
        
        # Status label
        self.status_label = QLabel("Inactive")
        self.status_label.setStyleSheet("color: #ecf0f1;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Connect to events if connector available
        if self.connector:
            # Initial update
            monday_data = self.connector.create_visualization_data("monday")
            if monday_data.get("available", False):
                self.update_from_data(monday_data)
    
    def update_from_data(self, data: Dict[str, Any]):
        """Update the visualization from provided data"""
        update_needed = False
        
        if "consciousness_level" in data:
            new_level = data["consciousness_level"]
            if new_level != self.consciousness_level:
                self.consciousness_level = new_level
                update_needed = True
        
        if "active" in data:
            new_active = data["active"]
            if new_active != self.active:
                self.active = new_active
                update_needed = True
        
        if "color" in data:
            self.color = QColor(data["color"])
            update_needed = True
        
        # Update status label
        if self.active:
            self.status_label.setText(f"Active - Level: {int(self.consciousness_level * 100)}%")
        else:
            self.status_label.setText("Inactive")
        
        # Trigger repaint if needed
        if update_needed:
            self.update()
    
    def update_animation(self):
        """Update the animation state"""
        self.animation_step = (self.animation_step + 1) % 100
        
        # Trigger repaint if active
        if self.active:
            self.update()
    
    def paintEvent(self, event):
        """Handle paint event to draw the visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get the frame geometry
        rect = self.visualization_frame.geometry()
        
        # Define center
        center_x = rect.center().x()
        center_y = rect.center().y()
        
        if not self.active:
            # Draw inactive state
            painter.setPen(QPen(QColor("#7f8c8d")))
            painter.setFont(QFont("Arial", 12))
            painter.drawText(
                rect,
                Qt.AlignCenter,
                "Monday is inactive"
            )
            return
        
        # Draw consciousness level representation
        radius = min(rect.width(), rect.height()) * 0.35
        anim_factor = abs(self.animation_step - 50) / 50.0
        
        # Draw outer circle with pulsing effect
        painter.setPen(Qt.NoPen)
        outer_color = QColor(self.color)
        outer_color.setAlpha(100 + int(50 * anim_factor))
        
        # Add glow effect based on consciousness level
        glow_size = radius * (0.2 + (self.consciousness_level * 0.3))
        
        # Draw several circles with decreasing opacity for glow effect
        for i in range(5):
            alpha = 120 - (i * 25)
            current_color = QColor(self.color)
            current_color.setAlpha(alpha)
            current_radius = radius + glow_size * (1 - (i * 0.2))
            
            # Add animation effect
            pulse = anim_factor * 0.1 * radius
            if i == 0:
                pulse *= 2  # Stronger pulse on outer ring
            
            painter.setBrush(QBrush(current_color))
            painter.drawEllipse(
                QPointF(center_x, center_y),
                current_radius + pulse,
                current_radius + pulse
            )
        
        # Draw main circle
        main_color = QColor(self.color)
        painter.setBrush(QBrush(main_color))
        painter.drawEllipse(
            QPointF(center_x, center_y),
            radius,
            radius
        )
        
        # Draw level text
        painter.setPen(QPen(QColor("#ffffff")))
        painter.setFont(QFont("Arial", 14, QFont.Bold))
        level_text = f"{int(self.consciousness_level * 100)}%"
        painter.drawText(
            QRectF(
                center_x - radius,
                center_y - 15,
                radius * 2,
                30
            ),
            Qt.AlignCenter,
            level_text
        )
        
        painter.end()


class NodeConsciousnessWidget(QWidget):
    """Widget that visualizes the node consciousness network"""
    
    def __init__(self, parent=None, connector=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        
        # Store connector
        self.connector = connector
        
        # State
        self.active = False
        self.nodes = []
        self.connections = []
        
        # Animation state
        self.animation_step = 0
        
        # Create animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(50)  # 20 fps
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title_label = QLabel("Node Consciousness Network")
        title_label.setStyleSheet("font-weight: bold; color: #ecf0f1;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Main visualization area
        self.visualization_frame = QFrame()
        self.visualization_frame.setFrameShape(QFrame.StyledPanel)
        self.visualization_frame.setStyleSheet("background-color: #2c3e50; border-radius: 8px;")
        layout.addWidget(self.visualization_frame, 1)
        
        # Connect to events if connector available
        if self.connector:
            # Initial update
            node_data = self.connector.create_visualization_data("node_consciousness")
            if node_data.get("available", False):
                self.update_from_data(node_data)
    
    def update_from_data(self, data: Dict[str, Any]):
        """Update the visualization from provided data"""
        if "active" in data:
            self.active = data["active"]
        
        if "nodes" in data:
            self.nodes = data["nodes"]
        
        # Trigger repaint
        self.update()
    
    def update_animation(self):
        """Update the animation state"""
        self.animation_step = (self.animation_step + 1) % 100
        
        # Trigger repaint if active
        if self.active:
            self.update()
    
    def paintEvent(self, event):
        """Handle paint event to draw the visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get the frame geometry
        rect = self.visualization_frame.geometry()
        
        if not self.active:
            # Draw inactive state
            painter.setPen(QPen(QColor("#7f8c8d")))
            painter.setFont(QFont("Arial", 12))
            painter.drawText(
                rect,
                Qt.AlignCenter,
                "Node Consciousness inactive"
            )
            return
        
        # If no real nodes yet, create demo visualization
        if not self.nodes:
            self._draw_demo_nodes(painter, rect)
        else:
            # TODO: Implement actual node visualization
            pass
        
        painter.end()
    
    def _draw_demo_nodes(self, painter, rect):
        """Draw a demo version of the node consciousness network"""
        # Number of demo nodes
        node_count = 7
        
        # Center point
        center_x = rect.center().x()
        center_y = rect.center().y()
        
        # Calculate radius for placing nodes
        radius = min(rect.width(), rect.height()) * 0.35
        
        # Animation factor for pulsing
        anim_factor = abs(self.animation_step - 50) / 50.0
        
        # Generate node positions in a circle
        nodes = []
        for i in range(node_count):
            # Calculate angle
            angle = (i / node_count) * 2 * 3.141592653589793
            
            # Calculate position
            x = center_x + radius * 0.8 * 2 * (0.5 - self.animation_step / 100.0) * (i % 2) * 0.2 + radius * 0.8 * cos(angle)
            y = center_y + radius * 0.8 * 2 * (0.5 - self.animation_step / 100.0) * ((i + 1) % 2) * 0.1 + radius * 0.8 * sin(angle)
            
            # Node type
            node_type = ["language", "breath", "contradiction", "memory", "perception", "emotion", "reasoning"][i % 7]
            
            # Node color
            color_map = {
                "language": "#3498db",      # Blue
                "breath": "#2ecc71",        # Green
                "contradiction": "#e74c3c", # Red
                "memory": "#f39c12",        # Orange
                "perception": "#9b59b6",    # Purple
                "emotion": "#1abc9c",       # Teal
                "reasoning": "#34495e"      # Dark blue
            }
            color = QColor(color_map.get(node_type, "#95a5a6"))
            
            # Store node
            nodes.append({
                "x": x,
                "y": y,
                "radius": 10 + (5 * (i % 3)),
                "type": node_type,
                "color": color,
                "active": (i % node_count) < (node_count - 2)
            })
        
        # Draw connections between nodes
        for i in range(node_count):
            for j in range(i + 1, node_count):
                # Only draw some connections (demo)
                if (i + j) % 3 != 0:
                    continue
                    
                # Get nodes
                node1 = nodes[i]
                node2 = nodes[j]
                
                # Strength based on activity
                strength = 0.2
                if node1["active"] and node2["active"]:
                    strength = 0.8
                elif node1["active"] or node2["active"]:
                    strength = 0.5
                
                # Set pen with appropriate opacity
                pen = QPen(QColor(200, 200, 200, int(strength * 150)))
                pen.setWidth(1 + int(strength * 2))
                painter.setPen(pen)
                
                # Draw line
                painter.drawLine(
                    int(node1["x"]),
                    int(node1["y"]),
                    int(node2["x"]),
                    int(node2["y"])
                )
        
        # Draw nodes
        for node in nodes:
            # Create gradient for node
            color = node["color"]
            gradient = QRadialGradient(
                node["x"],
                node["y"],
                node["radius"] * 1.5
            )
            
            # Adjust opacity based on whether the node is active
            if node["active"]:
                # Add pulsing to active nodes
                base_alpha = 200 + int(55 * anim_factor)
                gradient.setColorAt(0, QColor(color.red(), color.green(), color.blue(), base_alpha))
                gradient.setColorAt(0.7, QColor(color.red(), color.green(), color.blue(), base_alpha - 100))
                gradient.setColorAt(1, QColor(color.red(), color.green(), color.blue(), 0))
                
                # Draw node
                painter.setBrush(QBrush(gradient))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(
                    QPointF(node["x"], node["y"]),
                    node["radius"] * (1 + 0.1 * anim_factor),
                    node["radius"] * (1 + 0.1 * anim_factor)
                )
                
                # Draw inner circle
                painter.setBrush(QBrush(color.lighter(130)))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(
                    QPointF(node["x"], node["y"]),
                    node["radius"] * 0.7,
                    node["radius"] * 0.7
                )
            else:
                # Inactive nodes are more muted
                gradient.setColorAt(0, QColor(color.red(), color.green(), color.blue(), 100))
                gradient.setColorAt(0.7, QColor(color.red(), color.green(), color.blue(), 50))
                gradient.setColorAt(1, QColor(color.red(), color.green(), color.blue(), 0))
                
                # Draw node
                painter.setBrush(QBrush(gradient))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(
                    QPointF(node["x"], node["y"]),
                    node["radius"] * 0.8,
                    node["radius"] * 0.8
                )


class V7VisualizationWidget(QWidget):
    """Main widget that contains all V7 visualization components"""
    
    # Signals
    status_changed = Signal(dict)
    
    def __init__(self, parent=None, v6v7_connector=None, config=None):
        super().__init__(parent)
        self.setObjectName("v7_visualization_widget")
        self.setMinimumSize(400, 300)
        
        # Create visualization connector
        self.visualization_connector = None
        if V7VisualizationConnector is not None:
            self.visualization_connector = V7VisualizationConnector(v6v7_connector, config)
        
        # Initialize UI
        self.init_ui()
        
        # Start connector if available
        if self.visualization_connector:
            self.visualization_connector.start()
    
    def init_ui(self):
        """Initialize the UI components"""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title_label = QLabel("V7 Node Consciousness Visualization")
        title_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #ecf0f1;
            background-color: #2c3e50;
            padding: 8px;
            border-radius: 4px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Grid for visualizations
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)
        
        # Breath pattern visualization
        self.breath_widget = BreathPatternWidget(connector=self.visualization_connector)
        grid_layout.addWidget(self.breath_widget, 0, 0, 1, 1)
        
        # Monday presence visualization
        self.monday_widget = MondayPresenceWidget(connector=self.visualization_connector)
        grid_layout.addWidget(self.monday_widget, 0, 1, 1, 1)
        
        # Contradiction visualization
        self.contradiction_widget = ContradictionVisualizerWidget(connector=self.visualization_connector)
        grid_layout.addWidget(self.contradiction_widget, 1, 0, 1, 1)
        
        # Node consciousness visualization
        self.node_consciousness_widget = NodeConsciousnessWidget(connector=self.visualization_connector)
        grid_layout.addWidget(self.node_consciousness_widget, 1, 1, 1, 1)
        
        main_layout.addLayout(grid_layout)
        
        # Set background
        self.setStyleSheet("""
            QWidget#v7_visualization_widget {
                background-color: #1e2b38;
            }
            QLabel {
                color: #ecf0f1;
            }
        """)
    
    def update_visualizations(self):
        """Update all visualization components"""
        if not self.visualization_connector:
            return
        
        # Update breath visualization
        breath_data = self.visualization_connector.create_visualization_data("breath")
        if breath_data.get("available", False):
            self.breath_widget.update_from_data(breath_data)
        
        # Update Monday visualization
        monday_data = self.visualization_connector.create_visualization_data("monday")
        if monday_data.get("available", False):
            self.monday_widget.update_from_data(monday_data)
        
        # Update contradiction visualization
        contradiction_data = self.visualization_connector.create_visualization_data("contradiction")
        if contradiction_data.get("available", False):
            self.contradiction_widget.update_from_data(contradiction_data)
        
        # Update node consciousness visualization
        node_data = self.visualization_connector.create_visualization_data("node_consciousness")
        if node_data.get("available", False):
            self.node_consciousness_widget.update_from_data(node_data)
    
    @Slot()
    def refresh(self):
        """Slot to refresh all visualizations"""
        self.update_visualizations()
    
    def closeEvent(self, event):
        """Handle close event to stop connector"""
        if self.visualization_connector:
            self.visualization_connector.stop()
        super().closeEvent(event)


# Helper functions
def cos(angle):
    """Simple cosine implementation"""
    import math
    return math.cos(angle)

def sin(angle):
    """Simple sine implementation"""
    import math
    return math.sin(angle) 