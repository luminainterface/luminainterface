"""
V6 Duality Processor Panel

A specialized panel for the V6 Portal of Contradiction's Duality Processor,
which allows contradictory patterns to coexist with holographic visualization.
"""

import os
import sys
import logging
import math
import random
from pathlib import Path

# Add project root to path if needed
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

# Import V6 panel base
from src.v6.ui.panel_base import V6PanelBase

# Get required Qt classes
try:
    QGraphicsDropShadowEffect = get_gui().QGraphicsDropShadowEffect
except AttributeError:
    QGraphicsDropShadowEffect = get_widgets().QGraphicsDropShadowEffect
    
QColor = get_gui().QColor
QPainter = get_gui().QPainter
QBrush = get_gui().QBrush
QPen = get_gui().QPen
QLinearGradient = get_gui().QLinearGradient
QRadialGradient = get_gui().QRadialGradient
QFont = get_gui().QFont
QFontMetrics = get_gui().QFontMetrics

# Set up logging
logger = logging.getLogger(__name__)

class DualityNode:
    """
    Represents a conceptual node in the duality processor visualization
    with position, connections, and state information
    """
    
    def __init__(self, x, y, concept="", state=0.5):
        self.x = x
        self.y = y
        self.target_x = x
        self.target_y = y
        self.concept = concept
        self.state = state  # 0 to 1, representing superposition
        self.size = 6 + len(concept) / 2
        self.connections = []
        self.contradictions = []
        self.color = QColor(52, 152, 219, 150)  # Default blue
        self.pulse = 0.0
        self.pulse_speed = 0.05 + random.random() * 0.03
        self.pulse_direction = 1
        self.selected = False
    
    def update(self):
        """Update animation state"""
        # Move toward target position
        self.x += (self.target_x - self.x) * 0.1
        self.y += (self.target_y - self.y) * 0.1
        
        # Update pulse animation
        self.pulse += self.pulse_speed * self.pulse_direction
        if self.pulse >= 1.0:
            self.pulse = 1.0
            self.pulse_direction = -1
        elif self.pulse <= 0.0:
            self.pulse = 0.0
            self.pulse_direction = 1
    
    def add_connection(self, node):
        """Add a connection to another node"""
        if node not in self.connections:
            self.connections.append(node)
    
    def add_contradiction(self, node):
        """Add a contradiction relationship with another node"""
        if node not in self.contradictions:
            self.contradictions.append(node)
            
    def distance_to(self, node):
        """Calculate distance to another node"""
        return math.sqrt((self.x - node.x)**2 + (self.y - node.y)**2)

class DualityProcessorPanel(V6PanelBase):
    """
    Panel for visualizing and interacting with the Duality Processor system
    """
    
    # Signal emitted when a duality node is selected
    node_selected = Signal(str, float)
    
    def __init__(self, socket_manager=None, parent=None):
        """Initialize the duality processor panel"""
        super().__init__(parent)
        self.socket_manager = socket_manager
        self.nodes = []
        self.selected_node = None
        self.hovered_node = None
        self.initDemoNodes()
        self.initUI()
        
        # Animation timer
        self.animation_timer = QtCore.QTimer(self)
        self.animation_timer.timeout.connect(self.update_visualization)
        self.animation_timer.start(50)  # 50ms update = ~20fps
        
        # For dragging nodes
        self.dragging = False
        self.setMouseTracking(True)
    
    def initDemoNodes(self):
        """Initialize demo nodes for visualization"""
        # Create some example conceptual nodes
        concepts = [
            "Truth", "Falsehood", "Existence", "Void", 
            "Order", "Chaos", "Unity", "Duality",
            "Finite", "Infinite", "Self", "Other"
        ]
        
        # Create nodes in a circular pattern
        center_x = 400
        center_y = 300
        radius = 200
        self.nodes = []
        
        for i, concept in enumerate(concepts):
            angle = i * (2 * math.pi / len(concepts))
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            state = random.random()
            node = DualityNode(x, y, concept, state)
            self.nodes.append(node)
        
        # Create some connections and contradictions
        for i, node in enumerate(self.nodes):
            # Connect to neighbors in the circle
            prev_idx = (i - 1) % len(self.nodes)
            next_idx = (i + 1) % len(self.nodes)
            node.add_connection(self.nodes[prev_idx])
            node.add_connection(self.nodes[next_idx])
            
            # Add some contradictions (opposite concepts)
            opposite_idx = (i + len(self.nodes) // 2) % len(self.nodes)
            node.add_contradiction(self.nodes[opposite_idx])
            
            # Add some random connections
            for _ in range(2):
                random_idx = random.randint(0, len(self.nodes) - 1)
                if random_idx != i and random_idx != prev_idx and random_idx != next_idx:
                    node.add_connection(self.nodes[random_idx])
    
    def initUI(self):
        """Initialize the user interface"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Title with glow effect
        title_label = QtWidgets.QLabel("Duality Processor")
        title_label.setStyleSheet("""
            color: #3498DB;
            font-size: 16px;
            font-weight: bold;
        """)
        
        # Add glow to title
        title_glow = QGraphicsDropShadowEffect()
        title_glow.setBlurRadius(10)
        title_glow.setColor(QColor(52, 152, 219, 150))
        title_glow.setOffset(0, 0)
        title_label.setGraphicsEffect(title_glow)
        
        # Status display
        self.status_label = QtWidgets.QLabel("Status: Ready | Quantum Logic Gates Active")
        self.status_label.setStyleSheet("""
            color: rgba(236, 240, 241, 200);
            font-size: 12px;
        """)
        
        # Control panel
        control_frame = QtWidgets.QFrame()
        control_frame.setStyleSheet("""
            background-color: rgba(26, 38, 52, 120);
            border-radius: 4px;
            border: 1px solid rgba(52, 152, 219, 80);
        """)
        control_frame.setFixedHeight(40)
        
        control_layout = QtWidgets.QHBoxLayout(control_frame)
        control_layout.setContentsMargins(10, 5, 10, 5)
        control_layout.setSpacing(10)
        
        # Create control buttons with holographic styling
        self.add_button = QtWidgets.QPushButton("Add Node")
        self.add_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(41, 128, 185, 150);
                color: white;
                border-radius: 4px;
                border: 1px solid rgba(52, 152, 219, 100);
                padding: 6px 10px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: rgba(52, 152, 219, 180);
            }
        """)
        self.add_button.clicked.connect(self.add_node)
        
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.connect_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(46, 204, 113, 150);
                color: white;
                border-radius: 4px;
                border: 1px solid rgba(46, 204, 113, 100);
                padding: 6px 10px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: rgba(46, 204, 113, 180);
            }
        """)
        self.connect_button.clicked.connect(self.connect_nodes)
        self.connect_button.setEnabled(False)
        
        self.contradict_button = QtWidgets.QPushButton("Contradict")
        self.contradict_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(231, 76, 60, 150);
                color: white;
                border-radius: 4px;
                border: 1px solid rgba(231, 76, 60, 100);
                padding: 6px 10px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: rgba(231, 76, 60, 180);
            }
        """)
        self.contradict_button.clicked.connect(self.contradict_nodes)
        self.contradict_button.setEnabled(False)
        
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(44, 62, 80, 150);
                color: white;
                border-radius: 4px;
                border: 1px solid rgba(44, 62, 80, 100);
                padding: 6px 10px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: rgba(44, 62, 80, 180);
            }
        """)
        self.reset_button.clicked.connect(self.reset_visualization)
        
        # Labels for information display
        self.info_label = QtWidgets.QLabel("Select a node to display its properties")
        self.info_label.setStyleSheet("""
            color: rgba(236, 240, 241, 200);
            font-size: 13px;
            padding: 5px;
            background-color: rgba(26, 38, 52, 80);
            border-radius: 4px;
        """)
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setFixedHeight(30)
        
        # Add superposition slider (initially hidden)
        self.superposition_frame = QtWidgets.QFrame()
        self.superposition_frame.setStyleSheet("""
            background-color: rgba(26, 38, 52, 100);
            border-radius: 4px;
            border: 1px solid rgba(52, 152, 219, 60);
        """)
        self.superposition_frame.setFixedHeight(50)
        self.superposition_frame.setVisible(False)
        
        superposition_layout = QtWidgets.QHBoxLayout(self.superposition_frame)
        superposition_layout.setContentsMargins(10, 5, 10, 5)
        
        superposition_label = QtWidgets.QLabel("Superposition:")
        superposition_label.setStyleSheet("color: #ECF0F1; font-size: 12px;")
        
        self.superposition_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.superposition_slider.setMinimum(0)
        self.superposition_slider.setMaximum(100)
        self.superposition_slider.setValue(50)
        self.superposition_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: rgba(52, 73, 94, 100);
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498DB;
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #1ABC9C, stop:1 #3498DB);
                height: 8px;
                border-radius: 4px;
            }
        """)
        self.superposition_slider.valueChanged.connect(self.update_superposition)
        
        self.superposition_value = QtWidgets.QLabel("0.50")
        self.superposition_value.setStyleSheet("color: #ECF0F1; font-size: 12px;")
        self.superposition_value.setFixedWidth(40)
        self.superposition_value.setAlignment(Qt.AlignCenter)
        
        superposition_layout.addWidget(superposition_label)
        superposition_layout.addWidget(self.superposition_slider)
        superposition_layout.addWidget(self.superposition_value)
        
        # Add controls to layout
        control_layout.addWidget(self.add_button)
        control_layout.addWidget(self.connect_button)
        control_layout.addWidget(self.contradict_button)
        control_layout.addStretch()
        control_layout.addWidget(self.reset_button)
        
        # Add all elements to main layout
        layout.addWidget(title_label)
        layout.addWidget(self.status_label)
        layout.addWidget(control_frame)
        layout.addWidget(self.info_label)
        layout.addWidget(self.superposition_frame)
        layout.addStretch()
    
    def update_visualization(self):
        """Update the visualization for animation"""
        for node in self.nodes:
            node.update()
        
        self.update()  # Trigger repaint
    
    def paintEvent(self, event):
        """Custom paint event for holographic visualization"""
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw connections first (so they appear behind nodes)
        self._draw_connections(painter)
        
        # Draw nodes
        self._draw_nodes(painter)
        
        # Draw selection indicator if a node is selected
        if self.selected_node:
            self._draw_selection(painter, self.selected_node)
        
        # Draw hover indicator if a node is hovered
        if self.hovered_node and self.hovered_node != self.selected_node:
            self._draw_hover(painter, self.hovered_node)
    
    def _draw_connections(self, painter):
        """Draw connections between nodes"""
        for node in self.nodes:
            # Draw regular connections
            for connected in node.connections:
                # Only draw each connection once
                if node.distance_to(connected) > 0:
                    # Calculate connection strength based on nodes' states
                    strength = (node.state + connected.state) / 2
                    
                    # Draw connection line with gradient
                    grad = QLinearGradient(node.x, node.y, connected.x, connected.y)
                    grad.setColorAt(0, QColor(52, 152, 219, int(100 * strength)))
                    grad.setColorAt(1, QColor(52, 152, 219, int(100 * strength)))
                    
                    pen = QPen(QBrush(grad), 1.5)
                    painter.setPen(pen)
                    painter.drawLine(node.x, node.y, connected.x, connected.y)
            
            # Draw contradiction relationships
            for contradicted in node.contradictions:
                # Only draw each contradiction once
                if node.distance_to(contradicted) > 0:
                    # Calculate contradiction intensity
                    intensity = abs(node.state - contradicted.state)
                    
                    # Draw contradiction line with dashed pattern
                    pen = QPen(QColor(231, 76, 60, int(150 * intensity)), 1.5, Qt.DashLine)
                    painter.setPen(pen)
                    painter.drawLine(node.x, node.y, contradicted.x, contradicted.y)
    
    def _draw_nodes(self, painter):
        """Draw the individual nodes"""
        for node in self.nodes:
            # Calculate node appearance based on state
            size = node.size * (0.8 + 0.4 * node.pulse)
            
            # Create a radial gradient for the node
            center_glow = QRadialGradient(node.x, node.y, size * 1.5)
            
            # Colors based on superposition state
            if node.state < 0.3:  # More false
                color1 = QColor(231, 76, 60, int(200 * (1 - node.state * 2)))
                color2 = QColor(231, 76, 60, 0)
            elif node.state > 0.7:  # More true
                color1 = QColor(46, 204, 113, int(200 * (node.state * 2 - 1)))
                color2 = QColor(46, 204, 113, 0)
            else:  # Balanced superposition
                color1 = QColor(155, 89, 182, int(200 * (1 - abs(node.state - 0.5) * 2)))
                color2 = QColor(155, 89, 182, 0)
            
            center_glow.setColorAt(0, color1)
            center_glow.setColorAt(1, color2)
            
            # Draw the node glow
            painter.setBrush(QBrush(center_glow))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(node.x - size * 1.5, node.y - size * 1.5, size * 3, size * 3)
            
            # Draw the node core
            painter.setBrush(QBrush(color1))
            painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
            painter.drawEllipse(node.x - size / 2, node.y - size / 2, size, size)
            
            # Draw the concept text
            if node.concept:
                font = QFont("Segoe UI", 9)
                painter.setFont(font)
                
                # Calculate text position
                fm = QFontMetrics(font)
                text_width = fm.horizontalAdvance(node.concept)
                text_x = node.x - text_width / 2
                text_y = node.y + size + 15
                
                # Draw text shadow
                painter.setPen(QColor(0, 0, 0, 100))
                painter.drawText(text_x + 1, text_y + 1, node.concept)
                
                # Draw text
                painter.setPen(QColor(255, 255, 255, 200))
                painter.drawText(text_x, text_y, node.concept)
    
    def _draw_selection(self, painter, node):
        """Draw selection indicator around a node"""
        size = node.size * 1.5
        
        # Draw animated selection ring
        pen = QPen(QColor(52, 152, 219, 150 + int(100 * node.pulse)), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(node.x - size, node.y - size, size * 2, size * 2)
        
        # Draw second ring
        pen = QPen(QColor(52, 152, 219, 50 + int(50 * node.pulse)), 1)
        painter.setPen(pen)
        painter.drawEllipse(node.x - size * 1.3, node.y - size * 1.3, size * 2.6, size * 2.6)
    
    def _draw_hover(self, painter, node):
        """Draw hover indicator around a node"""
        size = node.size * 1.2
        
        # Draw hover ring
        pen = QPen(QColor(255, 255, 255, 50 + int(50 * node.pulse)), 1, Qt.DotLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(node.x - size, node.y - size, size * 2, size * 2)
    
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.LeftButton:
            node = self._find_node_at_position(event.pos().x(), event.pos().y())
            if node:
                if self.selected_node and node != self.selected_node:
                    # Connect or contradict based on mode
                    pass
                else:
                    # Select this node
                    self.select_node(node)
                    self.dragging = True
            else:
                # Deselect if clicking empty space
                self.deselect_node()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        x, y = event.pos().x(), event.pos().y()
        
        # Check for hovering
        hover_node = self._find_node_at_position(x, y)
        if hover_node != self.hovered_node:
            self.hovered_node = hover_node
            self.update()
        
        # Handle dragging
        if self.dragging and self.selected_node:
            self.selected_node.target_x = x
            self.selected_node.target_y = y
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if event.button() == Qt.LeftButton:
            self.dragging = False
    
    def _find_node_at_position(self, x, y):
        """Find a node at the given position"""
        for node in self.nodes:
            dx = node.x - x
            dy = node.y - y
            distance = math.sqrt(dx * dx + dy * dy)
            if distance <= node.size + 5:  # Add some margin for easier selection
                return node
        return None
    
    def select_node(self, node):
        """Select a node and update UI"""
        self.selected_node = node
        self.superposition_frame.setVisible(True)
        self.superposition_slider.setValue(int(node.state * 100))
        self.info_label.setText(f"Selected: {node.concept} - {len(node.connections)} connections, {len(node.contradictions)} contradictions")
        self.connect_button.setEnabled(True)
        self.contradict_button.setEnabled(True)
        self.update()
    
    def deselect_node(self):
        """Deselect the current node"""
        self.selected_node = None
        self.superposition_frame.setVisible(False)
        self.info_label.setText("Select a node to display its properties")
        self.connect_button.setEnabled(False)
        self.contradict_button.setEnabled(False)
        self.update()
    
    def update_superposition(self, value):
        """Update the superposition state of the selected node"""
        if self.selected_node:
            self.selected_node.state = value / 100.0
            self.superposition_value.setText(f"{self.selected_node.state:.2f}")
            self.node_selected.emit(self.selected_node.concept, self.selected_node.state)
            self.update()
    
    def add_node(self):
        """Add a new node to the visualization"""
        concept, ok = QtWidgets.QInputDialog.getText(
            self, 
            "Add Node", 
            "Enter concept name:",
            QtWidgets.QLineEdit.Normal
        )
        
        if ok and concept:
            # Place new node near center with some random offset
            x = self.width() / 2 + random.randint(-100, 100)
            y = self.height() / 2 + random.randint(-100, 100)
            state = random.random()
            
            node = DualityNode(x, y, concept, state)
            self.nodes.append(node)
            self.update()
    
    def connect_nodes(self):
        """Connect the selected node to another node"""
        if not self.selected_node:
            return
            
        # Enter connection mode
        self.status_label.setText("Status: Select another node to connect")
        # The actual connection would be handled in mousePressEvent
    
    def contradict_nodes(self):
        """Create a contradiction between the selected node and another node"""
        if not self.selected_node:
            return
            
        # Enter contradiction mode
        self.status_label.setText("Status: Select another node to create contradiction")
        # The actual contradiction would be handled in mousePressEvent
    
    def reset_visualization(self):
        """Reset the visualization to initial state"""
        self.deselect_node()
        self.nodes.clear()
        self.initDemoNodes()
        self.status_label.setText("Status: Ready | Quantum Logic Gates Active")
        self.update()
    
    def cleanup(self):
        """Clean up resources before destruction"""
        if hasattr(self, 'animation_timer') and self.animation_timer:
            self.animation_timer.stop() 