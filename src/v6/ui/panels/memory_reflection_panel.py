"""
V6 Memory Reflection Panel

Panel for the V6 Portal of Contradiction's Memory Reflection System,
which enables meta-cognitive processing and memory introspection.
"""

import os
import sys
import logging
import math
import random
from pathlib import Path
from datetime import datetime, timedelta

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

# Set up logging
logger = logging.getLogger(__name__)

class MemoryNode:
    """Represents a memory node in the visualization"""
    
    def __init__(self, text, timestamp, x, y, importance=0.5):
        self.text = text
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.target_x = x
        self.target_y = y
        self.importance = importance  # 0.0 to 1.0
        self.connections = []
        self.radius = 30 + importance * 20
        self.glow = 0.0
        self.selected = False
        self.contextual_shift = 0.0  # For recontextualization visualization
        
        # Animation properties
        self.pulse_speed = 0.02 + random.random() * 0.02
        self.pulse_direction = 1
    
    def update(self):
        """Update animation state"""
        # Move toward target position
        self.x += (self.target_x - self.x) * 0.1
        self.y += (self.target_y - self.y) * 0.1
        
        # Update pulse animation
        self.glow += self.pulse_speed * self.pulse_direction
        if self.glow >= 1.0:
            self.glow = 1.0
            self.pulse_direction = -1
        elif self.glow <= 0.0:
            self.glow = 0.0
            self.pulse_direction = 1

class MemoryReflectionPanel(V6PanelBase):
    """
    Panel for the Memory Reflection System - visualizes memory introspection,
    recontextualization, and temporal integration capabilities.
    """
    
    # Signal emitted when a memory is selected for inspection
    memory_selected = Signal(str, object)  # Text, timestamp
    
    def __init__(self, socket_manager=None, parent=None):
        """Initialize the memory reflection panel"""
        super().__init__(parent)
        self.socket_manager = socket_manager
        self.memory_nodes = []
        self.selected_node = None
        self.recontextualization_active = False
        self.temporal_view = False
        
        # For visual effects
        self.background_shift = 0.0
        self.shift_direction = 1
        
        # Initialize with demo data
        self.create_demo_memory_nodes()
        
        # Initialize UI
        self.initUI()
        
        # Animation timer
        self.animation_timer = QtCore.QTimer(self)
        self.animation_timer.timeout.connect(self.update_visualization)
        self.animation_timer.start(40)  # 40ms update = 25fps
        
        # For mouse interaction
        self.setMouseTracking(True)
    
    def create_demo_memory_nodes(self):
        """Create demo memory nodes for visualization"""
        # Sample memory texts
        memory_texts = [
            "Initial conversation about duality concepts",
            "Exploration of paradox resolution techniques",
            "Discussion of quantum logic principles",
            "Analysis of contradictory statements",
            "Review of multi-dimensional thinking framework",
            "Integration of opposing viewpoints",
            "Temporal pattern recognition session",
            "Metacognitive awareness enhancement"
        ]
        
        # Create nodes spread across visualization area
        now = datetime.now()
        center_x, center_y = 400, 300
        radius = 200
        
        for i, text in enumerate(memory_texts):
            # Create timestamp spreading back in time
            timestamp = now - timedelta(hours=(len(memory_texts)-i) * 2)
            
            # Position in a spiral pattern
            angle = i * 0.8
            r = 100 + i * 20
            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            
            # Random importance level
            importance = 0.3 + random.random() * 0.7
            
            # Create node
            node = MemoryNode(text, timestamp, x, y, importance)
            self.memory_nodes.append(node)
        
        # Create some connections between related memories
        for i, node in enumerate(self.memory_nodes):
            # Connect to some other nodes
            num_connections = random.randint(1, 3)
            for _ in range(num_connections):
                # Avoid self-connections
                target_idx = (i + random.randint(1, len(self.memory_nodes) - 1)) % len(self.memory_nodes)
                target_node = self.memory_nodes[target_idx]
                if target_node not in node.connections:
                    node.connections.append(target_node)
    
    def initUI(self):
        """Initialize the user interface"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Control panel at top
        control_panel = QtWidgets.QFrame()
        control_panel.setStyleSheet("""
            background-color: rgba(26, 38, 52, 120);
            border-radius: 4px;
            border: 1px solid rgba(52, 152, 219, 80);
        """)
        control_panel.setFixedHeight(40)
        
        control_layout = QtWidgets.QHBoxLayout(control_panel)
        control_layout.setContentsMargins(10, 0, 10, 0)
        
        # Title with glow effect
        title_label = QtWidgets.QLabel("Memory Reflection")
        title_label.setStyleSheet("""
            color: #3498DB;
            font-size: 16px;
            font-weight: bold;
        """)
        
        # Add glow to title
        title_glow = QGraphicsDropShadowEffect()
        title_glow.setBlurRadius(12)
        title_glow.setColor(QColor(52, 152, 219, 150))
        title_glow.setOffset(0, 0)
        title_label.setGraphicsEffect(title_glow)
        
        # Control buttons
        self.introspect_button = QtWidgets.QPushButton("Introspect")
        self.introspect_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(41, 128, 185, 150);
                color: white;
                border-radius: 4px;
                border: 1px solid rgba(52, 152, 219, 100);
                padding: 5px 10px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: rgba(52, 152, 219, 180);
            }
        """)
        self.introspect_button.clicked.connect(self.toggle_introspection)
        
        self.recontextualize_button = QtWidgets.QPushButton("Recontextualize")
        self.recontextualize_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(142, 68, 173, 150);
                color: white;
                border-radius: 4px;
                border: 1px solid rgba(142, 68, 173, 100);
                padding: 5px 10px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: rgba(142, 68, 173, 180);
            }
            QPushButton:disabled {
                background-color: rgba(44, 62, 80, 150);
                color: rgba(189, 195, 199, 150);
            }
        """)
        self.recontextualize_button.clicked.connect(self.toggle_recontextualization)
        self.recontextualize_button.setEnabled(False)
        
        self.temporal_button = QtWidgets.QPushButton("Temporal View")
        self.temporal_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(46, 204, 113, 150);
                color: white;
                border-radius: 4px;
                border: 1px solid rgba(46, 204, 113, 100);
                padding: 5px 10px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: rgba(46, 204, 113, 180);
            }
        """)
        self.temporal_button.clicked.connect(self.toggle_temporal_view)
        
        # Add elements to control layout
        control_layout.addWidget(title_label)
        control_layout.addStretch()
        control_layout.addWidget(self.introspect_button)
        control_layout.addWidget(self.recontextualize_button)
        control_layout.addWidget(self.temporal_button)
        
        # Information display
        self.info_panel = QtWidgets.QFrame()
        self.info_panel.setStyleSheet("""
            background-color: rgba(26, 38, 52, 100);
            border-radius: 4px;
            border: 1px solid rgba(52, 152, 219, 60);
            padding: 5px;
        """)
        self.info_panel.setFixedHeight(100)
        self.info_panel.setVisible(False)
        
        info_layout = QtWidgets.QVBoxLayout(self.info_panel)
        
        self.memory_title = QtWidgets.QLabel("Memory Inspection")
        self.memory_title.setStyleSheet("""
            color: #3498DB;
            font-size: 14px;
            font-weight: bold;
        """)
        
        self.memory_text = QtWidgets.QLabel("Select a memory node to view details")
        self.memory_text.setWordWrap(True)
        self.memory_text.setStyleSheet("""
            color: #ECF0F1;
            font-size: 12px;
        """)
        
        self.memory_meta = QtWidgets.QLabel("Timestamp: N/A | Importance: N/A")
        self.memory_meta.setStyleSheet("""
            color: rgba(189, 195, 199, 200);
            font-size: 11px;
        """)
        
        info_layout.addWidget(self.memory_title)
        info_layout.addWidget(self.memory_text)
        info_layout.addWidget(self.memory_meta)
        
        # Status bar
        self.status_label = QtWidgets.QLabel("Ready | 8 memory patterns loaded")
        self.status_label.setStyleSheet("""
            color: rgba(189, 195, 199, 200);
            font-size: 12px;
        """)
        
        # Add elements to main layout
        layout.addWidget(control_panel)
        layout.addWidget(self.info_panel)
        layout.addStretch()
        layout.addWidget(self.status_label)
    
    def update_visualization(self):
        """Update the visualization for animation"""
        # Update memory nodes
        for node in self.memory_nodes:
            node.update()
        
        # Update background effect
        self.background_shift += 0.01 * self.shift_direction
        if self.background_shift > 1.0:
            self.background_shift = 1.0
            self.shift_direction = -1
        elif self.background_shift < 0.0:
            self.background_shift = 0.0
            self.shift_direction = 1
        
        # Trigger repaint
        self.update()
    
    def paintEvent(self, event):
        """Custom paint event for holographic visualization"""
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw the background effect
        self._draw_background(painter)
        
        # Draw connections between memory nodes
        self._draw_connections(painter)
        
        # Draw memory nodes
        self._draw_memory_nodes(painter)
    
    def _draw_background(self, painter):
        """Draw the background effect"""
        width, height = self.width(), self.height()
        
        # Draw grid lines
        painter.setPen(QPen(QColor(52, 152, 219, 20), 1))
        
        # Horizontal lines
        spacing = 30
        offset = int(self.background_shift * spacing)
        for y in range(-offset, height, spacing):
            painter.drawLine(0, y, width, y)
        
        # Vertical lines
        for x in range(-offset, width, spacing):
            painter.drawLine(x, 0, x, height)
        
        # Draw temporal axis if in temporal view
        if self.temporal_view:
            # Draw time axis line
            painter.setPen(QPen(QColor(46, 204, 113, 100), 2))
            axis_y = height * 0.7
            painter.drawLine(50, axis_y, width - 50, axis_y)
            
            # Draw temporal markers
            now = datetime.now()
            earliest = now - timedelta(hours=24)
            time_range = (now - earliest).total_seconds()
            
            # Convert timestamp to x position
            def time_to_x(timestamp):
                seconds_ago = (now - timestamp).total_seconds()
                x_pos = width - 50 - (seconds_ago / time_range) * (width - 100)
                return max(50, min(width - 50, x_pos))
            
            # Draw time markers
            painter.setPen(QPen(QColor(46, 204, 113, 80), 1))
            for i in range(13):
                marker_time = earliest + timedelta(hours=i * 2)
                x_pos = time_to_x(marker_time)
                
                # Draw marker line
                painter.drawLine(x_pos, axis_y - 5, x_pos, axis_y + 5)
                
                # Draw time label
                time_text = marker_time.strftime("%H:%M")
                painter.setPen(QColor(46, 204, 113, 150))
                painter.drawText(x_pos - 20, axis_y + 20, 40, 20, Qt.AlignCenter, time_text)
    
    def _draw_connections(self, painter):
        """Draw the connections between memory nodes"""
        for node in self.memory_nodes:
            for connected in node.connections:
                # Calculate connection opacity based on recontextualization
                base_opacity = 120
                if self.recontextualization_active and self.selected_node:
                    if node == self.selected_node or connected == self.selected_node:
                        # Strengthen connections to the selected node
                        opacity = 180
                    else:
                        # Fade other connections
                        opacity = 50
                else:
                    opacity = base_opacity
                
                # Draw the connection line with gradient
                gradient = QLinearGradient(node.x, node.y, connected.x, connected.y)
                gradient.setColorAt(0, QColor(52, 152, 219, opacity))
                gradient.setColorAt(1, QColor(52, 152, 219, opacity))
                
                painter.setPen(QPen(QBrush(gradient), 1.5))
                painter.drawLine(node.x, node.y, connected.x, connected.y)
                
                # Draw direction indicator if recontextualizing
                if self.recontextualization_active and self.selected_node:
                    if node == self.selected_node or connected == self.selected_node:
                        # Calculate midpoint
                        mid_x = (node.x + connected.x) / 2
                        mid_y = (node.y + connected.y) / 2
                        
                        # Draw pulsing dot at midpoint
                        pulse = 0.5 + 0.5 * math.sin(self.background_shift * 6)
                        dot_size = 3 + pulse * 3
                        
                        painter.setBrush(QBrush(QColor(155, 89, 182, 200)))
                        painter.setPen(Qt.NoPen)
                        painter.drawEllipse(mid_x - dot_size, mid_y - dot_size, 
                                           dot_size * 2, dot_size * 2)
    
    def _draw_memory_nodes(self, painter):
        """Draw the memory nodes"""
        for node in self.memory_nodes:
            # Position adjustment if in temporal view
            if self.temporal_view:
                # Position vertically in the middle, horizontally by time
                now = datetime.now()
                earliest = now - timedelta(hours=24)
                time_range = (now - earliest).total_seconds()
                
                seconds_ago = (now - node.timestamp).total_seconds()
                x_pos = self.width() - 50 - (seconds_ago / time_range) * (self.width() - 100)
                node.target_x = max(50, min(self.width() - 50, x_pos))
                node.target_y = self.height() * 0.7 - 20 - node.importance * 40
            
            # Calculate visual properties
            radius = node.radius * (0.8 + 0.2 * node.glow)
            importance_color = self._importance_to_color(node.importance)
            
            # Add recontextualization effect
            if self.recontextualization_active and self.selected_node:
                if node == self.selected_node:
                    # Make the selected node pulse more dramatically
                    radius *= 1.1 + 0.1 * math.sin(self.background_shift * 10)
                    glow_intensity = 180 + 75 * node.glow
                elif node in self.selected_node.connections:
                    # Connected nodes show stronger relationship
                    radius *= 1.05
                    glow_intensity = 150 + 50 * node.glow
                else:
                    # Other nodes fade slightly
                    radius *= 0.9
                    glow_intensity = 100 + 30 * node.glow
            else:
                glow_intensity = 120 + 50 * node.glow
            
            # Draw outer glow
            glow = QRadialGradient(node.x, node.y, radius * 1.5)
            glow_color = QColor(importance_color.red(), 
                               importance_color.green(), 
                               importance_color.blue(), 
                               glow_intensity)
            glow.setColorAt(0, glow_color)
            glow.setColorAt(1, QColor(importance_color.red(), 
                                     importance_color.green(), 
                                     importance_color.blue(), 0))
            
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(node.x - radius * 1.5, node.y - radius * 1.5, 
                               radius * 3, radius * 3)
            
            # Draw inner circle
            painter.setBrush(QBrush(importance_color))
            painter.setPen(QPen(QColor(255, 255, 255, 50), 1))
            painter.drawEllipse(node.x - radius / 2, node.y - radius / 2, radius, radius)
            
            # Draw selection indicator if this is the selected node
            if node == self.selected_node:
                painter.setPen(QPen(QColor(255, 255, 255, 150 + int(100 * node.glow)), 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(node.x - radius * 0.8, node.y - radius * 0.8, 
                                   radius * 1.6, radius * 1.6)
    
    def _importance_to_color(self, importance):
        """Convert importance value to a color"""
        if importance < 0.3:
            # Low importance - blue
            return QColor(52, 152, 219, 180)
        elif importance < 0.7:
            # Medium importance - purple
            return QColor(155, 89, 182, 180)
        else:
            # High importance - golden
            return QColor(241, 196, 15, 180)
    
    def mousePressEvent(self, event):
        """Handle mouse click events"""
        if event.button() == Qt.LeftButton:
            # Find if a node was clicked
            clicked_node = self._find_node_at_position(event.pos().x(), event.pos().y())
            if clicked_node:
                self._select_node(clicked_node)
            else:
                self._deselect_node()
    
    def _find_node_at_position(self, x, y):
        """Find a node at the given position"""
        for node in self.memory_nodes:
            dx = node.x - x
            dy = node.y - y
            distance = math.sqrt(dx * dx + dy * dy)
            if distance <= node.radius:
                return node
        return None
    
    def _select_node(self, node):
        """Select a node and update UI"""
        self.selected_node = node
        self.info_panel.setVisible(True)
        
        # Update info panel
        self.memory_text.setText(node.text)
        self.memory_meta.setText(f"Timestamp: {node.timestamp.strftime('%Y-%m-%d %H:%M')} | Importance: {node.importance:.2f}")
        
        # Enable recontextualize button
        self.recontextualize_button.setEnabled(True)
        
        # Update status
        self.status_label.setText(f"Memory selected: {len(node.connections)} connections")
        
        # Emit signal
        self.memory_selected.emit(node.text, node.timestamp)
        
        self.update()
    
    def _deselect_node(self):
        """Deselect the current node"""
        self.selected_node = None
        self.info_panel.setVisible(False)
        self.recontextualize_button.setEnabled(False)
        
        # If recontextualization is active, turn it off
        if self.recontextualization_active:
            self.toggle_recontextualization()
        
        # Update status
        self.status_label.setText(f"Ready | {len(self.memory_nodes)} memory patterns loaded")
        
        self.update()
    
    def toggle_introspection(self):
        """Toggle memory introspection mode"""
        # In a real implementation, this would connect to backend systems
        if self.introspect_button.text() == "Introspect":
            self.introspect_button.setText("Normal View")
            self.status_label.setText("Memory introspection active - analyzing patterns")
            
            # Rearrange nodes in a more organized pattern
            center_x, center_y = self.width() / 2, self.height() / 2
            radius = min(self.width(), self.height()) * 0.4
            
            for i, node in enumerate(self.memory_nodes):
                angle = i * (2 * math.pi / len(self.memory_nodes))
                node.target_x = center_x + radius * math.cos(angle)
                node.target_y = center_y + radius * math.sin(angle)
        else:
            self.introspect_button.setText("Introspect")
            self.status_label.setText(f"Ready | {len(self.memory_nodes)} memory patterns loaded")
            
            # Randomize positions slightly
            for node in self.memory_nodes:
                node.target_x += random.randint(-30, 30)
                node.target_y += random.randint(-30, 30)
    
    def toggle_recontextualization(self):
        """Toggle memory recontextualization mode"""
        if not self.selected_node:
            return
            
        self.recontextualization_active = not self.recontextualization_active
        
        if self.recontextualization_active:
            self.recontextualize_button.setText("Stop Recontextualize")
            self.status_label.setText(f"Recontextualizing memory: {self.selected_node.text[:20]}...")
        else:
            self.recontextualize_button.setText("Recontextualize")
            self.status_label.setText(f"Memory selected: {len(self.selected_node.connections)} connections")
    
    def toggle_temporal_view(self):
        """Toggle between temporal and spatial views"""
        self.temporal_view = not self.temporal_view
        
        if self.temporal_view:
            self.temporal_button.setText("Spatial View")
            self.status_label.setText("Temporal integration view active - organizing by timestamp")
        else:
            self.temporal_button.setText("Temporal View")
            self.status_label.setText(f"Ready | {len(self.memory_nodes)} memory patterns loaded")
            
            # Reset to spatial positions when exiting temporal view
            center_x, center_y = self.width() / 2, self.height() / 2
            
            for i, node in enumerate(self.memory_nodes):
                angle = i * 0.8
                r = 100 + i * 20
                node.target_x = center_x + r * math.cos(angle) + random.randint(-20, 20)
                node.target_y = center_y + r * math.sin(angle) + random.randint(-20, 20)
    
    def cleanup(self):
        """Clean up resources before destruction"""
        if hasattr(self, 'animation_timer') and self.animation_timer:
            self.animation_timer.stop() 