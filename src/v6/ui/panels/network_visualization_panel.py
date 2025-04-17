"""
Neural Network Visualization Panel

Provides a real-time interactive visualization of the neural network with 
node selection, connection strength visualization, and pattern detection.
"""

import math
import random
import logging
from pathlib import Path

# Add project root to path if needed
import sys
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

# Import the panel base
from ..panel_base import V6PanelBase

# Set up logging
logger = logging.getLogger(__name__)

class NetworkVisualizationPanel(V6PanelBase):
    """Interactive neural network visualization panel with real-time updates"""
    
    # Signal emitted when a node is selected
    nodeSelected = Signal(int)
    
    def __init__(self, socket_manager=None, parent=None):
        super().__init__(parent)
        self.socket_manager = socket_manager
        
        # Network data
        self.nodes = []
        self.connections = []
        self.selected_node = None
        self.hovered_node = None
        self.connection_strength = 70  # Default connection strength (0-100)
        self.display_mode = "Default"  # Default, Activity, Relevance
        
        # Initialize UI
        self.init_ui()
        
        # Set up socket manager event handling if available
        if self.socket_manager:
            self.setup_socket_events()
        else:
            # Create mock data for testing
            self.generate_mock_data()
        
        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)
        
        # Start animation timer
        self.animation_timer = QtCore.QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(50)  # 50ms update = ~20fps
    
    def init_ui(self):
        """Initialize the user interface components"""
        # Create main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Create controls panel at top
        controls_layout = QtWidgets.QHBoxLayout()
        
        # Display mode selector
        self.mode_label = QtWidgets.QLabel("Display Mode:")
        self.mode_label.setStyleSheet("color: #ECF0F1; font-size: 13px;")
        
        self.mode_selector = QtWidgets.QComboBox()
        self.mode_selector.addItems(["Default", "Activity", "Relevance"])
        self.mode_selector.setStyleSheet("""
            background-color: rgba(52, 73, 94, 180);
            color: white;
            padding: 4px;
            border: 1px solid rgba(52, 152, 219, 120);
            border-radius: 4px;
            min-width: 120px;
        """)
        self.mode_selector.currentTextChanged.connect(self.change_display_mode)
        
        # Connection strength slider
        self.strength_label = QtWidgets.QLabel("Connection Strength:")
        self.strength_label.setStyleSheet("color: #ECF0F1; font-size: 13px;")
        
        self.strength_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(70)
        self.strength_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: rgba(52, 73, 94, 150);
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: rgba(52, 152, 219, 200);
                border: 1px solid rgba(41, 128, 185, 255);
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
        """)
        self.strength_slider.valueChanged.connect(self.change_connection_strength)
        
        # Reset view button
        self.reset_button = QtWidgets.QPushButton("Reset View")
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(41, 128, 185, 180);
                color: white;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 13px;
                border: 1px solid rgba(52, 152, 219, 120);
            }
            QPushButton:hover {
                background-color: rgba(52, 152, 219, 200);
                border: 1px solid rgba(52, 152, 219, 180);
            }
        """)
        self.reset_button.clicked.connect(self.reset_view)
        
        # Add widgets to controls layout
        controls_layout.addWidget(self.mode_label)
        controls_layout.addWidget(self.mode_selector)
        controls_layout.addSpacing(15)
        controls_layout.addWidget(self.strength_label)
        controls_layout.addWidget(self.strength_slider, 1)  # Stretch factor 1
        controls_layout.addSpacing(15)
        controls_layout.addWidget(self.reset_button)
        
        # Add controls to main layout
        layout.addLayout(controls_layout)
        
        # Rest of the layout is for the visualization canvas
        layout.addStretch(1)
    
    def setup_socket_events(self):
        """Set up event handlers for socket manager events"""
        if not self.socket_manager:
            return
            
        # Register for network update events
        self.socket_manager.register_handler("network_update", self.handle_network_update)
        self.socket_manager.register_handler("node_activation", self.handle_node_activation)
        
        # Request initial network data
        self.socket_manager.emit("request_network_data", {})
    
    def handle_network_update(self, data):
        """Handle network update events from socket manager"""
        if "nodes" in data:
            self.nodes = data["nodes"]
        if "connections" in data:
            self.connections = data["connections"]
        self.update()
    
    def handle_node_activation(self, data):
        """Handle node activation events from socket manager"""
        if "node_id" in data:
            node_id = data["node_id"]
            for node in self.nodes:
                if node["id"] == node_id:
                    node["activation"] = data.get("activation", 1.0)
                    node["pulse"] = 1.0  # Start a pulse animation
                    break
        self.update()
    
    def generate_mock_data(self):
        """Generate mock data for testing without a socket manager"""
        # Create mock nodes
        self.nodes = []
        node_count = 20
        
        # Create nodes in a circle
        center_x = self.width() / 2 if self.width() > 0 else 300
        center_y = self.height() / 2 if self.height() > 0 else 200
        radius = min(center_x, center_y) * 0.7
        
        for i in range(node_count):
            angle = i * (2 * math.pi / node_count)
            node = {
                "id": i,
                "x": center_x + radius * math.cos(angle),
                "y": center_y + radius * math.sin(angle),
                "size": 12 + random.randint(0, 8),
                "activation": random.random() * 0.5,
                "relevance": random.random(),
                "label": f"Node {i}",
                "color": QtGui.QColor(
                    100 + random.randint(0, 100),
                    150 + random.randint(0, 100),
                    200 + random.randint(0, 55)
                ),
                "pulse": 0.0,
                "pulse_dir": 1
            }
            self.nodes.append(node)
        
        # Create connections between nodes (not all nodes are connected)
        self.connections = []
        for i in range(node_count):
            # Each node connects to 2-5 others
            connections_count = random.randint(2, 5)
            for _ in range(connections_count):
                target = random.randint(0, node_count - 1)
                if target != i:  # No self-connections
                    # Check if connection already exists
                    exists = False
                    for conn in self.connections:
                        if (conn["source"] == i and conn["target"] == target) or \
                           (conn["source"] == target and conn["target"] == i):
                            exists = True
                            break
                    
                    if not exists:
                        connection = {
                            "source": i,
                            "target": target,
                            "weight": random.random() * 0.8 + 0.2,  # 0.2 to 1.0
                            "active": True
                        }
                        self.connections.append(connection)
    
    def update_animation(self):
        """Update animation state for nodes and connections"""
        update_needed = False
        
        # Update node pulses
        for node in self.nodes:
            if node.get("pulse", 0) > 0:
                node["pulse"] = max(0, node["pulse"] - 0.05)
                update_needed = True
                
            # Occasionally activate a random node
            if random.random() < 0.01:  # 1% chance per frame
                node["activation"] = min(1.0, node["activation"] + random.random() * 0.3)
                node["pulse"] = 1.0
                update_needed = True
            
            # Decay activation gradually
            if node["activation"] > 0:
                node["activation"] = max(0, node["activation"] - 0.005)
                update_needed = True
        
        # Update connection states
        for conn in self.connections:
            # Occasionally toggle active state based on connection strength
            if random.random() < 0.005:  # 0.5% chance per frame
                source_node = self.nodes[conn["source"]]
                target_node = self.nodes[conn["target"]]
                
                # More likely to activate if both nodes have higher activation
                activation_factor = (source_node["activation"] + target_node["activation"]) / 2
                
                # Connection strength influences activation threshold
                threshold = 1.0 - (self.connection_strength / 100.0)
                
                if activation_factor > threshold:
                    conn["active"] = True
                    update_needed = True
        
        if update_needed:
            self.update()
    
    def change_display_mode(self, mode):
        """Change the display mode for the network visualization"""
        self.display_mode = mode
        logger.info(f"Changed display mode to: {mode}")
        self.update()
    
    def change_connection_strength(self, value):
        """Change the connection strength threshold"""
        self.connection_strength = value
        logger.info(f"Changed connection strength to: {value}")
        # Update connections visibility based on new threshold
        self.update()
    
    def reset_view(self):
        """Reset the view to default state"""
        logger.info("Resetting network view")
        
        # Regenerate mock data if in test mode
        if not self.socket_manager:
            self.generate_mock_data()
        else:
            # Request fresh data from socket manager
            self.socket_manager.emit("request_network_data", {
                "reset": True
            })
        
        # Reset view parameters
        self.selected_node = None
        self.hovered_node = None
        self.display_mode = "Default"
        self.mode_selector.setCurrentText("Default")
        self.connection_strength = 70
        self.strength_slider.setValue(70)
        
        self.update()
    
    def resizeEvent(self, event):
        """Handle resize events to reposition nodes"""
        super().resizeEvent(event)
        
        if not self.nodes:
            return
            
        # Reposition nodes based on new size while maintaining relative positions
        old_width = event.oldSize().width() or 100
        old_height = event.oldSize().height() or 100
        width_ratio = self.width() / old_width
        height_ratio = self.height() / old_height
        
        center_x = self.width() / 2
        center_y = self.height() / 2
        
        for node in self.nodes:
            # Calculate position relative to center
            rel_x = node["x"] - (old_width / 2)
            rel_y = node["y"] - (old_height / 2)
            
            # Apply scaling while preserving relative positions
            node["x"] = center_x + rel_x * width_ratio
            node["y"] = center_y + rel_y * height_ratio
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for node hovering"""
        old_hover = self.hovered_node
        self.hovered_node = None
        
        # Check if mouse is over any node
        for i, node in enumerate(self.nodes):
            distance = math.sqrt((event.x() - node["x"])**2 + (event.y() - node["y"])**2)
            if distance < node["size"] + 5:
                self.hovered_node = i
                self.setCursor(Qt.PointingHandCursor)
                break
        
        if old_hover != self.hovered_node:
            if self.hovered_node is None:
                self.setCursor(Qt.ArrowCursor)
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events for node selection"""
        if self.hovered_node is not None:
            # Toggle selection
            if self.selected_node == self.hovered_node:
                self.selected_node = None
            else:
                self.selected_node = self.hovered_node
                # Emit signal with selected node ID
                self.nodeSelected.emit(self.nodes[self.selected_node]["id"])
            
            # For mock data testing, increase activation of selected node
            if self.selected_node is not None:
                self.nodes[self.selected_node]["activation"] = 1.0
                self.nodes[self.selected_node]["pulse"] = 1.0
            
            self.update()
    
    def paintEvent(self, event):
        """Custom paint event to render the network visualization"""
        super().paintEvent(event)
        
        if not self.nodes:
            return
            
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Draw connections first (behind nodes)
        self.draw_connections(painter)
        
        # Then draw nodes
        self.draw_nodes(painter)
        
        # Draw status info
        self.draw_status_info(painter)
    
    def draw_connections(self, painter):
        """Draw connections between nodes"""
        for conn in self.connections:
            source = self.nodes[conn["source"]]
            target = self.nodes[conn["target"]]
            
            # Get connection strength based on mode
            strength = conn["weight"]
            
            # Check if connection meets threshold
            if strength * 100 < self.connection_strength:
                continue
            
            # Determine color and opacity based on connection state
            if conn["active"]:
                # Active connection
                if self.display_mode == "Activity":
                    # Color based on average activation of connected nodes
                    activity = (source["activation"] + target["activation"]) / 2
                    color = QtGui.QColor(
                        int(52 + 203 * activity),  # More red for higher activity
                        int(152 + 103 * (1 - activity)),  # Less green for higher activity
                        219,
                        int(100 + 155 * strength)  # Opacity based on weight
                    )
                elif self.display_mode == "Relevance":
                    # Color based on relevance
                    relevance = (source["relevance"] + target["relevance"]) / 2
                    color = QtGui.QColor(
                        52,
                        int(152 + 103 * relevance),  # More green for higher relevance
                        int(219 - 150 * relevance),  # Less blue for higher relevance
                        int(100 + 155 * strength)  # Opacity based on weight
                    )
                else:
                    # Default color
                    color = QtGui.QColor(52, 152, 219, int(100 + 155 * strength))
                
                width = 1.5 + 2.5 * strength
                style = Qt.SolidLine
            else:
                # Inactive connection
                color = QtGui.QColor(52, 73, 94, int(60 + 60 * strength))
                width = 1
                style = Qt.DotLine
            
            # Highlight connections to selected node
            if self.selected_node is not None:
                if conn["source"] == self.selected_node or conn["target"] == self.selected_node:
                    color = QtGui.QColor(231, 76, 60, 200)  # Highlight in red
                    width += 1
            
            # Set pen for drawing
            pen = QtGui.QPen(color, width)
            pen.setStyle(style)
            painter.setPen(pen)
            
            # Draw the connection line
            painter.drawLine(
                int(source["x"]), int(source["y"]),
                int(target["x"]), int(target["y"])
            )
    
    def draw_nodes(self, painter):
        """Draw network nodes"""
        for i, node in enumerate(self.nodes):
            # Calculate node appearance based on mode and state
            if self.display_mode == "Activity":
                # Color based on activation level
                color = QtGui.QColor(
                    int(41 + 190 * node["activation"]),  # More red for higher activation
                    int(128 + 127 * (1 - node["activation"])),  # Less green for higher activation
                    185,
                    180
                )
            elif self.display_mode == "Relevance":
                # Color based on relevance
                color = QtGui.QColor(
                    41,
                    int(128 + 127 * node["relevance"]),  # More green for higher relevance
                    int(185 + 70 * (1 - node["relevance"])),  # Less blue for higher relevance
                    180
                )
            else:
                # Default color (use node's base color)
                color = node["color"]
            
            # Pulse effect size modification
            pulse = node.get("pulse", 0)
            base_size = node["size"]
            size = base_size * (1.0 + 0.5 * pulse)
            
            # Node glow for active nodes
            if node["activation"] > 0.1 or pulse > 0:
                glow_radius = size + 10
                glow = QtGui.QRadialGradient(node["x"], node["y"], glow_radius)
                glow_color = QtGui.QColor(color)
                glow_color.setAlpha(int(100 * max(node["activation"], pulse * 0.5)))
                glow.setColorAt(0, glow_color)
                glow.setColorAt(1, QtGui.QColor(color.red(), color.green(), color.blue(), 0))
                painter.setBrush(QtGui.QBrush(glow))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(
                    node["x"] - glow_radius, 
                    node["y"] - glow_radius,
                    glow_radius * 2, 
                    glow_radius * 2
                )
            
            # Handle selection and hover states
            if i == self.selected_node:
                # Selected node
                ring_color = QtGui.QColor(231, 76, 60)  # Red highlight
                border_width = 3
            elif i == self.hovered_node:
                # Hovered node
                ring_color = QtGui.QColor(241, 196, 15)  # Yellow highlight
                border_width = 2
            else:
                # Normal node
                ring_color = QtGui.QColor(255, 255, 255, 100)
                border_width = 1
            
            # Draw node circle
            painter.setBrush(QtGui.QBrush(color))
            painter.setPen(QtGui.QPen(ring_color, border_width))
            painter.drawEllipse(
                node["x"] - size/2, 
                node["y"] - size/2,
                size, 
                size
            )
            
            # Draw node ID in the center for larger nodes
            if base_size >= 15 or i == self.selected_node or i == self.hovered_node:
                font = QtGui.QFont("Segoe UI", 8)
                painter.setFont(font)
                painter.setPen(QtGui.QColor(255, 255, 255, 220))
                
                # Only show the number part
                id_text = str(node["id"])
                
                # Center the text
                metrics = QtGui.QFontMetrics(font)
                text_width = metrics.horizontalAdvance(id_text)
                text_height = metrics.height()
                
                painter.drawText(
                    int(node["x"] - text_width/2),
                    int(node["y"] + text_height/4),
                    id_text
                )
            
            # Show node info for selected or hovered node
            if i == self.selected_node or i == self.hovered_node:
                # Draw label above the node
                font = QtGui.QFont("Segoe UI", 9)
                painter.setFont(font)
                painter.setPen(QtGui.QColor(255, 255, 255, 220))
                
                # Prepare label text
                if self.display_mode == "Activity":
                    label_text = f"{node['label']}: {int(node['activation'] * 100)}%"
                elif self.display_mode == "Relevance":
                    label_text = f"{node['label']}: {int(node['relevance'] * 100)}%"
                else:
                    label_text = node["label"]
                
                # Measure text size
                metrics = QtGui.QFontMetrics(font)
                text_width = metrics.horizontalAdvance(label_text)
                text_height = metrics.height()
                
                # Create background rect
                bg_rect = QtCore.QRect(
                    int(node["x"] - text_width/2 - 5),
                    int(node["y"] - size/2 - text_height - 8),
                    text_width + 10,
                    text_height + 6
                )
                
                # Draw background
                painter.fillRect(
                    bg_rect,
                    QtGui.QColor(44, 62, 80, 200)
                )
                
                # Draw text
                painter.drawText(
                    int(node["x"] - text_width/2),
                    int(node["y"] - size/2 - 8),
                    label_text
                )
    
    def draw_status_info(self, painter):
        """Draw status information overlay"""
        # Draw pattern highlight message if applicable
        if self.selected_node is not None:
            # Get selected node info
            node = self.nodes[self.selected_node]
            
            # Create info string
            if self.display_mode == "Activity":
                info = f"Highlighting node: {node['label']} - Activity: {int(node['activation'] * 100)}%"
            elif self.display_mode == "Relevance":
                info = f"Highlighting node: {node['label']} - Relevance: {int(node['relevance'] * 100)}%"
            else:
                info = f"Highlighting node: {node['label']}"
            
            # Draw text at bottom of panel
            font = QtGui.QFont("Segoe UI", 10)
            painter.setFont(font)
            painter.setPen(QtGui.QColor(236, 240, 241))
            
            painter.drawText(15, self.height() - 15, info) 