"""
Node Consciousness Visualization Panel

Provides an interactive node consciousness visualization with metrics display
based on the V5 Fractal Echo Visualization.
"""

import math
import random
import logging
from pathlib import Path
import colorsys

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

class NodeConsciousnessPanel(V6PanelBase):
    """Interactive node consciousness visualization panel with real-time updates"""
    
    # Signal emitted when a node is selected
    nodeSelected = Signal(int)
    
    def __init__(self, socket_manager=None, parent=None):
        super().__init__(parent)
        self.socket_manager = socket_manager
        
        # Visualization state
        self.nodes = []
        self.connections = []
        self.selected_node = None
        self.hovered_node = None
        self.current_tab = "Global"  # Global, Node Details, Active Processing
        
        # Node metrics
        self.awareness_level = 87
        self.integration_index = 0.77
        self.neural_coherence = "High"
        self.responsiveness = 94
        
        # Animation
        self.animation_timer = None
        self.animation_time = 0
        
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
        
        # Start animation
        self.start_animation()
    
    def init_ui(self):
        """Initialize the user interface components"""
        # Create main layout with tabs and content area
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create tab bar
        tab_bar = QtWidgets.QWidget()
        tab_bar.setFixedHeight(40)
        tab_bar.setStyleSheet("""
            background-color: rgba(26, 38, 52, 220);
            border-bottom: 1px solid rgba(52, 73, 94, 150);
        """)
        
        tab_layout = QtWidgets.QHBoxLayout(tab_bar)
        tab_layout.setContentsMargins(10, 0, 10, 0)
        tab_layout.setSpacing(5)
        
        # Tab buttons
        self.global_btn = self.create_tab_button("Global Metrics")
        self.node_btn = self.create_tab_button("Node Details")
        self.process_btn = self.create_tab_button("Active Processing")
        
        # Connect tab buttons
        self.global_btn.clicked.connect(lambda: self.change_tab("Global"))
        self.node_btn.clicked.connect(lambda: self.change_tab("Node"))
        self.process_btn.clicked.connect(lambda: self.change_tab("Process"))
        
        # Set initial state
        self.global_btn.setProperty("active", True)
        self.global_btn.setStyleSheet(self.global_btn.styleSheet())
        
        # Add buttons to tab layout
        tab_layout.addWidget(self.global_btn)
        tab_layout.addWidget(self.node_btn)
        tab_layout.addWidget(self.process_btn)
        tab_layout.addStretch(1)
        
        # Add tab bar to main layout
        layout.addWidget(tab_bar)
        
        # Create content stacked widget
        self.content_stack = QtWidgets.QStackedWidget()
        self.content_stack.setStyleSheet("""
            background-color: transparent;
        """)
        
        # Create pages for each tab
        self.global_page = QtWidgets.QWidget()
        self.node_page = QtWidgets.QWidget()
        self.process_page = QtWidgets.QWidget()
        
        # Set up global metrics page
        global_layout = QtWidgets.QVBoxLayout(self.global_page)
        global_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add visualization area to global page
        self.viz_area = QtWidgets.QWidget()
        self.viz_area.setStyleSheet("""
            background-color: rgba(16, 26, 40, 180);
        """)
        self.viz_area.setMouseTracking(True)
        global_layout.addWidget(self.viz_area)
        
        # Set up node details page
        node_layout = QtWidgets.QVBoxLayout(self.node_page)
        node_layout.setContentsMargins(15, 15, 15, 15)
        
        # Node metrics
        metrics_group = QtWidgets.QGroupBox("Node Metrics")
        metrics_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid rgba(52, 152, 219, 120);
                border-radius: 4px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                color: #3498DB;
                font-weight: bold;
            }
        """)
        
        metrics_layout = QtWidgets.QFormLayout(metrics_group)
        
        # Awareness Level
        self.awareness_label = QtWidgets.QLabel()
        self.awareness_label.setStyleSheet("color: #ECF0F1; font-size: 24px; font-weight: bold;")
        metrics_layout.addRow("<b>Awareness Level:</b>", self.awareness_label)
        
        # Integration Index
        self.integration_label = QtWidgets.QLabel()
        self.integration_label.setStyleSheet("color: #ECF0F1; font-size: 24px; font-weight: bold;")
        metrics_layout.addRow("<b>Integration Index:</b>", self.integration_label)
        
        # Neural Coherence
        self.coherence_label = QtWidgets.QLabel()
        self.coherence_label.setStyleSheet("color: #ECF0F1; font-size: 24px; font-weight: bold;")
        metrics_layout.addRow("<b>Neural Coherence:</b>", self.coherence_label)
        
        # Responsiveness
        self.responsiveness_label = QtWidgets.QLabel()
        self.responsiveness_label.setStyleSheet("color: #ECF0F1; font-size: 24px; font-weight: bold;")
        metrics_layout.addRow("<b>Responsiveness:</b>", self.responsiveness_label)
        
        node_layout.addWidget(metrics_group)
        node_layout.addStretch(1)
        
        # Update node metrics display
        self.update_node_metrics()
        
        # Set up process page
        process_layout = QtWidgets.QVBoxLayout(self.process_page)
        process_layout.setContentsMargins(15, 15, 15, 15)
        
        process_title = QtWidgets.QLabel("Active Processing")
        process_title.setStyleSheet("color: #3498DB; font-size: 18px; font-weight: bold;")
        process_layout.addWidget(process_title)
        
        # Process visualization (placeholder)
        process_viz = QtWidgets.QLabel("Real-time neural processing visualization will appear here")
        process_viz.setAlignment(Qt.AlignCenter)
        process_viz.setStyleSheet("color: #ECF0F1; font-size: 14px;")
        process_layout.addWidget(process_viz, 1)
        
        # Add pages to stack
        self.content_stack.addWidget(self.global_page)
        self.content_stack.addWidget(self.node_page)
        self.content_stack.addWidget(self.process_page)
        
        # Add content stack to main layout
        layout.addWidget(self.content_stack, 1)  # 1 = stretch factor
        
        # Add status bar
        status_bar = QtWidgets.QWidget()
        status_bar.setFixedHeight(25)
        status_bar.setStyleSheet("""
            background-color: rgba(16, 26, 40, 220);
            border-top: 1px solid rgba(52, 73, 94, 150);
        """)
        
        status_layout = QtWidgets.QHBoxLayout(status_bar)
        status_layout.setContentsMargins(10, 0, 10, 0)
        
        self.status_label = QtWidgets.QLabel("Consciousness data updated")
        self.status_label.setStyleSheet("color: rgba(127, 140, 141, 220); font-size: 11px;")
        status_layout.addWidget(self.status_label)
        
        layout.addWidget(status_bar)
    
    def create_tab_button(self, text):
        """Create a styled tab button"""
        button = QtWidgets.QPushButton(text)
        button.setCheckable(True)
        button.setFlat(True)
        button.setFixedHeight(30)
        button.setMinimumWidth(100)
        button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #ECF0F1;
                border: none;
                border-bottom: 3px solid transparent;
                font-size: 13px;
                text-align: center;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: rgba(52, 73, 94, 120);
                border-bottom: 3px solid rgba(52, 152, 219, 120);
            }
            QPushButton[active="true"] {
                color: #3498DB;
                font-weight: bold;
                border-bottom: 3px solid #3498DB;
            }
        """)
        return button
    
    def change_tab(self, tab):
        """Change the active tab"""
        self.current_tab = tab
        logger.info(f"Changed to tab: {tab}")
        
        # Update button states
        self.global_btn.setProperty("active", tab == "Global")
        self.node_btn.setProperty("active", tab == "Node")
        self.process_btn.setProperty("active", tab == "Process")
        
        # Force style update
        self.global_btn.setStyleSheet(self.global_btn.styleSheet())
        self.node_btn.setStyleSheet(self.node_btn.styleSheet())
        self.process_btn.setStyleSheet(self.process_btn.styleSheet())
        
        # Switch content page
        if tab == "Global":
            self.content_stack.setCurrentWidget(self.global_page)
        elif tab == "Node":
            self.content_stack.setCurrentWidget(self.node_page)
        else:
            self.content_stack.setCurrentWidget(self.process_page)
    
    def update_node_metrics(self):
        """Update the node metrics display"""
        # Format awareness level as large blue number
        self.awareness_label.setText(f'<span style="color: #3498DB; font-size: 32px;">{self.awareness_level}</span>')
        
        # Format integration index as decimal number
        self.integration_label.setText(f'<span style="color: #3498DB; font-size: 32px;">{self.integration_index:.2f}</span>')
        
        # Format neural coherence as text
        self.coherence_label.setText(f'<span style="color: #3498DB; font-size: 32px;">{self.neural_coherence}</span>')
        
        # Format responsiveness as large blue number
        self.responsiveness_label.setText(f'<span style="color: #3498DB; font-size: 32px;">{self.responsiveness}</span>')
    
    def setup_socket_events(self):
        """Set up event handlers for socket manager events"""
        if not self.socket_manager:
            return
            
        # Register for node consciousness events
        self.socket_manager.register_handler("node_consciousness_update", self.handle_consciousness_update)
        
        # Request initial data
        self.socket_manager.emit("request_node_consciousness", {})
    
    def handle_consciousness_update(self, data):
        """Handle consciousness update events from socket manager"""
        if "nodes" in data:
            self.nodes = data["nodes"]
        if "connections" in data:
            self.connections = data["connections"]
        if "metrics" in data:
            metrics = data["metrics"]
            self.awareness_level = metrics.get("awareness", self.awareness_level)
            self.integration_index = metrics.get("integration", self.integration_index)
            self.neural_coherence = metrics.get("coherence", self.neural_coherence)
            self.responsiveness = metrics.get("responsiveness", self.responsiveness)
            self.update_node_metrics()
        
        self.status_label.setText("Consciousness data updated at " + QtCore.QDateTime.currentDateTime().toString("hh:mm:ss"))
        self.update()
    
    def generate_mock_data(self):
        """Generate mock data for testing without a socket manager"""
        # Create nodes arranged in a circle
        self.nodes = []
        node_count = 24
        
        # Circle parameters
        center_x = 0
        center_y = 0
        radius = 200
        
        # Create nodes on a circle
        for i in range(node_count):
            angle = 2 * math.pi * i / node_count
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # Group nodes by color (emotional centers)
            hue = (i % 6) / 6.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            
            node = {
                "id": i,
                "x": x,
                "y": y,
                "radius": 10 + random.randint(0, 10),  # Random size
                "color": QtGui.QColor(int(r * 255), int(g * 255), int(b * 255), 220),
                "value": random.random(),
                "activity": random.random() * 0.5,
                "label": f"Node {i}",
                "group": i % 6  # 6 emotional groups
            }
            self.nodes.append(node)
        
        # Create connections between nodes
        self.connections = []
        
        # Create connections within same group (stronger)
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                if self.nodes[i]["group"] == self.nodes[j]["group"]:
                    # Same group - always connect with higher weight
                    weight = 0.7 + random.random() * 0.3
                    self.connections.append({
                        "source": i,
                        "target": j,
                        "weight": weight,
                        "active": True
                    })
                elif random.random() < 0.15:  # 15% chance for cross-group connections
                    # Different group - occasional connections with lower weight
                    weight = 0.3 + random.random() * 0.4
                    self.connections.append({
                        "source": i,
                        "target": j,
                        "weight": weight,
                        "active": random.random() < 0.7  # 70% chance of being active
                    })
    
    def start_animation(self):
        """Start the consciousness animation"""
        if not self.animation_timer:
            self.animation_timer = QtCore.QTimer(self)
            self.animation_timer.timeout.connect(self.animate_frame)
            self.animation_timer.start(50)  # 50ms = 20fps
    
    def stop_animation(self):
        """Stop the consciousness animation"""
        if self.animation_timer and self.animation_timer.isActive():
            self.animation_timer.stop()
    
    def animate_frame(self):
        """Update animation for one frame"""
        # Update animation time
        self.animation_time += 0.05
        
        # No nodes to animate
        if not self.nodes:
            return
            
        update_needed = False
        
        # Pulsate nodes
        for node in self.nodes:
            # Pulsating effect based on activity
            activity_factor = node.get("activity", 0) * (0.5 + 0.5 * math.sin(self.animation_time + node["id"] * 0.5))
            node["current_radius"] = node["radius"] * (1.0 + 0.3 * activity_factor)
            
            # Occasionally activate random nodes
            if random.random() < 0.02:  # 2% chance per frame
                node["activity"] = min(1.0, node["activity"] + random.random() * 0.5)
                update_needed = True
            elif node["activity"] > 0:
                # Decay activity over time
                node["activity"] = max(0, node["activity"] - 0.01)
                update_needed = True
        
        # Update connections
        for conn in self.connections:
            # Random activity changes
            if random.random() < 0.01:  # 1% chance per frame
                conn["active"] = not conn["active"]
                update_needed = True
            
            # Connection strength based on node activity
            source_node = self.nodes[conn["source"]]
            target_node = self.nodes[conn["target"]]
            avg_activity = (source_node.get("activity", 0) + target_node.get("activity", 0)) / 2
            
            # More active nodes have stronger connections
            if avg_activity > 0.4 and not conn["active"]:
                conn["active"] = True
                update_needed = True
        
        if update_needed or self.current_tab == "Global":
            self.update()
    
    def mousePressEvent(self, event):
        """Handle mouse press events for interaction"""
        if self.current_tab == "Global" and event.button() == Qt.LeftButton:
            # Check if clicked on a node
            if self.hovered_node is not None:
                # Toggle selection
                if self.selected_node == self.hovered_node:
                    self.selected_node = None
                else:
                    self.selected_node = self.hovered_node
                    
                    # Update metrics for the selected node
                    if self.selected_node is not None:
                        node = self.nodes[self.selected_node]
                        
                        # Generate metrics based on node properties
                        self.awareness_level = int(70 + node.get("activity", 0) * 30)
                        self.integration_index = round(0.5 + node.get("value", 0) * 0.5, 2)
                        
                        # Text values based on activity
                        if node.get("activity", 0) > 0.7:
                            self.neural_coherence = "Very High"
                        elif node.get("activity", 0) > 0.4:
                            self.neural_coherence = "High"
                        elif node.get("activity", 0) > 0.2:
                            self.neural_coherence = "Medium"
                        else:
                            self.neural_coherence = "Low"
                            
                        self.responsiveness = int(80 + node.get("activity", 0) * 20)
                        
                        # Update display
                        self.update_node_metrics()
                        
                        # Switch to node details tab
                        self.change_tab("Node")
                    
                self.update()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for hover detection"""
        if self.current_tab == "Global":
            old_hover = self.hovered_node
            self.hovered_node = None
            
            # Find node under mouse
            pos = self.viz_area.mapFromParent(event.pos())
            for i, node in enumerate(self.nodes):
                # Convert from logical coordinates to widget coordinates
                widget_x, widget_y = self.logical_to_widget(node["x"], node["y"])
                
                # Calculate distance to mouse
                distance = math.sqrt((pos.x() - widget_x)**2 + (pos.y() - widget_y)**2)
                
                # Use the current (potentially pulsating) radius for hit detection
                radius = node.get("current_radius", node["radius"])
                
                if distance < radius:
                    self.hovered_node = i
                    self.setCursor(Qt.PointingHandCursor)
                    break
            
            # Update cursor and repaint if hover state changed
            if old_hover != self.hovered_node:
                if self.hovered_node is None:
                    self.setCursor(Qt.ArrowCursor)
                self.update()
    
    def logical_to_widget(self, x, y):
        """Convert logical coordinates to widget coordinates"""
        if not self.viz_area:
            return (0, 0)
            
        # Get widget center
        center_x = self.viz_area.width() / 2
        center_y = self.viz_area.height() / 2
        
        # Calculate the scale factor based on widget size
        scale = min(self.viz_area.width(), self.viz_area.height()) / 500.0
        
        # Convert coordinates
        widget_x = center_x + x * scale
        widget_y = center_y + y * scale
        
        return (widget_x, widget_y)
    
    def paintEvent(self, event):
        """Custom paint event to render the node consciousness visualization"""
        super().paintEvent(event)
        
        # Only paint in global tab
        if self.current_tab != "Global":
            return
            
        # Get the visualization area
        viz_rect = self.viz_area.geometry()
        
        # Skip if area is too small
        if viz_rect.width() < 10 or viz_rect.height() < 10:
            return
            
        # Skip if no nodes
        if not self.nodes:
            return
        
        # Create painter for the entire widget
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Set clipping to visualization area
        painter.setClipRect(viz_rect)
        
        # Calculate translation to viz area
        painter.translate(viz_rect.x(), viz_rect.y())
        
        # Draw connections first (behind nodes)
        self.draw_connections(painter)
        
        # Then draw nodes
        self.draw_nodes(painter)
    
    def draw_connections(self, painter):
        """Draw connections between nodes"""
        for conn in self.connections:
            source = self.nodes[conn["source"]]
            target = self.nodes[conn["target"]]
            
            # Convert from logical to widget coordinates
            source_x, source_y = self.logical_to_widget(source["x"], source["y"])
            target_x, target_y = self.logical_to_widget(target["x"], target["y"])
            
            # Determine color and width based on connection state
            if conn["active"]:
                # For active connections, use a color based on the source & target nodes
                source_color = source["color"]
                target_color = target["color"]
                
                r = (source_color.red() + target_color.red()) // 2
                g = (source_color.green() + target_color.green()) // 2
                b = (source_color.blue() + target_color.blue()) // 2
                
                # Apply weight to alpha
                a = int(150 * conn["weight"])
                
                color = QtGui.QColor(r, g, b, a)
                width = 1 + 3 * conn["weight"]
            else:
                # Inactive connections are more transparent
                color = QtGui.QColor(120, 120, 120, 50)
                width = 1
            
            # Highlight connections to selected nodes
            if self.selected_node is not None:
                if conn["source"] == self.selected_node or conn["target"] == self.selected_node:
                    color = QtGui.QColor(231, 76, 60, 180)  # Red highlight
                    width += 1
            
            # Set pen for drawing
            pen = QtGui.QPen(color, width)
            if not conn["active"]:
                pen.setStyle(Qt.DotLine)
            painter.setPen(pen)
            
            # Draw connection line
            painter.drawLine(int(source_x), int(source_y), int(target_x), int(target_y))
    
    def draw_nodes(self, painter):
        """Draw the consciousness nodes"""
        for i, node in enumerate(self.nodes):
            # Convert from logical to widget coordinates
            x, y = self.logical_to_widget(node["x"], node["y"])
            
            # Get the node color
            color = node["color"]
            
            # Get current radius (may be pulsating)
            radius = node.get("current_radius", node["radius"])
            
            # Draw glow for active nodes
            activity = node.get("activity", 0)
            if activity > 0.1:
                glow_radius = radius * 2
                glow = QtGui.QRadialGradient(x, y, glow_radius)
                glow_color = QtGui.QColor(color)
                glow_color.setAlpha(int(80 * activity))
                glow.setColorAt(0, glow_color)
                glow.setColorAt(1, QtGui.QColor(0, 0, 0, 0))
                
                painter.setBrush(QtGui.QBrush(glow))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(
                    x - glow_radius, y - glow_radius,
                    glow_radius * 2, glow_radius * 2
                )
            
            # Determine the border color based on selection state
            if i == self.selected_node:
                border_color = QtGui.QColor(231, 76, 60)  # Red for selected
                border_width = 3
            elif i == self.hovered_node:
                border_color = QtGui.QColor(241, 196, 15)  # Yellow for hover
                border_width = 2
            else:
                # Default white border
                border_color = QtGui.QColor(255, 255, 255, 120)
                border_width = 1
            
            # Draw node
            painter.setBrush(QtGui.QBrush(color))
            painter.setPen(QtGui.QPen(border_color, border_width))
            painter.drawEllipse(
                x - radius, y - radius,
                radius * 2, radius * 2
            )
            
            # Draw node id in center
            if radius >= 15 or i == self.selected_node or i == self.hovered_node:
                font = QtGui.QFont("Arial", 9)
                if i == self.selected_node:
                    font.setBold(True)
                painter.setFont(font)
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
                painter.drawText(
                    QtCore.QRectF(x - radius, y - radius, radius * 2, radius * 2),
                    Qt.AlignCenter,
                    str(i)
                )
    
    def closeEvent(self, event):
        """Clean up when panel is closed"""
        # Stop animation timer
        if self.animation_timer and self.animation_timer.isActive():
            self.animation_timer.stop()
        
        super().closeEvent(event) 