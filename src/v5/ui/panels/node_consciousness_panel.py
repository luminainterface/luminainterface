"""
Node Consciousness Panel for V5 Visualization System

This panel visualizes neural node consciousness metrics,
showing activation levels, integration, awareness, and more.
"""

import os
import sys
import time
import random
from pathlib import Path

# Add project root to Python path if needed
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import Qt compatibility layer
from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui, Qt, Signal, Slot
from src.v5.ui.qt_compat import get_widgets, get_gui, get_core

# Get required Qt classes
QSplitter = get_widgets().QSplitter
QTimer = get_core().QTimer
QPainter = get_gui().QPainter
QColor = get_gui().QColor
QFont = get_gui().QFont
QBrush = get_gui().QBrush
QPen = get_gui().QPen
QPointF = get_core().QPointF
QRectF = get_core().QRectF

import json
import logging
import math
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NodeConsciousnessPanel(QtWidgets.QWidget):
    """Panel for visualizing neural node consciousness metrics"""
    
    # Signals
    node_selected = Signal(dict)  # Emitted when a node is selected
    
    def __init__(self, socket_manager=None):
        """
        Initialize the Node Consciousness Panel
        
        Args:
            socket_manager: Optional socket manager for plugin communication
        """
        super().__init__()
        
        # Component name for state persistence
        self.component_name = "node_consciousness_panel"
        
        # Initialize data
        self.socket_manager = socket_manager
        self.consciousness_data = None
        self.active_processes = []
        self.animation_timer = None
        self.animation_phase = 0.0
        
        # Set up UI
        self.initUI()
        
        # Connect to consciousness analytics plugin
        if socket_manager:
            self.connect_to_consciousness_analytics()
        
        # Start animation
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(50)  # 20 fps
    
    def initUI(self):
        """Initialize the user interface"""
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QtWidgets.QLabel("Node Consciousness Visualization")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #4B6EAF;")
        layout.addWidget(title)
        
        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Consciousness visualization
        self.consciousness_view = ConsciousnessVisualizationView()
        
        # Connect the node_clicked signal to show_node_details
        self.consciousness_view.node_clicked.connect(self.show_node_details)
        
        # Right: Tab widget with details
        tabs = QtWidgets.QTabWidget()
        
        # Global metrics tab
        global_metrics_widget = QtWidgets.QWidget()
        global_metrics_layout = QtWidgets.QVBoxLayout(global_metrics_widget)
        self.global_metrics_panel = GlobalMetricsPanel()
        global_metrics_layout.addWidget(self.global_metrics_panel)
        tabs.addTab(global_metrics_widget, "Global Metrics")
        
        # Node details tab
        node_details_widget = QtWidgets.QWidget()
        node_details_layout = QtWidgets.QVBoxLayout(node_details_widget)
        self.node_details_panel = NodeDetailsPanel()
        node_details_layout.addWidget(self.node_details_panel)
        tabs.addTab(node_details_widget, "Node Details")
        
        # Active processes tab
        processes_widget = QtWidgets.QWidget()
        processes_layout = QtWidgets.QVBoxLayout(processes_widget)
        self.processes_panel = ActiveProcessesPanel()
        processes_layout.addWidget(self.processes_panel)
        tabs.addTab(processes_widget, "Active Processes")
        
        # Add views to splitter
        splitter.addWidget(self.consciousness_view)
        splitter.addWidget(tabs)
        
        # Set initial sizes (3:2 ratio)
        splitter.setSizes([int(self.width() * 0.6), int(self.width() * 0.4)])
        
        layout.addWidget(splitter, 1)  # 1 is stretch factor
        
        # Status bar
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet("color: #666;")
        layout.addWidget(self.status_label)
    
    def connect_to_consciousness_analytics(self):
        """Connect to the consciousness analytics plugin"""
        try:
            # Get direct access to the consciousness analytics plugin
            plugin = self.socket_manager.get_plugin("consciousness_analytics_1")
            
            if plugin:
                logger.info(f"Found consciousness analytics plugin directly: {plugin.node_id}")
                
                # Connect to socket
                plugin.socket.connect_to(self.socket_manager.manager_socket)
                
                # Register for data updates
                self.socket_manager.register_message_handler(
                    "consciousness_data_updated", 
                    self.handle_consciousness_update
                )
                
                # Request initial consciousness data
                self.request_consciousness_data()
                
                self.status_label.setText("Connected to Consciousness Analytics")
                return
                
            # Try to directly connect to the consciousness analytics plugin by ID
            connected = self.socket_manager.establish_direct_connection(
                self, 
                "consciousness_analytics_1", 
                "consciousness_view"
            )
            
            if connected:
                # Register for consciousness data updates if not already registered
                self.socket_manager.register_message_handler(
                    "consciousness_data_updated", 
                    self.handle_consciousness_update
                )
                
                # Request initial consciousness data
                self.request_consciousness_data()
                
                self.status_label.setText("Connected to Consciousness Analytics")
                return
                
            # Also try connecting with alternative component name
            connected = self.socket_manager.establish_direct_connection(
                self, 
                "consciousness_analytics_1", 
                "consciousness_meter"
            )
            
            if connected:
                # Register for consciousness data updates if not already registered
                self.socket_manager.register_message_handler(
                    "consciousness_data_updated", 
                    self.handle_consciousness_update
                )
                
                # Request initial consciousness data
                self.request_consciousness_data()
                
                self.status_label.setText("Connected to Consciousness Analytics")
                return
            
            # Fallback to traditional component provider approach
            providers = self.socket_manager.get_ui_component_providers("consciousness_meter")
            if providers:
                plugin_id = providers[0]["plugin"].node_id
                logger.info(f"Connecting to consciousness analytics plugin: {plugin_id}")
                
                # Connect socket
                self.socket_manager.connect_ui_to_plugin(self, plugin_id)
                
                # Register for consciousness data updates
                self.socket_manager.register_message_handler(
                    "consciousness_data_updated", 
                    self.handle_consciousness_update
                )
                
                # Request initial consciousness data
                self.request_consciousness_data()
                
                self.status_label.setText("Connected to Consciousness Analytics")
            else:
                logger.warning("No consciousness analytics plugin found")
                self.status_label.setText("No Consciousness Analytics plugin found")
        except Exception as e:
            logger.error(f"Error connecting to consciousness analytics: {str(e)}")
            self.status_label.setText(f"Connection error: {str(e)}")
    
    def request_consciousness_data(self):
        """Request consciousness data from plugin"""
        request_id = f"consciousness_request_{int(time.time())}"
        message = {
            "type": "request_consciousness_data",
            "request_id": request_id,
            "plugin_id": "consciousness_analytics_1",  # Explicitly target the plugin
            "content": {
                "include_details": True,
                "metrics": ["self_awareness", "integration", "coherence", "responsiveness"]
            }
        }
        logger.info(f"Sending consciousness data request with ID: {request_id}")
        self.socket_manager.send_message(message)
    
    def handle_consciousness_update(self, message):
        """Handle consciousness data update messages"""
        try:
            data = message.get("data", {})
            if "error" in data:
                error_msg = data.get("error", "Unknown error")
                self.status_label.setText(f"Error: {error_msg}")
                return
            
            # Update consciousness data
            self.consciousness_data = data
            
            # Extract nodes and connections
            nodes = data.get("nodes", [])
            connections = data.get("connections", [])
            
            # If we have no nodes, generate some placeholder data
            if not nodes or len(nodes) == 0:
                logger.warning("No nodes data received, using generated data for visualization")
                nodes, connections = self._generate_sample_nodes()
            
            # Update visualization
            self.consciousness_view.update_visualization(nodes, connections)
            
            # Update global metrics
            global_metrics = data.get("global_metrics", {})
            if not global_metrics:
                global_metrics = {
                    "awareness_level": 87,
                    "integration_index": 0.77,
                    "neural_coherence": "High",
                    "responsiveness": 94
                }
            
            self.global_metrics_panel.update_metrics(global_metrics)
            
            # Update active processes
            self.active_processes = data.get("active_processes", [])
            if not self.active_processes:
                self.active_processes = [
                    "Information integration across neural networks",
                    "Dynamic complexity modulation",
                    "Neural synchronization",
                    "Temporal binding"
                ]
            
            self.processes_panel.update_processes(self.active_processes)
            
            self.status_label.setText("Consciousness data updated")
        except Exception as e:
            logger.error(f"Error handling consciousness update: {str(e)}")
            self.status_label.setText(f"Error processing update: {str(e)}")
    
    def _generate_sample_nodes(self):
        """Generate sample nodes and connections for visualization when no data is available"""
        nodes = []
        connections = []
        node_count = 30
        
        # Generate nodes in a circular layout
        for i in range(node_count):
            # Determine node type
            node_types = ["perception", "processing", "integration", "motor"]
            node_type = node_types[i % len(node_types)]
            
            # Determine activation and consciousness levels
            activation = random.uniform(0.3, 0.9)
            consciousness = random.uniform(0.4, 0.8)
            
            # Create node
            node = {
                "id": str(i),
                "name": f"Node {i}",
                "type": node_type,
                "activation": activation,
                "consciousness": consciousness,
                "metrics": {
                    "self_awareness": random.uniform(0.3, 0.9),
                    "integration": random.uniform(0.5, 0.9),
                    "memory_access": random.uniform(0.2, 0.8),
                    "reflection": random.uniform(0.1, 0.7)
                }
            }
            nodes.append(node)
            
            # Generate connections
            # Each node connects to 2-4 other nodes
            connection_count = random.randint(2, 4)
            for _ in range(connection_count):
                target = random.randint(0, node_count - 1)
                if target != i:  # Don't connect to self
                    conn_type = "excitatory" if random.random() > 0.3 else "inhibitory"
                    connection = {
                        "source": str(i),
                        "target": str(target),
                        "strength": random.uniform(0.3, 0.9),
                        "type": conn_type
                    }
                    connections.append(connection)
        
        return nodes, connections
    
    def show_node_details(self, node_id):
        """Show details for a specific node"""
        try:
            # Log selection
            logger.info(f"Node selected: {node_id}")
            
            # If we don't have consciousness data or it's empty, generate some
            if not self.consciousness_data or not self.consciousness_data.get("nodes"):
                nodes, connections = self._generate_sample_nodes()
                self.consciousness_data = {
                    "nodes": nodes,
                    "connections": connections,
                    "global_metrics": {
                        "awareness_level": 87,
                        "integration_index": 0.77,
                        "neural_coherence": "High",
                        "responsiveness": 94
                    },
                    "active_processes": [
                        "Information integration across neural networks",
                        "Dynamic complexity modulation",
                        "Neural synchronization",
                        "Temporal binding"
                    ]
                }
            
            # Find the node
            nodes = self.consciousness_data.get("nodes", [])
            node = next((n for n in nodes if n.get("id") == node_id), None)
            
            if node:
                # Update node details panel
                self.node_details_panel.update_node_details(node)
                
                # Generate node-specific processes
                node_type = node.get("type", "")
                node_specific_processes = []
                
                if node_type == "perception":
                    node_specific_processes = [
                        "Sensory input processing",
                        "Feature detection",
                        "Pattern recognition",
                        "Input signal filtering"
                    ]
                elif node_type == "processing":
                    node_specific_processes = [
                        "Information transformation",
                        "Statistical analysis",
                        "Context integration",
                        "Parallel processing streams"
                    ]
                elif node_type == "integration":
                    node_specific_processes = [
                        "Cross-modal binding",
                        "Temporal synchronization",
                        "Global workspace integration",
                        "Information synthesis"
                    ]
                elif node_type == "motor":
                    node_specific_processes = [
                        "Action planning",
                        "Response coordination",
                        "Execution monitoring",
                        "Feedback processing"
                    ]
                else:
                    node_specific_processes = [
                        "General information processing",
                        "Network coordination",
                        "Adaptive response",
                        "Signal propagation"
                    ]
                
                # Update active processes panel with node-specific processes
                self.processes_panel.update_processes(node_specific_processes)
                
                # Update status
                self.status_label.setText(f"Node {node_id} selected")
                
                # Emit signal
                self.node_selected.emit(node)
            else:
                # If node not found, show a message
                self.status_label.setText(f"Node {node_id} not found")
        except Exception as e:
            logger.error(f"Error showing node details: {str(e)}")
            self.status_label.setText(f"Error showing node details: {str(e)}")
    
    def update_animation(self):
        """Update animation state"""
        self.animation_phase += 0.05
        if self.animation_phase > 2 * math.pi:
            self.animation_phase -= 2 * math.pi
            
        self.consciousness_view.set_animation_phase(self.animation_phase)
        self.consciousness_view.update()
    
    def update_visualization(self):
        """Update visualization with latest data or generate fallback data if needed"""
        try:
            self.request_consciousness_data()
        except Exception as e:
            logger.error(f"Error requesting consciousness data: {str(e)}")
            # Generate fallback data if request fails
            nodes, connections = self._generate_sample_nodes()
            self.consciousness_view.update_visualization(nodes, connections)
            
            # Update global metrics with fallback data
            global_metrics = {
                "awareness_level": 87,
                "integration_index": 0.77,
                "neural_coherence": "High",
                "responsiveness": 94
            }
            self.global_metrics_panel.update_metrics(global_metrics)
            
            # Update active processes with fallback data
            fallback_processes = [
                "Information integration across neural networks",
                "Dynamic complexity modulation",
                "Neural synchronization",
                "Temporal binding"
            ]
            self.processes_panel.update_processes(fallback_processes)
    
    def cleanup(self):
        """Clean up resources before closing"""
        # Stop animation timer
        if self.animation_timer and self.animation_timer.isActive():
            self.animation_timer.stop()
            
        # Deregister message handlers
        self.socket_manager.deregister_message_handler("consciousness_data_updated")

    def restore_state(self, state):
        """
        Restore panel state from saved data
        
        Args:
            state: State data dictionary
        """
        try:
            # Update consciousness data if available
            if "global_metrics" in state:
                if not self.consciousness_data:
                    self.consciousness_data = {}
                
                self.consciousness_data["global_metrics"] = state["global_metrics"]
                self.global_metrics_panel.update_metrics(state["global_metrics"])
                
            # Update nodes and connections if available
            if "nodes" in state and "connections" in state:
                nodes = state["nodes"]
                connections = state["connections"]
                
                if not self.consciousness_data:
                    self.consciousness_data = {}
                    
                self.consciousness_data["nodes"] = nodes
                self.consciousness_data["connections"] = connections
                
                self.consciousness_view.update_visualization(nodes, connections)
                
            # Update active processes if available
            if "active_processes" in state:
                self.active_processes = state["active_processes"]
                self.processes_panel.update_processes(self.active_processes)
                
            logger.info(f"Restored state for {self.component_name}")
        except Exception as e:
            logger.error(f"Error restoring state: {str(e)}")


class ConsciousnessVisualizationView(QtWidgets.QWidget):
    """Custom widget for visualizing neural network consciousness"""
    
    # Signals
    node_clicked = Signal(str)  # Emitted when a node is clicked
    
    def __init__(self):
        super().__init__()
        self.nodes = []
        self.connections = []
        self.node_positions = {}
        self.selected_node = None
        self.animation_phase = 0.0
        
        # Set up widget
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
    
    def update_visualization(self, nodes, connections):
        """Update the visualization with new data"""
        self.nodes = nodes
        self.connections = connections
        
        # Calculate node positions in a circle
        self._calculate_node_positions()
        
        # Trigger repaint
        self.update()
    
    def set_animation_phase(self, phase):
        """Set the animation phase"""
        self.animation_phase = phase
    
    def _calculate_node_positions(self):
        """Calculate positions for nodes in a circle layout"""
        if not self.nodes:
            return
            
        # Get widget center
        center_x = self.width() / 2
        center_y = self.height() / 2
        
        # Calculate radius based on widget size
        radius = min(center_x, center_y) * 0.8
        
        # Position nodes in a circle
        for i, node in enumerate(self.nodes):
            angle = (i / len(self.nodes)) * 2 * math.pi
            
            # Calculate position
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # Store position
            self.node_positions[node.get("id")] = (x, y)
    
    def paintEvent(self, event):
        """Paint the consciousness visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(240, 245, 250))
        
        # Skip if no data
        if not self.nodes or not self.connections:
            # Draw instructions
            painter.setPen(QColor(100, 100, 100))
            painter.setFont(QFont("Arial", 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No data available.\nClick Refresh to generate visualization.")
            
            # Draw refresh message in smaller text
            painter.setFont(QFont("Arial", 10))
            text_rect = QRectF(self.rect().x(), self.rect().y() + 50, self.rect().width(), 30)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, "(Nodes will be clickable when available)")
            return
        
        # Recalculate node positions if needed
        if len(self.node_positions) != len(self.nodes):
            self._calculate_node_positions()
        
        # Draw connections first
        for connection in self.connections:
            source_id = connection.get("source")
            target_id = connection.get("target")
            
            if source_id in self.node_positions and target_id in self.node_positions:
                # Get positions
                x1, y1 = self.node_positions[source_id]
                x2, y2 = self.node_positions[target_id]
                
                # Get connection properties
                strength = connection.get("strength", 0.5)
                conn_type = connection.get("type", "default")
                
                # Determine color and width based on type and strength
                if conn_type == "excitatory":
                    color = QColor(100, 200, 100, int(255 * strength))
                    width = 1 + 3 * strength
                elif conn_type == "inhibitory":
                    color = QColor(200, 100, 100, int(255 * strength))
                    width = 1 + 3 * strength
                else:
                    color = QColor(150, 150, 150, int(200 * strength))
                    width = 1 + 2 * strength
                
                # Set pen
                painter.setPen(QPen(color, width))
                
                # Draw animated connection
                if self.animation_phase is not None:
                    # Animated flow along connection
                    dx = x2 - x1
                    dy = y2 - y1
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    # Number of particles based on strength
                    particle_count = int(3 * strength) + 1
                    
                    for i in range(particle_count):
                        # Particle position along line
                        t = ((self.animation_phase + i/particle_count) % 1.0)
                        
                        # Particle position
                        px = x1 + t * dx
                        py = y1 + t * dy
                        
                        # Particle size based on strength
                        particle_size = 3 * strength
                        
                        # Draw particle
                        painter.setBrush(QBrush(color))
                        painter.setPen(Qt.NoPen)
                        painter.drawEllipse(QPointF(px, py), particle_size, particle_size)
                
                # Draw connection line
                painter.setPen(QPen(color.lighter(150), width/2))
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Draw nodes
        for node in self.nodes:
            node_id = node.get("id")
            if node_id not in self.node_positions:
                continue
                
            # Get position
            x, y = self.node_positions[node_id]
            
            # Get node properties
            activation = node.get("activation", 0.5)
            consciousness = node.get("consciousness", 0.5)
            node_type = node.get("type", "default")
            
            # Node size based on consciousness
            base_size = 20 + 20 * consciousness
            
            # Animation - subtle pulsing based on activation
            if self.animation_phase is not None:
                pulse = 1.0 + 0.2 * activation * math.sin(self.animation_phase)
                size = base_size * pulse
            else:
                size = base_size
            
            # Node color based on type
            if node_type == "perception":
                base_color = QColor(75, 110, 175)  # Blue
            elif node_type == "processing":
                base_color = QColor(75, 175, 110)  # Green
            elif node_type == "motor":
                base_color = QColor(175, 75, 110)  # Red
            elif node_type == "integration":
                base_color = QColor(175, 110, 75)  # Orange
            else:
                base_color = QColor(110, 110, 110)  # Gray
            
            # Adjust color based on activation
            color = base_color.lighter(100 + int(activation * 100))
            
            # Highlight selected node
            if node_id == self.selected_node:
                painter.setPen(QPen(QColor(40, 40, 40), 2))
                painter.setBrush(QBrush(color.lighter(120)))
            else:
                painter.setPen(QPen(QColor(40, 40, 40), 1))
                painter.setBrush(QBrush(color))
            
            # Draw node
            painter.drawEllipse(QPointF(x, y), size/2, size/2)
            
            # Draw consciousness level indicator
            if consciousness > 0:
                # Inner circle representing consciousness level
                inner_size = size * 0.6
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(255, 255, 255, 180)))
                painter.drawEllipse(QPointF(x, y), inner_size/2, inner_size/2)
            
            # Draw node label
            name = node.get("name", f"Node {node_id}")
            painter.setPen(QColor(40, 40, 40))
            painter.setFont(QFont("Arial", 9))
            painter.drawText(QRectF(x-60, y+size/2+5, 120, 20), Qt.AlignmentFlag.AlignCenter, name)
    
    def mousePressEvent(self, event):
        """Handle mouse press event"""
        # Check if mouse is over a node
        pos = event.position()
        x, y = pos.x(), pos.y()
        
        for node in self.nodes:
            node_id = node.get("id")
            if node_id not in self.node_positions:
                continue
                
            # Get node position
            node_x, node_y = self.node_positions[node_id]
            
            # Calculate distance to node center
            dx = x - node_x
            dy = y - node_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Node size based on consciousness
            consciousness = node.get("consciousness", 0.5)
            size = 20 + 20 * consciousness
            
            # Check if distance is within node
            if distance <= size/2:
                # Select node
                self.selected_node = node_id
                self.update()
                
                # Emit signal
                self.node_clicked.emit(node_id)
                return
        
        # Clear selection if clicking on background
        self.selected_node = None
        self.update()

    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        # Check if mouse is over a node and change cursor accordingly
        pos = event.position()
        x, y = pos.x(), pos.y()
        
        for node in self.nodes:
            node_id = node.get("id")
            if node_id not in self.node_positions:
                continue
                
            # Get node position
            node_x, node_y = self.node_positions[node_id]
            
            # Calculate distance to node center
            dx = x - node_x
            dy = y - node_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Node size based on consciousness
            consciousness = node.get("consciousness", 0.5)
            size = 20 + 20 * consciousness
            
            # Check if distance is within node
            if distance <= size/2:
                # Change cursor to indicate clickable node
                self.setCursor(Qt.PointingHandCursor)
                return
        
        # Reset cursor if not over a node
        self.setCursor(Qt.ArrowCursor)


class GlobalMetricsPanel(QtWidgets.QWidget):
    """Panel for displaying global consciousness metrics"""
    
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        """Initialize the user interface"""
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        
        # Metrics
        metrics_layout = QtWidgets.QVBoxLayout()
        
        # Awareness
        awareness_group = QtWidgets.QGroupBox("Awareness Level")
        awareness_layout = QtWidgets.QVBoxLayout(awareness_group)
        self.awareness_label = QtWidgets.QLabel("87")
        self.awareness_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4B6EAF;")
        self.awareness_label.setAlignment(Qt.AlignCenter)
        awareness_layout.addWidget(self.awareness_label)
        metrics_layout.addWidget(awareness_group)
        
        # Integration
        integration_group = QtWidgets.QGroupBox("Integration Index")
        integration_layout = QtWidgets.QVBoxLayout(integration_group)
        self.integration_label = QtWidgets.QLabel("0.72")
        self.integration_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4B6EAF;")
        self.integration_label.setAlignment(Qt.AlignCenter)
        integration_layout.addWidget(self.integration_label)
        metrics_layout.addWidget(integration_group)
        
        # Coherence
        coherence_group = QtWidgets.QGroupBox("Neural Coherence")
        coherence_layout = QtWidgets.QVBoxLayout(coherence_group)
        self.coherence_label = QtWidgets.QLabel("High")
        self.coherence_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4B6EAF;")
        self.coherence_label.setAlignment(Qt.AlignCenter)
        coherence_layout.addWidget(self.coherence_label)
        metrics_layout.addWidget(coherence_group)
        
        # Responsiveness
        responsiveness_group = QtWidgets.QGroupBox("Responsiveness")
        responsiveness_layout = QtWidgets.QVBoxLayout(responsiveness_group)
        self.responsiveness_label = QtWidgets.QLabel("94")
        self.responsiveness_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4B6EAF;")
        self.responsiveness_label.setAlignment(Qt.AlignCenter)
        responsiveness_layout.addWidget(self.responsiveness_label)
        metrics_layout.addWidget(responsiveness_group)
        
        layout.addLayout(metrics_layout)
        
        # Add stretch to fill remaining space
        layout.addStretch(1)
    
    def update_metrics(self, metrics):
        """Update metrics display with new data"""
        if not metrics:
            return
            
        self.awareness_label.setText(str(metrics.get("awareness_level", "N/A")))
        self.integration_label.setText(str(metrics.get("integration_index", "N/A")))
        self.coherence_label.setText(str(metrics.get("neural_coherence", "N/A")))
        self.responsiveness_label.setText(str(metrics.get("responsiveness", "N/A")))


class NodeDetailsPanel(QtWidgets.QWidget):
    """Panel for displaying node details"""
    
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        """Initialize the user interface"""
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        
        # Node name
        self.name_label = QtWidgets.QLabel("No node selected")
        self.name_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #4B6EAF;")
        layout.addWidget(self.name_label)
        
        # Node type
        self.type_label = QtWidgets.QLabel("Type: N/A")
        layout.addWidget(self.type_label)
        
        # Metrics
        metrics_group = QtWidgets.QGroupBox("Node Metrics")
        metrics_layout = QtWidgets.QVBoxLayout(metrics_group)
        
        # Self-awareness
        self.self_awareness_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.self_awareness_slider.setRange(0, 100)
        self.self_awareness_slider.setValue(0)
        self.self_awareness_slider.setEnabled(False)
        metrics_layout.addWidget(QtWidgets.QLabel("Self-Awareness:"))
        metrics_layout.addWidget(self.self_awareness_slider)
        
        # Integration
        self.integration_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.integration_slider.setRange(0, 100)
        self.integration_slider.setValue(0)
        self.integration_slider.setEnabled(False)
        metrics_layout.addWidget(QtWidgets.QLabel("Integration:"))
        metrics_layout.addWidget(self.integration_slider)
        
        # Memory access
        self.memory_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.memory_slider.setRange(0, 100)
        self.memory_slider.setValue(0)
        self.memory_slider.setEnabled(False)
        metrics_layout.addWidget(QtWidgets.QLabel("Memory Access:"))
        metrics_layout.addWidget(self.memory_slider)
        
        # Reflection
        self.reflection_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.reflection_slider.setRange(0, 100)
        self.reflection_slider.setValue(0)
        self.reflection_slider.setEnabled(False)
        metrics_layout.addWidget(QtWidgets.QLabel("Reflection:"))
        metrics_layout.addWidget(self.reflection_slider)
        
        layout.addWidget(metrics_group)
        
        # Add stretch to fill remaining space
        layout.addStretch(1)
    
    def update_node_details(self, node):
        """Update node details display with new data"""
        if not node:
            return
            
        # Update labels
        self.name_label.setText(node.get("name", f"Node {node.get('id')}"))
        self.type_label.setText(f"Type: {node.get('type', 'N/A')}")
        
        # Update metrics
        metrics = node.get("metrics", {})
        
        self.self_awareness_slider.setValue(int(metrics.get("self_awareness", 0) * 100))
        self.integration_slider.setValue(int(metrics.get("integration", 0) * 100))
        self.memory_slider.setValue(int(metrics.get("memory_access", 0) * 100))
        self.reflection_slider.setValue(int(metrics.get("reflection", 0) * 100))


class ActiveProcessesPanel(QtWidgets.QWidget):
    """Panel for displaying active processes"""
    
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        """Initialize the user interface"""
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title = QtWidgets.QLabel("Active Processes")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #4B6EAF;")
        layout.addWidget(title)
        
        # Process list
        self.process_list = QtWidgets.QLabel("No active processes")
        self.process_list.setStyleSheet("color: #666;")
        self.process_list.setWordWrap(True)
        layout.addWidget(self.process_list)
        
        # Add stretch to fill remaining space
        layout.addStretch(1)
    
    def update_processes(self, processes):
        """Update process list with new data"""
        if not processes:
            self.process_list.setText("No active processes")
            return
            
        # Format processes as bulleted list
        process_text = "<ul>"
        for process in processes:
            process_text += f"<li>{process}</li>"
        process_text += "</ul>"
        
        self.process_list.setText(process_text) 