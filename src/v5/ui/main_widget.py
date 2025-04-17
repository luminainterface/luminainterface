"""
Main widget for the V5 Fractal Echo Visualization System.
"""

# Replace direct PySide6 imports with compatibility layer
from ..ui.qt_compat import QtWidgets, QtCore, QtGui, Qt, Signal, Slot
from ..ui.qt_compat import get_widgets, get_gui, get_core

# Required Qt classes
QSplitter = get_widgets().QSplitter
QPainter = get_gui().QPainter
QLinearGradient = get_gui().QLinearGradient
QColor = get_gui().QColor

# Keep the existing panel imports
from .panels.fractal_pattern_panel import FractalPatternPanel
from .panels.node_consciousness_panel import NodeConsciousnessPanel
from .panels.network_visualization_panel import NetworkVisualizationPanel
from .panels.memory_synthesis_panel import MemorySynthesisPanel
from .panels.metrics_panel import MetricsPanel
from .panels.conversation_panel import ConversationPanel

# Add logging
import logging
logger = logging.getLogger(__name__)

class PanelContainer(QtWidgets.QWidget):
    """A container for panels with custom styling and a title bar"""
    
    def __init__(self, title, panel, parent=None):
        super().__init__(parent)
        self.panel = panel
        self.title = title
        self.initUI()
        
    def initUI(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create title bar
        title_bar = QtWidgets.QWidget()
        title_bar.setFixedHeight(30)
        title_bar.setStyleSheet("""
            background-color: #1A2634;
            border-bottom: 1px solid #34495E;
        """)
        
        title_layout = QtWidgets.QHBoxLayout(title_bar)
        title_layout.setContentsMargins(10, 0, 10, 0)
        
        title_label = QtWidgets.QLabel(self.title)
        title_label.setStyleSheet("""
            color: #3498DB;
            font-weight: bold;
            font-size: 14px;
        """)
        
        # Add buttons for expanding/collapsing (icons would be better in a real implementation)
        expand_button = QtWidgets.QPushButton("↕")
        expand_button.setFixedSize(24, 24)
        expand_button.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50;
                color: #ECF0F1;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #34495E;
            }
        """)
        expand_button.clicked.connect(self.toggleExpand)
        
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(expand_button)
        
        # Add title bar and panel to layout
        layout.addWidget(title_bar)
        layout.addWidget(self.panel, 1)
        
    def toggleExpand(self):
        """Toggle the panel between collapsed and expanded states"""
        if self.panel.isVisible():
            self.panel.setVisible(False)
        else:
            self.panel.setVisible(True)

class V5MainWidget(QtWidgets.QWidget):
    """Main widget containing all visualization panels."""
    
    def __init__(self, socket_manager):
        super().__init__()
        self.socket_manager = socket_manager
        self.initUI()
        
    def initUI(self):
        """Initialize the user interface."""
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Add toolbar for global controls
        toolbar = self.createToolbar()
        layout.addWidget(toolbar)
        
        # Create main splitter for resizable sections
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #34495E;
            }
        """)
        
        # Left section - Fractal Pattern and Node Consciousness
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        
        # Fractal pattern visualization
        self.fractal_panel = FractalPatternPanel(self.socket_manager)
        fractal_container = PanelContainer("Fractal Pattern Visualization", self.fractal_panel)
        
        # Node consciousness visualization
        self.consciousness_panel = NodeConsciousnessPanel(self.socket_manager)
        consciousness_container = PanelContainer("Node Consciousness", self.consciousness_panel)
        
        # Create vertical splitter for left panels
        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.setHandleWidth(1)
        left_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #34495E;
            }
        """)
        left_splitter.addWidget(fractal_container)
        left_splitter.addWidget(consciousness_container)
        left_splitter.setSizes([500, 500])
        
        left_layout.addWidget(left_splitter)
        
        # Center section - Network Visualization and Conversation Panel
        center_widget = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        
        # Create vertical splitter for center section
        center_splitter = QSplitter(Qt.Vertical)
        center_splitter.setHandleWidth(1)
        center_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #34495E;
            }
        """)
        
        # Network visualization
        self.network_panel = NetworkVisualizationPanel(self.socket_manager)
        network_container = PanelContainer("Neural Network Visualization", self.network_panel)
        
        # NN/LLM weighted conversation panel
        self.conversation_panel = ConversationPanel(self.socket_manager)
        conversation_container = PanelContainer("NN/LLM Weighted Conversation", self.conversation_panel)
        
        center_splitter.addWidget(network_container)
        center_splitter.addWidget(conversation_container)
        
        # Set initial sizes for center splitter (60% network, 40% conversation)
        center_splitter.setSizes([600, 400])
        
        center_layout.addWidget(center_splitter)
        
        # Right section - Memory Synthesis and Metrics
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        
        # Create vertical splitter for right panels
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setHandleWidth(1)
        right_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #34495E;
            }
        """)
        
        # Memory synthesis visualization
        self.memory_panel = MemorySynthesisPanel(self.socket_manager)
        memory_container = PanelContainer("Memory Synthesis", self.memory_panel)
        
        # Metrics display
        self.metrics_panel = MetricsPanel(self.socket_manager)
        metrics_container = PanelContainer("System Metrics", self.metrics_panel)
        
        right_splitter.addWidget(memory_container)
        right_splitter.addWidget(metrics_container)
        right_splitter.setSizes([500, 500])
        
        right_layout.addWidget(right_splitter)
        
        # Add widgets to main splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(center_widget)
        splitter.addWidget(right_widget)
        
        # Set initial sizes (1:2:1 ratio)
        total_width = self.width()
        splitter.setSizes([total_width//4, total_width//2, total_width//4])
        
        # Add splitter to main layout
        layout.addWidget(splitter, 1)
        
        # Add status bar
        status_bar = self.createStatusBar()
        layout.addWidget(status_bar)
        
        # Connect signals
        self.fractal_panel.pattern_selected.connect(self.network_panel.highlight_pattern)
        self.consciousness_panel.node_selected.connect(self.network_panel.focus_node)
        self.network_panel.node_activated.connect(self.consciousness_panel.show_node_details)
        self.memory_panel.memory_selected.connect(self.network_panel.show_memory_path)
        
        # Connect conversation panel signals
        self.conversation_panel.message_sent.connect(self.on_message_sent)
        self.conversation_panel.weight_changed.connect(self.on_weight_changed)
        
    def createToolbar(self):
        """Create a toolbar with global controls"""
        toolbar = QtWidgets.QWidget()
        toolbar.setFixedHeight(40)
        toolbar.setStyleSheet("""
            background-color: #1A2634;
            border-bottom: 1px solid #34495E;
        """)
        
        layout = QtWidgets.QHBoxLayout(toolbar)
        layout.setContentsMargins(10, 0, 10, 0)
        
        # Add title
        title = QtWidgets.QLabel("V5 Fractal Echo Visualization")
        title.setStyleSheet("""
            color: #3498DB;
            font-weight: bold;
            font-size: 16px;
        """)
        
        # Add control buttons
        refresh_button = QtWidgets.QPushButton("Refresh")
        refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #2980B9;
                color: white;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #3498DB;
            }
        """)
        refresh_button.clicked.connect(self.update_all)
        
        settings_button = QtWidgets.QPushButton("Settings")
        settings_button.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50;
                color: white;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #34495E;
            }
        """)
        
        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(refresh_button)
        layout.addWidget(settings_button)
        
        return toolbar
        
    def createStatusBar(self):
        """Create a status bar for system information"""
        status_bar = QtWidgets.QWidget()
        status_bar.setFixedHeight(30)
        status_bar.setStyleSheet("""
            background-color: #1A2634;
            border-top: 1px solid #34495E;
        """)
        
        layout = QtWidgets.QHBoxLayout(status_bar)
        layout.setContentsMargins(10, 0, 10, 0)
        
        # Add system status indicators
        memory_status = QtWidgets.QLabel("Memory System: Active")
        memory_status.setStyleSheet("""
            color: #2ECC71;
            font-size: 12px;
        """)
        
        neural_status = QtWidgets.QLabel("Neural Processing: Active")
        neural_status.setStyleSheet("""
            color: #2ECC71;
            font-size: 12px;
        """)
        
        version_info = QtWidgets.QLabel("V5.0.1")
        version_info.setStyleSheet("""
            color: #7F8C8D;
            font-size: 12px;
        """)
        version_info.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        layout.addWidget(memory_status)
        layout.addSpacing(20)
        layout.addWidget(neural_status)
        layout.addStretch()
        layout.addWidget(version_info)
        
        return status_bar
        
    def toggle_view(self, view_name):
        """Toggle visibility of a specific view."""
        views = {
            'fractal_pattern': self.fractal_panel,
            'node_consciousness': self.consciousness_panel,
            'network': self.network_panel,
            'memory_synthesis': self.memory_panel,
            'metrics': self.metrics_panel,
            'conversation': self.conversation_panel
        }
        
        if view_name in views:
            views[view_name].setVisible(not views[view_name].isVisible())
    
    def on_message_sent(self, message):
        """Handle a message sent from the conversation panel"""
        # In a full implementation, this would connect to the language memory system
        # and neural network processing pipeline
        pass
    
    def on_weight_changed(self, weight):
        """Handle NN/LLM weight change from conversation panel"""
        # In a full implementation, this would update the weighting in the 
        # neural network and language model integration components
        pass
            
    def update_all(self):
        """Update all visualization panels."""
        self.fractal_panel.update_visualization()
        self.consciousness_panel.update_visualization()
        self.network_panel.update_visualization()
        self.memory_panel.update_visualization()
        self.metrics_panel.update_metrics()
        
    def cleanup(self):
        """Clean up resources before closing."""
        self.fractal_panel.cleanup()
        self.consciousness_panel.cleanup()
        self.network_panel.cleanup()
        self.memory_panel.cleanup()
        self.metrics_panel.cleanup()
        self.conversation_panel.cleanup()
        
    def paintEvent(self, event):
        """Custom background paint event"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill with gradient background
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(14, 22, 33))  # Darker blue at top
        gradient.setColorAt(1, QColor(23, 36, 55))  # Lighter blue at bottom
        
        painter.fillRect(self.rect(), gradient)
        
    def handle_bridge_message(self, source, message_type, data):
        """
        Handle messages from the version bridge system.
        
        Args:
            source: Source of the message (e.g., "v1v2", "v3v4", "v5_language")
            message_type: Type of the message
            data: Message data
        
        Returns:
            bool: True if the message was handled successfully, False otherwise
        """
        logger.info(f"Received bridge message from {source}: {message_type}")
        
        try:
            # Handle messages based on type
            if message_type == "memory_update":
                # Update memory synthesis panel
                if hasattr(self, "memory_panel"):
                    self.memory_panel.update_memory(data)
                    logger.info("Updated memory synthesis panel")
                
            elif message_type == "topic_update":
                # Update conversation panel with topics
                if hasattr(self, "conversation_panel"):
                    self.conversation_panel.update_topics(data)
                    logger.info("Updated conversation panel with topics")
                
            elif message_type == "fractal_pattern":
                # Update fractal pattern panel
                if hasattr(self, "fractal_panel"):
                    self.fractal_panel.update_pattern(data)
                    logger.info("Updated fractal pattern panel")
                
            elif message_type == "v1v2_text_to_v5":
                # Handle text from v1v2 interface
                if hasattr(self, "conversation_panel"):
                    self.conversation_panel.add_message(data.get("text", ""), "v1v2")
                    logger.info("Added message from v1v2 interface")
                
            elif message_type == "v3v4_breath_to_v5":
                # Handle breath state from v3v4 interface
                if hasattr(self, "consciousness_panel"):
                    self.consciousness_panel.update_breath_state(data.get("state", "normal"))
                    logger.info("Updated breath state from v3v4 interface")
                
            elif message_type == "v3v4_glyph_to_v5":
                # Handle glyph from v3v4 interface
                if hasattr(self, "network_panel"):
                    self.network_panel.update_glyph(data.get("glyph", ""))
                    logger.info("Updated glyph from v3v4 interface")
                
            elif message_type == "v3v4_resonance_to_v5":
                # Handle neural resonance from v3v4 interface
                if hasattr(self, "metrics_panel"):
                    self.metrics_panel.update_resonance(data.get("resonance", 0.0))
                    logger.info("Updated resonance from v3v4 interface")
                
            else:
                logger.warning(f"Unknown message type from bridge: {message_type}")
                return False
            
            # Update status bar if available
            if hasattr(self, "status_label"):
                self.status_label.setText(f"Last bridge update: {message_type} from {source}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling bridge message: {str(e)}")
            return False
            
    def send_to_bridge(self, destination, message_type, data):
        """
        Send a message to the bridge system.
        
        Args:
            destination: Destination for the message (e.g., "v1v2", "v3v4", "v5_language")
            message_type: Type of the message
            data: Message data
        
        Returns:
            bool: True if the message was sent successfully, False otherwise
        """
        # This method will be used by panels to send messages through the bridge
        # The actual bridge manager will be connected in the main.py file
        
        # Emit a custom signal that main.py can connect to the bridge manager
        # For now, just log the attempt
        logger.info(f"Attempting to send bridge message to {destination}: {message_type}")
        
        # In the future, this could be implemented by having the main application
        # pass a bridge_manager reference to this widget
        return True
        
    def update_all(self):
        """Update all visualization panels."""
        self.fractal_panel.update_visualization()
        self.consciousness_panel.update_visualization()
        self.network_panel.update_visualization()
        self.memory_panel.update_visualization()
        self.metrics_panel.update_metrics()
        
    def cleanup(self):
        """Clean up resources before closing."""
        self.fractal_panel.cleanup()
        self.consciousness_panel.cleanup()
        self.network_panel.cleanup()
        self.memory_panel.cleanup()
        self.metrics_panel.cleanup()
        self.conversation_panel.cleanup() 