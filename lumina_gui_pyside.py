#!/usr/bin/env python
"""
Lumina GUI PySide6 - Modern UI implementation for Lumina Neural Network System
Implements a dynamic tab interface with improved interactivity and animations
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import traceback
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lumina_pyside.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LuminaPySideGUI")

# PySide6 imports - modern Qt6-based UI framework
try:
    from PySide6.QtCore import Qt, Signal, Slot, QSize, QTimer, QPoint, QRectF, QPropertyAnimation, Property
    from PySide6.QtGui import QIcon, QFont, QPixmap, QPainter, QColor, QPen, QBrush, QPainterPath, QAction
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
        QLineEdit, QPushButton, QLabel, QTabWidget, QGridLayout, QScrollArea,
        QFrame, QSplitter, QProgressBar, QComboBox, QFileDialog, QDialog, 
        QMessageBox, QSlider, QCheckBox, QGroupBox, QSizePolicy, QSpacerItem,
        QStackedWidget
    )
    logger.info("Successfully imported PySide6")
except ImportError as e:
    logger.error(f"Failed to import PySide6: {e}")
    print(f"Error: {e}")
    print("Please install PySide6 with: pip install PySide6")
    sys.exit(1)

# Import system components
try:
    from minimal_central import MinimalCentralNode
    try:
        from central_node import CentralNode, BaseComponent  
        logger.info("Successfully imported CentralNode")
    except ImportError as e:
        logger.error(f"Failed to import CentralNode: {e}")
        logger.info("Will use fallback minimal central node")
        CentralNode = None
except ImportError as e:
    logger.error(f"Failed to import required module: {e}")
    # Will create fallback classes later

# Load environment variables
try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    logger.warning("python-dotenv not installed. Environment variables must be set manually.")

# Import our custom UI components
try:
    from src.ui.components.GlyphInterfacePanel import GlyphInterfacePanel
    logger.info("Successfully imported GlyphInterfacePanel")
except ImportError as e:
    logger.error(f"Failed to import GlyphInterfacePanel: {e}")
    # Continue anyway, we'll use placeholder if needed

try:
    from src.ui.components.NetworkVisualizationPanelPySide6 import NetworkVisualizationPanelPySide6
    logger.info("Successfully imported NetworkVisualizationPanelPySide6")
except ImportError as e:
    logger.error(f"Failed to import NetworkVisualizationPanelPySide6: {e}")
    # Continue anyway, we'll use placeholder if needed

class ControlButton(QPushButton):
    """Custom button for the control panel with hover effects"""
    
    def __init__(self, text, icon_name=None, parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(50)
        self.setIconSize(QSize(24, 24))
        
        if icon_name:
            icon_path = f"icons/{icon_name}.png"
            if os.path.exists(icon_path):
                self.setIcon(QIcon(icon_path))
            else:
                # Use a text-based icon prefix if icon file doesn't exist
                prefix = icon_name[0].upper() if icon_name else "•"
                self.setText(f"{prefix} {text}")
                logger.debug(f"Icon not found: {icon_path}")
            
        self.setStyleSheet("""
            QPushButton {
                background-color: #1E2C3A;
                color: #ECF0F1;
                border: none;
                border-radius: 5px;
                padding: 10px;
                text-align: left;
                margin: 3px;
            }
            QPushButton:hover {
                background-color: #2C3E50;
            }
            QPushButton:pressed {
                background-color: #34495E;
            }
            QPushButton:checked {
                background-color: #2980B9;
                border-left: 4px solid #3498DB;
            }
        """)
        self.setCheckable(True)

class ControlPanel(QWidget):
    """Left control panel with navigation buttons that control the center panel content"""
    
    # Define signals
    button_clicked = Signal(str)  # Emits the ID of the clicked button
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 20, 10, 20)
        layout.setSpacing(10)
        
        # App title
        title_label = QLabel("LUMINA")
        title_label.setStyleSheet("color: #3498DB; font-size: 24px; font-weight: bold;")
        layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Neural Network System")
        subtitle_label.setStyleSheet("color: #7F8C8D; font-size: 12px;")
        layout.addWidget(subtitle_label)
        
        layout.addSpacing(20)
        
        # Navigation buttons
        self.buttons = {}
        
        button_data = [
            ("chat", "Chat", "chat"),
            ("profile", "Profile", "user"),
            ("settings", "Settings", "settings"),
            ("memory", "Memory", "brain"),
            ("model", "Neural Model", "model"),
            ("llm", "LLM Settings", "ai"),
            ("node", "Neural Nodes", "node"),
            ("glyph", "Glyphs", "symbol")
        ]
        
        for btn_id, btn_text, btn_icon in button_data:
            button = ControlButton(btn_text, btn_icon)
            button.clicked.connect(lambda checked, b_id=btn_id: self.on_button_clicked(b_id))
            layout.addWidget(button)
            self.buttons[btn_id] = button
        
        # Set chat button as initially selected
        self.buttons["chat"].setChecked(True)
        
        # Add spacer at the bottom
        layout.addStretch()
        
        # Status indicator
        status_layout = QHBoxLayout()
        status_indicator = QLabel()
        status_indicator.setFixedSize(10, 10)
        status_indicator.setStyleSheet("background-color: #2ECC71; border-radius: 5px;")
        
        status_text = QLabel("System Active")
        status_text.setStyleSheet("color: #7F8C8D; font-size: 12px;")
        
        status_layout.addWidget(status_indicator)
        status_layout.addWidget(status_text)
        status_layout.addStretch()
        
        layout.addLayout(status_layout)
        
    def on_button_clicked(self, button_id):
        # Uncheck all buttons except the clicked one
        for btn_id, button in self.buttons.items():
            if btn_id != button_id:
                button.setChecked(False)
        
        # Emit signal with button ID
        self.button_clicked.emit(button_id)

class ChatPanel(QWidget):
    """Chat panel for communicating with Lumina"""
    
    message_sent = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Chat history
        self.chat_area = QScrollArea()
        self.chat_area.setWidgetResizable(True)
        self.chat_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_area.setStyleSheet("background-color: #121A24; border: none;")
        
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setSpacing(10)
        self.chat_container.setLayout(self.chat_layout)
        
        self.chat_area.setWidget(self.chat_container)
        layout.addWidget(self.chat_area, 1)
        
        # Input area
        input_widget = QWidget()
        input_layout = QHBoxLayout(input_widget)
        input_layout.setContentsMargins(10, 10, 10, 10)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: #1E2C3A;
                color: #ECF0F1;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
        """)
        self.input_field.returnPressed.connect(self.send_message)
        
        send_button = QPushButton("Send")
        send_button.setStyleSheet("""
            QPushButton {
                background-color: #2980B9;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #3498DB;
            }
            QPushButton:pressed {
                background-color: #1C587F;
            }
        """)
        send_button.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.input_field, 1)
        input_layout.addWidget(send_button)
        
        layout.addWidget(input_widget)
        
    def add_message(self, is_user, text):
        """Add a new message to the chat"""
        message_frame = QFrame()
        message_frame.setObjectName("userMessage" if is_user else "luminaMessage")
        message_frame.setStyleSheet("""
            #userMessage {
                background-color: #2C3E50;
                border-radius: 10px;
                margin: 5px;
            }
            #luminaMessage {
                background-color: #1E2C3A;
                border-radius: 10px;
                margin: 5px;
            }
        """)
        
        message_layout = QVBoxLayout(message_frame)
        message_layout.setContentsMargins(10, 10, 10, 10)
        
        name_label = QLabel("You" if is_user else "Lumina")
        name_label.setStyleSheet("color: #3498DB; font-weight: bold;")
        
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        text_label.setStyleSheet("color: #ECF0F1;")
        
        message_layout.addWidget(name_label)
        message_layout.addWidget(text_label)
        
        self.chat_layout.addWidget(message_frame)
        
        # Scroll to bottom
        self.chat_area.verticalScrollBar().setValue(
            self.chat_area.verticalScrollBar().maximum()
        )
        
    def send_message(self):
        """Send the current message"""
        text = self.input_field.text().strip()
        if not text:
            return
            
        # Add user message to chat
        self.add_message(True, text)
        self.input_field.clear()
        
        # Emit signal with message text
        self.message_sent.emit(text)

class PlaceholderPanel(QWidget):
    """Placeholder panel for unimplemented features"""
    
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        icon_label = QLabel("🔧")
        icon_label.setStyleSheet("font-size: 48px;")
        
        title_label = QLabel(f"{self.title} Panel")
        title_label.setStyleSheet("color: #ECF0F1; font-size: 24px; font-weight: bold;")
        
        desc_label = QLabel("This feature is under development")
        desc_label.setStyleSheet("color: #7F8C8D; font-size: 16px;")
        
        layout.addWidget(icon_label, 0, Qt.AlignCenter)
        layout.addWidget(title_label, 0, Qt.AlignCenter)
        layout.addWidget(desc_label, 0, Qt.AlignCenter)

class NeuralNetworkVis(QWidget):
    """Interactive neural network visualization panel"""
    
    node_clicked = Signal(str)  # Emits node ID when a node is clicked
    
    def __init__(self, parent=None, central_node=None):
        super().__init__(parent)
        self.central_node = central_node
        self.nodes = {}  # Dictionary of nodes: {id: {pos: QPoint, type: str, connections: [ids]}}
        self.highlighted_node = None
        self.animation_step = 0
        self.node_activities = {}  # Activity level of each node (0.0-1.0)
        self.error_nodes = set()  # Nodes with errors
        self.diagnostics_mode = False  # Toggle for showing diagnostics
        self.initUI()
        
    def initUI(self):
        self.setMinimumSize(400, 300)
        
        # Create a timer for animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(50)  # Update every 50ms
        
        if self.central_node:
            self.create_network_from_central_node()
        else:
            self.create_default_network()
            
    def create_default_network(self):
        """Create a default network when no central node is available"""
        # Create some sample nodes
        self.nodes = {
            "input1": {"pos": QPoint(100, 100), "type": "input", "connections": ["hidden1", "hidden2"]},
            "input2": {"pos": QPoint(100, 200), "type": "input", "connections": ["hidden1", "hidden3"]},
            "input3": {"pos": QPoint(100, 300), "type": "input", "connections": ["hidden2", "hidden3"]},
            "hidden1": {"pos": QPoint(250, 150), "type": "hidden", "connections": ["output1"]},
            "hidden2": {"pos": QPoint(250, 250), "type": "hidden", "connections": ["output1", "output2"]},
            "hidden3": {"pos": QPoint(250, 350), "type": "hidden", "connections": ["output2"]},
            "output1": {"pos": QPoint(400, 200), "type": "output", "connections": []},
            "output2": {"pos": QPoint(400, 300), "type": "output", "connections": []}
        }
    
    def create_network_from_central_node(self):
        """Create a network visualization from the central node's components"""
        if not self.central_node:
            self.create_default_network()
            return
            
        try:
            # Extract component information from central node
            if hasattr(self.central_node, 'get_component_dependencies'):
                components = self.central_node.get_component_dependencies()
                
                if isinstance(components, dict) and 'components' in components:
                    component_list = components['components']
                    
                    # Create a basic visualization based on components
                    self.nodes = {}
                    
                    # Create central node
                    central_pos = QPoint(300, 200)
                    self.nodes["central"] = {"pos": central_pos, "type": "central", "connections": []}
                    
                    # Create component nodes around the central node
                    num_components = len(component_list)
                    radius = 150
                    
                    for i, component in enumerate(component_list):
                        # Calculate position in a circle around central node
                        angle = (i / num_components) * 6.28  # 2*pi radians
                        x = central_pos.x() + int(radius * math.cos(angle))
                        y = central_pos.y() + int(radius * math.sin(angle))
                        
                        # Create node
                        node_id = f"comp{i}"
                        self.nodes[node_id] = {
                            "pos": QPoint(x, y),
                            "type": self.get_node_type(component),
                            "connections": [],
                            "name": component
                        }
                        
                        # Connect to central node
                        self.nodes["central"]["connections"].append(node_id)
                    
                    logger.info(f"Created network visualization with {num_components} components")
                    return
            
            # Fallback if we couldn't get component information
            self.create_default_network()
            
        except Exception as e:
            logger.error(f"Error creating network from central node: {e}")
            self.create_default_network()
            
    def get_node_type(self, node_name):
        """Determine node type based on name"""
        node_name = node_name.lower()
        
        if "input" in node_name or "sensor" in node_name:
            return "input"
        elif "output" in node_name or "response" in node_name:
            return "output"
        elif "process" in node_name or "compute" in node_name:
            return "process"
        elif "memory" in node_name or "storage" in node_name:
            return "memory"
        else:
            return "hidden"

    def update_animation(self):
        """Update animation step for flowing connections"""
        self.animation_step = (self.animation_step + 1) % 20
        self.update()  # Trigger repaint
        
    def mousePressEvent(self, event):
        """Handle mouse clicks on nodes"""
        for node_id, node_data in self.nodes.items():
            # Check if click position is within node
            node_rect = QRectF(node_data["pos"].x() - 15, node_data["pos"].y() - 15, 30, 30)
            if node_rect.contains(event.position().toPoint()):
                self.highlighted_node = node_id
                self.node_clicked.emit(node_id)
                self.update()
                break
    
    def toggle_diagnostics(self):
        """Toggle the diagnostics mode"""
        self.diagnostics_mode = not self.diagnostics_mode
        self.update()
        
    def highlight_error_node(self, node_id):
        """Mark a node as having an error"""
        if node_id in self.nodes:
            self.error_nodes.add(node_id)
            self.update()
            
    def set_node_activity(self, node_id, activity):
        """Set the activity level for a node (0.0-1.0)"""
        if node_id in self.nodes:
            self.node_activities[node_id] = min(1.0, max(0.0, activity))
            self.update()
            
    def paintEvent(self, event):
        """Draw the neural network visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw connections first (so they appear behind nodes)
        for node_id, node_data in self.nodes.items():
            start_pos = node_data["pos"]
            
            for conn_id in node_data["connections"]:
                if conn_id in self.nodes:
                    end_pos = self.nodes[conn_id]["pos"]
                    
                    # Adjust color based on source node's activity
                    if self.diagnostics_mode and node_id in self.node_activities:
                        activity = self.node_activities[node_id]
                        color = QColor(
                            min(255, int(52 + 203 * activity)),  # More red for higher activity
                            min(255, int(152 + 50 * activity)),
                            219, 
                            150
                        )
                    else:
                        color = QColor(52, 152, 219, 150)  # Default blue
                        
                    # Use red for connections from error nodes
                    if node_id in self.error_nodes:
                        color = QColor(231, 76, 60, 180)  # Red for error
                        
                    pen = QPen(color)
                    pen.setWidth(2)
                    painter.setPen(pen)
                    
                    # Draw the main connection line
                    painter.drawLine(start_pos, end_pos)
                    
                    # Draw animated flow particles
                    flow_pen = QPen(QColor(46, 204, 113))
                    flow_pen.setWidth(4)
                    painter.setPen(flow_pen)
                    
                    # Only show flow animation for active or non-diagnostic nodes
                    if not self.diagnostics_mode or node_id in self.node_activities and self.node_activities[node_id] > 0.2:
                        # Calculate position along the line based on animation step
                        dx = end_pos.x() - start_pos.x()
                        dy = end_pos.y() - start_pos.y()
                        for i in range(3):  # 3 particles per connection
                            particle_pos = (i * 5 + self.animation_step) % 15  # 0-14 position value
                            particle_pos = particle_pos / 15.0  # Convert to 0.0-1.0 range
                            
                            x = start_pos.x() + dx * particle_pos
                            y = start_pos.y() + dy * particle_pos
                            
                            painter.drawPoint(int(x), int(y))
        
        # Draw nodes
        for node_id, node_data in self.nodes.items():
            pos = node_data["pos"]
            node_type = node_data["type"]
            
            # Choose color based on node type
            if node_type == "input":
                color = QColor(41, 128, 185)  # Blue
            elif node_type == "hidden":
                color = QColor(142, 68, 173)  # Purple
            elif node_type == "output":
                color = QColor(39, 174, 96)  # Green
            elif node_type == "central":
                color = QColor(230, 126, 34)  # Orange for central node
            else:
                color = QColor(149, 165, 166)  # Gray
                
            # In diagnostic mode, adjust color based on activity
            if self.diagnostics_mode and node_id in self.node_activities:
                activity = self.node_activities[node_id]
                # Blend with white based on activity
                color = QColor(
                    min(255, int(color.red() + (255 - color.red()) * activity)),
                    min(255, int(color.green() + (255 - color.green()) * activity)),
                    min(255, int(color.blue() + (255 - color.blue()) * activity))
                )
                
            # Highlight error nodes with red border
            if node_id in self.error_nodes:
                error_pen = QPen(QColor(231, 76, 60), 3)  # Red for error
                painter.setPen(error_pen)
                painter.setBrush(QBrush(color.lighter(120)))
                painter.drawEllipse(pos, 17, 17)
                
            # Highlight selected node
            if node_id == self.highlighted_node:
                highlight = QColor(243, 156, 18)  # Orange
                painter.setPen(QPen(highlight, 2))
                painter.setBrush(QBrush(highlight.lighter(120)))
                painter.drawEllipse(pos, 18, 18)
            
            # Draw node
            painter.setPen(QPen(color.darker(120), 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(pos, 15, 15)
            
            # Draw node label
            painter.setPen(QPen(QColor(236, 240, 241), 1))
            
            # Use the name if available, otherwise use node_id
            label = node_data.get("name", node_id)
            label = label.split("_")[0][:4]  # Use first part, limit to 4 chars
            
            label_width = painter.fontMetrics().horizontalAdvance(label)
            painter.drawText(pos.x() - label_width // 2, pos.y() + 5, label)
            
            # Draw activity indicator in diagnostic mode
            if self.diagnostics_mode and node_id in self.node_activities:
                activity = self.node_activities[node_id]
                # Draw activity as small text below the node
                activity_text = f"{activity:.1f}"
                painter.drawText(pos.x() - 10, pos.y() + 25, activity_text)

class MetricsPanel(QWidget):
    """Right metrics panel with neural network visualization and controls"""
    
    # Add signals
    weight_changed = Signal(float)  # Emits the new weight value (0.0-1.0)
    breathe_clicked = Signal()
    resonance_clicked = Signal()
    echo_clicked = Signal()
    mirror_clicked = Signal()
    diagnose_clicked = Signal()  # New signal for diagnostics
    
    def __init__(self, parent=None, central_node=None):
        super().__init__(parent)
        self.central_node = central_node
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title area with diagnostics toggle
        title_layout = QHBoxLayout()
        title_label = QLabel("Neural Network")
        title_label.setStyleSheet("color: #ECF0F1; font-size: 18px; font-weight: bold;")
        
        self.diagnose_btn = QPushButton("Diagnose")
        self.diagnose_btn.setCheckable(True)
        self.diagnose_btn.setStyleSheet("""
            QPushButton {
                background-color: #34495E;
                color: #ECF0F1;
                border-radius: 5px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:pressed, QPushButton:checked {
                background-color: #E74C3C;
                color: white;
            }
        """)
        
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(self.diagnose_btn)
        
        layout.addLayout(title_layout)
        
        # Neural network visualization
        self.nn_vis = NeuralNetworkVis(central_node=self.central_node)
        self.nn_vis.node_clicked.connect(self.on_node_clicked)
        layout.addWidget(self.nn_vis, 1)
        
        # System diagnostics - initially hidden
        self.diagnostics_group = QGroupBox("System Diagnostics")
        self.diagnostics_group.setVisible(False)
        self.diagnostics_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #E74C3C;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                color: #ECF0F1;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #E74C3C;
            }
        """)
        
        diag_layout = QVBoxLayout()
        
        # Status indicators
        self.nn_status = QLabel("Neural Network: Unknown")
        self.nn_status.setStyleSheet("color: #ECF0F1;")
        
        self.mem_status = QLabel("Memory System: Unknown")
        self.mem_status.setStyleSheet("color: #ECF0F1;")
        
        self.llm_status = QLabel("LLM Integration: Unknown")
        self.llm_status.setStyleSheet("color: #ECF0F1;")
        
        # Error log
        self.error_log = QTextEdit()
        self.error_log.setReadOnly(True)
        self.error_log.setMaximumHeight(80)
        self.error_log.setStyleSheet("""
            QTextEdit {
                background-color: #121A24;
                color: #E74C3C;
                border: 1px solid #34495E;
                border-radius: 3px;
                font-family: monospace;
            }
        """)
        
        # Add to diagnostic layout
        diag_layout.addWidget(self.nn_status)
        diag_layout.addWidget(self.mem_status)
        diag_layout.addWidget(self.llm_status)
        diag_layout.addWidget(QLabel("Latest Errors:"))
        diag_layout.addWidget(self.error_log)
        
        self.diagnostics_group.setLayout(diag_layout)
        layout.addWidget(self.diagnostics_group)
        
        # Process control buttons
        process_group = QGroupBox("Process Controls")
        process_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #34495E;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                color: #ECF0F1;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        process_layout = QGridLayout()
        
        # Create process buttons with hover effects
        self.breathe_btn = QPushButton("Breathe")
        self.resonance_btn = QPushButton("Resonance")
        self.echo_btn = QPushButton("Echo")
        self.mirror_btn = QPushButton("Mirror")
        
        # Apply styling to all buttons
        for btn in [self.breathe_btn, self.resonance_btn, self.echo_btn, self.mirror_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #34495E;
                    color: #ECF0F1;
                    border-radius: 5px;
                    padding: 8px;
                }
                QPushButton:hover {
                    background-color: #2980B9;
                }
                QPushButton:pressed {
                    background-color: #1C587F;
                }
            """)
        
        process_layout.addWidget(self.breathe_btn, 0, 0)
        process_layout.addWidget(self.resonance_btn, 0, 1)
        process_layout.addWidget(self.echo_btn, 1, 0)
        process_layout.addWidget(self.mirror_btn, 1, 1)
        
        process_group.setLayout(process_layout)
        layout.addWidget(process_group)
        
        # NN/LLM Weight slider
        weight_group = QGroupBox("NN/LLM Weight")
        weight_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #34495E;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                color: #ECF0F1;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        weight_layout = QVBoxLayout()
        
        # Weight slider with labels
        slider_layout = QHBoxLayout()
        nn_label = QLabel("NN")
        nn_label.setStyleSheet("color: #ECF0F1;")
        
        self.weight_slider = QSlider(Qt.Horizontal)
        self.weight_slider.setRange(0, 100)
        self.weight_slider.setValue(50)  # Default to 50/50 balance
        self.weight_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #455A64;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2980B9, stop:1 #16A085);
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #ECF0F1;
                border: 1px solid #607D8B;
                width: 18px;
                margin: -8px 0;
                border-radius: 9px;
            }
        """)
        
        llm_label = QLabel("LLM")
        llm_label.setStyleSheet("color: #ECF0F1;")
        
        slider_layout.addWidget(nn_label)
        slider_layout.addWidget(self.weight_slider, 1)
        slider_layout.addWidget(llm_label)
        
        # Weight display
        self.weight_display = QLabel("NN 50% / LLM 50%")
        self.weight_display.setAlignment(Qt.AlignCenter)
        self.weight_display.setStyleSheet("color: #ECF0F1;")
        
        # Preset buttons
        preset_layout = QHBoxLayout()
        
        self.nn_only_btn = QPushButton("NN Only")
        self.balanced_btn = QPushButton("Balanced")
        self.llm_only_btn = QPushButton("LLM Only")
        
        for btn in [self.nn_only_btn, self.balanced_btn, self.llm_only_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #34495E;
                    color: #ECF0F1;
                    border-radius: 5px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #2980B9;
                }
                QPushButton:pressed {
                    background-color: #1C587F;
                }
            """)
        
        preset_layout.addWidget(self.nn_only_btn)
        preset_layout.addWidget(self.balanced_btn)
        preset_layout.addWidget(self.llm_only_btn)
        
        # Add layouts to weight group
        weight_layout.addLayout(slider_layout)
        weight_layout.addWidget(self.weight_display)
        weight_layout.addLayout(preset_layout)
        
        weight_group.setLayout(weight_layout)
        layout.addWidget(weight_group)
        
        # Connect signals
        self.weight_slider.valueChanged.connect(self.on_weight_changed)
        self.nn_only_btn.clicked.connect(lambda: self.weight_slider.setValue(0))
        self.balanced_btn.clicked.connect(lambda: self.weight_slider.setValue(50))
        self.llm_only_btn.clicked.connect(lambda: self.weight_slider.setValue(100))
        
        # Connect process button signals
        self.breathe_btn.clicked.connect(lambda: self.breathe_clicked.emit())
        self.resonance_btn.clicked.connect(lambda: self.resonance_clicked.emit())
        self.echo_btn.clicked.connect(lambda: self.echo_clicked.emit())
        self.mirror_btn.clicked.connect(lambda: self.mirror_clicked.emit())
        
        # Connect diagnose button
        self.diagnose_btn.clicked.connect(self.toggle_diagnostics)
        
    def toggle_diagnostics(self):
        """Toggle the diagnostics display"""
        is_visible = self.diagnostics_group.isVisible()
        self.diagnostics_group.setVisible(not is_visible)
        self.nn_vis.toggle_diagnostics()
        
        # Emit signal when entering diagnostics mode
        if not is_visible:
            self.diagnose_clicked.emit()
            
    def update_diagnostics(self, nn_status="Unknown", mem_status="Unknown", 
                          llm_status="Unknown", error_log=None):
        """Update the diagnostics display with current system status"""
        # Update status labels
        self.nn_status.setText(f"Neural Network: {nn_status}")
        self.mem_status.setText(f"Memory System: {mem_status}")
        self.llm_status.setText(f"LLM Integration: {llm_status}")
        
        # Set color based on status
        for label, status in [(self.nn_status, nn_status), 
                             (self.mem_status, mem_status), 
                             (self.llm_status, llm_status)]:
            if "Error" in status or "Failed" in status:
                label.setStyleSheet("color: #E74C3C;")  # Red for error
            elif "Warning" in status or "Limited" in status:
                label.setStyleSheet("color: #F39C12;")  # Orange for warning
            elif "OK" in status or "Active" in status:
                label.setStyleSheet("color: #2ECC71;")  # Green for OK
            else:
                label.setStyleSheet("color: #ECF0F1;")  # Default color
        
        # Update error log
        if error_log:
            self.error_log.append(error_log)
        
    def on_weight_changed(self, value):
        """Update weight display when slider value changes"""
        nn_weight = 100 - value
        llm_weight = value
        self.weight_display.setText(f"NN {nn_weight}% / LLM {llm_weight}%")
        
        # Emit signal with normalized weight (0.0 - 1.0)
        self.weight_changed.emit(value / 100.0)
        
    def on_node_clicked(self, node_id):
        """Handle node click events from the visualization"""
        logger.info(f"Node clicked: {node_id}")
        # You could show node details or trigger activations here

class LuminaGUIPySide(QMainWindow):
    """Main window for the PySide6 Lumina GUI application"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lumina Neural Network System")
        self.setGeometry(100, 100, 1280, 720)  # 16:9 aspect ratio
        self.setMinimumSize(800, 450)
        
        # Dark theme
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0A1014;
                color: #ECF0F1;
            }
            QScrollBar:vertical {
                border: none;
                background: #121A24;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #2C3E50;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # Initialize system components
        self.init_system()
        
        # Set up the UI
        self.initUI()
        
    def init_system(self):
        """Initialize system components"""
        # Try to initialize central node
        try:
            if CentralNode:
                self.central_node = self.create_central_node()
                logger.info("Using CentralNode")
            else:
                self.central_node = MinimalCentralNode()
                logger.info("Using MinimalCentralNode")
        except Exception as e:
            logger.error(f"Failed to initialize central node: {e}")
            self.central_node = self.create_dummy_central_node()
            logger.info("Using dummy central node")
            
        # NN/LLM weight value (0.0 = NN only, 1.0 = LLM only)
        self.nn_llm_weight = 0.5  # Start at balanced
            
    def create_central_node(self):
        """Create and initialize the central node"""
        try:
            return CentralNode()
        except Exception as e:
            logger.error(f"Error creating CentralNode: {e}")
            return self.create_dummy_central_node()
            
    def create_dummy_central_node(self):
        """Create a dummy central node for testing"""
        class DummyNode:
            def __init__(self):
                self.id = "dummy_node"
                self.type = "dummy"
                self.components = {}
                self.vocabulary = ["dummy", "test", "example", "neural", "network"]
                
            def process_complete_flow(self, data):
                logger.info(f"Dummy node processing: {data}")
                return {
                    "response": f"Dummy response to: {data}",
                    "confidence": 0.7,
                    "processing_time": 0.1
                }
                
            def get_system_status(self):
                return {
                    "status": "OK",
                    "active_components": 5,
                    "total_components": 8,
                    "memory_size": 143
                }
                
            def get_component_dependencies(self):
                return {"components": ["dummy1", "dummy2", "dummy3"]}
                
        return DummyNode()
        
    def initUI(self):
        """Initialize the user interface"""
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main layout (horizontal split)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Left control panel (1:5 ratio)
        self.control_panel = ControlPanel()
        self.control_panel.setFixedWidth(200)  # approx 1/5 of a 1000px width
        self.control_panel.button_clicked.connect(self.on_control_button_clicked)
        
        # Center content area (dynamic, changes based on left panel selection)
        self.center_panel = QStackedWidget()
        
        # Chat panel
        self.chat_panel = ChatPanel()
        self.chat_panel.message_sent.connect(self.process_message)
        self.center_panel.addWidget(self.chat_panel)
        
        # Other panels
        self.profile_panel = PlaceholderPanel("Profile")
        self.settings_panel = PlaceholderPanel("Settings")
        self.memory_panel = PlaceholderPanel("Memory")
        
        # Use the NetworkVisualizationPanelPySide6 for model visualization
        try:
            self.model_panel = NetworkVisualizationPanelPySide6()
            logger.info("Using NetworkVisualizationPanelPySide6 implementation")
        except Exception as e:
            logger.error(f"Failed to create NetworkVisualizationPanelPySide6: {e}")
            self.model_panel = PlaceholderPanel("Neural Model")
            logger.info("Using placeholder for NetworkVisualizationPanelPySide6")
            
        self.llm_panel = PlaceholderPanel("LLM Settings")
        self.node_panel = PlaceholderPanel("Neural Node")
        
        # Use the proper GlyphInterfacePanel instead of placeholder
        try:
            self.glyph_panel = GlyphInterfacePanel()
            logger.info("Using GlyphInterfacePanel implementation")
        except Exception as e:
            logger.error(f"Failed to create GlyphInterfacePanel: {e}")
            self.glyph_panel = PlaceholderPanel("Glyph")
            logger.info("Using placeholder for GlyphInterfacePanel")
        
        self.center_panel.addWidget(self.profile_panel)
        self.center_panel.addWidget(self.settings_panel)
        self.center_panel.addWidget(self.memory_panel)
        self.center_panel.addWidget(self.model_panel)
        self.center_panel.addWidget(self.llm_panel)
        self.center_panel.addWidget(self.node_panel)
        self.center_panel.addWidget(self.glyph_panel)
        
        # Right metrics panel (1:3 ratio)
        self.metrics_panel = MetricsPanel(central_node=self.central_node)
        self.metrics_panel.setFixedWidth(320)
        
        # Add panels to main layout
        main_layout.addWidget(self.control_panel)
        main_layout.addWidget(self.center_panel, 1)  # Center panel takes most space
        main_layout.addWidget(self.metrics_panel)
        
        # Start with chat panel active
        self.on_control_button_clicked("chat")
        
        # Add system message to chat
        self.add_system_message("Lumina Neural Network System initialized. How can I help you today?")
        
        # Connect metrics panel signals
        self.metrics_panel.weight_changed.connect(self.on_weight_changed)
        self.metrics_panel.breathe_clicked.connect(self.on_breathe)
        self.metrics_panel.resonance_clicked.connect(self.on_resonance)
        self.metrics_panel.echo_clicked.connect(self.on_echo)
        self.metrics_panel.mirror_clicked.connect(self.on_mirror)
        self.metrics_panel.diagnose_clicked.connect(self.on_diagnose)
        
    def on_control_button_clicked(self, button_id):
        """Handle clicks on the control panel buttons"""
        # Map button IDs to panel indices
        panel_map = {
            "chat": 0,
            "profile": 1,
            "settings": 2,
            "memory": 3,
            "model": 4,
            "llm": 5,
            "node": 6,
            "glyph": 7
        }
        
        if button_id in panel_map:
            self.center_panel.setCurrentIndex(panel_map[button_id])
            
            # Special handling for glyph panel - update data from other agents if needed
            if button_id == "glyph" and hasattr(self.glyph_panel, "update_collaborator_data"):
                try:
                    # Get data from other systems
                    neural_data = self.get_neural_data_for_glyphs() if hasattr(self, "get_neural_data_for_glyphs") else None
                    knowledge_data = self.get_knowledge_data_for_glyphs() if hasattr(self, "get_knowledge_data_for_glyphs") else None
                    
                    # Update the glyph panel with data from collaborating agents
                    self.glyph_panel.update_collaborator_data(neural_data, knowledge_data)
                    logger.info("Updated glyph panel with collaborator data")
                except Exception as e:
                    logger.error(f"Error updating glyph panel: {e}")
        
    def add_system_message(self, text):
        """Add a system message to the chat"""
        self.chat_panel.add_message(False, text)
        
    def process_message(self, text):
        """Process a message from the user"""
        logger.info(f"Processing message: {text}")
        
        # Show typing indicator
        self.add_system_message("Thinking...")
        
        try:
            # Process with central node - use a timer to simulate processing time
            # and make the UI more responsive
            QTimer.singleShot(300, lambda: self._complete_response(text))
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.add_system_message(f"I encountered an error processing your request. Please try again.")
    
    def _complete_response(self, text):
        """Complete the response after a short delay"""
        try:
            # Process the text
            response = self.process_with_central_node(text)
            
            # Remove the "Thinking..." message
            self.chat_panel.chat_layout.itemAt(self.chat_panel.chat_layout.count()-1).widget().setParent(None)
            
            # Add the actual response
            self.add_system_message(response)
            
        except Exception as e:
            logger.error(f"Error completing response: {e}")
            self.add_system_message(f"I encountered an error processing your request. Please try again.")
    
    def process_with_central_node(self, text):
        """Process text with the central node"""
        try:
            # Get the neural network and LLM responses
            nn_response = self.get_neural_network_response(text)
            
            # If weight is fully on NN side, just return the NN response
            if self.nn_llm_weight <= 0.05:
                return nn_response
                
            # If we have an LLM component and weight is not on NN only
            llm_response = self.get_llm_response(text)
            
            # If weight is fully on LLM side, just return the LLM response
            if self.nn_llm_weight >= 0.95:
                return llm_response
                
            # Otherwise, blend the responses based on weight
            nn_part = (1 - self.nn_llm_weight)
            llm_part = self.nn_llm_weight
            
            # Improved blending logic
            # Get first sentence of each response for contribution
            def get_first_sentence(text):
                # Handle None or empty strings
                if not text:
                    return ""
                    
                # Find the first sentence end (., !, or ?)
                for i, char in enumerate(text):
                    if char in ['.', '!', '?'] and (i+1 == len(text) or text[i+1] == ' '):
                        return text[:i+1]
                # If no sentence end found, return the whole text or a portion
                return text[:min(100, len(text))]
            
            if nn_part > llm_part:
                # NN is the primary response with LLM contribution
                nn_first = get_first_sentence(nn_response)
                llm_contrib = get_first_sentence(llm_response)
                if llm_contrib:
                    return f"{nn_response}\n\nLLM insight: {llm_contrib}"
                else:
                    return nn_response
            else:
                # LLM is the primary response with NN contribution
                llm_first = get_first_sentence(llm_response)
                nn_contrib = get_first_sentence(nn_response)
                if nn_contrib:
                    return f"{llm_response}\n\nNN insight: {nn_contrib}"
                else:
                    return llm_response
                
        except Exception as e:
            logger.error(f"Error in central node processing: {e}")
            traceback.print_exc()
            return "I'm having trouble processing that right now. The system may need maintenance."
            
    def get_neural_network_response(self, text):
        """Get a response from the neural network"""
        try:
            if hasattr(self.central_node, 'process_complete_flow'):
                logger.info(f"Processing via neural network: {text[:50]}...")
                
                # Simulate processing delay
                QTimer.singleShot(200, lambda: self.update_network_visualization())
                
                result = self.central_node.process_complete_flow(text)
                
                # Detailed logging to diagnose issues
                logger.info(f"NN result type: {type(result)}")
                if isinstance(result, dict):
                    logger.info(f"NN result keys: {result.keys()}")
                
                if isinstance(result, dict) and 'response' in result:
                    response = result['response']
                    if response and isinstance(response, str) and len(response.strip()) > 0:
                        return response
                    else:
                        logger.warning("Empty or invalid response from neural network")
                        self.update_error_visualization()
                        return self._get_fallback_nn_response(text)
                
                # Handle non-dictionary or dictionary without 'response'
                if result:
                    return str(result)
                else:
                    logger.warning("Empty result from neural network")
                    self.update_error_visualization()
                    return self._get_fallback_nn_response(text)
                    
            logger.warning("Central node doesn't have process_complete_flow method")
            self.update_error_visualization()
            return self._get_fallback_nn_response(text)
            
        except Exception as e:
            logger.error(f"Neural network processing error: {e}")
            traceback.print_exc()
            self.update_error_visualization()
            return self._get_fallback_nn_response(text)
            
    def _get_fallback_nn_response(self, text):
        """Get a fallback neural network response when the main processing fails"""
        # Use a more varied set of fallback responses
        fallbacks = [
            "I'm analyzing your input through neural pathways, but encountering some resistance.",
            "My neural processing is currently limited. I'm working to improve this capability.",
            "I'm processing your request through alternate neural circuits.",
            f"I notice your message about '{text.split()[0:3]}...' but my neural system needs recalibration.",
            "My neural networks are currently operating at reduced capacity. I can still assist with basic requests."
        ]
        import random
        return random.choice(fallbacks)
            
    def get_llm_response(self, text):
        """Get a response from the LLM component"""
        # For now, return a more informative placeholder response
        responses = [
            f"Based on my understanding, '{text}' relates to neural network concepts. I would analyze this in terms of activation patterns and knowledge structures.",
            f"I understand you're interested in '{text}'. This concept connects to several neural pathways in my knowledge base.",
            f"Your query about '{text}' is interesting. I can offer several perspectives on this from my language model analysis.",
            f"From an LLM perspective, '{text}' has multiple semantic dimensions worth exploring. Let me elaborate on the key aspects."
        ]
        import random
        return random.choice(responses)

    def on_weight_changed(self, value):
        """Update the NN/LLM weight value"""
        self.nn_llm_weight = value
        logger.info(f"NN/LLM weight set to {value:.2f}")
        
    def on_breathe(self):
        """Handle breathe button click"""
        logger.info("Breathe process activated")
        self.add_system_message("Breath calibration initiated. Take a deep breath...")
        # Show animation or effect here
        
    def on_resonance(self):
        """Handle resonance button click"""
        logger.info("Resonance process activated")
        self.add_system_message("Resonance session started. Aligning neural pathways...")
        # Show resonance effect here
        
    def on_echo(self):
        """Handle echo button click"""
        logger.info("Echo process activated")
        self.add_system_message("Echo feedback engaged. Retrieving memory patterns...")
        # Show echo effect here
        
    def on_mirror(self):
        """Handle mirror button click"""
        logger.info("Mirror process activated")
        self.add_system_message("Mirror reflection mode activated. Analyzing patterns...")
        # Show mirror effect here

    def on_diagnose(self):
        """Run diagnostics on the system"""
        logger.info("Running system diagnostics")
        
        # Collect diagnostic information
        nn_status = "Limited - Processing errors detected"
        mem_status = "OK"
        llm_status = "Simulation mode - No API connection"
        
        # Error log
        error_log = f"[{datetime.now().strftime('%H:%M:%S')}] Neural network processing error detected"
        
        # Update the diagnostics panel
        self.metrics_panel.update_diagnostics(
            nn_status=nn_status,
            mem_status=mem_status,
            llm_status=llm_status,
            error_log=error_log
        )
        
        # Simulate some node activities and errors for visualization
        # In a real implementation, this would come from actual system state
        vis = self.metrics_panel.nn_vis
        
        # Clear previous diagnostics
        vis.error_nodes.clear()
        vis.node_activities.clear()
        
        # Set activity levels
        for node_id in vis.nodes:
            # Random activity levels for demo
            import random
            activity = random.random()
            vis.set_node_activity(node_id, activity)
            
            # Mark some nodes as having errors
            if activity < 0.3 and random.random() < 0.5:
                vis.highlight_error_node(node_id)
                
        # Update visualization
        vis.update()
        
        # Add diagnostic message to chat
        self.add_system_message("System diagnostics complete. Neural processing is limited.")

    def update_network_visualization(self):
        """Update the network visualization with activity"""
        try:
            # Only update if diagnostics mode is active
            if not self.metrics_panel.diagnostics_group.isVisible():
                return
                
            vis = self.metrics_panel.nn_vis
            
            # Set activity levels for nodes - in real implementation this would 
            # come from actual NN activation levels
            import random
            for node_id in vis.nodes:
                vis.set_node_activity(node_id, random.random())
                
            vis.update()
            
        except Exception as e:
            logger.error(f"Error updating visualization: {e}")
            
    def update_error_visualization(self):
        """Update visualization to show error state"""
        try:
            vis = self.metrics_panel.nn_vis
            
            # Mark a random node as having an error
            if vis.nodes:
                import random
                error_node = random.choice(list(vis.nodes.keys()))
                vis.highlight_error_node(error_node)
                
                # Log this in the diagnostics panel if visible
                if self.metrics_panel.diagnostics_group.isVisible():
                    error_time = datetime.now().strftime("%H:%M:%S")
                    self.metrics_panel.update_diagnostics(
                        nn_status="Error - Processing failure",
                        error_log=f"[{error_time}] Error in node: {error_node}"
                    )
                
        except Exception as e:
            logger.error(f"Error updating error visualization: {e}")

    def get_neural_data_for_glyphs(self):
        """
        Get neural network data from agent 1 for the glyph panel
        This simulates getting data from another AI agent working on the neural part
        """
        # This would normally come from the neural network training agent
        mock_neural_data = {
            "network_status": "online",
            "available_models": ["basic_glyph_encoder", "symbol_recognition", "pattern_generator"],
            "active_connections": [
                {"glyph_id": 0, "neuron_path": "layer3.node42", "activation": 0.78},
                {"glyph_id": 2, "neuron_path": "layer2.node17", "activation": 0.63},
                {"glyph_id": 5, "neuron_path": "layer4.node91", "activation": 0.82}
            ]
        }
        logger.debug("Generated mock neural data for glyphs")
        return mock_neural_data
    
    def get_knowledge_data_for_glyphs(self):
        """
        Get knowledge/database data from agent 2 for the glyph panel
        This simulates getting data from another AI agent working on the knowledge part
        """
        # This would normally come from the knowledge/database agent
        mock_knowledge_data = {
            "symbol_meanings": {
                "circle": ["unity", "wholeness", "completion"],
                "triangle": ["transformation", "ascension", "balance"],
                "square": ["stability", "foundation", "structure"],
                "cross": ["intersection", "decision point", "connection"],
                "spiral": ["growth", "evolution", "journey"]
            },
            "related_concepts": {
                "circle": ["zero", "infinity", "cycle"],
                "spiral": ["fibonacci", "golden ratio", "fractal"]
            },
            "recent_activations": [
                {"glyph": "triangle", "timestamp": "2023-07-15T14:22:31", "context": "transformation process"},
                {"glyph": "spiral", "timestamp": "2023-07-15T15:17:42", "context": "evolutionary algorithm"}
            ]
        }
        logger.debug("Generated mock knowledge data for glyphs")
        return mock_knowledge_data

# Run the application when the script is executed directly
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LuminaGUIPySide()
    window.show()
    sys.exit(app.exec()) 