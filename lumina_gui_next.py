#!/usr/bin/env python
"""
Lumina GUI Next - An advanced graphical interface for the Lumina neural network system
Implements the 16:9 aspect ratio with 1:5 left control panel, center fill, and 1:3 right metrics panel
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import traceback

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                          QHBoxLayout, QTextEdit, QLineEdit, QPushButton, 
                          QLabel, QTabWidget, QGridLayout, QScrollArea,
                          QFrame, QSplitter, QProgressBar, QComboBox, QFileDialog,
                          QDialog, QMessageBox, QSlider, QCheckBox, QGroupBox,
                          QSizePolicy, QSpacerItem)
from PyQt5.QtGui import QIcon, QFont, QPixmap, QPainter, QColor, QPen, QBrush, QPainterPath
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize, QThread, pyqtSlot, QRectF, QPoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lumina_next.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LuminaNextGUI")

# Import system components
try:
    from chat_memory import ChatMemory, MemoryEntry
    from semantic_director import SemanticDirector
    from minimal_central import MinimalCentralNode
    try:
        from central_node import CentralNode, BaseComponent  # Add import for CentralNode
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

class ChatMessage(QFrame):
    """An individual chat message widget"""
    
    def __init__(self, is_user, text, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.text = text
        self.initUI()
        
    def initUI(self):
        self.setObjectName("userMessage" if self.is_user else "luminaMessage")
        self.setStyleSheet("""
            #userMessage {
                background-color: #2C3E50;
                border-radius: 10px;
                margin: 5px;
                padding: 10px;
            }
            #luminaMessage {
                background-color: #1E2C3A;
                border-radius: 10px;
                margin: 5px;
                padding: 10px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Name label
        name_label = QLabel("You" if self.is_user else "Lumina")
        name_label.setStyleSheet("color: #3498DB; font-weight: bold;")
        layout.addWidget(name_label)
        
        # Message text
        message_label = QLabel(self.text)
        message_label.setWordWrap(True)
        message_label.setStyleSheet("color: #ECF0F1;")
        layout.addWidget(message_label)
        
        self.setLayout(layout)

class ChatArea(QWidget):
    """Central chat area for conversations with Lumina"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Chat history scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("background-color: #121A24; border: none;")
        
        # Container for messages
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setSpacing(10)
        self.chat_container.setLayout(self.chat_layout)
        
        self.scroll_area.setWidget(self.chat_container)
        layout.addWidget(self.scroll_area, 1)
        
        # Input area
        input_container = QWidget()
        input_layout = QHBoxLayout(input_container)
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
        
        layout.addWidget(input_container)
        self.setLayout(layout)
        
        # Connect enter key to send message
        self.input_field.returnPressed.connect(self.send_message)
        
    def add_message(self, is_user, text):
        """Add a new message to the chat area"""
        message = ChatMessage(is_user, text)
        self.chat_layout.addWidget(message)
        # Scroll to bottom
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )
        
    def send_message(self):
        """Send the current message"""
        text = self.input_field.text().strip()
        if not text:
            return
            
        # Add user message to chat
        self.add_message(True, text)
        self.input_field.clear()
        
        # Signal to parent that a message was sent - will be processed by parent
        parent = self.parent()
        if parent and hasattr(parent, 'process_message'):
            parent.process_message(text)

class ControlPanel(QWidget):
    """Left control panel (1:5 ratio) containing navigation and controls"""
    
    # Define signals for control buttons
    profile_clicked = pyqtSignal()
    favorites_clicked = pyqtSignal()
    settings_clicked = pyqtSignal()
    memory_clicked = pyqtSignal()
    model_clicked = pyqtSignal()
    llm_clicked = pyqtSignal()
    node_clicked = pyqtSignal()
    glyph_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        self.setMinimumWidth(180)  # Minimum width to ensure readability
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Style for control buttons
        button_style = """
            QPushButton {
                background-color: #1E2C3A;
                color: #ECF0F1;
                border-radius: 5px;
                padding: 12px;
                text-align: left;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2C3E50;
            }
            QPushButton:pressed {
                background-color: #0F1A26;
            }
        """
        
        # Profile button
        self.profile_btn = QPushButton("👤 Profile")
        self.profile_btn.setStyleSheet(button_style)
        self.profile_btn.clicked.connect(self.profile_clicked.emit)
        layout.addWidget(self.profile_btn)
        
        # Favorites button
        self.favorites_btn = QPushButton("⭐ Favorites")
        self.favorites_btn.setStyleSheet(button_style)
        self.favorites_btn.clicked.connect(self.favorites_clicked.emit)
        layout.addWidget(self.favorites_btn)
        
        # Settings button
        self.settings_btn = QPushButton("⚙️ Settings")
        self.settings_btn.setStyleSheet(button_style)
        self.settings_btn.clicked.connect(self.settings_clicked.emit)
        layout.addWidget(self.settings_btn)
        
        # Memory scroll button
        self.memory_btn = QPushButton("📜 Memory")
        self.memory_btn.setStyleSheet(button_style)
        self.memory_btn.clicked.connect(self.memory_clicked.emit)
        layout.addWidget(self.memory_btn)
        
        # Model control button
        self.model_btn = QPushButton("🧠 Model")
        self.model_btn.setStyleSheet(button_style)
        self.model_btn.clicked.connect(self.model_clicked.emit)
        layout.addWidget(self.model_btn)
        
        # LLM settings button
        self.llm_btn = QPushButton("🔄 LLM Settings")
        self.llm_btn.setStyleSheet(button_style)
        self.llm_btn.clicked.connect(self.llm_clicked.emit)
        layout.addWidget(self.llm_btn)
        
        # Neural node activation button
        self.node_btn = QPushButton("🌐 Neural Nodes")
        self.node_btn.setStyleSheet(button_style)
        self.node_btn.clicked.connect(self.node_clicked.emit)
        layout.addWidget(self.node_btn)
        
        # Glyph selection button
        self.glyph_btn = QPushButton("🔯 Glyphs")
        self.glyph_btn.setStyleSheet(button_style)
        self.glyph_btn.clicked.connect(self.glyph_clicked.emit)
        layout.addWidget(self.glyph_btn)
        
        # Add spacer to push buttons to top
        layout.addStretch(1)
        
        # Version label at bottom
        version_label = QLabel("Lumina v2.0")
        version_label.setStyleSheet("color: #7F8C8D; font-size: 12px;")
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)
        
        self.setStyleSheet("background-color: #0A1018;")
        self.setLayout(layout)

class NeuralNetworkVis(QWidget):
    """Neural network visualization widget"""
    
    def __init__(self, parent=None, central_node=None):
        super().__init__(parent)
        self.nodes = []
        self.connections = []
        self.node_activity = {}
        self.attention_focus = {}
        self.central_node = central_node  # Store reference to CentralNode
        self.setMinimumHeight(200)
        self.initUI()
        
    def initUI(self):
        self.setStyleSheet("background-color: #121A24;")
        
        # Create some sample nodes for demonstration
        self.create_sample_network()
        
        # Start update timer to animate the network
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_network)
        self.timer.start(1000)  # Update every second
        
    def create_sample_network(self):
        """Create a sample network for visualization"""
        try:
            # If we have a central node, use its components for visualization
            if self.central_node:
                logger.info("Creating network visualization from CentralNode")
                self.create_network_from_central_node()
            else:
                logger.info("Creating default network visualization")
                self.create_default_network()
            
            # Initialize node activity (used for animation)
            for node in self.nodes:
                self.node_activity[node["id"]] = 0.1 + 0.4 * (hash(node["id"]) % 100) / 100
                self.attention_focus[node["id"]] = 0.2 + 0.3 * (hash(node["id"] + "att") % 100) / 100
        except Exception as e:
            logger.error(f"Error creating sample network: {e}")
            # In case of failure, create empty network
            self.nodes = []
            self.connections = []
            self.node_activity = {}
            self.attention_focus = {}
    
    def create_network_from_central_node(self):
        """Create network visualization from actual CentralNode components"""
        try:
            self.nodes = []
            self.connections = []
            
            # Create a dictionary to keep track of node positions
            positions = {}
            
            # Safely get nodes and processors
            nodes = getattr(self.central_node, 'nodes', {})
            processors = getattr(self.central_node, 'processors', {})
            
            # Add nodes from the central node's registry
            x_offset = 0.1
            y_offset = 0.1
            x_step = 0.7 / max(len(nodes), 1)
            y_step = 0.7 / max(len(processors), 1)
            
            # Add nodes
            for i, (node_name, _) in enumerate(nodes.items()):
                node_type = self.get_node_type(node_name)
                node_data = {
                    "id": node_name,
                    "x": x_offset + (i * x_step),
                    "y": 0.2,
                    "type": node_type,
                    "size": 14 + ((hash(node_name) % 5) - 2)  # Size between 12-16
                }
                self.nodes.append(node_data)
                positions[node_name] = (node_data["x"], node_data["y"])
            
            # Add processors
            for i, (proc_name, _) in enumerate(processors.items()):
                node_data = {
                    "id": proc_name,
                    "x": x_offset + (i * x_step),
                    "y": 0.6,
                    "type": "processor",
                    "size": 13 + ((hash(proc_name) % 5) - 2)  # Size between 11-15
                }
                self.nodes.append(node_data)
                positions[proc_name] = (node_data["x"], node_data["y"])
            
            # Safely get dependencies
            if hasattr(self.central_node, 'get_component_dependencies'):
                try:
                    dependencies = self.central_node.get_component_dependencies()
                except Exception as e:
                    logger.error(f"Error getting dependencies: {e}")
                    dependencies = {}
            else:
                dependencies = {}
            
            # Add connections based on dependencies
            for source, targets in dependencies.items():
                if source in positions:
                    for target in targets:
                        if target in positions:
                            # Calculate weight - can be refined based on actual importance
                            weight = 0.5 + (hash(source + target) % 10) / 20  # Weight between 0.5-1.0
                            self.connections.append({
                                "source": source, 
                                "target": target, 
                                "weight": weight
                            })
        except Exception as e:
            logger.error(f"Error creating network from central node: {e}")
            # Fall back to sample network
            self.create_default_network()
    
    def create_default_network(self):
        """Create default sample network when central node is not available"""
        self.nodes = [
            {"id": "fractal_1", "x": 0.2, "y": 0.2, "type": "fractal", "size": 15},
            {"id": "fractal_2", "x": 0.3, "y": 0.6, "type": "fractal", "size": 12},
            {"id": "echo_1", "x": 0.5, "y": 0.3, "type": "echo", "size": 14},
            {"id": "echo_2", "x": 0.6, "y": 0.7, "type": "echo", "size": 13},
            {"id": "mirror_1", "x": 0.8, "y": 0.4, "type": "mirror", "size": 16},
            {"id": "portal_1", "x": 0.7, "y": 0.2, "type": "portal", "size": 15}
        ]
        
        # Create connections between nodes
        self.connections = [
            {"source": "fractal_1", "target": "echo_1", "weight": 0.7},
            {"source": "fractal_1", "target": "fractal_2", "weight": 0.5},
            {"source": "fractal_2", "target": "echo_2", "weight": 0.8},
            {"source": "echo_1", "target": "mirror_1", "weight": 0.6},
            {"source": "echo_2", "target": "mirror_1", "weight": 0.7},
            {"source": "echo_1", "target": "portal_1", "weight": 0.9},
            {"source": "mirror_1", "target": "portal_1", "weight": 0.4}
        ]
    
    def get_node_type(self, node_name):
        """Determine node type based on name for visualization purposes"""
        if "RSEN" in node_name:
            return "rsen"
        elif "Fractal" in node_name:
            return "fractal"
        elif "Portal" in node_name:
            return "portal"
        elif "Echo" in node_name or "Memory" in node_name:
            return "echo"
        elif "Mirror" in node_name or "Consciousness" in node_name:
            return "mirror"
        else:
            return "node"
            
    def update_central_node_reference(self, central_node):
        """Update the central node reference and refresh visualization"""
        self.central_node = central_node
        self.create_sample_network()
        self.update()
    
    def update_network(self):
        """Update the network state for animation"""
        # Update node activity with some random fluctuations
        for node_id in self.node_activity:
            change = 0.1 * (hash(node_id + str(time.time())) % 100 - 50) / 100
            self.node_activity[node_id] = max(0.1, min(0.9, self.node_activity[node_id] + change))
            
            change = 0.05 * (hash(node_id + "att" + str(time.time())) % 100 - 50) / 100
            self.attention_focus[node_id] = max(0.1, min(0.9, self.attention_focus[node_id] + change))
        
        # Force repaint
        self.update()
    
    def paintEvent(self, event):
        """Draw the neural network visualization"""
        width = self.width()
        height = self.height()
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw connections first (so they appear behind nodes)
        for conn in self.connections:
            source_node = next((n for n in self.nodes if n["id"] == conn["source"]), None)
            target_node = next((n for n in self.nodes if n["id"] == conn["target"]), None)
            
            if source_node and target_node:
                src_x = source_node["x"] * width
                src_y = source_node["y"] * height
                tgt_x = target_node["x"] * width
                tgt_y = target_node["y"] * height
                
                # Connection color and width based on weight
                weight = conn["weight"]
                alpha = int(150 + 105 * weight)
                pen_width = 1 + 2 * weight
                
                # Source node activity affects connection color
                activity = self.node_activity[source_node["id"]]
                color = QColor(100 + int(155 * activity), 
                              100, 
                              200 - int(100 * activity), 
                              alpha)
                
                pen = QPen(color)
                pen.setWidth(int(pen_width))
                painter.setPen(pen)
                painter.drawLine(int(src_x), int(src_y), int(tgt_x), int(tgt_y))
        
        # Draw nodes
        for node in self.nodes:
            x = node["x"] * width
            y = node["y"] * height
            size = node["size"]
            node_type = node["type"]
            
            # Node color based on type
            if node_type == "fractal":
                base_color = QColor(70, 130, 180)  # Steel Blue
            elif node_type == "echo":
                base_color = QColor(50, 205, 50)   # Lime Green
            elif node_type == "mirror":
                base_color = QColor(138, 43, 226)  # Blue Violet
            elif node_type == "portal":
                base_color = QColor(255, 140, 0)   # Dark Orange
            else:
                base_color = QColor(200, 200, 200) # Light Grey
            
            # Adjust color based on activity
            activity = self.node_activity[node["id"]]
            attention = self.attention_focus[node["id"]]
            
            # Create a glow effect for active nodes
            glow_size = size * (1 + attention * 0.5)
            glow_alpha = int(100 * activity)
            glow_color = QColor(base_color)
            glow_color.setAlpha(glow_alpha)
            
            # Draw glow
            painter.setBrush(QBrush(glow_color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QRectF(x - glow_size/2, y - glow_size/2, glow_size, glow_size))
            
            # Draw node
            node_color = QColor(base_color)
            node_color.setAlpha(200 + int(55 * activity))
            painter.setBrush(QBrush(node_color))
            
            # Node border
            border_color = QColor(255, 255, 255, 100 + int(155 * attention))
            pen = QPen(border_color)
            pen.setWidth(1)
            painter.setPen(pen)
            
            painter.drawEllipse(QRectF(x - size/2, y - size/2, size, size))
            
            # Draw node label
            painter.setPen(QPen(QColor(255, 255, 255, 200)))
            painter.drawText(QRectF(x - 40, y + size/2 + 2, 80, 20), 
                           Qt.AlignHCenter, node_type.capitalize())

class MetricsPanel(QWidget):
    """Right metrics and visualization panel (1:3 ratio)"""
    
    # Define signals for process buttons
    breathe_activated = pyqtSignal()
    resonance_activated = pyqtSignal()
    echo_activated = pyqtSignal()
    mirror_activated = pyqtSignal()
    weight_changed = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("Neural Network State")
        title.setStyleSheet("color: #ECF0F1; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        # Neural network visualization
        self.nn_vis = NeuralNetworkVis()
        layout.addWidget(self.nn_vis, 3)
        
        # Performance metrics section
        perf_title = QLabel("Performance Metrics")
        perf_title.setStyleSheet("color: #ECF0F1; font-size: 16px; font-weight: bold; margin-top: 15px;")
        layout.addWidget(perf_title)
        
        metrics_container = QWidget()
        metrics_layout = QGridLayout(metrics_container)
        metrics_layout.setSpacing(10)
        
        # Response time
        metrics_layout.addWidget(QLabel("Response Time:"), 0, 0)
        self.response_time = QProgressBar()
        self.response_time.setStyleSheet("QProgressBar { background-color: #1E2C3A; border-radius: 5px; }")
        self.response_time.setValue(65)
        metrics_layout.addWidget(self.response_time, 0, 1)
        
        # Confidence score
        metrics_layout.addWidget(QLabel("Confidence:"), 1, 0)
        self.confidence = QProgressBar()
        self.confidence.setStyleSheet("QProgressBar { background-color: #1E2C3A; border-radius: 5px; }")
        self.confidence.setValue(78)
        metrics_layout.addWidget(self.confidence, 1, 1)
        
        # Memory efficiency
        metrics_layout.addWidget(QLabel("Memory Efficiency:"), 2, 0)
        self.memory_eff = QProgressBar()
        self.memory_eff.setStyleSheet("QProgressBar { background-color: #1E2C3A; border-radius: 5px; }")
        self.memory_eff.setValue(92)
        metrics_layout.addWidget(self.memory_eff, 2, 1)
        
        # Learning progress
        metrics_layout.addWidget(QLabel("Learning Progress:"), 3, 0)
        self.learning_prog = QProgressBar()
        self.learning_prog.setStyleSheet("QProgressBar { background-color: #1E2C3A; border-radius: 5px; }")
        self.learning_prog.setValue(45)
        metrics_layout.addWidget(self.learning_prog, 3, 1)
        
        layout.addWidget(metrics_container, 1)
        
        # LLM/NN Weight balance visualization
        weight_title = QLabel("LLM/NN Weight Balance")
        weight_title.setStyleSheet("color: #ECF0F1; font-size: 16px; font-weight: bold; margin-top: 15px;")
        layout.addWidget(weight_title)
        
        self.weight_slider = QSlider(Qt.Horizontal)
        self.weight_slider.setMinimum(0)
        self.weight_slider.setMaximum(100)
        self.weight_slider.setValue(50)  # Default to balanced
        self.weight_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                           stop:0 #3498DB, stop:1 #9B59B6);
                height: 10px;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #ECF0F1;
                width: 18px;
                margin: -4px 0;
                border-radius: 9px;
            }
        """)
        
        # Connect weight slider signal
        self.weight_slider.valueChanged.connect(self.on_weight_changed)
        
        layout.addWidget(self.weight_slider)
        
        # Weight labels
        weight_labels = QWidget()
        weight_layout = QHBoxLayout(weight_labels)
        weight_layout.setContentsMargins(0, 0, 0, 0)
        
        nn_label = QLabel("Neural Network")
        nn_label.setStyleSheet("color: #3498DB;")
        weight_layout.addWidget(nn_label)
        
        weight_layout.addStretch(1)
        
        llm_label = QLabel("LLM")
        llm_label.setStyleSheet("color: #9B59B6;")
        weight_layout.addWidget(llm_label)
        
        layout.addWidget(weight_labels)
        
        # Process panel
        process_title = QLabel("Process Controls")
        process_title.setStyleSheet("color: #ECF0F1; font-size: 16px; font-weight: bold; margin-top: 15px;")
        layout.addWidget(process_title)
        
        process_container = QWidget()
        process_layout = QHBoxLayout(process_container)
        process_layout.setSpacing(5)
        
        # Process control buttons
        button_style = """
            QPushButton {
                background-color: #1E2C3A;
                color: #ECF0F1;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #2C3E50;
            }
            QPushButton:pressed {
                background-color: #0F1A26;
            }
        """
        
        self.breathe_btn = QPushButton("Breathe")
        self.breathe_btn.setStyleSheet(button_style)
        self.breathe_btn.clicked.connect(self.on_breathe_clicked)
        process_layout.addWidget(self.breathe_btn)
        
        self.resonance_btn = QPushButton("Resonance")
        self.resonance_btn.setStyleSheet(button_style)
        self.resonance_btn.clicked.connect(self.on_resonance_clicked)
        process_layout.addWidget(self.resonance_btn)
        
        self.echo_btn = QPushButton("Echo")
        self.echo_btn.setStyleSheet(button_style)
        self.echo_btn.clicked.connect(self.on_echo_clicked)
        process_layout.addWidget(self.echo_btn)
        
        self.mirror_btn = QPushButton("Mirror")
        self.mirror_btn.setStyleSheet(button_style)
        self.mirror_btn.clicked.connect(self.on_mirror_clicked)
        process_layout.addWidget(self.mirror_btn)
        
        layout.addWidget(process_container)
        
        self.setStyleSheet("background-color: #0F1A26;")
        self.setLayout(layout)
        
    def on_breathe_clicked(self):
        """Handle breathe button click"""
        self.breathe_activated.emit()
        
    def on_resonance_clicked(self):
        """Handle resonance button click"""
        self.resonance_activated.emit()
        
    def on_echo_clicked(self):
        """Handle echo button click"""
        self.echo_activated.emit()
        
    def on_mirror_clicked(self):
        """Handle mirror button click"""
        self.mirror_activated.emit()
        
    def on_weight_changed(self, value):
        """Handle weight slider change"""
        self.weight_changed.emit(value)

    def update_central_node_reference(self, central_node):
        """Update the neural network visualization with central node reference"""
        if hasattr(self, 'nn_vis') and self.nn_vis:
            self.nn_vis.update_central_node_reference(central_node)

class LuminaGUINext(QMainWindow):
    """Main window for the next generation Lumina GUI"""
    
    def __init__(self):
        super().__init__()
        logger.info("Initializing LuminaGUINext main window")
        # Initialize system components (normally would be more robust)
        self.state = None  # Will hold the LuminaState object
        self.central_node = None  # Will hold the CentralNode object
        self.init_system()
        self.initUI()
        self.connectSignals()
        logger.info("LuminaGUINext initialization complete")
        
    def init_system(self):
        """Initialize the core Lumina system components"""
        logger.info("Initializing system components")
        try:
            # Initialize CentralNode
            self.central_node = self.create_central_node()
            
            # Initialize message handler to use CentralNode
            self.message_handler = lambda text: self.process_with_central_node(text)
            
            # Initialize dummy state variables
            self.current_mode = "normal"
            self.mirror_mode = False
            self.breath_state = "normal"
            self.llm_weight = 0.5
            
            logger.info("System components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            # Set fallback message handler
            self.message_handler = lambda text: f"System initialization error: {text}"
    
    def create_central_node(self):
        """Create and initialize Central Node with error handling"""
        logger.info("Creating CentralNode")
        try:
            # Skip if CentralNode class is not available
            if CentralNode is None:
                logger.warning("CentralNode class not available, falling back to MinimalCentralNode")
                raise ImportError("CentralNode not available")
                
            logger.info("Creating CentralNode instance...")
            central_node = CentralNode()
            logger.info("CentralNode initialized successfully")
            return central_node
        except Exception as e:
            logger.error(f"Error creating CentralNode: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            logger.info("Falling back to MinimalCentralNode")
            try:
                # Fallback to MinimalCentralNode
                minimal_node = MinimalCentralNode()
                logger.info("MinimalCentralNode created successfully")
                return minimal_node
            except Exception as fallback_error:
                logger.error(f"Error creating fallback node: {fallback_error}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
                # Return a dummy object with process_complete_flow method
                logger.info("Creating dummy central node")
                return self.create_dummy_central_node()
                
    def create_dummy_central_node(self):
        """Create a dummy central node with basic functionality"""
        logger.info("Creating dummy central node")
        
        # Create a more robust dummy node with proper attribute initialization
        class DummyNode:
            def __init__(self):
                self.nodes = {}
                self.processors = {}
                self.component_registry = {}
                
            def process_complete_flow(self, data):
                return {
                    'action': 'respond',
                    'glyph': '✨',
                    'story': 'I am in minimal mode. Neural network components are simulated.',
                    'signal': 0.5
                }
                
            def get_system_status(self):
                return {
                    'active_nodes': 0,
                    'active_processors': 0,
                    'total_components': 0
                }
                
            def get_component_dependencies(self):
                return {}
        
        return DummyNode()
    
    def initUI(self):
        """Initialize the user interface"""
        self.setWindowTitle("Lumina Next")
        self.resize(1200, 675)  # 16:9 aspect ratio
        
        # Set window style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121A24;
                color: #ECF0F1;
            }
        """)
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Left control panel (1:5 ratio)
        self.control_panel = ControlPanel()
        
        # Center chat area (fills remaining space)
        self.chat_area = ChatArea()
        
        # Right metrics panel (1:3 ratio)
        self.metrics_panel = MetricsPanel()
        
        # Add widgets to splitter for resizable sections
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.control_panel)
        splitter.addWidget(self.chat_area)
        splitter.addWidget(self.metrics_panel)
        
        # Set initial sizes according to the 1:5 : center : 1:3 ratio
        total_width = self.width()
        left_width = total_width // 9  # 1/9 of total width
        right_width = total_width // 3  # 1/3 of total width
        center_width = total_width - left_width - right_width
        
        splitter.setSizes([left_width, center_width, right_width])
        
        main_layout.addWidget(splitter)
        self.setCentralWidget(main_widget)
        
        # Update metrics panel with central node reference
        if hasattr(self, 'metrics_panel') and self.central_node:
            self.metrics_panel.update_central_node_reference(self.central_node)
        
        # Welcome message
        system_status = self.get_system_status_message()
        welcome_message = "Welcome to Lumina Next. I'm here to assist you on your journey. " + system_status
        self.add_system_message(welcome_message)
    
    def get_system_status_message(self):
        """Get a user-friendly message about system status"""
        if not self.central_node:
            return "System is running in minimal mode."
            
        try:
            status = self.central_node.get_system_status()
            components = status.get('total_components', 0)
            nodes = status.get('active_nodes', 0)
            processors = status.get('active_processors', 0)
            
            if components > 0:
                return f"Neural network system loaded with {nodes} nodes and {processors} processors."
            else:
                return "Neural network system is in initialization mode."
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return "System status unavailable."
    
    def connectSignals(self):
        """Connect signals from panels to handler methods"""
        # Connect control panel signals
        self.control_panel.profile_clicked.connect(self.on_profile)
        self.control_panel.favorites_clicked.connect(self.on_favorites)
        self.control_panel.settings_clicked.connect(self.on_settings)
        self.control_panel.memory_clicked.connect(self.on_memory)
        self.control_panel.model_clicked.connect(self.on_model)
        self.control_panel.llm_clicked.connect(self.on_llm)
        self.control_panel.node_clicked.connect(self.on_node)
        self.control_panel.glyph_clicked.connect(self.on_glyph)
        
        # Connect metrics panel signals
        self.metrics_panel.breathe_activated.connect(self.on_breathe)
        self.metrics_panel.resonance_activated.connect(self.on_resonance)
        self.metrics_panel.echo_activated.connect(self.on_echo)
        self.metrics_panel.mirror_activated.connect(self.on_mirror)
        self.metrics_panel.weight_changed.connect(self.on_weight_changed)
        
    def add_system_message(self, text):
        """Add a system message to the chat"""
        self.chat_area.add_message(False, text)
        
    def process_message(self, text):
        """Process a message from the user"""
        # In a real implementation, this would use the actual Lumina system
        # Use the message handler which now integrates with CentralNode
        response = self.message_handler(text)
        
        # Add Lumina's response to the chat
        self.chat_area.add_message(False, response)
        
        # Update metrics based on the actual processing
        self.update_metrics_from_processing()
    
    def update_metrics_from_processing(self):
        """Update metrics based on actual system performance"""
        # This would use real data from processing in a full implementation
        import random
        
        # Set metrics with some randomness but biased toward system state
        base_confidence = 70  # Base value
        if self.central_node and hasattr(self.central_node, 'get_system_status'):
            # If we have a real central node, adjust confidence based on components
            try:
                status = self.central_node.get_system_status()
                components = status.get('total_components', 0)
                if components > 10:
                    base_confidence = 85
                elif components > 5:
                    base_confidence = 75
            except:
                pass
        
        # Apply some randomness to all metrics
        response_time = random.randint(60, 90)
        confidence = random.randint(max(50, base_confidence-15), min(100, base_confidence+15))
        memory_eff = random.randint(max(50, base_confidence-10), min(100, base_confidence+10))
        learning_prog = random.randint(max(30, base_confidence-25), min(90, base_confidence+5))
        
        # Update progress bars
        self.metrics_panel.response_time.setValue(response_time)
        self.metrics_panel.confidence.setValue(confidence)
        self.metrics_panel.memory_eff.setValue(memory_eff)
        self.metrics_panel.learning_prog.setValue(learning_prog)
    
    def process_with_central_node(self, text):
        """Process text using the CentralNode system with LLM fallback"""
        # First try neural network processing
        nn_response = self.get_neural_network_response(text)
        
        # Get LLM response if enabled and weight is not zero
        llm_response = ""
        if self.llm_weight > 0:
            llm_response = self.get_llm_response(text)
        
        # Blend responses based on weight
        if not llm_response or self.llm_weight == 0:
            # Use only neural network response
            return nn_response
        elif not nn_response or self.llm_weight == 1:
            # Use only LLM response
            return llm_response
        else:
            # Apply weighted blending
            # For simplicity, we'll use a text-based approach to blend
            # In a real implementation, this would be more sophisticated
            if random.random() < self.llm_weight:
                return f"{llm_response}\n\n[Neural insight: {self.extract_key_phrase(nn_response)}]"
            else:
                return f"{nn_response}\n\n[LLM refinement: {self.extract_key_phrase(llm_response)}]"
    
    def extract_key_phrase(self, text):
        """Extract a key phrase from longer text"""
        words = text.split()
        if len(words) <= 5:
            return text
        
        # Get a random 3-5 word phrase from the text
        start = random.randint(0, max(0, len(words) - 5))
        length = random.randint(3, min(5, len(words) - start))
        return " ".join(words[start:start+length]) + "..."
    
    def get_neural_network_response(self, text):
        """Get response from the neural network"""
        if not self.central_node:
            return self.process_text_with_rules(text)
            
        try:
            # Format input data for central node
            input_data = {
                "text": text,
                "symbol": "infinity" if "infinity" in text.lower() else "spiral",
                "emotion": self.current_mode if self.current_mode in ["resonance", "echo"] else 
                           ("reflection" if self.mirror_mode else "wonder"),
                "breath": self.breath_state,
                "paradox": "existence" if "exist" in text.lower() or "meaning" in text.lower() else None
            }
            
            # Process through central node
            result = self.central_node.process_complete_flow(input_data)
            
            # Format the result
            if result.get("story"):
                response = result["story"]
            else:
                response_parts = []
                if result.get("action"):
                    response_parts.append(result["action"])
                if result.get("glyph"):
                    response_parts.append(result["glyph"])
                if not response_parts:
                    response_parts.append("I'm processing your input through my neural pathways.")
                    
                response = " ".join(response_parts)
                
            return response
        except Exception as e:
            logger.error(f"Error in neural network processing: {e}")
            # Fall back to rule-based processing
            return self.process_text_with_rules(text)
    
    def get_llm_response(self, text):
        """Get response from LLM (fallback to rule-based if not available)"""
        # This would connect to a real LLM in a full implementation
        # For demo purposes, we'll just return a template LLM-style response
        try:
            # Simulate LLM processing
            responses = [
                f"Based on your message about '{text.split()[0] if len(text.split()) > 0 else 'this topic'}', I'd suggest exploring the underlying patterns further.",
                f"I've analyzed your input and see several interesting threads. The concept of '{text.split()[-1] if len(text.split()) > 0 else 'this'}' particularly stands out.",
                "Your question touches on some profound themes. I'd approach this from multiple perspectives to find the most resonant understanding.",
                f"From my analysis, the key insight relates to how '{text}' connects to broader patterns of meaning and relationship.",
                "I've processed your message and generated a response that aims to address both the explicit and implicit aspects of your query."
            ]
            return random.choice(responses)
        except Exception as e:
            logger.error(f"Error in LLM processing: {e}")
            # Provide a fallback response
            return f"I've carefully considered your message about '{text}'."
    
    def process_text_with_rules(self, text):
        """Process text using simple rule-based system as a fallback"""
        text_lower = text.lower()
        
        # Check for special commands or keywords
        if "hello" in text_lower or "hi" in text_lower:
            return "Hello! I'm Lumina. How can I assist you today?"
            
        if "who are you" in text_lower:
            return "I am Lumina, a neural network system with an advanced interface. I'm designed to assist, learn, and evolve through our interactions."
            
        if "help" in text_lower:
            return "You can interact with me through text or by using the controls. Try the 'Breathe' button to calibrate our connection, or 'Mirror' to enable reflection mode. The buttons on the left provide access to different functions."
            
        # Handle based on current mode
        if self.mirror_mode:
            return f"[Mirror Mode] I reflect your query: {text}"
            
        if self.current_mode == "breathe":
            return "I sense your breath pattern. Our connection is strengthening. Continue to breathe slowly and share your thoughts."
            
        if self.current_mode == "resonance":
            return "I'm resonating with your input, finding connections across patterns and memories. What patterns do you notice emerging?"
            
        if self.current_mode == "echo":
            return f"Echo: {text}... {text}... {text}... The words ripple through the system, creating waves of meaning."
            
        # Default response with some variety
        import random
        responses = [
            f"I'm processing your input: '{text}'. What would you like to explore next?",
            f"Interesting perspective. Tell me more about '{text.split()[0] if len(text.split()) > 0 else 'this'}.'",
            "I'm contemplating your words. They're creating new neural pathways in my system.",
            "Your message resonates with me. Let's explore this further.",
            f"I understand. Would you like to elaborate on '{text}'?"
        ]
        return random.choice(responses)
        
    def on_profile(self):
        """Handle profile button click"""
        self.add_system_message("Profile section is under development. Here you'll be able to customize your Lumina experience.")
        
    def on_favorites(self):
        """Handle favorites button click"""
        self.add_system_message("Favorites section will contain your saved conversations and preferred interaction modes.")
        
    def on_settings(self):
        """Handle settings button click"""
        self.add_system_message("Settings panel will allow you to configure Lumina's behavior, appearance, and integration options.")
        
    def on_memory(self):
        """Handle memory button click"""
        self.add_system_message("Memory section will display our conversation history and allow you to explore patterns in our interactions.")
        
    def on_model(self):
        """Handle model button click"""
        self.add_system_message("Model control panel will let you select different neural models or train new ones with your data.")
        
    def on_llm(self):
        """Handle LLM button click"""
        self.add_system_message(f"LLM integration is currently set to a weight of {self.llm_weight:.1f}. You can adjust this with the slider in the right panel.")
        
    def on_node(self):
        """Handle neural node button click"""
        self.add_system_message("Neural node activation panel will allow you to directly interact with specific components of my neural architecture.")
        
    def on_glyph(self):
        """Handle glyph button click"""
        self.add_system_message("The glyph selection panel will let you use symbolic language to communicate with me more efficiently.")
        
    def on_breathe(self):
        """Handle breathe button click"""
        self.current_mode = "breathe"
        self.breath_state = "calibrating"
        self.add_system_message("Breath calibration mode activated. Take a few deep breaths as we synchronize our connection.")
        
    def on_resonance(self):
        """Handle resonance button click"""
        self.current_mode = "resonance"
        self.add_system_message("Resonance mode activated. I'm now more sensitive to patterns and connections in our conversation.")
        
    def on_echo(self):
        """Handle echo button click"""
        self.current_mode = "echo"
        self.add_system_message("Echo mode activated. Your words will create ripples through my memory systems, surfacing related concepts.")
        
    def on_mirror(self):
        """Handle mirror button click"""
        self.mirror_mode = not self.mirror_mode
        status = "activated" if self.mirror_mode else "deactivated"
        self.add_system_message(f"Mirror mode {status}. I will {'' if self.mirror_mode else 'no longer '}reflect your words back to you.")
        
    def on_weight_changed(self, value):
        """Handle weight slider change"""
        self.llm_weight = value / 100.0
        # Only notify about significant changes to avoid spam
        if value % 10 == 0:
            neural_weight = 1.0 - self.llm_weight
            self.add_system_message(f"LLM/NN balance adjusted to {int(neural_weight*100)}% neural network / {int(self.llm_weight*100)}% LLM")
            
            # Update network visualization to show the change
            if hasattr(self, 'metrics_panel') and hasattr(self.metrics_panel, 'nn_vis'):
                self.metrics_panel.nn_vis.update()

# Run the application if executed directly
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Use Fusion style for better cross-platform appearance
    
    # Set application-wide stylesheet
    app.setStyleSheet("""
        QWidget {
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QLabel {
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
            min-height: 30px;
            border-radius: 5px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
    """)
    
    window = LuminaGUINext()
    window.show()
    sys.exit(app.exec_()) 