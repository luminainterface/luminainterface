#!/usr/bin/env python
"""
LUMINA V7.5 Holographic Frontend
This module provides compatibility with the v7 holographic frontend,
with added support for v7.5 specific features.
"""

import os
import sys
import logging
from pathlib import Path
import math
import random
from typing import Dict, List, Any, Optional

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'holographic_frontend.log'), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HolographicFrontend")

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(project_root)

# Import Qt components
from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QWidget, QHBoxLayout, QPushButton, QFrame)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QRadialGradient

class SimplifiedCentralNode:
    """Simplified central node for visualization testing"""
    
    def __init__(self):
        self.logger = logging.getLogger('SimplifiedCentralNode')
        self.nodes = {}
        self.processors = {}
        self.component_registry = {}
        self.connections = {}
        
        # Initialize with mock components
        self._initialize_mock_components()
    
    def _initialize_mock_components(self):
        """Initialize mock components for visualization"""
        # Mock nodes
        self.nodes = {
            'RSEN': {'activation': 0.7, 'type': 'node'},
            'HybridNode': {'activation': 0.5, 'type': 'node'},
            'NodeZero': {'activation': 0.8, 'type': 'node'},
            'PortalNode': {'activation': 0.6, 'type': 'node'},
            'WormholeNode': {'activation': 0.4, 'type': 'node'},
            'ZPENode': {'activation': 0.9, 'type': 'node'},
            'NeutrinoNode': {'activation': 0.3, 'type': 'node'},
            'GameTheoryNode': {'activation': 0.5, 'type': 'node'},
            'ConsciousnessNode': {'activation': 0.7, 'type': 'node'},
            'GaugeTheoryNode': {'activation': 0.6, 'type': 'node'},
            'FractalNodes': {'activation': 0.8, 'type': 'node'},
            'InfiniteMindsNode': {'activation': 0.4, 'type': 'node'},
            'VoidInfinityNode': {'activation': 0.5, 'type': 'node'}
        }
        
        # Mock processors
        self.processors = {
            'NeuralProcessor': {'activation': 0.7, 'type': 'processor'},
            'LanguageProcessor': {'activation': 0.6, 'type': 'processor'},
            'LuminaProcessor': {'activation': 0.8, 'type': 'processor'},
            'MoodProcessor': {'activation': 0.5, 'type': 'processor'},
            'NodeManager': {'activation': 0.9, 'type': 'processor'},
            'WikiLearner': {'activation': 0.4, 'type': 'processor'},
            'WikiVocabulary': {'activation': 0.6, 'type': 'processor'},
            'WikipediaTrainingModule': {'activation': 0.7, 'type': 'processor'},
            'WikipediaTrainer': {'activation': 0.5, 'type': 'processor'},
            'LuminaNeural': {'activation': 0.8, 'type': 'processor'},
            'PhysicsEngine': {'activation': 0.6, 'type': 'processor'},
            'CalculusEngine': {'activation': 0.7, 'type': 'processor'},
            'PhysicsMetaphysicsFramework': {'activation': 0.5, 'type': 'processor'},
            'HyperdimensionalThought': {'activation': 0.8, 'type': 'processor'},
            'QuantumInfection': {'activation': 0.6, 'type': 'processor'},
            'NodeIntegration': {'activation': 0.7, 'type': 'processor'}
        }
        
        # Combine all components
        self.component_registry = {**self.nodes, **self.processors}
    
    def get_component_dependencies(self):
        """Get mock component dependencies"""
        return {
            'RSEN': ['NeuralProcessor', 'NodeManager'],
            'HybridNode': ['NeuralProcessor', 'NodeManager'],
            'NodeZero': ['NodeManager'],
            'PortalNode': ['PhysicsEngine', 'CalculusEngine'],
            'WormholeNode': ['PhysicsEngine', 'QuantumInfection'],
            'ZPENode': ['PhysicsEngine', 'QuantumInfection'],
            'NeutrinoNode': ['PhysicsEngine', 'PhysicsMetaphysicsFramework'],
            'GameTheoryNode': ['NodeManager', 'HyperdimensionalThought'],
            'ConsciousnessNode': ['LuminaNeural', 'HyperdimensionalThought'],
            'GaugeTheoryNode': ['PhysicsEngine', 'PhysicsMetaphysicsFramework'],
            'FractalNodes': ['CalculusEngine', 'HyperdimensionalThought'],
            'InfiniteMindsNode': ['LuminaNeural', 'HyperdimensionalThought'],
            'VoidInfinityNode': ['PhysicsEngine', 'QuantumInfection'],
            'WikiLearner': ['WikiVocabulary', 'WikipediaTrainingModule'],
            'WikipediaTrainer': ['WikiVocabulary', 'WikipediaTrainingModule']
        }
    
    def get_component(self, name: str) -> Optional[Dict]:
        """Get a component by name"""
        return self.component_registry.get(name)

class NetworkVisualization(QFrame):
    """Interactive network visualization widget"""
    
    def __init__(self, parent=None, central_node=None):
        super().__init__(parent)
        self.central_node = central_node
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: rgba(40, 44, 52, 200); border-radius: 4px;")
        
        # Network data
        self.nodes = []
        self.connections = []
        self.selected_node = None
        self.hovered_node = None
        
        # Animation state
        self.animation_step = 0
        self.node_activities = {}
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        
        # Start animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(50)  # 50ms update = ~20fps
    
    def update_network_data(self, nodes: List[Dict], connections: List[Dict]):
        """Update network visualization data"""
        self.nodes = nodes
        self.connections = connections
        self.update()
    
    def update_animation(self):
        """Update animation state"""
        self.animation_step = (self.animation_step + 1) % 360
        self.update()
    
    def paintEvent(self, event):
        """Custom paint event to render the network visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw connections first (behind nodes)
        self.draw_connections(painter)
        
        # Draw nodes
        self.draw_nodes(painter)
    
    def draw_connections(self, painter):
        """Draw connections between nodes"""
        for conn in self.connections:
            source = next((n for n in self.nodes if n["id"] == conn["source"]), None)
            target = next((n for n in self.nodes if n["id"] == conn["target"]), None)
            
            if not source or not target:
                continue
            
            # Get connection strength
            strength = conn.get("weight", 0.5)
            
            # Determine color and opacity based on connection state
            if conn.get("active", False):
                color = QColor(52, 152, 219, int(100 + 155 * strength))
                width = 1.5 + 2.5 * strength
                style = Qt.SolidLine
            else:
                color = QColor(52, 73, 94, int(60 + 60 * strength))
                width = 1
                style = Qt.DotLine
            
            # Set pen for drawing
            pen = QPen(color, width)
            pen.setStyle(style)
            painter.setPen(pen)
            
            # Draw the connection line
            painter.drawLine(
                int(source["x"]), int(source["y"]),
                int(target["x"]), int(target["y"])
            )
    
    def draw_nodes(self, painter):
        """Draw network nodes"""
        for node in self.nodes:
            x = node["x"]
            y = node["y"]
            size = node.get("size", 12)
            activation = node.get("activation", 0.0)
            
            # Calculate pulse effect
            pulse = math.sin(self.animation_step * math.pi / 180) * 0.5 + 0.5
            
            # Determine node color based on type and activation
            if node.get("type") == "central":
                color = QColor(231, 76, 60)  # Red for central node
            else:
                # Interpolate between blue and yellow based on activation
                r = int(52 + 179 * activation)
                g = int(152 + 103 * activation)
                b = int(219 - 150 * activation)
                color = QColor(r, g, b)
            
            # Node glow for active nodes
            if activation > 0.1 or pulse > 0:
                glow_radius = size + 10
                glow = QRadialGradient(x, y, glow_radius)
                glow_color = QColor(color)
                glow_color.setAlpha(int(100 * max(activation, pulse * 0.5)))
                glow.setColorAt(0, glow_color)
                glow.setColorAt(1, QColor(color.red(), color.green(), color.blue(), 0))
                painter.setBrush(QBrush(glow))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(
                    x - glow_radius, y - glow_radius,
                    glow_radius * 2, glow_radius * 2
                )
            
            # Draw node circle
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
            painter.drawEllipse(
                x - size/2, y - size/2,
                size, size
            )

class HolographicMainWindow(QMainWindow):
    """Main window for the holographic frontend"""
    def __init__(self, mock=False, port=5678, gui_framework="PySide6"):
        super().__init__()
        self.setWindowTitle("LUMINA V7.5 Holographic Frontend")
        self.resize(800, 600)
        
        # Initialize central node
        self.central_node = SimplifiedCentralNode()
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add status label
        status_label = QLabel("LUMINA Holographic UI Initialized")
        status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(status_label)
        
        # Add network visualization
        self.network_viz = NetworkVisualization(central_node=self.central_node)
        layout.addWidget(self.network_viz)
        
        # Add control buttons
        controls_layout = QHBoxLayout()
        refresh_button = QPushButton("Refresh Network")
        refresh_button.clicked.connect(self.refresh_network)
        controls_layout.addWidget(refresh_button)
        layout.addLayout(controls_layout)
        
        logger.info("HolographicMainWindow initialized")
        
        # Initialize with real data from central node
        self.refresh_network()
    
    def refresh_network(self):
        """Refresh network visualization data from central node"""
        nodes = []
        connections = []
        
        # Get component information from central node
        if hasattr(self.central_node, 'get_component_dependencies'):
            components = self.central_node.get_component_dependencies()
            
            # Create central node
            center_x = self.network_viz.width() / 2
            center_y = self.network_viz.height() / 2
            nodes.append({
                "id": "central",
                "x": center_x,
                "y": center_y,
                "size": 20,
                "type": "central",
                "activation": 1.0
            })
            
            # Create component nodes
            component_list = list(components.keys())
            num_components = len(component_list)
            radius = min(center_x, center_y) * 0.7
            
            for i, component_name in enumerate(component_list):
                angle = i * (2 * math.pi / num_components)
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                
                # Get component status
                component = self.central_node.get_component(component_name)
                activation = component.get('activation', 0.5) if component else 0.5
                
                nodes.append({
                    "id": component_name,
                    "x": x,
                    "y": y,
                    "size": 12 + random.randint(0, 8),
                    "type": "component",
                    "activation": activation
                })
                
                # Create connections to central node
                connections.append({
                    "source": "central",
                    "target": component_name,
                    "weight": 0.8,  # Default weight
                    "active": True
                })
                
                # Add connections between components based on dependencies
                dependencies = components[component_name]
                for dep in dependencies:
                    if dep in component_list:
                        connections.append({
                            "source": component_name,
                            "target": dep,
                            "weight": 0.5,
                            "active": True
                        })
        
        # Update visualization
        self.network_viz.update_network_data(nodes, connections)

def run_holographic_frontend(mock=False, port=5678, gui_framework="PySide6"):
    """
    Run the holographic frontend with v7.5 enhancements
    
    Args:
        mock (bool): Whether to use mock data
        port (int): Port for backend communication
        gui_framework (str): GUI framework to use
    """
    logger.info(f"Starting v7.5 holographic frontend (mock={mock}, port={port}, gui={gui_framework})")
    
    try:
        # Create and show the main window
        app = QApplication(sys.argv)
        window = HolographicMainWindow(mock=mock, port=port, gui_framework=gui_framework)
        window.show()
        return app.exec_()
    except Exception as e:
        logger.error(f"Fatal error launching holographic frontend: {e}")
        return 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LUMINA V7.5 Holographic Frontend")
    parser.add_argument("--mock", action="store_true", help="Run with mock data")
    parser.add_argument("--port", type=int, default=5678, help="Port for backend communication")
    parser.add_argument("--gui-framework", default="PySide6", help="GUI framework to use")
    args = parser.parse_args()
    
    sys.exit(run_holographic_frontend(
        mock=args.mock,
        port=args.port,
        gui_framework=args.gui_framework
    )) 