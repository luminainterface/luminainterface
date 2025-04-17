from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox
from PySide6.QtCore import Qt
from typing import Optional, List
import logging

class NetworkPanel(QWidget):
    """Panel for neural network visualization and control."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.setMinimumWidth(250)
        
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop)
        
        self.initialize_ui()
        
    def initialize_ui(self):
        """Initialize the network panel UI."""
        try:
            # Network header
            header = QLabel("Neural Network")
            header.setStyleSheet("font-size: 18px; font-weight: bold;")
            self.layout.addWidget(header)
            
            # Network type selector
            type_label = QLabel("Network Type:")
            self.type_combo = QComboBox()
            self.type_combo.addItems(["Feedforward", "Recurrent", "Convolutional", "Transformer"])
            self.layout.addWidget(type_label)
            self.layout.addWidget(self.type_combo)
            
            # Layer controls
            layer_label = QLabel("Layer Configuration:")
            self.layer_combo = QComboBox()
            self.layer_combo.addItems(["2 Layers", "3 Layers", "4 Layers", "Custom"])
            self.layout.addWidget(layer_label)
            self.layout.addWidget(self.layer_combo)
            
            # Visualization controls
            viz_label = QLabel("Visualization:")
            self.viz_combo = QComboBox()
            self.viz_combo.addItems(["2D", "3D", "Fractal", "Quantum"])
            self.layout.addWidget(viz_label)
            self.layout.addWidget(self.viz_combo)
            
            # Control buttons
            train_button = QPushButton("Train Network")
            train_button.clicked.connect(self.train_network)
            self.layout.addWidget(train_button)
            
            visualize_button = QPushButton("Visualize")
            visualize_button.clicked.connect(self.visualize_network)
            self.layout.addWidget(visualize_button)
            
            # Add stretch to push everything to top
            self.layout.addStretch()
            
            # Set styles
            self.setStyleSheet("""
                QWidget {
                    background-color: #2d2d2d;
                    border-radius: 5px;
                    padding: 10px;
                }
                QLabel {
                    color: #ffffff;
                }
                QComboBox {
                    background-color: #3d3d3d;
                    color: #ffffff;
                    border: 1px solid #4d4d4d;
                    border-radius: 3px;
                    padding: 5px;
                }
                QPushButton {
                    background-color: #4d4d4d;
                    color: #ffffff;
                    border: none;
                    border-radius: 3px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #5d5d5d;
                }
            """)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize network panel UI: {str(e)}")
            raise
            
    def train_network(self):
        """Start network training."""
        try:
            network_type = self.type_combo.currentText()
            layers = self.layer_combo.currentText()
            self.logger.info(f"Starting training for {network_type} network with {layers}")
            # Training logic here
        except Exception as e:
            self.logger.error(f"Failed to start training: {str(e)}")
            
    def visualize_network(self):
        """Visualize the network."""
        try:
            viz_type = self.viz_combo.currentText()
            self.logger.info(f"Starting {viz_type} visualization")
            # Visualization logic here
        except Exception as e:
            self.logger.error(f"Failed to start visualization: {str(e)}")
            
    def get_network_config(self) -> dict:
        """Get current network configuration."""
        return {
            "type": self.type_combo.currentText(),
            "layers": self.layer_combo.currentText(),
            "visualization": self.viz_combo.currentText()
        } 