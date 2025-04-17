from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QSlider
from PySide6.QtCore import Qt
from typing import Optional
import logging

class VisualizationPanel(QWidget):
    """Panel for neural network visualization."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop)
        
        self.initialize_ui()
        
    def initialize_ui(self):
        """Initialize the visualization panel UI."""
        try:
            # Visualization header
            header = QLabel("Network Visualization")
            header.setStyleSheet("font-size: 18px; font-weight: bold;")
            self.layout.addWidget(header)
            
            # Visualization type selector
            type_label = QLabel("Visualization Type:")
            self.type_combo = QComboBox()
            self.type_combo.addItems([
                "2D Network",
                "3D Network",
                "Fractal Pattern",
                "Quantum Field",
                "Cosmic Pattern"
            ])
            self.layout.addWidget(type_label)
            self.layout.addWidget(self.type_combo)
            
            # Visualization parameters
            params_label = QLabel("Parameters:")
            self.layout.addWidget(params_label)
            
            # Opacity control
            opacity_label = QLabel("Opacity:")
            self.opacity_slider = QSlider(Qt.Horizontal)
            self.opacity_slider.setRange(0, 100)
            self.opacity_slider.setValue(100)
            self.layout.addWidget(opacity_label)
            self.layout.addWidget(self.opacity_slider)
            
            # Scale control
            scale_label = QLabel("Scale:")
            self.scale_slider = QSlider(Qt.Horizontal)
            self.scale_slider.setRange(50, 200)
            self.scale_slider.setValue(100)
            self.layout.addWidget(scale_label)
            self.layout.addWidget(self.scale_slider)
            
            # Control buttons
            refresh_button = QPushButton("Refresh Visualization")
            refresh_button.clicked.connect(self.refresh_visualization)
            self.layout.addWidget(refresh_button)
            
            capture_button = QPushButton("Capture Visualization")
            capture_button.clicked.connect(self.capture_visualization)
            self.layout.addWidget(capture_button)
            
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
                QSlider::groove:horizontal {
                    background: #3d3d3d;
                    height: 8px;
                    border-radius: 4px;
                }
                QSlider::handle:horizontal {
                    background: #ffffff;
                    width: 16px;
                    margin: -4px 0;
                    border-radius: 8px;
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
            self.logger.error(f"Failed to initialize visualization panel UI: {str(e)}")
            raise
            
    def refresh_visualization(self):
        """Refresh the current visualization."""
        try:
            viz_type = self.type_combo.currentText()
            params = self.get_visualization_params()
            self.logger.info(f"Refreshing {viz_type} visualization with parameters: {params}")
            # Refresh logic here
        except Exception as e:
            self.logger.error(f"Failed to refresh visualization: {str(e)}")
            
    def capture_visualization(self):
        """Capture the current visualization."""
        try:
            self.logger.info("Capturing current visualization...")
            # Capture logic here
        except Exception as e:
            self.logger.error(f"Failed to capture visualization: {str(e)}")
            
    def get_visualization_params(self) -> dict:
        """Get current visualization parameters."""
        return {
            "type": self.type_combo.currentText(),
            "opacity": self.opacity_slider.value() / 100.0,
            "scale": self.scale_slider.value() / 100.0
        } 