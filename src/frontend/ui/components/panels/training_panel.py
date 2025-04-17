from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox
from PySide6.QtCore import Qt
from typing import Optional
import logging

class TrainingPanel(QWidget):
    """Panel for managing neural network training parameters."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.setMinimumWidth(250)
        
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop)
        
        self.initialize_ui()
        
    def initialize_ui(self):
        """Initialize the training panel UI."""
        try:
            # Training header
            header = QLabel("Training Parameters")
            header.setStyleSheet("font-size: 18px; font-weight: bold;")
            self.layout.addWidget(header)
            
            # Epochs control
            epochs_label = QLabel("Epochs:")
            self.epochs_spin = QSpinBox()
            self.epochs_spin.setRange(1, 1000)
            self.epochs_spin.setValue(100)
            self.layout.addWidget(epochs_label)
            self.layout.addWidget(self.epochs_spin)
            
            # Learning rate control
            lr_label = QLabel("Learning Rate:")
            self.lr_spin = QDoubleSpinBox()
            self.lr_spin.setRange(0.0001, 1.0)
            self.lr_spin.setValue(0.001)
            self.lr_spin.setDecimals(4)
            self.layout.addWidget(lr_label)
            self.layout.addWidget(self.lr_spin)
            
            # Batch size control
            batch_label = QLabel("Batch Size:")
            self.batch_spin = QSpinBox()
            self.batch_spin.setRange(1, 256)
            self.batch_spin.setValue(32)
            self.layout.addWidget(batch_label)
            self.layout.addWidget(self.batch_spin)
            
            # Training options
            options_label = QLabel("Options:")
            self.use_gpu_check = QCheckBox("Use GPU")
            self.use_gpu_check.setChecked(True)
            self.early_stopping_check = QCheckBox("Early Stopping")
            self.early_stopping_check.setChecked(True)
            self.layout.addWidget(options_label)
            self.layout.addWidget(self.use_gpu_check)
            self.layout.addWidget(self.early_stopping_check)
            
            # Control buttons
            start_button = QPushButton("Start Training")
            start_button.clicked.connect(self.start_training)
            self.layout.addWidget(start_button)
            
            stop_button = QPushButton("Stop Training")
            stop_button.clicked.connect(self.stop_training)
            self.layout.addWidget(stop_button)
            
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
                QSpinBox, QDoubleSpinBox {
                    background-color: #3d3d3d;
                    color: #ffffff;
                    border: 1px solid #4d4d4d;
                    border-radius: 3px;
                    padding: 5px;
                }
                QCheckBox {
                    color: #ffffff;
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
            self.logger.error(f"Failed to initialize training panel UI: {str(e)}")
            raise
            
    def start_training(self):
        """Start the training process."""
        try:
            params = self.get_training_params()
            self.logger.info(f"Starting training with parameters: {params}")
            # Training logic here
        except Exception as e:
            self.logger.error(f"Failed to start training: {str(e)}")
            
    def stop_training(self):
        """Stop the training process."""
        try:
            self.logger.info("Stopping training...")
            # Stop training logic here
        except Exception as e:
            self.logger.error(f"Failed to stop training: {str(e)}")
            
    def get_training_params(self) -> dict:
        """Get current training parameters."""
        return {
            "epochs": self.epochs_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "batch_size": self.batch_spin.value(),
            "use_gpu": self.use_gpu_check.isChecked(),
            "early_stopping": self.early_stopping_check.isChecked()
        } 