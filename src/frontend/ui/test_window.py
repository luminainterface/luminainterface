import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QGroupBox, QGridLayout,
                             QFrame)
from PySide6.QtCore import Qt, QTimer

from src.frontend.ui.components.widgets.network_2d_widget import Network2DWidget

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network 2D Visualization")
        self.setGeometry(100, 100, 1200, 800)  # Increased window size
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for network visualization
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        main_layout.addWidget(left_panel, stretch=2)
        
        # Create and set up the network widget
        self.network_widget = Network2DWidget()
        left_layout.addWidget(self.network_widget)
        
        # Right panel for controls and metrics
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel, stretch=1)
        
        # Create metrics group
        metrics_group = QGroupBox("Real-time Metrics")
        metrics_layout = QGridLayout(metrics_group)
        
        # Add metric labels
        self.metric_labels = {}
        metrics = [
            ("Node Count", "get_current_node_count"),
            ("Connection Count", "get_connection_count"),
            ("Current Complexity", "get_current_complexity"),
            ("Animation Speed", "animation_speed"),
            ("Signal Speed", "signal_speed"),
            ("Signal Frequency", "signal_frequency"),
            ("Node Oscillation", "node_oscillation_speed"),
            ("Connection Oscillation", "connection_oscillation_speed"),
            ("Reverse Signal Speed", "reverse_signal_speed_multiplier"),
            ("Reverse Signal Strength", "reverse_signal_strength_multiplier"),
            ("Diagonal Signal Speed", "diagonal_signal_speed_multiplier"),
            ("Diagonal Signal Strength", "diagonal_signal_strength_multiplier")
        ]
        
        for i, (name, attr) in enumerate(metrics):
            metrics_layout.addWidget(QLabel(f"{name}:"), i, 0)
            label = QLabel("0")
            label.setAlignment(Qt.AlignRight)
            self.metric_labels[attr] = label
            metrics_layout.addWidget(label, i, 1)
        
        right_layout.addWidget(metrics_group)
        
        # Create controls group
        controls_group = QGroupBox("Animation Controls")
        controls_layout = QGridLayout(controls_group)
        
        # Node complexity controls
        controls_layout.addWidget(QLabel("Node Complexity:"), 0, 0)
        self.complexity_slider = QSlider(Qt.Horizontal)
        self.complexity_slider.setRange(0, 100)
        self.complexity_slider.setValue(0)
        self.complexity_slider.valueChanged.connect(self._update_complexity)
        controls_layout.addWidget(self.complexity_slider, 0, 1)
        
        controls_layout.addWidget(QLabel("Complexity Transition Speed:"), 1, 0)
        self.complexity_speed_slider = QSlider(Qt.Horizontal)
        self.complexity_speed_slider.setRange(1, 100)
        self.complexity_speed_slider.setValue(10)
        self.complexity_speed_slider.valueChanged.connect(self._update_complexity_speed)
        controls_layout.addWidget(self.complexity_speed_slider, 1, 1)
        
        # Animation speed control
        controls_layout.addWidget(QLabel("Animation Speed:"), 2, 0)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(10, 100)
        self.speed_slider.setValue(20)
        self.speed_slider.valueChanged.connect(self._update_animation_speed)
        controls_layout.addWidget(self.speed_slider, 2, 1)
        
        # Signal speed control
        controls_layout.addWidget(QLabel("Signal Speed:"), 3, 0)
        self.signal_speed_slider = QSlider(Qt.Horizontal)
        self.signal_speed_slider.setRange(10, 200)
        self.signal_speed_slider.setValue(50)
        self.signal_speed_slider.valueChanged.connect(self._update_signal_speed)
        controls_layout.addWidget(self.signal_speed_slider, 3, 1)
        
        # Signal frequency control
        controls_layout.addWidget(QLabel("Signal Frequency:"), 4, 0)
        self.signal_freq_slider = QSlider(Qt.Horizontal)
        self.signal_freq_slider.setRange(1, 100)
        self.signal_freq_slider.setValue(10)
        self.signal_freq_slider.valueChanged.connect(self._update_signal_frequency)
        controls_layout.addWidget(self.signal_freq_slider, 4, 1)
        
        # Bidirectional signal controls
        controls_layout.addWidget(QLabel("Reverse Signal Speed:"), 5, 0)
        self.reverse_speed_slider = QSlider(Qt.Horizontal)
        self.reverse_speed_slider.setRange(10, 100)
        self.reverse_speed_slider.setValue(80)
        self.reverse_speed_slider.valueChanged.connect(self._update_reverse_speed)
        controls_layout.addWidget(self.reverse_speed_slider, 5, 1)
        
        controls_layout.addWidget(QLabel("Reverse Signal Strength:"), 6, 0)
        self.reverse_strength_slider = QSlider(Qt.Horizontal)
        self.reverse_strength_slider.setRange(10, 100)
        self.reverse_strength_slider.setValue(70)
        self.reverse_strength_slider.valueChanged.connect(self._update_reverse_strength)
        controls_layout.addWidget(self.reverse_strength_slider, 6, 1)
        
        # Diagonal signal controls
        controls_layout.addWidget(QLabel("Diagonal Signal Speed:"), 7, 0)
        self.diagonal_speed_slider = QSlider(Qt.Horizontal)
        self.diagonal_speed_slider.setRange(10, 200)
        self.diagonal_speed_slider.setValue(120)
        self.diagonal_speed_slider.valueChanged.connect(self._update_diagonal_speed)
        controls_layout.addWidget(self.diagonal_speed_slider, 7, 1)
        
        controls_layout.addWidget(QLabel("Diagonal Signal Strength:"), 8, 0)
        self.diagonal_strength_slider = QSlider(Qt.Horizontal)
        self.diagonal_strength_slider.setRange(10, 100)
        self.diagonal_strength_slider.setValue(80)
        self.diagonal_strength_slider.valueChanged.connect(self._update_diagonal_strength)
        controls_layout.addWidget(self.diagonal_strength_slider, 8, 1)
        
        right_layout.addWidget(controls_group)
        
        # Initialize with test parameters
        params = {
            "num_layers": 1,
            "nodes_per_layer": 1,
            "min_nodes": 1,
            "max_nodes": 300,
            "initial_complexity": 0.0,
            "complexity_transition_speed": 0.1,
            "animation_speed": 1.0,
            "node_oscillation_speed": 0.5,
            "connection_oscillation_speed": 0.25,
            "signal_speed": 0.5,
            "signal_frequency": 0.1,
            "bidirectional_enabled": True,
            "reverse_signal_probability": 0.3,
            "reverse_signal_speed_multiplier": 0.8,
            "reverse_signal_strength_multiplier": 0.7,
            "diagonal_signal_enabled": True,
            "diagonal_signal_probability": 0.3,
            "diagonal_signal_speed_multiplier": 1.2,
            "diagonal_signal_strength_multiplier": 0.8
        }
        self.network_widget.initialize(params)
        
        # Start animation
        self.network_widget.start_animation()
        
        # Set up metrics update timer
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self._update_metrics)
        self.metrics_timer.start(100)  # Update every 100ms
        
    def _update_metrics(self):
        """Update all metric displays."""
        try:
            # Update method-based metrics
            self.metric_labels["get_current_node_count"].setText(str(self.network_widget.get_current_node_count()))
            self.metric_labels["get_connection_count"].setText(str(self.network_widget.get_connection_count()))
            self.metric_labels["get_current_complexity"].setText(f"{self.network_widget.get_current_complexity():.3f}")
            
            # Update attribute-based metrics
            for attr in ["animation_speed", "signal_speed", "signal_frequency", 
                        "node_oscillation_speed", "connection_oscillation_speed",
                        "reverse_signal_speed_multiplier", "reverse_signal_strength_multiplier",
                        "diagonal_signal_speed_multiplier", "diagonal_signal_strength_multiplier"]:
                value = getattr(self.network_widget, attr)
                self.metric_labels[attr].setText(f"{value:.3f}")
                
        except Exception as e:
            print(f"Error updating metrics: {str(e)}")
            
    def _update_complexity(self, value: int):
        """Update node complexity based on slider value."""
        complexity = value / 100.0
        self.network_widget.set_node_complexity(complexity)
        
    def _update_complexity_speed(self, value: int):
        """Update complexity transition speed based on slider value."""
        speed = value / 100.0
        self.network_widget.set_complexity_transition_speed(speed)
        
    def _update_animation_speed(self, value: int):
        """Update animation speed based on slider value."""
        speed = value / 20.0
        self.network_widget.set_animation_speed(speed)
        
    def _update_signal_speed(self, value: int):
        """Update signal speed based on slider value."""
        speed = value / 100.0
        self.network_widget.set_signal_speed(speed)
        
    def _update_signal_frequency(self, value: int):
        """Update signal frequency based on slider value."""
        frequency = value / 100.0
        self.network_widget.set_signal_frequency(frequency)

    def _update_reverse_speed(self, value: int):
        """Update reverse signal speed multiplier based on slider value."""
        multiplier = value / 100.0
        self.network_widget.set_reverse_signal_speed_multiplier(multiplier)
        
    def _update_reverse_strength(self, value: int):
        """Update reverse signal strength multiplier based on slider value."""
        multiplier = value / 100.0
        self.network_widget.set_reverse_signal_strength_multiplier(multiplier)

    def _update_diagonal_speed(self, value: int):
        """Update diagonal signal speed multiplier based on slider value."""
        multiplier = value / 100.0
        self.network_widget.set_diagonal_signal_speed_multiplier(multiplier)
        
    def _update_diagonal_strength(self, value: int):
        """Update diagonal signal strength multiplier based on slider value."""
        multiplier = value / 100.0
        self.network_widget.set_diagonal_signal_strength_multiplier(multiplier)

def main():
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 