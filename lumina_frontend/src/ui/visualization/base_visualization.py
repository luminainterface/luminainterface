"""
Base visualization class for all visualizations in the Lumina Frontend.
Provides common functionality and interface for all visualization components.
"""

from typing import Dict, List, Optional, Tuple
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QSlider, QCheckBox, QLabel
)
from PySide6.QtCore import Qt, Signal, QTimer
import pyqtgraph as pg
import numpy as np

class BaseVisualization(QWidget):
    """Base class for all visualization components."""
    
    # Signal emitted when new data is available
    data_updated = Signal(dict)
    
    def __init__(self, title: str, parent: Optional[QWidget] = None, mini: bool = False):
        """Initialize the base visualization.
        
        Args:
            title: Title of the visualization
            parent: Parent widget
            mini: Whether this is a mini version
        """
        super().__init__(parent)
        
        # Initialize properties
        self.title = title
        self.mini = mini
        self.paused = False
        self.auto_scale = True
        self.time_range = 60  # Default 60 seconds
        
        # Setup UI
        self._setup_ui()
        
        # Initialize data
        self._initialize_data()
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(1000)  # Update every second
        
    def _setup_ui(self) -> None:
        """Setup the user interface components."""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(5)
        
        # Header layout
        header_layout = QHBoxLayout()
        
        # Title label
        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(self.title_label)
        
        # Controls
        self.auto_scale_check = QCheckBox("Auto Scale")
        self.auto_scale_check.setChecked(True)
        self.auto_scale_check.stateChanged.connect(self._on_auto_scale_changed)
        header_layout.addWidget(self.auto_scale_check)
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self._on_pause_clicked)
        header_layout.addWidget(self.pause_button)
        
        self.time_range_slider = QSlider(Qt.Horizontal)
        self.time_range_slider.setRange(10, 300)  # 10 to 300 seconds
        self.time_range_slider.setValue(60)
        self.time_range_slider.valueChanged.connect(self._on_time_range_changed)
        header_layout.addWidget(self.time_range_slider)
        
        self.layout.addLayout(header_layout)
        
        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True)
        self.layout.addWidget(self.plot_widget)
        
        # Mini plot if enabled
        if not self.mini:
            self.mini_plot = pg.PlotWidget()
            self.mini_plot.setBackground('w')
            self.mini_plot.showGrid(x=True, y=True)
            self.layout.addWidget(self.mini_plot)
            
    def _initialize_data(self) -> None:
        """Initialize data arrays and structures."""
        self.time_data = np.zeros(self.time_range)
        self.current_data: Dict[str, np.ndarray] = {}
        
    def _on_auto_scale_changed(self, state: int) -> None:
        """Handle auto-scale checkbox state change."""
        self.auto_scale = bool(state)
        if self.auto_scale:
            self.plot_widget.enableAutoRange()
            
    def _on_pause_clicked(self) -> None:
        """Handle pause button click."""
        self.paused = not self.paused
        self.pause_button.setText("Resume" if self.paused else "Pause")
        
    def _on_time_range_changed(self, value: int) -> None:
        """Handle time range slider value change."""
        self.time_range = value
        self._initialize_data()
        
    def update(self) -> None:
        """Update the visualization with new data."""
        if not self.paused:
            self._update_plot()
            
    def _update_plot(self) -> None:
        """Update the plot with new data. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _update_plot")
        
    def resizeEvent(self, event) -> None:
        """Handle resize events."""
        super().resizeEvent(event)
        if not self.mini:
            # Adjust plot sizes
            height = self.height() // 2
            self.plot_widget.setFixedHeight(height)
            self.mini_plot.setFixedHeight(height)
            
    def get_data(self) -> Dict[str, np.ndarray]:
        """Get the current data.
        
        Returns:
            Dictionary of data arrays
        """
        return self.current_data.copy()
        
    def set_data(self, data: Dict[str, np.ndarray]) -> None:
        """Set new data.
        
        Args:
            data: Dictionary of data arrays
        """
        self.current_data = data.copy()
        self.data_updated.emit(data) 