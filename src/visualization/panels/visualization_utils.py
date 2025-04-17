#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Utilities for LUMINA V7 Dashboard
===============================================

Common visualization functions and utilities used across different panels.
"""

import os
import numpy as np
import colorsys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VisualizationUtils")

# Try to determine which Qt framework is being used
try:
    from PySide6 import QtWidgets
    QT_FRAMEWORK = "PySide6"
    logger.info("Using PySide6 for visualization utilities")
except ImportError:
    try:
        from PyQt5 import QtWidgets
        QT_FRAMEWORK = "PyQt5"
        logger.info("Using PyQt5 for visualization utilities")
    except ImportError:
        QT_FRAMEWORK = None
        logger.warning("No Qt framework found, some visualization features may be limited")

# Try to import PyQtGraph for advanced visualizations
try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
    logger.info("PyQtGraph successfully imported")
except ImportError:
    HAS_PYQTGRAPH = False
    logger.warning("PyQtGraph not available, falling back to Matplotlib")
    
# For matplotlib fallback
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Color schemes for consistent visualization
LUMINA_COLORS = {
    "neural": "#4b7bec",     # Blue for neural activity
    "language": "#26de81",   # Green for language processing
    "dream": "#a55eea",      # Purple for dream mode
    "memory": "#fd9644",     # Orange for memory system
    "system": "#fc5c65",     # Red for system resources
    "breath": "#2bcbba",     # Teal for breath detection
    "background": "#f5f6fa", # Light gray background
    "text": "#2f3542",       # Dark gray text
    "warning": "#fed330",    # Yellow for warnings
    "error": "#eb3b5a",      # Red for errors
}

class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas for embedding in Qt"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        Initialize matplotlib canvas
        
        Args:
            parent: Parent widget
            width: Figure width in inches
            height: Figure height in inches
            dpi: Dots per inch
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        
        # Set style
        self.fig.patch.set_facecolor(LUMINA_COLORS["background"])
        self.axes.set_facecolor(LUMINA_COLORS["background"])
        self.axes.tick_params(colors=LUMINA_COLORS["text"])
        for spine in self.axes.spines.values():
            spine.set_edgecolor(LUMINA_COLORS["text"])

# Only define PyQtGraphWidget if PyQtGraph is available
if HAS_PYQTGRAPH:
    class PyQtGraphWidget(pg.PlotWidget):
        """PyQtGraph widget for advanced interactive plots"""
        
        def __init__(self, parent=None, title=None, background=None):
            """
            Initialize PyQtGraph widget
            
            Args:
                parent: Parent widget
                title: Plot title
                background: Background color (defaults to LUMINA_COLORS["background"])
            """
            super().__init__(parent=parent)
            
            # Set background color
            background = background or LUMINA_COLORS["background"]
            self.setBackground(background)
            
            # Configure plot
            self.showGrid(x=True, y=True)
            self.plotItem.addLegend()
            
            # Set title if provided
            if title:
                self.setTitle(title, color=LUMINA_COLORS["text"])
                
            # Store plot items
            self.plot_items = {}
        
        def update_plot(self, name, x_data, y_data, color=None, width=2):
            """
            Update plot with new data
            
            Args:
                name: Plot name for legend
                x_data: X-axis data
                y_data: Y-axis data
                color: Line color
                width: Line width
            """
            if name in self.plot_items:
                # Update existing plot
                self.plot_items[name].setData(x_data, y_data)
            else:
                # Create new plot
                color = color or LUMINA_COLORS.get(name, LUMINA_COLORS["neural"])
                pen = pg.mkPen(color=color, width=width)
                self.plot_items[name] = self.plot(x_data, y_data, name=name, pen=pen)

def generate_color_gradient(n, start_color, end_color):
    """
    Generate a gradient of n colors between start_color and end_color
    
    Args:
        n: Number of colors to generate
        start_color: Start color in hex format (#RRGGBB)
        end_color: End color in hex format (#RRGGBB)
        
    Returns:
        List of n colors in hex format
    """
    # Convert hex to RGB
    start_rgb = tuple(int(start_color[i:i+2], 16) / 255.0 for i in (1, 3, 5))
    end_rgb = tuple(int(end_color[i:i+2], 16) / 255.0 for i in (1, 3, 5))
    
    # Convert RGB to HSV
    start_hsv = colorsys.rgb_to_hsv(*start_rgb)
    end_hsv = colorsys.rgb_to_hsv(*end_rgb)
    
    # Generate colors in HSV space
    colors = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0
        h = start_hsv[0] + t * (end_hsv[0] - start_hsv[0])
        s = start_hsv[1] + t * (end_hsv[1] - start_hsv[1])
        v = start_hsv[2] + t * (end_hsv[2] - start_hsv[2])
        
        # Convert back to RGB and then to hex
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        hex_color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        colors.append(hex_color)
        
    return colors

def smooth_data(data, window_size=5):
    """
    Apply a simple moving average to smooth the data
    
    Args:
        data: Data to smooth
        window_size: Window size for moving average
        
    Returns:
        Smoothed data
    """
    if len(data) < window_size:
        return data
        
    # Pad the data to handle edges
    padded_data = np.pad(data, (window_size//2, window_size//2), mode='edge')
    
    # Apply moving average
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(padded_data, kernel, mode='valid')
    
    return smoothed

def format_value(value, precision=2):
    """
    Format a numeric value for display
    
    Args:
        value: Value to format
        precision: Decimal precision
        
    Returns:
        Formatted string
    """
    if value is None:
        return "N/A"
        
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
        
    return str(value)

def create_mock_time_series(length=100, trend="random", noise_level=0.1):
    """
    Create mock time series data for testing
    
    Args:
        length: Number of data points
        trend: Trend type ("random", "increasing", "decreasing", "sine")
        noise_level: Amount of noise to add
        
    Returns:
        Tuple of (x_data, y_data)
    """
    x_data = np.arange(length)
    
    if trend == "random":
        y_data = np.random.rand(length)
    elif trend == "increasing":
        y_data = np.linspace(0.1, 0.9, length)
    elif trend == "decreasing":
        y_data = np.linspace(0.9, 0.1, length)
    elif trend == "sine":
        y_data = 0.5 + 0.4 * np.sin(np.linspace(0, 4*np.pi, length))
    else:
        y_data = np.random.rand(length)
        
    # Add noise
    if noise_level > 0:
        y_data += noise_level * (np.random.rand(length) - 0.5)
        
    # Ensure values are in [0, 1]
    y_data = np.clip(y_data, 0, 1)
    
    return x_data, y_data 