#!/usr/bin/env python3
"""
Enhanced Language System - Visualization Utilities

This module provides visualization tools for the Enhanced Language System
using PySide6 and Matplotlib. It handles:
1. Semantic network visualization
2. Language pattern graphs
3. Consciousness level tracking
4. LLM weight effects visualization
"""

import logging
import math
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visualization_utils")

try:
    from PySide6.QtCore import Qt, QRectF
    from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QPainterPath, QFont
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem
    HAS_PYSIDE6 = True
    logger.info("Successfully imported PySide6 for visualization")
except ImportError:
    HAS_PYSIDE6 = False
    logger.error("PySide6 not found for visualization utilities")

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import numpy as np
    HAS_MATPLOTLIB = True
    logger.info("Successfully imported Matplotlib for visualization")
except ImportError:
    HAS_MATPLOTLIB = False
    logger.error("Matplotlib not found for visualization utilities")

try:
    import networkx as nx
    HAS_NETWORKX = True
    logger.info("Successfully imported NetworkX for graph visualization")
except ImportError:
    HAS_NETWORKX = False
    logger.error("NetworkX not found for graph visualization")


class SemanticNetworkWidget(QWidget):
    """Widget to display a semantic network using Qt graphics"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.network_data = None
        self.node_items = {}
        self.edge_items = []
        self.min_strength = 0.1
    
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create graphics view
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        
        layout.addWidget(self.view)
    
    def set_network_data(self, network_data, min_strength=0.1):
        """Set the semantic network data to display"""
        self.network_data = network_data
        self.min_strength = min_strength
        self.update_visualization()
    
    def update_visualization(self):
        """Update the network visualization"""
        if not self.network_data:
            return
        
        # Clear previous visualization
        self.scene.clear()
        self.node_items = {}
        self.edge_items = []
        
        # Get base word and connections
        base_word = self.network_data.get('word', '')
        connections = self.network_data.get('connections', [])
        
        if not base_word or not connections:
            return
        
        # Filter connections by minimum strength
        filtered_connections = [conn for conn in connections 
                               if conn.get('strength', 0) >= self.min_strength]
        
        # Calculate layout
        center_x, center_y = 0, 0
        radius = 200
        
        # Create base word node at center
        base_node = self._create_node_item(base_word, center_x, center_y, is_base=True)
        self.scene.addItem(base_node)
        self.node_items[base_word] = base_node
        
        # Create connected nodes in a circle
        node_count = len(filtered_connections)
        
        for i, connection in enumerate(filtered_connections):
            word = connection.get('word', '')
            strength = connection.get('strength', 0)
            
            # Calculate position in a circle
            angle = 2 * math.pi * i / node_count
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # Create node
            node = self._create_node_item(word, x, y, strength=strength)
            self.scene.addItem(node)
            self.node_items[word] = node
            
            # Create edge
            edge = self._create_edge_item(base_node, node, strength)
            self.scene.addItem(edge)
            self.edge_items.append(edge)
        
        # Fit the view to show all items
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
    
    def _create_node_item(self, word, x, y, is_base=False, strength=1.0):
        """Create a node item for the graph"""
        # Determine node size based on whether it's the base word
        radius = 30 if is_base else 20 * min(1.0, strength)
        
        # Determine node color based on strength
        if is_base:
            color = QColor(255, 100, 100)  # Red for base word
        else:
            # Gradient from yellow to green based on strength
            green = min(255, int(strength * 255))
            color = QColor(255, green, 100)
        
        # Create node
        node = QGraphicsEllipseItem(-radius, -radius, 2*radius, 2*radius)
        node.setPen(QPen(Qt.black, 1))
        node.setBrush(QBrush(color))
        node.setPos(x, y)
        node.setZValue(1)
        
        # Add text label
        text = QGraphicsTextItem(word)
        text.setPos(x - text.boundingRect().width() / 2, 
                   y - text.boundingRect().height() / 2)
        text.setZValue(2)
        self.scene.addItem(text)
        
        return node
    
    def _create_edge_item(self, source_node, target_node, strength=1.0):
        """Create an edge between two nodes"""
        # Get center points
        source_pos = source_node.pos()
        target_pos = target_node.pos()
        
        # Create edge
        edge = QGraphicsLineItem(source_pos.x(), source_pos.y(), 
                               target_pos.x(), target_pos.y())
        
        # Set edge appearance based on strength
        pen_width = max(1, min(5, strength * 5))
        pen = QPen(Qt.darkGray, pen_width)
        edge.setPen(pen)
        edge.setZValue(0)
        
        return edge
    
    def resizeEvent(self, event):
        """Handle resize events to fit the view"""
        super().resizeEvent(event)
        if self.scene.items():
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)


class NetworkXVisualizer(QWidget):
    """NetworkX-based visualization for semantic networks"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.network_data = None
    
    def init_ui(self):
        """Initialize the UI components"""
        if not HAS_MATPLOTLIB or not HAS_NETWORKX:
            # Create placeholder widget with error message
            layout = QVBoxLayout(self)
            msg = "NetworkX or Matplotlib not available. Please install with: pip install networkx matplotlib"
            layout.addWidget(QWidget())
            logger.error(msg)
            return
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        layout.addWidget(self.canvas)
    
    def set_network_data(self, network_data, min_strength=0.1):
        """Set the semantic network data to display"""
        if not HAS_MATPLOTLIB or not HAS_NETWORKX:
            return
            
        self.network_data = network_data
        self.update_visualization(min_strength)
    
    def update_visualization(self, min_strength=0.1):
        """Update the network visualization using NetworkX"""
        if not HAS_MATPLOTLIB or not HAS_NETWORKX or not self.network_data:
            return
        
        # Get base word and connections
        base_word = self.network_data.get('word', '')
        connections = self.network_data.get('connections', [])
        
        if not base_word or not connections:
            return
        
        # Create graph
        G = nx.Graph()
        
        # Add base node
        G.add_node(base_word, type='base')
        
        # Add connections
        for conn in connections:
            word = conn.get('word', '')
            strength = conn.get('strength', 0)
            
            if strength >= min_strength:
                G.add_node(word, type='connection')
                G.add_edge(base_word, word, weight=strength)
        
        # Clear figure
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Calculate positions using spring layout
        pos = nx.spring_layout(G, seed=42, k=0.3)
        
        # Create node colors based on type
        node_colors = ['red' if G.nodes[n].get('type') == 'base' else 'lightblue' 
                      for n in G.nodes()]
        
        # Create edge widths based on weight
        edge_widths = [G[u][v].get('weight', 1) * 2 for u, v in G.edges()]
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax, node_size=500)
        nx.draw_networkx_edges(G, pos, width=edge_widths, ax=ax, alpha=0.7)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        
        ax.set_title(f"Semantic Network for '{base_word}'")
        ax.axis('off')
        
        # Update canvas
        self.figure.tight_layout()
        self.canvas.draw()


class ConsciousnessLevelChart(QWidget):
    """Widget to display consciousness level over time"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.consciousness_data = []
        self.max_data_points = 100  # Maximum number of data points to show
    
    def init_ui(self):
        """Initialize the UI components"""
        if not HAS_MATPLOTLIB:
            # Create placeholder widget with error message
            layout = QVBoxLayout(self)
            msg = "Matplotlib not available. Please install with: pip install matplotlib"
            layout.addWidget(QWidget())
            logger.error(msg)
            return
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        layout.addWidget(self.canvas)
        
        # Create initial plot
        self.update_plot()
    
    def add_data_point(self, consciousness_level, timestamp=None):
        """Add a new consciousness level data point"""
        if not HAS_MATPLOTLIB:
            return
            
        if timestamp is None:
            timestamp = datetime.now()
            
        self.consciousness_data.append((timestamp, consciousness_level))
        
        # Trim data if exceeding maximum
        if len(self.consciousness_data) > self.max_data_points:
            self.consciousness_data = self.consciousness_data[-self.max_data_points:]
            
        self.update_plot()
    
    def update_plot(self):
        """Update the consciousness level plot"""
        if not HAS_MATPLOTLIB or not self.consciousness_data:
            return
        
        # Clear figure
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Extract data
        timestamps, levels = zip(*self.consciousness_data) if self.consciousness_data else ([], [])
        
        # Plot data
        ax.plot(timestamps, levels, marker='o', linestyle='-', color='purple', markersize=3)
        
        # Format plot
        ax.set_title("Consciousness Level Over Time")
        ax.set_ylabel("Consciousness Level")
        ax.set_ylim(0, max(0.2, max(levels) * 1.2) if levels else 0.2)
        
        # Format x-axis to show times
        self.figure.autofmt_xdate()
        
        # Update canvas
        self.figure.tight_layout()
        self.canvas.draw()


class LLMWeightEffectsChart(QWidget):
    """Widget to display effects of different LLM weights"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.weight_data = {}  # Format: {weight: {score_type: score}}
    
    def init_ui(self):
        """Initialize the UI components"""
        if not HAS_MATPLOTLIB:
            # Create placeholder widget with error message
            layout = QVBoxLayout(self)
            msg = "Matplotlib not available. Please install with: pip install matplotlib"
            layout.addWidget(QWidget())
            logger.error(msg)
            return
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        layout.addWidget(self.canvas)
        
        # Create initial plot
        self.update_plot()
    
    def add_data_point(self, llm_weight, scores):
        """
        Add a data point for a specific LLM weight
        
        Args:
            llm_weight: The LLM weight value (0.0-1.0)
            scores: Dictionary of scores keyed by score type
                    (e.g., {'unified': 0.54, 'neural': 0.62, 'consciousness': 0.21})
        """
        if not HAS_MATPLOTLIB:
            return
            
        self.weight_data[llm_weight] = scores
        self.update_plot()
    
    def update_plot(self):
        """Update the LLM weight effects plot"""
        if not HAS_MATPLOTLIB or not self.weight_data:
            return
        
        # Clear figure
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Extract data
        weights = sorted(self.weight_data.keys())
        
        # Get score types (unified, neural, consciousness, etc.)
        all_score_types = set()
        for scores in self.weight_data.values():
            all_score_types.update(scores.keys())
        
        # Plot each score type
        for score_type in sorted(all_score_types):
            scores = [self.weight_data[w].get(score_type, 0) for w in weights]
            ax.plot(weights, scores, marker='o', linestyle='-', label=score_type.capitalize())
        
        # Format plot
        ax.set_title("Effect of LLM Weight on Scores")
        ax.set_xlabel("LLM Weight")
        ax.set_ylabel("Score Value")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add exact weights as ticks
        ax.set_xticks(weights)
        
        # Update canvas
        self.figure.tight_layout()
        self.canvas.draw()


# Example usage function
def example_usage():
    """Show example usage of the visualization utilities"""
    from PySide6.QtWidgets import QApplication
    import sys
    
    # Sample semantic network data
    sample_network = {
        'word': 'language',
        'connections': [
            {'word': 'learning', 'strength': 0.9},
            {'word': 'memory', 'strength': 0.8},
            {'word': 'communication', 'strength': 0.75},
            {'word': 'neural', 'strength': 0.7},
            {'word': 'processing', 'strength': 0.65},
            {'word': 'understanding', 'strength': 0.6},
            {'word': 'consciousness', 'strength': 0.55},
            {'word': 'thought', 'strength': 0.5},
            {'word': 'symbol', 'strength': 0.45},
            {'word': 'meaning', 'strength': 0.4}
        ]
    }
    
    # Sample consciousness data
    sample_consciousness_data = [
        (datetime(2023, 1, 1, 12, 0), 0.05),
        (datetime(2023, 1, 1, 12, 10), 0.07),
        (datetime(2023, 1, 1, 12, 20), 0.10),
        (datetime(2023, 1, 1, 12, 30), 0.12),
        (datetime(2023, 1, 1, 12, 40), 0.15),
        (datetime(2023, 1, 1, 12, 50), 0.13),
        (datetime(2023, 1, 1, 13, 0), 0.14)
    ]
    
    # Sample LLM weight data
    sample_weight_data = {
        0.0: {'unified': 0.54, 'neural': 0.62, 'consciousness': 0.21},
        0.2: {'unified': 0.59, 'neural': 0.57, 'consciousness': 0.22},
        0.5: {'unified': 0.52, 'neural': 0.55, 'consciousness': 0.25},
        0.8: {'unified': 0.47, 'neural': 0.49, 'consciousness': 0.30},
        1.0: {'unified': 0.46, 'neural': 0.42, 'consciousness': 0.35}
    }
    
    app = QApplication(sys.argv)
    
    # Test SemanticNetworkWidget
    semantic_widget = SemanticNetworkWidget()
    semantic_widget.set_network_data(sample_network)
    semantic_widget.setWindowTitle("Semantic Network (Qt)")
    semantic_widget.resize(600, 500)
    semantic_widget.show()
    
    # Test NetworkXVisualizer
    if HAS_NETWORKX:
        networkx_widget = NetworkXVisualizer()
        networkx_widget.set_network_data(sample_network)
        networkx_widget.setWindowTitle("Semantic Network (NetworkX)")
        networkx_widget.resize(600, 500)
        networkx_widget.show()
    
    # Test ConsciousnessLevelChart
    if HAS_MATPLOTLIB:
        consciousness_widget = ConsciousnessLevelChart()
        for timestamp, level in sample_consciousness_data:
            consciousness_widget.add_data_point(level, timestamp)
        consciousness_widget.setWindowTitle("Consciousness Level Chart")
        consciousness_widget.resize(600, 300)
        consciousness_widget.show()
    
    # Test LLMWeightEffectsChart
    if HAS_MATPLOTLIB:
        llm_weight_widget = LLMWeightEffectsChart()
        for weight, scores in sample_weight_data.items():
            llm_weight_widget.add_data_point(weight, scores)
        llm_weight_widget.setWindowTitle("LLM Weight Effects")
        llm_weight_widget.resize(600, 400)
        llm_weight_widget.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    example_usage() 