#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Visualization Panel for LUMINA V7 Dashboard
=================================================

Visualizes memory patterns, associations, and statistics from the neural memory system.
"""

import os
import sys
import time
import random
import logging
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

# Import base panel
from src.visualization.panels.base_panel import BasePanel, QT_FRAMEWORK

# Qt compatibility layer
if QT_FRAMEWORK == "PySide6":
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout, 
        QFrame, QTabWidget, QPushButton, QComboBox, QSpinBox, QTableWidget,
        QTableWidgetItem, QHeaderView, QSplitter
    )
    from PySide6.QtCore import Qt, Signal, Slot, QTimer
    from PySide6.QtGui import QFont, QPainter, QColor, QPen, QBrush, QLinearGradient
else:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout, 
        QFrame, QTabWidget, QPushButton, QComboBox, QSpinBox, QTableWidget,
        QTableWidgetItem, QHeaderView, QSplitter
    )
    from PyQt5.QtCore import Qt, pyqtSignal as Signal, pyqtSlot as Slot, QTimer
    from PyQt5.QtGui import QFont, QPainter, QColor, QPen, QBrush, QLinearGradient

# Try to import PyQtGraph for enhanced visualizations
try:
    import pyqtgraph as pg
    import numpy as np
    HAVE_PYQTGRAPH = True
    
    # Configure PyQtGraph for dark theme
    pg.setConfigOption('background', (40, 44, 52))
    pg.setConfigOption('foreground', (255, 255, 255))
except ImportError:
    HAVE_PYQTGRAPH = False

# Setup logging
logger = logging.getLogger("LuminaPanels.MemoryVisualization")

class MemoryVisualizationPanel(BasePanel):
    """Panel for visualizing neural memory patterns and associations"""
    
    def __init__(self, parent=None, db_path=None, refresh_rate=2000, active=True, gui_framework=None):
        """
        Initialize the memory visualization panel
        
        Args:
            parent: Parent widget
            db_path: Path to metrics database
            refresh_rate: Data refresh rate in milliseconds
            active: Whether panel is active at startup
            gui_framework: GUI framework to use
        """
        super().__init__(
            parent=parent, 
            panel_name="Memory Visualization", 
            db_path=db_path, 
            refresh_rate=refresh_rate, 
            active=active, 
            gui_framework=gui_framework
        )
        
        # Memory visualization parameters
        self.memory_stats = {}
        self.recent_memories = []
        self.memory_categories = ["language", "visual", "conceptual", "procedural"]
        self.selected_category = "language"
        self.memory_network = {}
        self.strongest_connections = []
        
        # Colors for memory types
        self.memory_colors = {
            "language": QColor(52, 152, 219),     # Blue
            "visual": QColor(46, 204, 113),       # Green
            "conceptual": QColor(155, 89, 182),   # Purple
            "procedural": QColor(230, 126, 34),   # Orange
            "default": QColor(149, 165, 166)      # Gray
        }
        
        # Set up the UI components
        self._setup_memory_ui()
        
        # Connect signals
        self.refresh_button.clicked.connect(self.refresh_data)
        self.category_combo.currentTextChanged.connect(self._update_selected_category)
        self.visualization_combo.currentIndexChanged.connect(self._update_visualization_mode)
        
        # Initial refresh
        self.refresh_data()
    
    def _setup_memory_ui(self):
        """Set up UI components specific to memory visualization"""
        # Create tab widget for different memory visualizations
        self.tabs = QTabWidget()
        
        # Overview tab
        self.overview_tab = QWidget()
        self.overview_layout = QVBoxLayout(self.overview_tab)
        
        # Memory stats section
        self.stats_frame = QFrame()
        self.stats_frame.setFrameShape(QFrame.StyledPanel)
        self.stats_frame.setStyleSheet("background-color: rgba(60, 65, 75, 150); border-radius: 4px;")
        self.stats_layout = QGridLayout(self.stats_frame)
        
        # Add stats labels
        self.stats_labels = {}
        stats_fields = [
            "Total Memories", "Active Memories", "Memory Capacity", 
            "Retrieval Speed", "Association Strength", "Memory Coherence"
        ]
        
        for i, field in enumerate(stats_fields):
            label = QLabel(f"{field}:")
            value = QLabel("0")
            value.setStyleSheet("color: #3498db; font-weight: bold;")
            self.stats_layout.addWidget(label, i % 3, (i // 3) * 2)
            self.stats_layout.addWidget(value, i % 3, (i // 3) * 2 + 1)
            self.stats_labels[field] = value
        
        self.overview_layout.addWidget(self.stats_frame)
        
        # Memory timeline/history
        self.memory_history_frame = QFrame()
        self.memory_history_frame.setFrameShape(QFrame.StyledPanel)
        self.memory_history_frame.setMinimumHeight(150)
        self.memory_history_frame.setStyleSheet("background-color: rgba(60, 65, 75, 150); border-radius: 4px;")
        self.memory_history_layout = QVBoxLayout(self.memory_history_frame)
        
        # Add title
        history_title = QLabel("Recent Memory Activity")
        history_title.setStyleSheet("font-weight: bold; color: white;")
        self.memory_history_layout.addWidget(history_title)
        
        # Memory table
        self.memory_table = QTableWidget(0, 4)
        self.memory_table.setHorizontalHeaderLabels(["Time", "Type", "Content", "Strength"])
        self.memory_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.memory_table.setAlternatingRowColors(True)
        self.memory_table.setStyleSheet("alternate-background-color: rgba(45, 50, 60, 150);")
        self.memory_history_layout.addWidget(self.memory_table)
        
        self.overview_layout.addWidget(self.memory_history_frame)
        
        # Network tab
        self.network_tab = QWidget()
        self.network_layout = QVBoxLayout(self.network_tab)
        
        # Control panel for network view
        self.network_controls = QFrame()
        self.network_controls.setFrameShape(QFrame.StyledPanel)
        self.network_controls.setStyleSheet("background-color: rgba(60, 65, 75, 150); border-radius: 4px;")
        self.network_controls_layout = QHBoxLayout(self.network_controls)
        
        # Category selector
        category_label = QLabel("Memory Category:")
        self.network_controls_layout.addWidget(category_label)
        
        self.category_combo = QComboBox()
        self.category_combo.addItems(self.memory_categories)
        self.network_controls_layout.addWidget(self.category_combo)
        
        # Visualization selector
        viz_label = QLabel("Visualization:")
        self.network_controls_layout.addWidget(viz_label)
        
        self.visualization_combo = QComboBox()
        self.visualization_combo.addItems(["Network", "Force Directed", "Heatmap", "Connections"])
        self.network_controls_layout.addWidget(self.visualization_combo)
        
        # Depth selector
        depth_label = QLabel("Connection Depth:")
        self.network_controls_layout.addWidget(depth_label)
        
        self.depth_spinner = QSpinBox()
        self.depth_spinner.setRange(1, 5)
        self.depth_spinner.setValue(2)
        self.network_controls_layout.addWidget(self.depth_spinner)
        
        # Refresh button
        self.refresh_button = QPushButton("Refresh")
        self.network_controls_layout.addWidget(self.refresh_button)
        
        self.network_layout.addWidget(self.network_controls)
        
        # Network visualization area
        self.network_view = QFrame()
        self.network_view.setFrameShape(QFrame.StyledPanel)
        self.network_view.setMinimumHeight(300)
        self.network_view.setStyleSheet("background-color: rgba(40, 44, 52, 200); border-radius: 4px;")
        
        if HAVE_PYQTGRAPH:
            # Create PyQtGraph widget for network visualization
            self.network_view_layout = QVBoxLayout(self.network_view)
            
            # Use GraphicsLayoutWidget to have a more flexible layout
            self.plot_widget = pg.GraphicsLayoutWidget()
            
            # Create plot for network visualization
            self.network_plot = self.plot_widget.addPlot(title="Memory Network")
            self.network_plot.setAspectLocked(True)
            
            # Create scatter plot item for nodes
            self.node_scatter = pg.ScatterPlotItem(
                size=15, 
                pen=pg.mkPen(None), 
                brush=pg.mkBrush(255, 255, 255, 120),
                symbol='o'
            )
            self.network_plot.addItem(self.node_scatter)
            
            # Create GraphItem for connections
            self.graph_item = pg.GraphItem()
            self.network_plot.addItem(self.graph_item)
            
            self.network_view_layout.addWidget(self.plot_widget)
        else:
            # Custom rendering will be used
            self.network_view_layout = QVBoxLayout(self.network_view)
            self.custom_network_viz = MemoryNetworkWidget()
            self.network_view_layout.addWidget(self.custom_network_viz)
            
            # Add note about missing PyQtGraph
            note = QLabel("Note: Install PyQtGraph for enhanced visualizations")
            note.setStyleSheet("color: orange;")
            self.network_view_layout.addWidget(note)
        
        self.network_layout.addWidget(self.network_view)
        
        # Connections tab
        self.connections_tab = QWidget()
        self.connections_layout = QVBoxLayout(self.connections_tab)
        
        # Strongest connections section
        self.connections_frame = QFrame()
        self.connections_frame.setFrameShape(QFrame.StyledPanel)
        self.connections_frame.setStyleSheet("background-color: rgba(60, 65, 75, 150); border-radius: 4px;")
        self.connections_layout.addWidget(self.connections_frame)
        
        self.connections_inner_layout = QVBoxLayout(self.connections_frame)
        
        # Table for strongest connections
        connections_title = QLabel("Strongest Memory Connections")
        connections_title.setStyleSheet("font-weight: bold; color: white;")
        self.connections_inner_layout.addWidget(connections_title)
        
        self.connections_table = QTableWidget(0, 4)
        self.connections_table.setHorizontalHeaderLabels(["Source", "Target", "Strength", "Type"])
        self.connections_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.connections_table.setAlternatingRowColors(True)
        self.connections_table.setStyleSheet("alternate-background-color: rgba(45, 50, 60, 150);")
        self.connections_inner_layout.addWidget(self.connections_table)
        
        # Add tabs to tab widget
        self.tabs.addTab(self.overview_tab, "Overview")
        self.tabs.addTab(self.network_tab, "Network View")
        self.tabs.addTab(self.connections_tab, "Connections")
        
        # Add tab widget to main layout
        self.layout.insertWidget(3, self.tabs)
    
    def refresh_data(self):
        """Refresh memory visualization data"""
        try:
            # Generate mock data when in mock mode or no real data available
            if self.is_mock_mode or not self.db_path:
                memory_data = self._generate_mock_memory_data()
            else:
                # Fetch real memory data from database
                memory_data = self._fetch_memory_data()
            
            # Update the UI
            self.update_signal.emit(memory_data)
            
        except Exception as e:
            logger.error(f"Error refreshing memory data: {e}")
            self.status_label.setText(f"Error: {str(e)}")
    
    def _fetch_memory_data(self):
        """Fetch real memory data from database"""
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query memory statistics
            cursor.execute("""
                SELECT metric_name, metric_value
                FROM memory_metrics
                WHERE metric_type = 'statistic'
                ORDER BY timestamp DESC
                LIMIT 20
            """)
            
            stats = {}
            for name, value in cursor.fetchall():
                stats[name] = float(value)
            
            # Query recent memories
            cursor.execute("""
                SELECT timestamp, memory_type, content, strength
                FROM memory_metrics
                WHERE metric_type = 'memory_formation'
                ORDER BY timestamp DESC
                LIMIT 50
            """)
            
            recent_memories = []
            for timestamp, memory_type, content, strength in cursor.fetchall():
                recent_memories.append({
                    "timestamp": timestamp,
                    "type": memory_type,
                    "content": content,
                    "strength": float(strength)
                })
            
            # Query memory network for selected category
            cursor.execute("""
                SELECT network_data
                FROM memory_metrics
                WHERE metric_type = 'memory_network' AND memory_type = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (self.selected_category,))
            
            row = cursor.fetchone()
            network = {}
            if row and row[0]:
                network = json.loads(row[0])
            
            # Query strongest connections
            cursor.execute("""
                SELECT source, target, strength, connection_type
                FROM memory_connections
                ORDER BY strength DESC
                LIMIT 50
            """)
            
            connections = []
            for source, target, strength, connection_type in cursor.fetchall():
                connections.append({
                    "source": source,
                    "target": target,
                    "strength": float(strength),
                    "type": connection_type
                })
            
            conn.close()
            
            return {
                "stats": stats,
                "recent_memories": recent_memories,
                "network": network,
                "connections": connections,
                "category": self.selected_category,
                "timestamp": datetime.now().isoformat()
            }
                
        except Exception as e:
            logger.error(f"Error fetching memory data: {e}")
            # Fall back to mock data on error
            return self._generate_mock_memory_data()
    
    def _generate_mock_memory_data(self):
        """Generate mock memory data for testing"""
        import random
        
        # Generate statistics
        stats = {
            "Total Memories": random.randint(5000, 50000),
            "Active Memories": random.randint(500, 5000),
            "Memory Capacity": random.randint(70, 95),
            "Retrieval Speed": random.uniform(80, 99.9),
            "Association Strength": random.uniform(0.4, 0.9),
            "Memory Coherence": random.uniform(0.5, 0.95)
        }
        
        # Generate memory types and content templates
        memory_types = {
            "language": ["Word", "Phrase", "Grammar", "Context", "Concept"],
            "visual": ["Shape", "Color", "Pattern", "Object", "Scene"],
            "conceptual": ["Relation", "Category", "Property", "Abstract", "Rule"],
            "procedural": ["Sequence", "Action", "Skill", "Routine", "Method"]
        }
        
        content_templates = {
            "language": [
                "Association between '{0}' and '{1}'",
                "Grammatical pattern for '{0}'",
                "Semantic meaning of '{0}'",
                "Usage context for '{0}'",
                "Synonym relation between '{0}' and '{1}'"
            ],
            "visual": [
                "Recognition pattern for {0}",
                "Visual features of {0}",
                "Spatial relationship between {0} and {1}",
                "Color patterns in {0}",
                "Shape classification of {0}"
            ],
            "conceptual": [
                "Categorization of {0} as {1}",
                "Abstract relation between {0} and {1}",
                "Property extraction from {0}",
                "Conceptual mapping of {0} to {1}",
                "Hierarchical position of {0} in {1}"
            ],
            "procedural": [
                "Action sequence for {0}",
                "Method for achieving {0}",
                "Skill component for {0}",
                "Optimization of {0} procedure",
                "Learning progress for {0} skill"
            ]
        }
        
        # Word pool for generating content
        words = [
            "neural", "pattern", "concept", "language", "vision", 
            "memory", "learning", "structure", "function", "system",
            "process", "module", "network", "connection", "association",
            "model", "sequence", "time", "space", "action",
            "object", "attribute", "relation", "category", "instance"
        ]
        
        # Generate recent memories
        recent_memories = []
        for i in range(20):
            memory_type = random.choice(list(memory_types.keys()))
            subtype = random.choice(memory_types[memory_type])
            
            # Generate content from template
            template = random.choice(content_templates[memory_type])
            word1 = random.choice(words)
            word2 = random.choice(words)
            content = template.format(word1, word2)
            
            # Create memory entry
            timestamp = (datetime.now() - timedelta(minutes=random.randint(0, 60))).isoformat()
            strength = random.uniform(0.3, 1.0)
            
            recent_memories.append({
                "timestamp": timestamp,
                "type": memory_type,
                "content": f"[{subtype}] {content}",
                "strength": strength
            })
        
        # Sort by timestamp (newest first)
        recent_memories.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Generate network data
        network_nodes = []
        network_edges = []
        
        # Generate nodes for the selected category
        for i in range(15):
            node_id = f"node_{i}"
            node_type = random.choice(memory_types[self.selected_category])
            word = random.choice(words)
            
            network_nodes.append({
                "id": node_id,
                "label": f"{node_type}: {word}",
                "type": node_type,
                "strength": random.uniform(0.4, 1.0),
                "age": random.randint(1, 100)
            })
        
        # Generate edges between nodes
        for i in range(25):
            if len(network_nodes) < 2:
                continue
                
            source = random.choice(network_nodes)["id"]
            target = random.choice(network_nodes)["id"]
            
            # Avoid self-loops
            while source == target:
                target = random.choice(network_nodes)["id"]
            
            network_edges.append({
                "source": source,
                "target": target,
                "strength": random.uniform(0.1, 0.9),
                "type": random.choice(["associative", "hierarchical", "sequential", "causal"])
            })
        
        network = {
            "nodes": network_nodes,
            "edges": network_edges
        }
        
        # Generate strongest connections
        connections = []
        for i in range(30):
            memory_type = random.choice(list(memory_types.keys()))
            subtype = random.choice(memory_types[memory_type])
            
            word1 = random.choice(words)
            word2 = random.choice(words)
            
            connections.append({
                "source": f"{subtype}: {word1}",
                "target": f"{subtype}: {word2}",
                "strength": random.uniform(0.5, 1.0),
                "type": random.choice(["associative", "hierarchical", "sequential", "causal"])
            })
        
        # Sort by strength (strongest first)
        connections.sort(key=lambda x: x["strength"], reverse=True)
        
        return {
            "stats": stats,
            "recent_memories": recent_memories,
            "network": network,
            "connections": connections,
            "category": self.selected_category,
            "timestamp": datetime.now().isoformat()
        }
    
    def _update_ui_from_data(self, data):
        """Update UI with memory data"""
        # Call parent implementation
        super()._update_ui_from_data(data)
        
        # Update statistics
        stats = data.get("stats", {})
        for field, value in stats.items():
            if field in self.stats_labels:
                if isinstance(value, float):
                    self.stats_labels[field].setText(f"{value:.2f}")
                else:
                    self.stats_labels[field].setText(str(value))
        
        # Update recent memories table
        recent_memories = data.get("recent_memories", [])
        self.memory_table.setRowCount(0)  # Clear existing rows
        
        for i, memory in enumerate(recent_memories):
            self.memory_table.insertRow(i)
            
            # Format timestamp to just show time
            timestamp = memory.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%H:%M:%S")
                except ValueError:
                    time_str = timestamp
            else:
                time_str = ""
            
            # Create items
            time_item = QTableWidgetItem(time_str)
            type_item = QTableWidgetItem(memory.get("type", ""))
            content_item = QTableWidgetItem(memory.get("content", ""))
            strength_item = QTableWidgetItem(f"{memory.get('strength', 0):.2f}")
            
            # Set colors based on memory type
            memory_type = memory.get("type", "default")
            if memory_type in self.memory_colors:
                color = self.memory_colors[memory_type]
                type_item.setForeground(color)
            
            # Add items to table
            self.memory_table.setItem(i, 0, time_item)
            self.memory_table.setItem(i, 1, type_item)
            self.memory_table.setItem(i, 2, content_item)
            self.memory_table.setItem(i, 3, strength_item)
        
        # Update network visualization
        network = data.get("network", {})
        self._update_network_visualization(network)
        
        # Update connections table
        connections = data.get("connections", [])
        self.connections_table.setRowCount(0)  # Clear existing rows
        
        for i, connection in enumerate(connections):
            self.connections_table.insertRow(i)
            
            # Create items
            source_item = QTableWidgetItem(connection.get("source", ""))
            target_item = QTableWidgetItem(connection.get("target", ""))
            strength_item = QTableWidgetItem(f"{connection.get('strength', 0):.2f}")
            type_item = QTableWidgetItem(connection.get("type", ""))
            
            # Add items to table
            self.connections_table.setItem(i, 0, source_item)
            self.connections_table.setItem(i, 1, target_item)
            self.connections_table.setItem(i, 2, strength_item)
            self.connections_table.setItem(i, 3, type_item)
    
    def _update_network_visualization(self, network):
        """Update network visualization with new data"""
        nodes = network.get("nodes", [])
        edges = network.get("edges", [])
        
        if HAVE_PYQTGRAPH:
            # Update PyQtGraph visualization
            
            # Create positions for nodes using a circular layout
            positions = {}
            colors = []
            sizes = []
            symbols = []
            node_data = []
            
            if nodes:
                # Create a circular layout
                import math
                radius = 200
                center = (0, 0)
                
                for i, node in enumerate(nodes):
                    angle = 2 * math.pi * i / len(nodes)
                    x = center[0] + radius * math.cos(angle)
                    y = center[1] + radius * math.sin(angle)
                    
                    node_id = node.get("id", f"node_{i}")
                    positions[node_id] = (x, y)
                    
                    # Node styling
                    strength = node.get("strength", 0.5)
                    node_type = node.get("type", "")
                    
                    # Determine color based on node type
                    if node_type in ["Word", "Phrase", "Grammar", "Context", "Concept"]:
                        color = (52, 152, 219, 200)  # Blue
                    elif node_type in ["Shape", "Color", "Pattern", "Object", "Scene"]:
                        color = (46, 204, 113, 200)  # Green
                    elif node_type in ["Relation", "Category", "Property", "Abstract", "Rule"]:
                        color = (155, 89, 182, 200)  # Purple
                    elif node_type in ["Sequence", "Action", "Skill", "Routine", "Method"]:
                        color = (230, 126, 34, 200)  # Orange
                    else:
                        color = (149, 165, 166, 200)  # Gray
                    
                    colors.append(color)
                    sizes.append(10 + strength * 20)  # Size based on strength
                    symbols.append('o')
                    
                    # Store node data
                    node_data.append({
                        'pos': (x, y),
                        'size': 10 + strength * 20,
                        'brush': pg.mkBrush(color),
                        'symbol': 'o',
                        'data': node
                    })
                
                # Update scatter plot with node data
                self.node_scatter.setData(
                    pos=np.array([d['pos'] for d in node_data]),
                    size=np.array([d['size'] for d in node_data]),
                    brush=[d['brush'] for d in node_data],
                    symbol=[d['symbol'] for d in node_data]
                )
                
                # Prepare edge data for GraphItem
                if edges:
                    adj_list = []
                    edge_pens = []
                    
                    for edge in edges:
                        source = edge.get("source", "")
                        target = edge.get("target", "")
                        strength = edge.get("strength", 0.5)
                        
                        if source in positions and target in positions:
                            source_idx = next((i for i, n in enumerate(nodes) if n.get("id") == source), None)
                            target_idx = next((i for i, n in enumerate(nodes) if n.get("id") == target), None)
                            
                            if source_idx is not None and target_idx is not None:
                                adj_list.append([source_idx, target_idx])
                                
                                # Edge styling based on strength
                                alpha = int(strength * 200) + 55
                                width = 1 + strength * 3
                                edge_pens.append(pg.mkPen((200, 200, 200, alpha), width=width))
                    
                    # Update GraphItem
                    self.graph_item.setData(
                        pos=np.array([d['pos'] for d in node_data]),
                        adj=np.array(adj_list),
                        pen=edge_pens
                    )
        else:
            # Update custom visualization
            if hasattr(self, 'custom_network_viz'):
                self.custom_network_viz.update_data(nodes, edges)
    
    def _update_selected_category(self, category):
        """Update selected memory category"""
        self.selected_category = category
        self.refresh_data()
    
    def _update_visualization_mode(self, index):
        """Update visualization mode"""
        modes = ["Network", "Force Directed", "Heatmap", "Connections"]
        mode = modes[index]
        logger.debug(f"Changed visualization mode to {mode}")
        
        # This would update the visualization type
        self.refresh_data()


class MemoryNetworkWidget(QWidget):
    """Custom widget for memory network visualization when PyQtGraph is not available"""
    
    def __init__(self, parent=None):
        """Initialize the visualization widget"""
        super().__init__(parent)
        self.nodes = []
        self.edges = []
        
        # Set minimum size
        self.setMinimumSize(400, 300)
    
    def update_data(self, nodes, edges):
        """Update with new network data"""
        self.nodes = nodes
        self.edges = edges
        self.update()  # Trigger repaint
    
    def paintEvent(self, event):
        """Paint the visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        
        # Draw background
        painter.fillRect(0, 0, width, height, QColor(40, 44, 52))
        
        # Early return if no data
        if not self.nodes:
            painter.setPen(QPen(QColor(200, 200, 200)))
            painter.drawText(width/2 - 100, height/2, "No memory network data available")
            return
        
        # Create positions for nodes using a circular layout
        positions = {}
        radius = min(width, height) * 0.4
        center_x = width / 2
        center_y = height / 2
        
        import math
        for i, node in enumerate(self.nodes):
            angle = 2 * math.pi * i / len(self.nodes)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            node_id = node.get("id", f"node_{i}")
            positions[node_id] = (x, y)
        
        # Draw edges first (so they're behind nodes)
        for edge in self.edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            strength = edge.get("strength", 0.5)
            
            if source in positions and target in positions:
                start_x, start_y = positions[source]
                end_x, end_y = positions[target]
                
                # Set edge style based on strength
                alpha = int(strength * 200) + 55
                width = 1 + int(strength * 3)
                painter.setPen(QPen(QColor(200, 200, 200, alpha), width))
                
                # Draw edge
                painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))
        
        # Draw nodes
        for node in self.nodes:
            node_id = node.get("id", "")
            if node_id in positions:
                x, y = positions[node_id]
                
                # Node styling
                strength = node.get("strength", 0.5)
                node_type = node.get("type", "")
                
                # Determine color based on node type
                if node_type in ["Word", "Phrase", "Grammar", "Context", "Concept"]:
                    color = QColor(52, 152, 219, 200)  # Blue
                elif node_type in ["Shape", "Color", "Pattern", "Object", "Scene"]:
                    color = QColor(46, 204, 113, 200)  # Green
                elif node_type in ["Relation", "Category", "Property", "Abstract", "Rule"]:
                    color = QColor(155, 89, 182, 200)  # Purple
                elif node_type in ["Sequence", "Action", "Skill", "Routine", "Method"]:
                    color = QColor(230, 126, 34, 200)  # Orange
                else:
                    color = QColor(149, 165, 166, 200)  # Gray
                
                # Calculate radius based on strength (5-15 pixels)
                radius = 5 + int(strength * 10)
                
                # Draw node
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(color))
                painter.drawEllipse(int(x - radius), int(y - radius), radius * 2, radius * 2)
                
                # Draw border
                painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(int(x - radius), int(y - radius), radius * 2, radius * 2)
                
                # Draw label if node has one
                label = node.get("label", "")
                if label:
                    painter.setPen(QPen(QColor(255, 255, 255)))
                    rect = painter.fontMetrics().boundingRect(label)
                    painter.drawText(int(x - rect.width()/2), int(y + radius + 15), label)


# For testing outside of the dashboard
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    # Create and show the panel
    panel = MemoryVisualizationPanel(db_path=None, refresh_rate=1000)
    panel.set_mock_mode(True)
    panel.show()
    
    sys.exit(app.exec_()) 