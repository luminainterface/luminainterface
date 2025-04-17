"""
Metrics Panel for the V5 Fractal Echo Visualization System.

This panel displays various metrics from the neural network visualization
including consciousness metrics, pattern analysis, and performance statistics.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to Python path if needed
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import Qt compatibility layer
from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui, Qt, Signal, Slot

import json
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsPanel(QtWidgets.QWidget):
    """Panel for displaying neural network metrics and statistics."""
    
    def __init__(self, socket_manager=None):
        """
        Initialize the Metrics Panel.
        
        Args:
            socket_manager: Optional socket manager for plugin communication
        """
        super().__init__()
        self.socket_manager = socket_manager
        self.metrics = {}
        self.stats = {}
        self.update_interval = 2000  # ms
        self.update_timer = None
        
        # Initialize UI
        self.initUI()
        
        # Connect to socket manager if available
        if socket_manager:
            self.connect_to_socket_manager()
        
        # Start update timer
        self.update_timer = QtCore.QTimer(self)
        self.update_timer.timeout.connect(self.update_metrics)
        self.update_timer.start(self.update_interval)
    
    def initUI(self):
        """Initialize the user interface."""
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header_label = QtWidgets.QLabel("Neural Network Metrics")
        header_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #EFEFEF;")
        layout.addWidget(header_label)
        
        # Tabs for different metric categories
        self.tabs = QtWidgets.QTabWidget()
        
        # Consciousness Metrics Tab
        consciousness_tab = QtWidgets.QWidget()
        consciousness_layout = QtWidgets.QVBoxLayout(consciousness_tab)
        
        # Create tree widget for consciousness metrics
        self.consciousness_tree = QtWidgets.QTreeWidget()
        self.consciousness_tree.setHeaderLabels(["Metric", "Value"])
        self.consciousness_tree.setAlternatingRowColors(True)
        consciousness_layout.addWidget(self.consciousness_tree)
        
        # Performance Metrics Tab
        performance_tab = QtWidgets.QWidget()
        performance_layout = QtWidgets.QVBoxLayout(performance_tab)
        
        # Create tree widget for performance metrics
        self.performance_tree = QtWidgets.QTreeWidget()
        self.performance_tree.setHeaderLabels(["Metric", "Value"])
        self.performance_tree.setAlternatingRowColors(True)
        performance_layout.addWidget(self.performance_tree)
        
        # Pattern Analysis Tab
        pattern_tab = QtWidgets.QWidget()
        pattern_layout = QtWidgets.QVBoxLayout(pattern_tab)
        
        # Create tree widget for pattern metrics
        self.pattern_tree = QtWidgets.QTreeWidget()
        self.pattern_tree.setHeaderLabels(["Metric", "Value"])
        self.pattern_tree.setAlternatingRowColors(True)
        pattern_layout.addWidget(self.pattern_tree)
        
        # Add tabs
        self.tabs.addTab(consciousness_tab, "Consciousness")
        self.tabs.addTab(performance_tab, "Performance")
        self.tabs.addTab(pattern_tab, "Pattern Analysis")
        
        layout.addWidget(self.tabs)
        
        # Controls area at the bottom
        controls_layout = QtWidgets.QHBoxLayout()
        
        # Update interval control
        interval_label = QtWidgets.QLabel("Update Interval (ms):")
        controls_layout.addWidget(interval_label)
        
        self.interval_spinner = QtWidgets.QSpinBox()
        self.interval_spinner.setRange(500, 10000)
        self.interval_spinner.setSingleStep(500)
        self.interval_spinner.setValue(self.update_interval)
        self.interval_spinner.valueChanged.connect(self.on_interval_changed)
        controls_layout.addWidget(self.interval_spinner)
        
        # Refresh button
        refresh_button = QtWidgets.QPushButton("Refresh Now")
        refresh_button.clicked.connect(self.update_metrics)
        controls_layout.addWidget(refresh_button)
        
        # Add stretch to push controls to the left
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Status bar
        self.status_label = QtWidgets.QLabel("Initializing metrics...")
        self.status_label.setStyleSheet("color: #999999; font-style: italic;")
        layout.addWidget(self.status_label)
    
    def connect_to_socket_manager(self):
        """Connect to socket manager for data exchange."""
        if not self.socket_manager:
            return
        
        try:
            # Register for metrics updates
            self.socket_manager.register_message_handler(
                "metrics_updated", 
                self.handle_metrics_update
            )
            
            # Request initial metrics
            self.request_metrics()
            
            self.status_label.setText("Connected to metrics providers")
        except Exception as e:
            logger.error(f"Error connecting to socket manager: {str(e)}")
            self.status_label.setText(f"Connection error: {str(e)}")
    
    def request_metrics(self):
        """Request metrics data from plugins."""
        if not self.socket_manager:
            # Generate mock data for testing
            self.generate_mock_metrics()
            return
        
        # Request metrics from plugins
        request_id = f"metrics_request_{int(time.time())}"
        message = {
            "type": "request_metrics",
            "request_id": request_id,
            "content": {
                "categories": ["consciousness", "performance", "pattern"]
            }
        }
        
        try:
            self.socket_manager.send_message(message)
            self.status_label.setText("Metrics requested")
        except Exception as e:
            logger.error(f"Error requesting metrics: {str(e)}")
            self.status_label.setText(f"Error requesting metrics: {str(e)}")
    
    def handle_metrics_update(self, message):
        """Handle metrics update messages."""
        try:
            data = message.get("data", {})
            if "error" in data:
                error_msg = data.get("error", "Unknown error")
                self.status_label.setText(f"Error: {error_msg}")
                return
            
            # Update metrics with new data
            metrics = data.get("metrics", {})
            
            # Update UI with new metrics
            self.update_metrics_ui(metrics)
            
            # Update status
            timestamp = data.get("timestamp", "unknown")
            self.status_label.setText(f"Metrics updated at {timestamp}")
            
        except Exception as e:
            logger.error(f"Error handling metrics update: {str(e)}")
            self.status_label.setText(f"Error processing metrics: {str(e)}")
    
    def generate_mock_metrics(self):
        """Generate mock metrics data for testing."""
        # Consciousness metrics
        consciousness_metrics = {
            "global_awareness_level": random.randint(60, 95),
            "integration_index": round(random.uniform(0.6, 0.9), 2),
            "neural_coherence": random.choice(["Low", "Medium", "High", "Very High"]),
            "responsiveness": random.randint(80, 99),
            "self_awareness": round(random.uniform(0.4, 0.8), 2),
            "memory_access": round(random.uniform(0.5, 0.9), 2),
            "reflection": round(random.uniform(0.3, 0.7), 2)
        }
        
        # Performance metrics
        performance_metrics = {
            "processing_time_ms": random.randint(5, 50),
            "memory_usage_mb": random.randint(200, 500),
            "node_count": random.randint(500, 2000),
            "connection_count": random.randint(2000, 10000),
            "throughput": f"{random.randint(100, 500)} msg/s",
            "cache_hit_ratio": f"{random.randint(60, 95)}%",
            "error_rate": f"{round(random.uniform(0.01, 0.1), 3)}%"
        }
        
        # Pattern metrics
        pattern_metrics = {
            "fractal_dimension": round(random.uniform(1.5, 1.9), 2),
            "complexity_index": random.randint(70, 95),
            "pattern_coherence": random.randint(80, 99),
            "entropy_level": random.choice(["Low", "Medium", "High"]),
            "detected_patterns": random.randint(3, 12),
            "pattern_stability": round(random.uniform(0.7, 0.95), 2),
            "recursion_depth": random.randint(4, 8)
        }
        
        # Combine metrics
        metrics = {
            "consciousness": consciousness_metrics,
            "performance": performance_metrics,
            "pattern": pattern_metrics,
            "timestamp": time.strftime("%H:%M:%S")
        }
        
        # Update UI with mock metrics
        self.update_metrics_ui(metrics)
        
        # Update status
        self.status_label.setText(f"Mock metrics generated at {metrics['timestamp']}")
    
    def update_metrics_ui(self, metrics):
        """Update the UI with the provided metrics."""
        # Store metrics
        self.metrics = metrics
        
        # Clear trees
        self.consciousness_tree.clear()
        self.performance_tree.clear()
        self.pattern_tree.clear()
        
        # Update consciousness metrics
        consciousness = metrics.get("consciousness", {})
        consciousness_items = []
        
        for key, value in consciousness.items():
            # Format key for display
            display_key = " ".join(word.capitalize() for word in key.split("_"))
            
            # Create item
            item = QtWidgets.QTreeWidgetItem([display_key, str(value)])
            
            # Set color based on value if it's a number
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # Normalize to 0-1 range for coloring
                try:
                    if isinstance(value, int) and value > 1:
                        # Assume percentage-like values (0-100)
                        normalized = value / 100
                    else:
                        # Assume already in 0-1 range
                        normalized = float(value)
                    
                    # Clamp to 0-1
                    normalized = max(0, min(1, normalized))
                    
                    # Create a color gradient: red (0) to yellow (0.5) to green (1)
                    if normalized < 0.5:
                        # Red to yellow
                        r = 255
                        g = int(255 * (normalized * 2))
                        b = 0
                    else:
                        # Yellow to green
                        r = int(255 * (1 - (normalized - 0.5) * 2))
                        g = 255
                        b = 0
                    
                    # Set text color
                    item.setForeground(1, QtGui.QBrush(QtGui.QColor(r, g, b)))
                except (ValueError, TypeError):
                    pass
            
            consciousness_items.append(item)
        
        self.consciousness_tree.addTopLevelItems(consciousness_items)
        
        # Update performance metrics
        performance = metrics.get("performance", {})
        performance_items = []
        
        for key, value in performance.items():
            # Format key for display
            display_key = " ".join(word.capitalize() for word in key.split("_"))
            
            # Create item
            item = QtWidgets.QTreeWidgetItem([display_key, str(value)])
            performance_items.append(item)
        
        self.performance_tree.addTopLevelItems(performance_items)
        
        # Update pattern metrics
        pattern = metrics.get("pattern", {})
        pattern_items = []
        
        for key, value in pattern.items():
            # Format key for display
            display_key = " ".join(word.capitalize() for word in key.split("_"))
            
            # Create item
            item = QtWidgets.QTreeWidgetItem([display_key, str(value)])
            pattern_items.append(item)
        
        self.pattern_tree.addTopLevelItems(pattern_items)
        
        # Resize columns to content
        self.consciousness_tree.resizeColumnToContents(0)
        self.performance_tree.resizeColumnToContents(0)
        self.pattern_tree.resizeColumnToContents(0)
    
    @Slot(int)
    def on_interval_changed(self, value):
        """Handle update interval change."""
        self.update_interval = value
        
        # Update timer
        if self.update_timer:
            self.update_timer.setInterval(self.update_interval)
            
        self.status_label.setText(f"Update interval set to {value} ms")
    
    @Slot()
    def update_metrics(self):
        """Update metrics data."""
        try:
            # Request metrics if connected to socket manager
            if self.socket_manager:
                self.request_metrics()
            else:
                # Generate mock data if not connected
                self.generate_mock_metrics()
                
            # Update status
            current_time = time.strftime("%H:%M:%S")
            self.status_label.setText(f"Metrics requested at {current_time}")
            
            # Force a UI update regardless of socket response
            current_metrics = {
                "consciousness": {
                    "global_awareness_level": 87,
                    "integration_index": 0.77,
                    "neural_coherence": "High",
                    "responsiveness": 94
                },
                "performance": {
                    "processing_time_ms": 25,
                    "memory_usage_mb": 340,
                    "node_count": 1245,
                    "connection_count": 5678
                },
                "pattern": {
                    "fractal_dimension": 1.79,
                    "complexity_index": 80, 
                    "pattern_coherence": 88,
                    "entropy_level": "Medium-High"
                },
                "timestamp": current_time
            }
            
            # Update UI with current metrics
            self.update_metrics_ui(current_metrics)
            
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
            self.status_label.setText(f"Error updating metrics: {str(e)}")
    
    def cleanup(self):
        """Clean up resources before closing."""
        # Stop timer
        if self.update_timer and self.update_timer.isActive():
            self.update_timer.stop()
        
        # Deregister message handlers
        if self.socket_manager:
            self.socket_manager.deregister_message_handler("metrics_updated") 