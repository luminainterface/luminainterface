"""
Network Visualization Panel for the V5 Fractal Echo Visualization System.

This panel provides visualization of neural network connections and relationships.
"""

import os
import sys
import time
from pathlib import Path
import csv
from datetime import datetime
import json

# Add project root to Python path if needed
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import Qt compatibility layer
from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui, Qt, Signal, Slot

class NetworkVisualizationPanel(QtWidgets.QWidget):
    """Panel for visualizing neural network connections and relationships."""
    
    # Signal emitted when a node is activated
    node_activated = Signal(dict)
    
    def __init__(self, socket_manager=None):
        """
        Initialize the Network Visualization Panel.
        
        Args:
            socket_manager: Optional socket manager for plugin communication
        """
        super().__init__()
        self.socket_manager = socket_manager
        self.nodes = []
        self.connections = []
        self.selected_node = None
        self.hover_node = None
        self.zoom_level = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.dragging = False
        self.last_mouse_pos = None
        self.neural_weight = 0.5
        
        # Save state management
        self.auto_save_timer = QtCore.QTimer()
        self.auto_save_timer.timeout.connect(self._auto_save_tick)
        self.last_save_time = datetime.now()
        self.save_pending = False
        self.last_save_path = None
        
        # Initialize UI
        self.initUI()
        
        # Connect to socket manager if available
        if socket_manager:
            self.connect_to_socket_manager()
    
    def initUI(self):
        """Initialize the user interface."""
        # Panel layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Top control bar
        control_frame = QtWidgets.QFrame()
        control_frame.setMaximumHeight(40)
        control_frame.setStyleSheet("""
            QFrame {
                background-color: #1A1A1A;
                border-bottom: 1px solid #D4AF37;
            }
        """)
        control_layout = QtWidgets.QHBoxLayout(control_frame)
        control_layout.setContentsMargins(10, 5, 10, 5)
        
        # Connection strength slider
        self.strength_label = QtWidgets.QLabel("60%")
        self.strength_label.setStyleSheet("color: #D4AF37;")
        control_layout.addWidget(self.strength_label)
        
        self.strength_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(60)
        self.strength_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #D4AF37;
                height: 4px;
                background: #2D2D2D;
                margin: 0px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #D4AF37;
                border: 1px solid #D4AF37;
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
        """)
        control_layout.addWidget(self.strength_slider, 1)  # 1 = stretch factor
        
        # Display mode dropdown
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Connection Strength", "Activity", "Relevance"])
        self.mode_combo.setStyleSheet("""
            QComboBox {
                background-color: #2D2D2D;
                border: 1px solid #D4AF37;
                border-radius: 4px;
                color: #D4AF37;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(assets/icons/dropdown.svg);
                width: 12px;
                height: 12px;
            }
        """)
        control_layout.addWidget(self.mode_combo)
        
        # Reset view button
        reset_btn = QtWidgets.QPushButton("Reset View")
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #2D2D2D;
                border: 1px solid #D4AF37;
                border-radius: 4px;
                color: #D4AF37;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #3D3D3D;
            }
            QPushButton:pressed {
                background-color: #1A1A1A;
            }
        """)
        reset_btn.clicked.connect(self.reset_view)
        control_layout.addWidget(reset_btn)
        
        # Neural Chat button
        chat_btn = QtWidgets.QPushButton("Neural Chat")
        chat_btn.setStyleSheet(reset_btn.styleSheet())
        control_layout.addWidget(chat_btn)
        
        layout.addWidget(control_frame)
        
        # Network visualization area
        self.network_view = NetworkCanvas(self)
        layout.addWidget(self.network_view, 1)  # 1 = stretch factor
        
        # Graphs section at bottom
        graphs_frame = QtWidgets.QFrame()
        graphs_frame.setMinimumHeight(200)
        graphs_frame.setStyleSheet("""
            QFrame {
                background-color: #1A1A1A;
                border-top: 1px solid #D4AF37;
            }
        """)
        graphs_layout = QtWidgets.QVBoxLayout(graphs_frame)
        
        # Tab widget for different graph types
        self.graph_tabs = QtWidgets.QTabWidget()
        self.graph_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #D4AF37;
                background: #1A1A1A;
            }
            QTabBar::tab {
                background: #2D2D2D;
                color: #D4AF37;
                border: 1px solid #D4AF37;
                padding: 5px 10px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #1A1A1A;
            }
        """)
        
        # Add graph tabs
        self.graph_tabs.addTab(QtWidgets.QWidget(), "Network Activity")
        self.graph_tabs.addTab(QtWidgets.QWidget(), "Performance Metrics")
        self.graph_tabs.addTab(QtWidgets.QWidget(), "Learning")
        self.graph_tabs.addTab(QtWidgets.QWidget(), "Model Saving")
        
        graphs_layout.addWidget(self.graph_tabs)
        
        # Time scale and view controls
        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        
        controls_layout.addWidget(QtWidgets.QLabel("Time Scale:"))
        time_combo = QtWidgets.QComboBox()
        time_combo.addItems(["1m", "5m", "15m", "1h", "6h", "24h"])
        time_combo.setStyleSheet(self.mode_combo.styleSheet())
        controls_layout.addWidget(time_combo)
        
        controls_layout.addWidget(QtWidgets.QLabel("View:"))
        view_combo = QtWidgets.QComboBox()
        view_combo.addItems(["Linear", "Logarithmic", "Normalized"])
        view_combo.setStyleSheet(self.mode_combo.styleSheet())
        controls_layout.addWidget(view_combo)
        
        controls_layout.addStretch()
        
        graphs_layout.addWidget(controls_widget)
        layout.addWidget(graphs_frame)
        
        # Status bar at the bottom
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet("padding: 5px; background-color: rgba(0, 0, 0, 0.1);")
        layout.addWidget(self.status_label)
        
        # Set mouse tracking for hover effects
        self.setMouseTracking(True)
        
        # Set focus policy to accept keyboard input
        self.setFocusPolicy(Qt.StrongFocus)
    
    def connect_to_socket_manager(self):
        """Connect to socket manager for data exchange."""
        if not self.socket_manager:
            return
            
        # Register for network data updates
        try:
            # Add any specific socket registrations here
            pass
        except Exception as e:
            self.status_label.setText(f"Error connecting to socket manager: {str(e)}")
    
    def update_visualization(self):
        """Update the network visualization."""
        # In a real implementation, this would update based on current data
        # For now, generate some mock data if we don't have any
        if not self.nodes:
            self.generate_mock_data()
        
        # Trigger repaint
        self.network_view.update()
        
        # Update status
        mode = self.mode_combo.currentText()
        self.status_label.setText(f"Displaying: {mode} | Nodes: {len(self.nodes)} | Connections: {len(self.connections)}")
    
    def generate_mock_data(self):
        """Generate mock data for visualization."""
        import random
        
        # Clear existing data
        self.nodes = []
        self.connections = []
        
        # Generate nodes
        node_count = 20
        for i in range(node_count):
            node = {
                "id": i,
                "name": f"Node {i}",
                "type": random.choice(["input", "hidden", "output"]),
                "x": random.uniform(0.1, 0.9),
                "y": random.uniform(0.1, 0.9),
                "activation": random.uniform(0, 1),
                "consciousness": random.uniform(0, 1)
            }
            self.nodes.append(node)
        
        # Generate connections (not fully connected)
        for i in range(node_count):
            # Each node connects to 2-5 other nodes
            targets = random.sample(range(node_count), min(random.randint(2, 5), node_count))
            for target in targets:
                if target != i:  # No self-connections
                    connection = {
                        "source": i,
                        "target": target,
                        "strength": random.uniform(0.1, 1.0),
                        "type": random.choice(["excitatory", "inhibitory"])
                    }
                    self.connections.append(connection)
    
    def zoom_in(self):
        """Zoom in the visualization."""
        self.zoom_level = min(self.zoom_level * 1.2, 5.0)
        self.network_view.update()
    
    def zoom_out(self):
        """Zoom out the visualization."""
        self.zoom_level = max(self.zoom_level / 1.2, 0.2)
        self.network_view.update()
    
    def reset_view(self):
        """Reset the view to default."""
        self.zoom_level = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.network_view.update()
    
    def highlight_pattern(self, pattern_data):
        """
        Highlight nodes that are part of a pattern.
        
        Args:
            pattern_data: Data about the pattern to highlight
        """
        # In a real implementation, this would identify nodes associated with the pattern
        self.status_label.setText(f"Highlighting pattern: {pattern_data.get('name', 'Unknown')}")
        self.network_view.update()
    
    def focus_node(self, node_data):
        """
        Focus on a specific node.
        
        Args:
            node_data: Data about the node to focus on
        """
        node_id = node_data.get('id')
        for node in self.nodes:
            if node['id'] == node_id:
                self.selected_node = node
                self.status_label.setText(f"Selected node: {node.get('name', f'Node {node_id}')}")
                
                # Notify that a node was activated
                self.node_activated.emit(node)
                
                self.network_view.update()
                break
    
    def show_memory_path(self, memory_data):
        """
        Visualize the path through the network for a specific memory.
        
        Args:
            memory_data: Data about the memory to show
        """
        # In a real implementation, this would highlight the nodes and connections
        # involved in processing the specified memory
        self.status_label.setText(f"Showing memory path: {memory_data.get('type', 'Unknown')}")
        self.network_view.update()
    
    def cleanup(self):
        """Clean up resources before closing."""
        pass

    def set_neural_weight(self, weight):
        """
        Set the neural network weight for network rendering
        
        Args:
            weight: Neural network weight (0.0-1.0)
        """
        # Store the weight
        self.neural_weight = weight
        
        # Adjust network rendering parameters based on the weight
        if hasattr(self, 'network_view'):
            # Adjust node size based on neural weight
            self.network_view.node_scale_factor = 0.8 + (weight * 0.4)
            
            # Adjust edge thickness based on neural weight
            self.network_view.edge_scale_factor = 0.7 + (weight * 0.6)
            
            # Adjust physics parameters
            if hasattr(self.network_view, 'physics_enabled'):
                # Higher neural weight means more dynamic physics
                self.network_view.physics_strength = 0.5 + (weight * 0.5)
                
                # Update physics enabled state if needed
                if weight > 0.7 and not self.network_view.physics_enabled:
                    self.network_view.physics_enabled = True
                    self.network_view.restart_physics()
            
            # Log the change
            import logging
            logging.getLogger(__name__).info(
                f"Updated network parameters: node_scale={self.network_view.node_scale_factor:.2f}, "
                f"edge_scale={self.network_view.edge_scale_factor:.2f}"
            )
            
            # Trigger an update

    def update_spiderweb_metrics(self, metrics):
        """Update the visualization with Spiderweb metrics."""
        try:
            if not metrics:
                return
                
            # Update quantum metrics if available
            if 'quantum' in metrics:
                self.update_quantum_metrics(metrics['quantum'])
                
            # Update cosmic metrics if available
            if 'cosmic' in metrics:
                self.update_cosmic_metrics(metrics['cosmic'])
                
            # Trigger a repaint
            self.network_view.update()
            
        except Exception as e:
            logger.error(f"Error updating Spiderweb metrics in visualization: {str(e)}")
            
    def update_quantum_metrics(self, quantum_metrics):
        """Update quantum-specific metrics visualization."""
        try:
            # Update quantum field strength indicators
            field_strength = quantum_metrics.get('field_strength', 0)
            self.network_view.quantum_field_strength = field_strength
            
            # Update entanglement network visualization
            entangled_nodes = quantum_metrics.get('entangled_nodes', [])
            self.network_view.entangled_connections = entangled_nodes
            
        except Exception as e:
            logger.error(f"Error updating quantum metrics: {str(e)}")
            
    def update_cosmic_metrics(self, cosmic_metrics):
        """Update cosmic-specific metrics visualization."""
        try:
            # Update cosmic field strength indicators
            field_strength = cosmic_metrics.get('field_strength', 0)
            self.network_view.cosmic_field_strength = field_strength
            
            # Update dimensional resonance visualization
            resonance = cosmic_metrics.get('dimensional_resonance', 0)
            self.network_view.dimensional_resonance = resonance
            
        except Exception as e:
            logger.error(f"Error updating cosmic metrics: {str(e)}")

    def _setup_bottom_tabs(self):
        """Setup the bottom tab panel with monitoring and control features"""
        bottom_panel = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QVBoxLayout(bottom_panel)
        
        # Create tab widget
        self.tab_widget = QtWidgets.QTabWidget()
        
        # Network Activity Tab
        activity_tab = QtWidgets.QWidget()
        activity_layout = QtWidgets.QVBoxLayout(activity_tab)
        
        # Activity Monitor
        self.activity_text = QtWidgets.QTextEdit()
        self.activity_text.setReadOnly(True)
        self.activity_text.setStyleSheet("background-color: #1E1E1E; color: #00FF00;")
        activity_layout.addWidget(self.activity_text)
        
        # Activity Controls
        activity_controls = QtWidgets.QHBoxLayout()
        self.pause_activity = QtWidgets.QPushButton("Pause Monitoring")
        self.clear_activity = QtWidgets.QPushButton("Clear Log")
        self.export_activity = QtWidgets.QPushButton("Export Log")
        activity_controls.addWidget(self.pause_activity)
        activity_controls.addWidget(self.clear_activity)
        activity_controls.addWidget(self.export_activity)
        activity_layout.addLayout(activity_controls)
        
        # Performance Metrics Tab
        metrics_tab = QtWidgets.QWidget()
        metrics_layout = QtWidgets.QGridLayout(metrics_tab)
        
        # Metrics Display
        self.cpu_usage = QtWidgets.QProgressBar()
        self.memory_usage = QtWidgets.QProgressBar()
        self.network_latency = QtWidgets.QProgressBar()
        self.operation_count = QtWidgets.QLabel("Operations: 0")
        
        metrics_layout.addWidget(QtWidgets.QLabel("CPU Usage:"), 0, 0)
        metrics_layout.addWidget(self.cpu_usage, 0, 1)
        metrics_layout.addWidget(QtWidgets.QLabel("Memory Usage:"), 1, 0)
        metrics_layout.addWidget(self.memory_usage, 1, 1)
        metrics_layout.addWidget(QtWidgets.QLabel("Network Latency:"), 2, 0)
        metrics_layout.addWidget(self.network_latency, 2, 1)
        metrics_layout.addWidget(self.operation_count, 3, 0, 1, 2)
        
        # Learning Tab
        learning_tab = QtWidgets.QWidget()
        learning_layout = QtWidgets.QVBoxLayout(learning_tab)
        
        # Learning Parameters
        params_group = QtWidgets.QGroupBox("Learning Parameters")
        params_layout = QtWidgets.QFormLayout(params_group)
        
        self.learning_rate = QtWidgets.QDoubleSpinBox()
        self.learning_rate.setRange(0.0001, 1.0)
        self.learning_rate.setValue(0.01)
        self.learning_rate.setSingleStep(0.001)
        
        self.batch_size = QtWidgets.QSpinBox()
        self.batch_size.setRange(1, 1000)
        self.batch_size.setValue(32)
        
        self.epochs = QtWidgets.QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(10)
        
        params_layout.addRow("Learning Rate:", self.learning_rate)
        params_layout.addRow("Batch Size:", self.batch_size)
        params_layout.addRow("Epochs:", self.epochs)
        
        learning_layout.addWidget(params_group)
        
        # Learning Controls
        learning_controls = QtWidgets.QHBoxLayout()
        self.start_learning = QtWidgets.QPushButton("Start Learning")
        self.pause_learning = QtWidgets.QPushButton("Pause")
        self.reset_learning = QtWidgets.QPushButton("Reset")
        learning_controls.addWidget(self.start_learning)
        learning_controls.addWidget(self.pause_learning)
        learning_controls.addWidget(self.reset_learning)
        learning_layout.addLayout(learning_controls)
        
        # Learning Progress
        self.learning_progress = QtWidgets.QProgressBar()
        learning_layout.addWidget(self.learning_progress)
        
        # Model Saving Tab
        saving_tab = QtWidgets.QWidget()
        saving_layout = QtWidgets.QVBoxLayout(saving_tab)
        
        # Save Configuration
        config_group = QtWidgets.QGroupBox("Save Configuration")
        config_layout = QtWidgets.QFormLayout(config_group)
        
        self.save_path = QtWidgets.QLineEdit()
        self.save_path.setPlaceholderText("Save directory path...")
        self.browse_button = QtWidgets.QPushButton("Browse...")
        path_layout = QtWidgets.QHBoxLayout()
        path_layout.addWidget(self.save_path)
        path_layout.addWidget(self.browse_button)
        
        self.save_format = QtWidgets.QComboBox()
        self.save_format.addItems(["Binary (.bin)", "JSON (.json)", "Checkpoint (.ckpt)"])
        
        self.compression = QtWidgets.QComboBox()
        self.compression.addItems(["None", "Light", "Medium", "Heavy"])
        
        self.auto_save = QtWidgets.QCheckBox("Enable Auto-save")
        self.save_interval = QtWidgets.QSpinBox()
        self.save_interval.setRange(1, 60)
        self.save_interval.setValue(5)
        self.save_interval.setSuffix(" min")
        
        config_layout.addRow("Save Path:", path_layout)
        config_layout.addRow("Format:", self.save_format)
        config_layout.addRow("Compression:", self.compression)
        config_layout.addRow("Auto-save:", self.auto_save)
        config_layout.addRow("Interval:", self.save_interval)
        
        saving_layout.addWidget(config_group)
        
        # Save History
        history_group = QtWidgets.QGroupBox("Save History")
        history_layout = QtWidgets.QVBoxLayout(history_group)
        
        self.save_history = QtWidgets.QTableWidget()
        self.save_history.setColumnCount(4)
        self.save_history.setHorizontalHeaderLabels(["Timestamp", "Size", "Format", "Status"])
        self.save_history.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        
        history_layout.addWidget(self.save_history)
        saving_layout.addWidget(history_group)
        
        # Save Controls
        save_controls = QtWidgets.QHBoxLayout()
        self.save_button = QtWidgets.QPushButton("Save Now")
        self.export_button = QtWidgets.QPushButton("Export History")
        self.clear_history = QtWidgets.QPushButton("Clear History")
        save_controls.addWidget(self.save_button)
        save_controls.addWidget(self.export_button)
        save_controls.addWidget(self.clear_history)
        saving_layout.addLayout(save_controls)
        
        # Add all tabs
        self.tab_widget.addTab(activity_tab, "Network Activity")
        self.tab_widget.addTab(metrics_tab, "Performance Metrics")
        self.tab_widget.addTab(learning_tab, "Learning")
        self.tab_widget.addTab(saving_tab, "Model Saving")
        
        # Add tab widget to bottom layout
        bottom_layout.addWidget(self.tab_widget)
        
        # Connect signals
        self._connect_tab_signals()
        
        return bottom_panel
        
    def _connect_tab_signals(self):
        """Connect signals for tab functionality"""
        # Activity tab
        self.pause_activity.clicked.connect(self._toggle_activity_monitoring)
        self.clear_activity.clicked.connect(self.activity_text.clear)
        self.export_activity.clicked.connect(self._export_activity_log)
        
        # Learning tab
        self.start_learning.clicked.connect(self._toggle_learning)
        self.reset_learning.clicked.connect(self._reset_learning)
        
        # Save tab
        self.browse_button.clicked.connect(self._browse_save_path)
        self.save_button.clicked.connect(self._save_model)
        self.auto_save.stateChanged.connect(self._toggle_auto_save)
        self.export_button.clicked.connect(self._export_save_history)
        self.clear_history.clicked.connect(self._clear_save_history)
        
    def _toggle_activity_monitoring(self):
        """Toggle network activity monitoring"""
        if self.pause_activity.text() == "Pause Monitoring":
            self.pause_activity.setText("Resume Monitoring")
        else:
            self.pause_activity.setText("Pause Monitoring")
            
    def _export_activity_log(self):
        """Export network activity log"""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Activity Log",
            "",
            "Text Files (*.txt);;All Files (*)"
        )
        if filename:
            with open(filename, 'w') as f:
                f.write(self.activity_text.toPlainText())
                
    def _toggle_learning(self):
        """Toggle learning process"""
        if self.start_learning.text() == "Start Learning":
            self.start_learning.setText("Stop Learning")
            self.learning_progress.setValue(0)
            # Start learning process
        else:
            self.start_learning.setText("Start Learning")
            # Stop learning process
            
    def _reset_learning(self):
        """Reset learning parameters and progress"""
        self.learning_rate.setValue(0.01)
        self.batch_size.setValue(32)
        self.epochs.setValue(10)
        self.learning_progress.setValue(0)
        
    def _browse_save_path(self):
        """Open file dialog to select save path"""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Save Directory",
            "",
            QtWidgets.QFileDialog.ShowDirsOnly
        )
        if directory:
            self.save_path.setText(directory)
            
    def _save_model(self):
        """Save the model with current configuration"""
        if not self.save_path.text():
            QtWidgets.QMessageBox.warning(
                self,
                "Save Error",
                "Please select a save directory first."
            )
            return
            
        try:
            save_path = self.save_path.text()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            format_ext = self.save_format.currentText().split('(')[1].strip(')')
            filename = f"neural_state_{timestamp}{format_ext}"
            full_path = os.path.join(save_path, filename)
            
            # Prepare state data
            state_data = {
                'timestamp': timestamp,
                'nodes': self.nodes,
                'connections': self.connections,
                'neural_weight': self.neural_weight,
                'metrics': {
                    'cpu_usage': self.cpu_usage.value(),
                    'memory_usage': self.memory_usage.value(),
                    'network_latency': self.network_latency.value(),
                    'operations': self.operation_count.text().split(': ')[1]
                },
                'learning_params': {
                    'learning_rate': self.learning_rate.value(),
                    'batch_size': self.batch_size.value(),
                    'epochs': self.epochs.value()
                }
            }
            
            # Apply compression if selected
            compression_level = self.compression.currentText()
            if compression_level != "None":
                import zlib
                compression_levels = {
                    "Light": 1,
                    "Medium": 6,
                    "Heavy": 9
                }
                state_data = zlib.compress(
                    json.dumps(state_data).encode(),
                    level=compression_levels[compression_level]
                )
            
            # Save based on selected format
            format_type = self.save_format.currentText()
            if "Binary" in format_type:
                with open(full_path, 'wb') as f:
                    import pickle
                    pickle.dump(state_data, f)
            elif "JSON" in format_type:
                with open(full_path, 'w') as f:
                    json.dump(state_data, f, indent=2)
            else:  # Checkpoint
                import torch
                torch.save(state_data, full_path)
            
            # Calculate file size
            size = os.path.getsize(full_path)
            size_str = self._format_size(size)
            
            # Add save entry to history
            row_position = self.save_history.rowCount()
            self.save_history.insertRow(row_position)
            
            timestamp_item = QtWidgets.QTableWidgetItem(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            size_item = QtWidgets.QTableWidgetItem(size_str)
            format_item = QtWidgets.QTableWidgetItem(self.save_format.currentText())
            status_item = QtWidgets.QTableWidgetItem("Success")
            
            self.save_history.setItem(row_position, 0, timestamp_item)
            self.save_history.setItem(row_position, 1, size_item)
            self.save_history.setItem(row_position, 2, format_item)
            self.save_history.setItem(row_position, 3, status_item)
            
            # Update last save info
            self.last_save_time = datetime.now()
            self.last_save_path = full_path
            self.save_pending = False
            
            # Show success message
            self.status_label.setText(f"Model saved successfully to {filename}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save model: {str(e)}"
            )
            # Add failed save to history
            row_position = self.save_history.rowCount()
            self.save_history.insertRow(row_position)
            
            self.save_history.setItem(row_position, 0, QtWidgets.QTableWidgetItem(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            self.save_history.setItem(row_position, 1, QtWidgets.QTableWidgetItem("N/A"))
            self.save_history.setItem(row_position, 2, QtWidgets.QTableWidgetItem(self.save_format.currentText()))
            self.save_history.setItem(row_position, 3, QtWidgets.QTableWidgetItem("Failed"))
            
    def _format_size(self, size_bytes):
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
            
    def _toggle_auto_save(self, state):
        """Toggle auto-save functionality"""
        self.save_interval.setEnabled(state)
        if state:
            # Start auto-save timer
            interval_mins = self.save_interval.value()
            self.auto_save_timer.start(interval_mins * 60 * 1000)  # Convert to milliseconds
            self.status_label.setText(f"Auto-save enabled ({interval_mins} min interval)")
        else:
            # Stop auto-save timer
            self.auto_save_timer.stop()
            self.status_label.setText("Auto-save disabled")
            
    def _auto_save_tick(self):
        """Handle auto-save timer tick"""
        if not self.save_path.text():
            self.status_label.setText("Auto-save failed: No save path selected")
            return
            
        if self.save_pending:
            self.status_label.setText("Auto-save pending: Previous save not completed")
            return
            
        self.save_pending = True
        self._save_model()
        
    def load_state(self, filepath):
        """Load a previously saved state"""
        try:
            format_type = os.path.splitext(filepath)[1]
            
            if format_type == '.bin':
                with open(filepath, 'rb') as f:
                    import pickle
                    state_data = pickle.load(f)
            elif format_type == '.json':
                with open(filepath, 'r') as f:
                    state_data = json.load(f)
            elif format_type == '.ckpt':
                import torch
                state_data = torch.load(filepath)
            else:
                raise ValueError(f"Unsupported file format: {format_type}")
                
            # Check if data is compressed
            if isinstance(state_data, bytes):
                import zlib
                state_data = json.loads(zlib.decompress(state_data).decode())
            
            # Restore state
            self.nodes = state_data['nodes']
            self.connections = state_data['connections']
            self.neural_weight = state_data['neural_weight']
            
            # Restore metrics if available
            if 'metrics' in state_data:
                metrics = state_data['metrics']
                self.cpu_usage.setValue(int(metrics['cpu_usage']))
                self.memory_usage.setValue(int(metrics['memory_usage']))
                self.network_latency.setValue(int(metrics['network_latency']))
                self.operation_count.setText(f"Operations: {metrics['operations']}")
            
            # Restore learning parameters if available
            if 'learning_params' in state_data:
                params = state_data['learning_params']
                self.learning_rate.setValue(params['learning_rate'])
                self.batch_size.setValue(params['batch_size'])
                self.epochs.setValue(params['epochs'])
            
            # Update visualization
            self.network_view.update()
            
            # Show success message
            self.status_label.setText(f"State loaded successfully from {os.path.basename(filepath)}")
            
            return True
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load state: {str(e)}"
            )
            return False
            
    def verify_save_state(self, filepath):
        """Verify the integrity of a saved state file"""
        try:
            # Try loading the file
            format_type = os.path.splitext(filepath)[1]
            
            if format_type == '.bin':
                with open(filepath, 'rb') as f:
                    import pickle
                    state_data = pickle.load(f)
            elif format_type == '.json':
                with open(filepath, 'r') as f:
                    state_data = json.load(f)
            elif format_type == '.ckpt':
                import torch
                state_data = torch.load(filepath)
            else:
                return False, "Unsupported file format"
                
            # Check if data is compressed
            if isinstance(state_data, bytes):
                import zlib
                state_data = json.loads(zlib.decompress(state_data).decode())
            
            # Verify required fields
            required_fields = ['timestamp', 'nodes', 'connections', 'neural_weight']
            for field in required_fields:
                if field not in state_data:
                    return False, f"Missing required field: {field}"
            
            # Verify data types
            if not isinstance(state_data['nodes'], list):
                return False, "Invalid nodes data type"
            if not isinstance(state_data['connections'], list):
                return False, "Invalid connections data type"
            if not isinstance(state_data['neural_weight'], (int, float)):
                return False, "Invalid neural weight data type"
            
            return True, "Save state verified successfully"
            
        except Exception as e:
            return False, f"Verification failed: {str(e)}"

class NetworkCanvas(QtWidgets.QWidget):
    """Canvas for drawing the network visualization."""
    
    def __init__(self, parent=None):
        """Initialize the network canvas."""
        super().__init__(parent)
        self.parent = parent
        self.setMinimumSize(400, 300)
        self.dragging = False
        self.last_pos = None
        self.node_scale_factor = 1.0  # Default scale factor
        self.min_scale = 0.1
        self.max_scale = 5.0
        
        # Set mouse tracking for hover effects
        self.setMouseTracking(True)
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(15, 15, 25))
        self.setPalette(palette)
        
        # Animation properties
        self.animation_step = 0
        self.animation_timer = QtCore.QTimer(self)
        self.animation_timer.timeout.connect(lambda: self.update_network(True))
        self.animation_timer.start(50)  # 20 FPS
        
        # Node properties
        self.node_color = QtGui.QColor(212, 175, 55)  # Golden color
        self.node_size = 8
        self.node_pulse = 0
        self.pulse_direction = 1
        
    def update_network(self, animation_update=False):
        """
        Update the network visualization.
        
        Args:
            animation_update: Boolean indicating if this is an animation frame update
        """
        if animation_update:
            # Update pulse effect
            if self.pulse_direction > 0:
                self.node_pulse = min(1.0, self.node_pulse + 0.1)
                if self.node_pulse >= 1.0:
                    self.pulse_direction = -1
            else:
                self.node_pulse = max(0.0, self.node_pulse - 0.1)
                if self.node_pulse <= 0.0:
                    self.pulse_direction = 1
            
            self.animation_step = (self.animation_step + 1) % 360
        
        self.repaint()
    
    def paintEvent(self, event):
        """Paint the network visualization."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        
        # Draw connections first
        self._draw_connections(painter, width, height)
        
        # Then draw nodes
        self._draw_nodes(painter, width, height)
    
    def _draw_connections(self, painter, width, height):
        """Draw network connections."""
        if not self.parent.nodes:
            return
            
        # Get connection strength from slider
        strength = self.parent.strength_slider.value() / 100.0
        
        for conn in self.parent.connections:
            source = self.parent.nodes[conn['source']]
            target = self.parent.nodes[conn['target']]
            
            # Calculate connection opacity based on strength
            opacity = int(255 * strength * conn['weight'])
            
            # Create gradient for connection
            gradient = QtGui.QLinearGradient(
                source['x'], source['y'],
                target['x'], target['y']
            )
            gradient.setColorAt(0, QtGui.QColor(212, 175, 55, opacity))
            gradient.setColorAt(1, QtGui.QColor(212, 175, 55, opacity))
            
            # Draw connection line
            pen = QtGui.QPen(QtGui.QBrush(gradient), 2)
            pen.setStyle(Qt.SolidLine)
            painter.setPen(pen)
            painter.drawLine(
                source['x'], source['y'],
                target['x'], target['y']
            )
    
    def _draw_nodes(self, painter, width, height):
        """Draw network nodes."""
        if not self.parent.nodes:
            return
            
        for node in self.parent.nodes:
            # Calculate node size with pulse effect
            size = self.node_size * (1.0 + 0.3 * self.node_pulse)
            
            # Draw node glow
            glow = QtGui.QRadialGradient(node['x'], node['y'], size * 2)
            glow.setColorAt(0, QtGui.QColor(212, 175, 55, 100))
            glow.setColorAt(1, QtGui.QColor(212, 175, 55, 0))
            painter.setBrush(QtGui.QBrush(glow))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(
                node['x'] - size * 2,
                node['y'] - size * 2,
                size * 4,
                size * 4
            )
            
            # Draw node
            painter.setBrush(QtGui.QBrush(self.node_color))
            painter.setPen(QtGui.QPen(Qt.white, 1))
            painter.drawEllipse(
                node['x'] - size,
                node['y'] - size,
                size * 2,
                size * 2
            )
    
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.LeftButton:
            # Store initial position for dragging
            self.last_pos = event.pos()
            self.dragging = True
            
            # Check if clicked on a node
            clicked_node = self._find_node_at_pos(event.pos())
            if clicked_node:
                self.parent.selected_node = clicked_node
                if self.parent.on_node_selected:
                    self.parent.on_node_selected(clicked_node)
            else:
                self.parent.selected_node = None
            
            self.update()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for dragging nodes and hover effects."""
        if self.dragging and self.parent.selected_node:
            # Update node position while dragging
            self.parent.selected_node['x'] = event.pos().x()
            self.parent.selected_node['y'] = event.pos().y()
            self.update()
        else:
            # Check for hover effects
            hover_node = self._find_node_at_pos(event.pos())
            if hover_node:
                self.parent.status_label.setText(f"Node: {hover_node.get('name', f'Node {hover_node['id']}')} | Type: {hover_node['type']}")
            elif not self.parent.selected_node:
                self.parent.status_label.setText("Ready")
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events to end node dragging."""
        self.dragging = False
        self.update()
    
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        zoom_factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        self.node_scale_factor *= zoom_factor
        self.edge_scale_factor *= zoom_factor
        self.update()
    
    def _find_node_at_pos(self, pos):
        """Find a node at the given position."""
        if not self.parent.nodes:
            return None
            
        for node in self.parent.nodes:
            dx = node['x'] - pos.x()
            dy = node['y'] - pos.y()
            if (dx * dx + dy * dy) <= (self.node_size * self.node_size):
                return node
        
        return None

    def update(self):
        """Redraw the network visualization."""
        self.repaint()
    
    def set_node_properties(self, **properties):
        """
        Set multiple node properties at once.
        
        Args:
            **properties: Dictionary of properties to update:
                - color: QColor for node color
                - size: Base size of nodes
                - pulse: Pulse effect value
                - pulse_direction: Direction of pulse animation
                - scale_factor: Overall scale factor for nodes
        """
        for prop, value in properties.items():
            if prop == 'color':
                self.node_color = value
            elif prop == 'size':
                self.node_size = value
            elif prop == 'pulse':
                self.node_pulse = value
            elif prop == 'pulse_direction':
                self.pulse_direction = value
            elif prop == 'scale_factor':
                self.node_scale_factor = value
        self.update()

    def adjust_view(self, action, value=None):
        """
        Adjust the view based on the specified action.
        
        Args:
            action: String indicating the action ('zoom_in', 'zoom_out', 'reset', 'scale')
            value: Optional value for scaling
        """
        if action == 'zoom_in':
            self.node_scale_factor *= 1.2
            self.edge_scale_factor *= 1.2
        elif action == 'zoom_out':
            self.node_scale_factor /= 1.2
            self.edge_scale_factor /= 1.2
        elif action == 'reset':
            self.node_scale_factor = 1.0
            self.edge_scale_factor = 1.0
        elif action == 'scale' and value is not None:
            self.node_scale_factor = value
            self.edge_scale_factor = value
        self.update()

    # Network data setters
    def set_node_positions(self, positions):
        """Set the positions of the network nodes."""
        self.parent.nodes = positions
        self.update()
    
    def set_connections(self, connections):
        """Set the connections of the network."""
        self.parent.connections = connections
        self.update()
    
    def set_neural_weight(self, weight):
        """Set the neural network weight for network rendering."""
        self.parent.neural_weight = weight
        self.update()
    
    # UI control setters
    def set_strength_slider(self, value):
        """Set the value of the connection strength slider."""
        self.parent.strength_slider.setValue(value)
        self.update()
    
    def set_mode_combo(self, text):
        """Set the selected mode of the display mode combo box."""
        self.parent.mode_combo.setCurrentText(text)
        self.update()
    
    def set_time_combo(self, text):
        """Set the selected time scale of the time combo box."""
        self.parent.time_combo.setCurrentText(text)
        self.update()
    
    def set_view_combo(self, text):
        """Set the selected view mode of the view combo box."""
        self.parent.view_combo.setCurrentText(text)
        self.update()
    
    def set_status_label(self, text):
        """Set the text of the status label."""
        self.parent.status_label.setText(text)
        self.update()
    
    def set_graph_tabs(self, index):
        """Set the selected tab of the graph tabs."""
        self.parent.graph_tabs.setCurrentIndex(index)
        self.update()