"""
Node Manager UI for Neural Network Node Manager
"""

import sys
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QTextEdit, QTreeWidget,
    QTreeWidgetItem, QTabWidget, QProgressBar, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QIcon
from node_manager import NodeManager

class NodeManagerUI(QMainWindow):
    """Main window for Node Manager UI"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LUMINA Neural Network Node Manager")
        self.setMinimumSize(1280, 720)
        
        # Initialize components
        self.node_manager = NodeManager()
        self.logger = logging.getLogger("NodeManagerUI")
        self.update_timer = QTimer()
        
        # Initialize UI
        self._setup_ui()
        self._setup_logging()
        
        # Initialize node manager
        if not self.node_manager.initialize():
            QMessageBox.critical(self, "Error", "Failed to initialize Node Manager")
            sys.exit(1)
            
        # Start update timer
        self.update_timer.timeout.connect(self._update_ui)
        self.update_timer.start(1000)  # Update every second
        
    def _setup_ui(self):
        """Setup the main UI components"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create left panel (node tree and controls)
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, stretch=1)
        
        # Create right panel (tabs and details)
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, stretch=2)
        
    def _create_left_panel(self) -> QWidget:
        """Create the left panel with node tree and controls"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Node type selector
        type_layout = QHBoxLayout()
        type_label = QLabel("Node Type:")
        self.node_type_combo = QComboBox()
        self.node_type_combo.addItems(["RSEN", "HybridNode", "FractalNode"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.node_type_combo)
        layout.addLayout(type_layout)
        
        # Create node button
        create_btn = QPushButton("Create Node")
        create_btn.clicked.connect(self._create_node)
        layout.addWidget(create_btn)
        
        # Node tree
        self.node_tree = QTreeWidget()
        self.node_tree.setHeaderLabels(["Nodes"])
        self.node_tree.itemSelectionChanged.connect(self._node_selected)
        layout.addWidget(self.node_tree)
        
        # Node controls
        controls_layout = QHBoxLayout()
        self.activate_btn = QPushButton("Activate")
        self.activate_btn.clicked.connect(self._activate_selected_node)
        self.deactivate_btn = QPushButton("Deactivate")
        self.deactivate_btn.clicked.connect(self._deactivate_selected_node)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self._remove_selected_node)
        
        controls_layout.addWidget(self.activate_btn)
        controls_layout.addWidget(self.deactivate_btn)
        controls_layout.addWidget(self.remove_btn)
        layout.addLayout(controls_layout)
        
        return panel
        
    def _create_right_panel(self) -> QWidget:
        """Create the right panel with tabs"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Status tab
        status_tab = self._create_status_tab()
        self.tab_widget.addTab(status_tab, "Status")
        
        # Details tab
        details_tab = self._create_details_tab()
        self.tab_widget.addTab(details_tab, "Details")
        
        # Logs tab
        logs_tab = self._create_logs_tab()
        self.tab_widget.addTab(logs_tab, "Logs")
        
        layout.addWidget(self.tab_widget)
        return panel
        
    def _create_status_tab(self) -> QWidget:
        """Create the status tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # System status
        status_group = QWidget()
        status_layout = QVBoxLayout(status_group)
        
        self.total_nodes_label = QLabel("Total Nodes: 0")
        self.active_nodes_label = QLabel("Active Nodes: 0")
        self.system_status_label = QLabel("System Status: Initializing")
        
        status_layout.addWidget(self.total_nodes_label)
        status_layout.addWidget(self.active_nodes_label)
        status_layout.addWidget(self.system_status_label)
        
        layout.addWidget(status_group)
        
        # Node status grid
        self.node_status_tree = QTreeWidget()
        self.node_status_tree.setHeaderLabels(["Node", "Status", "Activation"])
        layout.addWidget(self.node_status_tree)
        
        return tab
        
    def _create_details_tab(self) -> QWidget:
        """Create the details tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Node details
        self.node_details = QTextEdit()
        self.node_details.setReadOnly(True)
        layout.addWidget(self.node_details)
        
        return tab
        
    def _create_logs_tab(self) -> QWidget:
        """Create the logs tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Log viewer
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        layout.addWidget(self.log_viewer)
        
        return tab
        
    def _setup_logging(self):
        """Setup logging handlers"""
        # Create logs directory
        Path('logs').mkdir(exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler('logs/node_manager_ui.log')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # UI handler
        class QTextEditHandler(logging.Handler):
            def __init__(self, text_edit):
                super().__init__()
                self.text_edit = text_edit
                
            def emit(self, record):
                msg = self.format(record)
                self.text_edit.append(msg)
                
        ui_handler = QTextEditHandler(self.log_viewer)
        ui_handler.setLevel(logging.INFO)
        ui_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ui_handler.setFormatter(ui_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(ui_handler)
        
    def _create_node(self):
        """Create a new node"""
        node_type = self.node_type_combo.currentText()
        try:
            node = self.node_manager.create_node(node_type)
            if node:
                self.logger.info(f"Created node: {node.node_id}")
                self._update_node_tree()
            else:
                self.logger.error(f"Failed to create node of type: {node_type}")
        except Exception as e:
            self.logger.error(f"Error creating node: {str(e)}")
            
    def _node_selected(self):
        """Handle node selection"""
        items = self.node_tree.selectedItems()
        if items:
            node_id = items[0].text(0)
            node = self.node_manager.get_node(node_id)
            if node:
                # Update details
                status = node.get_status()
                details = "\n".join(f"{k}: {v}" for k, v in status.items())
                self.node_details.setText(details)
                
    def _activate_selected_node(self):
        """Activate the selected node"""
        items = self.node_tree.selectedItems()
        if items:
            node_id = items[0].text(0)
            if self.node_manager.activate_node(node_id):
                self.logger.info(f"Activated node: {node_id}")
            else:
                self.logger.error(f"Failed to activate node: {node_id}")
                
    def _deactivate_selected_node(self):
        """Deactivate the selected node"""
        items = self.node_tree.selectedItems()
        if items:
            node_id = items[0].text(0)
            if self.node_manager.deactivate_node(node_id):
                self.logger.info(f"Deactivated node: {node_id}")
            else:
                self.logger.error(f"Failed to deactivate node: {node_id}")
                
    def _remove_selected_node(self):
        """Remove the selected node"""
        items = self.node_tree.selectedItems()
        if items:
            node_id = items[0].text(0)
            if self.node_manager.remove_node(node_id):
                self.logger.info(f"Removed node: {node_id}")
                self._update_node_tree()
            else:
                self.logger.error(f"Failed to remove node: {node_id}")
                
    def _update_ui(self):
        """Update UI components"""
        try:
            # Update system status
            status = self.node_manager.get_system_status()
            self.total_nodes_label.setText(f"Total Nodes: {status['total_nodes']}")
            self.active_nodes_label.setText(f"Active Nodes: {status['active_nodes']}")
            
            # Update node status tree
            self._update_node_status_tree()
            
            # Update node tree if needed
            self._update_node_tree()
            
        except Exception as e:
            self.logger.error(f"Error updating UI: {str(e)}")
            
    def _update_node_tree(self):
        """Update the node tree"""
        try:
            self.node_tree.clear()
            for node_id in self.node_manager.active_nodes:
                item = QTreeWidgetItem([node_id])
                self.node_tree.addTopLevelItem(item)
        except Exception as e:
            self.logger.error(f"Error updating node tree: {str(e)}")
            
    def _update_node_status_tree(self):
        """Update the node status tree"""
        try:
            self.node_status_tree.clear()
            states = self.node_manager.get_all_node_states()
            
            for node_id, state in states.items():
                item = QTreeWidgetItem([
                    node_id,
                    state.get('status', 'unknown'),
                    str(state.get('activation_level', 0.0))
                ])
                self.node_status_tree.addTopLevelItem(item)
                
        except Exception as e:
            self.logger.error(f"Error updating node status tree: {str(e)}")
            
def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    window = NodeManagerUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 