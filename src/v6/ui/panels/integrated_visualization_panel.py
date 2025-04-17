"""
Integrated Visualization Panel

Brings together multiple visualization components into a unified
page-based interface similar to the V5 Fractal Echo Visualization.
"""

import logging
from pathlib import Path

# Add project root to path if needed
import sys
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    # Import Qt compatibility layer from V5
    from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui, Qt, Signal, Slot
    from src.v5.ui.qt_compat import get_widgets, get_gui, get_core
except ImportError:
    logging.warning("V5 Qt compatibility layer not found. Using direct PySide6 imports.")
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        from PySide6.QtCore import Qt, Signal, Slot
        
        # Simple compatibility functions
        def get_widgets():
            return QtWidgets
            
        def get_gui():
            return QtGui
            
        def get_core():
            return QtCore
    except ImportError:
        logging.error("PySide6 not found. Please install PySide6 or configure the V5 Qt compatibility layer.")
        sys.exit(1)

# Import the panel base and the paged visualization base
from ..panel_base import V6PanelBase
from .paged_visualization_panel import PagedVisualizationPanel

# Import the specific visualization panels
from .fractal_pattern_panel import FractalPatternPanel
from .network_visualization_panel import NetworkVisualizationPanel
from .node_consciousness_panel import NodeConsciousnessPanel

# Import memory synthesis panel (will be implemented if needed)
# from .memory_synthesis_panel import MemorySynthesisPanel

# Set up logging
logger = logging.getLogger(__name__)

class IntegratedVisualizationPanel(V6PanelBase):
    """
    Integrated visualization panel that combines multiple visualization components
    with page-based navigation similar to the V5 Fractal Echo Visualization.
    """
    
    # Signal emitted when a visualization page changes
    pageChanged = Signal(str)
    
    def __init__(self, socket_manager=None, parent=None):
        super().__init__(parent)
        self.socket_manager = socket_manager
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface components"""
        # Create main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create paged visualization panel
        self.paged_viz = PagedVisualizationPanel(self.socket_manager)
        self.paged_viz.pageChanged.connect(self.on_page_changed)
        
        # Create visualization panels
        self.fractal_panel = FractalPatternPanel(self.socket_manager)
        self.network_panel = NetworkVisualizationPanel(self.socket_manager)
        self.node_panel = NodeConsciousnessPanel(self.socket_manager)
        
        # Create a simple memory synthesis panel
        self.memory_panel = self.create_memory_synthesis_panel()
        
        # Add panels to paged visualization
        self.paged_viz.add_page("fractal", "Fractal Pattern Visualization", self.fractal_panel)
        self.paged_viz.add_page("network", "Neural Network Visualization", self.network_panel)
        self.paged_viz.add_page("node", "Node Consciousness", self.node_panel)
        self.paged_viz.add_page("memory", "Memory Synthesis", self.memory_panel)
        
        # Add paged visualization to layout
        layout.addWidget(self.paged_viz)
    
    def create_memory_synthesis_panel(self):
        """Create a simple memory synthesis panel"""
        panel = QtWidgets.QWidget()
        panel.setObjectName("memory_synthesis_panel")
        
        # Create layout
        layout = QtWidgets.QVBoxLayout(panel)
        
        # Create header widget
        header = QtWidgets.QWidget()
        header.setFixedHeight(40)
        header.setStyleSheet("""
            background-color: rgba(26, 38, 52, 220);
            border-bottom: 1px solid rgba(52, 73, 94, 150);
        """)
        
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(10, 0, 10, 0)
        
        # Topic selector
        topic_label = QtWidgets.QLabel("Topic:")
        topic_label.setStyleSheet("color: #ECF0F1; font-size: 13px;")
        
        topic_combo = QtWidgets.QComboBox()
        topic_combo.addItems(["neural_networks", "consciousness", "pattern_recognition", "emergence"])
        topic_combo.setStyleSheet("""
            background-color: rgba(52, 73, 94, 180);
            color: white;
            padding: 4px;
            border: 1px solid rgba(52, 152, 219, 120);
            border-radius: 4px;
            min-width: 180px;
        """)
        
        # Depth selector
        depth_label = QtWidgets.QLabel("Depth:")
        depth_label.setStyleSheet("color: #ECF0F1; font-size: 13px;")
        
        depth_spin = QtWidgets.QSpinBox()
        depth_spin.setRange(1, 5)
        depth_spin.setValue(3)
        depth_spin.setStyleSheet("""
            background-color: rgba(52, 73, 94, 180);
            color: white;
            padding: 4px;
            border: 1px solid rgba(52, 152, 219, 120);
            border-radius: 4px;
        """)
        
        # Synthesize button
        synthesize_btn = QtWidgets.QPushButton("Synthesize")
        synthesize_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(41, 128, 185, 180);
                color: white;
                border-radius: 4px;
                padding: 5px 15px;
                font-size: 13px;
                border: 1px solid rgba(52, 152, 219, 120);
            }
            QPushButton:hover {
                background-color: rgba(52, 152, 219, 200);
                border: 1px solid rgba(52, 152, 219, 180);
            }
        """)
        
        # Add widgets to header layout
        header_layout.addWidget(topic_label)
        header_layout.addWidget(topic_combo)
        header_layout.addSpacing(15)
        header_layout.addWidget(depth_label)
        header_layout.addWidget(depth_spin)
        header_layout.addSpacing(15)
        header_layout.addWidget(synthesize_btn)
        header_layout.addStretch(1)
        
        # Add header to main layout
        layout.addWidget(header)
        
        # Create content area with dark background
        content = QtWidgets.QWidget()
        content.setStyleSheet("""
            background-color: rgba(16, 26, 40, 180);
        """)
        
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(15, 15, 15, 15)
        
        # Core Insights section
        insights_label = QtWidgets.QLabel("Core Insights")
        insights_label.setStyleSheet("""
            color: #3498DB;
            font-weight: bold;
            font-size: 16px;
            border-bottom: 1px solid rgba(52, 152, 219, 150);
            padding-bottom: 5px;
        """)
        
        insights_text = QtWidgets.QTextEdit()
        insights_text.setReadOnly(True)
        insights_text.setStyleSheet("""
            background-color: rgba(44, 62, 80, 150);
            color: #ECF0F1;
            border: 1px solid rgba(52, 73, 94, 150);
            border-radius: 4px;
            padding: 10px;
        """)
        insights_text.setText(
            "The concept of neural_networks appears frequently in neural processing discussions\n"
            "neural_networks demonstrates connections to language understanding\n"
            "Pattern recognition related to neural_networks shows high coherence\n"
            "Processing neural_networks activates both logical and intuitive pathways"
        )
        
        # Related Topics section
        topics_label = QtWidgets.QLabel("Related Topics")
        topics_label.setStyleSheet("""
            color: #3498DB;
            font-weight: bold;
            font-size: 16px;
            border-bottom: 1px solid rgba(52, 152, 219, 150);
            padding-bottom: 5px;
            margin-top: 15px;
        """)
        
        # Create table for related topics
        topics_table = QtWidgets.QTableWidget(2, 3)
        topics_table.setStyleSheet("""
            background-color: rgba(44, 62, 80, 150);
            color: #ECF0F1;
            border: 1px solid rgba(52, 73, 94, 150);
            border-radius: 4px;
            gridline-color: rgba(52, 73, 94, 150);
        """)
        
        topics_table.setHorizontalHeaderLabels(["#", "Topic", "Relevance"])
        topics_table.horizontalHeader().setStyleSheet("""
            QHeaderView::section {
                background-color: rgba(52, 73, 94, 180);
                color: #ECF0F1;
                padding: 5px;
                border: 1px solid rgba(44, 62, 80, 200);
            }
        """)
        
        # Add sample data
        topics_table.setItem(0, 0, QtWidgets.QTableWidgetItem("1"))
        topics_table.setItem(0, 1, QtWidgets.QTableWidgetItem("neural_networks_related_0"))
        topics_table.setItem(0, 2, QtWidgets.QTableWidgetItem("0.84"))
        
        topics_table.setItem(1, 0, QtWidgets.QTableWidgetItem("2"))
        topics_table.setItem(1, 1, QtWidgets.QTableWidgetItem("neural_networks_related_1"))
        topics_table.setItem(1, 2, QtWidgets.QTableWidgetItem("0.72"))
        
        # Set column widths
        topics_table.setColumnWidth(0, 40)
        topics_table.setColumnWidth(1, 300)
        topics_table.setColumnWidth(2, 100)
        
        # Add widgets to content layout
        content_layout.addWidget(insights_label)
        content_layout.addWidget(insights_text)
        content_layout.addWidget(topics_label)
        content_layout.addWidget(topics_table)
        
        # Add content area to main layout
        layout.addWidget(content, 1)  # 1 = stretch factor
        
        return panel
    
    def on_page_changed(self, page_id):
        """Handle page change events"""
        logger.info(f"Page changed to: {page_id}")
        
        # Emit signal
        self.pageChanged.emit(page_id)
        
        # Update animation states based on visibility
        self.update_animation_states(page_id)
    
    def update_animation_states(self, active_page_id):
        """Update animation states for all panels based on visibility"""
        # Pause/resume animations based on visibility to save resources
        if hasattr(self.fractal_panel, 'animation_timer'):
            if active_page_id == "fractal":
                if hasattr(self.fractal_panel, 'start_animation'):
                    self.fractal_panel.start_animation()
            else:
                if hasattr(self.fractal_panel, 'stop_animation'):
                    self.fractal_panel.stop_animation()
        
        if hasattr(self.network_panel, 'animation_timer'):
            if active_page_id == "network":
                if hasattr(self.network_panel, 'start_animation'):
                    self.network_panel.start_animation()
            else:
                if hasattr(self.network_panel, 'stop_animation'):
                    self.network_panel.stop_animation()
        
        if hasattr(self.node_panel, 'animation_timer'):
            if active_page_id == "node":
                if hasattr(self.node_panel, 'start_animation'):
                    self.node_panel.start_animation()
            else:
                if hasattr(self.node_panel, 'stop_animation'):
                    self.node_panel.stop_animation()
    
    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        
        # Update active page if needed
        if hasattr(self.paged_viz, 'current_page'):
            self.update_animation_states(self.paged_viz.current_page)
    
    def showEvent(self, event):
        """Handle show events"""
        super().showEvent(event)
        
        # Update active page when shown
        if hasattr(self.paged_viz, 'current_page'):
            self.update_animation_states(self.paged_viz.current_page)
    
    def hideEvent(self, event):
        """Handle hide events"""
        # Pause all animations when hidden
        if hasattr(self.fractal_panel, 'stop_animation'):
            self.fractal_panel.stop_animation()
        
        if hasattr(self.network_panel, 'stop_animation'):
            self.network_panel.stop_animation()
        
        if hasattr(self.node_panel, 'stop_animation'):
            self.node_panel.stop_animation()
        
        super().hideEvent(event) 