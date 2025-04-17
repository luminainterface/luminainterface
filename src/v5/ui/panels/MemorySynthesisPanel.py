from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui, Qt, Signal, Slot
from src.v5.ui.qt_compat import Signal
from src.v5.ui.qt_compat import Slot

"""
Memory Synthesis Panel for the V5 Fractal Echo Visualization System.

This panel displays synthesized memories from the Language Memory System
and provides an interface for exploring memory relationships.
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

# Import language memory integration
try:
    from src.language_memory_api_compat import memory_api
    HAS_MEMORY_API = True
except ImportError:
    HAS_MEMORY_API = False


class MemorySynthesisPanel(QtWidgets.QtWidgets.QWidget):
    """Panel for visualizing and interacting with the Language Memory System."""
    
    # Signal emitted when a memory item is selected
    memory_selected = Signal(dict)
    
    def __init__(self, socket_manager=None):
        """
        Initialize the Memory Synthesis Panel.
        
        Args:
            socket_manager: Optional socket manager for plugin communication
        """
        super().__init__()
        self.socket_manager = socket_manager
        self.current_topic = None
        self.current_depth = 3
        self.synthesis_results = {}
        
        # Initialize UI
        self.initUI()
        
        # Connect to memory API if available
        if HAS_MEMORY_API:
            memory_api.signals.topics_updated.connect(self.handle_topics_updated)
            memory_api.signals.error_occurred.connect(self.handle_error)
    
    def initUI(self):
        """Initialize the user interface."""
        # Panel layout
        layout = QtWidgets.QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Panel title
        title_label = QtWidgets.QtWidgets.QLabel("Memory Synthesis")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #FFFFFF;")
        layout.addWidget(title_label)
        
        # Topic selection area
        topic_layout = QtWidgets.QtWidgets.QHBoxLayout()
        
        topic_label = QtWidgets.QtWidgets.QLabel("Topic:")
        self.topic_combo = QtWidgets.QtWidgets.QComboBox()
        self.topic_combo.setEditable(True)
        self.topic_combo.setMinimumWidth(200)
        self.depth_spinner = QtWidgets.QtWidgets.QSpinBox()
        self.depth_spinner.setRange(1, 5)
        self.depth_spinner.setValue(3)
        self.depth_spinner.setPrefix("Depth: ")
        self.synthesize_button = QtWidgets.QtWidgets.QPushButton("Synthesize")
        
        topic_layout.addWidget(topic_label)
        topic_layout.addWidget(self.topic_combo, 1)
        topic_layout.addWidget(self.depth_spinner)
        topic_layout.addWidget(self.synthesize_button)
        
        layout.addLayout(topic_layout)
        
        # Splitter for resizable sections
        splitter = QtWidgets.QSplitter(Qt.Vertical)
        
        # Insights area
        insights_widget = QtWidgets.QtWidgets.QWidget()
        insights_layout = QtWidgets.QtWidgets.QVBoxLayout(insights_widget)
        insights_layout.setContentsMargins(0, 5, 0, 5)
        
        insights_header = QtWidgets.QtWidgets.QLabel("Core Insights")
        insights_header.setStyleSheet("font-weight: bold;")
        insights_layout.addWidget(insights_header)
        
        self.insights_list = QtWidgets.QListWidget()
        self.insights_list.setAlternatingRowColors(True)
        insights_layout.addWidget(self.insights_list)
        
        # Related topics area
        related_widget = QtWidgets.QtWidgets.QWidget()
        related_layout = QtWidgets.QtWidgets.QVBoxLayout(related_widget)
        related_layout.setContentsMargins(0, 5, 0, 5)
        
        related_header = QtWidgets.QtWidgets.QLabel("Related Topics")
        related_header.setStyleSheet("font-weight: bold;")
        related_layout.addWidget(related_header)
        
        self.related_table = QtWidgets.QTableWidget(0, 2)
        self.related_table.setHorizontalHeaderLabels(["Topic", "Relevance"])
        self.related_table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        self.related_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.related_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        related_layout.addWidget(self.related_table)
        
        # Core understanding area
        understanding_widget = QtWidgets.QtWidgets.QWidget()
        understanding_layout = QtWidgets.QtWidgets.QVBoxLayout(understanding_widget)
        understanding_layout.setContentsMargins(0, 5, 0, 5)
        
        understanding_header = QtWidgets.QtWidgets.QLabel("Core Understanding")
        understanding_header.setStyleSheet("font-weight: bold;")
        understanding_layout.addWidget(understanding_header)
        
        self.understanding_text = QtWidgets.QtWidgets.QTextEdit()
        self.understanding_text.setReadOnly(True)
        understanding_layout.addWidget(self.understanding_text)
        
        # Add widgets to splitter
        splitter.addWidget(insights_widget)
        splitter.addWidget(related_widget)
        splitter.addWidget(understanding_widget)
        
        # Set initial sizes
        splitter.setSizes([200, 200, 100])
        
        layout.addWidget(splitter, 1)
        
        # Status area
        self.status_label = QtWidgets.QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet("font-style: italic; color: #999999;")
        layout.addWidget(self.status_label)
        
        # Connect signals
        self.synthesize_button.clicked.connect(self.on_synthesize_clicked)
        self.topic_combo.currentTextChanged.connect(self.on_topic_changed)
        self.depth_spinner.valueChanged.connect(self.on_depth_changed)
        self.related_table.cellDoubleClicked.connect(self.on_related_topic_selected)
        self.insights_list.itemClicked.connect(self.on_insight_selected)
        
        # Initial population
        self.populate_topic_list()
    
    def populate_topic_list(self):
        """Populate the topic combo box with available topics."""
        if not HAS_MEMORY_API:
            self.topic_combo.addItems(["consciousness", "neural_networks", "language", "memory"])
            return
            
        try:
            self.status_label.setText("Loading topics...")
            
            # Get topics from memory API
            topics = memory_api.get_topics(limit=20)
            
            # Add topics to combo box
            self.topic_combo.clear()
            if topics:
                self.topic_combo.addItems(topics)
                self.status_label.setText(f"Loaded {len(topics)} topics")
            else:
                self.topic_combo.addItems(["consciousness", "neural_networks", "language", "memory"])
                self.status_label.setText("Using default topics")
                
        except Exception as e:
            self.status_label.setText(f"Error loading topics: {str(e)}")
    
    def update_visualization(self):
        """Update the visualization with the latest data."""
        if self.current_topic:
            self.synthesize_memory(self.current_topic, self.current_depth)
    
    def on_topic_changed(self, topic):
        """Handle topic selection change."""
        self.current_topic = topic
    
    def on_depth_changed(self, depth):
        """Handle depth value change."""
        self.current_depth = depth
    
    def on_synthesize_clicked(self):
        """Handle synthesize button click."""
        topic = self.topic_combo.currentText().strip()
        depth = self.depth_spinner.value()
        
        if not topic:
            self.status_label.setText("Please enter a topic")
            return
            
        self.synthesize_memory(topic, depth)
    
    def on_related_topic_selected(self, row, column):
        """Handle selection of a related topic."""
        topic = self.related_table.item(row, 0).text()
        self.topic_combo.setCurrentText(topic)
        self.synthesize_memory(topic, self.current_depth)
    
    def on_insight_selected(self, item):
        """Handle selection of an insight."""
        insight = item.text()
        
        # Emit signal with selected insight info
        self.memory_selected.emit({
            "type": "insight",
            "content": insight,
            "topic": self.current_topic
        })
    
    def synthesize_memory(self, topic, depth=3):
        """
        Synthesize memory for the given topic.
        
        Args:
            topic: The topic to synthesize
            depth: How deep to explore related topics
        """
        self.current_topic = topic
        self.current_depth = depth
        
        # Update UI state
        self.status_label.setText(f"Synthesizing {topic}...")
        self.synthesize_button.setEnabled(False)
        
        if not HAS_MEMORY_API:
            # Generate mock data
            import random
            from time import sleep
            
            # Simulate processing delay
            QtCore.QTimer.singleShot(500, lambda: self._handle_mock_synthesis(topic, depth))
            return
            
        try:
            # Asynchronous processing
            memory_api.synthesize_topic(topic, depth, async_mode=True)
            
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.synthesize_button.setEnabled(True)
    
    def handle_topics_updated(self, response):
        """
        Handle topic synthesis response from the memory API.
        
        Args:
            response: The synthesis response
        """
        if "error" in response:
            self.status_label.setText(f"Error: {response['error']}")
            self.synthesize_button.setEnabled(True)
            return
            
        # Store results
        self.synthesis_results = response
        
        # Update the UI with the results
        self.update_ui_with_synthesis(response)
        
        # Update status
        self.status_label.setText(f"Synthesized {response.get('topic', '')}")
        self.synthesize_button.setEnabled(True)
    
    def handle_error(self, error_message):
        """
        Handle error from the memory API.
        
        Args:
            error_message: The error message
        """
        self.status_label.setText(f"Error: {error_message}")
        self.synthesize_button.setEnabled(True)
    
    def _handle_mock_synthesis(self, topic, depth):
        """Generate and display mock synthesis results."""
        import random
        
        # Create mock insights
        insights = [
            f"The concept of {topic} involves pattern recognition",
            f"{topic} demonstrates emergent properties",
            f"Information about {topic} is stored across multiple memory components",
            f"Neural patterns related to {topic} show high coherence",
            f"Processing {topic} activates both concrete and abstract reasoning pathways"
        ]
        
        # Create mock related topics
        related_topics = []
        for i in range(3 + depth):
            related_topics.append({
                "topic": f"{topic}_related_{i}",
                "relevance": round(random.uniform(0.3, 0.95), 2)
            })
        
        # Create mock response
        response = {
            "topic": topic,
            "depth": depth,
            "synthesized_memory": {
                "core_understanding": f"{topic} is a fundamental concept in neural processing",
                "core_insights": insights,
                "source_count": random.randint(5, 20)
            },
            "related_topics": related_topics,
            "mock": True
        }
        
        # Store results
        self.synthesis_results = response
        
        # Update the UI with the results
        self.update_ui_with_synthesis(response)
        
        # Update status
        self.status_label.setText(f"Synthesized {topic} (mock data)")
        self.synthesize_button.setEnabled(True)
    
    def update_ui_with_synthesis(self, response):
        """
        Update the UI with synthesis results.
        
        Args:
            response: The synthesis response
        """
        # Clear existing data
        self.insights_list.clear()
        self.related_table.setRowCount(0)
        self.understanding_text.clear()
        
        # Extract data from response
        topic = response.get("topic", "")
        synthesized_memory = response.get("synthesized_memory", {})
        related_topics = response.get("related_topics", [])
        
        # Update insights list
        insights = synthesized_memory.get("core_insights", [])
        for insight in insights:
            item = QtWidgets.QListWidgetItem(insight)
            self.insights_list.addItem(item)
        
        # Update related topics table
        self.related_table.setRowCount(len(related_topics))
        for i, topic_data in enumerate(related_topics):
            if isinstance(topic_data, dict):
                topic_name = topic_data.get("topic", f"related_{i}")
                relevance = topic_data.get("relevance", 0.5)
            else:
                topic_name = f"related_{i}"
                relevance = 0.5
                
            # Add topic name
            topic_item = QtWidgets.QTableWidgetItem(topic_name)
            self.related_table.setItem(i, 0, topic_item)
            
            # Add relevance score
            relevance_item = QtWidgets.QTableWidgetItem(f"{relevance:.2f}")
            relevance_item.setTextAlignment(Qt.AlignCenter)
            self.related_table.setItem(i, 1, relevance_item)
            
            # Set row background based on relevance
            color_val = int(255 - (relevance * 100))
            color = QtGui.QColor(20, 20, 20 + color_val)
            topic_item.setBackground(color)
            relevance_item.setBackground(color)
        
        # Update core understanding
        core_understanding = synthesized_memory.get("core_understanding", "")
        self.understanding_text.setText(core_understanding)
        
        # Emit signal with the current topic
        self.memory_selected.emit({
            "type": "topic",
            "content": topic,
            "related_topics": related_topics
        })
    
    def cleanup(self):
        """Clean up resources before closing."""
        pass 