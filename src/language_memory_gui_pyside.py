#!/usr/bin/env python3
"""
Language Memory GUI PySide6

A modern PySide6-based GUI for interacting with the language memory system.
Designed to integrate with the V5 Fractal Echo Visualization system.
"""

import os
import sys
import logging
import threading
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("language_memory_pyside.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("language_memory_gui_pyside")

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# PySide6 imports
try:
    from PySide6.QtCore import Qt, Signal, Slot, QSize, QTimer
    from PySide6.QtGui import QIcon, QFont, QPixmap, QPainter, QColor
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
        QLineEdit, QPushButton, QLabel, QTabWidget, QGridLayout, QScrollArea,
        QFrame, QSplitter, QTreeWidget, QTreeWidgetItem, QComboBox, QSpinBox,
        QMessageBox, QGroupBox, QRadioButton, QButtonGroup, QSizePolicy, QSpacerItem
    )
    logger.info("Successfully imported PySide6")
except ImportError as e:
    logger.error(f"Failed to import PySide6: {e}")
    print(f"Error: PySide6 is required but not installed.")
    print("Please install PySide6 with: pip install PySide6")
    sys.exit(1)

# Try to import required components
try:
    from src.language_memory_synthesis_integration import LanguageMemorySynthesisIntegration
    logger.info("Successfully imported LanguageMemorySynthesisIntegration")
except ImportError as e:
    logger.error(f"Failed to import LanguageMemorySynthesisIntegration: {str(e)}")
    logger.error("Please ensure the language_memory_synthesis_integration.py file exists")
    
    # Continue execution, but the app will show appropriate error messages

# Try to import V5 visualization components (optional)
try:
    from src.v5.frontend_socket_manager import FrontendSocketManager
    from src.v5.language_memory_integration import LanguageMemoryIntegrationPlugin
    HAS_V5_COMPONENTS = True
    logger.info("Successfully imported V5 visualization components")
except ImportError as e:
    HAS_V5_COMPONENTS = False
    logger.warning(f"V5 visualization components not available: {str(e)}")
    logger.warning("The application will run without V5 visualization features")


class MemoryTab(QWidget):
    """Tab for storing and retrieving memories"""
    
    def __init__(self, memory_system=None, parent=None):
        super().__init__(parent)
        self.memory_system = memory_system
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components"""
        # Main layout
        main_layout = QHBoxLayout(self)
        
        # Left side - Input new memories
        input_frame = QGroupBox("Store New Memory")
        input_layout = QVBoxLayout(input_frame)
        
        # Memory input area
        input_layout.addWidget(QLabel("Enter Memory Content:"))
        self.memory_text = QTextEdit()
        input_layout.addWidget(self.memory_text)
        
        # Topic and emotion inputs
        input_layout.addWidget(QLabel("Topic:"))
        self.topic_entry = QLineEdit()
        input_layout.addWidget(self.topic_entry)
        
        input_layout.addWidget(QLabel("Emotion:"))
        self.emotion_entry = QLineEdit()
        input_layout.addWidget(self.emotion_entry)
        
        input_layout.addWidget(QLabel("Keywords (comma separated):"))
        self.keywords_entry = QLineEdit()
        input_layout.addWidget(self.keywords_entry)
        
        # Store button
        self.store_button = QPushButton("Store Memory")
        self.store_button.clicked.connect(self.store_memory)
        input_layout.addWidget(self.store_button)
        
        # Right side - Search memories
        search_frame = QGroupBox("Search Memories")
        search_layout = QVBoxLayout(search_frame)
        
        # Search options
        search_options_layout = QHBoxLayout()
        search_options_layout.addWidget(QLabel("Search By:"))
        
        self.search_type_group = QButtonGroup(self)
        
        self.topic_radio = QRadioButton("Topic")
        self.topic_radio.setChecked(True)
        self.search_type_group.addButton(self.topic_radio, 1)
        search_options_layout.addWidget(self.topic_radio)
        
        self.keyword_radio = QRadioButton("Keyword")
        self.search_type_group.addButton(self.keyword_radio, 2)
        search_options_layout.addWidget(self.keyword_radio)
        
        self.text_radio = QRadioButton("Text")
        self.search_type_group.addButton(self.text_radio, 3)
        search_options_layout.addWidget(self.text_radio)
        
        search_layout.addLayout(search_options_layout)
        
        # Search query
        search_layout.addWidget(QLabel("Search Query:"))
        self.search_entry = QLineEdit()
        search_layout.addWidget(self.search_entry)
        
        # Search button
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_memories)
        search_layout.addWidget(self.search_button)
        
        # Results area
        search_layout.addWidget(QLabel("Search Results:"))
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        search_layout.addWidget(self.results_text)
        
        # Add frames to main layout
        main_layout.addWidget(input_frame)
        main_layout.addWidget(search_frame)
    
    def store_memory(self):
        """Store a new memory from user input"""
        # Get input values
        memory_content = self.memory_text.toPlainText().strip()
        topic = self.topic_entry.text().strip()
        emotion = self.emotion_entry.text().strip()
        keywords = [k.strip() for k in self.keywords_entry.text().split(",") if k.strip()]
        
        if not memory_content:
            QMessageBox.warning(self, "Input Required", "Please enter memory content")
            return
        
        if not self.memory_system:
            QMessageBox.warning(self, "System Unavailable", 
                              "Memory system is not available")
            return
        
        try:
            # Check if we have a conversation memory component
            if "conversation_memory" in self.memory_system.components:
                memory_component = self.memory_system.components["conversation_memory"]
                
                # Store the memory
                result = memory_component.store(
                    content=memory_content,
                    metadata={
                        "topic": topic,
                        "emotion": emotion,
                        "keywords": keywords,
                        "source": "gui_input"
                    }
                )
                
                # Clear input fields
                self.memory_text.clear()
                self.topic_entry.clear()
                self.emotion_entry.clear()
                self.keywords_entry.clear()
                
                QMessageBox.information(self, "Success", "Memory stored successfully")
                logger.info(f"Memory stored with ID: {result.get('id', 'unknown')}")
            else:
                QMessageBox.warning(self, "Component Missing", 
                                  "Conversation memory component is not available")
        except Exception as e:
            error_msg = f"Error storing memory: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
    
    def search_memories(self):
        """Search for memories based on user input"""
        search_query = self.search_entry.text().strip()
        
        # Get search type
        if self.topic_radio.isChecked():
            search_type = "topic"
        elif self.keyword_radio.isChecked():
            search_type = "keyword"
        elif self.text_radio.isChecked():
            search_type = "text"
        else:
            search_type = "topic"  # Default
        
        if not search_query:
            QMessageBox.warning(self, "Input Required", "Please enter a search query")
            return
        
        if not self.memory_system:
            QMessageBox.warning(self, "System Unavailable", 
                              "Memory system is not available")
            return
        
        try:
            # Check if we have a conversation memory component
            if "conversation_memory" in self.memory_system.components:
                memory_component = self.memory_system.components["conversation_memory"]
                
                # Perform search based on selected type
                if search_type == "topic":
                    results = memory_component.retrieve_by_topic(search_query)
                elif search_type == "keyword":
                    results = memory_component.retrieve_by_keyword(search_query)
                elif search_type == "text":
                    results = memory_component.search_text(search_query)
                else:
                    results = []
                
                # Display results
                self.results_text.clear()
                
                if results:
                    result_text = f"Found {len(results)} memories:\n\n"
                    
                    for i, memory in enumerate(results):
                        result_text += f"Memory {i+1}:\n"
                        result_text += f"Content: {memory.get('content', 'No content')}\n"
                        
                        if "metadata" in memory:
                            metadata = memory["metadata"]
                            result_text += f"Topic: {metadata.get('topic', 'N/A')}\n"
                            result_text += f"Emotion: {metadata.get('emotion', 'N/A')}\n"
                            
                            if "keywords" in metadata and metadata["keywords"]:
                                result_text += f"Keywords: {', '.join(metadata['keywords'])}\n"
                        
                        result_text += f"Timestamp: {memory.get('timestamp', 'N/A')}\n\n"
                    
                    self.results_text.setPlainText(result_text)
                else:
                    self.results_text.setPlainText("No results found.")
                
                logger.info(f"Search completed for '{search_query}' with {len(results)} results")
            else:
                QMessageBox.warning(self, "Component Missing", 
                                  "Conversation memory component is not available")
        except Exception as e:
            error_msg = f"Error searching memories: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(self, "Error", error_msg)


class SynthesisTab(QWidget):
    """Tab for synthesizing memories around topics"""
    
    def __init__(self, memory_system=None, parent=None):
        super().__init__(parent)
        self.memory_system = memory_system
        self.v5_integration = None
        self.init_ui()
        
        # Try to initialize V5 visualization if available
        if HAS_V5_COMPONENTS:
            self.init_v5_integration()
    
    def init_ui(self):
        """Initialize the UI components"""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Controls area
        controls_frame = QFrame()
        controls_layout = QHBoxLayout(controls_frame)
        
        # Topic input
        controls_layout.addWidget(QLabel("Topic to Synthesize:"))
        self.synthesis_topic_entry = QLineEdit()
        controls_layout.addWidget(self.synthesis_topic_entry)
        
        # Depth selector
        controls_layout.addWidget(QLabel("Depth:"))
        self.depth_spinner = QSpinBox()
        self.depth_spinner.setMinimum(1)
        self.depth_spinner.setMaximum(5)
        self.depth_spinner.setValue(3)
        controls_layout.addWidget(self.depth_spinner)
        
        # Synthesize button
        self.synthesize_button = QPushButton("Synthesize")
        self.synthesize_button.clicked.connect(self.synthesize_topic)
        controls_layout.addWidget(self.synthesize_button)
        
        main_layout.addWidget(controls_frame)
        
        # Create visualization layout that will hold either the standard results
        # or the V5 visualization if available
        self.visualization_layout = QVBoxLayout()
        
        # Results area (default, will be replaced with visualization if V5 is available)
        results_frame = QGroupBox("Synthesis Results")
        results_layout = QVBoxLayout(results_frame)
        
        self.synthesis_results = QTextEdit()
        self.synthesis_results.setReadOnly(True)
        results_layout.addWidget(self.synthesis_results)
        
        self.visualization_layout.addWidget(results_frame)
        main_layout.addLayout(self.visualization_layout)
    
    def init_v5_integration(self):
        """Initialize V5 visualization integration if available"""
        try:
            # Create a socket manager
            self.socket_manager = FrontendSocketManager()
            
            # Create a language memory integration plugin
            self.v5_integration = LanguageMemoryIntegrationPlugin(
                language_memory_synthesis=self.memory_system
            )
            
            # Register with socket manager
            self.socket_manager.register_plugin(self.v5_integration)
            
            # Try to create the visualization panel
            from src.v5.ui.panels.fractal_pattern_panel import FractalPatternPanel
            
            # Replace the text results with visualization
            self.clear_visualization_layout()
            
            # Create visualization frame
            viz_frame = QGroupBox("Memory Synthesis Visualization")
            viz_layout = QVBoxLayout(viz_frame)
            
            # Add the fractal pattern panel
            self.fractal_panel = FractalPatternPanel(self.socket_manager)
            viz_layout.addWidget(self.fractal_panel)
            
            self.visualization_layout.addWidget(viz_frame)
            
            logger.info("V5 visualization integration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize V5 visualization: {str(e)}")
            # Fall back to standard text display (already set up)
    
    def clear_visualization_layout(self):
        """Clear all widgets from the visualization layout"""
        while self.visualization_layout.count():
            item = self.visualization_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    
    def synthesize_topic(self):
        """Synthesize memories around a topic"""
        topic = self.synthesis_topic_entry.text().strip()
        depth = self.depth_spinner.value()
        
        if not topic:
            QMessageBox.warning(self, "Input Required", "Please enter a topic to synthesize")
            return
        
        if not self.memory_system:
            QMessageBox.warning(self, "System Unavailable", 
                              "Memory system is not available")
            return
        
        # Disable button while processing
        self.synthesize_button.setEnabled(False)
        self.synthesis_results.setPlainText(f"Synthesizing topic '{topic}'... Please wait.")
        
        # Run synthesis in a separate thread
        threading.Thread(target=self._run_synthesis, args=(topic, depth)).start()
    
    def _run_synthesis(self, topic, depth):
        """
        Run the synthesis operation in a background thread
        
        Args:
            topic: Topic to synthesize
            depth: Search depth
        """
        try:
            # Perform the synthesis
            results = self.memory_system.synthesize_topic(topic, depth)
            
            # If we have V5 integration, also process with that
            if self.v5_integration:
                self.v5_integration.process_language_data(topic, depth)
            
            # Update the UI on the main thread
            QApplication.instance().postEvent(
                self, 
                SynthesisResultEvent(results)
            )
        except Exception as e:
            error_msg = f"Error during synthesis: {str(e)}"
            logger.error(error_msg)
            
            # Signal error on main thread
            QApplication.instance().postEvent(
                self, 
                SynthesisErrorEvent(error_msg)
            )
    
    def event(self, event):
        """Handle custom events"""
        if isinstance(event, SynthesisResultEvent):
            self._display_synthesis_results(event.results)
            self.synthesize_button.setEnabled(True)
            return True
        elif isinstance(event, SynthesisErrorEvent):
            QMessageBox.critical(self, "Synthesis Error", event.error_message)
            self.synthesize_button.setEnabled(True)
            return True
        return super().event(event)
    
    def _display_synthesis_results(self, results):
        """
        Display the synthesis results in the GUI
        
        Args:
            results: The results from the synthesis operation
        """
        self.synthesis_results.clear()
        
        if "synthesis_results" in results and results["synthesis_results"]:
            synthesis = results["synthesis_results"]["synthesized_memory"]
            
            result_text = "SYNTHESIS RESULTS\n"
            result_text += "================\n\n"
            
            result_text += f"Topic: {', '.join(synthesis['topics'])}\n"
            result_text += f"ID: {synthesis['id']}\n"
            result_text += f"Created: {synthesis['timestamp']}\n\n"
            
            result_text += "CORE UNDERSTANDING:\n"
            result_text += f"{synthesis['core_understanding']}\n\n"
            
            result_text += "NOVEL INSIGHTS:\n"
            for insight in synthesis['novel_insights']:
                result_text += f"• {insight}\n"
            
            result_text += "\nCOMPONENT CONTRIBUTIONS:\n"
            for component, count in synthesis['component_contributions'].items():
                result_text += f"• {component}: {count} items\n"
            
            # Display related topics if available
            if "related_topics" in results["synthesis_results"]:
                related = results["synthesis_results"]["related_topics"]
                result_text += f"\nRELATED TOPICS: {', '.join(related)}\n"
            
            self.synthesis_results.setPlainText(result_text)
        
        elif "errors" in results and results["errors"]:
            result_text = "ERROR DURING SYNTHESIS:\n\n"
            for error in results["errors"]:
                result_text += f"• {error}\n"
            
            self.synthesis_results.setPlainText(result_text)
        else:
            self.synthesis_results.setPlainText("No synthesis results generated.")


class StatsTab(QWidget):
    """Tab for displaying memory system statistics"""
    
    def __init__(self, memory_system=None, parent=None):
        super().__init__(parent)
        self.memory_system = memory_system
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components"""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Controls area
        controls_frame = QFrame()
        controls_layout = QHBoxLayout(controls_frame)
        
        # Refresh button
        self.refresh_button = QPushButton("Refresh Statistics")
        self.refresh_button.clicked.connect(self.refresh_stats)
        controls_layout.addWidget(self.refresh_button)
        
        # Add stretch to push controls to the left
        controls_layout.addStretch()
        
        main_layout.addWidget(controls_frame)
        
        # Stats tree
        self.stats_tree = QTreeWidget()
        self.stats_tree.setHeaderLabels(["Statistic", "Value"])
        self.stats_tree.setColumnWidth(0, 250)
        main_layout.addWidget(self.stats_tree)
        
        # Initial stats load
        self.refresh_stats()
    
    def refresh_stats(self):
        """Refresh the statistics display"""
        if not self.memory_system:
            self.stats_tree.clear()
            root = QTreeWidgetItem(self.stats_tree, ["Memory System", "Unavailable"])
            self.stats_tree.addTopLevelItem(root)
            return
            
        try:
            # Clear existing items
            self.stats_tree.clear()
            
            # Get current stats
            stats = self.memory_system.get_stats()
            
            # Add synthesis stats
            synthesis_node = QTreeWidgetItem(["Synthesis Statistics", ""])
            self.stats_tree.addTopLevelItem(synthesis_node)
            
            synthesis_stats = stats.get("synthesis_stats", {})
            
            synthesis_count = QTreeWidgetItem(["Synthesis Count", 
                                             str(synthesis_stats.get("synthesis_count", 0))])
            synthesis_node.addChild(synthesis_count)
            
            topics = synthesis_stats.get("topics_synthesized", [])
            topics_str = ", ".join(topics[:5]) + (", ..." if len(topics) > 5 else "")
            topics_item = QTreeWidgetItem(["Topics Synthesized", 
                                         f"{len(topics)} topics" if topics else "None"])
            synthesis_node.addChild(topics_item)
            
            timestamp = synthesis_stats.get("last_synthesis_timestamp", "Never")
            timestamp_item = QTreeWidgetItem(["Last Synthesis", timestamp])
            synthesis_node.addChild(timestamp_item)
            
            # Add language memory stats
            language_node = QTreeWidgetItem(["Language Memory Statistics", ""])
            self.stats_tree.addTopLevelItem(language_node)
            
            language_stats = stats.get("language_memory_stats", {})
            
            memory_count = QTreeWidgetItem(["Memory Count", 
                                          str(language_stats.get("memory_count", 0))])
            language_node.addChild(memory_count)
            
            sentence_count = QTreeWidgetItem(["Sentence Count", 
                                            str(language_stats.get("sentence_count", 0))])
            language_node.addChild(sentence_count)
            
            topics = language_stats.get("topics", [])
            topics_str = ", ".join([t[0] for t in topics[:5]]) + (", ..." if len(topics) > 5 else "")
            topics_item = QTreeWidgetItem(["Top Topics", topics_str if topics else "None"])
            language_node.addChild(topics_item)
            
            # Add component stats
            components_node = QTreeWidgetItem(["Components", ""])
            self.stats_tree.addTopLevelItem(components_node)
            
            component_stats = stats.get("component_stats", {})
            for component, component_data in component_stats.items():
                status = "Active" if component_data.get("active", False) else "Inactive"
                component_item = QTreeWidgetItem([component, status])
                components_node.addChild(component_item)
            
            # Expand all items
            self.stats_tree.expandAll()
            
            logger.info("Statistics refreshed")
        except Exception as e:
            error_msg = f"Error refreshing statistics: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(self, "Statistics Error", error_msg)


# Custom events for thread communication
class SynthesisResultEvent(QApplication.instance().Event):
    """Event for synthesis results"""
    
    EVENT_TYPE = QApplication.instance().registerEventType()
    
    def __init__(self, results):
        super().__init__(self.EVENT_TYPE)
        self.results = results


class SynthesisErrorEvent(QApplication.instance().Event):
    """Event for synthesis errors"""
    
    EVENT_TYPE = QApplication.instance().registerEventType()
    
    def __init__(self, error_message):
        super().__init__(self.EVENT_TYPE)
        self.error_message = error_message


class LanguageMemoryMainWindow(QMainWindow):
    """Main window for the Language Memory System"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize core components
        self.memory_system = None
        self.init_memory_system()
        
        # Set up the UI
        self.init_ui()
        
        logger.info("Language Memory GUI initialized")
    
    def init_memory_system(self):
        """Initialize the memory system integration"""
        try:
            self.memory_system = LanguageMemorySynthesisIntegration()
            logger.info("Memory system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {str(e)}")
            # We'll continue without the memory system and show an error message
            # when the user tries to use a feature that requires it
    
    def init_ui(self):
        """Initialize the UI components"""
        # Window setup
        self.setWindowTitle("Language Memory System")
        self.resize(1000, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.memory_tab = MemoryTab(self.memory_system)
        self.tab_widget.addTab(self.memory_tab, "Memory")
        
        self.synthesis_tab = SynthesisTab(self.memory_system)
        self.tab_widget.addTab(self.synthesis_tab, "Synthesis")
        
        self.stats_tab = StatsTab(self.memory_system)
        self.tab_widget.addTab(self.stats_tab, "Statistics")
        
        # Status bar setup
        self.statusBar().showMessage("Ready")
        
        # Add version info to status bar
        version_label = QLabel("Language Memory System v1.0")
        self.statusBar().addPermanentWidget(version_label)


def main():
    """Main function for running the Language Memory GUI"""
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Apply style
    app.setStyle("Fusion")
    
    # Create main window
    window = LanguageMemoryMainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 