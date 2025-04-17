#!/usr/bin/env python3
"""
LUMINA v7.5 Main GUI
Integrates the node system with a graphical interface
"""

import sys
import logging
import asyncio
import nltk
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QLabel, QPushButton, QSplitter,
                              QScrollArea, QFrame, QMessageBox)
from PySide6.QtCore import Qt, QTimer, QObject, Signal
import qasync
from src.v7_5.nodes.node_manager import NodeManager
from src.v7_5.nodes.wiki_processor_node import WikiProcessorNode
from src.v7_5.chat_widget import ChatWidget

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]'
)
logger = logging.getLogger(__name__)

def initialize_nltk():
    """Initialize NLTK resources"""
    try:
        logger.debug("Initializing NLTK resources")
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.debug("NLTK resources initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize NLTK resources: {e}", exc_info=True)
        return False

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        logger.debug("Checking dependencies")
        # Check NLTK
        if not initialize_nltk():
            raise Exception("Failed to initialize NLTK resources")
        
        # Check Wikipedia
        import wikipedia
        logger.debug("Wikipedia package available")
        
        # Check PySide6
        from PySide6 import QtCore
        logger.debug("PySide6 available")
        
        logger.debug("All dependencies available")
        return True
    except Exception as e:
        logger.error(f"Dependency check failed: {e}", exc_info=True)
        return False

def show_error_dialog(message):
    """Show an error dialog"""
    app = QApplication.instance()
    if app:
        QMessageBox.critical(None, "Error", message)
    else:
        logger.error(f"Error dialog requested but no QApplication instance: {message}")

def debug_qt_state():
    """Debug function to check Qt state"""
    app = QApplication.instance()
    if app:
        logger.debug(f"Qt Application exists: {app}")
        logger.debug(f"Qt Application state: {app.applicationState()}")
    else:
        logger.debug("No Qt Application instance exists")

class AsyncHelper(QObject):
    """Helper class to run async tasks in Qt's event loop"""
    
    def __init__(self):
        logger.debug("Initializing AsyncHelper")
        super().__init__()
        self._loop = None
        
    def run_async(self, coro):
        logger.debug("Running async task")
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return asyncio.create_task(coro)

class NodeGraphWidget(QWidget):
    """Widget for visualizing and managing nodes"""
    
    def __init__(self, manager: NodeManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Add node controls
        controls = QHBoxLayout()
        add_button = QPushButton("Add Node")
        add_button.clicked.connect(self.add_node)
        controls.addWidget(add_button)
        layout.addLayout(controls)
        
        # Add node display area
        self.node_area = QScrollArea()
        self.node_area.setWidgetResizable(True)
        self.node_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.node_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        
        self.node_container = QWidget()
        self.node_layout = QVBoxLayout(self.node_container)
        self.node_area.setWidget(self.node_container)
        
        layout.addWidget(self.node_area)
        
    def add_node(self):
        """Add a new node to the graph"""
        node = self.manager.create_node("wiki_processor")
        if node:
            self.add_node_widget(node)
            
    def add_node_widget(self, node):
        """Add a widget representation of a node"""
        widget = NodeWidget(node)
        self.node_layout.addWidget(widget)

class NodeWidget(QFrame):
    """Widget representing a single node"""
    
    def __init__(self, node, parent=None):
        super().__init__(parent)
        self.node = node
        self.setup_ui()
        
        # Connect to node signals
        self.node.node_updated.connect(self.update_display)
        self.node.error_occurred.connect(self.show_error)
        
    def setup_ui(self):
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        layout = QVBoxLayout(self)
        
        # Node header
        header = QHBoxLayout()
        title = QLabel(self.node.metadata.name)
        title.setStyleSheet(f"color: {self.node.metadata.color};")
        header.addWidget(title)
        layout.addLayout(header)
        
        # Input ports
        if self.node.input_ports:
            inputs = QVBoxLayout()
            inputs.addWidget(QLabel("Inputs:"))
            for name, port in self.node.input_ports.items():
                port_layout = QHBoxLayout()
                port_layout.addWidget(QLabel(f"• {name}: {port.description}"))
                inputs.addLayout(port_layout)
            layout.addLayout(inputs)
            
        # Output ports
        if self.node.output_ports:
            outputs = QVBoxLayout()
            outputs.addWidget(QLabel("Outputs:"))
            for name, port in self.node.output_ports.items():
                port_layout = QHBoxLayout()
                port_layout.addWidget(QLabel(f"• {name}: {port.description}"))
                outputs.addLayout(port_layout)
            layout.addLayout(outputs)
            
        # Status display
        self.status_label = QLabel("Status: Ready")
        layout.addWidget(self.status_label)
        
    def update_display(self):
        """Update the node's display"""
        # Update status if available
        if "status" in self.node.output_ports:
            status = self.node.output_ports["status"].value
            if status:
                self.status_label.setText(f"Status: {status}")
                
    def show_error(self, error_msg):
        """Display an error message"""
        self.status_label.setText(f"Error: {error_msg}")
        self.status_label.setStyleSheet("color: red;")

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        logger.debug("Initializing MainWindow")
        try:
            super().__init__()
            self.setWindowTitle("LUMINA v7.5")
            self.resize(1200, 800)
            logger.debug("MainWindow basic properties set")
            
            # Initialize async helper
            logger.debug("Creating AsyncHelper")
            self.async_helper = AsyncHelper()
            
            # Initialize node system
            logger.debug("Initializing NodeManager")
            try:
                self.node_manager = NodeManager()
                logger.debug("NodeManager initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize NodeManager: {e}", exc_info=True)
                raise
            
            logger.debug("Registering node types")
            try:
                self.node_manager.register_node_type("wiki_processor", WikiProcessorNode)
                logger.debug("Node types registered successfully")
            except Exception as e:
                logger.error(f"Failed to register node types: {e}", exc_info=True)
                raise
            
            # Create central widget and layout
            logger.debug("Creating central widget")
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            layout = QHBoxLayout(central_widget)
            
            # Create main splitter
            logger.debug("Creating main splitter")
            splitter = QSplitter(Qt.Orientation.Horizontal)
            layout.addWidget(splitter)
            
            # Add node graph
            logger.debug("Creating node graph widget")
            try:
                self.node_graph = NodeGraphWidget(self.node_manager)
                splitter.addWidget(self.node_graph)
                logger.debug("Node graph widget created successfully")
            except Exception as e:
                logger.error(f"Failed to create node graph widget: {e}", exc_info=True)
                raise
            
            # Add chat widget
            logger.debug("Creating chat widget")
            try:
                self.chat_widget = ChatWidget(self)
                splitter.addWidget(self.chat_widget)
                logger.debug("Chat widget created successfully")
            except Exception as e:
                logger.error(f"Failed to create chat widget: {e}", exc_info=True)
                raise
            
            # Set up update timer
            logger.debug("Setting up update timer")
            self.update_timer = QTimer()
            self.update_timer.timeout.connect(self.process_nodes)
            self.update_timer.start(1000)  # Update every second
            
            logger.debug("MainWindow initialization complete")
            
        except Exception as e:
            logger.error(f"Critical error in MainWindow initialization: {e}", exc_info=True)
            raise
        
    def show(self):
        logger.debug("MainWindow.show() called")
        try:
            super().show()
            logger.debug("MainWindow shown")
            logger.debug(f"MainWindow visible: {self.isVisible()}")
            logger.debug(f"MainWindow geometry: {self.geometry()}")
            
            # Force window to be visible and on top
            self.raise_()
            self.activateWindow()
            self.setWindowState(Qt.WindowState.WindowActive)
            
            # Ensure the window is properly sized and positioned
            screen = QApplication.primaryScreen().geometry()
            self.setGeometry(
                (screen.width() - self.width()) // 2,
                (screen.height() - self.height()) // 2,
                self.width(),
                self.height()
            )
            
            logger.debug("MainWindow positioning complete")
            
        except Exception as e:
            logger.error(f"Error in MainWindow.show(): {e}", exc_info=True)
            raise

    def process_nodes(self):
        """Process all nodes in the system"""
        if self.node_manager._is_running:
            return
            
        async def run_nodes():
            try:
                await self.node_manager.execute()
            except Exception as e:
                logger.error(f"Error in node processing: {e}", exc_info=True)
                
        self.async_helper.run_async(run_nodes())
        
    def closeEvent(self, event):
        """Handle application shutdown"""
        logger.debug("MainWindow close event")
        self.update_timer.stop()
        self.node_manager.stop()
        self.node_manager.clear()
        super().closeEvent(event)

async def async_main(app):
    """Async main application entry point"""
    try:
        logger.debug("Starting async_main")
        debug_qt_state()
        
        # Create main window
        logger.debug("Creating main window")
        window = MainWindow()
        
        # Show window
        logger.debug("Showing main window")
        window.show()
        
        # Force window to be visible
        logger.debug("Ensuring window is visible")
        window.raise_()
        window.activateWindow()
        
        logger.debug("Main window display complete")
        
    except Exception as e:
        logger.error(f"Error in async_main: {e}", exc_info=True)
        raise

def main():
    """Main application entry point"""
    try:
        logger.debug("Starting main function")
        
        # Create Qt application first
        logger.debug("Checking for existing QApplication")
        app = QApplication.instance()
        if app is None:
            logger.debug("Creating new QApplication")
            app = QApplication(sys.argv)
        else:
            logger.debug("Using existing QApplication")
        
        debug_qt_state()
        
        # Create and run async event loop
        logger.debug("Creating QEventLoop")
        loop = qasync.QEventLoop(app)
        logger.debug("Setting event loop")
        asyncio.set_event_loop(loop)
        
        # Create main window
        logger.debug("Creating main window")
        window = MainWindow()
        window.show()
        
        # Run the event loop
        logger.debug("Starting event loop")
        with loop:
            loop.run_forever()
            
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        show_error_dialog(f"Application error: {str(e)}")
        return 1
    finally:
        logger.debug("Cleaning up")
        if 'loop' in locals() and loop.is_running():
            logger.debug("Closing event loop")
            loop.close()
        return 0

if __name__ == "__main__":
    logger.debug("Script started")
    sys.exit(main())
    logger.debug("Script ended") 