#!/usr/bin/env python
"""
V5 Visualization System Launcher
Initializes and launches the V5 Fractal Echo Visualization system
"""

import os
import sys
import time
import logging
import threading
import multiprocessing
from pathlib import Path

# Try to import Qt
try:
    # Try PySide6 first
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
        QWidget, QLabel, QPushButton, QTextEdit, QComboBox,
        QTabWidget, QSplitter, QFrame, QStatusBar
    )
    from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread
    from PySide6.QtGui import QFont, QColor, QPalette
    USE_PYSIDE6 = True
except ImportError:
    try:
        # Fallback to PyQt5
        from PyQt5.QtWidgets import (
            QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
            QWidget, QLabel, QPushButton, QTextEdit, QComboBox,
            QTabWidget, QSplitter, QFrame, QStatusBar
        )
        from PyQt5.QtCore import Qt, QTimer, pyqtSignal as Signal, pyqtSlot as Slot, QThread
        from PyQt5.QtGui import QFont, QColor, QPalette
        USE_PYSIDE6 = False
    except ImportError:
        USE_PYSIDE6 = None

# Setup logging
logger = logging.getLogger("V5-Launcher")

class V5MainWindow(QMainWindow):
    """
    Main window for the V5 Fractal Echo Visualization system
    """
    
    def __init__(self, config=None):
        """
        Initialize the V5 main window
        
        Args:
            config (dict): Configuration dictionary
        """
        super().__init__()
        
        self.config = config or {}
        self.mock_mode = self.config.get("mock_mode", False)
        
        # Initialize UI
        self._init_ui()
        
        # Status update timer
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(2000)  # Update every 2 seconds
        
        # Add initial log message
        self._add_log_message("V5 Visualization System started")
        if self.mock_mode:
            self._add_log_message("Running in mock mode")
        
        logger.info("V5 Main Window initialized")
    
    def _init_ui(self):
        """Initialize the user interface"""
        # Set window properties
        self.setWindowTitle("V5 Fractal Echo Visualization System")
        self.setMinimumSize(800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create splitter for main areas
        splitter = QSplitter(Qt.Vertical)
        
        # Create tabs for different visualizations
        tabs = QTabWidget()
        
        # Fractal Pattern tab
        fractal_tab = QWidget()
        fractal_layout = QVBoxLayout(fractal_tab)
        fractal_view = QFrame()
        fractal_view.setFrameShape(QFrame.Box)
        fractal_view.setMinimumHeight(300)
        fractal_layout.addWidget(QLabel("Fractal Pattern Visualization"))
        fractal_layout.addWidget(fractal_view)
        tabs.addTab(fractal_tab, "Fractal Patterns")
        
        # Network Visualization tab
        network_tab = QWidget()
        network_layout = QVBoxLayout(network_tab)
        network_view = QFrame()
        network_view.setFrameShape(QFrame.Box)
        network_view.setMinimumHeight(300)
        network_layout.addWidget(QLabel("Neural Network Visualization"))
        network_layout.addWidget(network_view)
        tabs.addTab(network_tab, "Network Visualization")
        
        # Control Panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # Visualization controls
        control_box = QFrame()
        control_box.setFrameShape(QFrame.Box)
        control_box_layout = QHBoxLayout(control_box)
        
        # Memory topic selector
        topic_layout = QVBoxLayout()
        topic_layout.addWidget(QLabel("Memory Topic:"))
        self.topic_combo = QComboBox()
        for topic in ["Neural Networks", "Language Processing", "Consciousness", "Fractal Patterns"]:
            self.topic_combo.addItem(topic)
        topic_layout.addWidget(self.topic_combo)
        
        # Pattern generation
        pattern_layout = QVBoxLayout()
        pattern_layout.addWidget(QLabel("Pattern Type:"))
        self.pattern_combo = QComboBox()
        for pattern in ["Recursive", "Echo", "Resonance", "Wave"]:
            self.pattern_combo.addItem(pattern)
        pattern_layout.addWidget(self.pattern_combo)
        
        # Generate button
        btn_layout = QVBoxLayout()
        btn_layout.addStretch()
        self.generate_btn = QPushButton("Generate Visualization")
        self.generate_btn.clicked.connect(self._generate_visualization)
        btn_layout.addWidget(self.generate_btn)
        
        control_box_layout.addLayout(topic_layout)
        control_box_layout.addLayout(pattern_layout)
        control_box_layout.addLayout(btn_layout)
        
        # Log area
        log_box = QFrame()
        log_box.setFrameShape(QFrame.Box)
        log_box_layout = QVBoxLayout(log_box)
        log_box_layout.addWidget(QLabel("System Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_box_layout.addWidget(self.log_text)
        
        # Add to control panel
        control_layout.addWidget(control_box)
        control_layout.addWidget(log_box)
        
        # Add widgets to splitter
        splitter.addWidget(tabs)
        splitter.addWidget(control_panel)
        
        # Set splitter sizes
        splitter.setSizes([400, 200])
        
        # Add to main layout
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("V5 Visualization System Ready")
        
        # Set central widget
        self.setCentralWidget(central_widget)
    
    def _generate_visualization(self):
        """Generate visualization based on selected parameters"""
        topic = self.topic_combo.currentText()
        pattern = self.pattern_combo.currentText()
        
        self._add_log_message(f"Generating {pattern} visualization for topic: {topic}")
        self.status_bar.showMessage(f"Generating visualization: {pattern} - {topic}")
        
        # In a real implementation, this would connect to the bridge
        # and request a visualization from the backend
        
        # For mock mode, just simulate a delay
        QTimer.singleShot(500, lambda: self._visualization_complete(topic, pattern))
    
    def _visualization_complete(self, topic, pattern):
        """Simulate visualization completion"""
        self._add_log_message(f"Visualization generated: {pattern} pattern for {topic}")
        self.status_bar.showMessage(f"Visualization complete: {pattern} - {topic}")
    
    def _add_log_message(self, message):
        """Add a message to the log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def _update_status(self):
        """Update status information"""
        # In a real implementation, this would query the bridge
        # for status information
        
        # For now, just update the timestamp in the status bar
        current_status = self.status_bar.currentMessage().split(" | ")[0]
        timestamp = time.strftime("%H:%M:%S")
        self.status_bar.showMessage(f"{current_status} | Last update: {timestamp}")

def _run_qt_app_process(mock_mode=False, debug=False):
    """
    Run the Qt application in a separate process.
    This function is called in the new process.
    
    Args:
        mock_mode (bool): Whether to use mock/simulated data
        debug (bool): Whether to enable debug logging
    """
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout
    )
    
    logger.info("Starting V5 GUI application process")
    
    # Create QApplication instance
    app = QApplication([])
    
    # Create configuration
    config = {
        "mock_mode": mock_mode,
        "debug": debug
    }
    
    # Create and show main window
    window = V5MainWindow(config)
    window.show()
    
    # Run the application event loop (this must be done in the main thread)
    logger.info(f"Running V5 GUI application using {'PySide6' if USE_PYSIDE6 else 'PyQt5'}")
    sys.exit(app.exec())

def launch_v5_system(mock_mode=False, debug=False):
    """
    Launch the V5 Visualization System
    
    Args:
        mock_mode (bool): Whether to use mock/simulated data
        debug (bool): Whether to enable debug logging
    
    Returns:
        bool: True if the system was launched successfully
    """
    logger.info("Launching V5 Visualization System")
    
    # Check if Qt is available
    if USE_PYSIDE6 is None:
        logger.error("Neither PySide6 nor PyQt5 are installed. Cannot launch V5 GUI.")
        return False
    
    try:
        # METHOD 1: For use in standalone mode - start a separate process
        if __name__ == "__main__":
            # Start a new process for the Qt application
            process = multiprocessing.Process(
                target=_run_qt_app_process,
                args=(mock_mode, debug),
                daemon=True
            )
            process.start()
            logger.info(f"Started V5 GUI in separate process (PID: {process.pid})")
            return True
        
        # METHOD 2: For use when imported from run_system.py
        else:
            # We create a subprocess to handle the GUI
            import subprocess
            
            # Construct the command to run this script directly
            script_path = os.path.abspath(__file__)
            cmd = [sys.executable, script_path]
            
            if mock_mode:
                cmd.append("--mock")
            if debug:
                cmd.append("--debug")
                
            # Use DETACHED_PROCESS flag to run process independently on Windows
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creationflags
            )
            
            logger.info(f"Started V5 GUI in subprocess (PID: {process.pid})")
            
            # Check if process started successfully
            return process.poll() is None
            
    except Exception as e:
        logger.error(f"Error launching V5 Visualization System: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

# Test code
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout
    )
    
    # Parse arguments
    mock_mode = "--mock" in sys.argv
    debug_mode = "--debug" in sys.argv
    
    # Launch V5 system
    launch_v5_system(mock_mode=mock_mode, debug=debug_mode) 