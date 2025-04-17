#!/usr/bin/env python3
"""
Simplified main entry point for the V5 Fractal Echo Visualization System.
This version is a minimal implementation to test the integration with the bridge system.
"""

import sys
import logging
import argparse
import time
import os
from pathlib import Path

# Add the parent directory to the Python path
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import bridge system components
try:
    from version_bridge_manager import VersionBridgeManager
    HAS_BRIDGE_SYSTEM = True
except ImportError:
    HAS_BRIDGE_SYSTEM = False
    print("WARNING: Bridge system not found. Continuing without bridge integration.")

# Try to import Qt - this is optional in the simplified version
try:
    # Try PySide6 first (preferred)
    from PySide6 import QtWidgets, QtCore, QtGui
    from PySide6.QtCore import Qt
    HAS_QT = True
    QT_FRAMEWORK = "PySide6"
except ImportError:
    try:
        # Fallback to PyQt5
        from PyQt5 import QtWidgets, QtCore, QtGui
        from PyQt5.QtCore import Qt
        HAS_QT = True
        QT_FRAMEWORK = "PyQt5"
    except ImportError:
        HAS_QT = False
        QT_FRAMEWORK = "None"
        print("WARNING: Neither PySide6 nor PyQt5 was found. GUI will not be available.")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="V5 Fractal Echo Visualization System (Simplified)")
    parser.add_argument("--mock", action="store_true", help="Run with mock data")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI (for testing)")
    parser.add_argument("--no-bridge", action="store_true", help="Run without bridge system")
    return parser.parse_args()

def setup_logging(debug=False):
    """Configure logging for the application."""
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Create logs directory if it doesn't exist
    logs_dir = Path(__file__).resolve().parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    log_file = logs_dir / "v5_visualization_simple.log"
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    return logging.getLogger("V5-Simple")

class SimpleV5MainWindow(QtWidgets.QMainWindow):
    """A simplified main window for the V5 Fractal Echo Visualization System."""
    
    def __init__(self, mock_mode=False):
        super().__init__()
        
        self.mock_mode = mock_mode
        self.logger = logging.getLogger("V5-Simple-UI")
        
        # Set up the UI
        self.setWindowTitle("V5 Fractal Echo Visualization System (Simplified)")
        self.setMinimumSize(800, 600)
        
        # Create central widget
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create main layout
        self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)
        
        # Add a title label
        title_label = QtWidgets.QLabel("V5 Fractal Echo Visualization System")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        title_label.setFont(font)
        self.main_layout.addWidget(title_label)
        
        # Add a description label
        description = (
            "This is a simplified version of the V5 Visualization System for testing integration.\n\n"
            f"Running with: {QT_FRAMEWORK}\n"
            f"Mock Mode: {'Enabled' if mock_mode else 'Disabled'}\n"
            f"Bridge System: {'Available' if HAS_BRIDGE_SYSTEM else 'Not Available'}"
        )
        description_label = QtWidgets.QLabel(description)
        description_label.setAlignment(QtCore.Qt.AlignCenter)
        self.main_layout.addWidget(description_label)
        
        # Add a spacer
        self.main_layout.addStretch()
        
        # Add a log text area
        log_label = QtWidgets.QLabel("System Log:")
        font = QtGui.QFont()
        font.setBold(True)
        log_label.setFont(font)
        self.main_layout.addWidget(log_label)
        
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.main_layout.addWidget(self.log_text)
        
        # Add a test button
        self.test_button = QtWidgets.QPushButton("Test Memory Bridge")
        self.test_button.clicked.connect(self.test_memory_bridge)
        self.main_layout.addWidget(self.test_button)
        
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("V5 System Ready")
        
        # Log the initialization
        self.log_message("V5 Visualization System initialized")
        if self.mock_mode:
            self.log_message("Running in MOCK mode")
    
    def log_message(self, message):
        """Add a message to the log display."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.logger.info(message)
    
    def test_memory_bridge(self):
        """Test the memory bridge connection."""
        self.log_message("Testing Memory Bridge connection...")
        
        if not HAS_BRIDGE_SYSTEM:
            self.log_message("Bridge system not available")
            return
        
        try:
            # Try to access the bridge manager from the parent's scope
            # This assumes the bridge manager is available as a global variable
            if 'bridge_manager' in globals():
                bridge = globals()['bridge_manager']
                
                language_bridge = bridge.get_component("language_memory_v5_bridge")
                if language_bridge:
                    self.log_message("Found Language Memory V5 Bridge component")
                    try:
                        topics = language_bridge.get_available_topics()
                        if topics:
                            self.log_message(f"Successfully retrieved {len(topics)} topics")
                            self.log_message(f"Topics: {', '.join(topics[:5])}...")
                        else:
                            self.log_message("No topics available")
                    except Exception as e:
                        self.log_message(f"Error retrieving topics: {e}")
                else:
                    self.log_message("Language Memory V5 Bridge component not found")
            else:
                self.log_message("Bridge manager not available in global scope")
                
                # Try creating a local instance
                self.log_message("Creating local bridge manager instance")
                bridge_config = {"mock_mode": self.mock_mode}
                bridge = VersionBridgeManager(bridge_config)
                
                success = bridge.start()
                if success:
                    self.log_message("Successfully started local bridge manager")
                    
                    language_bridge = bridge.get_component("language_memory_v5_bridge")
                    if language_bridge:
                        self.log_message("Found Language Memory V5 Bridge component")
                        try:
                            topics = language_bridge.get_available_topics()
                            if topics:
                                self.log_message(f"Successfully retrieved {len(topics)} topics")
                                self.log_message(f"Topics: {', '.join(topics[:5])}...")
                            else:
                                self.log_message("No topics available")
                        except Exception as e:
                            self.log_message(f"Error retrieving topics: {e}")
                    else:
                        self.log_message("Language Memory V5 Bridge component not found")
                    
                    # Stop the local bridge manager
                    bridge.stop()
                    self.log_message("Stopped local bridge manager")
                else:
                    self.log_message("Failed to start local bridge manager")
        
        except Exception as e:
            self.log_message(f"Error testing bridge: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

def main():
    """Main entry point for the application."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(args.debug)
    logger.info("Starting V5 Fractal Echo Visualization System (Simplified)")
    
    # For headless mode, we need the bridge system
    if args.no_gui and not HAS_BRIDGE_SYSTEM and not args.no_bridge:
        logger.error("Headless mode requires bridge system. Cannot continue.")
        return 1
    
    # Initialize bridge system if available and not explicitly disabled
    bridge_manager = None
    if HAS_BRIDGE_SYSTEM and not args.no_bridge:
        try:
            logger.info("Initializing Version Bridge Manager")
            
            # Initialize Version Bridge Manager with mock mode
            bridge_config = {"mock_mode": args.mock}
            bridge_manager = VersionBridgeManager(bridge_config)
            
            # Make the bridge manager available globally
            globals()['bridge_manager'] = bridge_manager
            
            # Start the bridge manager
            bridge_success = bridge_manager.start()
            if bridge_success:
                logger.info("Version Bridge Manager started successfully")
            else:
                logger.error("Version Bridge Manager failed to start")
                return 1
                
        except Exception as e:
            logger.error(f"Error initializing bridge system: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            if not HAS_QT or args.no_gui:
                # We need the bridge for headless mode
                return 1
    
    # Initialize GUI if available and not explicitly disabled
    if HAS_QT and not args.no_gui:
        try:
            # Initialize Qt application
            app = QtWidgets.QApplication(sys.argv)
            
            # Create and show the main window
            window = SimpleV5MainWindow(mock_mode=args.mock)
            window.show()
            
            # Run the application event loop
            logger.info("Entering Qt application main loop")
            return app.exec_()
            
        except Exception as e:
            logger.error(f"Error initializing GUI: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # Continue with headless mode if the bridge is available
            if not HAS_BRIDGE_SYSTEM or args.no_bridge:
                return 1
    
    # Headless mode - just keep the bridge running until interrupted
    if bridge_manager:
        try:
            logger.info("Running in headless mode. Press Ctrl+C to exit.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt. Shutting down...")
        finally:
            # Clean up resources
            if bridge_manager:
                logger.info("Stopping bridge manager")
                bridge_manager.stop()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 