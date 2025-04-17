#!/usr/bin/env python3
"""
LUMINA v7.5 Minimal GUI
A minimal GUI implementation for testing the LUMINA system
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QProgressBar, QLabel, QPushButton, QStatusBar
)
from PySide6.QtCore import Qt, QTimer
import qasync

from .lumina_core import LUMINACore
from .chat_widget import ChatWidget
from .signal_system import SignalBus
from .signal_component import SignalComponent

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """Main window for the LUMINA v7.5 GUI"""
    
    def __init__(self, core: LUMINACore):
        logger.debug("Initializing MainWindow")
        super().__init__()
        self.setWindowTitle("LUMINA v7.5")
        self.setGeometry(100, 100, 800, 600)
        
        # Store core reference
        self.core = core
        
        # Initialize signal system
        self.signal_bus = core.signal_bus
        self.signal_component = SignalComponent("minimal_gui", self.signal_bus)
        
        # Initialize UI components
        self.chat_widget: Optional[ChatWidget] = None
        self.progress_bar: Optional[QProgressBar] = None
        self.status_label: Optional[QLabel] = None
        
        # Initialize component status
        self.component_status = {
            "central_node": False,
            "mistral": False,
            "neural": False,
            "memory": False,
            "autowiki": False
        }
        
        # Message handling state
        self.waiting_for_response = False
        self.response_received = False
        
        # Set up UI and signals
        self._setup_ui()
        self._connect_signals()
        
        # Update initial status
        self._update_status_display()
        
        # Add intro message
        if self.chat_widget:
            self.chat_widget.append_message(
                "System",
                "Welcome to LUMINA v7.5! I'm here to assist you. Feel free to ask any questions."
            )
        
        logger.debug("MainWindow initialized")
        
    def _setup_ui(self):
        """Setup the user interface"""
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Add status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Add chat widget
        self.chat_widget = ChatWidget(self.signal_bus)
        layout.addWidget(self.chat_widget)
        
        # Setup status bar
        self.statusBar().showMessage("Starting up...")
        
        # Enable chat widget by default
        self.chat_widget.setEnabled(True)
        self.chat_widget.message_input.setEnabled(True)
        self.chat_widget.send_button.setEnabled(True)
        
    def _connect_signals(self):
        """Connect signals and slots"""
        try:
            # Connect core signals
            self.core.system_ready.connect(self.on_system_ready)
            self.core.system_error.connect(self.on_system_error)
            self.core.component_initialized.connect(self.on_component_initialized)
            
            # Connect chat widget signals
            if self.chat_widget:
                self.chat_widget.message_sent.connect(self.on_message_sent_sync)
                
            # Subscribe to signal bus messages
            self.signal_component.register_handler("status_response", self.on_status_response)
            self.signal_component.register_handler("component_response", self.on_component_response)
            self.signal_component.register_handler("message_response", self.on_message_response)
            self.signal_component.register_handler("system_message", self.handle_system_message)
            
        except Exception as e:
            logger.error(f"Error connecting signals: {e}")
            self.statusBar().showMessage(f"Error: {str(e)}")
        
    def on_system_ready(self):
        """Handle system ready signal"""
        logger.info("System ready")
        self.progress_bar.setValue(100)
        self.status_label.setText("System Ready")
        self.statusBar().showMessage("System is ready")
        self.chat_widget.setEnabled(True)
        
        # Request component status
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self.signal_component.emit_signal("status_request", {}))
        except Exception as e:
            logger.error(f"Error requesting status: {e}")
        
    def on_system_error(self, error_msg: str):
        """Handle system error signal"""
        logger.error(f"System error: {error_msg}")
        self.status_label.setText(f"Error: {error_msg}")
        self.statusBar().showMessage(f"Error: {error_msg}")
        
    def on_component_initialized(self, component_name: str):
        """Handle component initialization"""
        logger.info(f"Component initialized: {component_name}")
        self.component_status[component_name] = True
        self._update_progress()
        
    def on_status_response(self, data: Dict[str, Any]):
        """Handle status response from core"""
        if data.get("type") == "status_response":
            self.component_status.update(data.get("status", {}))
            self._update_status_display()
            
    def on_component_response(self, data: Dict[str, Any]):
        """Handle component availability response"""
        if data.get("type") == "component_response":
            component = data.get("component")
            available = data.get("available", False)
            if component:
                self.component_status[component] = available
                self._update_status_display()
                
    def _update_progress(self):
        """Update the progress bar based on initialized components"""
        try:
            total_components = len(self.component_status)
            if total_components > 0:
                initialized = sum(1 for status in self.component_status.values() if status)
                progress = int((initialized / total_components) * 100)
                self.progress_bar.setValue(progress)
        except Exception as e:
            logger.error(f"Error updating progress: {e}")
        
    def _update_status_display(self):
        """Update the status display with component information"""
        try:
            status_text = "Available Components:\n"
            for component, available in self.component_status.items():
                status = "✓" if available else "✗"
                status_text += f"{component}: {status}\n"
            self.status_label.setText(status_text)
        except Exception as e:
            logger.error(f"Error updating status display: {e}")
        
    def on_message_sent_sync(self, message: str):
        """Synchronous handler for message sent signal that launches async processing"""
        if not message.strip() or self.waiting_for_response:
            return
            
        # Create event loop if it doesn't exist
        loop = asyncio.get_event_loop()
        
        # Schedule the async processing
        asyncio.create_task(self._process_message(message))
        
        # Display user message immediately
        self.chat_widget.append_message("User", message)
        
        # Clear input field
        self.chat_widget.message_input.clear()
        
    async def _process_message(self, message: str):
        """Process a message asynchronously"""
        # Show processing state
        self.chat_widget.is_processing = True
        self.chat_widget.update_status_display()
        self.statusBar().showMessage("Processing message...")
        
        try:
            # Send message to core for processing
            await self.signal_component.emit_signal("process_message", {
                "type": "process_message",
                "content": message,
                "source": "minimal_gui"
            })
            
            # Wait for response (timeout after 10 seconds)
            start_time = asyncio.get_event_loop().time()
            while not self.response_received:
                if asyncio.get_event_loop().time() - start_time > 10:
                    raise TimeoutError("Message processing timed out")
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.chat_widget.append_system_message(f"Error: {str(e)}")
            
        finally:
            # Reset state
            self.waiting_for_response = False
            self.response_received = False
            self.chat_widget.is_processing = False
            self.chat_widget.update_status_display()
            self.statusBar().showMessage("Ready")
            
    def on_message_response(self, data: Dict[str, Any]):
        """Handle message response from core"""
        try:
            if data.get("type") == "message_response":
                if "error" in data:
                    self.chat_widget.append_system_message(f"Error: {data['error']}")
                else:
                    self.chat_widget.append_message("Assistant", data.get("content", ""))
                self.response_received = True
                self.waiting_for_response = False
        except Exception as e:
            logger.error(f"Error handling message response: {e}")
            self.chat_widget.append_system_message(f"Error handling response: {str(e)}")
        finally:
            self.response_received = True
            self.waiting_for_response = False
            
    def handle_system_message(self, data: Dict[str, Any]):
        """Handle system messages"""
        try:
            if isinstance(data, dict) and "message" in data:
                self.chat_widget.append_system_message(data["message"])
        except Exception as e:
            logger.error(f"Error handling system message: {e}")
            
    def closeEvent(self, event):
        """Handle window close event"""
        logger.info("Closing application...")
        try:
            self.signal_component.unregister_all_handlers()
            self.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        event.accept()
        
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.chat_widget:
                self.chat_widget.cleanup()
            self.signal_component.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def run_async_app():
    """Run the async application"""
    try:
        # Create Qt application
        app = QApplication(sys.argv)
        
        # Set up event loop first
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)
        
        # Create core and main window
        core = LUMINACore()
        window = MainWindow(core)
        window.show()
        
        # Initialize core and signal components
        async def init_components():
            try:
                # Initialize signal bus first
                await core.signal_bus.initialize()
                
                # Initialize core components
                await core.initialize()
                
                # Initialize GUI components
                await window.signal_component.initialize()
                
                # Enable chat widget after initialization
                window.chat_widget.setEnabled(True)
                window.chat_widget.message_input.setEnabled(True)
                window.chat_widget.send_button.setEnabled(True)
                
            except Exception as e:
                logger.error(f"Error during initialization: {e}")
                window.statusBar().showMessage(f"Error: {str(e)}")
                window.chat_widget.append_system_message(f"Initialization error: {str(e)}")
        
        # Schedule initialization
        loop.create_task(init_components())
        
        # Run event loop
        loop.run_forever()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Set environment variables if not set
    if not os.environ.get("MISTRAL_API_KEY"):
        os.environ["MISTRAL_API_KEY"] = "nLKZEpq29OihnaArxV7s6KtzsNEiky2A"
    if not os.environ.get("MODEL_NAME"):
        os.environ["MODEL_NAME"] = "mistral-medium"
    if not os.environ.get("MOCK_MODE"):
        os.environ["MOCK_MODE"] = "false"
        
    # Run the application
    run_async_app() 