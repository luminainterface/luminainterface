#!/usr/bin/env python3
"""
Main Application

This is the main entry point for the Lumina Neural Network application,
integrating all components through the SystemIntegrator.
"""

import sys
import asyncio
import logging
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from integration import (
    SystemIntegrator,
    LOGGING_CONFIG
)
from ui.main_window import MainWindow

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class LuminaApp:
    """Main application class."""
    
    def __init__(self):
        """Initialize the application."""
        # Create Qt application
        self.app = QApplication(sys.argv)
        
        # Create system integrator
        self.integrator = SystemIntegrator()
        
        # Create main window
        self.main_window = MainWindow(self.integrator)
        
        # Connect signals
        self._connect_signals()
        
        # Setup async event loop
        self.loop = asyncio.get_event_loop()
        self.timer = QTimer()
        self.timer.timeout.connect(lambda: self._process_events())
        self.timer.start(10)  # Process every 10ms
        
    def _connect_signals(self):
        """Connect application signals."""
        # System ready signal
        self.integrator.system_ready.connect(self._handle_system_ready)
        
        # State change signal
        self.integrator.state_changed.connect(self.main_window.update_state)
        
        # Error signal
        self.integrator.error_occurred.connect(self.main_window.show_error)
        
    def _handle_system_ready(self):
        """Handle system ready signal."""
        logger.info("System initialization complete")
        self.main_window.show()
        
    def _process_events(self):
        """Process async events."""
        self.loop.stop()
        self.loop.run_forever()
        
    async def start(self):
        """Start the application."""
        try:
            # Initialize system
            if not await self.integrator.initialize():
                raise RuntimeError("Failed to initialize system")
                
            # Start Qt event loop
            exit_code = self.app.exec()
            
            # Cleanup
            await self.cleanup()
            
            return exit_code
            
        except Exception as e:
            logger.error(f"Application error: {e}")
            return 1
            
    async def cleanup(self):
        """Clean up resources."""
        try:
            # Disconnect from backend
            await self.integrator.backend.disconnect()
            
            # Clear signal processor
            self.integrator.signal_processor.clear_buffer()
            
            logger.info("Application cleanup complete")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def main():
    """Main entry point."""
    app = LuminaApp()
    exit_code = asyncio.run(app.start())
    sys.exit(exit_code)

if __name__ == "__main__":
    main()