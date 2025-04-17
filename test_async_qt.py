#!/usr/bin/env python3
"""
Minimal async Qt test script
"""

import sys
import logging
import asyncio
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel
from PySide6.QtCore import QTimer
import qasync

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AsyncTestWindow(QMainWindow):
    def __init__(self):
        logger.debug("Initializing AsyncTestWindow")
        super().__init__()
        self.setWindowTitle("Async Qt Test Window")
        self.resize(400, 300)
        
        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Add some test widgets
        self.label = QLabel("Test Window")
        layout.addWidget(self.label)
        
        button = QPushButton("Test Async")
        button.clicked.connect(self.run_async_task)
        layout.addWidget(button)
        
        # Setup timer for periodic updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_label)
        self.timer.start(1000)  # Update every second
        
        self.counter = 0
        logger.debug("AsyncTestWindow setup complete")
        
    def update_label(self):
        self.counter += 1
        self.label.setText(f"Counter: {self.counter}")
        logger.debug(f"Updated counter to {self.counter}")
    
    def run_async_task(self):
        async def example_task():
            logger.debug("Starting async task")
            await asyncio.sleep(2)
            logger.debug("Async task complete")
            self.label.setText("Async task finished!")
        
        asyncio.create_task(example_task())

def main():
    try:
        logger.debug("Starting async test application")
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)
        
        window = AsyncTestWindow()
        window.show()
        
        logger.debug("Running async application")
        with loop:
            loop.run_forever()
        
    except Exception as e:
        logger.error(f"Error in async test application: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 