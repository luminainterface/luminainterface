#!/usr/bin/env python3
"""
AutoWiki Integration for LUMINA v7.5
Connects AutoWikiProcessor to the main node manager UI
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from .auto_wiki_processor import AutoWikiProcessor
from .auto_wiki_monitor import AutoWikiMonitor
from .auto_wiki_backend import AutoWikiBackend

class AutoWikiTab(QWidget):
    """AutoWiki integration tab for the node manager"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_components()
        self.setup_ui()
        self.connect_components()
    
    def setup_components(self):
        """Initialize system components"""
        # Create components in the correct order
        self.backend = AutoWikiBackend()
        self.processor = AutoWikiProcessor()
        self.monitor = AutoWikiMonitor(self.processor)
        
        # Register components with backend
        self.backend.register_processor(self.processor)
        self.backend.register_monitor(self.monitor)
    
    def setup_ui(self):
        """Set up the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create tab widget
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: white;
            }
            QTabBar::tab {
                padding: 8px 16px;
                margin-right: 4px;
                border: none;
                background: #F5F5F2;
                color: #1A1A1A;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom: 2px solid #C6A962;
                color: #000000;
            }
            QTabBar::tab:hover:!selected {
                background: #E5E5E2;
            }
        """)
        
        # Add monitor tab
        tabs.addTab(self.monitor, "Monitor")
        
        layout.addWidget(tabs)
    
    def connect_components(self):
        """Connect component signals and slots"""
        # Monitor to processor connections are handled by the monitor itself
        # Backend connections are handled in the backend's register methods
        pass
    
    def start(self):
        """Start the system when the node manager starts"""
        self.processor.start_processing()
    
    def stop(self):
        """Stop the system when the node manager stops"""
        self.processor.stop_processing()
        self.backend.shutdown()
        self.processor.shutdown()
    
    def get_processor(self) -> AutoWikiProcessor:
        """Get the processor instance for external use"""
        return self.processor
    
    def get_backend(self) -> AutoWikiBackend:
        """Get the backend instance for external use"""
        return self.backend 
 
 
"""
AutoWiki Integration for LUMINA v7.5
Connects AutoWikiProcessor to the main node manager UI
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from .auto_wiki_processor import AutoWikiProcessor
from .auto_wiki_monitor import AutoWikiMonitor
from .auto_wiki_backend import AutoWikiBackend

class AutoWikiTab(QWidget):
    """AutoWiki integration tab for the node manager"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_components()
        self.setup_ui()
        self.connect_components()
    
    def setup_components(self):
        """Initialize system components"""
        # Create components in the correct order
        self.backend = AutoWikiBackend()
        self.processor = AutoWikiProcessor()
        self.monitor = AutoWikiMonitor(self.processor)
        
        # Register components with backend
        self.backend.register_processor(self.processor)
        self.backend.register_monitor(self.monitor)
    
    def setup_ui(self):
        """Set up the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create tab widget
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: white;
            }
            QTabBar::tab {
                padding: 8px 16px;
                margin-right: 4px;
                border: none;
                background: #F5F5F2;
                color: #1A1A1A;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom: 2px solid #C6A962;
                color: #000000;
            }
            QTabBar::tab:hover:!selected {
                background: #E5E5E2;
            }
        """)
        
        # Add monitor tab
        tabs.addTab(self.monitor, "Monitor")
        
        layout.addWidget(tabs)
    
    def connect_components(self):
        """Connect component signals and slots"""
        # Monitor to processor connections are handled by the monitor itself
        # Backend connections are handled in the backend's register methods
        pass
    
    def start(self):
        """Start the system when the node manager starts"""
        self.processor.start_processing()
    
    def stop(self):
        """Stop the system when the node manager stops"""
        self.processor.stop_processing()
        self.backend.shutdown()
        self.processor.shutdown()
    
    def get_processor(self) -> AutoWikiProcessor:
        """Get the processor instance for external use"""
        return self.processor
    
    def get_backend(self) -> AutoWikiBackend:
        """Get the backend instance for external use"""
        return self.backend 
 