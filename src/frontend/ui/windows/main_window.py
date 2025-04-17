from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from PySide6.QtCore import Qt
from typing import Optional
import logging

from ..components.panels.profile_panel import ProfilePanel
from ..components.panels.network_panel import NetworkPanel
from ..components.panels.training_panel import TrainingPanel
from ..components.panels.visualization_panel import VisualizationPanel

class MainWindow(QMainWindow):
    """Main window for the Lumina frontend."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.setWindowTitle("Lumina Neural Network System")
        self.setMinimumSize(1280, 720)
        
        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # Initialize panels
        self.profile_panel: Optional[ProfilePanel] = None
        self.network_panel: Optional[NetworkPanel] = None
        self.training_panel: Optional[TrainingPanel] = None
        self.visualization_panel: Optional[VisualizationPanel] = None
        
        self.initialize_ui()
        
    def initialize_ui(self):
        """Initialize the user interface."""
        try:
            # Create left sidebar
            left_sidebar = QWidget()
            left_layout = QVBoxLayout(left_sidebar)
            left_layout.setAlignment(Qt.AlignTop)
            
            # Create main content area
            main_content = QWidget()
            main_layout = QVBoxLayout(main_content)
            
            # Add panels to left sidebar
            self.profile_panel = ProfilePanel()
            self.network_panel = NetworkPanel()
            self.training_panel = TrainingPanel()
            
            left_layout.addWidget(self.profile_panel)
            left_layout.addWidget(self.network_panel)
            left_layout.addWidget(self.training_panel)
            
            # Add visualization panel to main content
            self.visualization_panel = VisualizationPanel()
            main_layout.addWidget(self.visualization_panel)
            
            # Add widgets to main layout
            self.main_layout.addWidget(left_sidebar, 1)
            self.main_layout.addWidget(main_content, 3)
            
            # Set styles
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #1e1e1e;
                }
                QWidget {
                    color: #ffffff;
                }
            """)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize UI: {str(e)}")
            raise
            
    def closeEvent(self, event):
        """Handle window close event."""
        try:
            self.logger.info("Closing main window...")
            # Cleanup logic here
            event.accept()
        except Exception as e:
            self.logger.error(f"Error during window close: {str(e)}")
            event.accept() 