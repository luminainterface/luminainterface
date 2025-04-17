"""
Main window for the Lumina Frontend System.
Provides the primary user interface and component integration.
"""

from typing import Optional
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from PySide6.QtCore import Qt
from ..components.panels.profile_panel import ProfilePanel
from ..components.panels.network_panel import NetworkPanel
from ..components.panels.training_panel import TrainingPanel
from ..components.panels.visualization_panel import VisualizationPanel
from ...core.main_controller import MainController

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self, controller: MainController):
        super().__init__()
        self._controller = controller
        self._panels = {}
        
        self._initialize_ui()
        self._setup_layout()
        self._connect_signals()
    
    def _initialize_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("Lumina Neural Network")
        self.setMinimumSize(1280, 720)
        
        # Create central widget
        self._central_widget = QWidget()
        self.setCentralWidget(self._central_widget)
        
        # Create main layout
        self._main_layout = QVBoxLayout(self._central_widget)
        self._main_layout.setContentsMargins(10, 10, 10, 10)
        self._main_layout.setSpacing(10)
        
        # Create panel container
        self._panel_container = QWidget()
        self._panel_layout = QHBoxLayout(self._panel_container)
        self._panel_layout.setContentsMargins(0, 0, 0, 0)
        self._panel_layout.setSpacing(10)
        
        # Add panel container to main layout
        self._main_layout.addWidget(self._panel_container)
    
    def _setup_layout(self) -> None:
        """Set up the window layout."""
        # Create panels
        self._panels['profile'] = ProfilePanel(self._controller)
        self._panels['network'] = NetworkPanel(self._controller)
        self._panels['training'] = TrainingPanel(self._controller)
        self._panels['visualization'] = VisualizationPanel(self._controller)
        
        # Add panels to layout
        for panel in self._panels.values():
            self._panel_layout.addWidget(panel)
    
    def _connect_signals(self) -> None:
        """Connect signals and slots."""
        # Connect controller signals
        self._controller.system_initialized.connect(self._on_system_initialized)
        self._controller.version_changed.connect(self._on_version_changed)
    
    def _on_system_initialized(self) -> None:
        """Handle system initialization."""
        # Update UI state
        for panel in self._panels.values():
            panel.on_system_initialized()
    
    def _on_version_changed(self, version: str) -> None:
        """Handle version change."""
        # Update window title
        self.setWindowTitle(f"Lumina Neural Network - {version}")
        
        # Update panels
        for panel in self._panels.values():
            panel.on_version_changed(version)
    
    def get_panel(self, name: str) -> Optional[QWidget]:
        """Get a panel by name."""
        return self._panels.get(name)
    
    def closeEvent(self, event) -> None:
        """Handle window close event."""
        # Shutdown system
        self._controller.shutdown()
        event.accept() 
Main window for the Lumina Frontend System.
Provides the primary user interface and component integration.
"""

from typing import Optional
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from PySide6.QtCore import Qt
from ..components.panels.profile_panel import ProfilePanel
from ..components.panels.network_panel import NetworkPanel
from ..components.panels.training_panel import TrainingPanel
from ..components.panels.visualization_panel import VisualizationPanel
from ...core.main_controller import MainController

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self, controller: MainController):
        super().__init__()
        self._controller = controller
        self._panels = {}
        
        self._initialize_ui()
        self._setup_layout()
        self._connect_signals()
    
    def _initialize_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("Lumina Neural Network")
        self.setMinimumSize(1280, 720)
        
        # Create central widget
        self._central_widget = QWidget()
        self.setCentralWidget(self._central_widget)
        
        # Create main layout
        self._main_layout = QVBoxLayout(self._central_widget)
        self._main_layout.setContentsMargins(10, 10, 10, 10)
        self._main_layout.setSpacing(10)
        
        # Create panel container
        self._panel_container = QWidget()
        self._panel_layout = QHBoxLayout(self._panel_container)
        self._panel_layout.setContentsMargins(0, 0, 0, 0)
        self._panel_layout.setSpacing(10)
        
        # Add panel container to main layout
        self._main_layout.addWidget(self._panel_container)
    
    def _setup_layout(self) -> None:
        """Set up the window layout."""
        # Create panels
        self._panels['profile'] = ProfilePanel(self._controller)
        self._panels['network'] = NetworkPanel(self._controller)
        self._panels['training'] = TrainingPanel(self._controller)
        self._panels['visualization'] = VisualizationPanel(self._controller)
        
        # Add panels to layout
        for panel in self._panels.values():
            self._panel_layout.addWidget(panel)
    
    def _connect_signals(self) -> None:
        """Connect signals and slots."""
        # Connect controller signals
        self._controller.system_initialized.connect(self._on_system_initialized)
        self._controller.version_changed.connect(self._on_version_changed)
    
    def _on_system_initialized(self) -> None:
        """Handle system initialization."""
        # Update UI state
        for panel in self._panels.values():
            panel.on_system_initialized()
    
    def _on_version_changed(self, version: str) -> None:
        """Handle version change."""
        # Update window title
        self.setWindowTitle(f"Lumina Neural Network - {version}")
        
        # Update panels
        for panel in self._panels.values():
            panel.on_version_changed(version)
    
    def get_panel(self, name: str) -> Optional[QWidget]:
        """Get a panel by name."""
        return self._panels.get(name)
    
    def closeEvent(self, event) -> None:
        """Handle window close event."""
        # Shutdown system
        self._controller.shutdown()
        event.accept() 
 