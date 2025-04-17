"""
Main controller for the Lumina Frontend System.
Handles system initialization, version management, and component coordination.
"""

from typing import Dict, Optional
from PySide6.QtCore import QObject, Signal
from ..ui.windows.main_window import MainWindow
from ..integration.bridges.manager import BridgeManager
from ..neural.playground.manager import NeuralPlaygroundManager
from ..visualization.manager import VisualizationManager

class MainController(QObject):
    """Main controller class for the Lumina Frontend System."""
    
    # Signals
    system_initialized = Signal()
    version_changed = Signal(str)
    component_loaded = Signal(str)
    
    def __init__(self):
        super().__init__()
        self._version = "v7.5"
        self._components: Dict[str, object] = {}
        self._initialized = False
        
        # Initialize managers
        self._bridge_manager = BridgeManager()
        self._neural_manager = NeuralPlaygroundManager()
        self._visualization_manager = VisualizationManager()
        
        # Initialize main window
        self._main_window = MainWindow(self)
    
    def initialize(self) -> None:
        """Initialize the system and all components."""
        if self._initialized:
            return
            
        # Initialize components in order
        self._initialize_bridges()
        self._initialize_neural_system()
        self._initialize_visualization()
        
        self._initialized = True
        self.system_initialized.emit()
    
    def _initialize_bridges(self) -> None:
        """Initialize bridge connections."""
        self._bridge_manager.initialize()
        self._components['bridges'] = self._bridge_manager
    
    def _initialize_neural_system(self) -> None:
        """Initialize neural playground and breathing system."""
        self._neural_manager.initialize()
        self._components['neural'] = self._neural_manager
    
    def _initialize_visualization(self) -> None:
        """Initialize visualization system."""
        self._visualization_manager.initialize()
        self._components['visualization'] = self._visualization_manager
    
    def get_component(self, name: str) -> Optional[object]:
        """Get a component by name."""
        return self._components.get(name)
    
    def get_version(self) -> str:
        """Get current system version."""
        return self._version
    
    def show_main_window(self) -> None:
        """Show the main application window."""
        self._main_window.show()
    
    def shutdown(self) -> None:
        """Shutdown the system and all components."""
        # Shutdown components in reverse order
        self._visualization_manager.shutdown()
        self._neural_manager.shutdown()
        self._bridge_manager.shutdown()
        
        self._initialized = False 
Main controller for the Lumina Frontend System.
Handles system initialization, version management, and component coordination.
"""

from typing import Dict, Optional
from PySide6.QtCore import QObject, Signal
from ..ui.windows.main_window import MainWindow
from ..integration.bridges.manager import BridgeManager
from ..neural.playground.manager import NeuralPlaygroundManager
from ..visualization.manager import VisualizationManager

class MainController(QObject):
    """Main controller class for the Lumina Frontend System."""
    
    # Signals
    system_initialized = Signal()
    version_changed = Signal(str)
    component_loaded = Signal(str)
    
    def __init__(self):
        super().__init__()
        self._version = "v7.5"
        self._components: Dict[str, object] = {}
        self._initialized = False
        
        # Initialize managers
        self._bridge_manager = BridgeManager()
        self._neural_manager = NeuralPlaygroundManager()
        self._visualization_manager = VisualizationManager()
        
        # Initialize main window
        self._main_window = MainWindow(self)
    
    def initialize(self) -> None:
        """Initialize the system and all components."""
        if self._initialized:
            return
            
        # Initialize components in order
        self._initialize_bridges()
        self._initialize_neural_system()
        self._initialize_visualization()
        
        self._initialized = True
        self.system_initialized.emit()
    
    def _initialize_bridges(self) -> None:
        """Initialize bridge connections."""
        self._bridge_manager.initialize()
        self._components['bridges'] = self._bridge_manager
    
    def _initialize_neural_system(self) -> None:
        """Initialize neural playground and breathing system."""
        self._neural_manager.initialize()
        self._components['neural'] = self._neural_manager
    
    def _initialize_visualization(self) -> None:
        """Initialize visualization system."""
        self._visualization_manager.initialize()
        self._components['visualization'] = self._visualization_manager
    
    def get_component(self, name: str) -> Optional[object]:
        """Get a component by name."""
        return self._components.get(name)
    
    def get_version(self) -> str:
        """Get current system version."""
        return self._version
    
    def show_main_window(self) -> None:
        """Show the main application window."""
        self._main_window.show()
    
    def shutdown(self) -> None:
        """Shutdown the system and all components."""
        # Shutdown components in reverse order
        self._visualization_manager.shutdown()
        self._neural_manager.shutdown()
        self._bridge_manager.shutdown()
        
        self._initialized = False 
 