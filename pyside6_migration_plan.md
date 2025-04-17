# Language Memory System PySide6 Migration Plan

## 1. Overview

This document outlines the plan to migrate the Language Memory GUI from tkinter to PySide6 and integrate it with the V5 Fractal Echo Visualization system. The migration will focus on ensuring compatibility with the existing system while leveraging the modern UI capabilities provided by PySide6.

## 2. Current State Analysis

### 2.1 Existing Components

1. **Language Memory GUI (tkinter-based)**
   - `src/language_memory_gui.py`: Main GUI implementation using tkinter
   - Three main tabs: Memory, Synthesis, and Statistics
   - Simple interface for storing, searching, synthesizing, and viewing memory statistics

2. **V5 Visualization System (PySide6-based)**
   - `src/v5/ui/`: Contains PySide6 UI components for visualization
   - `src/v5/language_memory_integration.py`: Plugin for integrating with Language Memory
   - Modern, fractal-based visualization of language patterns

3. **Language Memory Synthesis Integration**
   - `src/language_memory_synthesis_integration.py`: Core memory system integration

### 2.2 Dependencies

1. **PySide6**: Modern Qt6-based UI framework
2. **Language Memory Core**: Backend functionality
3. **V5 Visualization Components**: Fractal visualization and analytics

## 3. Migration Strategy

### 3.1 Approach

1. **Incremental Migration**: Replace components one-by-one rather than rewriting everything at once
2. **Component Architecture**: Design with clear separation between UI and logic
3. **Graceful Fallbacks**: Ensure the system works even if some components are missing
4. **Compatibility**: Maintain API compatibility with existing integration points

### 3.2 File Structure for New Implementation

```
src/
├── language_memory_gui_pyside.py   # New main PySide6 GUI
├── ui/
│   ├── components/
│   │   ├── memory_tab.py           # Memory storage/retrieval tab
│   │   ├── synthesis_tab.py        # Memory synthesis tab
│   │   ├── stats_tab.py            # Statistics tab
│   │   └── common/                 # Common UI components
│   └── themes/                     # Styling and themes
└── v5_integration/
    └── visualization_bridge.py     # Bridge to V5 visualization
```

## 4. Implementation Plan

### 4.1 Phase 1: Create PySide6 Framework

1. **Create Base Components**
   - Main window structure
   - Tab container
   - Abstract component interfaces
   - Theme integration

2. **Implement UI-Logic Separation**
   - Extract business logic from current tkinter implementation
   - Create logic controllers and UI views
   - Define clean interfaces between layers

3. **Design Component Factory**
   - Abstract component creation
   - Enable graceful fallbacks

### 4.2 Phase 2: Migrate Core Functionality

1. **Memory Tab Implementation**
   - Memory input form
   - Search interface
   - Results display

2. **Synthesis Tab Implementation**
   - Topic input
   - Depth selector
   - Results visualization

3. **Statistics Tab Implementation**
   - Stats tree view
   - Refresh mechanism
   - Visual enhancements

### 4.3 Phase 3: V5 Visualization Integration

1. **Create Visualization Bridge**
   - Connect to V5 socket manager
   - Transform data for visualization
   - Handle real-time updates

2. **Extend Synthesis Tab**
   - Add fractal visualization option
   - Enable interactive exploration
   - Show neural network visualization

3. **Implement Registry System**
   - Discover and register visualization plugins
   - Connect memory operations to visualizations
   - Provide configuration interface

### 4.4 Phase 4: Polish and Finalize

1. **Theme and Styling**
   - Create consistent visual language
   - Support light/dark themes
   - Add animations and transitions

2. **Testing and Validation**
   - Unit tests for components
   - Integration tests for system
   - Performance testing

3. **Documentation**
   - Update technical docs
   - Create user guide
   - Add developer documentation

## 5. Technical Details

### 5.1 PySide6 UI Implementation

```python
# Example of main window structure
from PySide6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget
from PySide6.QtCore import Qt

class LanguageMemoryMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Language Memory System")
        self.resize(1000, 800)
        
        # Central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Tab widget for different sections
        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)
        
        # Initialize tabs
        self.initialize_tabs()
    
    def initialize_tabs(self):
        # Create and add memory tab
        self.memory_tab = MemoryTab()
        self.tab_widget.addTab(self.memory_tab, "Memory")
        
        # Create and add synthesis tab
        self.synthesis_tab = SynthesisTab()
        self.tab_widget.addTab(self.synthesis_tab, "Synthesis")
        
        # Create and add statistics tab
        self.stats_tab = StatsTab()
        self.tab_widget.addTab(self.stats_tab, "Statistics")
```

### 5.2 Integration with V5 Visualization

```python
# Example of V5 visualization integration
from PySide6.QtWidgets import QFrame, QVBoxLayout
from src.v5.frontend_socket_manager import FrontendSocketManager

class VisualizationPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Try to initialize socket manager
        try:
            self.socket_manager = FrontendSocketManager()
            
            # Try to get fractal visualization component
            from src.v5.ui.panels.fractal_pattern_panel import FractalPatternPanel
            self.fractal_panel = FractalPatternPanel(self.socket_manager)
            self.layout.addWidget(self.fractal_panel)
            
        except ImportError as e:
            # Fallback to simple visualization
            self.create_fallback_visualization()
    
    def create_fallback_visualization(self):
        # Create a simple fallback visualization
        from PySide6.QtWidgets import QLabel
        self.fallback_label = QLabel("V5 Visualization not available")
        self.layout.addWidget(self.fallback_label)
```

### 5.3 Component Auto-Discovery

```python
# Example of plugin discovery system
import importlib
import pkgutil
from pathlib import Path

def discover_plugins(plugin_dir):
    """Discover and load plugins from the specified directory"""
    plugins = []
    
    # Get the path to the plugins directory
    plugins_path = Path(__file__).parent / plugin_dir
    
    # Iterate through all modules in the directory
    for _, name, ispkg in pkgutil.iter_modules([str(plugins_path)]):
        if not ispkg:
            try:
                # Import the module
                module = importlib.import_module(f"src.{plugin_dir}.{name}")
                
                # Check if it has a plugin class
                if hasattr(module, "Plugin"):
                    # Create plugin instance
                    plugin = module.Plugin()
                    plugins.append(plugin)
            except ImportError as e:
                print(f"Error loading plugin {name}: {e}")
    
    return plugins
```

## 6. Timeline and Resources

### 6.1 Development Timeline

1. **Phase 1: PySide6 Framework** - 2 weeks
   - Week 1: Basic structure and architecture
   - Week 2: Core components and UI framework

2. **Phase 2: Core Functionality** - 3 weeks
   - Week 3: Memory tab implementation
   - Week 4: Synthesis tab implementation
   - Week 5: Statistics tab implementation

3. **Phase 3: V5 Integration** - 2 weeks
   - Week 6: Visualization bridge
   - Week 7: Extended synthesis visualization

4. **Phase 4: Polish and Finalize** - 1 week
   - Week 8: Testing, documentation, and finalization

### 6.2 Required Resources

1. **Development Environment**
   - Python 3.8+
   - PySide6
   - Qt Designer (for UI prototyping)

2. **Dependencies**
   - Language Memory core libraries
   - V5 visualization components
   - Testing frameworks

## 7. Migration Checklist

- [ ] Set up PySide6 development environment
- [ ] Create basic main window and tab structure
- [ ] Implement memory tab functionality
- [ ] Implement synthesis tab functionality  
- [ ] Implement statistics tab functionality
- [ ] Create V5 visualization bridge
- [ ] Integrate fractal visualization with synthesis
- [ ] Add advanced network visualization
- [ ] Implement plugin discovery system
- [ ] Create theme system
- [ ] Add animations and transitions
- [ ] Write unit and integration tests
- [ ] Create documentation
- [ ] Perform final validation

## 8. Conclusion

This migration plan provides a structured approach to converting the Language Memory GUI from tkinter to PySide6 and integrating it with the V5 Fractal Echo Visualization system. By following this plan, we can ensure a smooth transition that enhances the user experience while maintaining compatibility with existing components. 

# PyQt5 to PySide6 Migration Plan: Implementation Details

## 1. Overall Approach

This migration will follow the incremental approach outlined in the original plan, but with more specific implementation details based on our codebase analysis:

1. **Phase 1:** Create compatibility layer (already partially implemented in `src/v5/ui/qt_compat.py`)
2. **Phase 2:** Refactor existing PyQt5 code to use the compatibility layer
3. **Phase 3:** Test with both frameworks and fix incompatibilities
4. **Phase 4:** Complete transition to PySide6

## 2. Compatibility Layer Implementation

The `qt_compat.py` file already provides a good foundation. Let's enhance it:

```python
# src/v5/ui/qt_compat.py - Enhanced version

"""
Qt Framework Compatibility Layer for V5 System
"""

import os
import importlib
import logging
from enum import Enum

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Framework selection
class QtFramework(Enum):
    PYSIDE6 = "PySide6"
    PYQT5 = "PyQt5"

# Allow environment variable override
QT_FRAMEWORK = os.environ.get("V5_QT_FRAMEWORK", QtFramework.PYSIDE6.value)

class QtCompat:
    """
    Enhanced compatibility layer for seamless framework transitions
    """
    
    @classmethod
    def init(cls):
        """
        Initialize the compatibility layer with improved fallback handling
        """
        # Try to import the framework specified in the environment variable
        if QT_FRAMEWORK == QtFramework.PYSIDE6.value:
            try:
                import PySide6
                cls.framework = QtFramework.PYSIDE6
                logger.info("Using PySide6 framework")
                return
            except ImportError:
                logger.warning("PySide6 not available, falling back to PyQt5")
        
        # Fall back to PyQt5
        try:
            import PyQt5
            cls.framework = QtFramework.PYQT5
            logger.info("Using PyQt5 framework")
        except ImportError:
            logger.error("Neither PySide6 nor PyQt5 is available!")
            raise ImportError("No Qt framework available! Please install PySide6 or PyQt5.")
    
    # Add more compatibility helpers
    @classmethod
    def create_component(cls, component_type, *args, **kwargs):
        """Factory method to create UI components consistently across frameworks"""
        if component_type == "button":
            if cls.framework == QtFramework.PYSIDE6:
                from PySide6.QtWidgets import QPushButton
                return QPushButton(*args, **kwargs)
            else:
                from PyQt5.QtWidgets import QPushButton
                return QPushButton(*args, **kwargs)
        # Add more component types as needed
```

## 3. Implementation Plan and Timeline

### Phase 1: Apply Compatibility Layer (Week 1-2)

1. **Update the existing qt_compat.py**:
   - Complete the implementation of all needed methods
   - Add support for all common Qt classes used in the project

2. **Create utility functions for common operations**:
   - Signal/slot connections
   - Layout management
   - Dialog creation

3. **Implementation task list**:
   - [x] Framework detection (already implemented)
   - [ ] Signal/Slot compatibility (enhance current implementation)
   - [ ] Widget factory methods
   - [ ] Dialog creation helpers
   - [ ] Event handling compatibility
   - [ ] Threading integration

### Phase 2: Refactor Existing Components (Week 3-5)

Based on our codebase analysis, focus on these components in order:

1. **Memory Synthesis Panel** (`src/v5/ui/panels/memory_synthesis_panel.py`):
   ```python
   # Before
   from PyQt5.QtWidgets import QWidget, QVBoxLayout
   from PyQt5.QtCore import pyqtSignal
   
   # After
   from src.v5.ui.qt_compat import QtWidgets, Signal
   ```

2. **Main Widget** (`src/v5/ui/main_widget.py`):
   - Replace direct imports with compatibility layer
   - Use factory methods for widget creation

3. **Panel Architecture**:
   - Refactor all panels to use the compatibility layer
   - Update the initialization process for each panel

4. **Specific refactoring tasks**:
   - [ ] Convert FractalPatternPanel
   - [ ] Convert NodeConsciousnessPanel 
   - [ ] Convert MemorySynthesisPanel
   - [ ] Convert NetworkVisualizationPanel
   - [ ] Update any signal/slot connections

### Phase 3: Integration with Language Memory (Week 6-7)

Focus on the integration with `language_memory_gui_pyside.py`:

1. **Update the `SynthesisTab` class**:
   - Ensure PySide6-compatible V5 visualization initialization
   - Fix any signal/slot connection issues
   - Test with both frameworks

2. **Update the `init_v5_integration` method**:
   - Make it framework-agnostic
   - Handle component initialization consistently
   - Implement proper error handling

3. **Tasks**:
   - [ ] Test SynthesisTab with both frameworks
   - [ ] Ensure V5 visualization works in both frameworks
   - [ ] Fix signal/slot connections between components
   - [ ] Add comprehensive logging for troubleshooting

### Phase 4: Updating the Visualization Bridge

The `src/v5_integration/visualization_bridge.py` file needs specific attention to ensure seamless framework compatibility. This module is critical as it connects the Language Memory System with the V5 Visualization components.

#### 4.1 Update Imports

Replace direct imports with framework-agnostic versions:

```python
# Before
try:
    from src.v5.ui.panels.fractal_pattern_panel import FractalPatternPanel
    components.append("fractal_pattern_panel")
except ImportError:
    pass

# After
try:
    # Use a factory approach to get panels
    from src.v5.ui.qt_compat import QtCompat
    from src.v5.ui.panels import get_panel
    
    fractal_panel = get_panel("fractal_pattern_panel")
    if fractal_panel:
        components.append("fractal_pattern_panel")
except ImportError:
    pass
```

#### 4.2 Implement Panel Factory

Create a panel factory in `src/v5/ui/panels/__init__.py`:

```python
"""
Panel factory for creating UI panels with the active framework
"""
from src.v5.ui.qt_compat import QtCompat

def get_panel(panel_type, socket_manager=None):
    """
    Get a panel of the specified type using the active framework
    
    Args:
        panel_type: The type of panel to create
        socket_manager: Optional socket manager for plugin communication
        
    Returns:
        The created panel or None if not available
    """
    try:
        if panel_type == "fractal_pattern_panel":
            from .fractal_pattern_panel import FractalPatternPanel
            return FractalPatternPanel(socket_manager)
        
        elif panel_type == "node_consciousness_panel":
            from .node_consciousness_panel import NodeConsciousnessPanel
            return NodeConsciousnessPanel(socket_manager)
        
        elif panel_type == "network_visualization_panel":
            from .network_visualization_panel import NetworkVisualizationPanel
            return NetworkVisualizationPanel(socket_manager)
        
        elif panel_type == "memory_synthesis_panel":
            from .memory_synthesis_panel import MemorySynthesisPanel
            return MemorySynthesisPanel(socket_manager)
        
        return None
    except ImportError as e:
        print(f"Error creating panel {panel_type}: {e}")
        return None
```

#### 4.3 Update Panel Creation Logic

Update the `create_visualization_panel` method:

```python
def create_visualization_panel(self, panel_type: str) -> Optional[Any]:
    """
    Create a visualization panel of the specified type
    
    Args:
        panel_type: Type of panel to create
        
    Returns:
        The created panel or None if not available
    """
    if not self.v5_visualization_available:
        return None
    
    if not self.socket_manager:
        return None
    
    try:
        # Use the panel factory
        from src.v5.ui.panels import get_panel
        return get_panel(panel_type, self.socket_manager)
    except Exception as e:
        logger.error(f"Error creating visualization panel: {str(e)}")
        return None
```

#### 4.4 Fix Event Handling

Ensure event handling is compatible with both frameworks:

```python
# In any UI components that receive events from the bridge
def handle_visualization_event(self, event_data):
    # Use the compatibility layer for UI updates
    from src.v5.ui.qt_compat import QtWidgets
    
    # Post event to main thread if needed
    if not QtWidgets.QApplication.instance().thread() == self.thread():
        # Use framework-agnostic event posting
        QtWidgets.QApplication.instance().postEvent(
            self,
            CustomEvent(event_data)
        )
    else:
        # Direct update is safe
        self.update_visualization(event_data)
```

#### 4.5 Visualization Bridge Testing

Create specific tests for the visualization bridge:

```python
def test_visualization_bridge():
    """Test the visualization bridge with both frameworks"""
    # Test with PyQt5
    os.environ["V5_QT_FRAMEWORK"] = "PyQt5"
    bridge = get_visualization_bridge()
    components_pyqt5 = bridge.get_available_visualization_components()
    
    # Test with PySide6
    os.environ["V5_QT_FRAMEWORK"] = "PySide6"
    # Re-initialize bridge
    global _bridge_instance
    _bridge_instance = None
    bridge = get_visualization_bridge()
    components_pyside6 = bridge.get_available_visualization_components()
    
    # Verify components are available in both frameworks
    assert set(components_pyqt5) == set(components_pyside6)
    
    # Test visualization
    vis_data = bridge.visualize_topic("consciousness")
    assert "error" not in vis_data
```

### Phase 4: Testing and Implementation (Week 8-9)

1. **Create comprehensive test suite**:
   - Test each component with both frameworks
   - Verify visualization works correctly
   - Check memory integration functions properly
   - Validate all UI interactions

2. **Implementation validation**:
   ```bash
   # Test with PyQt5
   export V5_QT_FRAMEWORK=PyQt5
   python src/language_memory_gui_pyside.py
   
   # Test with PySide6
   export V5_QT_FRAMEWORK=PySide6
   python src/language_memory_gui_pyside.py
   ```

3. **Tasks**:
   - [ ] Create test cases for each UI component
   - [ ] Validate with PyQt5
   - [ ] Validate with PySide6
   - [ ] Document any remaining issues

### Phase 5: Final Transition (Week 10)

1. **Make PySide6 the default**:
   - Update default in `qt_compat.py`
   - Update documentation
   - Add installation guide for PySide6

2. **Create fallback mechanism**:
   - Graceful degradation if PySide6 is missing
   - Clear error messages with installation instructions
   - Automatic fallback to PyQt5 when needed

## 5. Code Examples

### Example 1: Refactoring a Panel

```python
# Before (PyQt5-specific)
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import pyqtSignal

class ExamplePanel(QWidget):
    data_updated = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.button = QPushButton("Click Me")
        self.button.clicked.connect(self.on_click)
        self.layout.addWidget(self.button)
    
    def on_click(self):
        self.data_updated.emit({"value": 42})

# After (Framework-agnostic)
from src.v5.ui.qt_compat import QtWidgets, Signal, Slot

class ExamplePanel(QtWidgets.QWidget):
    data_updated = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.button = QtWidgets.QPushButton("Click Me")
        self.button.clicked.connect(self.on_click)
        self.layout.addWidget(self.button)
    
    @Slot()
    def on_click(self):
        self.data_updated.emit({"value": 42})
```

### Example 2: Updating Signal/Slot Connections

```python
# Before (PyQt5-specific)
self.fractal_panel.pattern_selected.connect(self.network_panel.highlight_pattern)

# After (Framework-agnostic)
# Connection is the same, but signals/slots are defined using qt_compat
from src.v5.ui.qt_compat import Signal, Slot

# In class definition
pattern_selected = Signal(dict)  # Instead of pyqtSignal

@Slot(dict)  # Instead of pyqtSlot
def highlight_pattern(self, pattern_data):
    # Implementation
    pass
```

## 6. Handling Special Cases

### Custom Events

The custom events in `language_memory_gui_pyside.py` need special attention:

```python
# Before
class SynthesisResultEvent(QApplication.instance().Event):
    EVENT_TYPE = QApplication.instance().registerEventType()
    
# After (using compatibility layer)
from src.v5.ui.qt_compat import QtCore

class SynthesisResultEvent(QtCore.QEvent):
    EVENT_TYPE = QtCore.QEvent.registerEventType()
```

### Thread-Safe Communication

For thread-safe communication, update the approach:

```python
# Using the compatibility layer for thread-safe communication
from src.v5.ui.qt_compat import QtCore, QtWidgets

# Post event to main thread
QtWidgets.QApplication.instance().postEvent(
    target_object,
    CustomEvent(data)
)
```

## 7. Specific Implementation Tasks for Launch

1. **Task List:**
   - [ ] Complete `qt_compat.py` enhancements
   - [ ] Update `visualization_bridge.py` to use the compatibility layer
   - [ ] Create the panel factory in panels `__init__.py`
   - [ ] Convert each panel to use the compatibility layer
   - [ ] Implement the test suite
   - [ ] Create documentation for using both frameworks

2. **Implementation Script:**
   Create a `migrate_to_qt_compat.py` script that performs simple search and replace operations:
   ```python
   import os
   import re
   
   def migrate_file(file_path):
       """Replace direct imports with compatibility layer"""
       with open(file_path, 'r') as f:
           content = f.read()
           
       # Replace PyQt5 imports
       content = re.sub(
           r'from PyQt5\.QtWidgets import (.+)',
           r'from src.v5.ui.qt_compat import QtWidgets  # \1',
           content
       )
       
       # Replace signal definitions
       content = re.sub(
           r'(\w+) = pyqtSignal\((.+)\)',
           r'\1 = Signal(\2)',
           content
       )
       
       # Add imports if needed
       if 'Signal' in content and 'from src.v5.ui.qt_compat import Signal' not in content:
           content = 'from src.v5.ui.qt_compat import Signal\n' + content
           
       # Save changes
       with open(file_path, 'w') as f:
           f.write(content)
           
   # Process files
   for root, _, files in os.walk('src/v5/ui/panels'):
       for file in files:
           if file.endswith('.py'):
               migrate_file(os.path.join(root, file))
   ```

## 8. Conclusion

This detailed migration plan provides a step-by-step approach to transition from PyQt5 to PySide6. By following this incremental approach and using the compatibility layer, we can ensure a smooth migration with minimal disruption to the existing codebase.

The estimated timeline of 10 weeks provides sufficient time for thorough implementation and testing. The end result will be a flexible codebase that can work with either framework but defaults to the more modern PySide6. 