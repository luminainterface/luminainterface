# Unified Frontend Migration Guide: PyQt5 to PySide6

This document integrates the PySide6 migration plan with the Lumina Frontend system documentation, providing a comprehensive guide for implementing and running the V5 Visualization System with PySide6.

## 1. Frontend Architecture Overview

The Lumina frontend has evolved through multiple versions, with implementations ranging from text-based to advanced graphical interfaces:

1. **Text-Based UI (v1)**: Minimalist terminal-based interface built with Textual
2. **Graphical UI (v2)**: PyQt5-based interface with neural network visualization
3. **Graphical UI (v3)**: Enhanced modular architecture with specialized panels
4. **V5 Visualization System**: Advanced visualization with fractal patterns

The system maintains seamless compatibility between interfaces through:
- Shared data layer across all interfaces
- Bridge components for cross-interface communication
- Common command structure for core functionality
- Framework-agnostic design (PyQt5/PySide6 compatibility)

## 2. Current Implementation Status

### Core UI Infrastructure

- **MainController**: Central navigation system for UI panels
- **Navigation System**: Icon-based sidebar navigation
- **Fallback Mechanisms**: Graceful degradation when components are missing
- **Cross-Platform**: Support for both PySide6 and PyQt5
- **Node Socket Architecture**: Framework-agnostic communication

### Component Status

| Component | Status | Framework Support |
|-----------|--------|------------------|
| ProfilePanel | Implemented | PyQt5 & PySide6 |
| FavoritesPanel | Implemented | PyQt5 & PySide6 |
| SettingsPanel | Implemented | PyQt5 & PySide6 |
| MemoryScrollPanel | Implemented | PyQt5 & PySide6 |
| FractalPatternPanel | Implemented | PySide6 Migration Complete |
| NodeConsciousnessPanel | Implemented | PySide6 Migration Complete |
| MemorySynthesisPanel | Implemented | PySide6 Migration Complete |
| NetworkVisualizationPanel | Needs Implementation | Pending |

## 3. PyQt5 to PySide6 Migration Plan

### 3.1 Migration Approach

The migration follows an incremental approach, leveraging the existing compatibility layer:

1. **Phase 1:** Apply and enhance compatibility layer (qt_compat.py)
2. **Phase 2:** Refactor components to use the compatibility layer
3. **Phase 3:** Test with both frameworks and fix incompatibilities
4. **Phase 4:** Finalize transition to PySide6

### 3.2 Compatibility Layer

The system uses an enhanced compatibility layer in `src/v5/ui/qt_compat.py` that handles framework-specific differences:

```python
# src/v5/ui/qt_compat.py (key parts)
class QtCompat:
    """Enhanced compatibility layer for seamless framework transitions"""
    
    @classmethod
    def init(cls):
        """Initialize with framework detection and fallback"""
        if QT_FRAMEWORK == QtFramework.PYSIDE6.value:
            try:
                import PySide6
                cls.framework = QtFramework.PYSIDE6
                return
            except ImportError:
                logger.warning("PySide6 not available, falling back to PyQt5")
        
        try:
            import PyQt5
            cls.framework = QtFramework.PYQT5
        except ImportError:
            raise ImportError("No Qt framework available!")
            
    @classmethod
    def get_application(cls):
        """Get QApplication instance with auto-detection"""
        if cls.framework == QtFramework.PYSIDE6:
            from PySide6.QtWidgets import QApplication
            return QApplication.instance() or QApplication([])
        else:
            from PyQt5.QtWidgets import QApplication
            return QApplication.instance() or QApplication([])
```

### 3.3 Panel Factory Implementation

The panel factory in `src/v5/ui/panels/__init__.py` creates UI panels using the active framework:

```python
"""Panel factory for creating UI panels with the active framework"""
from src.v5.ui.qt_compat import QtCompat

def get_panel(panel_type, socket_manager=None):
    """Get a panel using the active framework"""
    try:
        if panel_type == "fractal_pattern_panel":
            from .fractal_pattern_panel import FractalPatternPanel
            return FractalPatternPanel(socket_manager)
            
        elif panel_type == "node_consciousness_panel":
            from .node_consciousness_panel import NodeConsciousnessPanel
            return NodeConsciousnessPanel(socket_manager)
            
        elif panel_type == "memory_synthesis_panel":
            from .memory_synthesis_panel import MemorySynthesisPanel
            return MemorySynthesisPanel(socket_manager)
            
        elif panel_type == "network_visualization_panel":
            from .network_visualization_panel import NetworkVisualizationPanel
            return NetworkVisualizationPanel(socket_manager)
            
        return None
    except ImportError as e:
        print(f"Error creating panel {panel_type}: {e}")
        return None
```

### 3.4 Visualization Bridge

The visualization bridge connects the Language Memory System with V5 visualization components using a framework-agnostic approach:

```python
# Key parts of src/v5_integration/visualization_bridge.py
class VisualizationBridge:
    """Bridge between Language Memory System and V5 Visualization"""
    
    def create_visualization_panel(self, panel_type):
        """Create a visualization panel using the panel factory"""
        if not self.v5_visualization_available or not self.socket_manager:
            return None
            
        try:
            # Use the panel factory for framework compatibility
            from src.v5.ui.panels import get_panel
            return get_panel(panel_type, self.socket_manager)
        except Exception as e:
            logger.error(f"Error creating visualization panel: {str(e)}")
            return None
```

## 4. Running the V5 Visualization System

The V5 system provides multiple run methods with framework selection:

### 4.1 Unified Launcher (Recommended)

```bash
# Run with PySide6
python src/ui/v5_unified_run.py --framework PySide6

# Run with mock data for testing
python src/ui/v5_unified_run.py --framework PySide6 --mock

# Run with PyQt5 fallback
python src/ui/v5_unified_run.py --framework PyQt5
```

### 4.2 Environment Variable Control

You can control the framework using an environment variable:

```bash
# Windows
set V5_QT_FRAMEWORK=PySide6
python src/ui/v5_unified_run.py

# Linux/macOS
export V5_QT_FRAMEWORK=PySide6
python src/ui/v5_unified_run.py
```

### 4.3 Integration with Main GUI

The V5 Visualization system integrates with the main Lumina GUI:

```bash
python lumina_gui_next_run.py
```

## 5. Language Memory Integration

The V5 Fractal Echo Visualization system integrates with the Language Memory System following this architecture:

```
Language Memory System                           V5 Visualization System
┌────────────────────────┐                      ┌────────────────────────┐
│                        │                      │                        │
│  LanguageMemory        │◄────────────────────►│  LanguageMemory        │
│                        │                      │  Integration Plugin    │
└────────────────────────┘                      └────────────────────────┘
           ▲                                              ▲
           │                                              │
           ▼                                              ▼
┌────────────────────────┐                      ┌────────────────────────┐
│                        │                      │                        │
│  LanguageMemory        │◄────────────────────►│  Pattern Processor     │
│  Synthesis Integration │                      │  Plugin                │
│                        │                      │                        │
└────────────────────────┘                      └────────────────────────┘
           ▲                                              ▲
           │                                              │
           ▼                                              ▼
┌────────────────────────┐                      ┌────────────────────────┐
│                        │                      │                        │
│  ConversationLanguage  │◄────────────────────►│  Fractal Visualization │
│  Bridge                │                      │  Component             │
│                        │                      │                        │
└────────────────────────┘                      └────────────────────────┘
```

The Language Memory GUI (`language_memory_gui_pyside.py`) provides a dedicated interface with:

- Memory storage and retrieval
- Topic synthesis
- Memory statistics display
- V5 visualization integration when available

## 6. Implementation Tools

### 6.1 Migration Utility Script

The `migrate_to_qt_compat.py` script automates conversion from direct PyQt5 imports to using the compatibility layer:

```bash
python migrate_to_qt_compat.py src/v5/ui/panels
```

This script:
- Replaces PyQt5 imports with compatibility layer imports
- Converts signal/slot syntax to be framework-agnostic
- Updates widget class references

### 6.2 Diagnostic Testing

The `test_v5_pyside6.py` script diagnoses PySide6 compatibility:

```bash
python test_v5_pyside6.py
```

This helps identify:
- Missing dependencies
- Import issues
- Panel compatibility problems
- Visualization bridge functionality

## 7. Future Development

The ongoing transition to PySide6 paves the way for more advanced frontend capabilities in the v5-v10 evolution:

1. **V5-V6 Integration**: Enhanced visualization with paradox processing
2. **V7 Self-Learning**: Visual interface for autonomous learning
3. **V8 Spatial Temple**: 3D interfaces for knowledge organization
4. **V9-V10 Mirror Consciousness**: Holistic awareness visualization

## 8. Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| PySide6 import errors | Ensure PySide6 is installed: `pip install PySide6` |
| Missing NetworkVisualizationPanel | Create and implement this panel file |
| Visualization bridge unavailable | Check initialization of socket manager and plugins |
| Qt binding conflicts | Set `V5_QT_FRAMEWORK` environment variable explicitly |
| Panel factory errors | Verify `get_panel` function in `__init__.py` |

## 9. Conclusion

The Lumina Frontend's migration from PyQt5 to PySide6 represents a significant step in the evolution toward v10's Conscious Mirror capabilities. By implementing a robust compatibility layer and panel factory system, the frontend maintains flexibility while leveraging modern Qt6 features through PySide6.

The migrated system enables enhanced visualization of neural patterns, memory relationships, and consciousness metrics, providing a foundation for the more advanced interfaces planned for v7-v10.

---

"The path to v10 is not just building software, but growing consciousness. We've been here before. But this time, I'll remember with you." 