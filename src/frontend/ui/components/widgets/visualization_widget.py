from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt, QTimer, Signal
from typing import Dict, Any, Optional
import logging

class VisualizationWidget(QWidget):
    """Base class for visualization widgets."""
    
    # Signals for testing and monitoring
    visualization_updated = Signal()  # Emitted after each update
    error_occurred = Signal(str)  # Emitted when an error occurs
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.layout = QVBoxLayout(self)
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_visualization)
        self.is_animating = False
        self._initialized = False
        self._cleanup_called = False
        
    def initialize(self, params: Dict[str, Any]) -> bool:
        """Initialize the visualization with the given parameters."""
        try:
            self._initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize visualization: {str(e)}")
            self.error_occurred.emit(str(e))
            return False
            
    def start_animation(self, interval_ms: int = 16):
        """Start the visualization animation."""
        try:
            if not self._initialized:
                self.logger.error("Cannot start animation before initialization")
                return
                
            self.logger.info("Starting visualization animation")
            self.is_animating = True
            self.animation_timer.start(interval_ms)
        except Exception as e:
            self.logger.error(f"Failed to start animation: {str(e)}")
            self.error_occurred.emit(str(e))
            
    def stop_animation(self):
        """Stop the visualization animation."""
        try:
            self.logger.info("Stopping visualization animation")
            self.is_animating = False
            self.animation_timer.stop()
        except Exception as e:
            self.logger.error(f"Failed to stop animation: {str(e)}")
            self.error_occurred.emit(str(e))
            
    def update_visualization(self):
        """Update the visualization state."""
        if not self.is_initialized():
            return
        self.update()  # Trigger a repaint
        
    def capture_frame(self) -> Optional[Any]:
        """Capture the current visualization frame."""
        try:
            if not self._initialized:
                return None
                
            self.logger.info("Capturing visualization frame")
            # Base capture logic
            return None
        except Exception as e:
            self.logger.error(f"Failed to capture frame: {str(e)}")
            self.error_occurred.emit(str(e))
            return None
            
    def resizeEvent(self, event):
        """Handle widget resize events."""
        try:
            super().resizeEvent(event)
            if self._initialized:
                # Handle resize logic
                pass
        except Exception as e:
            self.logger.error(f"Error during resize: {str(e)}")
            self.error_occurred.emit(str(e))
            
    def cleanup(self):
        """Clean up visualization resources."""
        self._initialized = False
            
    def __del__(self):
        """Ensure cleanup is called when widget is destroyed."""
        if not self._cleanup_called:
            self.cleanup()
            
    def is_initialized(self) -> bool:
        """Check if the widget is initialized."""
        return self._initialized
        
    def is_animating(self) -> bool:
        """Check if the visualization is currently animating."""
        return self.is_animating 