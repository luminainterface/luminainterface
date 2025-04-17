"""
Breath-Enhanced Conversation Panel

Extends the V5 conversation panel with V7's breath detection capabilities to
dynamically adjust the NN/LLM weighting based on detected breath patterns.
"""

import os
import sys
import logging
import time
from pathlib import Path

# Set up logging
logger = logging.getLogger("V7BreathConversation")

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import Qt compatibility layer - we'll try to use the V5 one if available
try:
    from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui, Qt, Signal, Slot
except ImportError:
    logger.warning("V5 Qt compatibility layer not found. Using direct PySide6 imports.")
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        from PySide6.QtCore import Qt, Signal, Slot
    except ImportError:
        logger.error("PySide6 not found. Please install PySide6 or configure the V5 Qt compatibility layer.")
        sys.exit(1)

# Import base ConversationPanel from V5
try:
    from src.v5.ui.panels.conversation_panel import ConversationPanel
    V5_CONVERSATION_AVAILABLE = True
except ImportError:
    logger.error("V5 ConversationPanel not found. Cannot extend functionality.")
    V5_CONVERSATION_AVAILABLE = False

# Import breath detector
from src.v7.breath_detector import BreathDetector, BreathPattern

class BreathPatternIndicator(QtWidgets.QWidget):
    """Widget displaying the current breath pattern and its effect on NN/LLM weighting"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.pattern = "relaxed"
        self.confidence = 0.85
        self.nn_weight = 0.5
        
        self.setMinimumHeight(30)
        self.setMaximumHeight(30)
        
        # Create tooltip
        self.setToolTip("Breath pattern affects NN/LLM weighting")
    
    def update_pattern(self, pattern, confidence, nn_weight):
        """Update the displayed pattern"""
        self.pattern = pattern
        self.confidence = confidence
        self.nn_weight = nn_weight
        self.update()
    
    def paintEvent(self, event):
        """Paint the pattern indicator"""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Calculate colors and positions
        width = self.width()
        height = self.height()
        
        # Draw background
        painter.fillRect(0, 0, width, height, QtGui.QColor(18, 26, 36))
        
        # Draw pattern label
        label_rect = QtCore.QRect(5, 0, width // 2 - 10, height)
        painter.setPen(QtGui.QPen(QtGui.QColor(52, 152, 219)))
        painter.setFont(QtGui.QFont("Segoe UI", 9))
        pattern_text = f"Pattern: {self.pattern.title()}"
        painter.drawText(label_rect, Qt.AlignLeft | Qt.AlignVCenter, pattern_text)
        
        # Draw NN/LLM weight
        nn_percent = int(self.nn_weight * 100)
        llm_percent = 100 - nn_percent
        weight_rect = QtCore.QRect(width // 2, 0, width // 2 - 5, height)
        
        # Color varies based on weighting
        if self.nn_weight > 0.7:
            color = QtGui.QColor(231, 76, 60)  # Red for NN heavy
        elif self.nn_weight < 0.3:
            color = QtGui.QColor(46, 204, 113)  # Green for LLM heavy
        else:
            color = QtGui.QColor(52, 152, 219)  # Blue for balanced
        
        painter.setPen(QtGui.QPen(color))
        weight_text = f"NN: {nn_percent}% / LLM: {llm_percent}%"
        painter.drawText(weight_rect, Qt.AlignRight | Qt.AlignVCenter, weight_text)
        
        # Draw confidence indicator as a small circle
        painter.setBrush(QtGui.QBrush(color))
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 100), 1))
        
        # Size circle based on confidence
        circle_size = int(8 * self.confidence)
        circle_x = width // 2 - 10
        circle_y = height // 2
        painter.drawEllipse(circle_x - circle_size//2, circle_y - circle_size//2, circle_size, circle_size)

class BreathEnhancedConversationPanel(ConversationPanel if V5_CONVERSATION_AVAILABLE else QtWidgets.QWidget):
    """
    Conversation panel enhanced with breath detection capabilities
    
    This panel extends the V5 ConversationPanel to integrate with breath patterns
    for dynamic adjustment of the NN/LLM weighting.
    """
    
    # Additional signals
    breath_pattern_changed = Signal(str, float, float)  # pattern, confidence, nn_weight
    
    def __init__(self, socket_manager=None, breath_detector=None):
        """Initialize the breath-enhanced conversation panel"""
        super().__init__(socket_manager)
        
        self.breath_detector = breath_detector
        self.auto_adjust_weights = True
        self.last_manual_weight = 0.5
        
        # Initialize or create breath detector if not provided
        if not self.breath_detector:
            self.breath_detector = BreathDetector(socket_manager)
            self.breath_detector.start()
        
        # Add breath pattern elements to the UI
        self._add_breath_elements()
        
        # Connect to breath pattern changes
        self.breath_detector.register_pattern_listener(self._on_breath_pattern_changed)
        
        # Sync initial weight with detector
        current_nn_weight = self.breath_detector.get_nn_weight_for_pattern()
        self._update_nn_llm_weight(current_nn_weight)
        
        logger.info("Breath-enhanced conversation panel initialized")
    
    def _add_breath_elements(self):
        """Add breath-related UI elements"""
        try:
            # Create pattern indicator
            self.pattern_indicator = BreathPatternIndicator()
            
            # Current pattern is relaxed by default
            self.pattern_indicator.update_pattern(
                "relaxed", 
                0.85, 
                self.breath_detector.get_nn_weight_for_pattern()
            )
            
            # Create auto-adjust toggle
            auto_adjust_layout = QtWidgets.QHBoxLayout()
            auto_adjust_layout.setContentsMargins(5, 0, 5, 0)
            
            auto_adjust_label = QtWidgets.QLabel("Auto-adjust based on breath:")
            auto_adjust_label.setStyleSheet("color: #7F8C8D; font-size: 12px;")
            
            self.auto_adjust_toggle = QtWidgets.QCheckBox()
            self.auto_adjust_toggle.setChecked(self.auto_adjust_weights)
            self.auto_adjust_toggle.stateChanged.connect(self._on_auto_adjust_toggled)
            self.auto_adjust_toggle.setStyleSheet("""
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                }
                QCheckBox::indicator:unchecked {
                    border: 2px solid #7F8C8D;
                    background-color: #2C3E50;
                    border-radius: 4px;
                }
                QCheckBox::indicator:checked {
                    border: 2px solid #3498DB;
                    background-color: #2980B9;
                    border-radius: 4px;
                }
            """)
            
            auto_adjust_layout.addWidget(auto_adjust_label)
            auto_adjust_layout.addWidget(self.auto_adjust_toggle)
            auto_adjust_layout.addStretch()
            
            # Find the location to insert breath elements in the ConversationPanel
            # Weight settings are typically in the lower section of the panel
            if hasattr(self, 'weight_container') and isinstance(self.weight_container, QtWidgets.QWidget):
                # Add the pattern indicator above the weight slider
                layout = self.weight_container.layout()
                
                # Add separator
                separator = QtWidgets.QFrame()
                separator.setFrameShape(QtWidgets.QFrame.HLine)
                separator.setFrameShadow(QtWidgets.QFrame.Sunken)
                separator.setMaximumHeight(1)
                separator.setStyleSheet("background-color: #34495E;")
                
                # Add breath header
                breath_header = QtWidgets.QLabel("Breath Integration")
                breath_header.setStyleSheet("""
                    color: #9B59B6;
                    font-weight: bold;
                    font-size: 14px;
                    margin-top: 10px;
                """)
                
                layout.addWidget(separator)
                layout.addWidget(breath_header)
                layout.addWidget(self.pattern_indicator)
                layout.addLayout(auto_adjust_layout)
                
                # Add another separator for visual distinction
                separator2 = QtWidgets.QFrame()
                separator2.setFrameShape(QtWidgets.QFrame.HLine)
                separator2.setFrameShadow(QtWidgets.QFrame.Sunken)
                separator2.setMaximumHeight(1)
                separator2.setStyleSheet("background-color: #34495E;")
                layout.addWidget(separator2)
            
            else:
                # If we can't find the weight container, add to the main layout
                logger.warning("Could not find weight container, adding breath elements to main layout")
                
                # Create a container for breath elements
                breath_container = QtWidgets.QWidget()
                breath_layout = QtWidgets.QVBoxLayout(breath_container)
                breath_layout.setContentsMargins(10, 10, 10, 10)
                
                # Add breath header
                breath_header = QtWidgets.QLabel("Breath Integration")
                breath_header.setStyleSheet("""
                    color: #9B59B6;
                    font-weight: bold;
                    font-size: 14px;
                """)
                
                breath_layout.addWidget(breath_header)
                breath_layout.addWidget(self.pattern_indicator)
                breath_layout.addLayout(auto_adjust_layout)
                
                # Add to main layout
                if hasattr(self, 'layout'):
                    self.layout().addWidget(breath_container)
        
        except Exception as e:
            logger.error(f"Error adding breath elements: {e}")
    
    def _on_breath_pattern_changed(self, data):
        """Handle breath pattern changes from the detector"""
        pattern = data.get("pattern", "relaxed")
        confidence = data.get("confidence", 0.85)
        nn_weight = data.get("nn_weight", 0.5)
        
        # Update the indicator
        if hasattr(self, 'pattern_indicator'):
            self.pattern_indicator.update_pattern(pattern, confidence, nn_weight)
        
        # Emit our own signal
        self.breath_pattern_changed.emit(pattern, confidence, nn_weight)
        
        # Auto-adjust the weight if enabled
        if self.auto_adjust_weights:
            self._update_nn_llm_weight(nn_weight)
        
        logger.debug(f"Breath pattern changed: {pattern} (conf: {confidence:.2f}, weight: {nn_weight:.2f})")
    
    def _on_auto_adjust_toggled(self, state):
        """Handle toggling of auto-adjust checkbox"""
        self.auto_adjust_weights = (state == QtCore.Qt.Checked)
        
        if self.auto_adjust_weights:
            # Immediately adjust to current breath pattern
            nn_weight = self.breath_detector.get_nn_weight_for_pattern()
            self._update_nn_llm_weight(nn_weight)
            logger.debug(f"Auto-adjust enabled, set weight to {nn_weight:.2f}")
        else:
            # Store current weight as manual setting
            self.last_manual_weight = self.nn_llm_weight
            logger.debug(f"Auto-adjust disabled, manual weight: {self.last_manual_weight:.2f}")
    
    def _update_nn_llm_weight(self, nn_weight):
        """Update the NN/LLM weight based on breath pattern"""
        # Only update if auto-adjust is enabled
        if not self.auto_adjust_weights:
            return
        
        # Update our internal weight
        self.nn_llm_weight = nn_weight
        
        # Update slider if it exists
        if hasattr(self, 'weight_slider'):
            # Block signals to prevent loops
            self.weight_slider.blockSignals(True)
            self.weight_slider.setValue(int(nn_weight * 100))
            self.weight_slider.blockSignals(False)
        
        # Emit weight changed signal for other components
        self.weight_changed.emit(nn_weight)
        
        logger.debug(f"NN/LLM weight updated to {nn_weight:.2f}")
    
    def on_weight_changed(self, value):
        """Override the base weight change handler to handle manual changes"""
        # Call parent implementation
        super().on_weight_changed(value)
        
        # If this is a manual change, disable auto-adjust
        if self.auto_adjust_weights:
            # This was probably triggered by our auto-adjustment, not a manual change
            pass
        else:
            # Store as last manual setting
            self.last_manual_weight = value / 100.0
    
    def closeEvent(self, event):
        """Handle panel close event"""
        # Stop the breath detector if we created it
        if self.breath_detector and not hasattr(self, '_breath_detector_external'):
            self.breath_detector.stop()
        
        # Call parent implementation if available
        if hasattr(super(), 'closeEvent'):
            super().closeEvent(event)


# For testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if not V5_CONVERSATION_AVAILABLE:
        logger.error("Cannot run test - V5 ConversationPanel not available")
        sys.exit(1)
    
    # Create Qt application
    app = QtWidgets.QApplication(sys.argv)
    
    # Create breath detector
    detector = BreathDetector()
    
    # Create conversation panel
    panel = BreathEnhancedConversationPanel(breath_detector=detector)
    panel.setWindowTitle("Breath-Enhanced Conversation")
    panel.resize(800, 600)
    panel.show()
    
    # Start breath detector
    detector.start()
    
    # Run the application
    sys.exit(app.exec()) 