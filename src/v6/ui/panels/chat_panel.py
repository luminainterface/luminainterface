"""
V6 Chat Panel

A proper chat panel for the V6 Portal of Contradiction with enhanced
sizing, holographic appearance, and glowing UI elements.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    # Import Qt compatibility layer from V5
    from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui, Qt, Signal, Slot
    from src.v5.ui.qt_compat import get_widgets, get_gui, get_core
except ImportError:
    logging.warning("V5 Qt compatibility layer not found. Using direct PySide6 imports.")
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        from PySide6.QtCore import Qt, Signal, Slot
        
        # Simple compatibility functions
        def get_widgets():
            return QtWidgets
            
        def get_gui():
            return QtGui
            
        def get_core():
            return QtCore
    except ImportError:
        logging.error("PySide6 not found. Please install PySide6 or configure the V5 Qt compatibility layer.")
        sys.exit(1)

# Import V6 panel base
from src.v6.ui.panel_base import V6PanelBase

# Get required Qt classes
try:
    QGraphicsDropShadowEffect = get_gui().QGraphicsDropShadowEffect
except AttributeError:
    QGraphicsDropShadowEffect = get_widgets().QGraphicsDropShadowEffect
    
QColor = get_gui().QColor
QPainter = get_gui().QPainter
QBrush = get_gui().QBrush
QPen = get_gui().QPen
QLinearGradient = get_gui().QLinearGradient
QRadialGradient = get_gui().QRadialGradient

# Set up logging
logger = logging.getLogger(__name__)

class HolographicBubble(QtWidgets.QWidget):
    """
    A holographic message bubble with glowing effects
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
        # For pulsing glow effect
        self.glow_opacity = 0.7
        self.glow_direction = 1  # 1 for increasing, -1 for decreasing
        self.glow_timer = QtCore.QTimer(self)
        self.glow_timer.timeout.connect(self.updateGlow)
        self.glow_timer.start(50)  # 50ms update
    
    def initUI(self):
        """Initialize UI settings"""
        self.setMinimumWidth(100)
        self.setAutoFillBackground(False)
        
        # Add glow effect
        self.glow_effect = QGraphicsDropShadowEffect()
        self.glow_effect.setBlurRadius(15)
        self.glow_effect.setOffset(0, 0)
        self.setGraphicsEffect(self.glow_effect)
    
    def updateGlow(self):
        """Update the glow animation"""
        # Update glow opacity for pulsing effect
        self.glow_opacity += 0.02 * self.glow_direction
        if self.glow_opacity >= 1.0:
            self.glow_opacity = 1.0
            self.glow_direction = -1
        elif self.glow_opacity <= 0.6:
            self.glow_opacity = 0.6
            self.glow_direction = 1
        
        # Update the glow effect
        self.update()
    
    def paintEvent(self, event):
        """Custom paint event for holographic bubble"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Update glow effect color
        color = QColor(52, 152, 219, int(130 * self.glow_opacity))
        self.glow_effect.setColor(color)
        
        # Draw background
        gradient = QLinearGradient(0, 0, 0, self.height())
        
        # User message vs system message
        if hasattr(self, 'is_user') and self.is_user:
            gradient.setColorAt(0, QColor(41, 128, 185, 180))  # Blue
            gradient.setColorAt(1, QColor(52, 152, 219, 150))
            border_color = QColor(52, 152, 219, int(200 * self.glow_opacity))
        else:
            gradient.setColorAt(0, QColor(44, 62, 80, 180))  # Dark
            gradient.setColorAt(1, QColor(52, 73, 94, 150))
            border_color = QColor(155, 89, 182, int(180 * self.glow_opacity))
        
        # Fill the background
        painter.setBrush(QBrush(gradient))
        
        # Draw border with glow
        pen = QPen(border_color)
        pen.setWidth(1)
        painter.setPen(pen)
        
        # Draw the rounded rectangle
        painter.drawRoundedRect(0, 0, self.width() - 1, self.height() - 1, 8, 8)
        
        # Let child widgets paint
        super().paintEvent(event)

class ChatMessageWidget(QtWidgets.QWidget):
    """
    Widget to display a single chat message with holographic styling
    """
    
    def __init__(self, message, is_user=False, timestamp=None, parent=None):
        """Initialize a chat message widget"""
        super().__init__(parent)
        self.message = message
        self.is_user = is_user
        self.timestamp = timestamp or time.strftime("%H:%M:%S")
        self.initUI()
    
    def initUI(self):
        """Initialize the user interface"""
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)  # Further reduced vertical spacing
        
        # Create the message bubble
        bubble = HolographicBubble()
        bubble.is_user = self.is_user  # Pass user status to bubble for styling
        
        bubble_layout = QtWidgets.QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(12, 8, 12, 8)  # Compact padding
        
        # Header with timestamp
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 3)  # Reduced spacing
        
        # Create sender label
        sender_label = QtWidgets.QLabel(self.is_user and "You" or "System")
        sender_label.setStyleSheet(f"""
            color: {self.is_user and '#ECF0F1' or '#E8DAEF'};
            font-weight: bold;
            font-size: 13px;
        """)
        
        # Create timestamp label
        time_label = QtWidgets.QLabel(self.timestamp)
        time_label.setStyleSheet("""
            color: rgba(236, 240, 241, 150);
            font-size: 11px;
        """)
        
        header_layout.addWidget(sender_label)
        header_layout.addStretch()
        header_layout.addWidget(time_label)
        
        # Create message label
        message_label = QtWidgets.QLabel(self.message)
        message_label.setWordWrap(True)
        message_label.setStyleSheet("""
            color: #ECF0F1;
            font-size: 13px;
        """)
        
        # Add to bubble layout
        bubble_layout.addLayout(header_layout)
        bubble_layout.addWidget(message_label)
        
        # Add bubble to main layout, aligned to left or right
        if self.is_user:
            layout.addStretch()
            layout.addWidget(bubble, 0, Qt.AlignRight)
        else:
            layout.addWidget(bubble, 0, Qt.AlignLeft)
            layout.addStretch()

class HolographicScrollArea(QtWidgets.QScrollArea):
    """
    A custom scroll area with holographic appearance
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background: rgba(26, 38, 52, 80);
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: rgba(52, 152, 219, 150);
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)

class HolographicTextEdit(QtWidgets.QTextEdit):
    """
    A custom text edit with holographic appearance
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QTextEdit {
                background-color: rgba(26, 38, 52, 120);
                color: #ECF0F1;
                font-size: 13px;
                border: 1px solid rgba(52, 152, 219, 100);
                border-radius: 4px;
                padding: 6px;
            }
        """)
        
        # Add glow effect
        self.glow_effect = QGraphicsDropShadowEffect()
        self.glow_effect.setBlurRadius(10)
        self.glow_effect.setColor(QColor(52, 152, 219, 100))
        self.glow_effect.setOffset(0, 0)
        self.setGraphicsEffect(self.glow_effect)
    
    def focusInEvent(self, event):
        """Change glow when focused"""
        self.glow_effect.setColor(QColor(52, 152, 219, 180))
        self.glow_effect.setBlurRadius(15)
        super().focusInEvent(event)
    
    def focusOutEvent(self, event):
        """Change glow when focus lost"""
        self.glow_effect.setColor(QColor(52, 152, 219, 100))
        self.glow_effect.setBlurRadius(10)
        super().focusOutEvent(event)

class ChatPanel(V6PanelBase):
    """
    A holographic chat panel with message history and input field for the V6 Portal
    """
    
    # Signal emitted when a message is sent
    message_sent = Signal(str)
    
    def __init__(self, socket_manager=None, parent=None):
        """Initialize the chat panel"""
        super().__init__(parent)
        self.socket_manager = socket_manager
        self.message_history = []
        self.initUI()
        
        # Initialize animation for typing indicator
        self.typing_dots = 0
        self.typing_timer = QtCore.QTimer(self)
        self.typing_timer.timeout.connect(self.updateTypingIndicator)
        self.is_typing = False
    
    def initUI(self):
        """Initialize the user interface"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Create a holographic frame for the header
        header_frame = QtWidgets.QFrame()
        header_frame.setFixedHeight(36)
        header_frame.setStyleSheet("""
            background-color: rgba(26, 38, 52, 120);
            border-radius: 4px;
            border: 1px solid rgba(52, 152, 219, 80);
        """)
        
        # Title and info area
        title_layout = QtWidgets.QHBoxLayout(header_frame)
        title_layout.setContentsMargins(10, 0, 10, 0)
        
        # Add a small glowing icon
        icon_label = QtWidgets.QLabel("≡")
        icon_label.setStyleSheet("""
            color: #3498DB;
            font-size: 16px;
            font-weight: bold;
        """)
        
        # Add glow to icon
        icon_glow = QGraphicsDropShadowEffect()
        icon_glow.setBlurRadius(12)
        icon_glow.setColor(QColor(52, 152, 219, 180))
        icon_glow.setOffset(0, 0)
        icon_label.setGraphicsEffect(icon_glow)
        
        title_label = QtWidgets.QLabel("Portal Conversation")
        title_label.setStyleSheet("""
            color: #3498DB;
            font-size: 14px;
            font-weight: bold;
        """)
        
        # Add glow to title
        title_glow = QGraphicsDropShadowEffect()
        title_glow.setBlurRadius(10)
        title_glow.setColor(QColor(52, 152, 219, 150))
        title_glow.setOffset(0, 0)
        title_label.setGraphicsEffect(title_glow)
        
        self.status_label = QtWidgets.QLabel("Connected")
        self.status_label.setStyleSheet("""
            color: #2ECC71;
            font-size: 12px;
        """)
        
        title_layout.addWidget(icon_label)
        title_layout.addSpacing(5)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(self.status_label)
        
        # Create a holographic scroll area for messages
        self.message_area = HolographicScrollArea()
        
        # Create widget to hold messages
        self.message_container = QtWidgets.QWidget()
        self.message_container.setStyleSheet("background: transparent;")
        self.message_layout = QtWidgets.QVBoxLayout(self.message_container)
        self.message_layout.setContentsMargins(5, 5, 5, 5)
        self.message_layout.setSpacing(6)
        self.message_layout.addStretch()
        
        self.message_area.setWidget(self.message_container)
        
        # Create typing indicator
        self.typing_indicator = QtWidgets.QLabel("")
        self.typing_indicator.setStyleSheet("""
            color: rgba(127, 140, 141, 180);
            font-size: 12px;
            font-style: italic;
            padding-left: 8px;
        """)
        self.typing_indicator.setVisible(False)
        
        # Create input area with holographic appearance
        input_frame = QtWidgets.QFrame()
        input_frame.setStyleSheet("""
            background-color: rgba(26, 38, 52, 80);
            border-radius: 4px;
            border: 1px solid rgba(52, 152, 219, 60);
        """)
        input_frame.setMinimumHeight(50)
        input_frame.setMaximumHeight(50)
        
        input_layout = QtWidgets.QHBoxLayout(input_frame)
        input_layout.setContentsMargins(8, 4, 8, 4)
        input_layout.setSpacing(8)
        
        # Create holographic text edit for input
        self.input_edit = HolographicTextEdit()
        self.input_edit.setPlaceholderText("Type your message here...")
        self.input_edit.setFixedHeight(35)
        
        # Connect key press event for Enter key
        self.input_edit.installEventFilter(self)
        
        # Create send button with glow effect
        self.send_button = QtWidgets.QPushButton("Send")
        self.send_button.setMinimumSize(70, 35)
        self.send_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(41, 128, 185, 180);
                color: white;
                border-radius: 4px;
                border: 1px solid rgba(52, 152, 219, 100);
                padding: 6px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: rgba(52, 152, 219, 200);
                border: 1px solid rgba(52, 152, 219, 150);
            }
        """)
        
        # Add glow effect to button
        button_glow = QGraphicsDropShadowEffect()
        button_glow.setBlurRadius(10)
        button_glow.setColor(QColor(52, 152, 219, 100))
        button_glow.setOffset(0, 0)
        self.send_button.setGraphicsEffect(button_glow)
        
        self.send_button.clicked.connect(self.sendMessage)
        
        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(self.send_button)
        
        # Add all elements to main layout
        layout.addWidget(header_frame)
        layout.addWidget(self.message_area, 1)
        layout.addWidget(self.typing_indicator)
        layout.addWidget(input_frame)
        
        # Add some initial messages
        self.addSystemMessage("Welcome to the Portal of Contradiction. How may I assist you today?")
    
    def eventFilter(self, obj, event):
        """Filter events to handle Enter key in the text edit"""
        if obj is self.input_edit and event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Return and not event.modifiers() & QtCore.Qt.ShiftModifier:
                self.sendMessage()
                return True
        return super().eventFilter(obj, event)
    
    def sendMessage(self):
        """Send the message from the input field"""
        message = self.input_edit.toPlainText().strip()
        if message:
            self.input_edit.clear()
            self.addUserMessage(message)
            self.message_sent.emit(message)
            
            # Show typing indicator for demo purposes
            self.showTypingIndicator(True)
            # In a real implementation, this would be triggered by the socket manager
            QtCore.QTimer.singleShot(2000, lambda: self.simulateResponse())
    
    def simulateResponse(self):
        """Simulate a response for demonstration purposes"""
        self.showTypingIndicator(False)
        
        # Demo responses
        responses = [
            "I understand your perspective. Let me analyze the contradictions within that concept.",
            "Your statement contains interesting paradoxical elements. The duality processor is integrating these patterns.",
            "The Memory Reflection System is contextualizing your input against previous interactions.",
            "Multiple consciousness threads are processing your request simultaneously.",
            "I've detected a pattern that connects to earlier symbols in our conversation.",
        ]
        
        import random
        response = random.choice(responses)
        self.addSystemMessage(response)
    
    def addUserMessage(self, message):
        """Add a user message to the chat"""
        self._addMessageWidget(message, True)
    
    def addSystemMessage(self, message):
        """Add a system message to the chat"""
        self._addMessageWidget(message, False)
    
    def _addMessageWidget(self, message, is_user):
        """Add a message widget to the message area"""
        self.message_history.append((message, is_user))
        message_widget = ChatMessageWidget(message, is_user)
        
        # Remove the stretch if it exists
        if self.message_layout.count() > 0:
            item = self.message_layout.itemAt(self.message_layout.count() - 1)
            if item.widget() is None:  # It's a stretch
                self.message_layout.removeItem(item)
        
        # Add the new message and a stretch
        self.message_layout.addWidget(message_widget)
        self.message_layout.addStretch()
        
        # Scroll to the bottom
        self._scrollToBottom()
    
    def _scrollToBottom(self):
        """Scroll the message area to the bottom"""
        QtCore.QTimer.singleShot(10, lambda: self.message_area.verticalScrollBar().setValue(
            self.message_area.verticalScrollBar().maximum()
        ))
    
    def clearChat(self):
        """Clear all messages from the chat"""
        self.message_history.clear()
        
        # Remove all message widgets
        while self.message_layout.count() > 0:
            item = self.message_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add stretch back
        self.message_layout.addStretch()
    
    def showTypingIndicator(self, visible):
        """Show or hide the typing indicator"""
        self.is_typing = visible
        if visible:
            self.typing_dots = 0
            self.typing_indicator.setText("System is typing")
            self.typing_indicator.setVisible(True)
            self.typing_timer.start(500)  # Update every 500ms
        else:
            self.typing_timer.stop()
            self.typing_indicator.setVisible(False)
    
    def updateTypingIndicator(self):
        """Update the typing indicator animation"""
        if not self.is_typing:
            return
            
        self.typing_dots = (self.typing_dots + 1) % 4
        dots = "." * self.typing_dots
        self.typing_indicator.setText(f"System is typing{dots}")
    
    def cleanup(self):
        """Clean up resources before destruction"""
        if hasattr(self, 'typing_timer') and self.typing_timer:
            self.typing_timer.stop()
            
        if self.socket_manager:
            # Disconnect from socket events if needed
            pass 