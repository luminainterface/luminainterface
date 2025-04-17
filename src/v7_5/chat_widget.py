from datetime import datetime
import asyncio
import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
    QLineEdit, QPushButton, QLabel
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QTextCursor, QFont

from .signal_component import SignalComponent
from .signal_system import SignalBus

# Configure logging
logger = logging.getLogger(__name__)

class ChatWidget(QWidget):
    # Signals for Qt
    message_sent = Signal(str)
    settings_updated = Signal(dict)
    
    def __init__(self, signal_bus: SignalBus, parent=None):
        logger.debug("Initializing ChatWidget")
        QWidget.__init__(self, parent)
        self.signal_bus = signal_bus
        self.signal_component = SignalComponent("chat_widget", signal_bus)
        
        self.setup_ui()
        self.setup_signals()
        self.model_status = "mistral"  # Default model
        self.temperature = 0.7  # Default temperature
        self.top_p = 1.0  # Default top_p
        self.settings_window = None
        self.is_processing = False
        self.update_status_display()
        logger.debug("ChatWidget initialized")
        
    def setup_ui(self):
        """Set up the chat widget UI"""
        logger.debug("Setting up ChatWidget UI")
        layout = QVBoxLayout()
        
        # Status bar
        status_layout = QHBoxLayout()
        self.model_status_label = QLabel("Initializing...")
        status_layout.addWidget(self.model_status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)
        
        # Chat history
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setFont(QFont("Consolas", 10))
        layout.addWidget(self.chat_history)
        
        # Input area
        input_layout = QHBoxLayout()
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Type your message...")
        self.send_button = QPushButton("Send")
        
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.send_button)
        layout.addLayout(input_layout)
        
        self.setLayout(layout)
        logger.debug("ChatWidget UI setup complete")
        
    def setup_signals(self):
        """Set up signal connections"""
        logger.debug("Setting up ChatWidget signals")
        self.send_button.clicked.connect(self.send_message)
        self.message_input.returnPressed.connect(self.send_message)
        
        # Register signal handlers
        self.signal_component.register_handler("chat_message", self.handle_chat_message)
        self.signal_component.register_handler("system_message", self.handle_system_message)
        logger.debug("ChatWidget signals setup complete")
        
    def send_message(self):
        """Send a message and emit signal"""
        message = self.message_input.text().strip()
        if message:
            logger.debug(f"Sending message: {message}")
            # Clear input field
            self.message_input.clear()
            # Emit the signal
            self.message_sent.emit(message)
            logger.debug("Message sent")
            
    def handle_chat_message(self, data: dict):
        """Handle incoming chat messages"""
        logger.debug(f"Handling chat message: {data}")
        if isinstance(data, dict):
            sender = data.get("sender", "unknown")
            message = data.get("message", "")
            timestamp = data.get("timestamp", "")
            
            self.append_message(sender, message, timestamp)
            logger.debug("Chat message handled and displayed")
            
    def handle_system_message(self, data: dict):
        """Handle system messages"""
        logger.debug(f"Handling system message: {data}")
        if isinstance(data, dict):
            message = data.get("message", "")
            self.append_system_message(message)
            logger.debug("System message handled and displayed")
            
    def append_message(self, sender: str, message: str, timestamp: str = None):
        """Append a message to the chat history"""
        try:
            logger.debug(f"Appending message from {sender}: {message}")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%H:%M:%S")
                except:
                    time_str = timestamp
            else:
                time_str = datetime.now().strftime("%H:%M:%S")
                
            if sender.lower() == "user":
                formatted_message = f'<div style="margin: 5px;"><b>[{time_str}] You:</b> {message}</div>'
            elif sender.lower() == "assistant":
                formatted_message = f'<div style="margin: 5px;"><b>[{time_str}] Assistant:</b> {message}</div>'
            elif sender.lower() == "system":
                formatted_message = f'<div style="margin: 5px; color: #666;"><i>[{time_str}] System:</i> {message}</div>'
            else:
                formatted_message = f'<div style="margin: 5px;"><b>[{time_str}] {sender}:</b> {message}</div>'
                
            self.chat_history.append(formatted_message)
            
            # Scroll to bottom
            cursor = self.chat_history.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.chat_history.setTextCursor(cursor)
            logger.debug("Message appended and displayed")
            
        except Exception as e:
            logger.error(f"Error appending message: {e}")
            
    def append_system_message(self, message: str):
        """Append a system message to the chat history"""
        logger.debug(f"Appending system message: {message}")
        self.append_message("System", message)
        
    def closeEvent(self, event):
        """Handle widget close event"""
        logger.debug("ChatWidget closing")
        self.cleanup()
        event.accept()
        
    def update_model_status(self, model_name: str):
        """Update the displayed model status"""
        logger.debug(f"Updating model status: {model_name}")
        self.model_status = model_name
        self.update_status_display()
        
    def update_status_display(self):
        """Update the status display with current settings"""
        status_text = (
            f"{self.model_status} "
            f"(temp={self.temperature:.1f}, "
            f"top_p={self.top_p:.1f})"
        )
        if self.is_processing:
            status_text += " [Processing...]"
        self.model_status_label.setText(status_text)
        logger.debug(f"Status display updated: {status_text}")
        
    def apply_settings(self, settings: dict):
        """Apply new settings and update the display"""
        logger.debug(f"Applying settings: {settings}")
        settings_type = settings.get('type')
        
        if settings_type == 'model':
            self.model_status = settings.get('model', self.model_status)
            self.temperature = settings.get('temperature', self.temperature)
            self.top_p = settings.get('top_p', self.top_p)
            self.update_status_display()
            
            # Notify parent and signal bus of settings update
            self.settings_updated.emit(settings)
            asyncio.create_task(self.signal_component.emit_signal("chat.settings_updated", settings))
            
            # Add system message about settings change
            settings_msg = (
                f"Model settings updated:\n"
                f"Model: {self.model_status}\n"
                f"Temperature: {self.temperature}\n"
                f"Top-P: {self.top_p}"
            )
            self.append_system_message(settings_msg)
            logger.debug("Settings applied and displayed")
        
    def cleanup(self):
        """Clean up resources when component is destroyed"""
        logger.debug("Cleaning up ChatWidget resources")
        if self.settings_window:
            self.settings_window.close()
            self.settings_window = None
        self.signal_component.cleanup()
        logger.debug("ChatWidget cleanup complete") 