#!/usr/bin/env python3
"""
Standalone Mistral PySide6 Chat Application

This application provides a simple chat interface for the Mistral AI LLM.
"""

import os
import sys
import json
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List

# Check if running in virtual environment
venv_path = os.environ.get('VIRTUAL_ENV')
if venv_path:
    site_packages = os.path.join(venv_path, 'Lib', 'site-packages')
    if site_packages not in sys.path:
        sys.path.append(site_packages)
        print(f"Added {site_packages} to Python path")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MistralApp")

try:
    # Import PySide6
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTextEdit, QLineEdit, QPushButton, QLabel, QSlider, QSplitter,
        QListWidget, QListWidgetItem, QStatusBar, QToolBar, QDialog,
        QDialogButtonBox, QFormLayout, QMessageBox
    )
    from PySide6.QtCore import Qt, QTimer, Signal, Slot, QSize, QObject
    from PySide6.QtGui import QIcon, QTextCursor, QColor, QPalette, QFont, QAction
    logger.info("PySide6 imported successfully")
except ImportError as e:
    logger.error(f"PySide6 import error: {e}")
    logger.error("PySide6 not found. Install with: pip install pyside6")
    sys.exit(1)

try:
    # Import Mistral client
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    MISTRAL_AVAILABLE = True
    logger.info("Mistral AI client imported successfully")
except ImportError as e:
    logger.error(f"Mistral import error: {e}")
    logger.error("Mistral AI client not found. Install with: pip install mistralai")
    MISTRAL_AVAILABLE = False
    sys.exit(1)

class MistralIntegration:
    """
    Mistral AI Integration for chat application
    
    This class provides a simple interface to the Mistral API.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistral-large-latest",
        llm_weight: float = 0.7,
        nn_weight: float = 0.3
    ):
        """
        Initialize the Mistral Integration
        
        Args:
            api_key: Mistral API key (uses MISTRAL_API_KEY env var if not provided)
            model: Mistral model to use
            llm_weight: Weight to give to LLM responses (0-1)
            nn_weight: Weight to give to neural network processing (0-1)
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.model = model
        self.llm_weight = max(0.0, min(1.0, llm_weight))
        self.nn_weight = max(0.0, min(1.0, nn_weight))
        
        # Initialize Mistral client if available
        self.client = None
        if MISTRAL_AVAILABLE and self.api_key:
            try:
                self.client = MistralClient(api_key=self.api_key)
                logger.info(f"Mistral AI client initialized with model {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize Mistral client: {e}")
        elif not self.api_key:
            logger.warning("Mistral API key not provided")
    
    @property
    def is_available(self) -> bool:
        """Check if Mistral integration is available and initialized"""
        return self.client is not None
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process a message using Mistral
        
        Args:
            message: User message to process
            
        Returns:
            Dict with processed results
        """
        result = {
            "input": message,
            "response": None,
            "error": None
        }
        
        # Get response if available
        if self.is_available:
            try:
                chat_response = self.client.chat(
                    model=self.model,
                    messages=[ChatMessage(role="user", content=message)]
                )
                result["response"] = chat_response.choices[0].message.content
                logger.info("Retrieved response from Mistral API")
            except Exception as e:
                logger.error(f"Error getting Mistral response: {e}")
                result["error"] = f"LLM error: {str(e)}"
        
        return result
    
    def adjust_weights(self, llm_weight: Optional[float] = None, nn_weight: Optional[float] = None) -> Dict[str, float]:
        """
        Adjust the weights of LLM components
        
        Args:
            llm_weight: New weight for LLM (0-1)
            nn_weight: New weight for neural networks (0-1)
            
        Returns:
            Dict with updated weights
        """
        # Update weights if provided
        if llm_weight is not None:
            self.llm_weight = max(0.0, min(1.0, llm_weight))
        
        if nn_weight is not None:
            self.nn_weight = max(0.0, min(1.0, nn_weight))
        
        return {
            "llm_weight": self.llm_weight,
            "nn_weight": self.nn_weight
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics and information
        
        Returns:
            Dict with system information
        """
        return {
            "mistral_available": self.is_available,
            "model": self.model if self.is_available else None,
            "weights": {
                "llm": self.llm_weight,
                "neural_network": self.nn_weight
            }
        }

class MistralSignals(QObject):
    """Signal wrapper for thread-safe communication with the UI"""
    message_received = Signal(dict)
    status_update = Signal(dict)
    response_received = Signal(str, dict)
    error_occurred = Signal(str)
    weights_updated = Signal(float, float)
    processing_started = Signal()
    processing_finished = Signal()

class ApiKeyDialog(QDialog):
    """Dialog for entering the Mistral API key"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mistral API Key")
        self.resize(400, 100)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create form layout
        form_layout = QFormLayout()
        
        # Create input field
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Enter your Mistral API key")
        self.api_key_input.setText(os.environ.get("MISTRAL_API_KEY", ""))
        form_layout.addRow("API Key:", self.api_key_input)
        
        layout.addLayout(form_layout)
        
        # Create buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def get_api_key(self):
        """Get the entered API key"""
        return self.api_key_input.text()

class ChatMessageWidget(QWidget):
    """Widget for displaying a chat message"""
    
    def __init__(self, role, content, parent=None):
        super().__init__(parent)
        
        # Set properties
        self.role = role
        self.content = content
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Create role label
        role_label = QLabel(role.capitalize())
        role_font = QFont()
        role_font.setBold(True)
        role_label.setFont(role_font)
        
        # Set role label color
        if role == "user":
            role_label.setStyleSheet("color: #0066cc;")
        else:
            role_label.setStyleSheet("color: #006633;")
        
        layout.addWidget(role_label)
        
        # Create content text edit
        content_text = QTextEdit()
        content_text.setReadOnly(True)
        content_text.setPlainText(content)
        content_text.setMinimumHeight(50)
        content_text.setMaximumHeight(200)
        layout.addWidget(content_text)
        
        # Set background color based on role
        if role == "user":
            self.setStyleSheet("background-color: #f0f8ff; border-radius: 5px;")
        else:
            self.setStyleSheet("background-color: #f0fff0; border-radius: 5px;")

class MistralChatWindow(QMainWindow):
    """
    Main window for the Mistral PySide6 frontend application
    
    This class provides a GUI for interacting with the Mistral LLM system.
    """
    
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("Mistral AI Chat")
        self.resize(800, 600)
        
        # Get API key
        api_key_dialog = ApiKeyDialog(self)
        if api_key_dialog.exec() == QDialog.Accepted:
            self.api_key = api_key_dialog.get_api_key()
        else:
            self.api_key = ""
        
        # Initialize Mistral integration
        self.integration = MistralIntegration(api_key=self.api_key)
        self.signals = MistralSignals()
        
        # Connect signals
        self.signals.response_received.connect(self._on_response_received)
        self.signals.error_occurred.connect(self._on_error_occurred)
        self.signals.status_update.connect(self._on_status_update)
        self.signals.weights_updated.connect(self._on_weights_updated)
        self.signals.processing_started.connect(self._on_processing_started)
        self.signals.processing_finished.connect(self._on_processing_finished)
        
        # Create UI
        self._create_ui()
        
        # Track conversation and processing state
        self.conversation_history = []
        self.is_processing = False
        
        # Start status timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._check_status)
        self.status_timer.start(3000)  # Update every 3 seconds
        
        logger.info("Mistral Chat Window initialized")
    
    def _create_ui(self):
        """Create the UI components"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Create chat area
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        
        # Create chat history
        self.chat_history = QWidget()
        self.chat_history_layout = QVBoxLayout(self.chat_history)
        self.chat_history_layout.setAlignment(Qt.AlignTop)
        self.chat_history_layout.setSpacing(10)
        
        # Create scroll area for chat history
        chat_scroll = QTextEdit()
        chat_scroll.setReadOnly(True)
        chat_scroll.setAcceptRichText(True)
        chat_scroll.setStyleSheet("background-color: #ffffff;")
        self.chat_scroll = chat_scroll
        chat_layout.addWidget(chat_scroll)
        
        # Create input area
        input_widget = QWidget()
        input_layout = QHBoxLayout(input_widget)
        
        # Create input field
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Type your message here...")
        self.message_input.returnPressed.connect(self._send_message)
        input_layout.addWidget(self.message_input)
        
        # Create send button
        send_button = QPushButton("Send")
        send_button.clicked.connect(self._send_message)
        input_layout.addWidget(send_button)
        
        chat_layout.addWidget(input_widget)
        
        # Create settings panel
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setAlignment(Qt.AlignTop)
        
        # Create settings label
        settings_label = QLabel("Settings")
        settings_font = QFont()
        settings_font.setBold(True)
        settings_font.setPointSize(12)
        settings_label.setFont(settings_font)
        settings_layout.addWidget(settings_label)
        
        # Create LLM weight slider
        llm_weight_label = QLabel(f"LLM Weight: {self.integration.llm_weight:.1f}")
        settings_layout.addWidget(llm_weight_label)
        
        llm_weight_slider = QSlider(Qt.Horizontal)
        llm_weight_slider.setMinimum(0)
        llm_weight_slider.setMaximum(100)
        llm_weight_slider.setValue(int(self.integration.llm_weight * 100))
        llm_weight_slider.valueChanged.connect(
            lambda value: self._update_weight_label(llm_weight_label, "LLM Weight", value)
        )
        settings_layout.addWidget(llm_weight_slider)
        self.llm_weight_slider = llm_weight_slider
        self.llm_weight_label = llm_weight_label
        
        # Create NN weight slider
        nn_weight_label = QLabel(f"NN Weight: {self.integration.nn_weight:.1f}")
        settings_layout.addWidget(nn_weight_label)
        
        nn_weight_slider = QSlider(Qt.Horizontal)
        nn_weight_slider.setMinimum(0)
        nn_weight_slider.setMaximum(100)
        nn_weight_slider.setValue(int(self.integration.nn_weight * 100))
        nn_weight_slider.valueChanged.connect(
            lambda value: self._update_weight_label(nn_weight_label, "NN Weight", value)
        )
        settings_layout.addWidget(nn_weight_slider)
        self.nn_weight_slider = nn_weight_slider
        self.nn_weight_label = nn_weight_label
        
        # Create apply weights button
        apply_weights_button = QPushButton("Apply Weights")
        apply_weights_button.clicked.connect(self._apply_weights)
        settings_layout.addWidget(apply_weights_button)
        
        # Create clear button
        clear_button = QPushButton("Clear Chat")
        clear_button.clicked.connect(self._clear_chat)
        settings_layout.addWidget(clear_button)
        
        # Add status label
        self.status_label = QLabel("Status: Initializing...")
        settings_layout.addWidget(self.status_label)
        
        # Add spacer
        settings_layout.addStretch()
        
        # Add widgets to splitter
        splitter.addWidget(chat_widget)
        splitter.addWidget(settings_widget)
        
        # Set splitter sizes (70% chat, 30% settings)
        splitter.setSizes([int(self.width() * 0.7), int(self.width() * 0.3)])
        
        # Create status bar
        self.statusBar().showMessage("Ready")
    
    def _update_weight_label(self, label, name, value):
        """Update weight label with slider value"""
        weight = value / 100.0
        label.setText(f"{name}: {weight:.1f}")
    
    def _apply_weights(self):
        """Apply weight changes from sliders"""
        llm_weight = self.llm_weight_slider.value() / 100.0
        nn_weight = self.nn_weight_slider.value() / 100.0
        
        # Update weights
        self.integration.adjust_weights(llm_weight=llm_weight, nn_weight=nn_weight)
        
        # Update UI
        self.statusBar().showMessage(f"Weights updated: LLM={llm_weight:.1f}, NN={nn_weight:.1f}")
        
        # Emit signal
        self.signals.weights_updated.emit(llm_weight, nn_weight)
    
    def _send_message(self):
        """Send a message to Mistral"""
        # Get message from input field
        message = self.message_input.text().strip()
        
        # Check if message is empty
        if not message:
            return
        
        # Check if processing is in progress
        if self.is_processing:
            QMessageBox.warning(self, "Processing", "Already processing a message. Please wait.")
            return
        
        # Clear input field
        self.message_input.clear()
        
        # Add message to chat
        self._add_chat_message("user", message)
        
        # Process message in a separate thread
        self._process_message_async(message)
    
    def _process_message_async(self, message):
        """Process a message asynchronously"""
        def process_thread():
            # Set processing flag
            self.is_processing = True
            
            # Emit signal
            self.signals.processing_started.emit()
            
            try:
                # Add to conversation history
                self.conversation_history.append({
                    "role": "user",
                    "content": message,
                    "timestamp": int(time.time())
                })
                
                # Process message
                result = self.integration.process_message(message)
                
                # Check for error
                if result.get("error"):
                    self.signals.error_occurred.emit(result["error"])
                    return
                
                # Get response
                response = result.get("response", "No response received")
                
                # Add to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": int(time.time())
                })
                
                # Emit signal
                self.signals.response_received.emit("", {"message": response})
            
            except Exception as e:
                # Emit error signal
                self.signals.error_occurred.emit(str(e))
            
            finally:
                # Clear processing flag
                self.is_processing = False
                
                # Emit signal
                self.signals.processing_finished.emit()
        
        # Start thread
        thread = threading.Thread(target=process_thread)
        thread.daemon = True
        thread.start()
    
    def _add_chat_message(self, role, content):
        """Add a message to the chat history"""
        # Create HTML for the message
        if role == "user":
            html = f'<div style="background-color: #f0f8ff; border-radius: 5px; padding: 10px; margin: 5px;">'
            html += f'<div style="font-weight: bold; color: #0066cc;">You</div>'
            html += f'<div>{content}</div>'
            html += f'</div>'
        else:
            html = f'<div style="background-color: #f0fff0; border-radius: 5px; padding: 10px; margin: 5px;">'
            html += f'<div style="font-weight: bold; color: #006633;">Assistant</div>'
            html += f'<div>{content}</div>'
            html += f'</div>'
        
        # Add to chat scroll
        cursor = self.chat_scroll.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(html)
        cursor.insertBlock()
        
        # Scroll to bottom
        self.chat_scroll.setTextCursor(cursor)
        self.chat_scroll.ensureCursorVisible()
    
    def _clear_chat(self):
        """Clear the chat history"""
        self.chat_scroll.clear()
        self.conversation_history = []
    
    @Slot(str, dict)
    def _on_response_received(self, request_id, response_data):
        """Handle response from Mistral"""
        message = response_data.get("message", "")
        
        # Add message to chat
        self._add_chat_message("assistant", message)
        
        # Update status
        self.statusBar().showMessage("Response received")
    
    @Slot(str)
    def _on_error_occurred(self, error_msg):
        """Handle error from Mistral"""
        # Display error in chat
        error_html = f'<div style="background-color: #fff0f0; border-radius: 5px; padding: 10px; margin: 5px;">'
        error_html += f'<div style="font-weight: bold; color: #cc0000;">Error</div>'
        error_html += f'<div>{error_msg}</div>'
        error_html += f'</div>'
        
        # Add to chat scroll
        cursor = self.chat_scroll.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(error_html)
        cursor.insertBlock()
        
        # Update status
        self.statusBar().showMessage(f"Error: {error_msg}")
    
    @Slot(dict)
    def _on_status_update(self, status_data):
        """Handle status update from Mistral"""
        # Update status label
        if self.integration.is_available:
            self.status_label.setText(f"Status: Connected (Model: {self.integration.model})")
        else:
            self.status_label.setText("Status: Not connected")
    
    @Slot(float, float)
    def _on_weights_updated(self, llm_weight, nn_weight):
        """Handle weights update from Mistral"""
        # Update slider values
        self.llm_weight_slider.setValue(int(llm_weight * 100))
        self.nn_weight_slider.setValue(int(nn_weight * 100))
        
        # Update labels
        self.llm_weight_label.setText(f"LLM Weight: {llm_weight:.1f}")
        self.nn_weight_label.setText(f"NN Weight: {nn_weight:.1f}")
        
        # Update status
        self.statusBar().showMessage(f"Weights updated: LLM={llm_weight:.1f}, NN={nn_weight:.1f}")
    
    @Slot()
    def _on_processing_started(self):
        """Handle processing started signal"""
        self.statusBar().showMessage("Processing message...")
        self.message_input.setEnabled(False)
    
    @Slot()
    def _on_processing_finished(self):
        """Handle processing finished signal"""
        self.statusBar().showMessage("Ready")
        self.message_input.setEnabled(True)
    
    def _check_status(self):
        """Check system status periodically"""
        if not self.integration:
            return
        
        # Get system stats
        stats = self.integration.get_system_stats()
        
        # Update status label
        if self.integration.is_available:
            self.status_label.setText(f"Status: Connected (Model: {stats['model']})")
        else:
            self.status_label.setText("Status: Not connected")
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop timer
        if hasattr(self, 'status_timer'):
            self.status_timer.stop()
        
        # Accept event
        event.accept()

def main():
    """Main application entry point"""
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create main window
    window = MistralChatWindow()
    window.show()
    
    # Run application
    return app.exec()

if __name__ == "__main__":
    sys.exit(main()) 