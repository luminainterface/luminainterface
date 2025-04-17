"""
LLM Panel for V5 Visualization System

This module provides a UI panel for interacting with LLMs through the LLM Bridge Plugin.
"""

import logging
import json
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Qt compatibility layer
from ..qt_compat import QtWidgets, QtCore, QtGui, Qt, Signal, Slot
from ..qt_compat import get_widgets, get_gui, get_core

# Get required Qt classes
QWidget = get_widgets().QWidget
QVBoxLayout = get_widgets().QVBoxLayout
QHBoxLayout = get_widgets().QHBoxLayout
QGridLayout = get_widgets().QGridLayout
QLabel = get_widgets().QLabel
QPushButton = get_widgets().QPushButton
QTextEdit = get_widgets().QTextEdit
QComboBox = get_widgets().QComboBox
QLineEdit = get_widgets().QLineEdit
QSpinBox = get_widgets().QSpinBox
QDoubleSpinBox = get_widgets().QDoubleSpinBox
QCheckBox = get_widgets().QCheckBox
QSplitter = get_widgets().QSplitter
QScrollArea = get_widgets().QScrollArea
QGroupBox = get_widgets().QGroupBox
QFormLayout = get_widgets().QFormLayout
QTabWidget = get_widgets().QTabWidget
QToolButton = get_widgets().QToolButton
QSizePolicy = get_widgets().QSizePolicy

# Import theme-related components
from ..themes import get_theme, apply_theme

# Primary color from fractal pattern panel
PRIMARY_COLOR = QtGui.QColor(100, 100, 150)
ACCENT_COLOR = QtGui.QColor(120, 160, 210)


class LLMPanel(QWidget):
    """
    UI panel for interacting with LLMs through the LLM Bridge Plugin.
    
    This panel provides a chat interface, configuration options, and
    visualizations of LLM responses and memory integration.
    """
    
    # Define signals
    llm_response_received = Signal(dict)
    config_updated = Signal(dict)
    
    def __init__(self, parent=None):
        """Initialize the LLM Panel"""
        super().__init__(parent)
        self.plugin_id = "llm_bridge"
        self.plugin = None
        self.plugin_connected = False
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []
        
        # Track requests in progress
        self.pending_requests = {}
        
        # Initialize UI
        self._init_ui()
        
        # Apply theme
        apply_theme(self, get_theme())
        
    def _init_ui(self):
        """Initialize the UI components"""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Vertical)
        
        # --- Chat section ---
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        
        # Message display area
        self.message_display = QTextEdit()
        self.message_display.setReadOnly(True)
        self.message_display.setMinimumHeight(200)
        chat_layout.addWidget(self.message_display)
        
        # Input area
        input_layout = QHBoxLayout()
        
        # System message button
        self.system_msg_btn = QToolButton()
        self.system_msg_btn.setText("🔧")
        self.system_msg_btn.setToolTip("Set System Message")
        self.system_msg_btn.clicked.connect(self._show_system_message_dialog)
        input_layout.addWidget(self.system_msg_btn)
        
        # Message input
        self.message_input = QTextEdit()
        self.message_input.setPlaceholderText("Type your message here...")
        self.message_input.setMaximumHeight(100)
        input_layout.addWidget(self.message_input)
        
        # Send button
        self.send_button = QPushButton("Send")
        self.send_button.setIcon(get_gui().QIcon.fromTheme("send"))
        self.send_button.clicked.connect(self._send_message)
        input_layout.addWidget(self.send_button)
        
        chat_layout.addLayout(input_layout)
        
        # --- Configuration section ---
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        
        # Configuration tabs
        config_tabs = QTabWidget()
        
        # LLM Config tab
        llm_config_widget = QWidget()
        llm_config_layout = QFormLayout(llm_config_widget)
        
        # Provider selection
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["openai", "anthropic"])
        llm_config_layout.addRow("LLM Provider:", self.provider_combo)
        
        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["gpt-3.5-turbo", "gpt-4", "claude-2", "claude-instant-1"])
        llm_config_layout.addRow("Model:", self.model_combo)
        
        # API Key
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Enter API key")
        llm_config_layout.addRow("API Key:", self.api_key_input)
        
        # Temperature
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(0.7)
        llm_config_layout.addRow("Temperature:", self.temperature_spin)
        
        # Max tokens
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(50, 4000)
        self.max_tokens_spin.setSingleStep(50)
        self.max_tokens_spin.setValue(1000)
        llm_config_layout.addRow("Max Tokens:", self.max_tokens_spin)
        
        # Update config button
        self.update_config_btn = QPushButton("Update Configuration")
        self.update_config_btn.clicked.connect(self._update_config)
        llm_config_layout.addRow("", self.update_config_btn)
        
        config_tabs.addTab(llm_config_widget, "LLM Settings")
        
        # Memory Config tab
        memory_config_widget = QWidget()
        memory_config_layout = QFormLayout(memory_config_widget)
        
        # Memory mode
        self.memory_mode_combo = QComboBox()
        self.memory_mode_combo.addItems(["none", "contextual", "synthesized", "combined"])
        memory_config_layout.addRow("Memory Mode:", self.memory_mode_combo)
        
        # Memory statistics display
        self.memory_stats_display = QTextEdit()
        self.memory_stats_display.setReadOnly(True)
        self.memory_stats_display.setMaximumHeight(100)
        memory_config_layout.addRow("Memory Stats:", self.memory_stats_display)
        
        # Refresh button
        self.refresh_stats_btn = QPushButton("Refresh Stats")
        self.refresh_stats_btn.clicked.connect(self._refresh_memory_stats)
        memory_config_layout.addRow("", self.refresh_stats_btn)
        
        config_tabs.addTab(memory_config_widget, "Memory Settings")
        
        # Add tabs to config layout
        config_layout.addWidget(config_tabs)
        
        # Status section
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Status: Not connected")
        status_layout.addWidget(self.status_label)
        
        # Connect button
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._connect_to_plugin)
        status_layout.addWidget(self.connect_btn)
        
        config_layout.addLayout(status_layout)
        
        # Add widgets to splitter
        splitter.addWidget(chat_widget)
        splitter.addWidget(config_widget)
        
        # Set splitter sizes
        splitter.setSizes([700, 300])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Set up system message
        self.system_message = "You are a helpful assistant integrated with the V5 visualization system."
        
    def connect_to_socket(self, socket, message_handler):
        """
        Connect to the LLM Bridge Plugin socket.
        
        Args:
            socket: The NodeSocket instance
            message_handler: Function to handle messages
        """
        # Store the socket reference
        self.socket = socket
        self.message_handler = message_handler
        
        # Update UI
        self.status_label.setText("Status: Connected to socket")
        
        # Request plugin configuration
        self._request_config()
        
    def set_request_handler(self, socket, request_handler):
        """
        Set the request handler for the plugin socket.
        
        Args:
            socket: The NodeSocket instance
            request_handler: Function to handle requests
        """
        self.socket = socket
        self.request_handler = request_handler
        
        # Enable sending requests
        self.send_button.setEnabled(True)
        
    def update(self, message_type, data):
        """
        Update the panel with data from the plugin.
        
        Args:
            message_type: Type of message received
            data: Message data
        """
        # Handle different message types
        if message_type == "llm_response":
            # Store the response
            request_id = data.get("request_id")
            if request_id in self.pending_requests:
                logger.info(f"Received LLM response for request {request_id}")
                
                # Remove from pending requests
                self.pending_requests.pop(request_id)
                
                # Update conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": data.get("response", ""),
                    "timestamp": data.get("timestamp", datetime.now().isoformat())
                })
                
                # Update UI
                self._display_message(data.get("response", ""), "assistant")
                
                # Emit signal
                self.llm_response_received.emit(data)
                
                # Update UI state
                self._update_ui_state()
                
        elif message_type == "config":
            # Update configuration fields
            config = data.get("config", {})
            
            # Update UI elements with config values
            self.provider_combo.setCurrentText(config.get("provider", "openai"))
            self.model_combo.setCurrentText(config.get("model", "gpt-3.5-turbo"))
            self.temperature_spin.setValue(config.get("temperature", 0.7))
            self.max_tokens_spin.setValue(config.get("max_tokens", 1000))
            self.memory_mode_combo.setCurrentText(config.get("memory_mode", "combined"))
            
            # Don't display API key for security
            if config.get("api_key"):
                self.api_key_input.setPlaceholderText("API key is set")
            
            # Emit signal
            self.config_updated.emit(config)
            
        elif message_type == "memory_stats":
            # Update memory statistics display
            stats = data.get("stats", {})
            
            # Format statistics for display
            stats_text = "Memory System Statistics:\n"
            stats_text += f"- Memory Count: {stats.get('language_memory_stats', {}).get('memory_count', 0)}\n"
            stats_text += f"- Topics Synthesized: {len(stats.get('synthesis_stats', {}).get('topics_synthesized', []))}\n"
            stats_text += f"- Synthesis Count: {stats.get('synthesis_stats', {}).get('synthesis_count', 0)}\n"
            
            # Display stats
            self.memory_stats_display.setText(stats_text)
            
    def _connect_to_plugin(self):
        """Connect to the LLM Bridge Plugin"""
        # This will be implemented by the frontend socket manager
        self.status_label.setText("Status: Connecting...")
        
    def _show_system_message_dialog(self):
        """Show dialog to edit system message"""
        # Create dialog
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Set System Message")
        dialog.setMinimumWidth(400)
        
        # Create layout
        layout = QVBoxLayout(dialog)
        
        # Text edit for system message
        system_edit = QTextEdit()
        system_edit.setPlainText(self.system_message)
        layout.addWidget(system_edit)
        
        # Button box
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # Show dialog
        result = dialog.exec_()
        
        # Update system message if accepted
        if result == QtWidgets.QDialog.Accepted:
            self.system_message = system_edit.toPlainText()
        
    def _send_message(self):
        """Send a message to the LLM Bridge Plugin"""
        # Get message text
        message_text = self.message_input.toPlainText().strip()
        
        if not message_text:
            return
        
        # Disable send button while processing
        self.send_button.setEnabled(False)
        
        # Display user message
        self._display_message(message_text, "user")
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message_text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Clear input
        self.message_input.clear()
        
        # Create request
        request_id = str(uuid.uuid4())
        request = {
            "plugin_id": self.plugin_id,
            "type": "llm_request",
            "request_id": request_id,
            "content": {
                "text": message_text,
                "system_message": self.system_message,
                "conversation_history": self.conversation_history,
                "session_id": self.session_id
            }
        }
        
        # Store pending request
        self.pending_requests[request_id] = {
            "timestamp": datetime.now().isoformat(),
            "text": message_text
        }
        
        # Send request
        try:
            self.request_handler(self, request, self._handle_response)
            self.status_label.setText("Status: Processing request...")
        except Exception as e:
            logger.error(f"Error sending request: {str(e)}")
            self.status_label.setText(f"Status: Error - {str(e)}")
            self.send_button.setEnabled(True)
        
    def _handle_response(self, response):
        """
        Handle response from the plugin.
        
        Args:
            response: Response data
        """
        # Check response status
        status = response.get("status")
        
        if status == "processing":
            # Request is being processed
            self.status_label.setText("Status: Processing request...")
        elif status == "success":
            # Process successful response (this gets called for non-LLM responses)
            if "config" in response:
                # Config response
                self.update("config", response)
                self.status_label.setText("Status: Configuration updated")
            elif "stats" in response:
                # Memory stats response
                self.update("memory_stats", response)
                self.status_label.setText("Status: Memory stats updated")
        elif status == "error":
            # Handle error
            error_msg = response.get("error", "Unknown error")
            self.status_label.setText(f"Status: Error - {error_msg}")
            logger.error(f"Error response: {error_msg}")
            
            # Re-enable send button
            self.send_button.setEnabled(True)
            
    def _update_config(self):
        """Update the LLM configuration"""
        # Create request
        request = {
            "plugin_id": self.plugin_id,
            "type": "update_config",
            "request_id": str(uuid.uuid4()),
            "content": {
                "provider": self.provider_combo.currentText(),
                "model": self.model_combo.currentText(),
                "temperature": self.temperature_spin.value(),
                "max_tokens": self.max_tokens_spin.value(),
                "memory_mode": self.memory_mode_combo.currentText()
            }
        }
        
        # Add API key if provided
        api_key = self.api_key_input.text()
        if api_key:
            request["content"]["api_key"] = api_key
            
        # Send request
        try:
            self.request_handler(self, request, self._handle_response)
            self.status_label.setText("Status: Updating configuration...")
        except Exception as e:
            logger.error(f"Error updating config: {str(e)}")
            self.status_label.setText(f"Status: Error - {str(e)}")
            
    def _request_config(self):
        """Request current configuration from the plugin"""
        # Create request
        request = {
            "plugin_id": self.plugin_id,
            "type": "get_config",
            "request_id": str(uuid.uuid4())
        }
        
        # Send request
        try:
            self.request_handler(self, request, self._handle_response)
        except Exception as e:
            logger.error(f"Error requesting config: {str(e)}")
            
    def _refresh_memory_stats(self):
        """Request memory statistics from the plugin"""
        # Create request
        request = {
            "plugin_id": self.plugin_id,
            "type": "get_memory_stats",
            "request_id": str(uuid.uuid4())
        }
        
        # Send request
        try:
            self.request_handler(self, request, self._handle_response)
            self.status_label.setText("Status: Refreshing memory stats...")
        except Exception as e:
            logger.error(f"Error refreshing memory stats: {str(e)}")
            self.status_label.setText(f"Status: Error - {str(e)}")
            
    def _display_message(self, message, role):
        """
        Display a message in the chat area.
        
        Args:
            message: The message text
            role: The role (user/assistant)
        """
        # Create formatted message
        cursor = self.message_display.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        
        # Create a block format for alignment
        block_format = QtGui.QTextBlockFormat()
        
        if role == "user":
            # User messages aligned right
            block_format.setAlignment(Qt.AlignRight)
            cursor.insertBlock(block_format)
            
            # Add timestamp
            time_format = QtGui.QTextCharFormat()
            time_format.setForeground(QtGui.QBrush(QtGui.QColor(100, 100, 100)))
            time_format.setFontPointSize(8)
            cursor.insertText(f"{datetime.now().strftime('%H:%M:%S')}\n", time_format)
            
            # Add user label
            user_format = QtGui.QTextCharFormat()
            user_format.setFontWeight(QtGui.QFont.Bold)
            user_format.setForeground(QtGui.QBrush(ACCENT_COLOR))
            cursor.insertText("You:\n", user_format)
            
            # Add message
            message_format = QtGui.QTextCharFormat()
            cursor.insertText(f"{message}\n\n", message_format)
            
        else:
            # Assistant messages aligned left
            block_format.setAlignment(Qt.AlignLeft)
            cursor.insertBlock(block_format)
            
            # Add timestamp
            time_format = QtGui.QTextCharFormat()
            time_format.setForeground(QtGui.QBrush(QtGui.QColor(100, 100, 100)))
            time_format.setFontPointSize(8)
            cursor.insertText(f"{datetime.now().strftime('%H:%M:%S')}\n", time_format)
            
            # Add assistant label
            assistant_format = QtGui.QTextCharFormat()
            assistant_format.setFontWeight(QtGui.QFont.Bold)
            assistant_format.setForeground(QtGui.QBrush(PRIMARY_COLOR))
            cursor.insertText("Lumina:\n", assistant_format)
            
            # Add message
            message_format = QtGui.QTextCharFormat()
            cursor.insertText(f"{message}\n\n", message_format)
            
        # Scroll to bottom
        self.message_display.setTextCursor(cursor)
        self.message_display.ensureCursorVisible()
        
    def _update_ui_state(self):
        """Update UI state based on current status"""
        # Re-enable send button
        self.send_button.setEnabled(True)
        
        # Update status
        if not self.pending_requests:
            self.status_label.setText("Status: Ready")
            
    def keyPressEvent(self, event):
        """Handle key press events"""
        # Send message when Ctrl+Enter is pressed in the input field
        if (event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter) and \
           (event.modifiers() & Qt.ControlModifier) and \
           self.message_input.hasFocus():
            self._send_message()
        else:
            super().keyPressEvent(event)


# Component factory function
def create_component():
    """Create and return an LLM Panel instance"""
    return LLMPanel() 