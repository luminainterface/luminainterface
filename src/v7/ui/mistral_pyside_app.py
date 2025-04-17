#!/usr/bin/env python3
"""
Mistral PySide6 Frontend Application

This module provides a PySide6-based GUI for interacting with the Mistral LLM system.
"""

import sys
import os
import logging
from pathlib import Path

# Add parent directory to path to allow imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import PySide6
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QSlider, QSplitter,
    QListWidget, QListWidgetItem, QStatusBar, QToolBar,
    QDialog, QDialogButtonBox, QFormLayout, QMessageBox, QTabWidget,
    QCheckBox, QInputDialog
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QSize
from PySide6.QtGui import QIcon, QTextCursor, QColor, QPalette, QFont, QAction

# Import Mistral integration
from src.v7.ui.v7_socket_manager import V7SocketManager
from src.v7.ui.mistral_pyside_integration import MistralPySideIntegration
from src.v7.ui.onsite_memory_integration import OnsiteMemoryIntegration

# Configure logging
logger = logging.getLogger("MistralPySideApp")

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
        self.resize(1000, 700)
        
        # Get API key
        api_key_dialog = ApiKeyDialog(self)
        if api_key_dialog.exec() == QDialog.Accepted:
            self.api_key = api_key_dialog.get_api_key()
        else:
            self.api_key = ""
        
        # Set up socket manager and integration
        self.socket_manager = V7SocketManager()
        self.integration = MistralPySideIntegration(self.socket_manager, self.api_key)
        
        # Set up onsite memory integration
        self.memory_integration = OnsiteMemoryIntegration(
            data_dir="data/onsite_memory",
            memory_file="mistral_memory.json"
        )
        
        # Connect signals
        self.integration.signals.response_received.connect(self._on_response_received)
        self.integration.signals.error_occurred.connect(self._on_error_occurred)
        self.integration.signals.status_update.connect(self._on_status_update)
        self.integration.signals.weights_updated.connect(self._on_weights_updated)
        self.integration.signals.processing_started.connect(self._on_processing_started)
        self.integration.signals.processing_finished.connect(self._on_processing_finished)
        
        # Initialize memory-related variables
        self.use_memory = True
        self.current_user_message = ""
        
        # Create UI
        self._create_ui()
        
        # Start status timer
        self.integration.start_status_timer(3000)  # Update every 3 seconds
        
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
        
        # Create settings tab widget
        settings_tabs = QTabWidget()
        
        # Create weights tab
        weights_tab = QWidget()
        weights_layout = QVBoxLayout(weights_tab)
        
        # Create LLM weight slider
        llm_layout = QVBoxLayout()
        llm_label = QLabel("LLM Weight: 0.7")
        self.llm_label = llm_label
        llm_layout.addWidget(llm_label)
        
        llm_slider = QSlider(Qt.Horizontal)
        llm_slider.setMinimum(0)
        llm_slider.setMaximum(100)
        llm_slider.setValue(70)
        llm_slider.valueChanged.connect(self._on_llm_weight_changed)
        self.llm_slider = llm_slider
        llm_layout.addWidget(llm_slider)
        
        weights_layout.addLayout(llm_layout)
        
        # Create NN weight slider
        nn_layout = QVBoxLayout()
        nn_label = QLabel("Neural Network Weight: 0.3")
        self.nn_label = nn_label
        nn_layout.addWidget(nn_label)
        
        nn_slider = QSlider(Qt.Horizontal)
        nn_slider.setMinimum(0)
        nn_slider.setMaximum(100)
        nn_slider.setValue(30)
        nn_slider.valueChanged.connect(self._on_nn_weight_changed)
        self.nn_slider = nn_slider
        nn_layout.addWidget(nn_slider)
        
        weights_layout.addLayout(nn_layout)
        
        # Create update button
        update_button = QPushButton("Update Weights")
        update_button.clicked.connect(self._update_weights)
        weights_layout.addWidget(update_button)
        
        weights_layout.addStretch(1)
        
        # Add weights tab
        settings_tabs.addTab(weights_tab, "Weights")
        
        # Create memory settings tab
        memory_tab = QWidget()
        memory_layout = QVBoxLayout(memory_tab)
        
        # Memory toggle checkbox
        self.memory_checkbox = QCheckBox("Use Memory for Context")
        self.memory_checkbox.setChecked(self.use_memory)
        self.memory_checkbox.stateChanged.connect(self._toggle_memory)
        memory_layout.addWidget(self.memory_checkbox)
        
        # Extract knowledge button
        extract_button = QPushButton("Extract Knowledge from Last Chat")
        extract_button.clicked.connect(self._extract_knowledge)
        memory_layout.addWidget(extract_button)
        
        # Memory stats
        memory_stats_button = QPushButton("View Memory Stats")
        memory_stats_button.clicked.connect(self._view_memory_stats)
        memory_layout.addWidget(memory_stats_button)
        
        memory_layout.addStretch(1)
        
        # Add memory tab
        settings_tabs.addTab(memory_tab, "Memory Settings")
        
        # Add tabs to settings layout
        settings_layout.addWidget(settings_tabs)
        
        # Add status display
        status_label = QLabel("Status: Ready")
        self.status_label = status_label
        settings_layout.addWidget(status_label)
        
        # Add widgets to splitter
        splitter.addWidget(chat_widget)
        splitter.addWidget(settings_widget)
        
        # Set splitter sizes
        splitter.setSizes([700, 300])
        
        # Create memory panel tab
        self.tab_widget = QTabWidget()
        
        # Add chat panel as the first tab
        self.tab_widget.addTab(central_widget, "Chat")
        
        # Add memory panel as the second tab
        memory_panel = self.memory_integration.get_memory_panel()
        self.tab_widget.addTab(memory_panel, "Memory")
        
        # Set the tab widget as the central widget
        self.setCentralWidget(self.tab_widget)
        
        # Create status bar
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        
        # Create menu bar
        menu_bar = self.menuBar()
        
        # Create file menu
        file_menu = menu_bar.addMenu("File")
        
        # Create exit action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Create edit menu
        edit_menu = menu_bar.addMenu("Edit")
        
        # Create clear action
        clear_action = QAction("Clear Chat", self)
        clear_action.triggered.connect(self._clear_chat)
        edit_menu.addAction(clear_action)
        
        # Create help menu
        help_menu = menu_bar.addMenu("Help")
        
        # Create about action
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _send_message(self):
        """Send a message to the Mistral system"""
        message = self.message_input.text().strip()
        if not message:
            return
        
        # Store message for potential knowledge extraction
        self.current_user_message = message
        
        # Check if we should use memory context
        if self.use_memory:
            # Get relevant context from memory
            context = self.memory_integration.search_context_for_query(message)
            
            if context:
                # Add context to the message
                enhanced_message = f"{context}\n\nWith that in mind, please answer: {message}"
                
                # Add memory context note to chat
                self._add_system_message("Using memory context to enhance your query...")
                
                # Send enhanced message
                self._add_message_to_chat("user", message)
                self.integration.process_message_async(enhanced_message)
            else:
                # No relevant context found, send original message
                self._add_message_to_chat("user", message)
                self.integration.process_message_async(message)
        else:
            # Send original message without memory context
            self._add_message_to_chat("user", message)
            self.integration.process_message_async(message)
        
        # Clear input field
        self.message_input.clear()
    
    def _add_message_to_chat(self, role, content):
        """Add a message to the chat history"""
        # Create message HTML
        message = f"<p><b>{role.capitalize()}:</b> "
        message += content.replace('\n', '<br>')
        message += "</p>"
        
        # Add to chat scroll
        self.chat_scroll.append(message)
        
        # Store in memory if enabled
        if self.use_memory and role != "system":
            self.memory_integration.store_exchange(
                role=role,
                content=content,
                timestamp=None  # Use current time
            )
    
    def _add_system_message(self, content):
        """Add a system message to the chat history"""
        # Create message widget
        message = f"<div style='margin-bottom: 10px; padding: 8px; border-radius: 5px; background-color: #f5f5f5; color: #666;'>"
        message += f"<i>{content}</i>"
        message += "</div>"
        
        # Add to chat scroll
        self.chat_scroll.insertHtml(message)
        
        # Scroll to bottom
        cursor = self.chat_scroll.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_scroll.setTextCursor(cursor)
    
    def _on_response_received(self, request_id, response_data):
        """Handle response from the Mistral system"""
        response = response_data.get("message", "")
        
        # Add response to chat
        self._add_message_to_chat("assistant", response)
        
        # Store the conversation in memory if enabled
        if self.use_memory and self.current_user_message and response:
            self.memory_integration.add_conversation(
                self.current_user_message,
                response,
                {"request_id": request_id}
            )
    
    def _on_error_occurred(self, error_message):
        """Handle error from the Mistral system"""
        # Add error message to chat
        self._add_system_message(f"Error: {error_message}")
    
    def _on_status_update(self, status):
        """Handle status update from the Mistral system"""
        # Update status label
        status_text = "Status: "
        
        if status.get("mistral_available", False):
            status_text += "Mistral LLM Connected | "
        else:
            status_text += "Mistral LLM Disconnected | "
        
        if status.get("enhanced_language_available", False):
            status_text += "Neural Processing Connected | "
        else:
            status_text += "Neural Processing Disconnected | "
        
        status_text += f"LLM: {status.get('llm_weight', 0.7):.1f} | "
        status_text += f"NN: {status.get('nn_weight', 0.3):.1f}"
        
        self.status_label.setText(status_text)
    
    def _on_weights_updated(self, llm_weight, nn_weight):
        """Handle weights update from the Mistral system"""
        # Update sliders and labels
        self.llm_slider.setValue(int(llm_weight * 100))
        self.nn_slider.setValue(int(nn_weight * 100))
        self.llm_label.setText(f"LLM Weight: {llm_weight:.1f}")
        self.nn_label.setText(f"Neural Network Weight: {nn_weight:.1f}")
    
    def _on_processing_started(self):
        """Handle processing started event"""
        # Update status
        self.status_label.setText("Status: Processing...")
        
        # Disable input
        self.message_input.setEnabled(False)
    
    def _on_processing_finished(self):
        """Handle processing finished event"""
        # Enable input
        self.message_input.setEnabled(True)
        
        # Set focus to input field
        self.message_input.setFocus()
    
    def _on_llm_weight_changed(self, value):
        """Handle LLM weight slider change"""
        weight = value / 100.0
        self.llm_label.setText(f"LLM Weight: {weight:.1f}")
    
    def _on_nn_weight_changed(self, value):
        """Handle NN weight slider change"""
        weight = value / 100.0
        self.nn_label.setText(f"Neural Network Weight: {weight:.1f}")
    
    def _update_weights(self):
        """Update weights in the Mistral system"""
        llm_weight = self.llm_slider.value() / 100.0
        nn_weight = self.nn_slider.value() / 100.0
        
        self.integration.set_weights(llm_weight, nn_weight)
    
    def _clear_chat(self):
        """Clear the chat history"""
        self.chat_scroll.clear()
        self._add_system_message("Chat history cleared")
    
    def _show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Mistral Chat",
            "Mistral Chat v1.0\n\n"
            "A PySide6 frontend for the Mistral AI language model.\n\n"
            "Features:\n"
            "- Chat with Mistral AI language models\n"
            "- Adjust LLM and Neural Network weights\n"
            "- Persistent memory of conversations\n"
            "- Knowledge extraction and retrieval"
        )
    
    def _toggle_memory(self, state):
        """Toggle memory usage"""
        self.use_memory = state == Qt.Checked
        
        if self.use_memory:
            self._add_system_message("Memory context enabled - your conversations will be stored and used for context")
        else:
            self._add_system_message("Memory context disabled - your conversations will not be stored")
    
    def _extract_knowledge(self):
        """Extract knowledge from the last conversation"""
        if not self.current_user_message:
            QMessageBox.warning(self, "No Conversation", "No recent conversation to extract knowledge from.")
            return
        
        # Get the last conversation from the memory
        conversations = self.memory_integration.memory.get_conversation_history(limit=1)
        if not conversations:
            QMessageBox.warning(self, "No Conversation", "No stored conversations found in memory.")
            return
        
        last_conversation = conversations[0]
        user_message = last_conversation.get("user_message", "")
        assistant_response = last_conversation.get("assistant_response", "")
        
        # Ask for a topic/key for this knowledge
        topic, ok = QInputDialog.getText(
            self, 
            "Extract Knowledge",
            "Enter a topic/keyword for this knowledge:",
            text=self._generate_topic_suggestion(user_message)
        )
        
        if ok and topic:
            # Add to knowledge base
            success = self.memory_integration.add_knowledge_from_conversation(
                topic,
                user_message,
                assistant_response
            )
            
            if success:
                self._add_system_message(f"Knowledge extracted and saved under topic: '{topic}'")
            else:
                self._add_system_message(f"Failed to extract knowledge. See logs for details.")
    
    def _generate_topic_suggestion(self, message):
        """Generate a topic suggestion from a message"""
        # Simple implementation - use the first few words
        words = message.split()
        if len(words) <= 3:
            return message
        return " ".join(words[:3]) + "..."
    
    def _view_memory_stats(self):
        """View memory statistics"""
        stats = self.memory_integration.memory.get_stats()
        
        stats_text = "Memory Statistics:\n\n"
        for key, value in stats.items():
            if key == "last_conversation" and value:
                # Format timestamp
                date_part = value.split("T")[0]
                stats_text += f"Last conversation: {date_part}\n"
            else:
                # Format other stats
                stats_text += f"{key.replace('_', ' ').title()}: {value}\n"
        
        QMessageBox.information(self, "Memory Statistics", stats_text)
    
    def closeEvent(self, event):
        """Handle application close event"""
        # Shutdown memory integration
        self.memory_integration.shutdown()
        
        # Call parent implementation
        super().closeEvent(event)

# Main entry point
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create main window
    window = MistralChatWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec()) 