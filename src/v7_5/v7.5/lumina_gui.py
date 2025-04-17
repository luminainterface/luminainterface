#!/usr/bin/env python3
"""
LUMINA v7.5 GUI Frontend
PySide6-based interface for the LUMINA chat system
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import PySide6 components
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTextEdit, QLineEdit, QPushButton,
                             QLabel, QFrame, QGridLayout, QTabWidget, QSlider)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QTextCursor

# Import Mistral integration
from src.api.mistral_integration_fixed import MistralIntegration
from src.v7_5.auto_wiki_processor import AutoWikiProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", f"lumina_gui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    ]
)
logger = logging.getLogger("LUMINA_GUI")

class ChatWidget(QWidget):
    """Main chat widget for LUMINA system"""
    message_sent = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Consolas", 10))
        self.layout.addWidget(self.chat_display)
        
        # Input area
        self.input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter your message...")
        self.input_field.returnPressed.connect(self.send_message)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        
        self.input_layout.addWidget(self.input_field)
        self.input_layout.addWidget(self.send_button)
        self.layout.addLayout(self.input_layout)
    
    def send_message(self):
        """Send a message from the input field"""
        message = self.input_field.text().strip()
        if message:
            self.message_sent.emit(message)
            self.input_field.clear()
    
    def add_user_message(self, message):
        """Add a user message to the chat display"""
        self.chat_display.append(f'<div style="margin: 5px;"><b>You: </b>{message}</div>')
    
    def add_system_message(self, message):
        """Add a system message to the chat display"""
        self.chat_display.append(f'<div style="margin: 5px;"><b>LUMINA: </b>{message}</div>')
        
    def add_process_message(self, message):
        """Add a process message to the chat display"""
        self.chat_display.append(f'<div style="margin: 5px;"><i>{message}</i></div>')

class SettingsWidget(QWidget):
    """Widget for adjusting system settings"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Title
        self.title = QLabel("System Settings")
        self.title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title)
        
        # LLM Weight Slider
        self.llm_weight_layout = QHBoxLayout()
        self.llm_weight_label = QLabel("LLM Weight:")
        self.llm_weight_slider = QSlider(Qt.Horizontal)
        self.llm_weight_slider.setMinimum(0)
        self.llm_weight_slider.setMaximum(100)
        self.llm_weight_slider.setValue(70)
        self.llm_weight_value = QLabel("0.7")
        
        self.llm_weight_layout.addWidget(self.llm_weight_label)
        self.llm_weight_layout.addWidget(self.llm_weight_slider)
        self.llm_weight_layout.addWidget(self.llm_weight_value)
        self.layout.addLayout(self.llm_weight_layout)
        
        # Neural Weight Slider
        self.nn_weight_layout = QHBoxLayout()
        self.nn_weight_label = QLabel("Neural Weight:")
        self.nn_weight_slider = QSlider(Qt.Horizontal)
        self.nn_weight_slider.setMinimum(0)
        self.nn_weight_slider.setMaximum(100)
        self.nn_weight_slider.setValue(30)
        self.nn_weight_value = QLabel("0.3")
        
        self.nn_weight_layout.addWidget(self.nn_weight_label)
        self.nn_weight_layout.addWidget(self.nn_weight_slider)
        self.nn_weight_layout.addWidget(self.nn_weight_value)
        self.layout.addLayout(self.nn_weight_layout)
        
        # Temperature Slider
        self.temp_layout = QHBoxLayout()
        self.temp_label = QLabel("Temperature:")
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setMinimum(0)
        self.temp_slider.setMaximum(100)
        self.temp_slider.setValue(70)
        self.temp_value = QLabel("0.7")
        
        self.temp_layout.addWidget(self.temp_label)
        self.temp_layout.addWidget(self.temp_slider)
        self.temp_layout.addWidget(self.temp_value)
        self.layout.addLayout(self.temp_layout)
        
        # Connect signals
        self.llm_weight_slider.valueChanged.connect(self.update_llm_weight)
        self.nn_weight_slider.valueChanged.connect(self.update_nn_weight)
        self.temp_slider.valueChanged.connect(self.update_temperature)
    
    def update_llm_weight(self, value):
        """Update LLM weight display"""
        weight = value / 100.0
        self.llm_weight_value.setText(f"{weight:.2f}")
    
    def update_nn_weight(self, value):
        """Update neural weight display"""
        weight = value / 100.0
        self.nn_weight_value.setText(f"{weight:.2f}")
    
    def update_temperature(self, value):
        """Update temperature display"""
        temp = value / 100.0
        self.temp_value.setText(f"{temp:.2f}")

class LuminaMainWindow(QMainWindow):
    """Main window for the LUMINA v7.5 interface"""
    
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("LUMINA v7.5")
        self.setMinimumSize(800, 600)
        
        # Initialize Mistral integration
        self.mistral = MistralIntegration(
            model=os.getenv("LLM_MODEL", "mistral-medium"),
            llm_weight=float(os.getenv("LLM_WEIGHT", "0.7")),
            nn_weight=float(os.getenv("NN_WEIGHT", "0.3")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            top_p=float(os.getenv("LLM_TOP_P", "0.9"))
        )
        
        # Initialize AutoWikiProcessor
        self.wiki_processor = AutoWikiProcessor()
        
        # Initialize conversation history
        self.conversation = [
            {"role": "system", "content": "You are LUMINA v7.5, an advanced AI assistant with integrated neural processing capabilities."}
        ]
        
        # Setup UI
        self.setup_ui()
        
        # Create conversations directory if it doesn't exist
        os.makedirs("data/conversations", exist_ok=True)
        
        logger.info("LUMINA v7.5 GUI initialized")
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main layout
        main_layout = QHBoxLayout(main_widget)
        
        # Create a tab widget for different sections
        self.tabs = QTabWidget()
        
        # Create chat tab
        chat_tab = QWidget()
        self.chat_widget = ChatWidget()
        chat_layout = QVBoxLayout(chat_tab)
        chat_layout.addWidget(self.chat_widget)
        self.tabs.addTab(chat_tab, "Chat")
        
        # Connect message signal
        self.chat_widget.message_sent.connect(self.process_user_message)
        
        # Create settings tab
        settings_tab = QWidget()
        self.settings_widget = SettingsWidget()
        settings_layout = QVBoxLayout(settings_tab)
        settings_layout.addWidget(self.settings_widget)
        self.tabs.addTab(settings_tab, "Settings")
        
        # Add tabs to main layout
        main_layout.addWidget(self.tabs)
    
    def process_user_message(self, message):
        """Process a user message"""
        logger.info(f"Processing user message: {message[:30]}...")
        
        # Add to chat display
        self.chat_widget.add_user_message(message)
        
        # Show "thinking" indicator
        self.chat_widget.add_process_message("Processing...")
        
        try:
            # Add user message to conversation
            self.conversation.append({"role": "user", "content": message})
            
            # Get response from Mistral integration
            response = self.mistral.process_message(self.conversation)
            
            # Add assistant response to conversation
            self.conversation.append({"role": "assistant", "content": response})
            
            # Update chat display
            self.chat_widget.add_system_message(response)
            
            # Run AutoWikiProcessor in background
            context = {
                "conversation": self.conversation,
                "weights": {
                    "llm": self.settings_widget.llm_weight_slider.value() / 100.0,
                    "nn": self.settings_widget.nn_weight_slider.value() / 100.0
                }
            }
            self.wiki_processor.run_in_background(message, context)
            
            # Update weights if changed
            llm_weight = self.settings_widget.llm_weight_slider.value() / 100.0
            nn_weight = self.settings_widget.nn_weight_slider.value() / 100.0
            temperature = self.settings_widget.temp_slider.value() / 100.0
            
            self.mistral.adjust_weights(
                llm_weight=llm_weight,
                nn_weight=nn_weight,
                temperature=temperature
            )
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.chat_widget.add_system_message(f"Error processing message: {str(e)}")
    
    def save_conversation(self):
        """Save the current conversation to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("data/conversations", f"conversation_{timestamp}.txt")
        
        with open(filename, "w", encoding="utf-8") as f:
            for message in self.conversation:
                f.write(f"{message['role'].upper()}: {message['content']}\n\n")
                
        logger.info(f"Conversation saved to {filename}")
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Cleanup resources
        self.wiki_processor.shutdown()
        event.accept()

def main():
    """Main entry point for the application"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show main window
    window = LuminaMainWindow()
    window.show()
    
    # Start event loop
    return app.exec()

if __name__ == "__main__":
    sys.exit(main()) 