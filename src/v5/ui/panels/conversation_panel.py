#!/usr/bin/env python3
"""
V5 Conversation Panel with NN/LLM Weighted Integration

This panel provides an interactive chat interface with integrated NN/LLM weighting capabilities,
allowing for fine-tuned control of the balance between neural network and language model processing.
"""

import os
import sys
import json
import logging
import time
import random
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime

# Add project root to Python path if needed
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import Qt compatibility layer
from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui, Qt, Signal, Slot
from src.v5.ui.qt_compat import get_widgets, get_gui, get_core

# Get required Qt classes
QSplitter = get_widgets().QSplitter
QFormLayout = get_widgets().QFormLayout
QSlider = get_widgets().QSlider
QTimer = get_core().QTimer
QPainter = get_gui().QPainter
QLinearGradient = get_gui().QLinearGradient
QColor = get_gui().QColor
QFont = get_gui().QFont
QPen = get_gui().QPen

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("conversation-panel")

try:
    # Try to import PySide6
    from PySide6.QtCore import Qt, Signal, Slot, QSize
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
        QPushButton, QSlider, QLabel, QComboBox, QFrame, QSplitter,
        QScrollArea, QGroupBox
    )
    from PySide6.QtGui import QFont, QColor, QPalette, QTextCursor
    logger.info("Using PySide6 for Conversation Panel")
except ImportError:
    try:
        # Fall back to PyQt5
        from PyQt5.QtCore import Qt, pyqtSignal as Signal, pyqtSlot as Slot, QSize
        from PyQt5.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
            QPushButton, QSlider, QLabel, QComboBox, QFrame, QSplitter,
            QScrollArea, QGroupBox
        )
        from PyQt5.QtGui import QFont, QColor, QPalette, QTextCursor
        logger.info("Using PyQt5 for Conversation Panel")
    except ImportError:
        logger.error("Failed to import Qt libraries. Conversation Panel will not be available.")
        raise

# Try to import memory API
try:
    from src.memory_api import MemoryAPI
    MEMORY_API_AVAILABLE = True
except ImportError:
    MEMORY_API_AVAILABLE = False
    logging.warning("Memory API not available. Conversation panel will use limited functionality.")

# Try to import language memory integration
try:
    from src.v5.language_memory_integration import LanguageMemoryIntegrationPlugin
    LANGUAGE_INTEGRATION_AVAILABLE = True
except ImportError:
    LANGUAGE_INTEGRATION_AVAILABLE = False
    logging.warning("Language Memory Integration not available. Using fallback mode.")

class ConversationPanel(QtWidgets.QWidget):
    """
    Interactive conversation panel with NN/LLM weighting for the V5 visualization system.
    
    This panel integrates with the Language Memory System to provide memory-enhanced
    conversation capabilities with adjustable balance between neural network and
    language model processing.
    """
    
    # Define signals
    message_sent = Signal(str)  # Emitted when a message is sent
    weight_changed = Signal(float)  # Emitted when NN/LLM weight changes
    
    def __init__(self, socket_manager=None):
        super().__init__()
        self.socket_manager = socket_manager
        self.nn_llm_weight = 0.5  # Default balanced weighting
        self.memory_mode = "combined"  # combined, contextual, or synthesized
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation_history = []
        
        # Initialize Memory API if available
        self.memory_api = None
        if MEMORY_API_AVAILABLE:
            try:
                self.memory_api = MemoryAPI()
                logger.info("Memory API initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Memory API: {e}")
        
        # Initialize Language Memory Integration if available
        self.language_integration = None
        if LANGUAGE_INTEGRATION_AVAILABLE and socket_manager:
            try:
                self.language_integration = LanguageMemoryIntegrationPlugin(plugin_id="language_memory_integration")
                logger.info("Language Memory Integration initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Language Memory Integration: {e}")
        
        self.initUI()
        
    def initUI(self):
        """Initialize the user interface"""
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Vertical)
        
        # Upper section - Chat interface
        chat_widget = QtWidgets.QWidget()
        chat_layout = QtWidgets.QVBoxLayout(chat_widget)
        chat_layout.setContentsMargins(15, 15, 15, 15)
        chat_layout.setSpacing(12)
        
        # Title and session info
        header_widget = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 10)
        
        title_label = QtWidgets.QLabel("Memory-Enhanced Conversation")
        title_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #3498DB;
        """)
        
        session_label = QtWidgets.QLabel(f"Session: {self.session_id.split('_')[1]}")
        session_label.setStyleSheet("""
            font-size: 12px;
            color: #7F8C8D;
        """)
        session_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(session_label)
        chat_layout.addWidget(header_widget)
        
        # Add separator line
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        separator.setStyleSheet("background-color: #34495E;")
        separator.setMaximumHeight(1)
        chat_layout.addWidget(separator)
        
        # Chat history
        self.chat_area = QtWidgets.QScrollArea()
        self.chat_area.setWidgetResizable(True)
        self.chat_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_area.setStyleSheet("""
            QScrollArea {
                background-color: #121A24; 
                border: none;
                border-radius: 8px;
            }
            QScrollBar:vertical {
                border: none;
                background: #1E2C3A;
                width: 10px;
                margin: 0px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #2C3E50;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
                height: 0px;
            }
        """)
        
        self.chat_container = QtWidgets.QWidget()
        self.chat_layout = QtWidgets.QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setSpacing(12)
        self.chat_layout.setContentsMargins(5, 5, 5, 5)
        self.chat_container.setLayout(self.chat_layout)
        
        self.chat_area.setWidget(self.chat_container)
        chat_layout.addWidget(self.chat_area, 1)
        
        # Input area with improved styling
        input_widget = QtWidgets.QWidget()
        input_widget.setStyleSheet("""
            background-color: #1A2634;
            border-radius: 10px;
        """)
        input_layout = QtWidgets.QHBoxLayout(input_widget)
        input_layout.setContentsMargins(12, 12, 12, 12)
        input_layout.setSpacing(10)
        
        self.input_field = QtWidgets.QLineEdit()
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: #1E2C3A;
                color: #ECF0F1;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                border: 1px solid #34495E;
            }
            QLineEdit:focus {
                border: 1px solid #3498DB;
            }
        """)
        self.input_field.returnPressed.connect(self.send_message)
        
        send_button = QtWidgets.QPushButton()
        send_button.setIcon(QtGui.QIcon.fromTheme("edit-redo"))
        send_button.setIconSize(QtCore.QSize(18, 18))
        send_button.setMinimumSize(40, 40)
        send_button.setCursor(QtCore.Qt.PointingHandCursor)
        send_button.setStyleSheet("""
            QPushButton {
                background-color: #2980B9;
                color: white;
                border-radius: 8px;
                padding: 5px;
                min-width: 40px;
                min-height: 40px;
                icon-size: 18px;
            }
            QPushButton:hover {
                background-color: #3498DB;
            }
            QPushButton:pressed {
                background-color: #1C587F;
            }
        """)
        send_button.clicked.connect(self.send_message)
        
        # Add a microphone button for future voice input
        mic_button = QtWidgets.QPushButton()
        mic_button.setIcon(QtGui.QIcon.fromTheme("audio-input-microphone"))
        mic_button.setIconSize(QtCore.QSize(18, 18))
        mic_button.setMinimumSize(40, 40)
        mic_button.setCursor(QtCore.Qt.PointingHandCursor)
        mic_button.setToolTip("Voice Input (Coming Soon)")
        mic_button.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50;
                color: white;
                border-radius: 8px;
                padding: 5px;
                min-width: 40px;
                min-height: 40px;
                icon-size: 18px;
            }
            QPushButton:hover {
                background-color: #34495E;
            }
            QPushButton:pressed {
                background-color: #1C2833;
            }
        """)
        mic_button.setEnabled(False)  # Disabled for now
        
        input_layout.addWidget(self.input_field, 1)
        input_layout.addWidget(send_button)
        input_layout.addWidget(mic_button)
        
        chat_layout.addWidget(input_widget)
        
        # Rest of the UI initialization remains unchanged...
        # Lower section - NN/LLM weighting and memory controls
        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(10)
        
        # Title for controls section
        controls_title = QtWidgets.QLabel("NN/LLM Weighting & Memory Controls")
        controls_title.setStyleSheet("color: #3498DB; font-weight: bold; font-size: 16px;")
        controls_layout.addWidget(controls_title)
        
        # NN/LLM Weighting slider
        weight_widget = QtWidgets.QWidget()
        weight_layout = QtWidgets.QVBoxLayout(weight_widget)
        weight_layout.setContentsMargins(0, 0, 0, 0)
        
        # Slider labels
        slider_labels_layout = QtWidgets.QHBoxLayout()
        nn_label = QtWidgets.QLabel("Neural Network")
        nn_label.setStyleSheet("color: #E74C3C; font-size: 12px;")
        llm_label = QtWidgets.QLabel("Language Model")
        llm_label.setStyleSheet("color: #2ECC71; font-size: 12px;")
        
        slider_labels_layout.addWidget(nn_label)
        slider_labels_layout.addStretch()
        slider_labels_layout.addWidget(llm_label)
        
        weight_layout.addLayout(slider_labels_layout)
        
        # Weighting slider
        self.weight_slider = QSlider(Qt.Horizontal)
        self.weight_slider.setRange(0, 100)
        self.weight_slider.setValue(int(self.nn_llm_weight * 100))
        self.weight_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #E74C3C, stop:1 #2ECC71);
                border-radius: 4px;
            }
            
            QSlider::handle:horizontal {
                background: #ECF0F1;
                border: 1px solid #7F8C8D;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
        """)
        self.weight_slider.valueChanged.connect(self.on_weight_changed)
        weight_layout.addWidget(self.weight_slider)
        
        # Weight display
        self.weight_display = QtWidgets.QLabel(f"Current balance: 50% NN / 50% LLM")
        self.weight_display.setAlignment(Qt.AlignCenter)
        self.weight_display.setStyleSheet("color: #ECF0F1; font-size: 14px;")
        weight_layout.addWidget(self.weight_display)
        
        controls_layout.addWidget(weight_widget)
        
        # Memory mode selection
        memory_group = QtWidgets.QGroupBox("Memory Enhancement Mode")
        memory_group.setStyleSheet("""
            QGroupBox {
                color: #ECF0F1;
                border: 1px solid #34495E;
                border-radius: 5px;
                margin-top: 1em;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        memory_layout = QtWidgets.QVBoxLayout(memory_group)
        
        # Memory mode radio buttons
        self.contextual_radio = QtWidgets.QRadioButton("Contextual")
        self.contextual_radio.setStyleSheet("color: #ECF0F1;")
        self.contextual_radio.setToolTip("Focus on conversation context and history")
        
        self.synthesized_radio = QtWidgets.QRadioButton("Synthesized")
        self.synthesized_radio.setStyleSheet("color: #ECF0F1;")
        self.synthesized_radio.setToolTip("Focus on synthesized understanding of topics")
        
        self.combined_radio = QtWidgets.QRadioButton("Combined")
        self.combined_radio.setStyleSheet("color: #ECF0F1;")
        self.combined_radio.setToolTip("Balance between contextual and synthesized memory")
        self.combined_radio.setChecked(True)  # Default
        
        # Connect radio buttons
        self.contextual_radio.toggled.connect(lambda: self.on_memory_mode_changed("contextual"))
        self.synthesized_radio.toggled.connect(lambda: self.on_memory_mode_changed("synthesized"))
        self.combined_radio.toggled.connect(lambda: self.on_memory_mode_changed("combined"))
        
        memory_layout.addWidget(self.contextual_radio)
        memory_layout.addWidget(self.synthesized_radio)
        memory_layout.addWidget(self.combined_radio)
        
        controls_layout.addWidget(memory_group)
        
        # Memory stats display
        self.memory_stats = QtWidgets.QLabel("Memory system not yet initialized")
        self.memory_stats.setStyleSheet("color: #7F8C8D; font-size: 12px;")
        controls_layout.addWidget(self.memory_stats)
        
        # Add widgets to splitter
        splitter.addWidget(chat_widget)
        splitter.addWidget(controls_widget)
        
        # Set initial sizes (70% chat, 30% controls)
        splitter.setSizes([700, 300])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Initialize timer for periodic updates
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_memory_stats)
        self.update_timer.start(10000)  # Update every 10 seconds
        
        # Add an initial system message
        self.add_system_message("Welcome to the V5 Memory-Enhanced Conversation System. This interface provides weighted NN/LLM processing with integrated language memory capabilities.")
        
        # Update memory stats
        self._update_memory_stats()
    
    def add_message(self, is_user, text):
        """Add a new message to the chat"""
        message_frame = QtWidgets.QFrame()
        message_frame.setObjectName("userMessage" if is_user else "systemMessage")
        
        # Use different styling for user vs system messages
        if is_user:
            message_frame.setStyleSheet("""
                #userMessage {
                    background-color: #2C3E50;
                    border-radius: 12px;
                    margin-left: 40px;
                }
            """)
        else:
            message_frame.setStyleSheet("""
                #systemMessage {
                    background-color: #1E2C3A;
                    border-radius: 12px;
                    margin-right: 40px;
                    border-left: 3px solid #16A085;
                }
            """)
        
        message_layout = QtWidgets.QVBoxLayout(message_frame)
        message_layout.setContentsMargins(12, 12, 12, 12)
        message_layout.setSpacing(8)
        
        # Create header with avatar, name and timestamp
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 5)
        
        # Simple avatar (colored circle with initial)
        avatar_label = QtWidgets.QLabel()
        avatar_label.setFixedSize(28, 28)
        avatar_label.setAlignment(Qt.AlignCenter)
        avatar_label.setStyleSheet(f"""
            background-color: {('#3498DB' if is_user else '#16A085')};
            color: white;
            font-weight: bold;
            border-radius: 14px;
        """)
        avatar_label.setText("Y" if is_user else "L")
        
        name_label = QtWidgets.QLabel("You" if is_user else "Lumina")
        name_label.setStyleSheet(f"color: {('#3498DB' if is_user else '#16A085')}; font-weight: bold;")
        
        time_label = QtWidgets.QLabel(datetime.now().strftime("%H:%M"))
        time_label.setStyleSheet("color: #7F8C8D; font-size: 10px;")
        time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        header_layout.addWidget(avatar_label)
        header_layout.addWidget(name_label, 1)
        header_layout.addWidget(time_label)
        
        # Main text content
        text_label = QtWidgets.QLabel(text)
        text_label.setWordWrap(True)
        text_label.setStyleSheet("color: #ECF0F1; font-size: 14px;")
        text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        message_layout.addLayout(header_layout)
        message_layout.addWidget(text_label)
        
        self.chat_layout.addWidget(message_frame)
        
        # Scroll to bottom
        self.chat_area.verticalScrollBar().setValue(
            self.chat_area.verticalScrollBar().maximum()
        )
    
    def add_system_message(self, text):
        """Add a system message to the chat"""
        message_frame = QtWidgets.QFrame()
        message_frame.setObjectName("infoMessage")
        message_frame.setStyleSheet("""
            #infoMessage {
                background-color: #2D4053;
                border-radius: 12px;
                margin-left: 20px;
                margin-right: 20px;
            }
        """)
        
        message_layout = QtWidgets.QVBoxLayout(message_frame)
        message_layout.setContentsMargins(12, 12, 12, 12)
        
        # Text content with a system icon
        text_label = QtWidgets.QLabel(text)
        text_label.setWordWrap(True)
        text_label.setStyleSheet("color: #BDC3C7; font-size: 13px; font-style: italic;")
        text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        message_layout.addWidget(text_label)
        
        self.chat_layout.addWidget(message_frame)
        
        # Scroll to bottom
        self.chat_area.verticalScrollBar().setValue(
            self.chat_area.verticalScrollBar().maximum()
        )
    
    def send_message(self):
        """Send a message from the input field"""
        message = self.input_field.text().strip()
        if not message:
            return
            
        # Add message to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Add message to chat display
        self.add_message(True, message)
        
        # Clear input field
        self.input_field.clear()
        
        # Emit signal for message sent
        self.message_sent.emit(message)
        
        # Process message with memory system
        self._process_message(message)
    
    def _process_message(self, message):
        """Process message with the memory system and generate a response"""
        try:
            # If language integration is available, use it
            if self.language_integration and self.socket_manager:
                # Create memory data object to pass to language integration
                memory_data = {
                    "text": message,
                    "session_id": self.session_id,
                    "timestamp": datetime.now().isoformat(),
                    "nn_weight": self.nn_llm_weight,
                    "llm_weight": 1 - self.nn_llm_weight,
                    "memory_mode": self.memory_mode
                }
                
                # In a real implementation, this would connect to the language memory system
                # and neural network processing pipeline
                self.language_integration.process_message(memory_data)
                
                # For demo purposes, generate a simulated response
                self._generate_simulated_response(message)
            
            # Otherwise use memory API if available
            elif self.memory_api:
                # For demo purposes, generate a simulated response
                self._generate_simulated_response(message)
            
            # No memory integration available
            else:
                self.add_system_message("Memory integration not available. Using echo mode.")
                self.add_message(False, f"You said: {message}")
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.add_system_message(f"Error processing message: {str(e)}")
    
    def _generate_simulated_response(self, message):
        """Generate a simulated response for demonstration purposes"""
        # In a real implementation, this would be replaced with actual NN/LLM processing
        
        # Simulate processing delay
        QtCore.QTimer.singleShot(800, lambda: self._add_simulated_response(message))
    
    def _add_simulated_response(self, message):
        """Add a simulated response to the chat area"""
        # Generate a response based on the message
        responses = [
            f"I understood your message about '{message[:20]}...' using the memory system with {int(self.nn_llm_weight * 100)}% neural network weighting.",
            f"The memory system has processed your input using {self.memory_mode} mode. It seems related to previous conversations about similar topics.",
            f"Based on the fractal memory patterns, I've analyzed your message with {int((1-self.nn_llm_weight) * 100)}% language model influence.",
            f"Your message has been synthesized with existing knowledge using the {self.memory_mode} memory enhancement mode."
        ]
        
        response = random.choice(responses)
        
        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Display in chat
        self.add_message(False, response)
        
        # Update memory stats (in a real implementation, this would show actual stats)
        self._update_memory_stats()
    
    def on_weight_changed(self, value):
        """Update the NN/LLM weight value"""
        self.nn_llm_weight = value / 100.0
        nn_percent = int(self.nn_llm_weight * 100)
        llm_percent = 100 - nn_percent
        
        # Update percentage indicators
        self.weight_slider.setValue(nn_percent)
        
        # Emit signal with weight value
        self.weight_changed.emit(self.nn_llm_weight)
        
        logger.info(f"NN/LLM weight set to {self.nn_llm_weight:.2f}")
        
        # Style the slider handle based on the current value
        handle_color = ""
        if value < 30:
            handle_color = "#2ECC71"  # Green for LLM
        elif value > 70:
            handle_color = "#E74C3C"  # Red for NN
        else:
            handle_color = "#3498DB"  # Blue for balanced
        
        self.weight_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 10px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #E74C3C, stop:1 #2ECC71);
                border-radius: 5px;
            }}
            
            QSlider::handle:horizontal {{
                background: {handle_color};
                border: 2px solid #7F8C8D;
                width: 22px;
                height: 22px;
                margin: -8px 0;
                border-radius: 11px;
            }}
            
            QSlider::handle:horizontal:hover {{
                background: {handle_color};
                border: 2px solid #3498DB;
            }}
        """)
        
        # Provide feedback about the current weighting
        if value == 0:
            self.add_system_message("Full language model processing. Responses will be comprehensive and nuanced.")
        elif value == 100:
            self.add_system_message("Full neural network processing. Responses will be pattern-focused and concise.")
        elif value % 25 == 0:  # Only show at 25%, 50%, 75% to avoid too many messages
            if value < 50:
                self.add_system_message(f"Weighting toward language model ({llm_percent}%). Responses will favor nuanced understanding.")
            elif value > 50:
                self.add_system_message(f"Weighting toward neural network ({nn_percent}%). Responses will favor pattern recognition.")
            else:
                self.add_system_message("Balanced weighting (50/50). Responses will combine pattern recognition with nuanced understanding.")
    
    def on_memory_mode_changed(self, mode):
        """Update the memory enhancement mode"""
        self.memory_mode = mode
        logger.info(f"Memory mode set to {mode}")
        
        # Update card styling for the selected mode
        contextual_card = self.contextual_radio.parent()
        synthesized_card = self.synthesized_radio.parent()
        combined_card = self.combined_radio.parent()
        
        # Reset all card styles
        contextual_card.setStyleSheet("""
            background-color: #121A24;
            border: 1px solid #34495E;
            border-radius: 8px;
        """)
        synthesized_card.setStyleSheet("""
            background-color: #121A24;
            border: 1px solid #34495E;
            border-radius: 8px;
        """)
        combined_card.setStyleSheet("""
            background-color: #121A24;
            border: 1px solid #34495E;
            border-radius: 8px;
        """)
        
        # Highlight the selected card
        if mode == "contextual":
            contextual_card.setStyleSheet("""
                background-color: #1A2634;
                border: 2px solid #3498DB;
                border-radius: 8px;
            """)
            self.add_system_message("Memory mode set to contextual. Responses will focus on conversation history.")
        elif mode == "synthesized":
            synthesized_card.setStyleSheet("""
                background-color: #1A2634;
                border: 2px solid #3498DB;
                border-radius: 8px;
            """)
            self.add_system_message("Memory mode set to synthesized. Responses will focus on topic understanding.")
        else:  # combined
            combined_card.setStyleSheet("""
                background-color: #1A2634;
                border: 2px solid #3498DB;
                border-radius: 8px;
            """)
            self.add_system_message("Memory mode set to combined. Responses will balance conversation history with topic understanding.")
    
    def _update_memory_stats(self):
        """Update memory statistics display"""
        if self.memory_api:
            try:
                # Get stats from memory API if available
                if hasattr(self.memory_api, 'get_stats'):
                    stats = self.memory_api.get_stats()
                    
                    # Format stats for display
                    stats_text = f"Total Memories: {stats.get('total_memories', 0)}\n"
                    stats_text += f"Total Conversations: {stats.get('total_conversations', 0)}\n"
                    stats_text += f"Top Topics: {', '.join([t.get('topic') for t in stats.get('top_topics', [])])}\n"
                    
                    # Update display
                    self.memory_stats.setText(stats_text)
                else:
                    self.memory_stats.setText("Memory API active but stats unavailable")
            except Exception as e:
                logger.error(f"Error getting memory stats: {e}")
                self.memory_stats.setText("Memory statistics unavailable")
        else:
            # Mock statistics
            stats_text = "Mock Mode Statistics:\n"
            stats_text += "Total Memories: 1250\n"
            stats_text += "Total Conversations: 85\n"
            stats_text += "Top Topics: neural networks, consciousness, machine learning"
            self.memory_stats.setText(stats_text)
    
    def paintEvent(self, event):
        """Paint the widget background"""
        # Create a custom gradient background
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(18, 26, 36))
        gradient.setColorAt(1, QColor(25, 35, 45))
        
        painter.fillRect(self.rect(), gradient)


# For testing
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = ConversationPanel()
    window.resize(600, 800)
    window.show()
    sys.exit(app.exec_()) 