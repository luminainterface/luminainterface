#!/usr/bin/env python3
"""
Mistral Neural Chat Plugin for LUMINA V7 Template UI
This plugin integrates the Mistral AI chat capabilities with neural network enhancements
"""

import os
import sys
import json
import logging
import datetime
import threading
import re
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

from PySide6.QtCore import Qt, Signal, Slot, QObject, QSize, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QLineEdit, QTextEdit, QScrollArea, QFrame, QSlider,
    QFileDialog, QMessageBox, QSplitter, QSpacerItem, 
    QSizePolicy, QComboBox, QInputDialog, QTabWidget,
    QProgressBar, QCheckBox
)
from PySide6.QtGui import QFont, QColor, QPixmap, QIcon

# Import Mistral integration 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from src.v7.mistral_integration import MistralIntegration
    from src.v7.onsite_memory import OnsiteMemory
    mistral_import_success = True
except ImportError as e:
    logging.error(f"Failed to import Mistral integration: {e}")
    mistral_import_success = False

# Set up logging
logger = logging.getLogger("MistralNeuralChat")
logger.setLevel(logging.INFO)

# Wiki data constants
WIKI_API_ENDPOINT = "https://en.wikipedia.org/w/api.php"
DEFAULT_WIKI_TOPICS = [
    "artificial intelligence", "neural networks", "machine learning",
    "consciousness", "cognition", "linguistics", "natural language processing",
    "deep learning", "reinforcement learning", "language models"
]

class ChatMessageWidget(QFrame):
    """Widget to display a single chat message"""
    
    def __init__(self, text: str, sender: str, parent=None):
        super().__init__(parent)
        self.sender = sender
        self.timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Set frame style based on sender
        if sender.lower() == "user":
            self.setStyleSheet("""
                QFrame {
                    background-color: #e1f5fe;
                    border-radius: 10px;
                    border: 1px solid #b3e5fc;
                    margin: 5px;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    background-color: #f1f8e9;
                    border-radius: 10px;
                    border: 1px solid #dcedc8;
                    margin: 5px;
                }
            """)
            
        # Create layout
        layout = QVBoxLayout(self)
        
        # Add header with sender and timestamp
        header_layout = QHBoxLayout()
        sender_label = QLabel(f"<b>{sender}</b>")
        time_label = QLabel(self.timestamp)
        time_label.setStyleSheet("color: #757575; font-size: 10px;")
        time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        header_layout.addWidget(sender_label)
        header_layout.addWidget(time_label)
        
        # Add message text
        message_label = QLabel(text)
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        layout.addLayout(header_layout)
        layout.addWidget(message_label)
        self.setLayout(layout)


class WikiReaderWidget(QWidget):
    """Widget for auto-reading wikis and learning from them"""
    
    progress_updated = Signal(int, str)  # progress value, status message
    knowledge_added = Signal(str, str)   # topic, summary
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.topics = DEFAULT_WIKI_TOPICS.copy()
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the wiki reader UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("<h2>Wikipedia Knowledge Reader</h2>")
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Automatically read Wikipedia articles and add them to the knowledge base")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Topics editor
        topics_layout = QVBoxLayout()
        topics_label = QLabel("Topics to read (one per line):")
        self.topics_edit = QTextEdit()
        self.topics_edit.setPlainText("\n".join(self.topics))
        self.topics_edit.setMaximumHeight(150)
        
        topics_layout.addWidget(topics_label)
        topics_layout.addWidget(self.topics_edit)
        layout.addLayout(topics_layout)
        
        # Options
        options_layout = QHBoxLayout()
        self.auto_summarize = QCheckBox("Auto-summarize articles")
        self.auto_summarize.setChecked(True)
        self.add_to_memory = QCheckBox("Add to onsite memory")
        self.add_to_memory.setChecked(True)
        self.add_to_autowiki = QCheckBox("Add to Mistral autowiki")
        self.add_to_autowiki.setChecked(True)
        
        options_layout.addWidget(self.auto_summarize)
        options_layout.addWidget(self.add_to_memory)
        options_layout.addWidget(self.add_to_autowiki)
        layout.addLayout(options_layout)
        
        # Progress area
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.status_label = QLabel("Ready")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        layout.addLayout(progress_layout)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Reading")
        self.start_button.clicked.connect(self.start_reading)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_reading)
        self.stop_button.setEnabled(False)
        
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)
        layout.addLayout(buttons_layout)
        
        # Results area
        results_layout = QVBoxLayout()
        results_label = QLabel("Knowledge Added:")
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        
        results_layout.addWidget(results_label)
        results_layout.addWidget(self.results_text)
        layout.addLayout(results_layout)
        
        # Connect signals
        self.progress_updated.connect(self.update_progress)
        self.knowledge_added.connect(self.add_knowledge_entry)
        
    @Slot()
    def start_reading(self):
        """Start the wiki reading process"""
        if self.running:
            return
            
        # Get topics from text edit
        text = self.topics_edit.toPlainText().strip()
        if text:
            self.topics = [topic.strip() for topic in text.split("\n") if topic.strip()]
        
        if not self.topics:
            QMessageBox.warning(self, "No Topics", "Please enter at least one topic to read")
            return
            
        # Start the process
        self.running = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting...")
        
        # Clear results
        self.results_text.clear()
        
        # Start in a thread
        threading.Thread(target=self.read_wiki_topics, daemon=True).start()
        
    @Slot()
    def stop_reading(self):
        """Stop the wiki reading process"""
        self.running = False
        self.status_label.setText("Stopping...")
        
    def read_wiki_topics(self):
        """Read all wiki topics in a background thread"""
        total_topics = len(self.topics)
        
        for i, topic in enumerate(self.topics):
            if not self.running:
                self.progress_updated.emit(100, "Stopped")
                break
                
            progress = int((i / total_topics) * 100)
            self.progress_updated.emit(progress, f"Reading: {topic}")
            
            try:
                # Get wiki content
                summary, content = self.get_wiki_content(topic)
                
                if summary and content:
                    self.knowledge_added.emit(topic, summary)
                    
                    # Wait a bit to not overload the API
                    for _ in range(10):
                        if not self.running:
                            break
                        threading.Event().wait(0.1)
            except Exception as e:
                logger.error(f"Error reading wiki for {topic}: {e}")
                
        # Complete
        if self.running:
            self.progress_updated.emit(100, "Complete")
            self.running = False
            
    def get_wiki_content(self, topic):
        """Get content from Wikipedia for a topic"""
        try:
            # Search for the topic
            search_params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": topic,
                "srprop": "snippet",
                "srlimit": 1
            }
            
            search_response = requests.get(WIKI_API_ENDPOINT, params=search_params)
            search_data = search_response.json()
            
            if "query" in search_data and "search" in search_data["query"] and search_data["query"]["search"]:
                page_id = search_data["query"]["search"][0]["pageid"]
                
                # Get summary and content
                content_params = {
                    "action": "query",
                    "format": "json",
                    "prop": "extracts",
                    "exintro": 1,
                    "explaintext": 1,
                    "pageids": page_id
                }
                
                content_response = requests.get(WIKI_API_ENDPOINT, params=content_params)
                content_data = content_response.json()
                
                # Get full content
                full_content_params = {
                    "action": "query",
                    "format": "json",
                    "prop": "extracts",
                    "explaintext": 1,
                    "pageids": page_id
                }
                
                full_content_response = requests.get(WIKI_API_ENDPOINT, params=full_content_params)
                full_content_data = full_content_response.json()
                
                if "query" in content_data and "pages" in content_data["query"]:
                    summary = content_data["query"]["pages"][str(page_id)]["extract"]
                    full_content = full_content_data["query"]["pages"][str(page_id)]["extract"]
                    
                    return summary, full_content
            
            return None, None
        except Exception as e:
            logger.error(f"Error getting wiki content: {e}")
            return None, None
    
    @Slot(int, str)
    def update_progress(self, value, status):
        """Update the progress bar and status label"""
        self.progress_bar.setValue(value)
        self.status_label.setText(status)
        
        if value >= 100:
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
    
    @Slot(str, str)
    def add_knowledge_entry(self, topic, summary):
        """Add a knowledge entry to the results"""
        self.results_text.append(f"<b>{topic}</b>: {summary[:100]}...<br><br>")
        
        # Signal to parent that knowledge was added
        parent = self.parent()
        if parent and hasattr(parent, "add_knowledge"):
            parent.add_knowledge(topic, summary)


class Plugin(QObject):
    """Mistral Neural Chat Plugin for Template UI"""
    
    # Define signals for communication
    message_received = Signal(str, str)  # message, sender
    status_update = Signal(str)
    wiki_knowledge_added = Signal(str, dict)  # topic, knowledge data
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = "Mistral Neural Chat"
        self.version = "0.2.0"
        self.description = "Integration with Mistral AI and neural network for chat"
        
        self.mistral = None
        self.memory = None
        self.api_key = None
        self.model = "mistral-medium"
        self.llm_weight = 0.65
        self.nn_weight = 0.50
        self.chat_history = []
        self.main_widget = None
        self.wiki_reader = None
        self.consciousness_plugin = None
        
        # Initialize if imports succeeded
        if mistral_import_success:
            self._initialize_components()
            
        # Set up auto-save timer
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.save_state)
        self.auto_save_timer.start(300000)  # Save every 5 minutes
        
    def _initialize_components(self):
        """Initialize Mistral and memory components"""
        try:
            # Try to load API key from config
            config_path = Path("config/mistral_config.json")
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                    self.api_key = config.get("api_key")
                    self.model = config.get("model", self.model)
            
            # Initialize memory
            memory_path = Path("data/chat_memory")
            memory_path.mkdir(parents=True, exist_ok=True)
            self.memory = OnsiteMemory(data_dir=str(memory_path))
            
            # Initialize Mistral if API key is available
            if self.api_key:
                self.mistral = MistralIntegration(
                    api_key=self.api_key,
                    model=self.model,
                    learning_enabled=True
                )
                logger.info(f"Mistral integration initialized with model: {self.model}")
                return True
            else:
                logger.warning("No Mistral API key found in config")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False
    
    def setup_ui(self, main_ui):
        """Set up the plugin UI in the main template"""
        try:
            # Create main widget with tabs
            self.main_widget = QTabWidget()
            
            # Create chat tab
            chat_tab = QWidget()
            self._setup_chat_tab(chat_tab)
            self.main_widget.addTab(chat_tab, "Neural Chat")
            
            # Create wiki reader tab
            wiki_tab = QWidget()
            self._setup_wiki_tab(wiki_tab)
            self.main_widget.addTab(wiki_tab, "Knowledge Reader")
            
            # Add to main UI
            if hasattr(main_ui, "add_tab"):
                main_ui.add_tab(self.main_widget, "Mistral Neural Chat")
            
            # Try to find consciousness plugin if available
            if hasattr(main_ui, "plugins"):
                for plugin in main_ui.plugins:
                    if hasattr(plugin, "name") and "consciousness" in plugin.name.lower():
                        self.consciousness_plugin = plugin
                        logger.info(f"Found consciousness plugin: {plugin.name}")
                        break
                
        except Exception as e:
            logger.error(f"Error setting up UI: {e}")
            
    def _setup_chat_tab(self, parent_widget):
        """Set up the chat interface tab"""
        main_layout = QVBoxLayout(parent_widget)
            
        # Create chat display area
        self.chat_scroll_area = QScrollArea()
        self.chat_scroll_area.setWidgetResizable(True)
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_scroll_area.setWidget(self.chat_container)
        
        # Create input area
        input_layout = QHBoxLayout()
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Type a message...")
        self.message_input.returnPressed.connect(self.send_message)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.message_input, 7)
        input_layout.addWidget(self.send_button, 1)
        
        # Create settings area
        settings_layout = QHBoxLayout()
        
        # LLM weight slider
        llm_layout = QVBoxLayout()
        llm_label = QLabel("LLM Weight:")
        self.llm_slider = QSlider(Qt.Horizontal)
        self.llm_slider.setMinimum(0)
        self.llm_slider.setMaximum(100)
        self.llm_slider.setValue(int(self.llm_weight * 100))
        self.llm_slider.valueChanged.connect(self.update_llm_weight)
        self.llm_value_label = QLabel(f"{self.llm_weight:.2f}")
        
        llm_layout.addWidget(llm_label)
        llm_layout.addWidget(self.llm_slider)
        llm_layout.addWidget(self.llm_value_label)
        
        # NN weight slider
        nn_layout = QVBoxLayout()
        nn_label = QLabel("NN Weight:")
        self.nn_slider = QSlider(Qt.Horizontal)
        self.nn_slider.setMinimum(0)
        self.nn_slider.setMaximum(100)
        self.nn_slider.setValue(int(self.nn_weight * 100))
        self.nn_slider.valueChanged.connect(self.update_nn_weight)
        self.nn_value_label = QLabel(f"{self.nn_weight:.2f}")
        
        nn_layout.addWidget(nn_label)
        nn_layout.addWidget(self.nn_slider)
        nn_layout.addWidget(self.nn_value_label)
        
        # Model selection
        model_layout = QVBoxLayout()
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "mistral-tiny", 
            "mistral-small", 
            "mistral-medium", 
            "mistral-large-latest"
        ])
        index = self.model_combo.findText(self.model)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        self.model_combo.currentTextChanged.connect(self.update_model)
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        
        # API key button
        api_layout = QVBoxLayout()
        api_label = QLabel("API Key:")
        self.api_button = QPushButton("Set API Key")
        self.api_button.clicked.connect(self.set_api_key)
        self.api_status = QLabel("Not Set" if not self.api_key else "Set")
        
        api_layout.addWidget(api_label)
        api_layout.addWidget(self.api_button)
        api_layout.addWidget(self.api_status)
        
        # Add all layouts to settings
        settings_layout.addLayout(llm_layout)
        settings_layout.addLayout(nn_layout)
        settings_layout.addLayout(model_layout)
        settings_layout.addLayout(api_layout)
        
        # Add all components to main layout
        main_layout.addWidget(self.chat_scroll_area, 7)
        main_layout.addLayout(input_layout, 1)
        main_layout.addLayout(settings_layout, 1)
        
        # Add welcome message
        self.add_message("Welcome to Mistral Neural Chat! How can I assist you today?", "Assistant")
        
        # Connect signals
        self.message_received.connect(self.add_message)
        
    def _setup_wiki_tab(self, parent_widget):
        """Set up the wiki reader tab"""
        layout = QVBoxLayout(parent_widget)
        
        self.wiki_reader = WikiReaderWidget()
        layout.addWidget(self.wiki_reader)
        
        # Connect signals
        self.wiki_reader.knowledge_added.connect(self.process_wiki_knowledge)
        
    @Slot(str, str)
    def process_wiki_knowledge(self, topic, summary):
        """Process wiki knowledge when added from the reader"""
        if not summary:
            return
            
        # Add to onsite memory if enabled
        if self.wiki_reader.add_to_memory.isChecked() and self.memory:
            self.memory.add_knowledge(topic, {"summary": summary, "source": "wikipedia"})
            logger.info(f"Added knowledge to memory: {topic}")
            
        # Add to mistral autowiki if enabled
        if self.wiki_reader.add_to_autowiki.isChecked() and self.mistral:
            if hasattr(self.mistral, "add_autowiki_entry"):
                try:
                    self.mistral.add_autowiki_entry(topic, summary)
                    logger.info(f"Added to Mistral autowiki: {topic}")
                except Exception as e:
                    logger.error(f"Error adding to autowiki: {e}")
                    
        # Add to consciousness system if available
        if self.consciousness_plugin:
            try:
                if hasattr(self.consciousness_plugin, "process_knowledge"):
                    self.consciousness_plugin.process_knowledge(topic, summary)
                    logger.info(f"Sent knowledge to consciousness system: {topic}")
            except Exception as e:
                logger.error(f"Error sending to consciousness system: {e}")
                
        # Emit signal for other plugins
        self.wiki_knowledge_added.emit(topic, {
            "summary": summary,
            "source": "wikipedia",
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    @Slot()
    def send_message(self):
        """Send a message from the input field"""
        message = self.message_input.text().strip()
        if not message:
            return
            
        # Clear input
        self.message_input.clear()
        
        # Add user message to chat
        self.add_message(message, "User")
        
        # Initialize mistral if needed
        if not self.mistral and self.api_key:
            success = self._initialize_components()
            if not success:
                self.add_message("Could not initialize Mistral. Please check your API key.", "System")
                return
        
        # Process message
        if self.mistral:
            try:
                # Add to memory
                if self.memory:
                    self.memory.add_conversation({"role": "user", "content": message})
                
                # Get consciousness context if available
                consciousness_context = None
                if self.consciousness_plugin:
                    try:
                        if hasattr(self.consciousness_plugin, "get_consciousness_context"):
                            consciousness_context = self.consciousness_plugin.get_consciousness_context()
                    except Exception as e:
                        logger.error(f"Error getting consciousness context: {e}")
                
                # Process with mistral
                response = self.mistral.process_message(
                    message, 
                    nn_weight=self.nn_weight,
                    context=consciousness_context
                )
                
                # Add to memory
                if self.memory and response:
                    self.memory.add_conversation({"role": "assistant", "content": response})
                
                # Update consciousness if available
                if self.consciousness_plugin and response:
                    try:
                        if hasattr(self.consciousness_plugin, "update_from_interaction"):
                            self.consciousness_plugin.update_from_interaction(message, response)
                    except Exception as e:
                        logger.error(f"Error updating consciousness: {e}")
                
                # Display response
                if response:
                    self.add_message(response, "Assistant")
                else:
                    self.add_message("No response received from Mistral", "System")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                self.add_message(f"Error: {str(e)}", "System")
        else:
            self.add_message("Please set your Mistral API key first.", "System")
    
    @Slot(str, str)
    def add_message(self, message: str, sender: str):
        """Add a message to the chat display"""
        # Create message widget
        message_widget = ChatMessageWidget(message, sender)
        self.chat_layout.addWidget(message_widget)
        
        # Add to history
        self.chat_history.append({"role": sender.lower(), "content": message})
        
        # Scroll to bottom
        self.chat_scroll_area.verticalScrollBar().setValue(
            self.chat_scroll_area.verticalScrollBar().maximum()
        )
    
    @Slot(int)
    def update_llm_weight(self, value: int):
        """Update the LLM weight from slider"""
        self.llm_weight = value / 100.0
        self.llm_value_label.setText(f"{self.llm_weight:.2f}")
        logger.info(f"LLM weight updated to {self.llm_weight}")
        
        # Update in mistral if available
        if self.mistral:
            # Note: there's no direct LLM weight in MistralIntegration
            # This would be handled by the developer based on actual implementation
            pass
    
    @Slot(int)
    def update_nn_weight(self, value: int):
        """Update the NN weight from slider"""
        self.nn_weight = value / 100.0
        self.nn_value_label.setText(f"{self.nn_weight:.2f}")
        logger.info(f"NN weight updated to {self.nn_weight}")
    
    @Slot(str)
    def update_model(self, model: str):
        """Update the Mistral model"""
        self.model = model
        logger.info(f"Model updated to {self.model}")
        
        # Reinitialize mistral if it exists
        if self.mistral:
            try:
                self.mistral = MistralIntegration(
                    api_key=self.api_key,
                    model=self.model,
                    learning_enabled=True
                )
                self.add_message(f"Model changed to {self.model}", "System")
            except Exception as e:
                logger.error(f"Error updating model: {e}")
                self.add_message(f"Error changing model: {str(e)}", "System")
    
    @Slot()
    def set_api_key(self):
        """Set the Mistral API key"""
        # Simple input dialog for API key
        api_key, ok = QInputDialog.getText(
            self.main_widget, 
            "Mistral API Key", 
            "Enter your Mistral API key:",
            QLineEdit.Password
        )
        
        if ok and api_key.strip():
            self.api_key = api_key.strip()
            self.api_status.setText("Set")
            
            # Save to config
            config_dir = Path("config")
            config_dir.mkdir(exist_ok=True)
            
            config = {
                "api_key": self.api_key,
                "model": self.model
            }
            
            with open(config_dir / "mistral_config.json", "w") as f:
                json.dump(config, f)
            
            # Initialize mistral
            self._initialize_components()
            self.add_message("API key set successfully", "System")
    
    def add_knowledge(self, topic, content):
        """Add knowledge to Mistral and memory systems"""
        if self.mistral and hasattr(self.mistral, "add_autowiki_entry"):
            try:
                self.mistral.add_autowiki_entry(topic, content)
                logger.info(f"Added knowledge to Mistral autowiki: {topic}")
            except Exception as e:
                logger.error(f"Error adding knowledge to Mistral: {e}")
                
        if self.memory:
            try:
                self.memory.add_knowledge(topic, {"content": content, "source": "manual"})
                logger.info(f"Added knowledge to memory: {topic}")
            except Exception as e:
                logger.error(f"Error adding knowledge to memory: {e}")
        
    def save_state(self):
        """Save plugin state"""
        logger.info("Saving Mistral Neural Chat plugin state")
        if self.memory:
            try:
                self.memory.save()
                logger.info("Memory saved successfully")
            except Exception as e:
                logger.error(f"Error saving memory: {e}")
        
        if self.mistral and hasattr(self.mistral, "save_learning_dictionary"):
            try:
                self.mistral.save_learning_dictionary()
                logger.info("Mistral learning dictionary saved successfully")
            except Exception as e:
                logger.error(f"Error saving Mistral learning dictionary: {e}")
                
    def shutdown(self):
        """Clean up resources before shutdown"""
        self.save_state()
        
        # Stop any running processes
        if self.wiki_reader and self.wiki_reader.running:
            self.wiki_reader.running = False 