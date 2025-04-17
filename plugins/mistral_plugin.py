"""
Mistral Integration Plugin for V7 Template

Connects the V7 PySide6 template with the Enhanced Language Mistral Integration system.
"""

import os
import sys
import threading
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
    QLineEdit, QPushButton, QLabel, QComboBox, QSlider, QCheckBox,
    QSplitter, QFrame, QProgressBar
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QIcon

# Import the plugin interface
try:
    from v7_pyside6_template import PluginInterface
except ImportError:
    # For development/testing
    class PluginInterface:
        def __init__(self, app_context):
            self.app_context = app_context

# Set up logging
logger = logging.getLogger("MistralPlugin")

# Try to import the Mistral integration
try:
    # Add src path to system path if needed
    if os.path.exists("src"):
        sys.path.insert(0, os.path.abspath("src"))
    
    # Import V7 Mistral integration components
    from src.v7.enhanced_language_mistral_integration import (
        get_enhanced_language_integration,
        EnhancedLanguageMistralIntegration
    )
    MISTRAL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Mistral integration import error: {e}")
    MISTRAL_AVAILABLE = False

class MistralChatWidget(QWidget):
    """Chat widget for interacting with Mistral AI"""
    
    message_sent = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.chat_history = []
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the chat UI components"""
        layout = QVBoxLayout(self)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        font = QFont("Arial", 10)
        self.chat_display.setFont(font)
        layout.addWidget(self.chat_display)
        
        # Status display
        self.status_bar = QProgressBar()
        self.status_bar.setTextVisible(True)
        self.status_bar.setRange(0, 100)
        self.status_bar.setValue(0)
        self.status_bar.setFormat("Ready")
        layout.addWidget(self.status_bar)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Type your message...")
        self.message_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.message_input)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        
        layout.addLayout(input_layout)
    
    def send_message(self):
        """Send a message to Mistral"""
        message = self.message_input.text().strip()
        if not message:
            return
        
        # Clear input
        self.message_input.clear()
        
        # Add to display
        self.chat_display.append(f"<b>You:</b> {message}")
        
        # Emit the message for processing
        self.message_sent.emit(message)
    
    def add_response(self, text, metrics=None):
        """Add a response from Mistral to the chat display"""
        # Add the response
        self.chat_display.append(f"<b>Mistral:</b> {text}")
        
        # Add metrics if available
        if metrics:
            metrics_text = "<i>"
            if "consciousness_level" in metrics:
                metrics_text += f"Consciousness: {metrics['consciousness_level']:.2f} "
            if "neural_linguistic_score" in metrics:
                metrics_text += f"Neural Score: {metrics['neural_linguistic_score']:.2f} "
            metrics_text += "</i>"
            self.chat_display.append(metrics_text)
        
        # Scroll to bottom
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )
    
    def set_status(self, message, progress=None):
        """Update the status bar with a message and optional progress"""
        self.status_bar.setFormat(message)
        if progress is not None:
            self.status_bar.setValue(int(progress * 100))

class MistralConfigWidget(QWidget):
    """Configuration widget for Mistral settings"""
    
    config_changed = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the configuration UI components"""
        layout = QVBoxLayout(self)
        
        # API key section
        api_layout = QHBoxLayout()
        api_layout.addWidget(QLabel("API Key:"))
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Enter Mistral API key")
        api_layout.addWidget(self.api_key_input)
        
        self.save_api_button = QPushButton("Save")
        self.save_api_button.clicked.connect(self.save_api_key)
        api_layout.addWidget(self.save_api_button)
        
        layout.addLayout(api_layout)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "mistral-tiny", 
            "mistral-small", 
            "mistral-medium", 
            "mistral-large"
        ])
        self.model_combo.setCurrentText("mistral-medium")
        self.model_combo.currentTextChanged.connect(self.update_config)
        model_layout.addWidget(self.model_combo)
        
        layout.addLayout(model_layout)
        
        # Learning mode
        self.learning_checkbox = QCheckBox("Enable Learning")
        self.learning_checkbox.setChecked(True)
        self.learning_checkbox.stateChanged.connect(self.update_config)
        layout.addWidget(self.learning_checkbox)
        
        # Weight sliders
        weights_layout = QVBoxLayout()
        weights_layout.addWidget(QLabel("Model Weights:"))
        
        # LLM weight slider
        llm_layout = QHBoxLayout()
        llm_layout.addWidget(QLabel("LLM:"))
        self.llm_slider = QSlider(Qt.Horizontal)
        self.llm_slider.setRange(1, 100)
        self.llm_slider.setValue(70)  # Default 0.7
        self.llm_slider.valueChanged.connect(self.update_config)
        llm_layout.addWidget(self.llm_slider)
        self.llm_value_label = QLabel("0.70")
        llm_layout.addWidget(self.llm_value_label)
        weights_layout.addLayout(llm_layout)
        
        # NN weight slider
        nn_layout = QHBoxLayout()
        nn_layout.addWidget(QLabel("NN:"))
        self.nn_slider = QSlider(Qt.Horizontal)
        self.nn_slider.setRange(1, 100)
        self.nn_slider.setValue(60)  # Default 0.6
        self.nn_slider.valueChanged.connect(self.update_config)
        nn_layout.addWidget(self.nn_slider)
        self.nn_value_label = QLabel("0.60")
        nn_layout.addWidget(self.nn_value_label)
        weights_layout.addLayout(nn_layout)
        
        layout.addLayout(weights_layout)
        
        # Status
        self.status_label = QLabel("Status: Not connected")
        layout.addWidget(self.status_label)
        
        # Apply button
        self.apply_button = QPushButton("Apply Configuration")
        self.apply_button.clicked.connect(self.apply_config)
        layout.addWidget(self.apply_button)
        
        # Add spacer
        layout.addStretch(1)
    
    def save_api_key(self):
        """Save the API key"""
        api_key = self.api_key_input.text().strip()
        if not api_key:
            return
        
        # Save to environment variable
        os.environ["MISTRAL_API_KEY"] = api_key
        
        # Update config
        self.update_config()
        
        # Clear input and show success
        self.api_key_input.clear()
        self.status_label.setText("Status: API key saved")
    
    def update_config(self):
        """Update configuration values based on UI state"""
        # Update slider labels
        llm_value = self.llm_slider.value() / 100.0
        nn_value = self.nn_slider.value() / 100.0
        
        self.llm_value_label.setText(f"{llm_value:.2f}")
        self.nn_value_label.setText(f"{nn_value:.2f}")
    
    def apply_config(self):
        """Apply the current configuration"""
        config = {
            "model": self.model_combo.currentText(),
            "learning": self.learning_checkbox.isChecked(),
            "llm_weight": self.llm_slider.value() / 100.0,
            "nn_weight": self.nn_slider.value() / 100.0,
            "api_key": os.environ.get("MISTRAL_API_KEY", "")
        }
        
        # Emit the config change signal
        self.config_changed.emit(config)
        
        # Update status
        self.status_label.setText(f"Status: Configuration applied - {config['model']}")

class MistralStatsWidget(QWidget):
    """Widget for displaying Mistral system statistics"""
    
    refresh_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.stats = {}
        self.setup_ui()
        
        # Set up timer for auto-refresh
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.request_refresh)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds
    
    def setup_ui(self):
        """Set up the statistics UI components"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Mistral System Statistics")
        title.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 12, QFont.Bold)
        title.setFont(font)
        layout.addWidget(title)
        
        # Stats display
        self.stats_display = QTextEdit()
        self.stats_display.setReadOnly(True)
        layout.addWidget(self.stats_display)
        
        # Refresh button
        self.refresh_button = QPushButton("Refresh Now")
        self.refresh_button.clicked.connect(self.request_refresh)
        layout.addWidget(self.refresh_button)
    
    def request_refresh(self):
        """Request a statistics refresh"""
        self.refresh_requested.emit()
    
    def update_stats(self, stats):
        """Update the statistics display"""
        self.stats = stats
        
        # Format stats as HTML
        html = "<style>table {width: 100%;} th {text-align: left;} td {padding: 2px;}</style>"
        html += "<table>"
        
        for key, value in self.stats.items():
            html += f"<tr><th>{key}</th><td>{value}</td></tr>"
        
        html += "</table>"
        
        # Update display
        self.stats_display.setHtml(html)

class Plugin(PluginInterface):
    """
    Mistral Integration Plugin
    
    Connects the V7 PySide6 template with the Enhanced Language Mistral Integration.
    """
    
    def __init__(self, app_context):
        super().__init__(app_context)
        self.name = "Mistral Integration"
        self.version = "1.0.0"
        self.author = "LUMINA"
        self.dependencies = []
        
        # Integration instance
        self.mistral_integration = None
        
        # Component status
        self.status = {
            "connected": False,
            "model": None,
            "learning_enabled": False
        }
        
        # Setup UI components
        self.setup_ui()
    
    def setup_ui(self):
        """Set up UI components for this plugin"""
        # Chat widget
        self.chat_widget = MistralChatWidget()
        self.chat_widget.message_sent.connect(self.process_message)
        
        # Create chat dock widget
        self.chat_dock = QDockWidget("Mistral Chat")
        self.chat_dock.setWidget(self.chat_widget)
        
        # Config widget
        self.config_widget = MistralConfigWidget()
        self.config_widget.config_changed.connect(self.apply_config)
        
        # Create config dock widget
        self.config_dock = QDockWidget("Mistral Configuration")
        self.config_dock.setWidget(self.config_widget)
        
        # Stats widget
        self.stats_widget = MistralStatsWidget()
        self.stats_widget.refresh_requested.connect(self.refresh_stats)
        
        # Create stats dock widget
        self.stats_dock = QDockWidget("Mistral Statistics")
        self.stats_dock.setWidget(self.stats_widget)
    
    def initialize(self) -> bool:
        """Initialize the Mistral integration"""
        if not MISTRAL_AVAILABLE:
            logger.warning("Mistral integration not available")
            self.chat_widget.add_response(
                "Mistral integration is not available. Please check your installation.",
                {}
            )
            return False
        
        # Initialize with default or environment settings
        api_key = os.environ.get("MISTRAL_API_KEY", "")
        model = "mistral-medium"
        
        try:
            # Initialize integration
            self.chat_widget.set_status("Initializing Mistral integration...", 0.2)
            
            # Initialize Mistral integration with mock mode if no API key
            self.mistral_integration = get_enhanced_language_integration(
                api_key=api_key,
                model=model,
                mock_mode=(not api_key)
            )
            
            # Set initial weights
            if hasattr(self.mistral_integration, "set_llm_weight"):
                self.mistral_integration.set_llm_weight(0.7)
            
            if hasattr(self.mistral_integration, "set_nn_weight"):
                self.mistral_integration.set_nn_weight(0.6)
            
            # Update status
            self.status["connected"] = True
            self.status["model"] = model
            self.status["learning_enabled"] = True
            
            # Update UI
            self.chat_widget.set_status("Connected to Mistral", 1.0)
            
            # Get initial stats
            self.refresh_stats()
            
            # Register for events
            self.app_context["register_event_handler"]("consciousness_update", self.handle_consciousness_update)
            
            return True
        except Exception as e:
            logger.error(f"Error initializing Mistral integration: {e}")
            self.chat_widget.add_response(
                f"Error initializing Mistral integration: {e}",
                {}
            )
            self.chat_widget.set_status("Error connecting to Mistral", 0)
            return False
    
    def apply_config(self, config):
        """Apply configuration changes"""
        if not self.mistral_integration:
            logger.warning("Cannot apply config: Mistral integration not initialized")
            return
        
        try:
            # Update API key if provided and different
            if config["api_key"] and config["api_key"] != os.environ.get("MISTRAL_API_KEY", ""):
                # Reinitialize with new API key
                self.shutdown()
                self.mistral_integration = get_enhanced_language_integration(
                    api_key=config["api_key"],
                    model=config["model"],
                    mock_mode=(not config["api_key"])
                )
            
            # Update model if different
            if config["model"] != self.status["model"]:
                if hasattr(self.mistral_integration, "set_model"):
                    self.mistral_integration.set_model(config["model"])
                self.status["model"] = config["model"]
            
            # Update learning mode
            if config["learning"] != self.status["learning_enabled"]:
                if hasattr(self.mistral_integration, "set_learning_enabled"):
                    self.mistral_integration.set_learning_enabled(config["learning"])
                self.status["learning_enabled"] = config["learning"]
            
            # Update weights
            if hasattr(self.mistral_integration, "set_llm_weight"):
                self.mistral_integration.set_llm_weight(config["llm_weight"])
            
            if hasattr(self.mistral_integration, "set_nn_weight"):
                self.mistral_integration.set_nn_weight(config["nn_weight"])
            
            # Update environment variables for subprocess consistency
            os.environ["LLM_WEIGHT"] = str(config["llm_weight"])
            os.environ["NN_WEIGHT"] = str(config["nn_weight"])
            
            # Update stats
            self.refresh_stats()
            
            # Notify success
            self.chat_widget.set_status(f"Configuration updated: {config['model']}", 1.0)
            
        except Exception as e:
            logger.error(f"Error applying configuration: {e}")
            self.chat_widget.set_status(f"Error applying configuration: {e}", 0.5)
    
    def process_message(self, message):
        """Process a message from the chat widget"""
        if not self.mistral_integration:
            self.chat_widget.add_response(
                "Mistral integration is not available. Please initialize first.",
                {}
            )
            return
        
        try:
            # Update status
            self.chat_widget.set_status("Processing message...", 0.5)
            
            # Process message
            response = self.mistral_integration.process_text(message)
            
            # Extract metrics
            metrics = {}
            if isinstance(response, dict):
                # Check for consciousness metrics
                if "consciousness_level" in response:
                    metrics["consciousness_level"] = response["consciousness_level"]
                
                # Check for neural metrics
                if "neural_linguistic_score" in response:
                    metrics["neural_linguistic_score"] = response["neural_linguistic_score"]
                
                # Get text response
                text_response = response.get("response", str(response))
            else:
                text_response = str(response)
            
            # Add response to chat
            self.chat_widget.add_response(text_response, metrics)
            
            # Trigger event for other plugins
            self.app_context["trigger_event"](
                "mistral_response", 
                {
                    "query": message,
                    "response": text_response,
                    "metrics": metrics
                }
            )
            
            # Update status
            self.chat_widget.set_status("Ready", 1.0)
            
            # Update stats after processing
            self.refresh_stats()
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.chat_widget.add_response(
                f"Error processing message: {e}",
                {}
            )
            self.chat_widget.set_status("Error processing message", 0.3)
    
    def refresh_stats(self):
        """Refresh Mistral system statistics"""
        if not self.mistral_integration:
            return
        
        try:
            # Get system stats
            if hasattr(self.mistral_integration, "get_system_stats"):
                stats = self.mistral_integration.get_system_stats()
            elif hasattr(self.mistral_integration, "get_stats"):
                stats = self.mistral_integration.get_stats()
            else:
                stats = {
                    "Status": "Connected",
                    "Model": self.status["model"],
                    "Learning Enabled": self.status["learning_enabled"],
                    "Mock Mode": getattr(self.mistral_integration, "mock_mode", False)
                }
            
            # Format stats for display
            display_stats = {}
            
            # Add basic info
            display_stats["Status"] = "Connected"
            display_stats["Model"] = self.status["model"]
            display_stats["Learning Enabled"] = "Yes" if self.status["learning_enabled"] else "No"
            display_stats["Mock Mode"] = "Yes" if getattr(self.mistral_integration, "mock_mode", False) else "No"
            
            # Add system stats
            if isinstance(stats, dict):
                # Add consciousness metrics
                if "consciousness_level" in stats:
                    display_stats["Consciousness Level"] = f"{stats['consciousness_level']:.4f}"
                
                # Add neural metrics
                if "neural_linguistic_score" in stats:
                    display_stats["Neural Linguistic Score"] = f"{stats['neural_linguistic_score']:.4f}"
                
                # Add learning stats
                if "total_exchanges" in stats:
                    display_stats["Total Exchanges"] = stats["total_exchanges"]
                
                if "dictionary_entries" in stats:
                    display_stats["Dictionary Entries"] = stats["dictionary_entries"]
                
                if "learning_dictionary_size" in stats:
                    display_stats["Learning Dictionary Size"] = f"{stats['learning_dictionary_size']/1024:.2f} KB"
            
            # Update stats widget
            self.stats_widget.update_stats(display_stats)
            
        except Exception as e:
            logger.error(f"Error refreshing stats: {e}")
            self.stats_widget.update_stats({"Error": str(e)})
    
    def handle_consciousness_update(self, data):
        """Handle consciousness update event from other plugins"""
        if not isinstance(data, dict):
            return
        
        # Extract consciousness level
        consciousness_level = data.get("consciousness_level")
        if consciousness_level is None:
            return
        
        # Update Mistral integration if available
        if self.mistral_integration and hasattr(self.mistral_integration, "set_consciousness_level"):
            try:
                self.mistral_integration.set_consciousness_level(consciousness_level)
                self.chat_widget.set_status(f"Consciousness updated: {consciousness_level:.2f}", 0.7)
            except Exception as e:
                logger.error(f"Error updating consciousness level: {e}")
    
    def get_dock_widgets(self) -> List[QDockWidget]:
        """Return list of dock widgets provided by this plugin"""
        return [self.chat_dock, self.config_dock, self.stats_dock]
    
    def get_tab_widgets(self) -> List[tuple]:
        """Return list of (name, widget) tuples for tab widgets"""
        return [
            ("Mistral Chat", self.chat_widget),
            ("Mistral Config", self.config_widget),
            ("Mistral Stats", self.stats_widget)
        ]
    
    def shutdown(self) -> None:
        """Clean shutdown of the plugin"""
        if self.mistral_integration:
            logger.info("Shutting down Mistral integration")
            
            # Shut down Mistral integration
            if hasattr(self.mistral_integration, "shutdown"):
                self.mistral_integration.shutdown()
            elif hasattr(self.mistral_integration, "close"):
                self.mistral_integration.close()
            
            # Clear integration reference
            self.mistral_integration = None
            
            # Update status
            self.status["connected"] = False 