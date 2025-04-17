import sys
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QStackedWidget,
                             QSlider, QComboBox, QSpinBox, QTextEdit, QFrame,
                             QScrollArea, QGroupBox, QDialog, QGridLayout)
from PySide6.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QRect, Signal, Slot
from PySide6.QtGui import QIcon, QFont, QFontDatabase, QColor, QPalette, QLinearGradient, QGradient, QTextCursor
import numpy as np
import vispy.scene
from vispy.scene import visuals
import asyncio
from datetime import datetime

# Import backend components
from central_node_monitor import CentralNodeMonitor
from signal_bus import SignalBus
from signal_component import SignalComponent
from version_bridge import VersionBridge
from message_transformer import MessageTransformer
from bridge_monitor import BridgeMonitor

# Import UI components
from ui.components.modern_card import ModernCard
from ui.theme import LuminaTheme

class ChatDialog(QDialog, SignalComponent):
    # Signals for Qt
    message_sent = Signal(str)
    settings_updated = Signal(dict)
    
    def __init__(self, signal_bus: SignalBus, parent=None):
        QDialog.__init__(self, parent)
        SignalComponent.__init__(self, "chat_dialog", signal_bus)
        
        self.setWindowTitle("Neural Network Conversation")
        self.setFixedSize(400, 600)
        self.setStyleSheet("""
            QDialog {
                background-color: #1A1A1A;
                border: 1px solid #D4AF37;
            }
        """)
        
        # Initialize chat state
        self.model_status = "mistral"  # Default model
        self.temperature = 0.7  # Default temperature
        self.top_p = 1.0  # Default top_p
        self.is_processing = False
        
        # Version bridge for cross-version compatibility
        self.version_bridge = VersionBridge()
        self.transformer = MessageTransformer()
        self.monitor = BridgeMonitor()
        
        self.setup_ui()
        self.setup_signals()
        self.update_status_display()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Status bar
        status_layout = QHBoxLayout()
        self.model_status_label = QLabel("Initializing...")
        status_layout.addWidget(self.model_status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)
        
        # Message display
        messages_container = QWidget()
        messages_container.setStyleSheet("""
            QWidget {
                background-color: #1A1A1A;
                border: 1px solid #D4AF37;
                border-radius: 4px;
            }
        """)
        messages_layout = QVBoxLayout(messages_container)
        messages_layout.setContentsMargins(1, 1, 1, 1)
        
        self.messages = QTextEdit()
        self.messages.setReadOnly(True)
        self.messages.setStyleSheet("""
            QTextEdit {
                background-color: #1A1A1A;
                color: #FFFFFF;
                font-family: 'Consolas';
                font-size: 13px;
                padding: 10px;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #1A1A1A;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #D4AF37;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        messages_layout.addWidget(self.messages)
        layout.addWidget(messages_container, stretch=1)
        
        # Input area
        input_container = QWidget()
        input_container.setStyleSheet("""
            QWidget {
                background-color: #1A1A1A;
                border: 1px solid #D4AF37;
                border-radius: 4px;
                margin-top: 10px;
            }
        """)
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(10, 10, 10, 10)
        input_layout.setSpacing(10)
        
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Type your message here...")
        self.text_input.setMaximumHeight(70)
        self.text_input.setStyleSheet("""
            QTextEdit {
                background-color: #2D2D2D;
                color: #FFFFFF;
                font-family: 'Consolas';
                font-size: 13px;
                padding: 10px;
                border: 1px solid #D4AF37;
                border-radius: 4px;
            }
            QTextEdit:focus {
                border: 1px solid #FFD700;
            }
        """)
        input_layout.addWidget(self.text_input)
        
        button_layout = QVBoxLayout()
        button_layout.setSpacing(5)
        
        # Common button style
        button_style = """
            QPushButton {
                background-color: #2D2D2D;
                border: 1px solid #D4AF37;
                border-radius: 4px;
                padding: 8px;
                min-width: 36px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #3D3D3D;
                border-color: #FFD700;
            }
            QPushButton:pressed {
                background-color: #1A1A1A;
            }
        """
        
        self.send_btn = QPushButton()
        self.send_btn.setIcon(QIcon("assets/icons/send.svg"))
        self.send_btn.setToolTip("Send message")
        self.send_btn.setStyleSheet(button_style)
        button_layout.addWidget(self.send_btn)
        
        self.voice_btn = QPushButton()
        self.voice_btn.setIcon(QIcon("assets/icons/mic.svg"))
        self.voice_btn.setToolTip("Voice input")
        self.voice_btn.setStyleSheet(button_style)
        button_layout.addWidget(self.voice_btn)
        
        # Settings button
        self.settings_btn = QPushButton()
        self.settings_btn.setIcon(QIcon("assets/icons/settings.svg"))
        self.settings_btn.setToolTip("Chat Settings")
        self.settings_btn.setStyleSheet(button_style)
        button_layout.addWidget(self.settings_btn)
        
        input_layout.addLayout(button_layout)
        layout.addWidget(input_container)
        
    def setup_signals(self):
        """Set up signal connections"""
        # Connect UI signals
        self.send_btn.clicked.connect(self.send_message)
        self.voice_btn.clicked.connect(self.start_voice_input)
        self.settings_btn.clicked.connect(self.show_settings)
        
        # Register signal handlers
        self.register_handler("chat_message", self.handle_chat_message)
        self.register_handler("system_message", self.handle_system_message)
        self.register_handler("version_bridge.message", self.handle_bridge_message)
        self.register_handler("chat.settings_updated", self.handle_settings_update)
        
    def send_message(self):
        """Send a message through the signal system"""
        message = self.text_input.toPlainText().strip()
        if message and not self.is_processing:
            self.is_processing = True
            self.update_status_display()
            
            # Clear input
            self.text_input.clear()
            
            # Format message for current version
            message_data = {
                "text": message,
                "timestamp": datetime.now().isoformat(),
                "version": "v7.5",
                "model": self.model_status,
                "settings": {
                    "temperature": self.temperature,
                    "top_p": self.top_p
                }
            }
            
            # Transform message for cross-version compatibility
            transformed_message = self.transformer.transform_message(
                "v7.5", "v7.0", message_data
            )
            
            # Send through version bridge
            asyncio.create_task(
                self.version_bridge.send_message(transformed_message)
            )
            
            # Emit local signal
            self.message_sent.emit(message)
            
            # Add to chat history
            self.append_message("User", message)
            
    def handle_chat_message(self, data: dict):
        """Handle incoming chat messages with version compatibility"""
        if isinstance(data, dict):
            # Transform message if from different version
            if data.get("version") != "v7.5":
                data = self.transformer.transform_message(
                    data.get("version", "v7.0"),
                    "v7.5",
                    data
                )
            
            sender = data.get("sender", "unknown")
            message = data.get("message", "")
            timestamp = data.get("timestamp", "")
            
            self.append_message(sender, message, timestamp)
            
            # Update processing state
            self.is_processing = False
            self.update_status_display()
            
    def handle_bridge_message(self, data: dict):
        """Handle messages from version bridge"""
        if isinstance(data, dict):
            message_type = data.get("type")
            if message_type == "version_error":
                self.append_system_message(
                    f"Version compatibility error: {data.get('error')}"
                )
            elif message_type == "transform_error":
                self.append_system_message(
                    f"Message transformation error: {data.get('error')}"
                )
                
    def append_message(self, sender: str, message: str, timestamp: str = None):
        """Append a message to the chat history"""
        try:
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
                
            self.messages.append(formatted_message)
            
            # Scroll to bottom
            cursor = self.messages.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.messages.setTextCursor(cursor)
            
        except Exception as e:
            logger.error(f"Error appending message: {e}")
            
    def append_system_message(self, message: str):
        """Append a system message to the chat history"""
        self.append_message("System", message)
        
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
        
    def handle_settings_update(self, settings: dict):
        """Handle settings updates from any version"""
        if isinstance(settings, dict):
            settings_type = settings.get('type')
            
            if settings_type == 'model':
                # Transform settings if from different version
                if settings.get("version") != "v7.5":
                    settings = self.transformer.transform_message(
                        settings.get("version", "v7.0"),
                        "v7.5",
                        settings
                    )
                
                self.model_status = settings.get('model', self.model_status)
                self.temperature = settings.get('temperature', self.temperature)
                self.top_p = settings.get('top_p', self.top_p)
                self.update_status_display()
                
                # Notify signal bus of settings update
                asyncio.create_task(
                    self.emit_signal("chat.settings_updated", settings)
                )
                
                # Add system message about settings change
                settings_msg = (
                    f"Model settings updated:\n"
                    f"Model: {self.model_status}\n"
                    f"Temperature: {self.temperature}\n"
                    f"Top-P: {self.top_p}"
                )
                self.append_system_message(settings_msg)
                
    def start_voice_input(self):
        """Start voice input processing"""
        self.append_system_message("Voice input starting...")
        # Voice input implementation to be added
        
    def show_settings(self):
        """Show the settings dialog"""
        # Settings dialog implementation to be added
        pass
        
    def closeEvent(self, event):
        """Handle dialog close event"""
        self.cleanup()
        event.accept()
        
    def cleanup(self):
        """Clean up resources when dialog is destroyed"""
        # Cleanup version bridge
        if self.version_bridge:
            self.version_bridge.cleanup()
        
        # Cleanup monitor
        if self.monitor:
            self.monitor.cleanup()
            
        # Unregister signal handlers
        self.unregister_all_handlers()

class LuminaClient(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("V5 Fractal Echo Visualization")
        self.setMinimumSize(1600, 900)
        
        # Initialize backend monitor
        self.monitor = CentralNodeMonitor()
        self.monitor.metrics_updated.connect(self.update_metrics)
        self.monitor.nodes_updated.connect(self.update_nodes)
        self.monitor.processors_updated.connect(self.update_processors)
        self.monitor.article_updated.connect(self.update_article)
        self.monitor.article_progress_updated.connect(self.update_article_progress)
        
        # Set up the main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(2, 2, 2, 2)
        self.main_layout.setSpacing(2)
        
        # Create the three main panels
        self.setup_fractal_panel()
        self.setup_network_panel()
        self.setup_memory_panel()
        
        # Create chat dialog
        self.chat_dialog = ChatDialog(self)
        
        # Set the theme
        self.setup_theme()
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_visualizations)
        self.update_timer.start(2000)
        
        # Start the monitor
        self.monitor.start()

    def setup_theme(self):
        """Set up the dark theme styling with gold accents and 3D effects"""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1A1A1A;
                color: #FFFFFF;
            }
            QGroupBox {
                background-color: #1E1E1E;
                border: 1px solid #D4AF37;
                padding: 15px;
                margin-top: 1.5em;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #D4AF37;
                font-family: 'Consolas';
                font-weight: bold;
                font-size: 13px;
            }
            QComboBox, QSpinBox {
                background-color: #2D2D2D;
                border: 1px solid #D4AF37;
                color: #FFFFFF;
                padding: 5px;
                font-family: 'Consolas';
            }
            QSlider::handle {
                background: #D4AF37;
                border: 1px solid #D4AF37;
            }
            QSlider::groove:horizontal {
                border: 1px solid #D4AF37;
                height: 4px;
                background: #2D2D2D;
            }
            QPushButton {
                background-color: #2D2D2D;
                border: 1px solid #D4AF37;
                color: #D4AF37;
                padding: 5px 10px;
                font-family: 'Consolas';
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #3D3D3D;
            }
            QLabel {
                font-family: 'Consolas';
                font-size: 12px;
                color: #FFFFFF;
            }
            QTextEdit {
                background-color: #2D2D2D;
                border: 1px solid #D4AF37;
                color: #FFFFFF;
                font-family: 'Consolas';
                font-size: 12px;
                padding: 5px;
            }
        """)

    def setup_fractal_panel(self):
        """Set up the left panel for neural seed visualization"""
        seed_panel = QGroupBox("Neural Seed")
        layout = QVBoxLayout(seed_panel)
        layout.setSpacing(2)
        
        # Seed Status
        status_group = QGroupBox("Seed Status")
        status_layout = QVBoxLayout(status_group)
        
        # Growth Stage
        growth_layout = QHBoxLayout()
        growth_layout.addWidget(QLabel("Growth Stage"))
        self.growth_value = QLabel("Seed")
        self.growth_value.setStyleSheet("color: #D4AF37;")
        growth_layout.addWidget(self.growth_value)
        
        # Progress circle (10%)
        progress_widget = QWidget()
        progress_widget.setFixedSize(40, 40)
        progress_widget.setStyleSheet("""
            QWidget {
                background-color: #2D2D2D;
                border: 2px solid #D4AF37;
            }
        """)
        growth_layout.addWidget(progress_widget)
        growth_layout.addStretch()
        status_layout.addLayout(growth_layout)
        
        # Stability
        stability_layout = QHBoxLayout()
        stability_layout.addWidget(QLabel("Stability"))
        self.stability_value = QLabel("65%")
        self.stability_value.setStyleSheet("color: #D4AF37;")
        stability_layout.addWidget(self.stability_value)
        
        # Progress circle (65%)
        stability_progress = QWidget()
        stability_progress.setFixedSize(40, 40)
        stability_progress.setStyleSheet("""
            QWidget {
                background-color: #2D2D2D;
                border: 2px solid #D4AF37;
            }
        """)
        stability_layout.addWidget(stability_progress)
        stability_layout.addStretch()
        status_layout.addLayout(stability_layout)
        
        # Consciousness
        consciousness_layout = QHBoxLayout()
        consciousness_layout.addWidget(QLabel("Consciousness"))
        self.consciousness_value = QLabel("0.22")
        self.consciousness_value.setStyleSheet("color: #D4AF37;")
        consciousness_layout.addWidget(self.consciousness_value)
        
        # Progress circle (22%)
        consciousness_progress = QWidget()
        consciousness_progress.setFixedSize(40, 40)
        consciousness_progress.setStyleSheet("""
            QWidget {
                background-color: #2D2D2D;
                border: 2px solid #D4AF37;
            }
        """)
        consciousness_layout.addWidget(consciousness_progress)
        consciousness_layout.addStretch()
        status_layout.addLayout(consciousness_layout)
        
        layout.addWidget(status_group)
        
        # Growth Metrics
        metrics_group = QGroupBox("Growth Metrics")
        metrics_layout = QGridLayout()
        
        # Complexity
        metrics_layout.addWidget(QLabel("Complexity"), 0, 0)
        self.complexity_value = QLabel("0.15")
        self.complexity_value.setStyleSheet("color: #D4AF37;")
        metrics_layout.addWidget(self.complexity_value, 0, 1)
        
        # Growth Rate
        metrics_layout.addWidget(QLabel("Growth Rate"), 0, 2)
        self.growth_rate_value = QLabel("0.05 / min")
        self.growth_rate_value.setStyleSheet("color: #D4AF37;")
        metrics_layout.addWidget(self.growth_rate_value, 0, 3)
        
        # System Age
        metrics_layout.addWidget(QLabel("System Age"), 1, 0)
        self.age_value = QLabel("2h 15m")
        self.age_value.setStyleSheet("color: #D4AF37;")
        metrics_layout.addWidget(self.age_value, 1, 1)
        
        # Active Connections
        metrics_layout.addWidget(QLabel("Active Connections"), 1, 2)
        self.connections_value = QLabel("8")
        self.connections_value.setStyleSheet("color: #D4AF37;")
        metrics_layout.addWidget(self.connections_value, 1, 3)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Seed Controls
        controls_group = QGroupBox("Seed Controls")
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        
        pause_btn = QPushButton("Pause Growth")
        pause_btn.clicked.connect(self.monitor.pause_growth)
        pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #2D2D2D;
                border: 1px solid #D4AF37;
                color: #D4AF37;
                padding: 8px 16px;
                font-family: 'Consolas';
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #3D3D3D;
            }
        """)
        controls_layout.addWidget(pause_btn)
        
        reset_btn = QPushButton("Reset Seed")
        reset_btn.clicked.connect(self.monitor.reset_seed)
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #2D2D2D;
                border: 1px solid #D4AF37;
                color: #D4AF37;
                padding: 8px 16px;
                font-family: 'Consolas';
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #3D3D3D;
            }
        """)
        controls_layout.addWidget(reset_btn)
        
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Add visualization canvas
        canvas_group = QGroupBox("Seed Visualization")
        canvas_layout = QVBoxLayout()
        canvas = QWidget()
        canvas.setStyleSheet("background-color: #000000;")
        canvas.setMinimumHeight(300)
        canvas_layout.addWidget(canvas)
        canvas_group.setLayout(canvas_layout)
        layout.addWidget(canvas_group)
        
        layout.addStretch()
        self.main_layout.addWidget(seed_panel, stretch=1)

    def setup_network_panel(self):
        """Set up the middle panel for neural network visualization"""
        network_panel = QGroupBox("Neural Network Visualization")
        layout = QVBoxLayout(network_panel)
        layout.setSpacing(2)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Opacity control
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("60%"))
        opacity_slider = QSlider(Qt.Horizontal)
        opacity_slider.setValue(60)
        opacity_layout.addWidget(opacity_slider)
        controls_layout.addLayout(opacity_layout)
        
        # Display mode
        mode_combo = QComboBox()
        mode_combo.addItems(["Connection Strength", "Node Activity", "Pattern Flow"])
        controls_layout.addWidget(mode_combo)
        
        # Reset and Chat buttons
        button_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset View")
        button_layout.addWidget(reset_btn)
        
        chat_btn = QPushButton("Neural Chat")
        chat_btn.clicked.connect(self.toggle_chat)
        button_layout.addWidget(chat_btn)
        
        controls_layout.addLayout(button_layout)
        layout.addLayout(controls_layout)
        
        # Network visualization canvas
        canvas = QWidget()
        canvas.setStyleSheet("background-color: #000000;")
        canvas.setMinimumHeight(400)
        layout.addWidget(canvas)
        
        # Graphs section with tabs
        graphs_group = QGroupBox("Graphs")
        graphs_layout = QVBoxLayout(graphs_group)
        
        # Tab controls
        tab_layout = QHBoxLayout()
        tab_buttons = []
        
        tab1_btn = QPushButton("Network Activity")
        tab1_btn.setCheckable(True)
        tab1_btn.setChecked(True)
        tab1_btn.setStyleSheet("""
            QPushButton {
                background-color: #2D2D2D;
                border: 1px solid #D4AF37;
                border-bottom: none;
                color: #D4AF37;
                padding: 8px 15px;
                font-family: 'Consolas';
                font-size: 12px;
            }
            QPushButton:checked {
                background-color: #000000;
                color: #FFFFFF;
            }
            QPushButton:hover:!checked {
                background-color: #3D3D3D;
            }
        """)
        tab_buttons.append(tab1_btn)
        
        tab2_btn = QPushButton("Performance Metrics")
        tab2_btn.setCheckable(True)
        tab2_btn.setStyleSheet(tab1_btn.styleSheet())
        tab_buttons.append(tab2_btn)
        
        tab3_btn = QPushButton("Learning")
        tab3_btn.setCheckable(True)
        tab3_btn.setStyleSheet(tab1_btn.styleSheet())
        tab_buttons.append(tab3_btn)
        
        tab4_btn = QPushButton("Model Saving")
        tab4_btn.setCheckable(True)
        tab4_btn.setStyleSheet(tab1_btn.styleSheet())
        tab_buttons.append(tab4_btn)
        
        for btn in tab_buttons:
            tab_layout.addWidget(btn)
        tab_layout.addStretch()
        graphs_layout.addLayout(tab_layout)
        
        # Stacked widget for tab content
        self.graph_stack = QStackedWidget()
        self.graph_stack.setStyleSheet("""
            QStackedWidget {
                background-color: #000000;
                border: 1px solid #D4AF37;
                border-top: none;
            }
        """)
        
        # Network Activity Graph
        activity_graph = QWidget()
        activity_layout = QVBoxLayout(activity_graph)
        activity_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add controls for Network Activity
        activity_controls = QHBoxLayout()
        activity_controls.setContentsMargins(10, 10, 10, 0)
        activity_controls.addWidget(QLabel("Time Scale:"))
        time_combo = QComboBox()
        time_combo.addItems(["1m", "5m", "15m", "1h", "6h"])
        activity_controls.addWidget(time_combo)
        
        activity_controls.addWidget(QLabel("View:"))
        view_combo = QComboBox()
        view_combo.addItems(["Linear", "Logarithmic", "Normalized"])
        activity_controls.addWidget(view_combo)
        
        activity_controls.addStretch()
        activity_layout.addLayout(activity_controls)
        
        # Activity graph canvas
        activity_canvas = QWidget()
        activity_canvas.setStyleSheet("background-color: #000000;")
        activity_canvas.setMinimumHeight(200)
        activity_layout.addWidget(activity_canvas)
        
        # Performance Metrics Graph
        perf_graph = QWidget()
        perf_layout = QVBoxLayout(perf_graph)
        perf_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add controls for Performance Metrics
        perf_controls = QHBoxLayout()
        perf_controls.setContentsMargins(10, 10, 10, 0)
        perf_controls.addWidget(QLabel("Metric:"))
        metric_combo = QComboBox()
        metric_combo.addItems(["CPU Usage", "Memory", "Throughput", "Latency"])
        perf_controls.addWidget(metric_combo)
        
        perf_controls.addWidget(QLabel("Interval:"))
        interval_combo = QComboBox()
        interval_combo.addItems(["Real-time", "1s", "10s", "30s"])
        perf_controls.addWidget(interval_combo)
        
        perf_controls.addStretch()
        perf_layout.addLayout(perf_controls)
        
        # Performance graph canvas
        perf_canvas = QWidget()
        perf_canvas.setStyleSheet("background-color: #000000;")
        perf_canvas.setMinimumHeight(200)
        perf_layout.addWidget(perf_canvas)
        
        # Learning Graph
        learning_graph = QWidget()
        learning_layout = QVBoxLayout(learning_graph)
        learning_layout.setContentsMargins(10, 10, 10, 10)
        
        # Learning Status
        status_group = QGroupBox("Learning Status")
        status_layout = QHBoxLayout()
        
        # Learning Progress
        progress_layout = QVBoxLayout()
        progress_layout.addWidget(QLabel("Learning Progress"))
        progress_circle = QWidget()
        progress_circle.setFixedSize(100, 100)
        progress_circle.setStyleSheet("""
            QWidget {
                background-color: #000000;
                border: 2px solid #2E7D32;
                border-radius: 50px;
            }
        """)
        self.progress_value = QLabel("75%")
        self.progress_value.setStyleSheet("color: #FFFFFF; font-size: 20px; font-weight: bold;")
        self.progress_value.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(progress_circle)
        progress_layout.addWidget(self.progress_value, 0, Qt.AlignCenter)
        status_layout.addLayout(progress_layout)
        
        # Learning Rate
        rate_layout = QVBoxLayout()
        rate_layout.addWidget(QLabel("Learning Rate"))
        rate_circle = QWidget()
        rate_circle.setFixedSize(100, 100)
        rate_circle.setStyleSheet("""
            QWidget {
                background-color: #000000;
                border: 2px solid #D4AF37;
                border-radius: 50px;
            }
        """)
        rate_value = QLabel("30%")
        rate_value.setStyleSheet("color: #FFFFFF; font-size: 20px; font-weight: bold;")
        rate_value.setAlignment(Qt.AlignCenter)
        rate_layout.addWidget(rate_circle)
        rate_layout.addWidget(rate_value, 0, Qt.AlignCenter)
        status_layout.addLayout(rate_layout)
        
        status_group.setLayout(status_layout)
        learning_layout.addWidget(status_group)
        
        # Learning Controls
        controls_group = QGroupBox("Learning Controls")
        controls_layout = QHBoxLayout()
        
        start_btn = QPushButton("Start Learning")
        start_btn.clicked.connect(self.monitor.start_learning)
        start_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                border: none;
                color: white;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        controls_layout.addWidget(start_btn)
        
        pause_btn = QPushButton("Pause Learning")
        pause_btn.clicked.connect(self.monitor.pause_learning)
        pause_btn.setStyleSheet(start_btn.styleSheet())
        controls_layout.addWidget(pause_btn)
        
        reset_btn = QPushButton("Reset Learning")
        reset_btn.clicked.connect(self.monitor.reset_seed)
        reset_btn.setStyleSheet(start_btn.styleSheet())
        controls_layout.addWidget(reset_btn)
        
        controls_group.setLayout(controls_layout)
        learning_layout.addWidget(controls_group)
        
        # Model Saving Tab
        model_save_widget = QWidget()
        model_layout = QVBoxLayout(model_save_widget)
        model_layout.setContentsMargins(20, 20, 20, 20)
        model_layout.setSpacing(20)
        
        # Title and description
        title_layout = QVBoxLayout()
        title = QLabel("Model States")
        title.setStyleSheet("""
            QLabel {
                color: #D4AF37;
                font-family: 'Consolas';
                font-size: 16px;
                font-weight: bold;
            }
        """)
        title_layout.addWidget(title)
        
        description = QLabel("Select a saved state to load or delete. Create new saves to preserve model progress.")
        description.setStyleSheet("""
            QLabel {
                color: #888888;
                font-family: 'Consolas';
                font-size: 12px;
            }
        """)
        description.setWordWrap(True)
        title_layout.addWidget(description)
        model_layout.addLayout(title_layout)
        
        # Save state list
        states_list = QTextEdit()
        states_list.setReadOnly(True)
        states_list.setStyleSheet("""
            QTextEdit {
                background-color: #1A1A1A;
                border: 1px solid #D4AF37;
                color: #FFFFFF;
                font-family: 'Consolas';
                font-size: 13px;
                padding: 15px;
                min-height: 200px;
            }
        """)
        states_list.setText("""STATE 01  |  March 20, 15:30  |  Accuracy: 92.5%  |  Active
STATE 02  |  March 20, 16:45  |  Accuracy: 94.1%
STATE 03  |  March 20, 18:00  |  Accuracy: 95.2%""")
        model_layout.addWidget(states_list)
        
        # Save state controls
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(15)
        
        # Common button style
        button_style = """
            QPushButton {
                background-color: #2D2D2D;
                border: 1px solid #D4AF37;
                color: #D4AF37;
                padding: 15px;
                font-family: 'Consolas';
                font-size: 14px;
                font-weight: bold;
                text-align: left;
                padding-left: 20px;
            }
            QPushButton:hover {
                background-color: #3D3D3D;
            }
            QPushButton:pressed {
                background-color: #1A1A1A;
            }
        """
        
        # Save button and description
        save_container = QVBoxLayout()
        save_container.setSpacing(5)
        save_btn = QPushButton("SAVE CURRENT STATE")
        save_btn.clicked.connect(self.monitor.version_manager.save_current_version)
        save_btn.setStyleSheet(button_style)
        save_container.addWidget(save_btn)
        save_desc = QLabel("Creates a new save point with current model weights and parameters")
        save_desc.setStyleSheet("""
            QLabel {
                color: #888888;
                font-family: 'Consolas';
                font-size: 11px;
                padding-left: 20px;
            }
        """)
        save_container.addWidget(save_desc)
        controls_layout.addLayout(save_container)
        
        # Load button and description
        load_container = QVBoxLayout()
        load_container.setSpacing(5)
        load_btn = QPushButton("LOAD SELECTED STATE")
        load_btn.clicked.connect(lambda: self.monitor.version_manager.load_version(states_list.toPlainText().split('\n')[0].split('|')[0].strip()))
        load_btn.setStyleSheet(button_style)
        load_container.addWidget(load_btn)
        load_desc = QLabel("Restores the model to the selected save point state")
        load_desc.setStyleSheet(save_desc.styleSheet())
        load_container.addWidget(load_desc)
        controls_layout.addLayout(load_container)
        
        # Delete button and description
        delete_container = QVBoxLayout()
        delete_container.setSpacing(5)
        delete_btn = QPushButton("DELETE SELECTED STATE")
        delete_btn.clicked.connect(lambda: self.monitor.version_manager.delete_version(states_list.toPlainText().split('\n')[0].split('|')[0].strip()))
        delete_btn.setStyleSheet(button_style)
        delete_container.addWidget(delete_btn)
        delete_desc = QLabel("Permanently removes the selected save point")
        delete_desc.setStyleSheet(save_desc.styleSheet())
        delete_container.addWidget(delete_desc)
        controls_layout.addLayout(delete_container)
        
        model_layout.addLayout(controls_layout)
        
        # Status text
        status_text = QLabel("Last saved: March 20, 18:00 (2 minutes ago)")
        status_text.setStyleSheet("""
            QLabel {
                color: #888888;
                font-family: 'Consolas';
                font-size: 12px;
            }
        """)
        model_layout.addWidget(status_text)
        
        # Add to stack
        self.graph_stack.addWidget(activity_graph)
        self.graph_stack.addWidget(perf_graph)
        self.graph_stack.addWidget(learning_graph)
        self.graph_stack.addWidget(model_save_widget)
        graphs_layout.addWidget(self.graph_stack)
        
        # Connect tab buttons
        def make_tab_handler(index):
            def handle_tab():
                self.graph_stack.setCurrentIndex(index)
                for btn in tab_buttons:
                    btn.setChecked(False)
                tab_buttons[index].setChecked(True)
            return handle_tab
        
        for i, btn in enumerate(tab_buttons):
            btn.clicked.connect(make_tab_handler(i))
        
        layout.addWidget(graphs_group)
        self.main_layout.addWidget(network_panel, stretch=2)

    def setup_memory_panel(self):
        """Set up the right panel for auto wiki system"""
        auto_wiki_panel = QGroupBox("Auto Wiki")
        layout = QVBoxLayout(auto_wiki_panel)
        layout.setSpacing(2)
        
        # Topic selection
        topic_layout = QHBoxLayout()
        topic_layout.addWidget(QLabel("Knowledge Base:"))
        topic_combo = QComboBox()
        topic_combo.addItems(["Neural Networks", "Machine Learning", "Deep Learning", "Cognitive Systems"])
        topic_layout.addWidget(topic_combo)
        
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("Search Depth:"))
        depth_spin = QSpinBox()
        depth_spin.setValue(3)
        depth_layout.addWidget(depth_spin)
        
        learn_btn = QPushButton("Auto Learn")
        learn_btn.clicked.connect(lambda: self.monitor.autowiki.start_learning(topic_combo.currentText(), depth_spin.value()))
        learn_btn.setStyleSheet("""
            QPushButton {
                background-color: #2D2D2D;
                border: 1px solid #D4AF37;
                color: #D4AF37;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3D3D3D;
            }
        """)
        depth_layout.addWidget(learn_btn)
        
        layout.addLayout(topic_layout)
        layout.addLayout(depth_layout)
        
        # Core Insights
        insights_group = QGroupBox("Knowledge Integration")
        insights_layout = QVBoxLayout(insights_group)
        self.insights_text = QTextEdit()
        self.insights_text.setReadOnly(True)
        self.insights_text.setStyleSheet("""
            QTextEdit {
                background-color: #2D2D2D;
                border: 1px solid #D4AF37;
                color: #FFFFFF;
                font-family: 'Consolas';
                font-size: 12px;
                padding: 5px;
            }
        """)
        self.insights_text.setText("• Auto-discovering neural network architectures\n• Integration with existing knowledge bases\n• Cross-referencing with academic papers\n• Learning rate: 84.5%")
        insights_layout.addWidget(self.insights_text)
        layout.addWidget(insights_group)
        
        # Nodes & Processors
        nodes_group = QGroupBox("Nodes & Processors")
        nodes_layout = QVBoxLayout(nodes_group)
        
        # Add processor status
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Active Processors:"))
        status_label = QLabel("12/16")
        status_label.setStyleSheet("color: #D4AF37; font-weight: bold;")
        status_layout.addWidget(status_label)
        nodes_layout.addLayout(status_layout)
        
        # Add node list
        self.nodes_list = QTextEdit()
        self.nodes_list.setReadOnly(True)
        self.nodes_list.setStyleSheet("""
            QTextEdit {
                background-color: #2D2D2D;
                border: 1px solid #D4AF37;
                color: #FFFFFF;
                font-family: 'Consolas';
                font-size: 12px;
                padding: 5px;
            }
        """)
        self.nodes_list.setText("PROC_01: Knowledge Acquisition  [ACTIVE]\nPROC_02: Pattern Recognition   [ACTIVE]\nPROC_03: Data Integration     [ACTIVE]\nPROC_04: Neural Synthesis     [STANDBY]")
        nodes_layout.addWidget(self.nodes_list)
        
        # Add processor controls
        controls_layout = QHBoxLayout()
        start_all_btn = QPushButton("Start All")
        start_all_btn.clicked.connect(lambda: [self.monitor.start_processor(proc_id) for proc_id in self.monitor.processors.keys()])
        controls_layout.addWidget(start_all_btn)
        
        stop_all_btn = QPushButton("Stop All")
        stop_all_btn.clicked.connect(lambda: [self.monitor.stop_processor(proc_id) for proc_id in self.monitor.processors.keys()])
        controls_layout.addWidget(stop_all_btn)
        
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.monitor.reset_seed)
        controls_layout.addWidget(reset_btn)
        nodes_layout.addLayout(controls_layout)
        
        layout.addWidget(nodes_group)
        
        # System Metrics
        metrics_group = QGroupBox("System Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        
        # Tabs for metrics
        metrics_buttons = QHBoxLayout()
        for text in ["Processing", "Memory", "Network Load"]:
            btn = QPushButton(text)
            metrics_buttons.addWidget(btn)
        metrics_layout.addLayout(metrics_buttons)
        
        # Metrics display
        metrics_display = QTextEdit()
        metrics_display.setReadOnly(True)
        metrics_display.setStyleSheet("""
            QTextEdit {
                background-color: #2D2D2D;
                border: 1px solid #D4AF37;
                color: #FFFFFF;
                font-family: 'Consolas';
                font-size: 12px;
                padding: 5px;
            }
        """)
        metrics_display.setText("Processing Load: 67%\nMemory Usage: 4.2GB\nNetwork Speed: 1.2GB/s\nActive Threads: 24")
        metrics_layout.addWidget(metrics_display)
        
        # Update controls
        update_layout = QHBoxLayout()
        update_layout.addWidget(QLabel("Update Rate:"))
        update_spin = QSpinBox()
        update_spin.setMaximum(10000)
        update_spin.setValue(2000)
        update_spin.valueChanged.connect(lambda value: self.update_timer.setInterval(value))
        update_layout.addWidget(update_spin)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.update_visualizations)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2D2D2D;
                border: 1px solid #D4AF37;
                color: #D4AF37;
            }
            QPushButton:hover {
                background-color: #3D3D3D;
            }
        """)
        update_layout.addWidget(refresh_btn)
        metrics_layout.addLayout(update_layout)
        
        layout.addWidget(metrics_group)
        
        self.main_layout.addWidget(auto_wiki_panel, stretch=1)

    def toggle_chat(self):
        """Toggle the chat dialog"""
        if self.chat_dialog.isVisible():
            self.chat_dialog.hide()
        else:
            # Position the dialog next to the main window
            pos = self.geometry()
            self.chat_dialog.move(pos.x() + pos.width(), pos.y())
            self.chat_dialog.show()

    def update_visualizations(self):
        """Update the visualizations periodically"""
        # Add real-time update logic here
        pass

    def update_metrics(self, metrics):
        """Update UI with current metrics"""
        try:
            # Update growth stage
            if hasattr(self, 'growth_value'):
                self.growth_value.setText(metrics.get('growth_stage', 'Seed'))
            
            # Update stability
            if hasattr(self, 'stability_value'):
                self.stability_value.setText(f"{metrics.get('stability', 0)}%")
            
            # Update consciousness
            if hasattr(self, 'consciousness_value'):
                self.consciousness_value.setText(f"{metrics.get('consciousness', 0)}")
            
            # Update complexity
            if hasattr(self, 'complexity_value'):
                self.complexity_value.setText(f"{metrics.get('complexity', 0)}")
            
            # Update growth rate
            if hasattr(self, 'growth_rate_value'):
                self.growth_rate_value.setText(f"{metrics.get('growth_rate', 0)} / min")
            
            # Update system age
            if hasattr(self, 'age_value'):
                self.age_value.setText(metrics.get('system_age', '0h 0m'))
            
            # Update active connections
            if hasattr(self, 'connections_value'):
                self.connections_value.setText(str(metrics.get('active_connections', 0)))
            
        except Exception as e:
            print(f"Error updating metrics: {e}")

    def update_nodes(self, nodes_data):
        """Update UI with current node status"""
        try:
            if hasattr(self, 'nodes_list'):
                text = ""
                for node in nodes_data:
                    text += f"{node['text']}\n"
                self.nodes_list.setText(text)
        except Exception as e:
            print(f"Error updating nodes: {e}")

    def update_processors(self, processors_data):
        """Update UI with current processor status"""
        try:
            if hasattr(self, 'processors_list'):
                text = ""
                for proc in processors_data:
                    text += f"{proc['text']}\n"
                self.processors_list.setText(text)
        except Exception as e:
            print(f"Error updating processors: {e}")

    def update_article(self, article_data):
        """Update UI with current article"""
        try:
            if hasattr(self, 'insights_text'):
                text = f"• {article_data['title']}\n• {article_data['content']}\n• Learning rate: {article_data.get('progress', 0)}%"
                self.insights_text.setText(text)
        except Exception as e:
            print(f"Error updating article: {e}")

    def update_article_progress(self, progress_data):
        """Update UI with article progress"""
        try:
            if hasattr(self, 'progress_value'):
                self.progress_value.setText(f"{progress_data.get('progress', 0)}%")
        except Exception as e:
            print(f"Error updating article progress: {e}")

def main():
    app = QApplication(sys.argv)
    
    # Load Consolas font for a more technical look
    font_id = QFontDatabase.addApplicationFont("assets/fonts/Consolas.ttf")
    
    window = LuminaClient()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 