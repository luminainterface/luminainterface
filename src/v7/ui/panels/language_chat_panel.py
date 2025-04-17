#!/usr/bin/env python3
"""
Language Chat Panel for LUMINA V7

This module provides a user interface for interacting with the 
Enhanced Language Integration System through a chat interface.

Features:
- Interactive chat panel with real-time metrics display
- Streaming text responses with typing animation
- Neural network and language model integration
- Consciousness level and neural-linguistic score visualization
- Adjustable weights for neural network and language model components

The streaming implementation uses a thread-safe approach with Qt signals
to communicate between the background worker thread and the UI thread,
preventing Qt timer issues across threads.
"""

import os
import sys
import json
import logging
import time
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine which Qt library to use
try:
    from PySide6.QtCore import Qt, QTimer, Signal, Slot, QSize, QRect, QPropertyAnimation, QEasingCurve, QObject
    from PySide6.QtGui import QFont, QColor, QPalette, QFontMetrics, QPainter, QBrush, QPen, QLinearGradient
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
        QTextEdit, QLineEdit, QPushButton, QLabel, 
        QScrollArea, QSlider, QSplitter, QFrame, 
        QGraphicsDropShadowEffect, QSizePolicy
    )
    USING_PYSIDE6 = True
except ImportError:
    try:
        from PyQt5.QtCore import Qt, QTimer, pyqtSignal as Signal, pyqtSlot as Slot, QSize, QRect, QPropertyAnimation, QEasingCurve, QObject
        from PyQt5.QtGui import QFont, QColor, QPalette, QFontMetrics, QPainter, QBrush, QPen, QLinearGradient
        from PyQt5.QtWidgets import (
            QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
            QTextEdit, QLineEdit, QPushButton, QLabel, 
            QScrollArea, QSlider, QSplitter, QFrame, 
            QGraphicsDropShadowEffect, QSizePolicy
        )
        USING_PYSIDE6 = False
    except ImportError:
        logger.error("Neither PySide6 nor PyQt5 is available. UI cannot be initialized.")
        raise ImportError("Qt libraries not found. Please install PySide6 or PyQt5.")

# Import the Enhanced Language Integration
try:
    from src.v7.enhanced_language_integration import EnhancedLanguageIntegration
    HAS_INTEGRATION = True
    logger.info("Enhanced Language Integration module loaded successfully")
except ImportError:
    logger.warning("Enhanced Language Integration module not found. Using mock integration.")
    HAS_INTEGRATION = False

# Create a mock integration if the real one is not available
class MockLanguageIntegration:
    """A mock implementation of the language integration for testing"""
    
    def __init__(self):
        self.llm_weight = 0.5
        self.nn_weight = 0.5
        
    def process_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process text input and return mock results"""
        time.sleep(0.5)  # Simulate processing delay
        
        return {
            "response": f"Mock response to: {text}",
            "consciousness_level": random.uniform(0.3, 0.9),
            "neural_linguistic_score": random.uniform(0.4, 0.8),
            "recursive_pattern_depth": random.randint(1, 5),
            "processing_time": random.uniform(0.1, 2.0)
        }
    
    def process_text_streaming(self, text: str, chunk_callback: Callable[[str, Optional[Dict[str, Any]]], None], 
                              done_callback: Callable[[Dict[str, Any]], None] = None, 
                              context: Optional[Dict[str, Any]] = None):
        """Process text input and generate streaming mock responses"""
        # Start a thread to simulate streaming
        import threading
        
        thread = threading.Thread(
            target=self._stream_mock_response,
            args=(text, chunk_callback, done_callback, context)
        )
        thread.daemon = True
        thread.start()
    
    def _stream_mock_response(self, text: str, chunk_callback: Callable, 
                             done_callback: Callable, context: Optional[Dict[str, Any]]):
        """Thread function to simulate streaming response"""
        # Initial "thinking" delay
        time.sleep(0.5)
        
        # Generate response message
        response = f"I'm responding to your message: '{text}' with a streaming mock response. "
        response += "This shows how the system can generate text character by character, "
        response += "simulating a more natural and engaging conversation flow. "
        response += "You can see metrics updating in real-time as the response is generated."
        
        # Break response into chunks for streaming
        words = response.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            if len(current_chunk.split()) < 4:  # Max 4 words per chunk
                if current_chunk:
                    current_chunk += " " + word
                else:
                    current_chunk = word
            else:
                chunks.append(current_chunk)
                current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Initialize metrics
        metrics = {
            "consciousness_level": 0.2,
            "neural_linguistic_score": 0.3,
            "recursive_pattern_depth": 1,
            "processing_time": 0.1
        }
        
        # Send chunks with updated metrics
        full_response = ""
        for i, chunk in enumerate(chunks):
            # Simulate typing delay
            time.sleep(0.1 + random.random() * 0.2)
            
            # Update metrics
            progress = (i + 1) / len(chunks)
            metrics["consciousness_level"] = min(0.2 + progress * 0.7, 0.9)
            metrics["neural_linguistic_score"] = min(0.3 + progress * 0.5, 0.8)
            metrics["recursive_pattern_depth"] = min(1 + int(progress * 4), 5)
            metrics["processing_time"] += 0.1 + random.random() * 0.1
            
            # Add to full response and send chunk
            full_response += chunk
            if i < len(chunks) - 1:
                full_response += " "
                chunk += " "
            
            if chunk_callback:
                chunk_callback(chunk, metrics.copy())
        
        # Final metrics
        final_metrics = {
            "response": full_response,
            "consciousness_level": metrics["consciousness_level"],
            "neural_linguistic_score": metrics["neural_linguistic_score"],
            "recursive_pattern_depth": metrics["recursive_pattern_depth"],
            "processing_time": metrics["processing_time"]
        }
        
        # Call done callback with final metrics
        if done_callback:
            done_callback(final_metrics)
    
    def get_status(self) -> Dict[str, Any]:
        """Get mock status information"""
        return {
            "status": "OK",
            "mode": "MOCK",
            "llm_weight": self.llm_weight,
            "nn_weight": self.nn_weight,
            "system_metrics": {
                "memory_usage": f"{random.randint(50, 200)}MB",
                "active_nodes": random.randint(80, 150),
                "language_models_ready": 1,
                "last_update": datetime.now().isoformat()
            }
        }
    
    def set_weights(self, llm_weight: float, nn_weight: float) -> bool:
        """Set the weights for the language model and neural network components"""
        self.llm_weight = max(0.0, min(1.0, llm_weight))
        self.nn_weight = max(0.0, min(1.0, nn_weight))
        return True


class MessageBubble(QFrame):
    """
    A custom widget that displays a chat message bubble
    with consciousness metrics and styling.
    """
    
    def __init__(
        self, 
        text: str, 
        is_user: bool = False, 
        consciousness_level: float = 0.0,
        neural_linguistic_score: float = 0.0,
        recursive_pattern_depth: int = 0,
        processing_time: float = 0.0,
        parent=None
    ):
        super().__init__(parent)
        self.text = text
        self.is_user = is_user
        self.consciousness_level = consciousness_level
        self.neural_linguistic_score = neural_linguistic_score
        self.recursive_pattern_depth = recursive_pattern_depth
        self.processing_time = processing_time
        
        # Set up appearance
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(3, 3)
        self.setGraphicsEffect(shadow)
        
        # Create layout
        self.layout = QVBoxLayout(self)
        
        # Create message content
        self.message_content = QTextEdit()
        self.message_content.setReadOnly(True)
        self.message_content.setPlainText(text)
        self.message_content.setFrameStyle(QFrame.NoFrame)
        self.message_content.setStyleSheet("background: transparent; border: none;")
        
        # Create metrics display if it's a system message
        if not is_user:
            self.metrics_layout = QHBoxLayout()
            
            # Consciousness level
            self.consciousness_label = QLabel(f"CL: {consciousness_level:.2f}")
            self.consciousness_label.setStyleSheet(
                f"color: {self._get_color_from_value(consciousness_level)}; font-size: 10px;"
            )
            
            # Neural linguistic score
            self.neural_label = QLabel(f"NLS: {neural_linguistic_score:.2f}")
            self.neural_label.setStyleSheet(
                f"color: {self._get_color_from_value(neural_linguistic_score)}; font-size: 10px;"
            )
            
            # Recursive pattern depth
            self.pattern_label = QLabel(f"RPD: {recursive_pattern_depth}")
            self.pattern_label.setStyleSheet("color: #666; font-size: 10px;")
            
            # Processing time
            self.time_label = QLabel(f"{processing_time:.2f}s")
            self.time_label.setStyleSheet("color: #666; font-size: 10px;")
            
            # Add metrics to layout
            self.metrics_layout.addWidget(self.consciousness_label)
            self.metrics_layout.addWidget(self.neural_label)
            self.metrics_layout.addWidget(self.pattern_label)
            self.metrics_layout.addWidget(self.time_label)
            self.metrics_layout.addStretch()
            
            self.layout.addWidget(self.message_content)
            self.layout.addLayout(self.metrics_layout)
        else:
            self.layout.addWidget(self.message_content)
        
        # Adjust appearance based on sender
        self._adjust_appearance()
        
    def _adjust_appearance(self):
        """Adjust the appearance based on who sent the message"""
        if self.is_user:
            # User message styling
            self.setStyleSheet("""
                QFrame {
                    background-color: #E3F2FD;
                    border-radius: 15px;
                    margin-left: 80px;
                    margin-right: 10px;
                    border: 1px solid #BBDEFB;
                }
            """)
        else:
            # System message styling - color based on consciousness level
            bg_color = self._get_bubble_bg_color()
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: {bg_color};
                    border-radius: 15px;
                    margin-left: 10px;
                    margin-right: 80px;
                    border: 1px solid #E0E0E0;
                }}
            """)
    
    def _get_bubble_bg_color(self) -> str:
        """Get the background color based on consciousness level"""
        if self.consciousness_level < 0.4:
            return "#F5F5F5"  # Light gray for low consciousness
        elif self.consciousness_level < 0.7:
            return "#F1F8E9"  # Light green for medium consciousness
        else:
            return "#E8F5E9"  # Green for high consciousness
    
    def _get_color_from_value(self, value: float) -> str:
        """Convert a value (0-1) to a color from red to green"""
        if value < 0.3:
            return "#F44336"  # Red
        elif value < 0.6:
            return "#FF9800"  # Orange
        else:
            return "#4CAF50"  # Green
    
    def sizeHint(self) -> QSize:
        """Suggest a size for the widget"""
        width = self.parent().width() if self.parent() else 400
        text_width = width - 150  # Account for margins
        
        # Calculate height based on text content
        font_metrics = QFontMetrics(self.message_content.font())
        text_height = font_metrics.boundingRect(
            QRect(0, 0, text_width, 1000), 
            Qt.TextWordWrap, 
            self.text
        ).height()
        
        # Add extra height for metrics if it's a system message
        extra_height = 30 if not self.is_user else 0
        
        return QSize(width, text_height + extra_height + 40)  # +40 for padding


class StreamingMessageBubble(MessageBubble):
    """
    An extension of MessageBubble that supports streaming text updates
    with a typing animation effect.
    """
    
    def __init__(
        self, 
        initial_text: str = "", 
        is_user: bool = False,
        consciousness_level: float = 0.0,
        neural_linguistic_score: float = 0.0,
        recursive_pattern_depth: int = 0,
        processing_time: float = 0.0,
        parent=None
    ):
        super().__init__(
            initial_text, 
            is_user,
            consciousness_level,
            neural_linguistic_score,
            recursive_pattern_depth,
            processing_time,
            parent
        )
        
        # Setup the typing animation
        self.full_text = initial_text
        self.current_position = 0
        self.typing_speed = 30  # milliseconds per character
        
        # Create typing timer
        self.typing_timer = QTimer(self)
        self.typing_timer.timeout.connect(self.update_typing_animation)
        
        # Setup cursor animation
        self.cursor_visible = True
        self.cursor_timer = QTimer(self)
        self.cursor_timer.timeout.connect(self.toggle_cursor)
        self.cursor_timer.start(500)  # Blink every 500ms
        
        # Show typing indicator initially if not user message
        if not is_user:
            self.show_typing_indicator()
    
    def set_full_text(self, text: str):
        """Set the full text to display and start the typing animation"""
        self.full_text = text
        self.current_position = 0
        self.update_typing_animation()
        self.typing_timer.start(self.typing_speed)
    
    def append_text(self, text: str):
        """Append text to the full text and update the animation"""
        self.full_text += text
        if not self.typing_timer.isActive():
            self.typing_timer.start(self.typing_speed)
    
    def show_typing_indicator(self):
        """Show a typing indicator before the message starts"""
        self.message_content.setPlainText("Thinking...")
        self.message_content.setStyleSheet("background: transparent; border: none; color: #999;")
    
    def update_typing_animation(self):
        """Update the displayed text to show one more character"""
        if self.current_position < len(self.full_text):
            self.current_position += 1
            display_text = self.full_text[:self.current_position]
            
            # If we're still typing, add a cursor
            if self.cursor_visible and self.current_position < len(self.full_text):
                display_text += "▌"
                
            self.message_content.setPlainText(display_text)
            self.message_content.setStyleSheet("background: transparent; border: none;")
        else:
            # Typing animation complete
            self.typing_timer.stop()
            
            # Update metrics display if needed
            if not self.is_user and hasattr(self, 'metrics_layout'):
                self.consciousness_label.setText(f"CL: {self.consciousness_level:.2f}")
                self.neural_label.setText(f"NLS: {self.neural_linguistic_score:.2f}")
                self.pattern_label.setText(f"RPD: {self.recursive_pattern_depth}")
                self.time_label.setText(f"{self.processing_time:.2f}s")
    
    def toggle_cursor(self):
        """Toggle the cursor visibility for blinking effect"""
        if self.current_position < len(self.full_text):
            self.cursor_visible = not self.cursor_visible
            self.update_typing_animation()
    
    def set_metrics(self, consciousness_level: float, neural_linguistic_score: float, 
                   recursive_pattern_depth: int, processing_time: float):
        """Update the metrics values"""
        self.consciousness_level = consciousness_level
        self.neural_linguistic_score = neural_linguistic_score
        self.recursive_pattern_depth = recursive_pattern_depth
        self.processing_time = processing_time
        
        if hasattr(self, 'metrics_layout') and not self.typing_timer.isActive():
            self.consciousness_label.setText(f"CL: {consciousness_level:.2f}")
            self.consciousness_label.setStyleSheet(
                f"color: {self._get_color_from_value(consciousness_level)}; font-size: 10px;"
            )
            
            self.neural_label.setText(f"NLS: {neural_linguistic_score:.2f}")
            self.neural_label.setStyleSheet(
                f"color: {self._get_color_from_value(neural_linguistic_score)}; font-size: 10px;"
            )
            
            self.pattern_label.setText(f"RPD: {recursive_pattern_depth}")
            self.time_label.setText(f"{processing_time:.2f}s")
            
        # Update appearance based on metrics
        self._adjust_appearance()


class StreamingHandler(QObject):
    """
    Handles streaming text updates from a worker thread to the UI thread safely.
    Uses Qt signals to communicate between threads.
    """
    # Define signals
    chunk_received = Signal(str, dict)
    streaming_done = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
    def on_chunk(self, chunk: str, metrics: Optional[Dict[str, Any]] = None):
        """Called from worker thread when a new chunk is received"""
        self.chunk_received.emit(chunk, metrics or {})
        
    def on_done(self, final_metrics: Dict[str, Any]):
        """Called from worker thread when streaming is complete"""
        self.streaming_done.emit(final_metrics)


class LanguageChatPanel(QWidget):
    """
    A chat panel for interacting with the Enhanced Language Integration System.
    This panel provides a chat-like interface with metrics display and settings.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize language integration
        if HAS_INTEGRATION:
            self.language_integration = EnhancedLanguageIntegration()
        else:
            self.language_integration = MockLanguageIntegration()
        
        # Set up UI
        self.setup_ui()
        
        # Set up streaming handler
        self.streaming_handler = StreamingHandler(self)
        self.streaming_handler.chunk_received.connect(self.on_streaming_chunk_received)
        self.streaming_handler.streaming_done.connect(self.on_streaming_done_received)
        
        # Track current streaming message
        self.current_streaming_message = None
        
        # Add welcome message
        self.add_system_message(
            "Welcome to the V7 Language Chat Panel. I'm ready to assist you with "
            "enhanced language capabilities. You can adjust the LLM and Neural Network "
            "weights using the sliders below to see how they affect my responses.",
            0.7, 0.65, 3, 0.1
        )
        
        # Set up metrics update timer
        self.metrics_timer = QTimer(self)
        self.metrics_timer.timeout.connect(self.update_metrics)
        self.metrics_timer.start(2000)  # Update every 2 seconds
        
        # Set up streaming simulation timer
        self.streaming_timer = QTimer(self)
        self.streaming_timer.timeout.connect(self.simulate_streaming)
        
        # Streaming simulation data
        self.streaming_text = ""
        self.streaming_position = 0
        self.streaming_metrics = {"cl": 0.0, "nls": 0.0, "rpd": 0, "time": 0.0}
        
        # Initial metrics update
        self.update_metrics()
    
    def setup_ui(self):
        """Set up the user interface"""
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # Create title
        self.title_label = QLabel("V7 Language Chat Panel")
        self.title_label.setStyleSheet("""
            font-size: 18px; 
            font-weight: bold; 
            color: #2196F3;
            margin-bottom: 10px;
        """)
        self.main_layout.addWidget(self.title_label)
        
        # Create system metrics display
        self.metrics_frame = QFrame()
        self.metrics_frame.setFrameShape(QFrame.StyledPanel)
        self.metrics_frame.setStyleSheet("""
            QFrame {
                background-color: #E3F2FD;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        
        self.metrics_layout = QHBoxLayout(self.metrics_frame)
        self.metrics_layout.setContentsMargins(10, 5, 10, 5)
        
        # Memory usage
        self.memory_label = QLabel("Memory: 0MB")
        self.memory_label.setStyleSheet("color: #333; font-size: 12px;")
        
        # Active nodes
        self.nodes_label = QLabel("Nodes: 0")
        self.nodes_label.setStyleSheet("color: #333; font-size: 12px;")
        
        # Language models
        self.models_label = QLabel("LMs: 0")
        self.models_label.setStyleSheet("color: #333; font-size: 12px;")
        
        # System status
        self.status_label = QLabel("Status: Unknown")
        self.status_label.setStyleSheet("color: #333; font-size: 12px;")
        
        # Add metrics to layout
        self.metrics_layout.addWidget(self.memory_label)
        self.metrics_layout.addWidget(self.nodes_label)
        self.metrics_layout.addWidget(self.models_label)
        self.metrics_layout.addWidget(self.status_label)
        self.metrics_layout.addStretch()
        
        self.main_layout.addWidget(self.metrics_frame)
        
        # Create splitter for main content
        self.content_splitter = QSplitter(Qt.Vertical)
        
        # Create messages area
        self.messages_area = QScrollArea()
        self.messages_area.setWidgetResizable(True)
        self.messages_area.setFrameShape(QFrame.NoFrame)
        self.messages_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.messages_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.messages_container = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_container)
        self.messages_layout.setAlignment(Qt.AlignTop)
        self.messages_layout.setContentsMargins(0, 0, 0, 0)
        self.messages_layout.setSpacing(15)
        
        self.messages_area.setWidget(self.messages_container)
        
        self.content_splitter.addWidget(self.messages_area)
        
        # Create the weight slider container
        self.weight_frame = QFrame()
        self.weight_frame.setFrameShape(QFrame.StyledPanel)
        self.weight_frame.setStyleSheet("""
            QFrame {
                background-color: #ECEFF1;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        
        self.weight_layout = QVBoxLayout(self.weight_frame)
        
        # LLM Weight slider
        self.llm_layout = QHBoxLayout()
        self.llm_label = QLabel("LLM Weight:")
        self.llm_label.setStyleSheet("font-size: 12px;")
        
        self.llm_slider = QSlider(Qt.Horizontal)
        self.llm_slider.setMinimum(0)
        self.llm_slider.setMaximum(100)
        self.llm_slider.setValue(50)
        self.llm_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #CFD8DC;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                border: 1px solid #1976D2;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        
        self.llm_value_label = QLabel("0.50")
        
        self.llm_layout.addWidget(self.llm_label)
        self.llm_layout.addWidget(self.llm_slider)
        self.llm_layout.addWidget(self.llm_value_label)
        
        # NN Weight slider
        self.nn_layout = QHBoxLayout()
        self.nn_label = QLabel("NN Weight:")
        self.nn_label.setStyleSheet("font-size: 12px;")
        
        self.nn_slider = QSlider(Qt.Horizontal)
        self.nn_slider.setMinimum(0)
        self.nn_slider.setMaximum(100)
        self.nn_slider.setValue(50)
        self.nn_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #CFD8DC;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #388E3C;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        
        self.nn_value_label = QLabel("0.50")
        
        self.nn_layout.addWidget(self.nn_label)
        self.nn_layout.addWidget(self.nn_slider)
        self.nn_layout.addWidget(self.nn_value_label)
        
        # Add sliders to weight layout
        self.weight_layout.addLayout(self.llm_layout)
        self.weight_layout.addLayout(self.nn_layout)
        
        # Add reset button
        self.reset_button = QPushButton("Reset Weights")
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #ECEFF1;
                border: 1px solid #B0BEC5;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #CFD8DC;
            }
            QPushButton:pressed {
                background-color: #B0BEC5;
            }
        """)
        
        self.weight_layout.addWidget(self.reset_button, alignment=Qt.AlignRight)
        
        # Input area
        self.input_frame = QFrame()
        self.input_frame.setFrameShape(QFrame.StyledPanel)
        self.input_frame.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 5px;
                border: 1px solid #E0E0E0;
            }
        """)
        
        self.input_layout = QHBoxLayout(self.input_frame)
        self.input_layout.setContentsMargins(10, 10, 10, 10)
        self.input_layout.setSpacing(10)
        
        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("Type your message here...")
        self.input_edit.setStyleSheet("""
            QTextEdit {
                border: none;
                background-color: transparent;
                font-size: 13px;
            }
        """)
        self.input_edit.setMaximumHeight(100)  # Initial height
        self.input_edit.setMinimumHeight(40)
        self.input_edit.textChanged.connect(self.adjust_input_height)
        
        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        self.send_button.setFixedWidth(80)
        
        self.input_layout.addWidget(self.input_edit)
        self.input_layout.addWidget(self.send_button)
        
        # Main layout setup
        self.bottom_layout = QVBoxLayout()
        self.bottom_layout.addWidget(self.weight_frame)
        self.bottom_layout.addWidget(self.input_frame)
        
        self.bottom_container = QWidget()
        self.bottom_container.setLayout(self.bottom_layout)
        self.bottom_container.setMaximumHeight(180)
        
        self.content_splitter.addWidget(self.bottom_container)
        
        # Set stretch factors for splitter
        self.content_splitter.setStretchFactor(0, 3)
        self.content_splitter.setStretchFactor(1, 1)
        
        self.main_layout.addWidget(self.content_splitter)
        
        # Connect signals and slots
        self.send_button.clicked.connect(self.send_message)
        self.input_edit.textChanged.connect(self.handle_input_changed)
        
        # Connect weight sliders
        self.llm_slider.valueChanged.connect(self.update_llm_weight)
        self.nn_slider.valueChanged.connect(self.update_nn_weight)
        self.reset_button.clicked.connect(self.reset_weights)
        
        # Set initial size
        self.resize(600, 800)
    
    def add_user_message(self, text: str):
        """Add a message from the user to the chat"""
        message = MessageBubble(text, is_user=True)
        self.messages_layout.addWidget(message)
        self.scroll_to_bottom()
    
    def add_system_message(self, text: str, consciousness_level: float, 
                          neural_linguistic_score: float, recursive_pattern_depth: int,
                          processing_time: float):
        """Add a message from the system to the chat"""
        message = MessageBubble(
            text=text, 
            is_user=False, 
            consciousness_level=consciousness_level,
            neural_linguistic_score=neural_linguistic_score,
            recursive_pattern_depth=recursive_pattern_depth,
            processing_time=processing_time
        )
        self.messages_layout.addWidget(message)
        self.scroll_to_bottom()
    
    def add_streaming_message(self, initial_text: str = ""):
        """Add a streaming message bubble that will update over time"""
        # If there's already a streaming message, finalize it first
        if self.current_streaming_message:
            self.finalize_streaming_message()
            
        message = StreamingMessageBubble(
            initial_text=initial_text,
            is_user=False
        )
        self.messages_layout.addWidget(message)
        self.current_streaming_message = message
        self.scroll_to_bottom()
        return message
    
    def update_streaming_message(self, text_chunk: str):
        """Add new text to the current streaming message"""
        if self.current_streaming_message:
            self.current_streaming_message.append_text(text_chunk)
            self.scroll_to_bottom()
    
    def finalize_streaming_message(self, metrics=None):
        """Finalize the current streaming message with complete metrics"""
        if self.current_streaming_message:
            if metrics:
                self.current_streaming_message.set_metrics(
                    consciousness_level=metrics.get("consciousness_level", 0.0),
                    neural_linguistic_score=metrics.get("neural_linguistic_score", 0.0),
                    recursive_pattern_depth=metrics.get("recursive_pattern_depth", 0),
                    processing_time=metrics.get("processing_time", 0.0)
                )
            self.current_streaming_message = None
    
    def simulate_streaming(self):
        """Simulate streaming response for demonstration purposes"""
        if self.streaming_position < len(self.streaming_text):
            # Determine chunk size (simulate variable speed)
            chunk_size = random.randint(1, 5)
            end_pos = min(self.streaming_position + chunk_size, len(self.streaming_text))
            
            # Get text chunk
            chunk = self.streaming_text[self.streaming_position:end_pos]
            self.streaming_position = end_pos
            
            # Update streaming message
            self.update_streaming_message(chunk)
            
            # Randomly update metrics during streaming
            if random.random() < 0.2:  # 20% chance per update
                metrics = {
                    "consciousness_level": min(self.streaming_metrics["cl"] + random.uniform(0.0, 0.05), 1.0),
                    "neural_linguistic_score": min(self.streaming_metrics["nls"] + random.uniform(0.0, 0.05), 1.0),
                    "recursive_pattern_depth": self.streaming_metrics["rpd"],
                    "processing_time": self.streaming_metrics["time"] + random.uniform(0.01, 0.1)
                }
                self.streaming_metrics = {
                    "cl": metrics["consciousness_level"],
                    "nls": metrics["neural_linguistic_score"],
                    "rpd": metrics["recursive_pattern_depth"],
                    "time": metrics["processing_time"]
                }
                
                # Update metrics in the streaming message
                if self.current_streaming_message:
                    self.current_streaming_message.set_metrics(**metrics)
        else:
            # Finalize with complete metrics
            final_metrics = {
                "consciousness_level": self.streaming_metrics["cl"],
                "neural_linguistic_score": self.streaming_metrics["nls"],
                "recursive_pattern_depth": self.streaming_metrics["rpd"] or random.randint(1, 5),
                "processing_time": self.streaming_metrics["time"]
            }
            
            self.finalize_streaming_message(final_metrics)
            self.streaming_timer.stop()
    
    def scroll_to_bottom(self):
        """Scroll the messages area to the bottom"""
        self.messages_area.verticalScrollBar().setValue(
            self.messages_area.verticalScrollBar().maximum()
        )
    
    def send_message(self):
        """Send the current message"""
        text = self.input_edit.toPlainText().strip()
        if not text:
            return
        
        # Add user message to chat
        self.add_user_message(text)
        
        # Clear input field
        self.input_edit.clear()
        
        # Create a minimal context
        context = {
            "session_id": "chat_session_1",
            "timestamp": time.time(),
            "interface": "chat_panel"
        }
        
        # Add initial streaming message
        self.add_streaming_message()
        
        # Check if the language integration supports streaming
        if hasattr(self.language_integration, 'process_text_streaming'):
            try:
                # Use streaming mode with thread-safe handler
                self.language_integration.process_text_streaming(
                    text,
                    self.streaming_handler.on_chunk,
                    self.streaming_handler.on_done,
                    context
                )
            except Exception as e:
                logger.error(f"Error starting streaming process: {e}")
                self.update_streaming_message("Sorry, there was an error processing your request.")
                self.finalize_streaming_message()
        else:
            # Fall back to non-streaming mode for compatibility
            try:
                result = self.language_integration.process_text(text, context)
                
                # Update the streaming message with the full response immediately
                if self.current_streaming_message:
                    self.current_streaming_message.set_full_text(result["response"])
                    
                    # Set final metrics
                    metrics = {
                        "consciousness_level": result["consciousness_level"],
                        "neural_linguistic_score": result["neural_linguistic_score"],
                        "recursive_pattern_depth": result["recursive_pattern_depth"],
                        "processing_time": result["processing_time"]
                    }
                    self.finalize_streaming_message(metrics)
            except Exception as e:
                logger.error(f"Error processing text: {e}")
                if self.current_streaming_message:
                    self.current_streaming_message.set_full_text("Sorry, there was an error processing your request.")
                    self.finalize_streaming_message()
    
    def on_streaming_chunk_received(self, text_chunk: str, metrics: Dict[str, Any]):
        """Signal handler for receiving streaming text chunks from worker thread"""
        # Add the text chunk to the current streaming message
        self.update_streaming_message(text_chunk)
        
        # Update metrics if provided
        if metrics and self.current_streaming_message:
            self.current_streaming_message.set_metrics(
                consciousness_level=metrics.get("consciousness_level", 0.0),
                neural_linguistic_score=metrics.get("neural_linguistic_score", 0.0),
                recursive_pattern_depth=metrics.get("recursive_pattern_depth", 0),
                processing_time=metrics.get("processing_time", 0.0)
            )
    
    def on_streaming_done_received(self, final_metrics: Dict[str, Any]):
        """Signal handler for when streaming is complete from worker thread"""
        # Finalize the streaming message with the final metrics
        self.finalize_streaming_message(final_metrics)
    
    def on_streaming_chunk(self, text_chunk: str, metrics: Optional[Dict[str, Any]] = None):
        """Callback for receiving streaming text chunks - used in worker thread"""
        # Use the handler to safely communicate with the UI thread
        self.streaming_handler.on_chunk(text_chunk, metrics)
    
    def on_streaming_done(self, final_metrics: Dict[str, Any]):
        """Callback for when streaming is complete - used in worker thread"""
        # Use the handler to safely communicate with the UI thread
        self.streaming_handler.on_done(final_metrics)
    
    def handle_input_changed(self):
        """Handle input changes for submit on enter"""
        # Submit on Ctrl+Enter
        if Qt.ControlModifier in QApplication.keyboardModifiers() and Qt.Key_Return in [
            Qt.Key(QApplication.keyboardModifiers())
        ]:
            self.send_message()
    
    def adjust_input_height(self):
        """Dynamically adjust input height based on content"""
        document_size = self.input_edit.document().size().toSize()
        height = min(document_size.height() + 20, 100)  # Max height 100
        height = max(height, 40)  # Min height 40
        self.input_edit.setFixedHeight(height)
    
    def update_metrics(self):
        """Update system metrics display"""
        status = self.language_integration.get_status()
        
        # Update memory usage
        memory_usage = status["system_metrics"]["memory_usage"]
        self.memory_label.setText(f"Memory: {memory_usage}")
        
        # Update active nodes
        active_nodes = status["system_metrics"]["active_nodes"]
        self.nodes_label.setText(f"Nodes: {active_nodes}")
        
        # Update language models
        language_models = status["system_metrics"]["language_models_ready"]
        self.models_label.setText(f"LMs: {language_models}")
        
        # Update system status
        system_status = status["status"]
        system_mode = status["mode"]
        self.status_label.setText(f"Status: {system_status} ({system_mode})")
    
    def update_llm_weight(self, value: int):
        """Update the LLM weight from the slider"""
        weight = value / 100.0
        self.llm_value_label.setText(f"{weight:.2f}")
        
        # Update weights in language integration
        self.language_integration.set_weights(
            llm_weight=weight, 
            nn_weight=float(self.nn_value_label.text())
        )
    
    def update_nn_weight(self, value: int):
        """Update the NN weight from the slider"""
        weight = value / 100.0
        self.nn_value_label.setText(f"{weight:.2f}")
        
        # Update weights in language integration
        self.language_integration.set_weights(
            llm_weight=float(self.llm_value_label.text()), 
            nn_weight=weight
        )
    
    def reset_weights(self):
        """Reset the weights to default values"""
        self.llm_slider.setValue(50)
        self.nn_slider.setValue(50)
        
        # Update weights in language integration
        self.language_integration.set_weights(llm_weight=0.5, nn_weight=0.5)


if __name__ == "__main__":
    # Create application for testing
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show widget
    widget = LanguageChatPanel()
    widget.show()
    
    # Run application
    sys.exit(app.exec() if USING_PYSIDE6 else app.exec_()) 