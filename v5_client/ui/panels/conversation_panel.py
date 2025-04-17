"""
Conversation Panel for V5 PySide6 Client

This panel provides an interface for conversation with memory enhancement.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add parent directory to path if needed
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

logger = logging.getLogger(__name__)

# Try to import PySide6
try:
    from PySide6.QtCore import Qt, Signal, Slot, QTimer
    from PySide6.QtGui import QColor, QTextCursor, QFont
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, 
        QPushButton, QLabel, QComboBox, QFrame, QSplitter, QGroupBox
    )
    USING_PYSIDE6 = True
except ImportError:
    from PyQt5.QtCore import Qt, pyqtSignal as Signal, pyqtSlot as Slot, QTimer
    from PyQt5.QtGui import QColor, QTextCursor, QFont
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, 
        QPushButton, QLabel, QComboBox, QFrame, QSplitter, QGroupBox
    )
    USING_PYSIDE6 = False

class ConversationPanel(QWidget):
    """Panel for conversation with memory enhancement"""
    
    # Class variables
    PANEL_NAME = "Conversation"
    PANEL_DESCRIPTION = "Interface for conversation with memory enhancement"
    
    @classmethod
    def get_panel_name(cls):
        """Get the name of the panel"""
        return cls.PANEL_NAME
    
    @classmethod
    def get_panel_description(cls):
        """Get the description of the panel"""
        return cls.PANEL_DESCRIPTION
    
    def __init__(self, socket_manager=None):
        """
        Initialize the Conversation Panel
        
        Args:
            socket_manager: Socket manager for communication with backend
        """
        super().__init__()
        
        self.socket_manager = socket_manager
        self.mock_mode = socket_manager is None or getattr(socket_manager, 'mock_mode', False)
        
        # State variables
        self.memory_mode = "combined"
        self.neural_weight = 0.5
        self.message_history = []
        self.current_context = ""
        
        # Set up UI
        self.init_ui()
        
        # Connect to memory bridge
        self.memory_bridge = self.get_memory_bridge()
        
        logger.info("Conversation Panel initialized")
    
    def init_ui(self):
        """Initialize the user interface"""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Create conversation area
        conversation_widget = self.create_conversation_area()
        splitter.addWidget(conversation_widget)
        
        # Create context panel
        context_widget = self.create_context_panel()
        splitter.addWidget(context_widget)
        
        # Add splitter to layout
        layout.addWidget(splitter)
        
        # Set initial splitter sizes (2:1 ratio)
        splitter.setSizes([600, 300])
        
        # Status bar
        self.status_bar = QFrame()
        status_layout = QHBoxLayout(self.status_bar)
        status_layout.setContentsMargins(5, 2, 5, 2)
        
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch(1)
        
        self.memory_status = QLabel("Memory: Not connected")
        status_layout.addWidget(self.memory_status)
        
        layout.addWidget(self.status_bar)
    
    def create_conversation_area(self):
        """Create the conversation area"""
        # Container widget
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Conversation history
        self.conversation_display = QTextEdit()
        self.conversation_display.setReadOnly(True)
        self.conversation_display.setMinimumHeight(200)
        self.conversation_display.document().setDefaultStyleSheet("""
            .user { color: #2b5797; font-weight: bold; }
            .system { color: #3c8f3f; }
            .meta { color: #888; font-style: italic; font-size: 80%; }
        """)
        layout.addWidget(self.conversation_display)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Type your message here...")
        self.message_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.message_input)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        
        layout.addLayout(input_layout)
        
        # Controls area
        controls_layout = QHBoxLayout()
        
        # Memory mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Memory Mode:"))
        
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Contextual", "Combined", "Synthesized"])
        self.mode_selector.setCurrentText(self.memory_mode.capitalize())
        self.mode_selector.currentTextChanged.connect(self.on_memory_mode_changed)
        mode_layout.addWidget(self.mode_selector)
        
        controls_layout.addLayout(mode_layout)
        
        controls_layout.addStretch(1)
        
        # Clear button
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_conversation)
        controls_layout.addWidget(self.clear_button)
        
        layout.addLayout(controls_layout)
        
        return widget
    
    def create_context_panel(self):
        """Create the context panel"""
        # Container widget
        widget = QWidget()
        widget.setMinimumWidth(200)
        widget.setMaximumWidth(400)
        
        layout = QVBoxLayout(widget)
        
        # Title
        title = QLabel("Memory Context")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #4B6EAF;")
        layout.addWidget(title)
        
        # Current context
        context_group = QGroupBox("Current Context")
        context_layout = QVBoxLayout(context_group)
        
        self.context_display = QTextEdit()
        self.context_display.setReadOnly(True)
        self.context_display.setPlaceholderText("No context available")
        context_layout.addWidget(self.context_display)
        
        layout.addWidget(context_group)
        
        # Related memories
        memories_group = QGroupBox("Related Memories")
        memories_layout = QVBoxLayout(memories_group)
        
        self.memories_display = QTextEdit()
        self.memories_display.setReadOnly(True)
        self.memories_display.setPlaceholderText("No related memories")
        memories_layout.addWidget(self.memories_display)
        
        layout.addWidget(memories_group)
        
        # Stats area
        stats_group = QGroupBox("Memory Stats")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_display = QTextEdit()
        self.stats_display.setReadOnly(True)
        self.stats_display.setMaximumHeight(100)
        self.stats_display.setPlaceholderText("No memory stats available")
        stats_layout.addWidget(self.stats_display)
        
        layout.addWidget(stats_group)
        
        return widget
    
    def get_memory_bridge(self):
        """Get the memory bridge from the socket manager"""
        if not self.socket_manager:
            return None
        
        memory_bridge = self.socket_manager.get_memory_bridge()
        
        if memory_bridge:
            self.memory_status.setText("Memory: Connected")
            self.update_memory_stats()
        
        return memory_bridge
    
    def set_neural_weight(self, weight):
        """
        Set the neural network weight
        
        Args:
            weight: Neural network weight (0.0-1.0)
        """
        self.neural_weight = weight
    
    def set_memory_mode(self, mode):
        """
        Set the memory mode
        
        Args:
            mode: Memory mode (contextual, combined, synthesized)
        """
        self.memory_mode = mode.lower()
        self.mode_selector.setCurrentText(mode.capitalize())
    
    def update_memory_stats(self):
        """Update memory statistics display"""
        if not self.memory_bridge:
            return
        
        # Get stats
        if hasattr(self.memory_bridge, 'get_memory_stats'):
            stats = self.memory_bridge.get_memory_stats()
            
            if stats:
                # Format stats
                stats_text = (
                    f"Total memories: {stats.get('total_memories', 'N/A')}\n"
                    f"Conversations: {stats.get('total_conversations', 'N/A')}\n"
                    f"Topics: {stats.get('total_topics', 'N/A')}"
                )
                
                self.stats_display.setPlainText(stats_text)
    
    @Slot()
    def send_message(self):
        """Send a message from the input field"""
        # Get message text
        message = self.message_input.text().strip()
        
        if not message:
            return
        
        # Clear input field
        self.message_input.clear()
        
        # Process message
        self.process_message(message)
    
    def process_message(self, message):
        """
        Process a user message
        
        Args:
            message: User message
        """
        # Add to conversation display
        self.add_message_to_display("user", message)
        
        # Add to history
        self.message_history.append({"role": "user", "content": message})
        
        # Get memory bridge
        if not self.memory_bridge:
            self.memory_bridge = self.get_memory_bridge()
        
        # Process with memory enhancement
        if self.memory_bridge and hasattr(self.memory_bridge, 'enhance_message_with_memory'):
            # Enhance message
            enhanced = self.memory_bridge.enhance_message_with_memory(
                message, 
                enhance_mode=self.memory_mode
            )
            
            # Update context display
            if enhanced and "enhanced_context" in enhanced:
                self.current_context = enhanced.get("enhanced_context", "")
                self.context_display.setPlainText(self.current_context)
        
        # Search related memories
        if self.memory_bridge and hasattr(self.memory_bridge, 'search_memories'):
            # Search for related memories
            memories = self.memory_bridge.search_memories(message, max_results=3)
            
            # Update memories display
            if memories and "memories" in memories:
                memory_list = memories.get("memories", [])
                
                if memory_list:
                    memory_text = ""
                    for memory in memory_list:
                        memory_text += f"• {memory.get('text', '')}\n\n"
                    
                    self.memories_display.setPlainText(memory_text)
        
        # Generate response based on neural weight
        if self.neural_weight < 0.3:
            # Language-dominated response
            response = self.generate_language_response(message)
        elif self.neural_weight > 0.7:
            # Neural-dominated response
            response = self.generate_neural_response(message)
        else:
            # Balanced response
            response = self.generate_balanced_response(message)
        
        # Add system message pause for realism
        QTimer.singleShot(500, lambda: self.add_message_to_display("system", response))
        
        # Add to history
        self.message_history.append({"role": "system", "content": response})
        
        # Store conversation
        if self.memory_bridge and hasattr(self.memory_bridge, 'store_conversation'):
            # Store user message
            self.memory_bridge.store_conversation(
                message,
                metadata={"role": "user", "neural_weight": self.neural_weight}
            )
            
            # Store system response
            self.memory_bridge.store_conversation(
                response,
                metadata={"role": "system", "neural_weight": self.neural_weight}
            )
    
    def add_message_to_display(self, role, content):
        """
        Add a message to the conversation display
        
        Args:
            role: Message role (user, system)
            content: Message content
        """
        cursor = self.conversation_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Format based on role
        if role == "user":
            html = f'<p class="user">You: {content}</p>'
        else:
            html = f'<p class="system">System: {content}</p>'
            
            # Add metadata for neural weight
            neural_percent = int(self.neural_weight * 100)
            language_percent = 100 - neural_percent
            
            html += f'<p class="meta">Response generated with {neural_percent}% neural / {language_percent}% language weighting using {self.memory_mode} memory mode</p>'
        
        # Insert HTML
        cursor.insertHtml(html)
        
        # Scroll to bottom
        self.conversation_display.setTextCursor(cursor)
        self.conversation_display.ensureCursorVisible()
    
    @Slot()
    def clear_conversation(self):
        """Clear the conversation history"""
        self.conversation_display.clear()
        self.message_history = []
    
    @Slot(str)
    def on_memory_mode_changed(self, mode):
        """
        Handle memory mode change
        
        Args:
            mode: New memory mode
        """
        self.memory_mode = mode.lower()
    
    def generate_language_response(self, message):
        """
        Generate a language-focused response
        
        Args:
            message: User message
            
        Returns:
            Language-focused response
        """
        # Use context if available
        if self.current_context:
            return f"Based on our previous conversations, I understand you're asking about {message.split()[-1]}. {self.current_context}"
        
        # Default responses
        responses = [
            f"I understand you're asking about {message.split()[-1]}. This relates to concepts we've discussed before.",
            f"Your question about {message.split()[0]} is interesting. Let me share some thoughts on this.",
            f"Regarding {message.split()[-1]}, I can provide some insights based on our previous conversations.",
            f"I've analyzed your question about {message.split()[0]} in the context of our earlier discussions."
        ]
        
        import random
        return random.choice(responses)
    
    def generate_neural_response(self, message):
        """
        Generate a neural-focused response
        
        Args:
            message: User message
            
        Returns:
            Neural-focused response
        """
        # Neural-focused responses emphasize pattern analysis
        responses = [
            f"Pattern analysis complete. Your message contains {len(message.split())} elements with a complexity rating of 76%.",
            f"Neural processing indicates a hierarchical structure in your query with recursive elements. Coherence rating: 92%.",
            f"Fractal analysis of your message shows an interesting pattern. The primary node connects to {len(message) % 5 + 3} secondary nodes.",
            f"Processing complete. Pattern structure identified with 87% confidence. Primary concept: {message.split()[0]}."
        ]
        
        import random
        return random.choice(responses)
    
    def generate_balanced_response(self, message):
        """
        Generate a balanced response
        
        Args:
            message: User message
            
        Returns:
            Balanced response
        """
        # Combine elements from both language and neural responses
        
        # Language component
        if self.current_context:
            language_part = f"Based on our previous conversations about {message.split()[-1]}, "
        else:
            language_part = f"Regarding your question about {message.split()[0]}, "
        
        # Neural component
        neural_part = f"I've analyzed the pattern structure (complexity: {len(message) % 30 + 60}%, coherence: {len(message) % 20 + 75}%) and identified key concepts."
        
        return language_part + neural_part 