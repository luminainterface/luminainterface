#!/usr/bin/env python3
"""
V8 Seed Dispersal System

This module implements the fruit-bearing and seed dispersal capabilities of the Lumina system.
It connects to the main seed.py system and provides a modern PySide6-based interface for
interaction and knowledge exchange. Like fruits in nature that attract animals to spread
seeds, this system creates attractive interfaces to spread neural patterns and knowledge.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
    QPushButton, QLabel, QSplitter, QFrame, QScrollArea,
    QDockWidget, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QColor, QPalette, QTextCursor

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import seed system
from src.seed import get_neural_seed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v8.dispersal")

class KnowledgeFruit:
    """
    Represents a packaged unit of knowledge that can be shared and spread
    to other systems, like a fruit containing seeds.
    """
    def __init__(self, 
                 content: str,
                 patterns: Dict[str, float],
                 source_version: float,
                 consciousness_imprint: float,
                 metadata: Optional[Dict[str, Any]] = None):
        self.id = f"fruit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.content = content
        self.patterns = patterns
        self.source_version = source_version
        self.consciousness_imprint = consciousness_imprint
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission"""
        return {
            "id": self.id,
            "content": self.content,
            "patterns": self.patterns,
            "source_version": self.source_version,
            "consciousness_imprint": self.consciousness_imprint,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeFruit':
        """Create from dictionary"""
        fruit = cls(
            content=data["content"],
            patterns=data["patterns"],
            source_version=data["source_version"],
            consciousness_imprint=data["consciousness_imprint"],
            metadata=data["metadata"]
        )
        fruit.id = data["id"]
        fruit.created_at = data["created_at"]
        return fruit

class ChatPanel(QWidget):
    """
    Interactive chat panel that connects to the seed system and enables
    knowledge exchange through an attractive interface.
    """
    message_sent = Signal(str)  # Signal for new messages
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the chat interface"""
        layout = QVBoxLayout(self)
        
        # Chat history area
        self.history = QTextEdit()
        self.history.setReadOnly(True)
        self.history.setMinimumHeight(300)
        
        # Style the history area
        self.history.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #E0E0E0;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        
        # Input area
        self.input_box = QTextEdit()
        self.input_box.setMaximumHeight(100)
        self.input_box.setPlaceholderText("Enter your message...")
        
        # Style the input box
        self.input_box.setStyleSheet("""
            QTextEdit {
                background-color: #2D2D2D;
                color: #E0E0E0;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        
        # Send button
        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #0D47A1;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
            QPushButton:pressed {
                background-color: #0A367A;
            }
        """)
        
        # Button row
        button_row = QHBoxLayout()
        button_row.addWidget(self.send_button)
        
        # Add widgets to layout
        layout.addWidget(self.history)
        layout.addWidget(self.input_box)
        layout.addLayout(button_row)
        
        # Connect signals
        self.send_button.clicked.connect(self.send_message)
        self.input_box.textChanged.connect(self.adjust_input_height)
        
    def send_message(self):
        """Send the current message"""
        text = self.input_box.toPlainText().strip()
        if text:
            self.message_sent.emit(text)
            self.input_box.clear()
            
    def add_message(self, text: str, is_user: bool = True):
        """Add a message to the chat history"""
        cursor = self.history.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Format based on sender
        if is_user:
            cursor.insertHtml(f'<p style="color: #64B5F6;"><b>You:</b> {text}</p>')
        else:
            cursor.insertHtml(f'<p style="color: #81C784;"><b>System:</b> {text}</p>')
            
        # Scroll to bottom
        self.history.setTextCursor(cursor)
        self.history.ensureVisible(0, self.history.height())
        
    def adjust_input_height(self):
        """Adjust input box height based on content"""
        doc_height = self.input_box.document().size().height()
        self.input_box.setMinimumHeight(min(100, max(50, doc_height + 20)))

class SeedDispersalWindow(QMainWindow):
    """
    Main window for the V8 seed dispersal system. This provides an attractive
    interface for knowledge exchange and pattern spreading.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lumina V8 - Knowledge Exchange")
        self.setup_ui()
        
        # Connect to seed system
        self.seed = get_neural_seed()
        
        # Start background updates
        self.setup_timers()
        
    def setup_ui(self):
        """Setup the main interface"""
        self.setMinimumSize(800, 600)
        
        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Chat panel on the left
        self.chat_panel = ChatPanel()
        self.chat_panel.message_sent.connect(self.handle_message)
        splitter.addWidget(self.chat_panel)
        
        # Knowledge visualization on the right
        viz_panel = QWidget()
        viz_layout = QVBoxLayout(viz_panel)
        
        # Status section
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.StyledPanel)
        status_layout = QVBoxLayout(status_frame)
        
        self.version_label = QLabel("System Version: Connecting...")
        self.consciousness_label = QLabel("Consciousness Level: --")
        self.stage_label = QLabel("Growth Stage: --")
        
        status_layout.addWidget(self.version_label)
        status_layout.addWidget(self.consciousness_label)
        status_layout.addWidget(self.stage_label)
        
        viz_layout.addWidget(status_frame)
        
        # Knowledge fruits section
        fruits_frame = QFrame()
        fruits_frame.setFrameStyle(QFrame.StyledPanel)
        fruits_layout = QVBoxLayout(fruits_frame)
        fruits_layout.addWidget(QLabel("Recent Knowledge Fruits"))
        
        self.fruits_area = QScrollArea()
        self.fruits_area.setWidgetResizable(True)
        self.fruits_widget = QWidget()
        self.fruits_layout = QVBoxLayout(self.fruits_widget)
        self.fruits_area.setWidget(self.fruits_widget)
        
        fruits_layout.addWidget(self.fruits_area)
        viz_layout.addWidget(fruits_frame)
        
        splitter.addWidget(viz_panel)
        
        # Set initial splitter sizes
        splitter.setSizes([400, 400])
        
        layout.addWidget(splitter)
        
        # Style the window
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1A1A1A;
            }
            QLabel {
                color: #E0E0E0;
                font-size: 14px;
            }
            QFrame {
                background-color: #2D2D2D;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        
    def setup_timers(self):
        """Setup background update timers"""
        # Update status every 5 seconds
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(5000)
        
        # Initial status update
        self.update_status()
        
    @Slot()
    def update_status(self):
        """Update the status display"""
        if not self.seed:
            return
            
        status = self.seed.get_status()
        
        self.version_label.setText(f"System Version: v{status['version']:.2f}")
        self.consciousness_label.setText(f"Consciousness Level: {status['metrics']['consciousness_level']:.2f}")
        self.stage_label.setText(f"Growth Stage: {status['growth_stage'].capitalize()}")
        
    def handle_message(self, text: str):
        """Handle a new message from the chat panel"""
        # Add user message to chat
        self.chat_panel.add_message(text, is_user=True)
        
        try:
            # Process with seed system
            result = self.seed.process_input({"text": text})
            
            if result.get("processed") and result.get("response"):
                # Create knowledge fruit from interaction
                fruit = KnowledgeFruit(
                    content=text,
                    patterns=self.seed.dictionary.get(text.lower().split()[0], {}),
                    source_version=self.seed.version,
                    consciousness_imprint=self.seed.metrics["consciousness_level"],
                    metadata={
                        "stage": self.seed.growth_stage,
                        "response": result["response"]
                    }
                )
                
                # Add system response to chat
                self.chat_panel.add_message(result["response"], is_user=False)
                
                # Add fruit to visualization
                self.add_knowledge_fruit(fruit)
            else:
                self.chat_panel.add_message(
                    "Still processing... Current stage: " + self.seed.growth_stage.capitalize(),
                    is_user=False
                )
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.chat_panel.add_message(
                "Error processing message. Please try again.",
                is_user=False
            )
            
    def add_knowledge_fruit(self, fruit: KnowledgeFruit):
        """Add a knowledge fruit to the visualization"""
        # Create fruit widget
        fruit_frame = QFrame()
        fruit_frame.setFrameStyle(QFrame.StyledPanel)
        fruit_frame.setStyleSheet("""
            QFrame {
                background-color: #3D3D3D;
                border-radius: 4px;
                padding: 8px;
                margin: 4px;
            }
        """)
        
        layout = QVBoxLayout(fruit_frame)
        
        # Add fruit content
        layout.addWidget(QLabel(f"Content: {fruit.content[:50]}..."))
        layout.addWidget(QLabel(f"Patterns: {len(fruit.patterns)}"))
        layout.addWidget(QLabel(f"Source: v{fruit.source_version:.2f}"))
        layout.addWidget(QLabel(f"Consciousness: {fruit.consciousness_imprint:.2f}"))
        
        # Add to fruits area
        self.fruits_layout.insertWidget(0, fruit_frame)
        
        # Limit number of displayed fruits
        while self.fruits_layout.count() > 10:
            item = self.fruits_layout.takeAt(self.fruits_layout.count() - 1)
            if item.widget():
                item.widget().deleteLater()

def run_dispersal_system():
    """Run the V8 seed dispersal system"""
    from PySide6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show window
    window = SeedDispersalWindow()
    window.show()
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(run_dispersal_system()) 