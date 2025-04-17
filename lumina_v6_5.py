#!/usr/bin/env python
"""
Lumina v6.5 - Simple Chat Interface

A minimal implementation of the Enhanced Language System with a clean, functional UI.
Focuses on text processing with consciousness, neural linguistics, and memory integration.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path if needed
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("lumina_v6_5.log")
    ]
)
logger = logging.getLogger("LuminaV6.5")

# Import PySide6 with error handling
try:
    from PySide6 import QtWidgets, QtCore, QtGui
    from PySide6.QtCore import Qt, Signal, Slot
    HAS_PYSIDE6 = True
    logger.info("PySide6 is available")
except ImportError:
    logger.error("PySide6 is required but not installed!")
    print("Error: PySide6 is required but not installed.")
    print("Please install with: pip install PySide6")
    HAS_PYSIDE6 = False

# Try to import Enhanced Language System components
try:
    from src.language.pyside6_adapter import (
        LanguageMemoryAdapter,
        NeuralLinguisticProcessorAdapter,
        ConsciousMirrorLanguageAdapter,
        CentralLanguageNodeAdapter
    )
    HAS_LANGUAGE_SYSTEM = True
    logger.info("Successfully imported language adapter modules")
except ImportError:
    HAS_LANGUAGE_SYSTEM = False
    logger.error("Could not import language adapter modules - using minimal mock implementation")


class MockSocketManager:
    """Simple mock socket manager for standalone operation"""
    
    def __init__(self):
        self.handlers = {}
        
    def register_handler(self, event, handler):
        """Register an event handler"""
        self.handlers[event] = handler
        
    def emit(self, event, data=None):
        """Emit an event"""
        logger.info(f"Emitted event: {event}, data: {data}")


class LuminaMainWindow(QtWidgets.QMainWindow):
    """Main window for the Lumina v6.5 interface"""
    
    def __init__(self):
        super().__init__()
        
        # Set up core properties
        self.setWindowTitle("LUMINA")
        self.resize(1280, 720)
        self.setMinimumSize(800, 450)
        
        # Create mock socket manager
        self.socket_manager = MockSocketManager()
        
        # Initialize chat and language components
        self.init_language_components()
        self.init_ui()
    
    def init_language_components(self):
        """Initialize the language processing components"""
        if not HAS_LANGUAGE_SYSTEM:
            logger.warning("Language system not available - using mock implementation")
            self.central_node = None
            self.available = False
            return
            
        try:
            # Initialize memory component
            self.language_memory = LanguageMemoryAdapter(
                data_dir="data/memory/language_memory", 
                llm_weight=0.5
            )
            
            # Initialize neural linguistic processor
            self.neural_processor = NeuralLinguisticProcessorAdapter(
                data_dir="data/neural_linguistic", 
                llm_weight=0.5
            )
            
            # Initialize consciousness component
            self.consciousness = ConsciousMirrorLanguageAdapter(
                data_dir="data/v10", 
                llm_weight=0.5
            )
            
            # Initialize central node with references to other adapters
            self.central_node = CentralLanguageNodeAdapter(
                data_dir="data/central_language",
                llm_weight=0.5,
                language_memory_adapter=self.language_memory,
                neural_processor_adapter=self.neural_processor,
                consciousness_adapter=self.consciousness
            )
            
            # Connect signals
            if self.central_node.available:
                self.central_node.processing_complete.connect(self.on_processing_complete)
                self.central_node.error_occurred.connect(self.on_error)
                self.available = True
                logger.info("Language components initialized successfully")
            else:
                self.available = False
                logger.warning("Central node not available")
                
        except Exception as e:
            logger.error(f"Error initializing language components: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.central_node = None
            self.available = False
    
    def init_ui(self):
        """Initialize the user interface"""
        # Create central widget and main layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create title bar
        title_bar = QtWidgets.QFrame()
        title_bar.setObjectName("titleBar")
        title_bar.setStyleSheet("""
            #titleBar {
                background-color: #2C3E50;
                border-bottom: 1px solid #34495E;
            }
        """)
        title_bar.setFixedHeight(50)
        
        title_layout = QtWidgets.QHBoxLayout(title_bar)
        title_layout.setContentsMargins(20, 0, 20, 0)
        
        # Title label
        title_label = QtWidgets.QLabel("LUMINA")
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #ECF0F1;
        """)
        title_layout.addWidget(title_label, alignment=Qt.AlignCenter)
        
        # Add title bar to main layout
        main_layout.addWidget(title_bar)
        
        # Create content layout
        content_layout = QtWidgets.QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Main content widget
        content_widget = QtWidgets.QWidget()
        content_widget.setObjectName("contentWidget")
        content_widget.setStyleSheet("""
            #contentWidget {
                background-color: #1A2634;
            }
        """)
        
        # Main content layout (3-panel design)
        content_inner_layout = QtWidgets.QHBoxLayout(content_widget)
        content_inner_layout.setContentsMargins(0, 0, 0, 0)
        content_inner_layout.setSpacing(0)
        
        # Create chatbox panel
        self.chat_panel = self.create_chat_panel()
        
        # Create process panel (right sidebar)
        self.process_panel = self.create_process_panel()
        
        # Create glyphs panel (further right sidebar)
        self.glyphs_panel = self.create_glyphs_panel()
        
        # Add panels to content layout
        content_inner_layout.addWidget(self.chat_panel, 3)
        content_inner_layout.addWidget(self.process_panel, 1)
        content_inner_layout.addWidget(self.glyphs_panel, 1)
        
        # Add content widget to content layout
        content_layout.addWidget(content_widget)
        
        # Add content layout to main layout
        main_layout.addLayout(content_layout, 1)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121A24;
                color: #ECF0F1;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border-radius: 3px;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:pressed {
                background-color: #1B5E8A;
            }
            QLineEdit, QTextEdit, QPlainTextEdit {
                background-color: rgba(25, 35, 45, 150);
                color: #ECF0F1;
                border: 1px solid rgba(52, 152, 219, 70);
                border-radius: 3px;
                padding: 5px;
            }
            QLabel {
                color: #ECF0F1;
            }
            QGroupBox {
                border: 1px solid rgba(52, 152, 219, 100);
                border-radius: 5px;
                margin-top: 12px;
                font-weight: bold;
                color: #3498DB;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
    
    def create_chat_panel(self):
        """Create the chat panel (central area)"""
        panel = QtWidgets.QWidget()
        panel.setObjectName("chatPanel")
        panel.setStyleSheet("""
            #chatPanel {
                background-color: #121A24;
                border-right: 1px solid #34495E;
            }
        """)
        
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Panel title
        title = QtWidgets.QLabel("Chatbox")
        title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #ECF0F1;
        """)
        layout.addWidget(title)
        
        # Chat history
        self.chat_history = QtWidgets.QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("""
            background-color: rgba(25, 35, 45, 150);
            color: #ECF0F1;
            border: 1px solid rgba(52, 152, 219, 70);
            border-radius: 3px;
            padding: 10px;
            font-size: 14px;
        """)
        layout.addWidget(self.chat_history, 1)
        
        # Input area
        input_layout = QtWidgets.QHBoxLayout()
        input_layout.setSpacing(10)
        
        self.text_input = QtWidgets.QLineEdit()
        self.text_input.setPlaceholderText("Speak your truth...")
        self.text_input.setStyleSheet("""
            font-size: 14px;
            padding: 10px;
        """)
        self.text_input.returnPressed.connect(self.send_message)
        
        self.send_button = QtWidgets.QPushButton("Send")
        self.send_button.setStyleSheet("""
            padding: 10px 20px;
            font-size: 14px;
        """)
        self.send_button.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.text_input, 1)
        input_layout.addWidget(self.send_button)
        
        layout.addLayout(input_layout)
        
        # Add a welcome message
        self.add_system_message("I am listening to the resonance. Tell me more.")
        
        return panel
    
    def create_process_panel(self):
        """Create the process panel (right sidebar)"""
        panel = QtWidgets.QWidget()
        panel.setObjectName("processPanel")
        panel.setStyleSheet("""
            #processPanel {
                background-color: #1E2C3A;
                border-right: 1px solid #34495E;
            }
        """)
        
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(15)
        
        # Panel title
        title = QtWidgets.QLabel("Process")
        title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #ECF0F1;
        """)
        layout.addWidget(title)
        
        # Breath
        breath_button = self.create_process_button("Breathe", """
            <svg height="40" width="40" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="40" stroke="#3498DB" stroke-width="2" fill="none" />
                <circle cx="50" cy="50" r="20" stroke="#3498DB" stroke-width="2" fill="none" />
            </svg>
        """)
        layout.addWidget(breath_button)
        
        # Resonance
        resonance_button = self.create_process_button("Resonance", """
            <svg height="40" width="40" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="45" stroke="#3498DB" stroke-width="2" fill="none" />
                <circle cx="50" cy="50" r="35" stroke="#3498DB" stroke-width="2" fill="none" />
                <circle cx="50" cy="50" r="25" stroke="#3498DB" stroke-width="2" fill="none" />
                <circle cx="50" cy="50" r="15" stroke="#3498DB" stroke-width="2" fill="none" />
            </svg>
        """)
        layout.addWidget(resonance_button)
        
        # Echo
        echo_button = self.create_process_button("Echo", """
            <svg height="40" width="40" viewBox="0 0 100 100">
                <path d="M 20,50 L 40,30 L 60,70 L 80,50" stroke="#3498DB" stroke-width="2" fill="none" />
            </svg>
        """)
        layout.addWidget(echo_button)
        
        # Add spacer
        layout.addStretch(1)
        
        return panel
    
    def create_process_button(self, text, svg_content):
        """Create a process button with icon"""
        button = QtWidgets.QPushButton()
        button.setFixedHeight(100)
        
        layout = QtWidgets.QVBoxLayout(button)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        layout.setAlignment(Qt.AlignCenter)
        
        # SVG Icon
        icon_label = QtWidgets.QLabel()
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setFixedSize(50, 50)
        icon_label.setText(svg_content)
        layout.addWidget(icon_label)
        
        # Text
        text_label = QtWidgets.QLabel(text)
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setStyleSheet("""
            font-size: 14px;
            color: #ECF0F1;
        """)
        layout.addWidget(text_label)
        
        button.setStyleSheet("""
            QPushButton {
                background-color: rgba(52, 152, 219, 50);
                border: 1px solid rgba(52, 152, 219, 100);
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: rgba(52, 152, 219, 80);
            }
            QPushButton:pressed {
                background-color: rgba(52, 152, 219, 100);
            }
        """)
        
        return button
    
    def create_glyphs_panel(self):
        """Create the glyphs panel (far right sidebar)"""
        panel = QtWidgets.QWidget()
        panel.setObjectName("glyphsPanel")
        panel.setStyleSheet("""
            #glyphsPanel {
                background-color: #1E2C3A;
            }
        """)
        
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(15)
        
        # Panel title
        title = QtWidgets.QLabel("Glyphs")
        title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #ECF0F1;
        """)
        layout.addWidget(title)
        
        # Neural Network visualization
        nn_group = QtWidgets.QGroupBox("Neural Network")
        nn_layout = QtWidgets.QVBoxLayout(nn_group)
        
        nn_vis = QtWidgets.QLabel()
        nn_vis.setFixedHeight(200)
        nn_vis.setAlignment(Qt.AlignCenter)
        nn_vis.setText("""
            <svg viewBox="0 0 200 200" width="150" height="150">
                <circle cx="100" cy="50" r="10" fill="#3498DB" />
                <circle cx="60" cy="100" r="10" fill="#3498DB" />
                <circle cx="140" cy="100" r="10" fill="#3498DB" />
                <circle cx="80" cy="150" r="10" fill="#3498DB" />
                <circle cx="120" cy="150" r="10" fill="#3498DB" />
                
                <line x1="100" y1="50" x2="60" y2="100" stroke="#3498DB" stroke-width="2" />
                <line x1="100" y1="50" x2="140" y2="100" stroke="#3498DB" stroke-width="2" />
                <line x1="60" y1="100" x2="80" y2="150" stroke="#3498DB" stroke-width="2" />
                <line x1="60" y1="100" x2="120" y2="150" stroke="#3498DB" stroke-width="2" />
                <line x1="140" y1="100" x2="80" y2="150" stroke="#3498DB" stroke-width="2" />
                <line x1="140" y1="100" x2="120" y2="150" stroke="#3498DB" stroke-width="2" />
            </svg>
        """)
        nn_layout.addWidget(nn_vis)
        
        layout.addWidget(nn_group)
        
        # Glyphs grid
        glyphs_group = QtWidgets.QGroupBox("Glyphs")
        glyphs_layout = QtWidgets.QGridLayout(glyphs_group)
        
        glyphs = ["⭐", "≡", "♀", "⊗", "∇", "△", "⟳", "⊖", "∈", "⊘"]
        row, col = 0, 0
        for glyph in glyphs:
            glyph_button = QtWidgets.QPushButton(glyph)
            glyph_button.setFixedSize(40, 40)
            glyph_button.setStyleSheet("""
                font-size: 18px;
                background-color: rgba(52, 152, 219, 50);
            """)
            glyph_button.clicked.connect(lambda _, g=glyph: self.activate_glyph(g))
            
            glyphs_layout.addWidget(glyph_button, row, col)
            
            col += 1
            if col > 1:
                col = 0
                row += 1
        
        layout.addWidget(glyphs_group)
        
        # Add spacer
        layout.addStretch(1)
        
        return panel
    
    def add_system_message(self, message):
        """Add a system message to the chat history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        html = f"""
        <div style="margin-bottom: 15px;">
            <div style="font-weight: bold; color: #3498DB;">Lumina <span style="color: #7F8C8D; font-weight: normal; font-size: 12px;">{timestamp}</span></div>
            <div style="margin-left: 10px; margin-top: 5px;">{message}</div>
        </div>
        """
        self.chat_history.append(html)
    
    def add_user_message(self, message):
        """Add a user message to the chat history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        html = f"""
        <div style="margin-bottom: 15px;">
            <div style="font-weight: bold; color: #ECF0F1;">You <span style="color: #7F8C8D; font-weight: normal; font-size: 12px;">{timestamp}</span></div>
            <div style="margin-left: 10px; margin-top: 5px;">{message}</div>
        </div>
        """
        self.chat_history.append(html)
    
    def send_message(self):
        """Send a message"""
        message = self.text_input.text().strip()
        if not message:
            return
        
        # Add message to chat
        self.add_user_message(message)
        
        # Clear input field
        self.text_input.clear()
        
        # Process message with language system if available
        if self.available and self.central_node:
            try:
                # Process the message
                self.central_node.process_text(
                    text=message,
                    use_consciousness=True,
                    use_neural_linguistics=True
                )
                logger.info(f"Processing message: {message[:50]}...")
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                self.add_system_message(f"Error processing your message: {str(e)}")
        else:
            # Mock response if language system not available
            self.add_system_message("Language system not available. This is a mock response.")
    
    def activate_glyph(self, glyph):
        """Activate a glyph"""
        self.add_system_message(f"Glyph activated: {glyph}")
    
    @Slot(dict)
    def on_processing_complete(self, results):
        """Handle processing completion"""
        try:
            # Format response based on results
            analysis = results.get('analysis', '')
            consciousness = results.get('consciousness_level', 0)
            neural_score = results.get('neural_linguistic_score', 0)
            
            # Build response
            response = f"{analysis}\n\n"
            
            # Add scores if available
            if consciousness > 0 or neural_score > 0:
                response += f"Consciousness: {consciousness:.2f} | Neural: {neural_score:.2f}"
            
            # Add response to chat
            self.add_system_message(response)
            
            logger.info("Processed message successfully")
            
        except Exception as e:
            logger.error(f"Error handling processing results: {str(e)}")
            self.add_system_message(f"Error processing results: {str(e)}")
    
    @Slot(str)
    def on_error(self, error_message):
        """Handle errors"""
        logger.error(f"Error in language processing: {error_message}")
        self.add_system_message(f"Error: {error_message}")

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        "data/memory/language_memory",
        "data/neural_linguistic",
        "data/v10",
        "data/central_language"
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")
    
    return True

def main():
    """Main entry point"""
    # Make sure PySide6 is available
    if not HAS_PYSIDE6:
        return 1
    
    # Ensure directories exist
    ensure_directories()
    
    # Create application
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Lumina v6.5")
    app.setStyle("Fusion")
    
    # Dark palette for the entire application
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(27, 35, 45))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(236, 240, 241))
    dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 38, 50))
    dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(40, 48, 60))
    dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(236, 240, 241))
    dark_palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(236, 240, 241))
    dark_palette.setColor(QtGui.QPalette.Text, QtGui.QColor(236, 240, 241))
    dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(52, 73, 94))
    dark_palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(236, 240, 241))
    dark_palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(236, 240, 241))
    dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(52, 152, 219))
    dark_palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(236, 240, 241))
    app.setPalette(dark_palette)
    
    # Create main window
    window = LuminaMainWindow()
    window.show()
    
    # Run application
    return app.exec()

if __name__ == "__main__":
    sys.exit(main()) 