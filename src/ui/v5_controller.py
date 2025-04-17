"""
V5 Controller for the Fractal Echo Visualization System

This module contains the main controller class for the V5 visualization system,
managing the UI components and connecting them to the backend plugins.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to use PySide6, fallback to PyQt5 if needed
try:
    from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                                QLabel, QFrame, QStackedWidget, QSplitter,
                                QToolBar, QToolButton, QMenu, QAction, QStatusBar)
    from PySide6.QtCore import Qt, QSize, Signal, Slot
    from PySide6.QtGui import QIcon, QAction
    logger.info("Using PySide6 for V5Controller")
    USING_PYSIDE6 = True
except ImportError:
    logger.warning("PySide6 not found, falling back to PyQt5")
    from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QLabel, QFrame, QStackedWidget, QSplitter,
                              QToolBar, QToolButton, QMenu, QAction, QStatusBar)
    from PyQt5.QtCore import Qt, QSize, pyqtSignal as Signal, pyqtSlot as Slot
    from PyQt5.QtGui import QIcon, QAction
    logger.info("Using PyQt5 for V5Controller")
    USING_PYSIDE6 = False

# Import V5 system components
sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    from src.v5.frontend_socket_manager import FrontendSocketManager
    from src.v5.node_socket import NodeSocket, QtSocketAdapter
    logger.info("Successfully imported V5 system components")
except ImportError as e:
    logger.error(f"Failed to import V5 system components: {e}")
    # Continue with limited functionality

# Try to import the UI panels
panels_imported = True
try:
    # Import panel components
    # First check if the panels are in the expected location
    v5_panels_path = Path(__file__).parent / "v5_components"
    panels_import_path = "src.ui.v5_components"
    if not v5_panels_path.exists():
        # Try alternative location
        if (Path(__file__).parent.parent / "v5" / "ui" / "panels").exists():
            panels_import_path = "src.v5.ui.panels"
        else:
            # Just use what we've defined directly
            panels_import_path = ".."
            
    # Import panels
    try:
        sys.path.append(str(Path(__file__).parent.parent.parent))
        FractalPatternPanel = __import__(f"{panels_import_path}.fractal_pattern_panel", fromlist=["FractalPatternPanel"]).FractalPatternPanel
        NodeConsciousnessPanel = __import__(f"{panels_import_path}.node_consciousness_panel", fromlist=["NodeConsciousnessPanel"]).NodeConsciousnessPanel
        
        # Try to import from ui/components if needed
        try:
            NetworkVisualizationPanel = __import__("src.ui.components.NetworkVisualizationPanel", fromlist=["NetworkVisualizationPanel"]).NetworkVisualizationPanel
        except ImportError:
            NetworkVisualizationPanel = __import__("src.ui.components.NetworkVisualizationPanelPySide6", fromlist=["NetworkVisualizationPanelPySide6"]).NetworkVisualizationPanelPySide6
        try:
            MemorySynthesisPanel = __import__("src.v5.ui.panels.memory_synthesis_panel", fromlist=["MemorySynthesisPanel"]).MemorySynthesisPanel
        except ImportError:
            MemorySynthesisPanel = __import__("src.ui.components.MemoryScrollPanel", fromlist=["MemoryScrollPanel"]).MemoryScrollPanel
            
        logger.info("Successfully imported panel components")
    except ImportError as e:
        logger.error(f"Error importing panels: {e}")
        panels_imported = False
except Exception as e:
    logger.error(f"Failed to import panel components: {e}")
    panels_imported = False

def create_placeholder_icons():
    """Create placeholder icons if they don't exist"""
    icons_dir = Path("assets/icons")
    icons_dir.mkdir(parents=True, exist_ok=True)
    
    # Create placeholders for each icon we need
    icon_names = [
        "fractal", "consciousness", "network", "memory", 
        "home", "settings", "help", "about",
        "play", "pause", "stop", "restart"
    ]
    
    for name in icon_names:
        icon_path = icons_dir / f"{name}.png"
        if not icon_path.exists():
            # Create empty icon file
            with open(icon_path, "wb") as f:
                # Minimal valid PNG
                f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x06\x00\x00\x00\x1f\xf3\xffa\x00\x00\x00\x01sRGB\x00\xae\xce\x1c\xe9\x00\x00\x00\x04IDAT8Oc\xf8\x0f\x04\x0c\x18\x10\x00\x00\x08\x00\x01\x00\x01\x00\x01\xf4\x86\x13\xe9\x00\x00\x00\x00IEND\xaeB`\x82')

class V5Controller(QMainWindow):
    """Main controller for the V5 GUI application"""
    
    def __init__(self):
        super().__init__()
        logger.info("Initializing V5Controller")
        
        # Create placeholder icons if needed
        create_placeholder_icons()
        
        # Initialize socket manager and plugins
        self.initializePlugins()
        
        # Initialize UI
        self.initUI()
        
    def initializePlugins(self):
        """Initialize all V5 plugins"""
        try:
            # Create frontend socket manager
            self.socket_manager = FrontendSocketManager()
            logger.info("Created FrontendSocketManager")
            
            # Initialize core plugins (if possible)
            try:
                from src.v5.neural_state_plugin import NeuralStatePlugin
                neural_state = NeuralStatePlugin()
                self.socket_manager.register_plugin(neural_state)
                logger.info("Registered NeuralStatePlugin")
            except ImportError as e:
                logger.warning(f"Could not import NeuralStatePlugin: {e}")
                
            try:
                from src.v5.pattern_processor_plugin import PatternProcessorPlugin
                pattern_processor = PatternProcessorPlugin()
                self.socket_manager.register_plugin(pattern_processor)
                logger.info("Registered PatternProcessorPlugin")
            except ImportError as e:
                logger.warning(f"Could not import PatternProcessorPlugin: {e}")
                
            try:
                from src.v5.consciousness_analytics_plugin import ConsciousnessAnalyticsPlugin
                consciousness_analytics = ConsciousnessAnalyticsPlugin()
                self.socket_manager.register_plugin(consciousness_analytics)
                logger.info("Registered ConsciousnessAnalyticsPlugin")
            except ImportError as e:
                logger.warning(f"Could not import ConsciousnessAnalyticsPlugin: {e}")
                
            try:
                from src.v5.api_service_plugin import ApiPlugin
                api_service = ApiPlugin()
                self.socket_manager.register_plugin(api_service)
                logger.info("Registered ApiPlugin")
            except ImportError as e:
                logger.warning(f"Could not import ApiPlugin: {e}")
                
            try:
                from src.v5.language_memory_integration import LanguageMemoryIntegrationPlugin
                language_memory = LanguageMemoryIntegrationPlugin()
                self.socket_manager.register_plugin(language_memory)
                logger.info("Registered LanguageMemoryIntegrationPlugin")
            except ImportError as e:
                logger.warning(f"Could not import LanguageMemoryIntegrationPlugin: {e}")
                
        except Exception as e:
            logger.error(f"Error initializing plugins: {e}")
            self.socket_manager = None
    
    def initUI(self):
        """Initialize the user interface"""
        # Main window setup
        self.setWindowTitle("Neural Network Visualizer - V5")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet("background-color: #1A1A30;")
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create navigation panel (left sidebar)
        nav_panel = self.create_navigation_panel()
        
        # Create content area
        self.content_stack = QStackedWidget()
        self.content_stack.setStyleSheet("background-color: #212140;")
        
        # Add panels to content stack
        self.add_content_panels()
        
        # Add panels to main layout
        main_layout.addWidget(nav_panel)
        main_layout.addWidget(self.content_stack, 1)  # Content takes remaining space
        
        # Set dark theme
        self.set_dark_theme()
        
        # Create status bar
        self.statusBar().showMessage("Ready")
    
    def create_navigation_panel(self):
        """Create the navigation panel (left sidebar)"""
        nav_panel = QFrame()
        nav_panel.setFrameShape(QFrame.StyledPanel)
        nav_panel.setStyleSheet("""
            QFrame {
                background-color: #1A1A30;
                border-right: 1px solid #333355;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                color: #CCCCDD;
                font-size: 14px;
                padding: 15px;
                text-align: left;
                border-radius: 0px;
            }
            QToolButton:hover {
                background-color: rgba(255, 255, 255, 0.1);
            }
            QToolButton:pressed {
                background-color: rgba(255, 255, 255, 0.2);
            }
            QToolButton:checked {
                background-color: #3A3A60;
                color: white;
            }
            QLabel {
                color: #CCCCDD;
            }
        """)
        nav_panel.setFixedWidth(250)
        
        # Layout for navigation panel
        nav_layout = QVBoxLayout(nav_panel)
        nav_layout.setContentsMargins(10, 20, 10, 20)
        nav_layout.setSpacing(5)
        
        # App title
        title_label = QLabel("V5 Visualization")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 20px;")
        nav_layout.addWidget(title_label)
        
        # Add navigation buttons
        nav_buttons = [
            {"name": "fractal", "text": "Fractal Patterns", "icon": "assets/icons/fractal.png"},
            {"name": "consciousness", "text": "Node Consciousness", "icon": "assets/icons/consciousness.png"},
            {"name": "network", "text": "Network Structure", "icon": "assets/icons/network.png"},
            {"name": "memory", "text": "Memory Synthesis", "icon": "assets/icons/memory.png"}
        ]
        
        for i, btn_info in enumerate(nav_buttons):
            btn = QToolButton()
            btn.setText(btn_info["text"])
            btn.setIcon(QIcon(btn_info["icon"]))
            btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            btn.setCheckable(True)
            btn.setFixedHeight(50)
            btn.setIconSize(QSize(24, 24))
            if i == 0:  # Select first button by default
                btn.setChecked(True)
            
            # Connect button to action
            btn.clicked.connect(lambda checked, name=btn_info["name"]: self.on_nav_button_clicked(name))
            
            nav_layout.addWidget(btn)
        
        # Add stretch to push buttons to top
        nav_layout.addStretch(1)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #333355;")
        nav_layout.addWidget(separator)
        
        # Add bottom buttons
        bottom_buttons = [
            {"name": "settings", "text": "Settings", "icon": "assets/icons/settings.png"},
            {"name": "help", "text": "Help", "icon": "assets/icons/help.png"},
            {"name": "about", "text": "About", "icon": "assets/icons/about.png"}
        ]
        
        for btn_info in bottom_buttons:
            btn = QToolButton()
            btn.setText(btn_info["text"])
            btn.setIcon(QIcon(btn_info["icon"]))
            btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            btn.setFixedHeight(50)
            btn.setIconSize(QSize(24, 24))
            
            # Connect button to action
            btn.clicked.connect(lambda checked, name=btn_info["name"]: self.on_nav_button_clicked(name))
            
            nav_layout.addWidget(btn)
        
        return nav_panel
    
    def add_content_panels(self):
        """Add content panels to the stack"""
        # Check if panels are imported
        if not panels_imported:
            # Add placeholder panels
            self.add_placeholder_panels()
            return
        
        try:
            # Fractal pattern panel
            self.fractal_panel = FractalPatternPanel(self.socket_manager)
            self.content_stack.addWidget(self.fractal_panel)
            
            # Node consciousness panel
            self.consciousness_panel = NodeConsciousnessPanel(self.socket_manager)
            self.content_stack.addWidget(self.consciousness_panel)
            
            # Network visualization panel
            self.network_panel = NetworkVisualizationPanel()
            self.content_stack.addWidget(self.network_panel)
            
            # Memory synthesis panel
            self.memory_panel = MemorySynthesisPanel(self.socket_manager)
            self.content_stack.addWidget(self.memory_panel)
            
            # Settings panel (placeholder)
            self.settings_panel = self.create_placeholder_panel("Settings")
            self.content_stack.addWidget(self.settings_panel)
            
            # Help panel (placeholder)
            self.help_panel = self.create_placeholder_panel("Help")
            self.content_stack.addWidget(self.help_panel)
            
            # About panel (placeholder)
            self.about_panel = self.create_placeholder_panel("About")
            self.content_stack.addWidget(self.about_panel)
            
            logger.info("Added content panels to stack")
        except Exception as e:
            logger.error(f"Error adding content panels: {e}")
            self.add_placeholder_panels()
    
    def add_placeholder_panels(self):
        """Add placeholder panels when real panels can't be loaded"""
        # Create placeholder panels for each section
        self.fractal_panel = self.create_placeholder_panel("Fractal Pattern Visualization")
        self.consciousness_panel = self.create_placeholder_panel("Node Consciousness Visualization")
        self.network_panel = self.create_placeholder_panel("Network Structure Visualization")
        self.memory_panel = self.create_placeholder_panel("Memory Synthesis")
        self.settings_panel = self.create_placeholder_panel("Settings")
        self.help_panel = self.create_placeholder_panel("Help")
        self.about_panel = self.create_placeholder_panel("About")
        
        # Add to content stack
        self.content_stack.addWidget(self.fractal_panel)
        self.content_stack.addWidget(self.consciousness_panel)
        self.content_stack.addWidget(self.network_panel)
        self.content_stack.addWidget(self.memory_panel)
        self.content_stack.addWidget(self.settings_panel)
        self.content_stack.addWidget(self.help_panel)
        self.content_stack.addWidget(self.about_panel)
        
        logger.warning("Using placeholder panels due to import errors")
    
    def create_placeholder_panel(self, title):
        """Create a placeholder panel with the given title"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(50, 50, 50, 50)
        
        # Add title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #CCCCDD; margin-bottom: 20px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Add message
        message = QLabel("This panel is not yet available or could not be loaded.")
        message.setStyleSheet("font-size: 18px; color: #AAAACC;")
        message.setAlignment(Qt.AlignCenter)
        message.setWordWrap(True)
        layout.addWidget(message)
        
        # Add stretch
        layout.addStretch(1)
        
        return panel
    
    def on_nav_button_clicked(self, name):
        """Handle navigation button clicks"""
        logger.info(f"Navigation button clicked: {name}")
        
        # Map button names to panel indices
        panel_map = {
            "fractal": 0,
            "consciousness": 1,
            "network": 2,
            "memory": 3,
            "settings": 4,
            "help": 5,
            "about": 6
        }
        
        # Show the selected panel
        if name in panel_map:
            self.content_stack.setCurrentIndex(panel_map[name])
            self.statusBar().showMessage(f"Viewing {name} panel")
    
    def set_dark_theme(self):
        """Set dark theme for the application"""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #212140;
                color: #CCCCDD;
            }
            QScrollBar:vertical {
                border: none;
                background: #1A1A30;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #333355;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                border: none;
                background: #1A1A30;
                height: 10px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #333355;
                min-width: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QTabWidget::pane {
                border: 1px solid #333355;
                top: -1px;
                background-color: #1A1A30;
            }
            QTabBar::tab {
                background-color: #1A1A30;
                color: #AAAACC;
                padding: 8px 12px;
                border: 1px solid #333355;
                border-bottom: none;
                min-width: 100px;
            }
            QTabBar::tab:selected {
                background-color: #2A2A50;
                color: #FFFFFF;
            }
            QTabBar::tab:hover {
                background-color: #333355;
            }
            QGroupBox {
                border: 1px solid #333355;
                border-radius: 5px;
                margin-top: 20px;
                font-weight: bold;
                color: #CCCCDD;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #3A3A60;
                color: #CCCCDD;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #4A4A80;
            }
            QPushButton:pressed {
                background-color: #5A5AA0;
            }
            QLineEdit, QTextEdit, QComboBox {
                background-color: #1A1A30;
                color: #CCCCDD;
                border: 1px solid #333355;
                border-radius: 5px;
                padding: 5px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #333355;
                height: 8px;
                background: #1A1A30;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3A3A60;
                border: 1px solid #5A5AA0;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QStatusBar {
                background-color: #1A1A30;
                color: #AAAACC;
            }
        """)
        
    def closeEvent(self, event):
        """Handle application close event"""
        # Clean up plugins
        if hasattr(self, 'socket_manager') and self.socket_manager:
            try:
                self.socket_manager.cleanup()
                logger.info("Cleaned up socket manager")
            except Exception as e:
                logger.error(f"Error cleaning up socket manager: {e}")
        
        # Clean up panels
        try:
            if hasattr(self, 'fractal_panel') and hasattr(self.fractal_panel, 'cleanup'):
                self.fractal_panel.cleanup()
            if hasattr(self, 'consciousness_panel') and hasattr(self.consciousness_panel, 'cleanup'):
                self.consciousness_panel.cleanup()
            if hasattr(self, 'network_panel') and hasattr(self.network_panel, 'cleanup'):
                self.network_panel.cleanup()
            if hasattr(self, 'memory_panel') and hasattr(self.memory_panel, 'cleanup'):
                self.memory_panel.cleanup()
            logger.info("Cleaned up panels")
        except Exception as e:
            logger.error(f"Error cleaning up panels: {e}")
        
        logger.info("V5 application closing")
        event.accept() 