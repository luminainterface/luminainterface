#!/usr/bin/env python
"""
V7 Main Widget

This module provides the main widget for the V7 Self-Learning Visualization System.
It integrates various UI components for monitoring and interacting with the V7 Node Consciousness system.
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Attempt to import Qt components from PySide6
    from PySide6.QtCore import Qt, QTimer, Signal, QSize
    from PySide6.QtGui import QPainter, QColor, QLinearGradient, QIcon, QPalette, QAction
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
        QPushButton, QSplitter, QTabWidget, QFrame, QToolBar, QSizePolicy
    )
    logger.info("Using PySide6 for V7 UI")
except ImportError:
    try:
        # Fall back to PyQt5 if PySide6 is not available
        from PyQt5.QtCore import Qt, QTimer, pyqtSignal as Signal, QSize
        from PyQt5.QtGui import QPainter, QColor, QLinearGradient, QIcon, QPalette, QAction
        from PyQt5.QtWidgets import (
            QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
            QPushButton, QSplitter, QTabWidget, QFrame, QToolBar, QSizePolicy
        )
        logger.info("Using PyQt5 for V7 UI")
    except ImportError:
        logger.error("Failed to import Qt. V7 UI components will not be available.")

# Import V7-specific components
try:
    from src.v7.ui.v7_socket_manager import V7SocketManager
except ImportError:
    logger.warning("V7SocketManager not found, using mock implementation")
    class V7SocketManager:
        def __init__(self):
            logger.info("Using mock V7SocketManager")
        
        def connect(self):
            logger.info("Mock socket manager connected")
            
        def disconnect(self):
            logger.info("Mock socket manager disconnected")

# Import visualization components
try:
    from src.v7.ui.v7_visualization_connector import V7VisualizationConnector
    from src.v7.ui.v7_visualization_widget import V7VisualizationWidget
    VISUALIZATION_AVAILABLE = True
    logger.info("V7 visualization components loaded successfully")
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("V7 visualization components not found, visualization features will be limited")

# Import language chat panel
try:
    from src.v7.ui.panels.language_chat_panel import LanguageChatPanel
    LANGUAGE_CHAT_AVAILABLE = True
    logger.info("Language Chat Panel loaded successfully")
except ImportError:
    LANGUAGE_CHAT_AVAILABLE = False
    logger.warning("Language Chat Panel not found, language chat features will not be available")


class PanelContainer(QFrame):
    """Container for UI panels with a title bar and styling"""
    
    def __init__(self, title, content_widget=None, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #1A2634;
                border: 1px solid #34495E;
                border-radius: 5px;
            }
        """)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create title bar
        title_bar = QWidget()
        title_bar.setFixedHeight(30)
        title_bar.setStyleSheet("""
            background-color: #34495E;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
        """)
        
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(10, 0, 10, 0)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            color: #ECF0F1;
            font-weight: bold;
        """)
        
        title_layout.addWidget(title_label)
        layout.addWidget(title_bar)
        
        # Add content widget if provided
        if content_widget:
            content_layout = QVBoxLayout()
            content_layout.setContentsMargins(10, 10, 10, 10)
            content_layout.addWidget(content_widget)
            layout.addLayout(content_layout, 1)


class PlaceholderPanel(QWidget):
    """Placeholder panel for components still in development"""
    
    def __init__(self, panel_name, accent_color="#3498DB", parent=None):
        super().__init__(parent)
        self.panel_name = panel_name
        self.accent_color = QColor(accent_color)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Add placeholder message
        message = QLabel(f"{panel_name} Component")
        message.setAlignment(Qt.AlignCenter)
        message.setStyleSheet(f"""
            color: {accent_color};
            font-weight: bold;
            font-size: 16px;
        """)
        
        status = QLabel("In Development")
        status.setAlignment(Qt.AlignCenter)
        status.setStyleSheet("""
            color: #7F8C8D;
            font-style: italic;
        """)
        
        layout.addStretch()
        layout.addWidget(message)
        layout.addWidget(status)
        layout.addStretch()


class V7MainWidget(QWidget):
    """Main widget for the V7 Self-Learning Visualization System"""
    
    def __init__(self, socket_manager, v6v7_connector=None, visualization_connector=None, parent=None, config=None):
        super().__init__(parent)
        self.socket_manager = socket_manager
        self.v6v7_connector = v6v7_connector
        
        # Create visualization connector if needed
        if visualization_connector is None and VISUALIZATION_AVAILABLE and self.v6v7_connector:
            self.visualization_connector = V7VisualizationConnector(v6v7_connector)
            logger.info("Created V7VisualizationConnector instance")
        else:
            self.visualization_connector = visualization_connector
        
        # Default configuration
        self.config = {
            "debug": False,
            "enable_v7": True,
            "enable_monday": False,
            "enable_auto_wiki": False,
            "enable_language_integration": True
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
            
        # Set up styling
        self.setStyleSheet("""
            QWidget {
                background-color: #121A24;
                color: #ECF0F1;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QScrollBar:vertical {
                border: none;
                background: #1E2C3A;
                width: 10px;
                margin: 0px;
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
        
        # Initialize UI
        self.initUI()
        
    def initUI(self):
        """Initialize the user interface"""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create toolbar
        toolbar = self.createToolbar()
        layout.addWidget(toolbar)
        
        # Create main content area with tabs for different views
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #34495E;
                background-color: #121A24;
            }
            QTabBar::tab {
                background-color: #1E2C3A;
                color: #7F8C8D;
                padding: 8px 16px;
                margin: 0px 2px 0px 0px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                border: 1px solid #34495E;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background-color: #121A24;
                color: #ECF0F1;
                border-bottom: 1px solid #121A24;
            }
            QTabBar::tab:hover:!selected {
                background-color: #2C3E50;
                color: #BDC3C7;
            }
        """)
        
        # Create different views
        self.dashboard_view = self.createDashboardView()
        self.knowledge_view = self.createKnowledgeView()
        self.learning_view = self.createLearningView()
        self.integration_view = self.createIntegrationView()
        self.language_chat_view = self.createLanguageChatView()
        
        # Add views to tabs
        self.tab_widget.addTab(self.dashboard_view, "Dashboard")
        self.tab_widget.addTab(self.knowledge_view, "Knowledge Explorer")
        self.tab_widget.addTab(self.learning_view, "Learning Pathways")
        self.tab_widget.addTab(self.integration_view, "System Integration")
        
        # Add language chat tab if available
        if LANGUAGE_CHAT_AVAILABLE and self.config.get("enable_language_integration", True):
            self.tab_widget.addTab(self.language_chat_view, "Language Chat")
        
        # Add tabs widget to main layout
        layout.addWidget(self.tab_widget, 1)
        
        # Create status bar
        status_bar = self.createStatusBar()
        layout.addWidget(status_bar)
    
    def createToolbar(self):
        """Create the toolbar with actions"""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        
        # Add the main menu button
        menu_button = QPushButton("☰ Menu")
        menu_button.setObjectName("toolbarMenuButton")
        menu_button.clicked.connect(self.toggle_menu)
        toolbar.addWidget(menu_button)
        
        # Add spacer to push help to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        
        # Add help button
        help_action = QAction(QIcon("icons/help.png"), "Help", self)
        help_action.triggered.connect(self.show_help)
        toolbar.addAction(help_action)
        
        return toolbar
    
    def createStatusBar(self):
        """Create the status bar with system information"""
        status_bar = QWidget()
        status_bar.setFixedHeight(25)
        status_bar.setStyleSheet("""
            background-color: #1A2634;
            border-top: 1px solid #34495E;
        """)
        
        layout = QHBoxLayout(status_bar)
        layout.setContentsMargins(10, 0, 10, 0)
        
        # Add status information
        status_label = QLabel("System: Active")
        status_label.setStyleSheet("""
            color: #2ECC71;
            font-size: 12px;
        """)
        
        # Add spacers and additional status info
        spacer1 = QWidget()
        spacer1.setSizePolicy(QWidget.expanding, QWidget.preferred)
        
        memory_label = QLabel("Memory Usage: Nominal")
        memory_label.setStyleSheet("""
            color: #7F8C8D;
            font-size: 12px;
        """)
        
        spacer2 = QWidget()
        spacer2.setSizePolicy(QWidget.expanding, QWidget.preferred)
        
        version_label = QLabel("V7.0.0.2")
        version_label.setStyleSheet("""
            color: #7F8C8D;
            font-size: 12px;
        """)
        
        # Add status components
        layout.addWidget(status_label)
        layout.addWidget(spacer1)
        layout.addWidget(memory_label)
        layout.addWidget(spacer2)
        layout.addWidget(version_label)
        
        return status_bar
    
    def createDashboardView(self):
        """Create the dashboard view with system overview"""
        dashboard = QWidget()
        dashboard_layout = QVBoxLayout(dashboard)
        dashboard_layout.setContentsMargins(20, 20, 20, 20)
        dashboard_layout.setSpacing(20)
        
        # Add title
        title_label = QLabel("V7 Node Consciousness Dashboard")
        title_label.setStyleSheet("""
            color: #3498DB;
            font-size: 24px;
            font-weight: bold;
        """)
        dashboard_layout.addWidget(title_label)
        
        # Add dashboard panels
        panels_layout = QHBoxLayout()
        panels_layout.setSpacing(20)
        
        # Consciousness metrics panel
        consciousness_panel = PanelContainer(
            "Consciousness Metrics",
            PlaceholderPanel("Consciousness Metrics", "#3498DB")
        )
        
        # Activity panel
        activity_panel = PanelContainer(
            "System Activity",
            PlaceholderPanel("Activity Monitor", "#E74C3C")
        )
        
        # Node status panel
        node_panel = PanelContainer(
            "Node Status",
            PlaceholderPanel("Node Status", "#2ECC71")
        )
        
        panels_layout.addWidget(consciousness_panel)
        panels_layout.addWidget(activity_panel)
        panels_layout.addWidget(node_panel)
        
        dashboard_layout.addLayout(panels_layout, 1)
        
        # Add some information text
        info_text = QLabel("V7 Node Consciousness provides self-awareness capabilities for neural network components")
        info_text.setStyleSheet("""
            color: #7F8C8D;
            font-size: 14px;
            font-style: italic;
        """)
        info_text.setAlignment(Qt.AlignCenter)
        dashboard_layout.addWidget(info_text)
        
        return dashboard
    
    def createKnowledgeView(self):
        """Create the knowledge explorer view"""
        knowledge = QWidget()
        knowledge_layout = QVBoxLayout(knowledge)
        knowledge_layout.setContentsMargins(20, 20, 20, 20)
        knowledge_layout.setSpacing(20)
        
        # Add title
        title_label = QLabel("Knowledge Explorer")
        title_label.setStyleSheet("""
            color: #3498DB;
            font-size: 24px;
            font-weight: bold;
        """)
        knowledge_layout.addWidget(title_label)
        
        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Knowledge Categories
        categories_panel = PanelContainer(
            "Knowledge Categories",
            PlaceholderPanel("Knowledge Categories", "#9B59B6")
        )
        
        # Right panel: Knowledge Graph
        graph_panel = PanelContainer(
            "Knowledge Graph",
            PlaceholderPanel("Knowledge Graph Visualization", "#E67E22")
        )
        
        # Add panels to splitter
        splitter.addWidget(categories_panel)
        splitter.addWidget(graph_panel)
        
        # Set initial sizes
        splitter.setSizes([300, 700])
        
        # Add splitter to layout
        knowledge_layout.addWidget(splitter, 1)
        
        return knowledge
    
    def createLearningView(self):
        """Create the learning pathways view"""
        learning = QWidget()
        learning_layout = QVBoxLayout(learning)
        learning_layout.setContentsMargins(20, 20, 20, 20)
        learning_layout.setSpacing(20)
        
        # Add title
        title_label = QLabel("Learning Pathways")
        title_label.setStyleSheet("""
            color: #3498DB;
            font-size: 24px;
            font-weight: bold;
        """)
        learning_layout.addWidget(title_label)
        
        # Add learning pathway visualization
        pathway_panel = PanelContainer(
            "Learning Path Visualization",
            PlaceholderPanel("Learning Pathway", "#27AE60")
        )
        
        learning_layout.addWidget(pathway_panel, 1)
        
        return learning
    
    def createIntegrationView(self):
        """Create the system integration view"""
        integration = QWidget()
        integration_layout = QVBoxLayout(integration)
        integration_layout.setContentsMargins(20, 20, 20, 20)
        integration_layout.setSpacing(20)
        
        # Add title
        title_label = QLabel("System Integration")
        title_label.setStyleSheet("""
            color: #3498DB;
            font-size: 24px;
            font-weight: bold;
        """)
        integration_layout.addWidget(title_label)
        
        # Create grid layout for integration panels
        grid_layout = QHBoxLayout()
        grid_layout.setSpacing(20)
        
        # Create integration panels
        v6_panel = PanelContainer(
            "V6 Integration",
            PlaceholderPanel("V6 Portal of Contradiction", "#16A085")
        )
        
        breath_panel = PanelContainer(
            "Breath Detection",
            PlaceholderPanel("Breath Detection System", "#8E44AD")
        )
        
        monday_panel = PanelContainer(
            "Monday Integration",
            PlaceholderPanel("Monday Node", "#D35400")
        )
        
        grid_layout.addWidget(v6_panel)
        grid_layout.addWidget(breath_panel)
        grid_layout.addWidget(monday_panel)
        
        integration_layout.addLayout(grid_layout, 1)
        
        return integration
    
    def createLanguageChatView(self):
        """Create the language chat view with enhanced language integration"""
        # If language chat panel is available, use it
        if LANGUAGE_CHAT_AVAILABLE:
            try:
                # Get language integration from system if available
                language_integration = None
                if hasattr(self.socket_manager, "get_component"):
                    language_integration = self.socket_manager.get_component("language_integration")
                
                # Create language chat panel
                chat_panel = LanguageChatPanel(
                    socket_manager=self.socket_manager,
                    language_integration=language_integration
                )
                
                # Return the panel directly
                return chat_panel
                
            except Exception as e:
                logger.error(f"Error creating language chat panel: {e}")
                # Fall back to placeholder on error
        
        # Create placeholder if language chat is not available
        language_chat = QWidget()
        language_chat_layout = QVBoxLayout(language_chat)
        language_chat_layout.setContentsMargins(20, 20, 20, 20)
        language_chat_layout.setSpacing(20)
        
        # Add title
        title_label = QLabel("Language Integration Chat")
        title_label.setStyleSheet("""
            color: #3498DB;
            font-size: 24px;
            font-weight: bold;
        """)
        language_chat_layout.addWidget(title_label)
        
        # Add placeholder
        placeholder = QWidget()
        placeholder_layout = QVBoxLayout(placeholder)
        
        message = QLabel("Language Chat Panel Not Available")
        message.setAlignment(Qt.AlignCenter)
        message.setStyleSheet("""
            color: #E74C3C;
            font-size: 18px;
            font-weight: bold;
        """)
        
        details = QLabel("The Language Chat Panel component could not be loaded.\nCheck that all required components are installed.")
        details.setAlignment(Qt.AlignCenter)
        details.setStyleSheet("""
            color: #7F8C8D;
            font-size: 14px;
            margin-top: 10px;
        """)
        
        placeholder_layout.addStretch()
        placeholder_layout.addWidget(message)
        placeholder_layout.addWidget(details)
        placeholder_layout.addStretch()
        
        language_chat_layout.addWidget(placeholder, 1)
        
        return language_chat
    
    def refreshAll(self):
        """Refresh all panels"""
        self.socket_manager.send_message("refresh", {"all": True})
        logger.info("Refresh all panels requested")

    def toggle_menu(self):
        """Toggle the main menu visibility"""
        # Implementation will depend on how your menu is structured
        logger.info("Menu toggle requested")
        # If you have a menu panel, you could toggle it here
        # For example:
        # if self.menu_panel.isVisible():
        #     self.menu_panel.hide()
        # else:
        #     self.menu_panel.show()

    def show_help(self):
        """Show the help dialog"""
        from PySide6.QtWidgets import QMessageBox
        
        help_text = """
        <h3>LUMINA V7 Help</h3>
        <p>This is the LUMINA V7 interface with the following features:</p>
        <ul>
            <li>Enhanced Language Integration</li>
            <li>Mistral AI Integration</li>
            <li>Neural Network Visualization</li>
            <li>Consciousness Metrics</li>
        </ul>
        <p>For more information, please refer to the documentation.</p>
        """
        
        QMessageBox.information(self, "LUMINA V7 Help", help_text)
        logger.info("Help dialog displayed")
    
    def paintEvent(self, event):
        """Custom paint event for gradient background"""
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor("#121A24"))  # Slightly lighter at top
        gradient.setColorAt(1, QColor("#0C1018"))  # Darker at bottom
        painter.fillRect(self.rect(), gradient)


# Running the widget directly for testing
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Create socket manager
    socket_manager = V7SocketManager()
    
    # Create and show the main widget
    widget = V7MainWidget(socket_manager)
    widget.setWindowTitle("V7 Self-Learning Visualization System")
    widget.resize(1600, 900)
    widget.show()
    
    sys.exit(app.exec()) 