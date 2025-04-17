"""
Main Window for V5 PySide6 Client

This module contains the MainWindow class for the V5 PySide6 client.
"""

import os
import sys
import logging
from pathlib import Path

# Try to import from PySide6
try:
    from PySide6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
        QTabWidget, QSplitter, QToolBar, QAction, QStatusBar,
        QLabel, QPushButton, QFrame, QSizePolicy, QMenu,
        QDockWidget, QScrollArea
    )
    from PySide6.QtCore import Qt, QSize, Signal, Slot, QTimer
    from PySide6.QtGui import QIcon, QPixmap, QFont
    USING_PYSIDE6 = True
except ImportError:
    from PyQt5.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
        QTabWidget, QSplitter, QToolBar, QAction, QStatusBar,
        QLabel, QPushButton, QFrame, QSizePolicy, QMenu,
        QDockWidget, QScrollArea
    )
    from PyQt5.QtCore import Qt, QSize, pyqtSignal as Signal, pyqtSlot as Slot, QTimer
    from PyQt5.QtGui import QIcon, QPixmap, QFont
    USING_PYSIDE6 = False

logger = logging.getLogger(__name__)

# Add parent directory to path if needed
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

class MainWindow(QMainWindow):
    """Main window for the V5 PySide6 client"""
    
    def __init__(self, mock_mode=False, enable_plugins=True):
        """
        Initialize the main window
        
        Args:
            mock_mode: Use mock mode for testing without backend services
            enable_plugins: Enable plugin discovery and loading
        """
        super().__init__()
        
        self.mock_mode = mock_mode
        self.enable_plugins = enable_plugins
        self.socket_manager = None
        
        # Set up UI
        self.setup_ui()
        
        # Set up status message
        self.status_message("Ready")
        
        logger.info("MainWindow initialized")
    
    def setup_ui(self):
        """Set up the user interface"""
        # Set window properties
        self.setWindowTitle("V5 Fractal Echo Visualization")
        self.resize(1200, 800)
        
        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Create splitter for sidebar and main content
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)
        
        # Create sidebar
        self.sidebar = self.create_sidebar()
        self.splitter.addWidget(self.sidebar)
        
        # Create main content
        self.content = self.create_content()
        self.splitter.addWidget(self.content)
        
        # Set splitter proportions (25% sidebar, 75% content)
        self.splitter.setSizes([300, 900])
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Status labels
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label, 1)
        
        self.memory_status = QLabel("Memory: Not connected")
        self.status_bar.addPermanentWidget(self.memory_status)
        
        self.neural_weight_status = QLabel("Neural Weight: 0.5")
        self.status_bar.addPermanentWidget(self.neural_weight_status)
    
    def create_sidebar(self):
        """Create the sidebar with controls"""
        # Sidebar container
        sidebar = QWidget()
        sidebar.setMinimumWidth(250)
        sidebar.setMaximumWidth(400)
        
        # Sidebar layout
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        
        # Add title
        title_label = QLabel("V5 Fractal Echo Visualization")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4B6EAF;")
        sidebar_layout.addWidget(title_label)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        sidebar_layout.addWidget(separator)
        
        # Add sections
        neural_section = self.create_neural_weight_section()
        sidebar_layout.addWidget(neural_section)
        
        memory_section = self.create_memory_controls_section()
        sidebar_layout.addWidget(memory_section)
        
        # Add stretcher to push everything to the top
        sidebar_layout.addStretch(1)
        
        # Create bottom buttons section
        bottom_section = QWidget()
        bottom_layout = QVBoxLayout(bottom_section)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add plugin management button
        if self.enable_plugins:
            plugins_button = QPushButton("Manage Plugins")
            plugins_button.clicked.connect(self.on_manage_plugins)
            bottom_layout.addWidget(plugins_button)
        
        # Add settings button
        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self.on_settings)
        bottom_layout.addWidget(settings_button)
        
        sidebar_layout.addWidget(bottom_section)
        
        return sidebar
    
    def create_neural_weight_section(self):
        """Create neural weight control section"""
        # Section widget
        section = QWidget()
        
        # Section layout
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 10)
        
        # Section title
        title = QLabel("Neural Network Weight")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)
        
        # Neural weight slider
        from PySide6.QtWidgets import QSlider
        self.nn_weight_slider = QSlider(Qt.Horizontal)
        self.nn_weight_slider.setRange(0, 100)
        self.nn_weight_slider.setValue(50)  # 0.5 default
        self.nn_weight_slider.setTickPosition(QSlider.TicksBelow)
        self.nn_weight_slider.setTickInterval(10)
        self.nn_weight_slider.valueChanged.connect(self.on_neural_weight_changed)
        layout.addWidget(self.nn_weight_slider)
        
        # Weight display
        weight_layout = QHBoxLayout()
        weight_layout.addWidget(QLabel("Language"))
        
        self.weight_value_label = QLabel("0.5")
        self.weight_value_label.setAlignment(Qt.AlignCenter)
        weight_layout.addWidget(self.weight_value_label)
        
        weight_layout.addWidget(QLabel("Neural"))
        
        layout.addLayout(weight_layout)
        
        return section
    
    def create_memory_controls_section(self):
        """Create memory controls section"""
        # Section widget
        section = QWidget()
        
        # Section layout
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 10)
        
        # Section title
        title = QLabel("Memory System")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)
        
        # Connect to memory button
        self.connect_memory_button = QPushButton("Connect to Memory System")
        self.connect_memory_button.clicked.connect(self.on_connect_memory)
        layout.addWidget(self.connect_memory_button)
        
        # Memory mode selection
        from PySide6.QtWidgets import QComboBox
        self.memory_mode_combo = QComboBox()
        self.memory_mode_combo.addItems(["Contextual", "Combined", "Synthesized"])
        self.memory_mode_combo.setCurrentText("Combined")
        self.memory_mode_combo.currentTextChanged.connect(self.on_memory_mode_changed)
        
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Memory Mode:"))
        mode_layout.addWidget(self.memory_mode_combo)
        
        layout.addLayout(mode_layout)
        
        return section
    
    def create_content(self):
        """Create the main content area with tabs"""
        # Content container
        content = QWidget()
        
        # Content layout
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setMovable(True)
        content_layout.addWidget(self.tab_widget)
        
        # Add "Loading" placeholder
        loading_widget = QWidget()
        loading_layout = QVBoxLayout(loading_widget)
        
        loading_label = QLabel("Loading panels...")
        loading_label.setAlignment(Qt.AlignCenter)
        loading_label.setStyleSheet("font-size: 16px; color: #888;")
        
        loading_layout.addStretch(1)
        loading_layout.addWidget(loading_label)
        loading_layout.addStretch(1)
        
        self.tab_widget.addTab(loading_widget, "Loading...")
        
        return content
    
    def create_menu_bar(self):
        """Create the menu bar"""
        # File menu
        file_menu = self.menuBar().addMenu("&File")
        
        # Connect action
        connect_action = QAction("&Connect to Memory System", self)
        connect_action.setStatusTip("Connect to the Language Memory System")
        connect_action.triggered.connect(self.on_connect_memory)
        file_menu.addAction(connect_action)
        
        # Settings action
        settings_action = QAction("&Settings", self)
        settings_action.setStatusTip("Configure application settings")
        settings_action.triggered.connect(self.on_settings)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = self.menuBar().addMenu("&View")
        
        # Toggle sidebar action
        self.toggle_sidebar_action = QAction("Toggle &Sidebar", self)
        self.toggle_sidebar_action.setStatusTip("Show or hide the sidebar")
        self.toggle_sidebar_action.triggered.connect(self.toggle_sidebar)
        view_menu.addAction(self.toggle_sidebar_action)
        
        view_menu.addSeparator()
        
        # Theme submenu
        theme_menu = view_menu.addMenu("&Theme")
        
        # Light theme action
        light_theme_action = QAction("&Light", self)
        light_theme_action.setStatusTip("Switch to light theme")
        light_theme_action.triggered.connect(lambda: self.change_theme("light"))
        theme_menu.addAction(light_theme_action)
        
        # Dark theme action
        dark_theme_action = QAction("&Dark", self)
        dark_theme_action.setStatusTip("Switch to dark theme")
        dark_theme_action.triggered.connect(lambda: self.change_theme("dark"))
        theme_menu.addAction(dark_theme_action)
        
        # System theme action
        system_theme_action = QAction("&System", self)
        system_theme_action.setStatusTip("Use system theme")
        system_theme_action.triggered.connect(lambda: self.change_theme("system"))
        theme_menu.addAction(system_theme_action)
        
        # Help menu
        help_menu = self.menuBar().addMenu("&Help")
        
        # About action
        about_action = QAction("&About", self)
        about_action.setStatusTip("Show information about the application")
        about_action.triggered.connect(self.on_about)
        help_menu.addAction(about_action)
    
    def create_toolbar(self):
        """Create the main toolbar"""
        # Create toolbar
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setIconSize(QSize(24, 24))
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)
        
        # Connect action
        connect_action = QAction("Connect", self)
        connect_action.setStatusTip("Connect to Language Memory System")
        connect_action.triggered.connect(self.on_connect_memory)
        self.toolbar.addAction(connect_action)
        
        self.toolbar.addSeparator()
        
        # Add panel button/menu
        add_panel_action = QAction("Add Panel", self)
        add_panel_action.setStatusTip("Add a new panel")
        
        # Create panels menu
        panels_menu = QMenu(self)
        
        # Add menu entries (these will be populated later)
        panels_menu.addAction("Loading panels...")
        
        add_panel_action.setMenu(panels_menu)
        self.toolbar.addAction(add_panel_action)
        self.panels_menu = panels_menu
    
    def initialize_components(self):
        """Initialize components after the window is shown"""
        # Clear the "Loading" tab
        self.tab_widget.clear()
        
        # Initialize socket manager
        self.initialize_socket_manager()
        
        # Load default panels
        self.load_default_panels()
    
    def initialize_socket_manager(self):
        """Initialize the socket manager"""
        try:
            # Import the socket manager
            from v5_client.core.socket_manager import ClientSocketManager
            
            # Create socket manager
            self.socket_manager = ClientSocketManager(mock_mode=self.mock_mode)
            
            if self.enable_plugins:
                # Start plugin discovery
                self.socket_manager.start_plugin_discovery()
            
            # Update status
            self.status_message("Socket manager initialized")
            
        except ImportError as e:
            logger.error(f"Failed to import socket manager: {e}")
            self.status_message("Failed to initialize socket manager", error=True)
    
    def load_default_panels(self):
        """Load default panels"""
        try:
            # Import panels
            from v5_client.ui.panels import get_available_panels
            
            # Create panels dictionary
            panels = get_available_panels()
            
            # Create panels menu
            self.panels_menu.clear()
            
            for panel_id, panel_class in panels.items():
                # Add to menu
                action = self.panels_menu.addAction(panel_class.get_panel_name())
                action.setData(panel_id)
                action.triggered.connect(lambda checked, pid=panel_id: self.add_panel(pid))
            
            # Add default panels
            self.add_panel("fractal_pattern")
            self.add_panel("memory_synthesis")
            self.add_panel("conversation")
            
            # Update status
            self.status_message("Default panels loaded")
            
        except ImportError as e:
            logger.error(f"Failed to load panels: {e}")
            self.status_message("Failed to load panels", error=True)
    
    def add_panel(self, panel_id):
        """
        Add a panel to the main interface
        
        Args:
            panel_id: ID of the panel to add
        """
        try:
            # Import panel class
            if panel_id == "fractal_pattern":
                from v5_client.ui.panels.fractal_pattern_panel import FractalPatternPanel
                panel_class = FractalPatternPanel
            elif panel_id == "memory_synthesis":
                from v5_client.ui.panels.memory_synthesis_panel import MemorySynthesisPanel
                panel_class = MemorySynthesisPanel
            elif panel_id == "node_consciousness":
                from v5_client.ui.panels.node_consciousness_panel import NodeConsciousnessPanel
                panel_class = NodeConsciousnessPanel
            elif panel_id == "conversation":
                from v5_client.ui.panels.conversation_panel import ConversationPanel
                panel_class = ConversationPanel
            else:
                logger.error(f"Unknown panel ID: {panel_id}")
                return
            
            # Create panel instance
            panel = panel_class(self.socket_manager)
            
            # Add to tab widget
            self.tab_widget.addTab(panel, panel.get_panel_name())
            
            # Set as current tab
            self.tab_widget.setCurrentWidget(panel)
            
            # Update status
            self.status_message(f"Added {panel.get_panel_name()} panel")
            
        except Exception as e:
            logger.error(f"Failed to add panel {panel_id}: {e}")
            self.status_message(f"Failed to add panel: {str(e)}", error=True)
    
    def on_neural_weight_changed(self, value):
        """
        Handle changes to the neural weight slider
        
        Args:
            value: New slider value (0-100)
        """
        # Convert to 0-1 range
        weight = value / 100.0
        
        # Update label
        self.weight_value_label.setText(f"{weight:.2f}")
        
        # Update status bar
        self.neural_weight_status.setText(f"Neural Weight: {weight:.2f}")
        
        # Update all panels
        for i in range(self.tab_widget.count()):
            panel = self.tab_widget.widget(i)
            if hasattr(panel, 'set_neural_weight'):
                panel.set_neural_weight(weight)
    
    def on_memory_mode_changed(self, mode):
        """
        Handle changes to the memory mode
        
        Args:
            mode: New memory mode ("Contextual", "Combined", "Synthesized")
        """
        logger.info(f"Memory mode changed to {mode}")
        
        # Update all panels that support memory mode
        for i in range(self.tab_widget.count()):
            panel = self.tab_widget.widget(i)
            if hasattr(panel, 'set_memory_mode'):
                panel.set_memory_mode(mode.lower())
    
    def on_connect_memory(self):
        """Handle connect to memory button"""
        try:
            # Import memory bridge
            from v5_client.bridge.language_memory_bridge import LanguageMemoryBridge
            
            # Create bridge
            bridge = LanguageMemoryBridge(mock_mode=self.mock_mode)
            
            # Connect to memory system
            if bridge.connect():
                self.memory_status.setText("Memory: Connected")
                self.connect_memory_button.setText("Disconnect from Memory System")
                self.connect_memory_button.clicked.disconnect()
                self.connect_memory_button.clicked.connect(self.on_disconnect_memory)
                
                # Store bridge in socket manager
                if self.socket_manager:
                    self.socket_manager.set_memory_bridge(bridge)
                
                # Update status
                self.status_message("Connected to Language Memory System")
            else:
                self.status_message("Failed to connect to memory system", error=True)
        except ImportError as e:
            logger.error(f"Failed to import memory bridge: {e}")
            self.status_message("Failed to load memory bridge", error=True)
    
    def on_disconnect_memory(self):
        """Handle disconnect from memory button"""
        # Update status
        self.memory_status.setText("Memory: Not connected")
        self.connect_memory_button.setText("Connect to Memory System")
        self.connect_memory_button.clicked.disconnect()
        self.connect_memory_button.clicked.connect(self.on_connect_memory)
        
        # Disconnect from socket manager
        if self.socket_manager:
            self.socket_manager.set_memory_bridge(None)
        
        # Update status
        self.status_message("Disconnected from Language Memory System")
    
    def on_manage_plugins(self):
        """Handle manage plugins button"""
        if not self.socket_manager:
            self.status_message("Socket manager not initialized", error=True)
            return
        
        try:
            # Import plugin manager dialog
            from v5_client.ui.plugin_manager_dialog import PluginManagerDialog
            
            # Create and show dialog
            dialog = PluginManagerDialog(self.socket_manager, self)
            dialog.exec()
        except ImportError as e:
            logger.error(f"Failed to import plugin manager dialog: {e}")
            self.status_message("Failed to load plugin manager", error=True)
    
    def on_settings(self):
        """Handle settings button"""
        try:
            # Import settings dialog
            from v5_client.ui.settings_dialog import SettingsDialog
            
            # Create and show dialog
            dialog = SettingsDialog(self)
            dialog.exec()
        except ImportError as e:
            logger.error(f"Failed to import settings dialog: {e}")
            self.status_message("Failed to load settings dialog", error=True)
    
    def on_about(self):
        """Handle about action"""
        from PySide6.QtWidgets import QMessageBox
        
        QMessageBox.about(
            self,
            "About V5 Fractal Echo Visualization",
            "<h2>V5 Fractal Echo Visualization</h2>"
            "<p>A comprehensive PySide6-based client for the V5 Fractal Echo "
            "Visualization System, integrating with the Language Memory System.</p>"
            "<p><b>Version:</b> 1.0.0</p>"
            "<p>Part of the Lumina Neural Network System</p>"
        )
    
    def toggle_sidebar(self):
        """Toggle the sidebar visibility"""
        if self.sidebar.isVisible():
            self.sidebar.hide()
        else:
            self.sidebar.show()
    
    def change_theme(self, theme_name):
        """
        Change the application theme
        
        Args:
            theme_name: Name of the theme to apply ('light', 'dark', 'system')
        """
        try:
            from v5_client.ui.theme_manager import ThemeManager
            theme_manager = ThemeManager()
            theme_manager.apply_theme(theme_name)
            self.status_message(f"Applied {theme_name} theme")
        except ImportError as e:
            logger.error(f"Failed to import theme manager: {e}")
            self.status_message("Failed to change theme", error=True)
    
    def status_message(self, message, timeout=5000, error=False):
        """
        Show a status message
        
        Args:
            message: Message to show
            timeout: Message timeout in milliseconds
            error: Whether this is an error message
        """
        self.status_label.setText(message)
        
        if error:
            self.status_label.setStyleSheet("color: #cc0000;")
        else:
            self.status_label.setStyleSheet("")
        
        logger.info(message) if not error else logger.error(message)
        
        # Reset after timeout if specified
        if timeout > 0:
            QTimer.singleShot(timeout, lambda: self.status_label.setText("Ready"))
            QTimer.singleShot(timeout, lambda: self.status_label.setStyleSheet(""))
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Clean up socket manager
        if self.socket_manager:
            self.socket_manager.cleanup()
        
        # Accept the close event
        event.accept() 