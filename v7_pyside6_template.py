#!/usr/bin/env python3
"""
LUMINA V7 PySide6 Template Application (16:9)

A template application for integrating V7, Mistral, and memory components
with a plugin architecture.
"""

import os
import sys
import time
import logging
import importlib
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDockWidget, QTabWidget, 
    QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QPushButton,
    QTextEdit, QLineEdit, QStatusBar, QToolBar, QMenu, QMenuBar,
    QScrollArea, QFrame, QSizePolicy, QComboBox, QMessageBox
)
from PySide6.QtCore import Qt, QSize, QTimer, QThread, Signal, Slot, QSettings
from PySide6.QtGui import QAction, QIcon, QFont, QColor, QPalette

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/v7_template.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
Path('logs').mkdir(exist_ok=True)

# Plugin interface class
class PluginInterface:
    """Base interface for all plugins"""
    
    def __init__(self, app_context):
        """Initialize the plugin with application context"""
        self.app_context = app_context
        self.name = "Base Plugin"
        self.version = "0.1.0"
        self.author = "LUMINA"
        self.dependencies = []
        self.ui_components = {}
        self.active = False
    
    def initialize(self) -> bool:
        """Initialize the plugin"""
        return True
    
    def get_dock_widgets(self) -> List[QDockWidget]:
        """Return list of dock widgets provided by this plugin"""
        return []
    
    def get_tab_widgets(self) -> List[tuple]:
        """Return list of (name, widget) tuples for tab widgets"""
        return []
    
    def get_toolbar_actions(self) -> List[QAction]:
        """Return list of toolbar actions"""
        return []
    
    def get_menu_actions(self) -> Dict[str, List[QAction]]:
        """Return dict of menu_name: [actions] for menu integration"""
        return {}
    
    def shutdown(self) -> None:
        """Clean shutdown of the plugin"""
        pass


class PluginManager:
    """Manages loading and integrating plugins"""
    
    def __init__(self, app_context):
        """Initialize with application context"""
        self.app_context = app_context
        self.plugins = {}
        self.plugin_paths = [
            "plugins",
            "src/v7/plugins",
            "src/plugins"
        ]
        
        # Check for environment variable overriding plugin paths
        if os.environ.get("TEMPLATE_PLUGINS_DIRS"):
            plugin_dirs = os.environ.get("TEMPLATE_PLUGINS_DIRS").split(";")
            # Use environment paths but keep the defaults if not specified
            if plugin_dirs:
                self.plugin_paths = plugin_dirs
        
        self.active_plugins = []
    
    def discover_plugins(self):
        """Discover available plugins in plugin directories"""
        plugin_modules = {}
        
        # Create plugin directories if they don't exist
        for path in self.plugin_paths:
            Path(path).mkdir(exist_ok=True)
            
            # Create __init__.py if it doesn't exist
            init_file = Path(path) / "__init__.py"
            if not init_file.exists():
                init_file.touch()
        
        # Add plugin directories to path
        for path in self.plugin_paths:
            if path not in sys.path and os.path.exists(path):
                sys.path.append(path)
                logger.info(f"Added plugin path to sys.path: {path}")
        
        # Discover plugins in each path
        for path in self.plugin_paths:
            if not os.path.exists(path):
                logger.warning(f"Plugin path does not exist: {path}")
                continue
                
            logger.info(f"Searching for plugins in: {path}")
            for item in os.listdir(path):
                if item.startswith('_') or not item.endswith('.py'):
                    continue
                
                module_name = item[:-3]  # Remove .py extension
                logger.info(f"Found potential plugin: {item}")
                
                try:
                    # Try to import the module
                    full_path = os.path.join(path, item)
                    logger.info(f"Trying to import: {full_path}")
                    
                    if path in ["plugins", "."]:
                        module_path = module_name
                    else:
                        module_path = f"{os.path.basename(path)}.{module_name}"
                    
                    logger.info(f"Importing module: {module_path}")
                    module = importlib.import_module(module_path)
                    
                    # Check if it has a Plugin class
                    if hasattr(module, 'Plugin') and issubclass(module.Plugin, PluginInterface):
                        plugin_modules[module_name] = module
                        logger.info(f"Discovered plugin: {module_name}")
                except Exception as e:
                    logger.error(f"Error loading plugin {module_name}: {e}")
        
        return plugin_modules
    
    def load_plugin(self, module_name, module):
        """Load a specific plugin"""
        try:
            # Instantiate plugin
            plugin = module.Plugin(self.app_context)
            plugin_id = f"{module_name}-{plugin.version}"
            
            # Initialize plugin
            if plugin.initialize():
                self.plugins[plugin_id] = plugin
                logger.info(f"Loaded plugin: {plugin.name} v{plugin.version}")
                return plugin
            else:
                logger.warning(f"Plugin {module_name} failed to initialize")
                return None
        except Exception as e:
            logger.error(f"Error initializing plugin {module_name}: {e}")
            return None
    
    def activate_plugin(self, plugin_id):
        """Activate a loaded plugin"""
        if plugin_id not in self.plugins:
            logger.warning(f"Cannot activate unknown plugin: {plugin_id}")
            return False
        
        plugin = self.plugins[plugin_id]
        
        # Check dependencies
        for dep in plugin.dependencies:
            if dep not in self.active_plugins:
                logger.warning(f"Cannot activate {plugin_id}: missing dependency {dep}")
                return False
        
        # Mark as active
        plugin.active = True
        self.active_plugins.append(plugin_id)
        
        # Integrate UI components
        try:
            self.app_context.integrate_plugin_ui(plugin)
        except Exception as e:
            logger.error(f"Error integrating {plugin_id} UI: {e}")
            plugin.active = False
            self.active_plugins.remove(plugin_id)
            return False
        
        logger.info(f"Activated plugin: {plugin.name} v{plugin.version}")
        return True
    
    def deactivate_plugin(self, plugin_id):
        """Deactivate a plugin"""
        if plugin_id not in self.plugins or plugin_id not in self.active_plugins:
            return False
        
        plugin = self.plugins[plugin_id]
        
        # Check if other active plugins depend on this one
        for pid, p in self.plugins.items():
            if pid != plugin_id and pid in self.active_plugins:
                if plugin_id in p.dependencies:
                    logger.warning(f"Cannot deactivate {plugin_id}: plugin {pid} depends on it")
                    return False
        
        # Remove UI components
        try:
            self.app_context.remove_plugin_ui(plugin)
        except Exception as e:
            logger.error(f"Error removing {plugin_id} UI: {e}")
        
        # Mark as inactive
        plugin.active = False
        self.active_plugins.remove(plugin_id)
        
        logger.info(f"Deactivated plugin: {plugin.name}")
        return True
    
    def shutdown_all_plugins(self):
        """Shut down all active plugins"""
        for plugin_id in list(self.active_plugins):
            plugin = self.plugins[plugin_id]
            try:
                plugin.shutdown()
                logger.info(f"Shut down plugin: {plugin.name}")
            except Exception as e:
                logger.error(f"Error shutting down plugin {plugin_id}: {e}")
        
        self.active_plugins = []


class V7MainWindow(QMainWindow):
    """Main window with 16:9 aspect ratio and plugin architecture"""
    
    def __init__(self, plugins_enabled=False, auto_load_plugins=None):
        super().__init__()
        
        self.plugins_enabled = plugins_enabled
        self.auto_load_plugins = auto_load_plugins or []
        
        # Get application title from environment variable
        self.title = os.environ.get("TEMPLATE_TITLE", "LUMINA V7 Template")
        
        # Application context (shared with plugins)
        self.app_context = {
            "main_window": self,
            "integrate_plugin_ui": self.integrate_plugin_ui,
            "remove_plugin_ui": self.remove_plugin_ui,
            "register_event_handler": self.register_event_handler,
            "trigger_event": self.trigger_event,
            "settings": QSettings("LUMINA", "V7Template")
        }
        
        # Plugin manager
        self.plugin_manager = PluginManager(self.app_context)
        
        # Event handlers
        self.event_handlers = {}
        
        # Setup UI
        self.setup_ui()
        
        # Discover plugins
        self.discover_plugins()
    
    def setup_ui(self):
        """Set up the main UI components"""
        # Set window properties
        self.setWindowTitle(self.title)
        
        # Set 16:9 aspect ratio (1280x720 base size)
        self.resize(1280, 720)
        
        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create main layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create main splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.main_splitter)
        
        # Left panel (navigation and controls)
        self.left_panel = QWidget()
        self.left_panel.setMinimumWidth(250)
        self.left_panel.setMaximumWidth(350)
        self.left_panel_layout = QVBoxLayout(self.left_panel)
        
        # Left panel header
        self.left_header = QLabel("LUMINA V7")
        self.left_header.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 14, QFont.Bold)
        self.left_header.setFont(font)
        self.left_panel_layout.addWidget(self.left_header)
        
        # Plugin list
        self.plugin_label = QLabel("Available Plugins:")
        self.left_panel_layout.addWidget(self.plugin_label)
        
        self.plugin_list = QComboBox()
        self.plugin_list.setMinimumHeight(30)
        self.left_panel_layout.addWidget(self.plugin_list)
        
        self.load_plugin_button = QPushButton("Load Plugin")
        self.load_plugin_button.clicked.connect(self.on_load_plugin)
        self.left_panel_layout.addWidget(self.load_plugin_button)
        
        # Add spacer
        self.left_panel_layout.addStretch(1)
        
        # Status section
        self.status_label = QLabel("System Status:")
        self.left_panel_layout.addWidget(self.status_label)
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)
        self.left_panel_layout.addWidget(self.status_text)
        self.status_text.append("System initialized")
        self.status_text.append("No plugins loaded")
        
        # Add left panel to splitter
        self.main_splitter.addWidget(self.left_panel)
        
        # Right content area with tabs
        self.right_content = QTabWidget()
        self.main_splitter.addWidget(self.right_content)
        
        # Welcome tab
        self.welcome_tab = QWidget()
        self.welcome_layout = QVBoxLayout(self.welcome_tab)
        self.welcome_text = QLabel("Welcome to LUMINA V7 Plugin Template")
        self.welcome_text.setAlignment(Qt.AlignCenter)
        welcome_font = QFont("Arial", 18, QFont.Bold)
        self.welcome_text.setFont(welcome_font)
        self.welcome_layout.addWidget(self.welcome_text)
        
        self.welcome_description = QLabel(
            "This template provides a 16:9 interface ready for plugin integration.\n"
            "Load plugins from the left panel to access V7, Mistral, and memory features."
        )
        self.welcome_description.setAlignment(Qt.AlignCenter)
        self.welcome_layout.addWidget(self.welcome_description)
        
        self.welcome_layout.addStretch(1)
        
        self.right_content.addTab(self.welcome_tab, "Welcome")
        
        # Set splitter proportions (1:3 ratio)
        self.main_splitter.setSizes([250, 750])
        
        # Create dock areas for plugins
        self.setDockOptions(
            QMainWindow.AnimatedDocks | 
            QMainWindow.AllowNestedDocks | 
            QMainWindow.AllowTabbedDocks
        )
        
        # Create status bar
        self.statusbar = self.statusBar()
        self.statusbar.showMessage("Ready")
        
        # Create menu bar
        self.menu_bar = self.menuBar()
        
        # File menu
        self.file_menu = self.menu_bar.addMenu("&File")
        
        self.exit_action = QAction("E&xit", self)
        self.exit_action.setStatusTip("Exit the application")
        self.exit_action.triggered.connect(self.close)
        self.file_menu.addAction(self.exit_action)
        
        # Plugins menu
        self.plugins_menu = self.menu_bar.addMenu("&Plugins")
        
        self.refresh_plugins_action = QAction("&Refresh Plugins", self)
        self.refresh_plugins_action.triggered.connect(self.discover_plugins)
        self.plugins_menu.addAction(self.refresh_plugins_action)
        
        # Help menu
        self.help_menu = self.menu_bar.addMenu("&Help")
        
        self.about_action = QAction("&About", self)
        self.about_action.triggered.connect(self.show_about)
        self.help_menu.addAction(self.about_action)
        
        # Create toolbar
        self.toolbar = self.addToolBar("Main")
        self.toolbar.setMovable(True)
        
        # Add refresh plugins action to toolbar
        self.toolbar.addAction(self.refresh_plugins_action)
    
    def discover_plugins(self):
        """Discover and load available plugins"""
        if not self.plugins_enabled:
            # Show template plugins instead
            show_template_plugins(self)
            return
        
        # Discover plugins
        modules = self.plugin_manager.discover_plugins()
        
        if not modules:
            logger.warning("No plugins discovered")
            # Show message in plugin list
            self.plugin_list.addItem("No plugins found")
            return
        
        # Clear plugin list
        self.plugin_list.clear()
        
        # Add discovered plugins to list
        for module_name in sorted(modules.keys()):
            self.plugin_list.addItem(module_name)
        
        # Auto-load plugins if specified
        if self.auto_load_plugins:
            logger.info(f"Auto-loading plugins: {self.auto_load_plugins}")
            
            # Load from environment variable if no list provided
            if not self.auto_load_plugins and os.environ.get("TEMPLATE_AUTO_LOAD_PLUGINS"):
                self.auto_load_plugins = os.environ.get("TEMPLATE_AUTO_LOAD_PLUGINS").split(";")
            
            for plugin_name in self.auto_load_plugins:
                if plugin_name.endswith('.py'):
                    plugin_name = plugin_name[:-3]  # Remove .py extension
                
                if plugin_name in modules:
                    plugin = self.plugin_manager.load_plugin(plugin_name, modules[plugin_name])
                    if plugin:
                        self.plugin_manager.activate_plugin(f"{plugin_name}-{plugin.version}")
                        self.statusbar.showMessage(f"Auto-loaded plugin: {plugin_name}", 5000)
                else:
                    logger.warning(f"Auto-load plugin not found: {plugin_name}")
    
    def on_load_plugin(self):
        """Load selected plugin"""
        index = self.plugin_list.currentIndex()
        if index < 0:
            return
        
        name = self.plugin_list.currentText()
        module = self.plugin_list.currentData()
        
        if not module:
            return
        
        # Load plugin
        plugin = self.plugin_manager.load_plugin(name, module)
        
        if plugin:
            # Activate plugin
            plugin_id = f"{name}-{plugin.version}"
            if self.plugin_manager.activate_plugin(plugin_id):
                self.status_text.append(f"Loaded and activated: {name} v{plugin.version}")
            else:
                self.status_text.append(f"Loaded but failed to activate: {name}")
        else:
            self.status_text.append(f"Failed to load plugin: {name}")
    
    def integrate_plugin_ui(self, plugin):
        """Integrate plugin UI components"""
        # Add dock widgets
        for dock in plugin.get_dock_widgets():
            self.addDockWidget(Qt.RightDockWidgetArea, dock)
            
            # Add to Window menu if not already there
            if not hasattr(self, 'window_menu'):
                self.window_menu = self.menu_bar.addMenu("&Window")
            
            self.window_menu.addAction(dock.toggleViewAction())
        
        # Add tab widgets
        for name, widget in plugin.get_tab_widgets():
            self.right_content.addTab(widget, name)
        
        # Add toolbar actions
        for action in plugin.get_toolbar_actions():
            self.toolbar.addAction(action)
        
        # Add menu actions
        for menu_name, actions in plugin.get_menu_actions().items():
            # Find or create menu
            menu = None
            for action in self.menu_bar.actions():
                if action.text().replace("&", "") == menu_name.replace("&", ""):
                    menu = action.menu()
                    break
            
            if not menu:
                menu = self.menu_bar.addMenu(menu_name)
            
            # Add actions to menu
            for action in actions:
                menu.addAction(action)
    
    def remove_plugin_ui(self, plugin):
        """Remove plugin UI components"""
        # Remove dock widgets
        for dock in plugin.get_dock_widgets():
            self.removeDockWidget(dock)
        
        # Remove tab widgets
        for name, widget in plugin.get_tab_widgets():
            index = self.right_content.indexOf(widget)
            if index >= 0:
                self.right_content.removeTab(index)
        
        # Remove toolbar actions
        for action in plugin.get_toolbar_actions():
            self.toolbar.removeAction(action)
        
        # Remove menu actions (tricky, so we'll just leave them)
        # Would need to track added actions per plugin for proper removal
    
    def register_event_handler(self, event_name, handler_fn):
        """Register an event handler function"""
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        
        self.event_handlers[event_name].append(handler_fn)
    
    def trigger_event(self, event_name, *args, **kwargs):
        """Trigger event and call all registered handlers"""
        if event_name not in self.event_handlers:
            return []
        
        results = []
        for handler in self.event_handlers[event_name]:
            try:
                result = handler(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in event handler for {event_name}: {e}")
        
        return results
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About LUMINA V7 Template",
            "<h1>LUMINA V7 Template</h1>"
            "<p>Version 1.0.0</p>"
            "<p>A plugin-based template for LUMINA V7 components.</p>"
            "<p>© 2025 LUMINA Labs</p>"
        )
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Shut down all plugins
        self.plugin_manager.shutdown_all_plugins()
        
        # Accept the close event
        event.accept()


class ChatPluginTemplate:
    """Template class for chat plugins (not a real plugin)"""
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.name = "Chat Plugin Template"
        self.version = "0.1.0"
        
        # Create UI components
        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_widget)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_layout.addWidget(self.chat_display)
        
        # Input area
        self.input_layout = QHBoxLayout()
        
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type your message...")
        self.input_layout.addWidget(self.chat_input)
        
        self.send_button = QPushButton("Send")
        self.input_layout.addWidget(self.send_button)
        
        self.chat_layout.addLayout(self.input_layout)
        
        # Create dock widget
        self.chat_dock = QDockWidget("Chat")
        self.chat_dock.setWidget(self.chat_widget)
    
    def get_dock_widgets(self):
        return [self.chat_dock]
    
    def get_tab_widgets(self):
        return [("Chat", self.chat_widget)]


class VisualizationPluginTemplate:
    """Template class for visualization plugins (not a real plugin)"""
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.name = "Visualization Template"
        self.version = "0.1.0"
        
        # Create UI components
        self.viz_widget = QWidget()
        self.viz_layout = QVBoxLayout(self.viz_widget)
        
        # Visualization placeholder
        self.viz_placeholder = QLabel("Visualization Area (16:9)")
        self.viz_placeholder.setAlignment(Qt.AlignCenter)
        self.viz_placeholder.setMinimumSize(640, 360)  # 16:9 ratio
        self.viz_placeholder.setStyleSheet("background-color: #2a2a2a; color: white;")
        self.viz_layout.addWidget(self.viz_placeholder)
        
        # Controls
        self.controls_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start")
        self.controls_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.controls_layout.addWidget(self.stop_button)
        
        self.viz_layout.addLayout(self.controls_layout)
        
        # Create dock widget
        self.viz_dock = QDockWidget("Visualization")
        self.viz_dock.setWidget(self.viz_widget)
    
    def get_dock_widgets(self):
        return [self.viz_dock]
    
    def get_tab_widgets(self):
        return [("Visualization", self.viz_widget)]


class MemoryPluginTemplate:
    """Template class for memory plugins (not a real plugin)"""
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.name = "Memory Template"
        self.version = "0.1.0"
        
        # Create UI components
        self.memory_widget = QWidget()
        self.memory_layout = QVBoxLayout(self.memory_widget)
        
        # Memory display
        self.memory_display = QTextEdit()
        self.memory_display.setReadOnly(True)
        self.memory_display.setPlaceholderText("Memory entries will appear here...")
        self.memory_layout.addWidget(self.memory_display)
        
        # Add memory controls
        self.memory_control_layout = QHBoxLayout()
        
        self.add_memory_button = QPushButton("Add Entry")
        self.memory_control_layout.addWidget(self.add_memory_button)
        
        self.search_memory_button = QPushButton("Search")
        self.memory_control_layout.addWidget(self.search_memory_button)
        
        self.clear_memory_button = QPushButton("Clear")
        self.memory_control_layout.addWidget(self.clear_memory_button)
        
        self.memory_layout.addLayout(self.memory_control_layout)
        
        # Create dock widget
        self.memory_dock = QDockWidget("Memory")
        self.memory_dock.setWidget(self.memory_widget)
    
    def get_dock_widgets(self):
        return [self.memory_dock]
    
    def get_tab_widgets(self):
        return [("Memory", self.memory_widget)]


def show_template_plugins(main_window):
    """Show template plugins for preview purposes"""
    # This is just to show what plugins might look like
    # In a real implementation, these would be loaded from plugin files
    
    # Create template plugins
    chat_template = ChatPluginTemplate(main_window.app_context)
    viz_template = VisualizationPluginTemplate(main_window.app_context)
    memory_template = MemoryPluginTemplate(main_window.app_context)
    
    # Add dock widgets
    main_window.addDockWidget(Qt.RightDockWidgetArea, chat_template.chat_dock)
    main_window.addDockWidget(Qt.BottomDockWidgetArea, viz_template.viz_dock)
    main_window.addDockWidget(Qt.LeftDockWidgetArea, memory_template.memory_dock)
    
    # Add tab widgets
    main_window.right_content.addTab(chat_template.chat_widget, "Chat")
    main_window.right_content.addTab(viz_template.viz_widget, "Visualization")
    main_window.right_content.addTab(memory_template.memory_widget, "Memory")
    
    # Add to status
    main_window.status_text.append("Added template plugin previews")
    main_window.status_text.append("Note: These are just UI previews, not functional plugins")


def create_sample_plugin_file():
    """Create a sample plugin file to demonstrate plugin structure"""
    plugins_dir = Path("plugins")
    plugins_dir.mkdir(exist_ok=True)
    
    sample_plugin_path = plugins_dir / "sample_plugin.py"
    
    if not sample_plugin_path.exists():
        with open(sample_plugin_path, 'w') as f:
            f.write('''"""
Sample Plugin for V7 Template

This demonstrates the basic structure of a plugin file.
"""

from PySide6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit
from PySide6.QtCore import Qt

# Import the plugin interface from main app
try:
    from v7_pyside6_template import PluginInterface
except ImportError:
    # For development/testing when not imported properly
    class PluginInterface:
        def __init__(self, app_context):
            self.app_context = app_context


class Plugin(PluginInterface):
    """Sample plugin implementation"""
    
    def __init__(self, app_context):
        super().__init__(app_context)
        self.name = "Sample Plugin"
        self.version = "1.0.0"
        self.author = "LUMINA"
        self.dependencies = []
        
        # Create UI components
        self.setup_ui()
    
    def setup_ui(self):
        """Set up UI components for this plugin"""
        # Main widget
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        
        # Add title
        self.title_label = QLabel("Sample Plugin")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.title_label)
        
        # Add text area
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setText("This is a sample plugin to demonstrate the plugin architecture.")
        self.main_layout.addWidget(self.text_area)
        
        # Add button
        self.action_button = QPushButton("Perform Action")
        self.action_button.clicked.connect(self.on_action_button_clicked)
        self.main_layout.addWidget(self.action_button)
        
        # Create dock widget
        self.dock_widget = QDockWidget("Sample Plugin")
        self.dock_widget.setWidget(self.main_widget)
    
    def initialize(self):
        """Initialize the plugin"""
        # Register for events we're interested in
        self.app_context["register_event_handler"]("sample_event", self.handle_sample_event)
        return True
    
    def get_dock_widgets(self):
        """Return dock widgets for this plugin"""
        return [self.dock_widget]
    
    def get_tab_widgets(self):
        """Return tab widgets for this plugin"""
        return [("Sample", self.main_widget)]
    
    def on_action_button_clicked(self):
        """Handle action button click"""
        self.text_area.append("Action button clicked!")
        
        # Trigger an event for other plugins to respond to
        results = self.app_context["trigger_event"]("sample_event", "Hello from Sample Plugin")
        
        self.text_area.append(f"Event triggered with {len(results)} handlers responding")
    
    def handle_sample_event(self, message):
        """Handle sample event from other plugins"""
        if message != "Hello from Sample Plugin":  # Avoid recursive loop
            self.text_area.append(f"Received event: {message}")
        return "Sample plugin received your message"
    
    def shutdown(self):
        """Clean shutdown of the plugin"""
        self.text_area.append("Plugin shutting down...")
''')
        return True
    return False


def main():
    """Run the application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LUMINA V7 PySide6 Template Application")
    parser.add_argument("--plugins-enabled", action="store_true", 
                        help="Enable plugin discovery and loading")
    parser.add_argument("--plugins-dirs", type=str, 
                        help="Semicolon-separated list of plugin directories")
    parser.add_argument("--auto-load-plugins", action="store_true",
                        help="Automatically load plugins from environment variable")
    parser.add_argument("--plugin-list", type=str,
                        help="Comma-separated list of plugins to auto-load")
    args = parser.parse_args()
    
    # Override with environment variables if specified
    plugins_enabled = args.plugins_enabled or os.environ.get("TEMPLATE_PLUGINS_ENABLED") == "true"
    
    # Get plugins to auto-load
    auto_load_plugins = []
    if args.auto_load_plugins and os.environ.get("TEMPLATE_AUTO_LOAD_PLUGINS"):
        auto_load_plugins = os.environ.get("TEMPLATE_AUTO_LOAD_PLUGINS").split(";")
    elif args.plugin_list:
        auto_load_plugins = args.plugin_list.split(",")
    
    # Set plugin directories from environment if specified
    if args.plugins_dirs:
        os.environ["TEMPLATE_PLUGINS_DIRS"] = args.plugins_dirs
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Set application name
    app.setApplicationName("LUMINA V7")
    app.setOrganizationName("LUMINA")
    
    # Create main window
    window = V7MainWindow(plugins_enabled=plugins_enabled, 
                         auto_load_plugins=auto_load_plugins)
    window.show()
    
    # Run the application
    return app.exec()

if __name__ == "__main__":
    sys.exit(main()) 