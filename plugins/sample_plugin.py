"""
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
