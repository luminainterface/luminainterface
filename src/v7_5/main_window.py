from PySide6.QtWidgets import QMainWindow, QMenuBar, QMenu, QWidget, QVBoxLayout
from PySide6.QtCore import Qt, Slot
from .chat_widget import ChatWidget
from .settings_window import SettingsWindow
from .auto_wiki_processor import AutoWikiProcessor
import threading

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LUMINA v7.5")
        self.resize(800, 600)

        # Initialize settings
        self.settings = {
            'model': 'mistral-tiny',
            'temperature': 0.7,
            'top_p': 0.9,
            'auto_wiki': False
        }

        # Create menu bar
        self.menu_bar = QMenuBar()
        self.file_menu = QMenu("File", self)
        self.settings_action = self.file_menu.addAction("Settings")
        self.settings_action.triggered.connect(self.show_settings)
        self.menu_bar.addMenu(self.file_menu)
        self.setMenuBar(self.menu_bar)

        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Initialize chat widget
        self.chat_widget = ChatWidget(self)
        self.layout.addWidget(self.chat_widget)

        # Initialize settings window
        self.settings_window = SettingsWindow(self)
        self.settings_window.settings_updated.connect(self.apply_settings)

        # Initialize wiki processor
        self.wiki_stop_event = threading.Event()
        self.wiki_processor = AutoWikiProcessor(
            stop_event=self.wiki_stop_event,
            update_callback=self.handle_wiki_update
        )

        # Apply initial settings
        self.apply_settings(self.settings)

    @Slot(dict)
    def apply_settings(self, settings):
        """Apply settings to the chat widget and update wiki processor."""
        self.settings.update(settings)
        self.chat_widget.apply_settings(settings)

        # Handle auto-wiki setting
        if settings.get('auto_wiki', False):
            if not self.wiki_processor.is_running():
                self.wiki_processor.start_processing()
        else:
            if self.wiki_processor.is_running():
                self.wiki_processor.stop_processing()

    def show_settings(self):
        """Show the settings window with current settings."""
        self.settings_window.load_settings(self.settings)
        self.settings_window.show()

    @Slot(str)
    def handle_wiki_update(self, wiki_content):
        """Handle wiki updates by displaying them as system messages."""
        if wiki_content:
            self.chat_widget.add_system_message(f"Wiki Info: {wiki_content}")

    def closeEvent(self, event):
        """Clean up resources when closing the window."""
        self.wiki_processor.stop_processing()
        self.wiki_stop_event.set()
        super().closeEvent(event) 