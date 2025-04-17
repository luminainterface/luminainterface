"""
Paged Visualization Panel

Provides a high-density visualization interface with page navigation,
based on the V5 Fractal Echo Visualization layout.
"""

import logging
from pathlib import Path

# Add project root to path if needed
import sys
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    # Import Qt compatibility layer from V5
    from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui, Qt, Signal, Slot
    from src.v5.ui.qt_compat import get_widgets, get_gui, get_core
except ImportError:
    logging.warning("V5 Qt compatibility layer not found. Using direct PySide6 imports.")
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        from PySide6.QtCore import Qt, Signal, Slot
        
        # Simple compatibility functions
        def get_widgets():
            return QtWidgets
            
        def get_gui():
            return QtGui
            
        def get_core():
            return QtCore
    except ImportError:
        logging.error("PySide6 not found. Please install PySide6 or configure the V5 Qt compatibility layer.")
        sys.exit(1)

# Import the panel base
from ..panel_base import V6PanelBase

# Set up logging
logger = logging.getLogger(__name__)

class PagedVisualizationPanel(V6PanelBase):
    """Base class for paged visualization panels with navigation"""
    
    # Signal emitted when a page changes
    pageChanged = Signal(str)
    
    def __init__(self, socket_manager=None, parent=None):
        super().__init__(parent)
        self.socket_manager = socket_manager
        
        # Initialize page tracking
        self.pages = {}
        self.current_page = None
        self.page_history = []
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface with page navigation"""
        # Create main layout
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Create navigation bar
        self.nav_bar = QtWidgets.QWidget()
        self.nav_bar.setFixedHeight(40)
        self.nav_bar.setStyleSheet("""
            background-color: rgba(44, 62, 80, 180);
            border-bottom: 1px solid rgba(52, 73, 94, 150);
        """)
        
        nav_layout = QtWidgets.QHBoxLayout(self.nav_bar)
        nav_layout.setContentsMargins(10, 0, 10, 0)
        
        # Add page navigation buttons
        self.page_combo = QtWidgets.QComboBox()
        self.page_combo.setStyleSheet("""
            background-color: rgba(52, 73, 94, 180);
            color: white;
            padding: 4px;
            border: 1px solid rgba(52, 152, 219, 120);
            border-radius: 4px;
            min-width: 150px;
        """)
        self.page_combo.currentTextChanged.connect(self.change_page)
        
        # Navigation buttons
        self.prev_button = QtWidgets.QPushButton("◀")
        self.prev_button.setFixedWidth(32)
        self.prev_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(41, 128, 185, 180);
                color: white;
                border-radius: 4px;
                padding: 4px;
                font-weight: bold;
                border: 1px solid rgba(52, 152, 219, 120);
            }
            QPushButton:hover {
                background-color: rgba(52, 152, 219, 200);
            }
            QPushButton:disabled {
                background-color: rgba(41, 128, 185, 80);
                color: rgba(255, 255, 255, 120);
            }
        """)
        self.prev_button.clicked.connect(self.go_to_previous_page)
        self.prev_button.setEnabled(False)
        
        self.next_button = QtWidgets.QPushButton("▶")
        self.next_button.setFixedWidth(32)
        self.next_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(41, 128, 185, 180);
                color: white;
                border-radius: 4px;
                padding: 4px;
                font-weight: bold;
                border: 1px solid rgba(52, 152, 219, 120);
            }
            QPushButton:hover {
                background-color: rgba(52, 152, 219, 200);
            }
            QPushButton:disabled {
                background-color: rgba(41, 128, 185, 80);
                color: rgba(255, 255, 255, 120);
            }
        """)
        self.next_button.clicked.connect(self.go_to_next_page)
        self.next_button.setEnabled(False)
        
        # Add fullscreen button
        self.fullscreen_button = QtWidgets.QPushButton("⤢")
        self.fullscreen_button.setFixedWidth(32)
        self.fullscreen_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(44, 62, 80, 180);
                color: white;
                border-radius: 4px;
                padding: 4px;
                font-weight: bold;
                border: 1px solid rgba(52, 73, 94, 120);
            }
            QPushButton:hover {
                background-color: rgba(52, 73, 94, 200);
            }
        """)
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        
        # Add widgets to navigation layout
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.page_combo)
        nav_layout.addWidget(self.next_button)
        nav_layout.addStretch(1)
        nav_layout.addWidget(self.fullscreen_button)
        
        # Add navigation bar to main layout
        self.main_layout.addWidget(self.nav_bar)
        
        # Create content area
        self.content_widget = QtWidgets.QStackedWidget()
        self.content_widget.setStyleSheet("""
            background-color: transparent;
        """)
        
        # Add content area to main layout
        self.main_layout.addWidget(self.content_widget, 1)  # 1 = stretch factor
    
    def add_page(self, page_id, page_title, widget):
        """
        Add a page to the visualization panel
        
        Args:
            page_id (str): Unique identifier for the page
            page_title (str): Display title for the page
            widget (QWidget): Widget to display for the page
        """
        if page_id in self.pages:
            logger.warning(f"Page {page_id} already exists, replacing")
            
            # Remove existing widget if it exists
            existing_index = self.content_widget.indexOf(self.pages[page_id]["widget"])
            if existing_index >= 0:
                self.content_widget.removeWidget(self.pages[page_id]["widget"])
        
        # Add page to tracking dictionary
        self.pages[page_id] = {
            "title": page_title,
            "widget": widget
        }
        
        # Add widget to stacked widget
        self.content_widget.addWidget(widget)
        
        # Update combo box
        current_text = self.page_combo.currentText()
        self.page_combo.clear()
        
        # Add all pages to combo box in alphabetical order by title
        sorted_pages = sorted(self.pages.items(), key=lambda x: x[1]["title"])
        for pid, page_info in sorted_pages:
            self.page_combo.addItem(page_info["title"], pid)
        
        # Restore selection or set to first page
        if current_text and self.page_combo.findText(current_text) >= 0:
            self.page_combo.setCurrentText(current_text)
        elif not self.current_page and sorted_pages:
            # Set to first page if none selected
            self.change_page(sorted_pages[0][1]["title"])
    
    def change_page(self, page_title):
        """
        Change to the specified page by title
        
        Args:
            page_title (str): Title of the page to switch to
        """
        # Find page ID from title
        page_id = None
        for pid, page_info in self.pages.items():
            if page_info["title"] == page_title:
                page_id = pid
                break
        
        if not page_id:
            logger.warning(f"Page title '{page_title}' not found")
            return
        
        # If changing to a different page
        if self.current_page != page_id:
            # Save the previous page for back navigation
            if self.current_page:
                self.page_history.append(self.current_page)
            
            # Update current page
            self.current_page = page_id
            
            # Switch to the widget
            widget = self.pages[page_id]["widget"]
            self.content_widget.setCurrentWidget(widget)
            
            # Update navigation buttons
            self.update_navigation_buttons()
            
            # Emit signal
            self.pageChanged.emit(page_id)
            
            logger.info(f"Changed to page: {page_title}")
    
    def go_to_previous_page(self):
        """Go back to the previous page in history"""
        if self.page_history:
            # Pop the last page from history
            prev_page_id = self.page_history.pop()
            
            # Get the title
            prev_page_title = self.pages[prev_page_id]["title"]
            
            # Block signals to avoid adding to history again
            self.page_combo.blockSignals(True)
            self.page_combo.setCurrentText(prev_page_title)
            self.page_combo.blockSignals(False)
            
            # Switch to the widget without modifying history
            self.current_page = prev_page_id
            widget = self.pages[prev_page_id]["widget"]
            self.content_widget.setCurrentWidget(widget)
            
            # Update navigation buttons
            self.update_navigation_buttons()
            
            # Emit signal
            self.pageChanged.emit(prev_page_id)
            
            logger.info(f"Navigated back to page: {prev_page_title}")
    
    def go_to_next_page(self):
        """Go to the next page based on ordering"""
        # Get current index in the combo box
        current_index = self.page_combo.currentIndex()
        
        # If there's a next page
        if current_index < self.page_combo.count() - 1:
            next_index = current_index + 1
            next_title = self.page_combo.itemText(next_index)
            self.change_page(next_title)
    
    def update_navigation_buttons(self):
        """Update navigation button states based on current page and history"""
        # Previous button enabled if there's history
        self.prev_button.setEnabled(len(self.page_history) > 0)
        
        # Next button enabled if not on last page
        current_index = self.page_combo.currentIndex()
        self.next_button.setEnabled(current_index < self.page_combo.count() - 1)
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode for the current visualization"""
        # Find the top-level window
        window = self.window()
        
        if window.isFullScreen():
            window.showNormal()
            self.fullscreen_button.setText("⤢")
        else:
            window.showFullScreen()
            self.fullscreen_button.setText("⤓")
    
    def set_fullscreen(self, fullscreen=True):
        """Set fullscreen mode explicitly"""
        window = self.window()
        
        if fullscreen:
            if not window.isFullScreen():
                window.showFullScreen()
                self.fullscreen_button.setText("⤓")
        else:
            if window.isFullScreen():
                window.showNormal()
                self.fullscreen_button.setText("⤢") 