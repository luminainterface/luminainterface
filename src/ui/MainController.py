import os
import sys
import logging

logger = logging.getLogger(__name__)

# Check if PySide6 is available, otherwise use PyQt5
try:
    from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                                QToolButton, QLabel, QFrame, QStackedWidget, QSplitter,
                                QPushButton)
    from PySide6.QtCore import Qt, QSize
    from PySide6.QtGui import QIcon, QFont, QPixmap
    logger.info("Using PySide6 for MainController")
    USING_PYSIDE6 = True
except ImportError:
    logger.warning("PySide6 not found, falling back to PyQt5")
    from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QToolButton, QLabel, QFrame, QStackedWidget, QSplitter,
                              QPushButton)
    from PyQt5.QtCore import Qt, QSize
    from PyQt5.QtGui import QIcon, QFont, QPixmap
    logger.info("Using PyQt5 for MainController")
    USING_PYSIDE6 = False

# Import components with proper error handling
def import_component(module_name, class_name):
    try:
        if USING_PYSIDE6:
            # Try to import PySide6 version
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        else:
            # Try to import PyQt5 version
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import {class_name} from {module_name}: {e}")
        
        # Create a simple fallback component
        if USING_PYSIDE6:
            from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
        else:
            from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
            
        class FallbackWidget(QWidget):
            def __init__(self, parent=None):
                super().__init__(parent)
                layout = QVBoxLayout(self)
                label = QLabel(f"Failed to load {class_name}")
                label.setStyleSheet("color: red; font-size: 16px;")
                layout.addWidget(label)
                error_label = QLabel(str(e))
                layout.addWidget(error_label)
        
        return FallbackWidget

# Import all panel components with fallback handling
ProfilePanel = import_component("src.ui.components.ProfilePanel", "ProfilePanel")
FavoritesPanel = import_component("src.ui.components.FavoritesPanel", "FavoritesPanel")
SettingsPanel = import_component("src.ui.components.SettingsPanel", "SettingsPanel")
MemoryScrollPanel = import_component("src.ui.components.MemoryScrollPanel", "MemoryScrollPanel")
NetworkVisualizationPanel = import_component("src.ui.components.NetworkVisualizationPanel", "NetworkVisualizationPanel")
TrainingPanel = import_component("src.ui.components.TrainingPanel", "TrainingPanel")
DatasetPanel = import_component("src.ui.components.DatasetPanel", "DatasetPanel")
JourneyVisualizationPanel = import_component("src.ui.components.JourneyVisualizationPanel", "JourneyVisualizationPanel")
JourneyInsightsPanel = import_component("src.ui.components.JourneyInsightsPanel", "JourneyInsightsPanel")
SpiritualGuidancePanel = import_component("src.ui.components.SpiritualGuidancePanel", "SpiritualGuidancePanel")

class NavigationButton(QToolButton):
    """Custom navigation button for the left sidebar"""
    
    def __init__(self, icon_name, text, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setIcon(QIcon(f"assets/icons/{icon_name}.png"))
        self.setIconSize(QSize(24, 24))
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.setCheckable(True)
        self.setAutoExclusive(True)
        self.setFixedHeight(40)
        self.setStyleSheet("""
            QToolButton {
                background-color: transparent;
                color: #BBBBBB;
                border: none;
                border-radius: 4px;
                padding: 5px 10px;
                text-align: left;
            }
            QToolButton:hover {
                background-color: rgba(255, 255, 255, 0.1);
                color: #FFFFFF;
            }
            QToolButton:checked {
                background-color: rgba(100, 150, 220, 0.3);
                color: #FFFFFF;
            }
        """)


class PlaceholderPanel(QWidget):
    """Placeholder panel for when actual components can't be loaded"""
    
    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.name = name
        self.initUI()
    
    def initUI(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel(f"{self.name} Panel")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #E0E0E0;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Description
        description = QLabel(f"This is a placeholder for the {self.name} panel. The actual component will be implemented as part of the V3 upgrade.")
        description.setWordWrap(True)
        description.setStyleSheet("font-size: 16px; color: #BBBBBB; margin: 20px 0;")
        description.setAlignment(Qt.AlignCenter)
        layout.addWidget(description)
        
        # Coming soon label
        coming_soon = QLabel("Coming Soon!")
        coming_soon.setStyleSheet("font-size: 20px; color: #66BBFF; margin: 10px 0;")
        coming_soon.setAlignment(Qt.AlignCenter)
        layout.addWidget(coming_soon)
        
        # Add some space
        layout.addStretch()


class MainController(QMainWindow):
    """Main controller for the Lumina GUI application"""
    
    def __init__(self):
        super().__init__()
        logger.info("Initializing MainController")
        self.initUI()
        
    def initUI(self):
        """Initialize the user interface"""
        self.setWindowTitle("Lumina Neural Network - V3")
        self.setMinimumSize(1200, 800)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #252535;
                color: #E0E0E0;
            }
            QSplitter::handle {
                background-color: #353545;
            }
            QStackedWidget {
                background-color: #252535;
            }
        """)
        
        # Main widget
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Left navigation panel
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_panel.setStyleSheet("""
            QFrame {
                background-color: #202030;
                border-right: 1px solid #353545;
            }
        """)
        left_panel.setFixedWidth(200)
        
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 20, 10, 20)
        left_layout.setSpacing(5)
        
        # Logo
        logo_layout = QHBoxLayout()
        logo_label = QLabel("LUMINA")
        logo_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #E0E0E0;")
        logo_layout.addWidget(logo_label)
        left_layout.addLayout(logo_layout)
        
        # Space after logo
        left_layout.addSpacing(20)
        
        # Navigation section label
        nav_label = QLabel("NAVIGATION")
        nav_label.setStyleSheet("font-size: 12px; color: #666677; margin-top: 10px; margin-bottom: 5px;")
        left_layout.addWidget(nav_label)
        
        # Create navigation buttons
        self.create_nav_buttons(left_layout)
        
        # Add stretch to push everything to the top
        left_layout.addStretch()
        
        # Add version info at bottom
        version_label = QLabel("Version 3.0")
        version_label.setStyleSheet("color: #666677; font-size: 10px;")
        left_layout.addWidget(version_label, 0, Qt.AlignBottom)
        
        # Main content area
        self.content_stack = QStackedWidget()
        
        # Create all panels
        self.create_panels()
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.content_stack)
        
        self.setCentralWidget(main_widget)
        
        # Set initial panel
        self.nav_buttons["journey"].setChecked(True)
        self.show_panel("journey")
        
        logger.info("MainController UI initialized")
    
    def create_nav_buttons(self, layout):
        """Create navigation buttons"""
        # Define buttons: {id: (icon_name, text)}
        button_defs = {
            "profile": ("user", "Profile"),
            "favorites": ("star", "Favorites"),
            "journey": ("path", "Journey"),
            "insights": ("bulb", "Insights"),
            "spiritual": ("lotus", "Spiritual"),
            "memory": ("memory", "Memory Scroll"),
            "network": ("network", "Network"),
            "training": ("brain", "Training"),
            "dataset": ("database", "Dataset"),
            "settings": ("settings", "Settings")
        }
        
        self.nav_buttons = {}
        
        for button_id, (icon, text) in button_defs.items():
            button = NavigationButton(icon, text)
            button.clicked.connect(lambda checked, bid=button_id: self.show_panel(bid))
            layout.addWidget(button)
            self.nav_buttons[button_id] = button
            
        logger.info(f"Created {len(button_defs)} navigation buttons")
    
    def create_panels(self):
        """Create all content panels"""
        logger.info("Creating content panels")
        
        # Define panel names
        panel_names = {
            "profile": "Profile",
            "favorites": "Favorites",
            "journey": "Journey Visualization",
            "insights": "Journey Insights",
            "spiritual": "Spiritual Guidance",
            "memory": "Memory Scroll",
            "network": "Network Visualization",
            "training": "Training",
            "dataset": "Dataset",
            "settings": "Settings"
        }
        
        # Create placeholder panels for each section
        self.panels = {}
        for panel_id, panel_name in panel_names.items():
            self.panels[panel_id] = PlaceholderPanel(panel_name)
            self.content_stack.addWidget(self.panels[panel_id])
        
        logger.info("All panels created")
    
    def show_panel(self, panel_id):
        """Show the specified panel"""
        if panel_id in self.panels:
            panel_index = list(self.panels.keys()).index(panel_id)
            self.content_stack.setCurrentIndex(panel_index)
            logger.info(f"Showing panel: {panel_id}")
    
    def handle_journey_milestone(self, data):
        """Handle milestone selection from journey visualization"""
        # In a real app, this would update the insights panel with relevant information
        logger.info(f"Milestone selected: {data}")
    
    def handle_neuron_selection(self, data):
        """Handle neuron selection from network visualization"""
        # In a real app, this would highlight related training data
        logger.info(f"Neuron selected: {data}")
    
    def handle_spiritual_insight(self, data):
        """Handle spiritual insight discovery"""
        # In a real app, this would add the insight to the insights panel
        logger.info(f"Spiritual insight discovered: {data}") 