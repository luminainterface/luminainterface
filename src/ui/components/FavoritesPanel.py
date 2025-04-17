from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                          QPushButton, QScrollArea, QListWidget, QListWidgetItem,
                          QFrame, QGroupBox, QMenu, QAction, QSizePolicy)
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal

class FavoriteItem(QFrame):
    """A single favorite item widget"""
    
    def __init__(self, title, category, timestamp, parent=None):
        super().__init__(parent)
        self.title = title
        self.category = category
        self.timestamp = timestamp
        self.initUI()
        
    def initUI(self):
        """Initialize the favorite item UI"""
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #1E2C3A;
                border-radius: 8px;
                margin: 5px;
                padding: 10px;
            }
            QFrame:hover {
                background-color: #2C3E50;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title_label = QLabel(self.title)
        title_label.setStyleSheet("color: #3498DB; font-weight: bold; font-size: 16px;")
        layout.addWidget(title_label)
        
        # Information row
        info_container = QHBoxLayout()
        
        # Category label
        category_label = QLabel(self.category)
        category_label.setStyleSheet("""
            color: #2ECC71;
            background-color: #0F1A26;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 12px;
        """)
        info_container.addWidget(category_label)
        
        # Timestamp
        timestamp_label = QLabel(self.timestamp)
        timestamp_label.setStyleSheet("color: #95A5A6; font-size: 12px;")
        timestamp_label.setAlignment(Qt.AlignRight)
        info_container.addWidget(timestamp_label)
        
        layout.addLayout(info_container)
        
        # Button row
        button_container = QHBoxLayout()
        
        # Open button
        open_button = QPushButton("Open")
        open_button.setStyleSheet("""
            QPushButton {
                background-color: #2980B9;
                color: white;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #3498DB;
            }
        """)
        button_container.addWidget(open_button)
        
        # Remove button
        remove_button = QPushButton("Remove")
        remove_button.setStyleSheet("""
            QPushButton {
                background-color: #C0392B;
                color: white;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #E74C3C;
            }
        """)
        button_container.addWidget(remove_button)
        
        layout.addLayout(button_container)
        
        self.setLayout(layout)
        
class FavoritesPanel(QWidget):
    """Favorites panel for saved conversations and patterns"""
    
    item_opened = pyqtSignal(str, str)  # Signal emitted when a favorite item is opened
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        """Initialize the favorites panel UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Header
        header_container = QHBoxLayout()
        
        header = QLabel("Favorites")
        header.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #3498DB;
        """)
        header_container.addWidget(header)
        
        # Filter dropdown
        self.filter_button = QPushButton("Filter ▼")
        self.filter_button.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50;
                color: white;
                border-radius: 5px;
                padding: 5px 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #34495E;
            }
        """)
        self.filter_button.setFixedWidth(100)
        self.filter_button.clicked.connect(self.show_filter_menu)
        header_container.addWidget(self.filter_button, alignment=Qt.AlignRight)
        
        layout.addLayout(header_container)
        
        # Search box
        self.search_box = QFrame()
        self.search_box.setStyleSheet("""
            QFrame {
                background-color: #1E2C3A;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        search_layout = QHBoxLayout(self.search_box)
        search_layout.setContentsMargins(10, 5, 10, 5)
        
        search_icon = QLabel("🔍")
        search_icon.setStyleSheet("color: #95A5A6; font-size: 16px;")
        search_layout.addWidget(search_icon)
        
        self.search_input = QLabel("Search favorites...")
        self.search_input.setStyleSheet("color: #95A5A6; font-size: 14px;")
        search_layout.addWidget(self.search_input, 1)
        
        layout.addWidget(self.search_box)
        
        # Favorite categories
        categories = [
            ("Conversations", ["Morning Reflection", "Neural Network Theory", "Quantum Insights"]),
            ("Patterns", ["Spiral Sequence", "Echo Pattern", "Resonance Field"]),
            ("Glyphs", ["Infinity Loop", "Trifold Symbol", "Quantum Gate"])
        ]
        
        # Create category sections
        for category_name, items in categories:
            group_box = QGroupBox(category_name)
            group_box.setStyleSheet("""
                QGroupBox {
                    font-size: 16px;
                    font-weight: bold;
                    border: 1px solid #2C3E50;
                    border-radius: 5px;
                    margin-top: 10px;
                    padding: 5px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                    color: #3498DB;
                }
            """)
            
            category_layout = QVBoxLayout(group_box)
            category_layout.setSpacing(10)
            
            # Add items to category
            for item in items:
                favorite_item = FavoriteItem(
                    title=item,
                    category=category_name,
                    timestamp="2025-04-12"
                )
                category_layout.addWidget(favorite_item)
            
            layout.addWidget(group_box)
        
        # Add spacer at the bottom
        layout.addStretch()
        
        self.setLayout(layout)
        
    def show_filter_menu(self):
        """Show the filter menu when filter button is clicked"""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1E2C3A;
                color: #ECF0F1;
                border: 1px solid #2C3E50;
                border-radius: 5px;
                padding: 5px;
            }
            QMenu::item {
                padding: 5px 20px;
            }
            QMenu::item:selected {
                background-color: #2980B9;
            }
        """)
        
        # Add filter options
        all_action = QAction("All", self)
        conversations_action = QAction("Conversations", self)
        patterns_action = QAction("Patterns", self)
        glyphs_action = QAction("Glyphs", self)
        
        # Add divider
        menu.addAction(all_action)
        menu.addSeparator()
        menu.addAction(conversations_action)
        menu.addAction(patterns_action)
        menu.addAction(glyphs_action)
        
        # Show menu at button position
        menu.exec_(self.filter_button.mapToGlobal(self.filter_button.rect().bottomLeft())) 