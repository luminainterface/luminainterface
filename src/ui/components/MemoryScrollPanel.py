from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                          QPushButton, QScrollArea, QFrame, QSizePolicy,
                          QLineEdit, QSlider, QToolButton, QComboBox, QGroupBox)
from PyQt5.QtGui import QIcon, QFont, QColor, QPainter, QPen, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QRect, QPoint

class MemoryNodeWidget(QWidget):
    """A widget representing a single memory node in the memory scroll"""
    
    node_clicked = pyqtSignal(str)  # Signal emitted when a node is clicked
    
    def __init__(self, memory_id, title, timestamp, content_preview, importance=50, parent=None):
        super().__init__(parent)
        self.memory_id = memory_id
        self.title = title
        self.timestamp = timestamp
        self.content_preview = content_preview
        self.importance = importance
        self.is_highlighted = False
        self.is_selected = False
        self.initUI()
        
    def initUI(self):
        """Initialize the memory node UI"""
        self.setMinimumHeight(80)
        self.setMaximumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        # Set cursor to pointing hand
        self.setCursor(Qt.PointingHandCursor)
        
        # Set mouse tracking
        self.setMouseTracking(True)
        
    def paintEvent(self, event):
        """Custom paint event for the memory node"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Define colors based on state
        if self.is_selected:
            bg_color = QColor(52, 152, 219, 200)  # Blue when selected
            border_color = QColor(41, 128, 185)
            text_color = QColor(255, 255, 255)
            title_color = QColor(255, 255, 255)
        elif self.is_highlighted:
            bg_color = QColor(52, 73, 94, 200)  # Dark blue-gray when highlighted
            border_color = QColor(44, 62, 80)
            text_color = QColor(236, 240, 241)
            title_color = QColor(52, 152, 219)
        else:
            bg_color = QColor(30, 44, 58, 200)  # Dark background normally
            border_color = QColor(44, 62, 80)
            text_color = QColor(189, 195, 199)
            title_color = QColor(52, 152, 219)
        
        # Draw background with rounded corners
        rect = self.rect().adjusted(5, 5, -5, -5)
        painter.setPen(QPen(border_color, 2))
        painter.setBrush(bg_color)
        painter.drawRoundedRect(rect, 10, 10)
        
        # Importance indicator on the left side
        indicator_width = 6
        indicator_rect = QRect(rect.left() + 2, rect.top() + 5, indicator_width, rect.height() - 10)
        
        # Determine color based on importance (red for high, yellow for medium, green for low)
        if self.importance > 70:
            indicator_color = QColor(231, 76, 60)  # Red for high importance
        elif self.importance > 30:
            indicator_color = QColor(241, 196, 15)  # Yellow for medium importance
        else:
            indicator_color = QColor(46, 204, 113)  # Green for low importance
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(indicator_color)
        painter.drawRoundedRect(indicator_rect, 3, 3)
        
        # Content area
        content_rect = rect.adjusted(indicator_width + 6, 0, 0, 0)
        
        # Draw title
        title_rect = QRect(content_rect.left() + 5, content_rect.top() + 5, 
                          content_rect.width() - 10, 20)
        painter.setPen(title_color)
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.drawText(title_rect, Qt.AlignLeft | Qt.AlignVCenter, self.title)
        
        # Draw timestamp
        timestamp_rect = QRect(content_rect.right() - 100, content_rect.top() + 5, 
                              95, 20)
        painter.setPen(text_color)
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(timestamp_rect, Qt.AlignRight | Qt.AlignVCenter, self.timestamp)
        
        # Draw content preview
        preview_rect = QRect(content_rect.left() + 5, title_rect.bottom() + 5,
                           content_rect.width() - 10, content_rect.height() - 35)
        painter.setPen(text_color)
        painter.setFont(QFont("Segoe UI", 9))
        
        # Elide text if it's too long
        elidedText = self.fontMetrics().elidedText(
            self.content_preview, Qt.ElideRight, preview_rect.width() * 2)
        
        # Draw text with word wrapping (simplified approach)
        painter.drawText(preview_rect, Qt.AlignLeft | Qt.TextWordWrap, elidedText)
    
    def enterEvent(self, event):
        """Mouse enter event"""
        self.is_highlighted = True
        self.update()
        
    def leaveEvent(self, event):
        """Mouse leave event"""
        self.is_highlighted = False
        self.update()
        
    def mousePressEvent(self, event):
        """Mouse press event"""
        if event.button() == Qt.LeftButton:
            self.is_selected = not self.is_selected
            self.update()
            self.node_clicked.emit(self.memory_id)

class MemoryScrollPanel(QWidget):
    """Panel for displaying and interacting with memory nodes"""
    
    memory_selected = pyqtSignal(str)  # Signal emitted when a memory is selected
    memory_filtered = pyqtSignal(dict)  # Signal emitted when memory filter changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.memories = []  # List to store memory nodes
        self.filter_text = ""
        self.importance_filter = 0  # Min importance (0-100)
        self.time_filter = "all"  # all, today, week, month
        self.category_filter = "all"  # all, conversation, concept, insight
        self.initUI()
        
    def initUI(self):
        """Initialize the memory scroll panel UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        
        # Header with title and controls
        header_layout = QHBoxLayout()
        
        # Title
        title = QLabel("Memory Scroll")
        title.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #3498DB;
            margin-bottom: 10px;
        """)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Search box
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search memories...")
        self.search_box.setStyleSheet("""
            QLineEdit {
                background-color: #1E2C3A;
                color: #ECF0F1;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                border: 1px solid #2C3E50;
            }
            QLineEdit:focus {
                border: 1px solid #3498DB;
            }
        """)
        self.search_box.textChanged.connect(self.filter_memories)
        header_layout.addWidget(self.search_box)
        
        # Filter button
        self.filter_button = QToolButton()
        self.filter_button.setText("Filter")
        self.filter_button.setPopupMode(QToolButton.InstantPopup)
        self.filter_button.setStyleSheet("""
            QToolButton {
                background-color: #2C3E50;
                color: white;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
            }
            QToolButton:hover {
                background-color: #34495E;
            }
        """)
        header_layout.addWidget(self.filter_button)
        
        main_layout.addLayout(header_layout)
        
        # Filter options panel (initially hidden, toggled by filter button)
        self.filter_panel = QGroupBox("Filter Options")
        self.filter_panel.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 1px solid #2C3E50;
                border-radius: 5px;
                margin-top: 10px;
                padding: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #3498DB;
            }
        """)
        filter_layout = QVBoxLayout(self.filter_panel)
        
        # Time filter
        time_layout = QHBoxLayout()
        time_label = QLabel("Time Period:")
        time_label.setStyleSheet("color: #ECF0F1; font-size: 14px;")
        self.time_combo = QComboBox()
        self.time_combo.addItems(["All Time", "Today", "This Week", "This Month", "This Year"])
        self.time_combo.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
            font-size: 14px;
        """)
        self.time_combo.currentTextChanged.connect(self.filter_memories)
        time_layout.addWidget(time_label)
        time_layout.addWidget(self.time_combo)
        filter_layout.addLayout(time_layout)
        
        # Category filter
        category_layout = QHBoxLayout()
        category_label = QLabel("Category:")
        category_label.setStyleSheet("color: #ECF0F1; font-size: 14px;")
        self.category_combo = QComboBox()
        self.category_combo.addItems(["All Categories", "Conversations", "Concepts", "Insights", "Decisions"])
        self.category_combo.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
            font-size: 14px;
        """)
        self.category_combo.currentTextChanged.connect(self.filter_memories)
        category_layout.addWidget(category_label)
        category_layout.addWidget(self.category_combo)
        filter_layout.addLayout(category_layout)
        
        # Importance filter
        importance_layout = QHBoxLayout()
        importance_label = QLabel("Min Importance:")
        importance_label.setStyleSheet("color: #ECF0F1; font-size: 14px;")
        self.importance_slider = QSlider(Qt.Horizontal)
        self.importance_slider.setMinimum(0)
        self.importance_slider.setMaximum(100)
        self.importance_slider.setValue(0)
        self.importance_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #2C3E50;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498DB;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        self.importance_value = QLabel("0")
        self.importance_value.setStyleSheet("color: #ECF0F1; font-size: 14px;")
        self.importance_slider.valueChanged.connect(self.update_importance_value)
        self.importance_slider.valueChanged.connect(self.filter_memories)
        importance_layout.addWidget(importance_label)
        importance_layout.addWidget(self.importance_slider)
        importance_layout.addWidget(self.importance_value)
        filter_layout.addLayout(importance_layout)
        
        # Apply button
        apply_layout = QHBoxLayout()
        apply_layout.addStretch()
        self.reset_button = QPushButton("Reset Filters")
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #7F8C8D;
                color: white;
                border-radius: 5px;
                padding: 8px 12px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #95A5A6;
            }
            QPushButton:pressed {
                background-color: #34495E;
            }
        """)
        self.reset_button.clicked.connect(self.reset_filters)
        apply_layout.addWidget(self.reset_button)
        filter_layout.addLayout(apply_layout)
        
        # Initially hidden
        self.filter_panel.setVisible(False)
        
        # Connect filter button to toggle filter panel
        self.filter_button.clicked.connect(self.toggle_filter_panel)
        
        main_layout.addWidget(self.filter_panel)
        
        # Memory nodes scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background: #1E2C3A;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #2C3E50;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        
        # Container for memory nodes
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setSpacing(10)
        self.scroll_layout.setContentsMargins(5, 5, 5, 5)
        self.scroll_area.setWidget(self.scroll_content)
        
        main_layout.addWidget(self.scroll_area)
        
        # Status bar showing number of visible memories
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Showing 0 memories")
        self.status_label.setStyleSheet("""
            color: #95A5A6;
            font-size: 12px;
        """)
        status_layout.addWidget(self.status_label)
        
        # Load more button
        self.load_more_button = QPushButton("Load More")
        self.load_more_button.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50;
                color: white;
                border-radius: 5px;
                padding: 5px 10px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #34495E;
            }
        """)
        status_layout.addWidget(self.load_more_button)
        
        main_layout.addLayout(status_layout)
        
        # Add mock data for demo
        self.add_mock_memories()
        self.update_status_label()
        
    def toggle_filter_panel(self):
        """Toggle visibility of the filter panel"""
        self.filter_panel.setVisible(not self.filter_panel.isVisible())
        
    def update_importance_value(self, value):
        """Update the importance value label when slider changes"""
        self.importance_value.setText(str(value))
        
    def filter_memories(self):
        """Filter memories based on current filter settings"""
        self.filter_text = self.search_box.text().lower()
        self.importance_filter = self.importance_slider.value()
        self.time_filter = self.time_combo.currentText().lower()
        self.category_filter = self.category_combo.currentText().lower()
        
        # Apply filters
        # For demo purposes, just hiding/showing existing nodes
        for i in range(self.scroll_layout.count()):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                should_show = True
                
                # Text filter
                if self.filter_text and self.filter_text not in widget.title.lower() and self.filter_text not in widget.content_preview.lower():
                    should_show = False
                
                # Importance filter
                if widget.importance < self.importance_filter:
                    should_show = False
                    
                # In a real implementation, we would also check time and category
                
                widget.setVisible(should_show)
        
        # Emit filter changed signal
        self.memory_filtered.emit({
            "text": self.filter_text,
            "importance": self.importance_filter,
            "time": self.time_filter,
            "category": self.category_filter
        })
        
        self.update_status_label()
        
    def update_status_label(self):
        """Update the status label with count of visible memories"""
        visible_count = 0
        for i in range(self.scroll_layout.count()):
            if self.scroll_layout.itemAt(i).widget().isVisible():
                visible_count += 1
                
        total_count = self.scroll_layout.count()
        self.status_label.setText(f"Showing {visible_count} of {total_count} memories")
        
    def reset_filters(self):
        """Reset all filters to default values"""
        self.search_box.clear()
        self.importance_slider.setValue(0)
        self.time_combo.setCurrentIndex(0)
        self.category_combo.setCurrentIndex(0)
        
    def add_memory_node(self, memory_id, title, timestamp, content_preview, importance=50):
        """Add a new memory node to the scroll area"""
        memory_node = MemoryNodeWidget(memory_id, title, timestamp, content_preview, importance)
        memory_node.node_clicked.connect(self.on_memory_clicked)
        self.scroll_layout.insertWidget(0, memory_node)  # Add new nodes at the top
        
    def on_memory_clicked(self, memory_id):
        """Handle memory node click event"""
        self.memory_selected.emit(memory_id)
        
    def add_mock_memories(self):
        """Add some mock memory nodes for demonstration"""
        # Some sample data
        memories = [
            {
                "id": "mem1",
                "title": "Conversation with User",
                "timestamp": "Today, 10:23 AM",
                "preview": "Discussed neural network architecture and learning rates for image classification problem.",
                "importance": 75
            },
            {
                "id": "mem2",
                "title": "Mathematical Insight",
                "timestamp": "Yesterday, 2:45 PM",
                "preview": "Connected Lagrangian mechanics to neural optimization through the lens of variational calculus.",
                "importance": 85
            },
            {
                "id": "mem3",
                "title": "Applied Learning",
                "timestamp": "3 days ago",
                "preview": "Used reinforcement learning principles to solve user's resource allocation problem.",
                "importance": 60
            },
            {
                "id": "mem4",
                "title": "Knowledge Integration",
                "timestamp": "Last week",
                "preview": "Merged concepts from quantum computing and neural networks to explain superposition states.",
                "importance": 90
            },
            {
                "id": "mem5",
                "title": "Pattern Recognition",
                "timestamp": "Last week",
                "preview": "Identified common pattern in user queries related to overfitting and regularization.",
                "importance": 40
            },
            {
                "id": "mem6",
                "title": "Conceptual Framework",
                "timestamp": "2 weeks ago",
                "preview": "Developed mental model for explaining gradient descent using physical analogies.",
                "importance": 70
            },
            {
                "id": "mem7",
                "title": "Learning Milestone",
                "timestamp": "3 weeks ago",
                "preview": "Integrated feedback from previous conversations to improve explanation quality.",
                "importance": 55
            },
            {
                "id": "mem8",
                "title": "Core Principle",
                "timestamp": "Last month",
                "preview": "Fundamental insight about the relationship between data distribution and model architecture.",
                "importance": 95
            }
        ]
        
        for memory in memories:
            self.add_memory_node(
                memory["id"], 
                memory["title"], 
                memory["timestamp"], 
                memory["preview"], 
                memory["importance"]
            ) 