from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                          QLineEdit, QPushButton, QComboBox, QScrollArea,
                          QFrame, QGroupBox, QSlider, QCheckBox, QSizePolicy)
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal

class ProfilePanel(QWidget):
    """Profile panel for user settings and customization"""
    
    save_clicked = pyqtSignal(dict)  # Signal emitted when changes are saved
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        """Initialize the profile panel UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Header
        header = QLabel("User Profile")
        header.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #3498DB;
            margin-bottom: 10px;
        """)
        layout.addWidget(header)
        
        # User information section
        user_group = QGroupBox("Personal Information")
        user_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 1px solid #2C3E50;
                border-radius: 5px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #3498DB;
            }
        """)
        
        user_layout = QVBoxLayout(user_group)
        
        # Username field
        username_container = QHBoxLayout()
        username_label = QLabel("Username:")
        username_label.setStyleSheet("color: #ECF0F1; font-size: 14px;")
        self.username_input = QLineEdit()
        self.username_input.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 8px;
            font-size: 14px;
        """)
        username_container.addWidget(username_label, 1)
        username_container.addWidget(self.username_input, 2)
        user_layout.addLayout(username_container)
        
        # Preferred language
        language_container = QHBoxLayout()
        language_label = QLabel("Language:")
        language_label.setStyleSheet("color: #ECF0F1; font-size: 14px;")
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "Spanish", "French", "German", "Japanese"])
        self.language_combo.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 8px;
            font-size: 14px;
        """)
        language_container.addWidget(language_label, 1)
        language_container.addWidget(self.language_combo, 2)
        user_layout.addLayout(language_container)
        
        layout.addWidget(user_group)
        
        # Interface preferences
        interface_group = QGroupBox("Interface Preferences")
        interface_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 1px solid #2C3E50;
                border-radius: 5px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #3498DB;
            }
        """)
        
        interface_layout = QVBoxLayout(interface_group)
        
        # Theme selection
        theme_container = QHBoxLayout()
        theme_label = QLabel("Theme:")
        theme_label.setStyleSheet("color: #ECF0F1; font-size: 14px;")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light", "Midnight", "Ocean", "Forest"])
        self.theme_combo.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 8px;
            font-size: 14px;
        """)
        theme_container.addWidget(theme_label, 1)
        theme_container.addWidget(self.theme_combo, 2)
        interface_layout.addLayout(theme_container)
        
        # Font size selection
        font_container = QHBoxLayout()
        font_label = QLabel("Font Size:")
        font_label.setStyleSheet("color: #ECF0F1; font-size: 14px;")
        self.font_slider = QSlider(Qt.Horizontal)
        self.font_slider.setMinimum(8)
        self.font_slider.setMaximum(18)
        self.font_slider.setValue(12)
        self.font_slider.setStyleSheet("""
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
        self.font_size_label = QLabel("12px")
        self.font_size_label.setStyleSheet("color: #ECF0F1; font-size: 14px;")
        self.font_slider.valueChanged.connect(self.update_font_size_label)
        font_container.addWidget(font_label, 1)
        font_container.addWidget(self.font_slider, 2)
        font_container.addWidget(self.font_size_label, 0)
        interface_layout.addLayout(font_container)
        
        # Notifications toggle
        notifications_container = QHBoxLayout()
        notifications_label = QLabel("Enable Notifications:")
        notifications_label.setStyleSheet("color: #ECF0F1; font-size: 14px;")
        self.notifications_checkbox = QCheckBox()
        self.notifications_checkbox.setChecked(True)
        self.notifications_checkbox.setStyleSheet("""
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 4px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #2C3E50;
                border: 2px solid #34495E;
            }
            QCheckBox::indicator:checked {
                background-color: #3498DB;
                border: 2px solid #2980B9;
            }
        """)
        notifications_container.addWidget(notifications_label, 1)
        notifications_container.addWidget(self.notifications_checkbox, 2, alignment=Qt.AlignLeft)
        interface_layout.addLayout(notifications_container)
        
        layout.addWidget(interface_group)
        
        # Usage statistics section (read-only)
        stats_group = QGroupBox("Usage Statistics")
        stats_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 1px solid #2C3E50;
                border-radius: 5px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #3498DB;
            }
        """)
        
        stats_layout = QVBoxLayout(stats_group)
        
        # Conversation count
        conv_label = QLabel("Total Conversations: 42")
        conv_label.setStyleSheet("color: #ECF0F1; font-size: 14px;")
        stats_layout.addWidget(conv_label)
        
        # Messages count
        msg_label = QLabel("Total Messages: 256")
        msg_label.setStyleSheet("color: #ECF0F1; font-size: 14px;")
        stats_layout.addWidget(msg_label)
        
        # Last session
        session_label = QLabel("Last Session: 2025-04-14 12:30:45")
        session_label.setStyleSheet("color: #ECF0F1; font-size: 14px;")
        stats_layout.addWidget(session_label)
        
        # Average session duration
        duration_label = QLabel("Average Session Duration: 23 minutes")
        duration_label.setStyleSheet("color: #ECF0F1; font-size: 14px;")
        stats_layout.addWidget(duration_label)
        
        layout.addWidget(stats_group)
        
        # Save button
        self.save_button = QPushButton("Save Changes")
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #2980B9;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3498DB;
            }
            QPushButton:pressed {
                background-color: #1C587F;
            }
        """)
        self.save_button.clicked.connect(self.save_profile)
        layout.addWidget(self.save_button, alignment=Qt.AlignCenter)
        
        # Add spacer at the bottom
        layout.addStretch()
        
        self.setLayout(layout)
        
    def update_font_size_label(self, value):
        """Update the font size label when slider value changes"""
        self.font_size_label.setText(f"{value}px")
        
    def save_profile(self):
        """Collect and emit profile settings"""
        profile_data = {
            "username": self.username_input.text(),
            "language": self.language_combo.currentText(),
            "theme": self.theme_combo.currentText(),
            "font_size": self.font_slider.value(),
            "notifications": self.notifications_checkbox.isChecked()
        }
        self.save_clicked.emit(profile_data) 