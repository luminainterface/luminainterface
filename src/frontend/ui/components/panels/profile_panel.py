from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit
from PySide6.QtCore import Qt
from typing import Optional
import logging

class ProfilePanel(QWidget):
    """Panel for managing user profile and settings."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.setMinimumWidth(250)
        
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop)
        
        self.initialize_ui()
        
    def initialize_ui(self):
        """Initialize the profile panel UI."""
        try:
            # Profile header
            header = QLabel("Profile")
            header.setStyleSheet("font-size: 18px; font-weight: bold;")
            self.layout.addWidget(header)
            
            # Username field
            username_label = QLabel("Username:")
            self.username_edit = QLineEdit()
            self.layout.addWidget(username_label)
            self.layout.addWidget(self.username_edit)
            
            # Version selector
            version_label = QLabel("Version:")
            self.version_edit = QLineEdit()
            self.version_edit.setReadOnly(True)
            self.layout.addWidget(version_label)
            self.layout.addWidget(self.version_edit)
            
            # Save button
            save_button = QPushButton("Save Profile")
            save_button.clicked.connect(self.save_profile)
            self.layout.addWidget(save_button)
            
            # Add stretch to push everything to top
            self.layout.addStretch()
            
            # Set styles
            self.setStyleSheet("""
                QWidget {
                    background-color: #2d2d2d;
                    border-radius: 5px;
                    padding: 10px;
                }
                QLabel {
                    color: #ffffff;
                }
                QLineEdit {
                    background-color: #3d3d3d;
                    color: #ffffff;
                    border: 1px solid #4d4d4d;
                    border-radius: 3px;
                    padding: 5px;
                }
                QPushButton {
                    background-color: #4d4d4d;
                    color: #ffffff;
                    border: none;
                    border-radius: 3px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #5d5d5d;
                }
            """)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize profile panel UI: {str(e)}")
            raise
            
    def save_profile(self):
        """Save profile changes."""
        try:
            username = self.username_edit.text()
            self.logger.info(f"Saving profile for user: {username}")
            # Save profile logic here
        except Exception as e:
            self.logger.error(f"Failed to save profile: {str(e)}")
            
    def set_version(self, version: str):
        """Set the current version."""
        self.version_edit.setText(version)
        
    def get_username(self) -> str:
        """Get the current username."""
        return self.username_edit.text() 