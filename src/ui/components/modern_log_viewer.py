"""
ModernLogViewer component for Lumina GUI
"""
from PySide6.QtWidgets import QTextEdit
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCharFormat, QColor, QTextCursor
import datetime

class ModernLogViewer(QTextEdit):
    """A modern styled log viewer component"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.WidgetWidth)
        
        # Set modern styling
        self.setStyleSheet("""
            QTextEdit {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                padding: 10px;
                font-family: 'Consolas', monospace;
                font-size: 13px;
            }
        """)
        
        # Define log level colors
        self.level_colors = {
            'INFO': '#1A1A1A',      # Dark gray
            'WARNING': '#8B7355',    # Muted brown
            'ERROR': '#8B4545',      # Muted red
            'SUCCESS': '#4A5D4F',    # Muted green
            'DEBUG': '#666666'       # Gray
        }
        
    def add_log(self, message: str, level: str = 'INFO'):
        """Add a log message
        
        Args:
            message: The log message to add
            level: Log level (INFO, WARNING, ERROR, SUCCESS, DEBUG)
        """
        # Create timestamp
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        
        # Create format for timestamp
        time_format = QTextCharFormat()
        time_format.setForeground(QColor('#666666'))
        
        # Create format for level
        level_format = QTextCharFormat()
        level_format.setForeground(QColor(self.level_colors.get(level, '#1A1A1A')))
        level_format.setFontWeight(700)  # Bold
        
        # Create format for message
        msg_format = QTextCharFormat()
        msg_format.setForeground(QColor(self.level_colors.get(level, '#1A1A1A')))
        
        # Get cursor
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Insert timestamp
        cursor.insertText(f'[{timestamp}] ', time_format)
        
        # Insert level
        cursor.insertText(f'[{level}] ', level_format)
        
        # Insert message
        cursor.insertText(f'{message}\n', msg_format)
        
        # Scroll to bottom
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        
    def clear_logs(self):
        """Clear all logs"""
        self.clear() 