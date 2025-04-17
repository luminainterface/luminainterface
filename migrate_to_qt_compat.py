#!/usr/bin/env python3
"""
Migration script to update PyQt5 imports to use the compatibility layer.
This helps with the transition from PyQt5 to PySide6.
"""

import os
import re
import sys
from pathlib import Path

def migrate_file(file_path):
    """Replace direct imports with compatibility layer"""
    print(f"Processing: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Store original content for comparison
    original_content = content
    
    # Replace PyQt5 imports
    content = re.sub(
        r'from PyQt5\.QtWidgets import (.+)',
        r'from src.v5.ui.qt_compat import QtWidgets  # \1',
        content
    )
    
    content = re.sub(
        r'from PyQt5\.QtCore import (.+)',
        r'from src.v5.ui.qt_compat import QtCore  # \1',
        content
    )
    
    content = re.sub(
        r'from PyQt5\.QtGui import (.+)',
        r'from src.v5.ui.qt_compat import QtGui  # \1',
        content
    )
    
    # Replace signal definitions
    content = re.sub(
        r'(\w+) = pyqtSignal\(([^)]+)\)',
        r'\1 = Signal(\2)',
        content
    )
    
    # Replace slot decorators
    content = re.sub(
        r'@pyqtSlot\(([^)]*)\)',
        r'@Slot(\1)',
        content
    )
    
    # Add imports if needed
    if 'Signal' in content and 'from src.v5.ui.qt_compat import Signal' not in content:
        content = 'from src.v5.ui.qt_compat import Signal\n' + content
    
    if 'Slot' in content and 'from src.v5.ui.qt_compat import Slot' not in content:
        content = 'from src.v5.ui.qt_compat import Slot\n' + content
    
    # Replace QWidget with QtWidgets.QWidget, etc.
    widget_classes = [
        'QWidget', 'QMainWindow', 'QDialog', 'QPushButton', 'QLabel', 
        'QLineEdit', 'QTextEdit', 'QVBoxLayout', 'QHBoxLayout', 'QGridLayout',
        'QFrame', 'QTabWidget', 'QComboBox', 'QSpinBox', 'QCheckBox',
        'QRadioButton', 'QSlider', 'QProgressBar', 'QScrollArea', 'QGroupBox'
    ]
    
    for widget_class in widget_classes:
        # Replace class inheritance and instantiation
        pattern = r'(?<!\w)' + widget_class + r'(?!\w)'
        replacement = 'QtWidgets.' + widget_class
        content = re.sub(pattern, replacement, content)
    
    # Clean up multiple imports of compatibility modules
    import_lines = set()
    content_lines = content.split('\n')
    cleaned_lines = []
    
    for line in content_lines:
        if line.startswith('from src.v5.ui.qt_compat import'):
            import_lines.add(line)
        else:
            cleaned_lines.append(line)
    
    # Add all unique imports at the top
    if import_lines:
        content = '\n'.join(sorted(list(import_lines))) + '\n\n' + '\n'.join(cleaned_lines)
    
    # Only write if changes were made
    if content != original_content:
        print(f"  Making changes to {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    else:
        print(f"  No changes needed for {file_path}")

def process_directory(directory):
    """Process all Python files in directory and subdirectories"""
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                migrate_file(full_path)
                count += 1
    
    print(f"Processed {count} files in {directory}")

def main():
    """Main function to run the script"""
    if len(sys.argv) < 2:
        print("Usage: python migrate_to_qt_compat.py <directory_path>")
        print("Example: python migrate_to_qt_compat.py src/v5/ui/panels")
        return
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        return
    
    process_directory(directory)
    print("Migration complete!")

if __name__ == "__main__":
    main() 