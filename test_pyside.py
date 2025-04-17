import sys
from PySide6.QtWidgets import QApplication, QLabel

print("PySide6 import successful")
print("Creating application...")
app = QApplication(sys.argv)
print("Creating label...")
label = QLabel("Test")
print("Showing label...")
label.show()
print("Starting event loop...")
sys.exit(app.exec()) 