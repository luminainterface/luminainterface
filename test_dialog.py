import sys
from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QPushButton

class TestDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test Dialog")
        self.setModal(True)  # Make it modal to ensure it stays on top
        
        layout = QVBoxLayout(self)
        
        label = QLabel("This is a test dialog")
        layout.addWidget(label)
        
        button = QPushButton("Close")
        button.clicked.connect(self.close)
        layout.addWidget(button)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = TestDialog()
    dialog.show()
    sys.exit(app.exec()) 