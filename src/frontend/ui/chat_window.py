from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTextEdit, QLineEdit, QPushButton, QLabel,
                             QSlider, QSpinBox, QComboBox)
from PySide6.QtCore import Qt, Signal
from frontend.bridge_integration import get_bridge
from mistralai import MistralClient
import os
from dotenv import load_dotenv

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mistral Chat Interface")
        self.setGeometry(200, 200, 800, 600)
        
        # Initialize version bridge
        self.bridge = get_bridge()
        
        # Load environment variables
        load_dotenv()
        
        # Initialize Mistral client
        self.mistral_client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Version selection
        version_layout = QHBoxLayout()
        version_label = QLabel("Version:")
        self.version_combo = QComboBox()
        for version in range(1, 6):
            self.version_combo.addItem(f"v{version}")
        version_layout.addWidget(version_label)
        version_layout.addWidget(self.version_combo)
        main_layout.addLayout(version_layout)
        
        # Create chat display area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        main_layout.addWidget(self.chat_display)
        
        # Create input area
        input_layout = QHBoxLayout()
        self.message_input = QLineEdit()
        self.send_button = QPushButton("Send")
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.send_button)
        main_layout.addLayout(input_layout)
        
        # Create parameter controls
        params_layout = QHBoxLayout()
        
        # Temperature control
        temp_layout = QVBoxLayout()
        temp_label = QLabel("Temperature:")
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(70)
        self.temp_value = QSpinBox()
        self.temp_value.setRange(0, 100)
        self.temp_value.setValue(70)
        temp_layout.addWidget(temp_label)
        temp_layout.addWidget(self.temp_slider)
        temp_layout.addWidget(self.temp_value)
        params_layout.addLayout(temp_layout)
        
        # Top-k control
        topk_layout = QVBoxLayout()
        topk_label = QLabel("Top-k:")
        self.topk_slider = QSlider(Qt.Horizontal)
        self.topk_slider.setRange(1, 100)
        self.topk_slider.setValue(50)
        self.topk_value = QSpinBox()
        self.topk_value.setRange(1, 100)
        self.topk_value.setValue(50)
        topk_layout.addWidget(topk_label)
        topk_layout.addWidget(self.topk_slider)
        topk_layout.addWidget(self.topk_value)
        params_layout.addLayout(topk_layout)
        
        # Top-p control
        topp_layout = QVBoxLayout()
        topp_label = QLabel("Top-p:")
        self.topp_slider = QSlider(Qt.Horizontal)
        self.topp_slider.setRange(0, 100)
        self.topp_slider.setValue(90)
        self.topp_value = QSpinBox()
        self.topp_value.setRange(0, 100)
        self.topp_value.setValue(90)
        topp_layout.addWidget(topp_label)
        topp_layout.addWidget(self.topp_slider)
        topp_layout.addWidget(self.topp_value)
        params_layout.addLayout(topp_layout)
        
        main_layout.addLayout(params_layout)
        
        # Connect signals
        self.send_button.clicked.connect(self.send_message)
        self.message_input.returnPressed.connect(self.send_message)
        self.temp_slider.valueChanged.connect(self.temp_value.setValue)
        self.temp_value.valueChanged.connect(self.temp_slider.setValue)
        self.topk_slider.valueChanged.connect(self.topk_value.setValue)
        self.topk_value.valueChanged.connect(self.topk_slider.setValue)
        self.topp_slider.valueChanged.connect(self.topp_value.setValue)
        self.topp_value.valueChanged.connect(self.topp_slider.setValue)
    
    def send_message(self):
        message = self.message_input.text()
        if not message:
            return
            
        # Get current version
        version = self.version_combo.currentText()
        
        # Get current parameters
        temperature = self.temp_value.value() / 100.0
        top_k = self.topk_value.value()
        top_p = self.topp_value.value() / 100.0
        
        # Display user message
        self.chat_display.append(f"You: {message}")
        self.message_input.clear()
        
        try:
            # Get version-specific context from bridge
            context = self.bridge.get_component(version, "context")
            
            # Prepare message for Mistral
            messages = [
                {"role": "system", "content": "You are a helpful assistant integrated with a neural network system."},
                {"role": "user", "content": message}
            ]
            
            if context:
                messages.insert(1, {"role": "system", "content": f"Context from {version}: {context}"})
            
            # Get response from Mistral
            response = self.mistral_client.chat(
                model="mistral-tiny",
                messages=messages,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # Display response
            self.chat_display.append(f"Assistant: {response.choices[0].message.content}")
            
            # Send response to version bridge for processing
            self.bridge.send_command(
                version,
                "process_response",
                {"message": response.choices[0].message.content}
            )
            
        except Exception as e:
            self.chat_display.append(f"Error: {str(e)}") 