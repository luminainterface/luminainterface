import pytest
from PySide6.QtWidgets import QApplication
from ui.monitor_window import MonitorWindow
from ui.chat_window import ChatWindow

@pytest.fixture
def app():
    return QApplication([])

def test_monitor_window_creation(app):
    window = MonitorWindow()
    assert window.windowTitle() == "System Monitor"
    assert window.cpu_label.text() == "CPU Usage:"
    assert window.memory_label.text() == "Memory Usage:"
    assert window.gpu_label.text() == "GPU Usage:"

def test_chat_window_creation(app):
    window = ChatWindow()
    assert window.windowTitle() == "Mistral Chat Interface"
    assert window.temp_slider.value() == 70
    assert window.topk_slider.value() == 50
    assert window.topp_slider.value() == 90

def test_chat_window_message_sending(app):
    window = ChatWindow()
    window.message_input.setText("Hello, world!")
    window.send_message()
    assert window.chat_display.toPlainText().endswith("You: Hello, world!")
    assert window.message_input.text() == "" 