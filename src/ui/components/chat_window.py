#!/usr/bin/env python3
"""
Chat Window Component

This module implements the chat window for real-time communication
using the V7.5 signal system with logic gate visualization.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QComboBox,
    QLabel, QStatusBar, QToolBar, QAction, QMenu,
    QScrollArea, QFrame, QSplitter, QGraphicsView,
    QGraphicsScene, QGraphicsItem, QGraphicsEllipseItem,
    QGraphicsLineItem, QGraphicsTextItem, QProgressBar,
    QDialog, QMessageBox, QInputDialog, QTabWidget,
    QTabBar, QStyle, QStyleOptionTab, QStylePainter
)
from PySide6.QtCore import (
    Qt, Signal, Slot, QTimer, QPointF, QRectF,
    QPropertyAnimation, QEasingCurve, QSize
)
from PySide6.QtGui import (
    QTextCursor, QTextCharFormat, QColor, QFont,
    QIcon, QActionGroup, QTextOption, QPen, QBrush,
    QPainter, QPainterPath, QLinearGradient, QKeySequence
)

from src.integration.signal_system import SignalBus, SignalComponent

logger = logging.getLogger(__name__)

class LogicGateItem(QGraphicsItem):
    """Graphics item representing a logic gate."""
    
    def __init__(self, gate_type: str, pos: QPointF, parent=None):
        super().__init__(parent)
        self._gate_type = gate_type
        self._pos = pos
        self._output = 0.0
        self._state = "closed"
        self._connections: List[QGraphicsLineItem] = []
        self._active = False
        self._pulse_animation = QPropertyAnimation(self, b"opacity")
        self._pulse_animation.setDuration(1000)
        self._pulse_animation.setLoopCount(-1)
        self._pulse_animation.setEasingCurve(QEasingCurve.InOutSine)
        self._pulse_animation.setStartValue(0.3)
        self._pulse_animation.setEndValue(1.0)
        
        # Set gate properties based on type
        self._set_gate_properties()
        
    def _set_gate_properties(self):
        """Set gate properties based on type."""
        gate_properties = {
            "AND": {"color": "#FFA500", "symbol": "&", "gradient": ["#FFA500", "#FFD700"]},
            "OR": {"color": "#0000FF", "symbol": "≥1", "gradient": ["#0000FF", "#4169E1"]},
            "XOR": {"color": "#800080", "symbol": "=1", "gradient": ["#800080", "#9932CC"]},
            "NOT": {"color": "#FF0000", "symbol": "1", "gradient": ["#FF0000", "#DC143C"]},
            "NAND": {"color": "#FFFF00", "symbol": "&", "gradient": ["#FFFF00", "#FFD700"]},
            "NOR": {"color": "#00FF00", "symbol": "≥1", "gradient": ["#00FF00", "#32CD32"]}
        }
        
        props = gate_properties.get(self._gate_type, gate_properties["AND"])
        self._color = QColor(props["color"])
        self._symbol = props["symbol"]
        self._gradient = props["gradient"]
            
    def boundingRect(self) -> QRectF:
        return QRectF(-20, -20, 40, 40)
        
    def paint(self, painter: QPainter, option, widget):
        # Create gradient
        gradient = QLinearGradient(-15, -15, 15, 15)
        gradient.setColorAt(0, QColor(self._gradient[0]))
        gradient.setColorAt(1, QColor(self._gradient[1]))
        
        # Draw gate body
        if self._active:
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.setBrush(QBrush(gradient))
        else:
            painter.setPen(QPen(self._color.darker(150), 2))
            painter.setBrush(QBrush(self._color.darker(200)))
            
        painter.drawEllipse(-15, -15, 30, 30)
        
        # Draw symbol
        painter.setPen(QPen(Qt.black, 1))
        painter.drawText(-10, 5, self._symbol)
        
        # Draw state indicator
        if self._active:
            state_color = QColor("#00FF00") if self._state == "open" else QColor("#FF0000")
            painter.setPen(QPen(state_color, 2))
            painter.setBrush(QBrush(state_color))
            painter.drawEllipse(-5, -5, 10, 10)
        
    def update_state(self, output: float, active: bool = False):
        """Update gate state based on output value and active status."""
        self._output = output
        self._state = "open" if output > 0.8 else "closed"
        
        if active != self._active:
            self._active = active
            if active:
                self._pulse_animation.start()
            else:
                self._pulse_animation.stop()
                self.setOpacity(1.0)
                
        self.update()
        
    def add_connection(self, line: QGraphicsLineItem):
        """Add a connection line to this gate."""
        self._connections.append(line)
        
    def get_pos(self) -> QPointF:
        """Get gate position."""
        return self._pos

class CustomTabBar(QTabBar):
    """Custom tab bar with add button and context menu."""
    
    add_tab_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMovable(True)
        self.setTabsClosable(True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QStylePainter(self)
        option = QStyleOptionTab()
        
        # Draw add button
        for i in range(self.count()):
            self.initStyleOption(option, i)
            if i == self.count() - 1:
                # Draw add button
                painter.drawControl(QStyle.CE_TabBarTab, option)
                
    def _show_context_menu(self, pos):
        menu = QMenu(self)
        add_action = menu.addAction("Add Tab")
        add_action.triggered.connect(self.add_tab_requested)
        menu.exec_(self.mapToGlobal(pos))

class SystemDialog(QDialog):
    """Dialog for system prompts and user input."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System Prompt")
        self.setModal(True)
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Message label
        self._message_label = QLabel()
        self._message_label.setWordWrap(True)
        layout.addWidget(self._message_label)
        
        # Input field (if needed)
        self._input_field = QLineEdit()
        self._input_field.setVisible(False)
        layout.addWidget(self._input_field)
        
        # Buttons
        button_layout = QHBoxLayout()
        self._ok_button = QPushButton("OK")
        self._cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self._ok_button)
        button_layout.addWidget(self._cancel_button)
        layout.addLayout(button_layout)
        
        # Connect signals
        self._ok_button.clicked.connect(self.accept)
        self._cancel_button.clicked.connect(self.reject)
        
    def set_message(self, message: str):
        self._message_label.setText(message)
        
    def set_input_required(self, required: bool):
        self._input_field.setVisible(required)
        if required:
            self._input_field.setFocus()
            
    def get_input(self) -> str:
        return self._input_field.text()

class ChatWindow(QMainWindow):
    """Main chat window for real-time communication."""
    
    # Signals
    message_sent = Signal(str, str)  # message, version
    version_changed = Signal(str)  # new version
    gate_state_changed = Signal(str, float)  # gate_type, output
    backend_state_changed = Signal(bool)  # active
    tab_changed = Signal(int)  # tab index
    new_tab_requested = Signal(str)  # tab name
    
    def __init__(self, signal_bus: SignalBus):
        """Initialize the chat window."""
        super().__init__()
        
        self._signal_bus = signal_bus
        self._current_version = 'v7.5'
        self._component = SignalComponent(self._current_version)
        self._message_history: List[Dict[str, Any]] = []
        self._typing_indicator = False
        self._typing_timer = QTimer()
        self._typing_timer.timeout.connect(self._update_typing_indicator)
        self._logic_gates: Dict[str, LogicGateItem] = {}
        self._gate_scene = QGraphicsScene()
        self._backend_active = False
        self._gate_update_timer = QTimer()
        self._gate_update_timer.timeout.connect(self._check_backend_state)
        self._last_backend_state = {}
        self._state_change_animation = QPropertyAnimation(self._gate_panel, b"maximumWidth")
        self._state_change_animation.setDuration(500)
        self._state_change_animation.setEasingCurve(QEasingCurve.InOutQuad)
        self._system_dialog = SystemDialog(self)
        self._pending_requests: List[Dict[str, Any]] = []
        
        self._setup_ui()
        self._connect_signals()
        self._apply_styles()
        
        # Start backend state checking
        self._gate_update_timer.start(100)  # Check every 100ms
        
    def _setup_ui(self):
        """Set up the user interface."""
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create toolbar
        self._create_toolbar()
        
        # Create tab widget
        self._tab_widget = QTabWidget()
        self._tab_widget.setTabBar(CustomTabBar())
        self._tab_widget.tabBar().add_tab_requested.connect(self._handle_new_tab_request)
        self._tab_widget.currentChanged.connect(self._handle_tab_change)
        layout.addWidget(self._tab_widget)
        
        # Create initial tab
        self._create_chat_tab()
        
        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._update_status("Disconnected")
        
        # Window settings
        self.setWindowTitle("Lumina Chat")
        self.setMinimumSize(800, 600)
        
    def _create_chat_tab(self, name: str = "Chat"):
        """Create a new chat tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel for chat
        chat_panel = QWidget()
        chat_layout = QVBoxLayout(chat_panel)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        
        # Chat display area
        chat_frame = QFrame()
        chat_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        chat_layout_inner = QVBoxLayout(chat_frame)
        chat_layout_inner.setContentsMargins(5, 5, 5, 5)
        
        # Chat display with scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        chat_display = QTextEdit()
        chat_display.setReadOnly(True)
        chat_display.setLineWrapMode(QTextEdit.WidgetWidth)
        chat_display.setWordWrapMode(QTextOption.WordWrap)
        scroll_area.setWidget(chat_display)
        
        chat_layout_inner.addWidget(scroll_area)
        chat_layout.addWidget(chat_frame)
        
        # Input area
        input_frame = QFrame()
        input_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        input_layout = QVBoxLayout(input_frame)
        input_layout.setContentsMargins(5, 5, 5, 5)
        
        # Version selector
        version_layout = QHBoxLayout()
        version_label = QLabel("Version:")
        version_combo = QComboBox()
        version_combo.addItems(['v7.5', 'v7.0', 'v6.0', 'v5.0'])
        version_combo.setCurrentText(self._current_version)
        version_layout.addWidget(version_label)
        version_layout.addWidget(version_combo)
        input_layout.addLayout(version_layout)
        
        # Message input
        input_row = QHBoxLayout()
        message_input = QLineEdit()
        message_input.setPlaceholderText("Type a message...")
        send_button = QPushButton("Send")
        send_button.setFixedWidth(80)
        input_row.addWidget(message_input)
        input_row.addWidget(send_button)
        input_layout.addLayout(input_row)
        
        # Typing indicator
        typing_label = QLabel()
        typing_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        typing_label.setStyleSheet("color: #666666; font-style: italic;")
        input_layout.addWidget(typing_label)
        
        chat_layout.addWidget(input_frame)
        
        # Right panel for logic gates
        gate_panel = QWidget()
        gate_layout = QVBoxLayout(gate_panel)
        gate_layout.setContentsMargins(0, 0, 0, 0)
        
        # Backend status indicator
        status_frame = QFrame()
        status_layout = QHBoxLayout(status_frame)
        backend_status = QLabel("Backend: Inactive")
        backend_status.setStyleSheet("color: #ff4d4d; font-weight: bold;")
        backend_progress = QProgressBar()
        backend_progress.setRange(0, 100)
        backend_progress.setValue(0)
        status_layout.addWidget(backend_status)
        status_layout.addWidget(backend_progress)
        gate_layout.addWidget(status_frame)
        
        # Gate visualization
        gate_view = QGraphicsView(self._gate_scene)
        gate_view.setRenderHint(QPainter.Antialiasing)
        gate_view.setBackgroundBrush(QBrush(QColor("#1e1e1e")))
        gate_layout.addWidget(gate_view)
        
        # Add panels to splitter
        splitter.addWidget(chat_panel)
        splitter.addWidget(gate_panel)
        
        # Set initial splitter sizes
        splitter.setSizes([600, 400])
        
        # Store references
        tab_data = {
            'widget': tab,
            'chat_display': chat_display,
            'message_input': message_input,
            'send_button': send_button,
            'version_combo': version_combo,
            'typing_label': typing_label,
            'backend_status': backend_status,
            'backend_progress': backend_progress,
            'gate_panel': gate_panel,
            'gate_view': gate_view
        }
        
        # Add tab
        index = self._tab_widget.addTab(tab, name)
        self._tab_widget.setTabData(index, tab_data)
        
        # Connect signals
        message_input.returnPressed.connect(lambda: self._send_message(index))
        send_button.clicked.connect(lambda: self._send_message(index))
        version_combo.currentTextChanged.connect(self._handle_version_change)
        message_input.textChanged.connect(self._handle_typing)
        
        return index
        
    def _handle_new_tab_request(self):
        """Handle request for new tab."""
        name, ok = QInputDialog.getText(
            self,
            "New Tab",
            "Enter tab name:",
            QLineEdit.Normal
        )
        
        if ok and name:
            self.new_tab_requested.emit(name)
            self._create_chat_tab(name)
            
    def _handle_tab_change(self, index: int):
        """Handle tab change."""
        self.tab_changed.emit(index)
        
    def _handle_backend_request(self, request: Dict[str, Any]):
        """Handle backend request for information."""
        try:
            request_type = request.get('type')
            message = request.get('message', '')
            require_input = request.get('require_input', False)
            
            # Show system dialog
            self._system_dialog.set_message(message)
            self._system_dialog.set_input_required(require_input)
            
            if self._system_dialog.exec_() == QDialog.Accepted:
                response = {
                    'type': request_type,
                    'success': True,
                    'input': self._system_dialog.get_input() if require_input else None
                }
                self._signal_bus.send_response(response)
            else:
                response = {
                    'type': request_type,
                    'success': False,
                    'error': 'User cancelled'
                }
                self._signal_bus.send_response(response)
                
        except Exception as e:
            logger.error(f"Error handling backend request: {e}")
            self._show_error(f"Error handling request: {str(e)}")
            
    def _show_error(self, message: str):
        """Show error message."""
        QMessageBox.critical(self, "Error", message)
        
    def _check_backend_state(self):
        """Check if backend is actively processing signals."""
        try:
            # Get backend state from signal bus
            backend_state = self._signal_bus.get_backend_state()
            
            # Check for pending requests
            requests = backend_state.get('requests', [])
            for request in requests:
                if request not in self._pending_requests:
                    self._pending_requests.append(request)
                    self._handle_backend_request(request)
            
            # Check if state has changed
            if backend_state != self._last_backend_state:
                self._last_backend_state = backend_state
                
                # Update backend active status
                new_active = backend_state.get('active', False)
                if new_active != self._backend_active:
                    self._backend_active = new_active
                    self._handle_backend_state_change(new_active)
                    
                # Update gate states if backend is active
                if self._backend_active:
                    self._update_gate_states_from_backend(backend_state)
                else:
                    # Reset all gates to inactive state
                    for gate in self._logic_gates.values():
                        gate.update_state(0.0, False)
                        
                # Update backend status indicator for current tab
                current_tab_data = self._get_current_tab_data()
                if current_tab_data:
                    self._update_backend_status(
                        current_tab_data['backend_status'],
                        current_tab_data['backend_progress'],
                        backend_state
                    )
                    
        except Exception as e:
            logger.error(f"Error checking backend state: {e}")
            self._show_error(f"Error checking backend state: {str(e)}")
            
    def _get_current_tab_data(self) -> Optional[Dict[str, Any]]:
        """Get data for current tab."""
        try:
            current_index = self._tab_widget.currentIndex()
            return self._tab_widget.tabData(current_index)
        except Exception as e:
            logger.error(f"Error getting current tab data: {e}")
            return None
            
    def _send_message(self, tab_index: int):
        """Send message through signal system."""
        try:
            tab_data = self._tab_widget.tabData(tab_index)
            if not tab_data:
                return
                
            message = tab_data['message_input'].text().strip()
            if not message:
                return
                
            # Send message
            self._signal_bus.send_message(message, self._current_version)
            
            # Add to history
            timestamp = datetime.now().strftime("%H:%M:%S")
            self._message_history.append({
                'timestamp': timestamp,
                'version': self._current_version,
                'message': message,
                'received': False
            })
            
            # Update gate states if backend is active
            if self._backend_active:
                self._update_gate_states(message)
            
            # Clear input
            tab_data['message_input'].clear()
            
            # Emit signal
            self.message_sent.emit(message, self._current_version)
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self._show_error(f"Error sending message: {str(e)}")
            
    def closeEvent(self, event):
        """Handle window close event."""
        try:
            # Clean up signal system
            self._signal_bus.unregister_component(self._current_version)
            self._component.cleanup()
            
            # Stop timers
            self._typing_timer.stop()
            self._gate_update_timer.stop()
            
            # Stop animations
            self._state_change_animation.stop()
            for gate in self._logic_gates.values():
                gate._pulse_animation.stop()
            
            # Accept close event
            event.accept()
            
        except Exception as e:
            logger.error(f"Error during close: {e}")
            event.accept() 