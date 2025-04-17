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
    QTabBar, QStyle, QStyleOptionTab, QStylePainter,
    QSizePolicy, QSpacerItem, QToolButton, QListWidget,
    QListWidgetItem
)
from PySide6.QtCore import (
    Qt, Signal, Slot, QTimer, QPointF, QRectF,
    QPropertyAnimation, QEasingCurve, QSize, QMargins
)
from PySide6.QtGui import (
    QTextCursor, QTextCharFormat, QColor, QFont,
    QIcon, QActionGroup, QTextOption, QPen, QBrush,
    QPainter, QPainterPath, QLinearGradient, QKeySequence,
    QPalette, QFontMetrics, QPainterPathStroker
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

class ModernTabBar(QTabBar):
    """Modern tab bar with enhanced styling and functionality."""
    
    add_tab_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMovable(True)
        self.setTabsClosable(True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.setDocumentMode(True)
        self.setExpanding(False)
        self.setElideMode(Qt.ElideRight)
        self.setUsesScrollButtons(True)
        self.customContextMenuRequested.connect(self._show_context_menu)
        
        # Add button
        self._add_button = QToolButton(self)
        self._add_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogNewFolder))
        self._add_button.setFixedSize(24, 24)
        self._add_button.clicked.connect(self.add_tab_requested)
        
    def paintEvent(self, event):
        painter = QStylePainter(self)
        option = QStyleOptionTab()
        
        for i in range(self.count()):
            self.initStyleOption(option, i)
            
            # Custom tab styling
            if self.currentIndex() == i:
                option.state |= QStyle.State_Selected
                
            # Draw tab
            painter.drawControl(QStyle.CE_TabBarTab, option)
            
        # Draw add button
        add_button_rect = QRectF(
            self.width() - 30,
            4,
            24,
            24
        )
        self._add_button.setGeometry(add_button_rect.toRect())
        
    def tabSizeHint(self, index):
        size = super().tabSizeHint(index)
        size.setWidth(min(size.width(), 150))
        return size
        
    def _show_context_menu(self, pos):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3d3d3d;
            }
            QMenu::item {
                padding: 5px 20px 5px 20px;
            }
            QMenu::item:selected {
                background-color: #3d3d3d;
            }
        """)
        
        add_action = menu.addAction("Add Tab")
        add_action.setIcon(self.style().standardIcon(QStyle.SP_FileDialogNewFolder))
        add_action.triggered.connect(self.add_tab_requested)
        
        menu.exec_(self.mapToGlobal(pos))

class ModernButton(QPushButton):
    """Modern styled button with hover effects."""
    
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
            QPushButton:pressed {
                background-color: #2d2d2d;
            }
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #666666;
            }
        """)

class ModernLineEdit(QLineEdit):
    """Modern styled line edit with focus effects."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QLineEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 8px;
                selection-background-color: #4d4d4d;
            }
            QLineEdit:focus {
                border: 1px solid #4d9eff;
            }
            QLineEdit:disabled {
                background-color: #1d1d1d;
                color: #666666;
            }
        """)

class ModernComboBox(QComboBox):
    """Modern styled combo box with custom dropdown."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QComboBox {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 8px;
                min-width: 100px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: url(resources/icons/down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                color: #ffffff;
                selection-background-color: #4d4d4d;
                border: 1px solid #3d3d3d;
            }
        """)

class ModernProgressBar(QProgressBar):
    """Modern styled progress bar with animation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QProgressBar {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                text-align: center;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #4d9eff;
                border-radius: 4px;
            }
        """)
        self.setTextVisible(False)

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

class BackendStateManager:
    """Manages backend state and updates."""
    
    def __init__(self):
        self._state = {
            'autowiki': {},
            'neural_seed': {},
            'spiderweb': {},
            'metrics': {},
            'connections': {},
            'errors': []
        }
        self._listeners = []
        self._error_handlers = []
        
    def add_listener(self, listener):
        """Add a state change listener."""
        self._listeners.append(listener)
        
    def add_error_handler(self, handler):
        """Add an error handler."""
        self._error_handlers.append(handler)
        
    def update_state(self, new_state):
        """Update backend state and notify listeners."""
        try:
            # Update state
            for key, value in new_state.items():
                if key in self._state:
                    self._state[key] = value
                    
            # Notify listeners
            for listener in self._listeners:
                listener(self._state)
                
        except Exception as e:
            self._handle_error(f"Error updating state: {str(e)}")
            
    def _handle_error(self, error_message):
        """Handle errors and notify error handlers."""
        self._state['errors'].append(error_message)
        for handler in self._error_handlers:
            handler(error_message)
            
    def get_state(self):
        """Get current backend state."""
        return self._state.copy()

class DockablePanel(QWidget):
    """Base class for dockable panels."""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._is_docked = True
        self._original_parent = None
        self._window = None
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create toolbar
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(16, 16))
        
        # Add dock/undock action
        self._dock_action = QAction("Dock", self)
        self._dock_action.setIcon(self.style().standardIcon(QStyle.SP_TitleBarNormalButton))
        self._dock_action.triggered.connect(self._toggle_dock)
        
        toolbar.addAction(self._dock_action)
        layout.addWidget(toolbar)
        
        # Create content widget
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self._content)
        
    def _toggle_dock(self):
        """Toggle between docked and undocked states."""
        if self._is_docked:
            self._undock()
        else:
            self._dock()
            
    def _undock(self):
        """Undock the panel into a separate window."""
        try:
            # Store original parent
            self._original_parent = self.parent()
            
            # Create new window
            self._window = QMainWindow()
            self._window.setWindowTitle(self._title)
            self._window.setCentralWidget(self)
            
            # Update action
            self._dock_action.setText("Dock")
            self._dock_action.setIcon(self.style().standardIcon(QStyle.SP_TitleBarNormalButton))
            
            # Show window
            self._window.show()
            self._is_docked = False
            
        except Exception as e:
            logger.error(f"Error undocking panel: {e}")
            
    def _dock(self):
        """Dock the panel back into its original position."""
        try:
            if self._window:
                # Return to original parent
                self.setParent(self._original_parent)
                
                # Update action
                self._dock_action.setText("Undock")
                self._dock_action.setIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
                
                # Close window
                self._window.close()
                self._window = None
                self._is_docked = True
                
        except Exception as e:
            logger.error(f"Error docking panel: {e}")
            
    def get_content_layout(self) -> QVBoxLayout:
        """Get the content layout."""
        return self._content_layout

class AutoWikiPanel(DockablePanel):
    """Panel for visualizing AutoWiki system state."""
    
    state_changed = Signal(str, dict)  # panel_name, state
    
    def __init__(self, parent=None):
        super().__init__("AutoWiki Panel", parent)
        self._setup_content()
        
    def _setup_content(self):
        """Set up the panel content."""
        layout = self.get_content_layout()
        
        # Article status
        article_frame = QFrame()
        article_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        article_layout = QVBoxLayout(article_frame)
        
        article_label = QLabel("Articles")
        article_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        self._article_count = QLabel("0")
        self._article_count.setStyleSheet("color: #4d9eff; font-size: 24px;")
        
        article_layout.addWidget(article_label)
        article_layout.addWidget(self._article_count)
        layout.addWidget(article_frame)
        
        # Suggestions status
        suggestion_frame = QFrame()
        suggestion_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        suggestion_layout = QVBoxLayout(suggestion_frame)
        
        suggestion_label = QLabel("Suggestions")
        suggestion_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        self._suggestion_count = QLabel("0")
        self._suggestion_count.setStyleSheet("color: #4dff4d; font-size: 24px;")
        
        suggestion_layout.addWidget(suggestion_label)
        suggestion_layout.addWidget(self._suggestion_count)
        layout.addWidget(suggestion_frame)
        
        # Learning progress
        learning_frame = QFrame()
        learning_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        learning_layout = QVBoxLayout(learning_frame)
        
        learning_label = QLabel("Learning Progress")
        learning_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        self._learning_progress = ModernProgressBar()
        
        learning_layout.addWidget(learning_label)
        learning_layout.addWidget(self._learning_progress)
        layout.addWidget(learning_frame)
        
    def update_state(self, state):
        try:
            # Update UI
            self._article_count.setText(str(state.get('articles_count', 0)))
            self._suggestion_count.setText(str(state.get('suggestions_count', 0)))
            self._learning_progress.setValue(int(state.get('learning_progress', 0) * 100))
            
            # Emit state change
            self.state_changed.emit('autowiki', state)
            
        except Exception as e:
            logger.error(f"Error updating AutoWiki panel: {e}")

class NeuralSeedPanel(DockablePanel):
    """Panel for visualizing Neural Seed growth and state."""
    
    state_changed = Signal(str, dict)  # panel_name, state
    
    def __init__(self, parent=None):
        super().__init__("Neural Seed Panel", parent)
        self._setup_content()
        
    def _setup_content(self):
        """Set up the panel content."""
        layout = self.get_content_layout()
        
        # Growth stage
        stage_frame = QFrame()
        stage_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        stage_layout = QVBoxLayout(stage_frame)
        
        stage_label = QLabel("Growth Stage")
        stage_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        self._stage_indicator = QLabel("Seed")
        self._stage_indicator.setStyleSheet("color: #f1c40f; font-size: 24px;")
        
        stage_layout.addWidget(stage_label)
        stage_layout.addWidget(self._stage_indicator)
        layout.addWidget(stage_frame)
        
        # Consciousness level
        consciousness_frame = QFrame()
        consciousness_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        consciousness_layout = QVBoxLayout(consciousness_frame)
        
        consciousness_label = QLabel("Consciousness Level")
        consciousness_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        self._consciousness_progress = ModernProgressBar()
        
        consciousness_layout.addWidget(consciousness_label)
        consciousness_layout.addWidget(self._consciousness_progress)
        layout.addWidget(consciousness_frame)
        
        # Stability
        stability_frame = QFrame()
        stability_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        stability_layout = QVBoxLayout(stability_frame)
        
        stability_label = QLabel("Stability")
        stability_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        self._stability_progress = ModernProgressBar()
        
        stability_layout.addWidget(stability_label)
        stability_layout.addWidget(self._stability_progress)
        layout.addWidget(stability_frame)
        
    def update_state(self, state):
        try:
            # Update UI
            stage = state.get('stage', 'Seed')
            self._stage_indicator.setText(stage)
            
            stage_colors = {
                'Seed': '#f1c40f',
                'Sprout': '#2ecc71',
                'Sapling': '#3498db',
                'Mature': '#9b59b6'
            }
            self._stage_indicator.setStyleSheet(f"color: {stage_colors.get(stage, '#f1c40f')}; font-size: 24px;")
            
            consciousness = state.get('consciousness_level', 0.0)
            self._consciousness_progress.setValue(int(consciousness * 100))
            
            stability = state.get('stability', 0.0)
            self._stability_progress.setValue(int(stability * 100))
            
            # Emit state change
            self.state_changed.emit('neural_seed', state)
            
        except Exception as e:
            logger.error(f"Error updating Neural Seed panel: {e}")

class SpiderwebPanel(DockablePanel):
    """Panel for visualizing Spiderweb Bridge system."""
    
    state_changed = Signal(str, dict)  # panel_name, state
    
    def __init__(self, parent=None):
        super().__init__("Spiderweb Panel", parent)
        self._setup_content()
        
    def _setup_content(self):
        """Set up the panel content."""
        layout = self.get_content_layout()
        
        # Version connections
        version_frame = QFrame()
        version_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        version_layout = QVBoxLayout(version_frame)
        
        version_label = QLabel("Version Connections")
        version_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        self._version_list = QListWidget()
        self._version_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                border: none;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #3d3d3d;
            }
        """)
        
        version_layout.addWidget(version_label)
        version_layout.addWidget(self._version_list)
        layout.addWidget(version_frame)
        
        # Quantum field
        quantum_frame = QFrame()
        quantum_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        quantum_layout = QVBoxLayout(quantum_frame)
        
        quantum_label = QLabel("Quantum Field")
        quantum_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        self._quantum_strength = ModernProgressBar()
        
        quantum_layout.addWidget(quantum_label)
        quantum_layout.addWidget(self._quantum_strength)
        layout.addWidget(quantum_frame)
        
        # Cosmic field
        cosmic_frame = QFrame()
        cosmic_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        cosmic_layout = QVBoxLayout(cosmic_frame)
        
        cosmic_label = QLabel("Cosmic Field")
        cosmic_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        self._cosmic_strength = ModernProgressBar()
        
        cosmic_layout.addWidget(cosmic_label)
        cosmic_layout.addWidget(self._cosmic_strength)
        layout.addWidget(cosmic_frame)
        
    def update_state(self, state):
        try:
            # Update UI
            self._version_list.clear()
            for version in state.get('versions', []):
                item = QListWidgetItem(f"Version {version}")
                self._version_list.addItem(item)
                
            quantum_strength = state.get('quantum_field_strength', 0.0)
            self._quantum_strength.setValue(int(quantum_strength * 100))
            
            cosmic_strength = state.get('cosmic_field_strength', 0.0)
            self._cosmic_strength.setValue(int(cosmic_strength * 100))
            
            # Emit state change
            self.state_changed.emit('spiderweb', state)
            
        except Exception as e:
            logger.error(f"Error updating Spiderweb panel: {e}")

class SystemMetricsPanel(DockablePanel):
    """Panel for displaying system metrics."""
    
    state_changed = Signal(str, dict)  # panel_name, state
    
    def __init__(self, parent=None):
        super().__init__("System Metrics Panel", parent)
        self._setup_content()
        
    def _setup_content(self):
        """Set up the panel content."""
        layout = self.get_content_layout()
        
        # CPU usage
        cpu_frame = QFrame()
        cpu_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        cpu_layout = QVBoxLayout(cpu_frame)
        
        cpu_label = QLabel("CPU Usage")
        cpu_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        self._cpu_usage = ModernProgressBar()
        
        cpu_layout.addWidget(cpu_label)
        cpu_layout.addWidget(self._cpu_usage)
        layout.addWidget(cpu_frame)
        
        # Memory usage
        memory_frame = QFrame()
        memory_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        memory_layout = QVBoxLayout(memory_frame)
        
        memory_label = QLabel("Memory Usage")
        memory_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        self._memory_usage = ModernProgressBar()
        
        memory_layout.addWidget(memory_label)
        memory_layout.addWidget(self._memory_usage)
        layout.addWidget(memory_frame)
        
        # Process count
        process_frame = QFrame()
        process_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        process_layout = QVBoxLayout(process_frame)
        
        process_label = QLabel("Process Count")
        process_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        self._process_count = QLabel("0")
        self._process_count.setStyleSheet("color: #ffffff; font-size: 24px;")
        
        process_layout.addWidget(process_label)
        process_layout.addWidget(self._process_count)
        layout.addWidget(process_frame)
        
    def update_state(self, state):
        try:
            # Update UI
            cpu_usage = state.get('cpu_usage', 0.0)
            self._cpu_usage.setValue(int(cpu_usage * 100))
            
            memory_usage = state.get('memory_usage', 0.0)
            self._memory_usage.setValue(int(memory_usage * 100))
            
            process_count = state.get('process_count', 0)
            self._process_count.setText(str(process_count))
            
            # Emit state change
            self.state_changed.emit('metrics', state)
            
        except Exception as e:
            logger.error(f"Error updating System Metrics panel: {e}")

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
        
        # Initialize state manager
        self._state_manager = BackendStateManager()
        self._state_manager.add_listener(self._handle_state_update)
        self._state_manager.add_error_handler(self._handle_error)
        
        # Initialize components
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
        
        # Initialize visualization panels
        self._initialize_panels()
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
        self._apply_styles()
        
        # Start state checking
        self._state_check_timer = QTimer()
        self._state_check_timer.timeout.connect(self._check_backend_state)
        self._state_check_timer.start(100)  # Check every 100ms
        
    def _initialize_panels(self):
        """Initialize visualization panels."""
        self._autowiki_panel = AutoWikiPanel(self)
        self._neural_seed_panel = NeuralSeedPanel(self)
        self._spiderweb_panel = SpiderwebPanel(self)
        self._system_metrics_panel = SystemMetricsPanel(self)
        
        # Connect panel signals
        self._autowiki_panel.state_changed.connect(self._handle_panel_state_change)
        self._neural_seed_panel.state_changed.connect(self._handle_panel_state_change)
        self._spiderweb_panel.state_changed.connect(self._handle_panel_state_change)
        self._system_metrics_panel.state_changed.connect(self._handle_panel_state_change)
        
    def _handle_state_update(self, state):
        """Handle backend state updates."""
        try:
            # Update panels
            self._autowiki_panel.update_state(state['autowiki'])
            self._neural_seed_panel.update_state(state['neural_seed'])
            self._spiderweb_panel.update_state(state['spiderweb'])
            self._system_metrics_panel.update_state(state['metrics'])
            
            # Update connection status
            self._update_connection_status(state['connections'])
            
            # Handle errors
            if state['errors']:
                self._handle_errors(state['errors'])
                
        except Exception as e:
            logger.error(f"Error handling state update: {e}")
            self._show_error(f"Error handling state update: {str(e)}")
            
    def _handle_panel_state_change(self, panel_name, state):
        """Handle panel state changes."""
        try:
            # Update backend state through signal bus
            self._signal_bus.send_panel_state(panel_name, state)
            
        except Exception as e:
            logger.error(f"Error handling panel state change: {e}")
            self._show_error(f"Error handling panel state change: {str(e)}")
            
    def _update_connection_status(self, connections):
        """Update connection status indicators."""
        try:
            current_tab_data = self._get_current_tab_data()
            if current_tab_data:
                status_text = "Connected" if connections.get('active', False) else "Disconnected"
                status_color = "#4dff4d" if connections.get('active', False) else "#ff4d4d"
                
                current_tab_data['backend_status'].setText(f"Backend: {status_text}")
                current_tab_data['backend_status'].setStyleSheet(f"color: {status_color}; font-weight: bold;")
                
                # Update progress bar
                current_tab_data['backend_progress'].setValue(
                    int(connections.get('progress', 0) * 100)
                )
                
        except Exception as e:
            logger.error(f"Error updating connection status: {e}")
            self._show_error(f"Error updating connection status: {str(e)}")
            
    def _handle_errors(self, errors):
        """Handle backend errors."""
        try:
            for error in errors:
                self._add_system_message(f"Error: {error}", QColor("#ff4d4d"))
                logger.error(f"Backend error: {error}")
                
        except Exception as e:
            logger.error(f"Error handling errors: {e}")
            self._show_error(f"Error handling errors: {str(e)}")
            
    def _check_backend_state(self):
        """Check backend state and update UI."""
        try:
            # Get backend state from signal bus
            backend_state = self._signal_bus.get_backend_state()
            
            # Update state manager
            self._state_manager.update_state(backend_state)
            
            # Check for pending requests
            requests = backend_state.get('requests', [])
            for request in requests:
                if request not in self._pending_requests:
                    self._pending_requests.append(request)
                    self._handle_backend_request(request)
                    
        except Exception as e:
            logger.error(f"Error checking backend state: {e}")
            self._show_error(f"Error checking backend state: {str(e)}")
            
    def _handle_backend_request(self, request):
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
            
    def closeEvent(self, event):
        """Handle window close event."""
        try:
            # Clean up signal system
            self._signal_bus.unregister_component(self._current_version)
            self._component.cleanup()
            
            # Stop timers
            self._typing_timer.stop()
            self._state_check_timer.stop()
            
            # Stop animations
            self._state_change_animation.stop()
            for gate in self._logic_gates.values():
                gate._pulse_animation.stop()
                
            # Close any undocked panels
            for panel in [self._autowiki_panel, self._neural_seed_panel, 
                         self._spiderweb_panel, self._system_metrics_panel]:
                if panel._window:
                    panel._window.close()
                    
            # Accept close event
            event.accept()
            
        except Exception as e:
            logger.error(f"Error during close: {e}")
            event.accept()

    def _apply_styles(self):
        """Apply global styles to the window."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QToolBar {
                background-color: #2d2d2d;
                border: none;
                border-bottom: 1px solid #3d3d3d;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                padding: 5px;
            }
            QToolButton:hover {
                background-color: #3d3d3d;
            }
            QToolButton:pressed {
                background-color: #4d4d4d;
            }
            QLabel {
                color: #ffffff;
            }
        """)
        
    def _create_toolbar(self):
        """Create the main toolbar."""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(16, 16))
        
        # Add actions
        new_tab_action = QAction("New Tab", self)
        new_tab_action.setShortcut(QKeySequence("Ctrl+T"))
        new_tab_action.triggered.connect(self._handle_new_tab_request)
        
        clear_chat_action = QAction("Clear Chat", self)
        clear_chat_action.setShortcut(QKeySequence("Ctrl+L"))
        clear_chat_action.triggered.connect(self._clear_chat)
        
        save_chat_action = QAction("Save Chat", self)
        save_chat_action.setShortcut(QKeySequence("Ctrl+S"))
        save_chat_action.triggered.connect(self._save_chat)
        
        # Add actions to toolbar
        toolbar.addAction(new_tab_action)
        toolbar.addSeparator()
        toolbar.addAction(clear_chat_action)
        toolbar.addAction(save_chat_action)
        
        self.addToolBar(toolbar)
        
    def _clear_chat(self):
        """Clear chat history."""
        try:
            current_tab_data = self._get_current_tab_data()
            if current_tab_data:
                current_tab_data['chat_display'].clear()
                self._message_history.clear()
                self._add_system_message("System: Chat history cleared")
                
        except Exception as e:
            logger.error(f"Error clearing chat: {e}")
            self._show_error(f"Error clearing chat: {str(e)}")
            
    def _save_chat(self):
        """Save chat history to file."""
        try:
            # TODO: Implement chat saving functionality
            self._add_system_message("System: Chat saving not implemented yet")
            
        except Exception as e:
            logger.error(f"Error saving chat: {e}")
            self._show_error(f"Error saving chat: {str(e)}")
            
    def _add_system_message(self, message: str, color: QColor = QColor("#666666")):
        """Add system message to chat."""
        try:
            current_tab_data = self._get_current_tab_data()
            if current_tab_data:
                format = QTextCharFormat()
                format.setForeground(color)
                format.setFontItalic(True)
                
                cursor = current_tab_data['chat_display'].textCursor()
                cursor.movePosition(QTextCursor.End)
                cursor.insertText(message + "\n", format)
                
        except Exception as e:
            logger.error(f"Error adding system message: {e}")
            self._show_error(f"Error adding system message: {str(e)}")
            
    def _update_status(self, message: str):
        """Update status bar message."""
        try:
            self._status_bar.showMessage(message)
            
        except Exception as e:
            logger.error(f"Error updating status: {e}")
            self._show_error(f"Error updating status: {str(e)}")
            
    def _handle_typing(self):
        """Handle typing indicator."""
        try:
            current_tab_data = self._get_current_tab_data()
            if current_tab_data:
                if not self._typing_indicator:
                    self._typing_indicator = True
                    self._typing_timer.start(1000)  # Update every second
                    
        except Exception as e:
            logger.error(f"Error handling typing: {e}")
            self._show_error(f"Error handling typing: {str(e)}")
            
    def _update_typing_indicator(self):
        """Update typing indicator."""
        try:
            current_tab_data = self._get_current_tab_data()
            if current_tab_data:
                if not current_tab_data['message_input'].text():
                    self._typing_indicator = False
                    self._typing_timer.stop()
                    current_tab_data['typing_label'].setText("")
                else:
                    current_tab_data['typing_label'].setText("Typing...")
                    
        except Exception as e:
            logger.error(f"Error updating typing indicator: {e}")
            self._show_error(f"Error updating typing indicator: {str(e)}")
            
    def _handle_version_change(self, version: str):
        """Handle version change."""
        try:
            # Unregister old component
            self._signal_bus.unregister_component(self._current_version)
            
            # Create new component
            self._current_version = version
            self._component = SignalComponent(version)
            
            # Register new component
            self._signal_bus.register_component(self._component)
            
            # Update status
            self._update_status(f"Switched to {version}")
            
            # Add system message
            self._add_system_message(f"System: Switched to version {version}")
            
            # Emit signal
            self.version_changed.emit(version)
            
        except Exception as e:
            logger.error(f"Error changing version: {e}")
            self._show_error(f"Error changing version: {str(e)}")
            
    def _handle_backend_state_change(self, active: bool):
        """Handle backend state change with animation."""
        try:
            # Emit signal
            self.backend_state_changed.emit(active)
            
            # Animate panel visibility
            current_tab_data = self._get_current_tab_data()
            if current_tab_data:
                if active:
                    current_tab_data['gate_panel'].setVisible(True)
                    self._state_change_animation.setStartValue(0)
                    self._state_change_animation.setEndValue(400)
                else:
                    self._state_change_animation.setStartValue(400)
                    self._state_change_animation.setEndValue(0)
                    self._state_change_animation.finished.connect(
                        lambda: current_tab_data['gate_panel'].setVisible(False)
                    )
                    
                self._state_change_animation.start()
                
        except Exception as e:
            logger.error(f"Error handling backend state change: {e}")
            self._show_error(f"Error handling backend state change: {str(e)}")
            
    def _update_backend_status(self, status_label: QLabel, progress_bar: QProgressBar, backend_state: Dict[str, Any]):
        """Update backend status indicator."""
        try:
            if self._backend_active:
                status_label.setText("Backend: Active")
                status_label.setStyleSheet("color: #4dff4d; font-weight: bold;")
                
                # Update progress bar
                progress = backend_state.get('progress', 0)
                progress_bar.setValue(int(progress * 100))
            else:
                status_label.setText("Backend: Inactive")
                status_label.setStyleSheet("color: #ff4d4d; font-weight: bold;")
                progress_bar.setValue(0)
                
        except Exception as e:
            logger.error(f"Error updating backend status: {e}")
            self._show_error(f"Error updating backend status: {str(e)}")
            
    def _handle_message(self, message: str, version: str):
        """Handle incoming message."""
        try:
            current_tab_data = self._get_current_tab_data()
            if current_tab_data:
                # Format message with timestamp and version
                timestamp = datetime.now().strftime("%H:%M:%S")
                formatted_message = f"[{timestamp}] [{version}] {message}"
                
                # Create message format
                format = QTextCharFormat()
                format.setForeground(QColor("#ffffff"))
                
                # Add version-specific color
                if version == 'v7.5':
                    format.setForeground(QColor("#4d9eff"))
                elif version == 'v7.0':
                    format.setForeground(QColor("#ff4d4d"))
                elif version == 'v6.0':
                    format.setForeground(QColor("#4dff4d"))
                else:
                    format.setForeground(QColor("#ffff4d"))
                
                # Append to chat display
                cursor = current_tab_data['chat_display'].textCursor()
                cursor.movePosition(QTextCursor.End)
                cursor.insertText(formatted_message + "\n", format)
                
                # Scroll to bottom
                current_tab_data['chat_display'].verticalScrollBar().setValue(
                    current_tab_data['chat_display'].verticalScrollBar().maximum()
                )
                
                # Add to history
                self._message_history.append({
                    'timestamp': timestamp,
                    'version': version,
                    'message': message,
                    'received': True
                })
                
                # Update gate states if backend is active
                if self._backend_active:
                    self._update_gate_states(message)
                    
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            self._show_error(f"Error handling message: {str(e)}")
            
    def _update_gate_states(self, message: str):
        """Update gate states based on incoming message."""
        try:
            # Implement logic to update gate states based on the incoming message
            # This is a placeholder and should be replaced with the actual implementation
            pass
            
        except Exception as e:
            logger.error(f"Error updating gate states: {e}")
            self._show_error(f"Error updating gate states: {str(e)}")
            
    def _update_gate_states_from_backend(self, backend_state: Dict[str, Any]):
        """Update gate states based on backend state."""
        try:
            # Implement logic to update gate states based on the backend state
            # This is a placeholder and should be replaced with the actual implementation
            pass
            
        except Exception as e:
            logger.error(f"Error updating gate states from backend: {e}")
            self._show_error(f"Error updating gate states from backend: {str(e)}") 