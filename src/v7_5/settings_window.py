#!/usr/bin/env python3
"""
LUMINA v7.5 Advanced Settings Window
Separate window for detailed settings and visualizations
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QGroupBox, QGridLayout, QSlider, QSpinBox,
                             QDoubleSpinBox, QCheckBox, QComboBox, QPushButton,
                             QFrame, QSplitter, QTabWidget, QScrollArea, QTextEdit,
                             QDialog, QLineEdit)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QColor, QPalette, QPainter, QTextCursor
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis, QPieSeries

class MetricsPanel(QWidget):
    """Panel for displaying performance metrics and charts"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        # Response Time Chart
        response_group = QGroupBox("Response Time History")
        response_layout = QVBoxLayout()
        
        self.response_chart = QChart()
        self.response_series = QLineSeries()
        self.response_chart.addSeries(self.response_series)
        
        # Configure axes
        self.axis_x = QValueAxis()
        self.axis_x.setTitleText("Messages")
        self.axis_x.setRange(0, 20)  # Show last 20 messages
        self.response_chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.response_series.attachAxis(self.axis_x)
        
        self.axis_y = QValueAxis()
        self.axis_y.setTitleText("Time (ms)")
        self.axis_y.setRange(0, 5000)
        self.response_chart.addAxis(self.axis_y, Qt.AlignLeft)
        self.response_series.attachAxis(self.axis_y)
        
        # Store points for managing the series
        self.data_points = []
        self.max_points = 20
        
        chart_view = QChartView(self.response_chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        response_layout.addWidget(chart_view)
        response_group.setLayout(response_layout)
        layout.addWidget(response_group)
        
        # AutoWiki Information
        wiki_group = QGroupBox("AutoWiki Status")
        wiki_layout = QVBoxLayout()
        
        # Status indicators
        status_layout = QGridLayout()
        
        # Processing Status
        status_layout.addWidget(QLabel("Processing Status:"), 0, 0)
        self.wiki_status_label = QLabel("Idle")
        self.wiki_status_label.setStyleSheet("color: gray;")
        status_layout.addWidget(self.wiki_status_label, 0, 1)
        
        # Active Tasks
        status_layout.addWidget(QLabel("Active Tasks:"), 1, 0)
        self.active_tasks_label = QLabel("0")
        status_layout.addWidget(self.active_tasks_label, 1, 1)
        
        # Success Rate
        status_layout.addWidget(QLabel("Success Rate:"), 2, 0)
        self.success_rate_label = QLabel("0%")
        status_layout.addWidget(self.success_rate_label, 2, 1)
        
        wiki_layout.addLayout(status_layout)
        
        # Recent Wiki Updates
        wiki_updates_label = QLabel("Recent Wiki Updates:")
        wiki_layout.addWidget(wiki_updates_label)
        
        self.wiki_updates = QTextEdit()
        self.wiki_updates.setReadOnly(True)
        self.wiki_updates.setMaximumHeight(200)
        self.wiki_updates.setFont(QFont("Consolas", 9))
        wiki_layout.addWidget(self.wiki_updates)
        
        # AutoWiki Settings
        settings_layout = QGridLayout()
        
        # Update Frequency
        settings_layout.addWidget(QLabel("Update Frequency:"), 0, 0)
        self.update_freq = QSpinBox()
        self.update_freq.setRange(5, 300)  # 5 seconds to 5 minutes
        self.update_freq.setValue(30)  # Default 30 seconds
        self.update_freq.setSuffix(" sec")
        settings_layout.addWidget(self.update_freq, 0, 1)
        
        # Max Concurrent Tasks
        settings_layout.addWidget(QLabel("Max Concurrent Tasks:"), 1, 0)
        self.max_tasks = QSpinBox()
        self.max_tasks.setRange(1, 10)
        self.max_tasks.setValue(3)
        settings_layout.addWidget(self.max_tasks, 1, 1)
        
        # Auto-update Toggle
        self.auto_update = QCheckBox("Enable Auto-updates")
        self.auto_update.setChecked(True)
        settings_layout.addWidget(self.auto_update, 2, 0, 1, 2)
        
        wiki_layout.addLayout(settings_layout)
        wiki_group.setLayout(wiki_layout)
        layout.addWidget(wiki_group)
        
        # Add metrics display
        metrics_layout = QGridLayout()
        metrics_layout.addWidget(QLabel("Average Response Time:"), 0, 0)
        self.avg_time_label = QLabel("0 ms")
        metrics_layout.addWidget(self.avg_time_label, 0, 1)
        
        metrics_layout.addWidget(QLabel("Total Messages:"), 1, 0)
        self.total_messages_label = QLabel("0")
        metrics_layout.addWidget(self.total_messages_label, 1, 1)
        
        layout.addLayout(metrics_layout)
    
    def update_response_time(self, time_ms: float):
        """Update response time chart with new data point"""
        # Add new point to our data
        self.data_points.append(time_ms)
        if len(self.data_points) > self.max_points:
            self.data_points.pop(0)
        
        # Clear and rebuild series
        self.response_series.clear()
        for i, point in enumerate(self.data_points):
            self.response_series.append(i, point)
        
        # Update axis range if needed
        max_time = max(self.data_points) if self.data_points else 5000
        self.axis_y.setRange(0, max_time * 1.1)  # Add 10% padding
        self.axis_x.setRange(0, self.max_points)
        
        # Update average
        avg_time = sum(self.data_points) / len(self.data_points) if self.data_points else 0
        self.avg_time_label.setText(f"{avg_time:.1f} ms")
        self.total_messages_label.setText(str(len(self.data_points)))
    
    def update_wiki_status(self, status: str, active_tasks: int, success_rate: float):
        """Update AutoWiki status information"""
        # Update status with color coding
        self.wiki_status_label.setText(status)
        if status.lower() == "idle":
            self.wiki_status_label.setStyleSheet("color: gray;")
        elif status.lower() == "processing":
            self.wiki_status_label.setStyleSheet("color: orange;")
        elif status.lower() == "complete":
            self.wiki_status_label.setStyleSheet("color: green;")
        else:
            self.wiki_status_label.setStyleSheet("color: red;")
        
        # Update other metrics
        self.active_tasks_label.setText(str(active_tasks))
        self.success_rate_label.setText(f"{success_rate:.1f}%")
    
    def add_wiki_update(self, wiki_data: str):
        """Add a new wiki update to the display"""
        self.wiki_updates.append(f"[{datetime.now().strftime('%H:%M:%S')}]\n{wiki_data}\n")
        # Scroll to bottom
        cursor = self.wiki_updates.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.wiki_updates.setTextCursor(cursor)

class ModelPanel(QWidget):
    """Panel for model settings"""
    settings_changed = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QGridLayout(self)
        
        # Model Selection
        layout.addWidget(QLabel("Model:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["mistral-medium", "mistral-small", "mistral-large"])
        self.model_combo.currentTextChanged.connect(self._emit_settings)
        layout.addWidget(self.model_combo, 0, 1)
        
        # Context Window
        layout.addWidget(QLabel("Context Window:"), 1, 0)
        self.context_spin = QSpinBox()
        self.context_spin.setRange(512, 8192)
        self.context_spin.setValue(2048)
        self.context_spin.setSingleStep(512)
        self.context_spin.valueChanged.connect(self._emit_settings)
        layout.addWidget(self.context_spin, 1, 1)
        
        # Temperature
        layout.addWidget(QLabel("Temperature:"), 2, 0)
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setValue(0.7)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.valueChanged.connect(self._emit_settings)
        layout.addWidget(self.temp_spin, 2, 1)
        
        # Top P
        layout.addWidget(QLabel("Top P:"), 3, 0)
        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setValue(0.9)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.valueChanged.connect(self._emit_settings)
        layout.addWidget(self.top_p_spin, 3, 1)
        
        # Top K
        layout.addWidget(QLabel("Top K:"), 4, 0)
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 100)
        self.top_k_spin.setValue(50)
        self.top_k_spin.setSingleStep(5)
        self.top_k_spin.valueChanged.connect(self._emit_settings)
        layout.addWidget(self.top_k_spin, 4, 1)
    
    def _emit_settings(self):
        """Emit current model settings"""
        settings = {
            'type': 'model',
            'model': self.model_combo.currentText(),
            'context_window': self.context_spin.value(),
            'temperature': self.temp_spin.value(),
            'top_p': self.top_p_spin.value(),
            'top_k': self.top_k_spin.value()
        }
        self.settings_changed.emit(settings)

class ProcessingPanel(QWidget):
    """Panel for processing settings"""
    
    settings_changed = Signal(dict)  # Single signal for all processing settings
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        # Processing Options
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout()
        
        self.parallel_check = QCheckBox("Enable Parallel Processing")
        self.parallel_check.setChecked(True)
        self.parallel_check.toggled.connect(self._emit_settings)
        options_layout.addWidget(self.parallel_check)
        
        self.cache_check = QCheckBox("Enable Response Caching")
        self.cache_check.setChecked(True)
        self.cache_check.toggled.connect(self._emit_settings)
        options_layout.addWidget(self.cache_check)
        
        self.fallback_check = QCheckBox("Enable Fallback Responses")
        self.fallback_check.setChecked(True)
        self.fallback_check.toggled.connect(self._emit_settings)
        options_layout.addWidget(self.fallback_check)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Thread Settings
        thread_group = QGroupBox("Thread Settings")
        thread_layout = QGridLayout()
        
        thread_layout.addWidget(QLabel("Worker Threads:"), 0, 0)
        self.thread_spin = QSpinBox()
        self.thread_spin.setRange(1, 16)
        self.thread_spin.setValue(4)
        self.thread_spin.valueChanged.connect(self._emit_settings)
        thread_layout.addWidget(self.thread_spin, 0, 1)
        
        thread_group.setLayout(thread_layout)
        layout.addWidget(thread_group)
    
    def _emit_settings(self):
        """Emit current processing settings"""
        settings = {
            'type': 'processing',
            'parallel_processing': self.parallel_check.isChecked(),
            'response_caching': self.cache_check.isChecked(),
            'fallback_responses': self.fallback_check.isChecked(),
            'worker_threads': self.thread_spin.value()
        }
        self.settings_changed.emit(settings)

class SettingsWindow(QDialog, SignalComponent):
    settings_updated = Signal(dict)
    
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        SignalComponent.__init__(self, "settings_window", parent.signal_bus if parent else None)
        
        self.setWindowTitle("LUMINA v7.5 Settings")
        self.setMinimumSize(600, 800)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create panels
        self.model_panel = ModelPanel()
        self.processing_panel = ProcessingPanel()
        self.metrics_panel = MetricsPanel()
        
        # Add panels to tabs
        self.tab_widget.addTab(self.model_panel, "Model")
        self.tab_widget.addTab(self.processing_panel, "Processing")
        self.tab_widget.addTab(self.metrics_panel, "Metrics")
        
        # Create main layout
        layout = QVBoxLayout()
        layout.addWidget(self.tab_widget)
        
        # Add buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_settings)
        button_layout.addWidget(save_button)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # Connect panel signals
        self.model_panel.settings_changed.connect(self.settings_updated)
        self.processing_panel.settings_changed.connect(self.settings_updated)
        
        # Register signal handlers
        self.register_handler("settings_update", self.handle_settings_update)
        
    def handle_settings_update(self, data: dict):
        """Handle settings updates from other components"""
        if isinstance(data, dict):
            self.load_settings(data)
            
    def save_settings(self):
        """Save current settings and emit update signal"""
        settings = {
            "model": self.model_panel.model_combo.currentText(),
            "temperature": self.model_panel.temp_spin.value(),
            "top_p": self.model_panel.top_p_spin.value(),
            "auto_wiki": self.metrics_panel.auto_update.isChecked(),
            "update_freq": self.metrics_panel.update_freq.value(),
            "max_tasks": self.metrics_panel.max_tasks.value(),
            "parallel_processing": self.processing_panel.parallel_check.isChecked(),
            "response_caching": self.processing_panel.cache_check.isChecked(),
            "fallback_responses": self.processing_panel.fallback_check.isChecked(),
            "worker_threads": self.processing_panel.thread_spin.value()
        }
        self.settings_updated.emit(settings)
        self.accept()
        
    def load_settings(self, settings: dict):
        """Load settings from dictionary"""
        if not isinstance(settings, dict):
            return
            
        # Model settings
        if "model" in settings:
            self.model_panel.model_combo.setCurrentText(settings["model"])
        if "temperature" in settings:
            self.model_panel.temp_spin.setValue(settings["temperature"])
        if "top_p" in settings:
            self.model_panel.top_p_spin.setValue(settings["top_p"])
            
        # Metrics settings
        if "auto_wiki" in settings:
            self.metrics_panel.auto_update.setChecked(settings["auto_wiki"])
        if "update_freq" in settings:
            self.metrics_panel.update_freq.setValue(settings["update_freq"])
        if "max_tasks" in settings:
            self.metrics_panel.max_tasks.setValue(settings["max_tasks"])
            
        # Processing settings
        if "parallel_processing" in settings:
            self.processing_panel.parallel_check.setChecked(settings["parallel_processing"])
        if "response_caching" in settings:
            self.processing_panel.cache_check.setChecked(settings["response_caching"])
        if "fallback_responses" in settings:
            self.processing_panel.fallback_check.setChecked(settings["fallback_responses"])
        if "worker_threads" in settings:
            self.processing_panel.thread_spin.setValue(settings["worker_threads"])
            
    def closeEvent(self, event):
        """Handle window close event"""
        self.cleanup()
        event.accept()

    def display_update(self, message: str):
        """Display a new update message"""
        current_text = self.update_display.toPlainText()
        if current_text:
            current_text += "\n\n"
        current_text += message
        self.update_display.setText(current_text)
        # Scroll to bottom
        cursor = self.update_display.textCursor()
        cursor.movePosition(cursor.End)
        self.update_display.setTextCursor(cursor)
    
    def update_response_time(self, time_ms: float):
        """Update response time chart with new data point"""
        self.metrics_panel.update_response_time(time_ms)
    
    def update_wiki_status(self, status: str, active_tasks: int, success_rate: float):
        """Update AutoWiki status information"""
        self.metrics_panel.update_wiki_status(status, active_tasks, success_rate)
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.update_timer.stop()
        event.accept() 