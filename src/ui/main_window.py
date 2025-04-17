#!/usr/bin/env python3
"""
Main Window

This module implements the main application window with metrics visualization
and system state monitoring.
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QMessageBox, QProgressBar,
    QMenu, QAction, QFileDialog, QSplitter, QDockWidget,
    QStatusBar
)
from PySide6.QtCore import Qt, Slot, QTimer, QEvent, QSettings, Signal
from PySide6.QtGui import QIcon, QActionGroup

from .components.system_overview import SystemOverviewWidget
from .components.neural_seed import NeuralSeedWidget
from .components.signal_system import SignalSystemWidget
from .components.spiderweb import SpiderwebWidget
from .components.metrics import MetricsWidget
from .components.health import HealthWidget
from .components.network_2d_widget import Network2DWidget
from .components.growth_visualizer import GrowthVisualizer
from .components.system_state import SystemState
from .components.metrics_dashboard import MetricsDashboard
from .components.chat_window import ChatWindow
from .components.signal_monitor import SignalMonitor

from visualization.cpu_graph import CPUGraph
from visualization.memory_graph import MemoryGraph
from visualization.disk_io_graph import DiskIOGraph
from visualization.network_graph import NetworkGraph
from visualization.process_graph import ProcessGraph

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """Main application window with metrics and system state."""
    
    # Signals
    chat_requested = Signal()
    metrics_updated = Signal(dict)
    system_state_changed = Signal(dict)
    growth_stage_changed = Signal(str)
    
    def __init__(self, integrator):
        """Initialize the main window."""
        super().__init__()
        self.integrator = integrator
        self._initialized = False
        self._shutting_down = False
        self._config = self._load_config()
        self._performance_data = []
        self._max_performance_samples = 1000
        self._current_growth_stage = "SEED"
        
        self.setWindowTitle("Lumina Neural Network Monitor")
        self.setMinimumSize(1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create status bar
        self.status_label = QLabel()
        self.statusBar().addWidget(self.status_label)
        
        # Create progress bar for system initialization
        self.init_progress = QProgressBar()
        self.init_progress.setRange(0, 100)
        self.init_progress.setValue(0)
        self.statusBar().addPermanentWidget(self.init_progress)
        
        # Create main splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(self.main_splitter)
        
        # Create left panel for metrics
        self.metrics_panel = QWidget()
        metrics_layout = QVBoxLayout(self.metrics_panel)
        
        # Create metrics dashboard
        self.metrics_dashboard = MetricsDashboard()
        metrics_layout.addWidget(self.metrics_dashboard)
        
        # Create right panel for system state
        self.state_panel = QWidget()
        state_layout = QVBoxLayout(self.state_panel)
        
        # Create system state widget
        self.system_state = SystemState()
        self.system_state.set_config(self._config.get('data_sources', {}))
        state_layout.addWidget(self.system_state)
        
        # Create network visualization
        self.network_widget = Network2DWidget()
        self.network_widget.set_config(self._config.get('network', {}))
        metrics_layout.addWidget(self.network_widget)
        
        # Create growth visualizer
        self.growth_visualizer = GrowthVisualizer()
        self.growth_visualizer.set_config(self._config.get('appearance', {}))
        metrics_layout.addWidget(self.growth_visualizer)
        
        # Add panels to splitter
        self.main_splitter.addWidget(self.metrics_panel)
        self.main_splitter.addWidget(self.state_panel)
        
        # Set initial splitter sizes
        self.main_splitter.setSizes([800, 400])
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create dock widgets
        self._create_dock_widgets()
        
        # Initialize components
        self._initialize_components()
        
        # Set up update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_components)
        self.update_timer.start(1000)  # Update every second
        
        # Set up performance monitoring timer
        self.perf_timer = QTimer()
        self.perf_timer.timeout.connect(self._update_performance)
        self.perf_timer.start(5000)  # Update every 5 seconds
        
        # Set up growth monitoring timer
        self.growth_timer = QTimer()
        self.growth_timer.timeout.connect(self._update_growth)
        self.growth_timer.start(1000)  # Update every second
        
        # Set dark theme
        self._set_dark_theme()
        
        # Connect signals
        self.integrator.system_ready.connect(self._on_system_ready)
        self.integrator.state_changed.connect(self.update_state)
        self.integrator.error_occurred.connect(self.show_error)
        
        # Install event filter for cleanup
        self.installEventFilter(self)
        
        # Restore window state
        self._restore_window_state()
        
        # Create chat window (initially hidden)
        self.chat_window = ChatWindow()
        self.chat_window.hide()
        
        self._initialized = True
        
    def _create_dock_widgets(self):
        """Create dock widgets for additional controls."""
        # Create neural seed dock
        neural_seed_dock = QDockWidget("Neural Seed", self)
        neural_seed_dock.setWidget(NeuralSeedWidget(self.integrator))
        self.addDockWidget(Qt.LeftDockWidgetArea, neural_seed_dock)
        
        # Create signal system dock
        signal_system_dock = QDockWidget("Signal System", self)
        signal_system_dock.setWidget(SignalSystemWidget(self.integrator))
        self.addDockWidget(Qt.RightDockWidgetArea, signal_system_dock)
        
        # Create spiderweb dock
        spiderweb_dock = QDockWidget("Spiderweb", self)
        spiderweb_dock.setWidget(SpiderwebWidget(self.integrator))
        self.addDockWidget(Qt.RightDockWidgetArea, spiderweb_dock)
        
        # Create signal monitor dock
        signal_dock = QDockWidget("Signal Monitor", self)
        signal_dock.setWidget(SignalMonitor())
        self.addDockWidget(Qt.RightDockWidgetArea, signal_dock)
        
        # Create metrics history dock
        history_dock = QDockWidget("Metrics History", this)
        history_dock.setWidget(MetricsHistory())
        self.addDockWidget(Qt.LeftDockWidgetArea, history_dock)
        
    def _update_growth(self):
        """Update growth visualization."""
        try:
            # Get current growth stage
            stage = self.system_state.get_growth_stage()
            
            if stage != self._current_growth_stage:
                self._current_growth_stage = stage
                self.growth_stage_changed.emit(stage)
                
            # Update growth visualization
            self.growth_visualizer.update_growth(
                stage=stage,
                health=self.system_state.get_health_metrics(),
                stability=self.system_state.get_stability()
            )
            
            # Update network visualization
            self.network_widget.update_growth(
                stage=stage,
                nodes=self.system_state.get_active_nodes(),
                connections=self.system_state.get_active_connections()
            )
            
        except Exception as e:
            logger.error(f"Error updating growth: {e}")
            
    def _update_components(self):
        """Update all components with latest data."""
        if not self._initialized or self._shutting_down:
            return
            
        try:
            # Get latest system state
            state = self.integrator.get_system_state()
            metrics = self.integrator.get_metrics()
            health = self.integrator.get_health_status()
            
            # Update system state
            self.system_state.update_state(state, metrics, health)
            
            # Update metrics dashboard
            self.metrics_dashboard.update_metrics(metrics)
            
            # Update network visualization
            self.network_widget.update_data(
                nodes=state.get('nodes', []),
                connections=state.get('connections', []),
                signals=state.get('signals', [])
            )
            
            # Update growth visualization
            self.growth_visualizer.update_data(
                health=health,
                stability=metrics.get('stability', 0.0)
            )
            
            # Emit system state changed signal
            self.system_state_changed.emit(state)
            
        except Exception as e:
            logger.error(f"Error updating components: {e}")
            
    def _cleanup(self):
        """Clean up resources before shutdown."""
        if self._shutting_down:
            return
            
        self._shutting_down = True
        logger.info("Cleaning up main window resources...")
        
        try:
            # Stop timers
            for timer in [self.update_timer, self.perf_timer, self.growth_timer]:
                if hasattr(self, timer):
                    timer.stop()
                    timer.deleteLater()
                    
            # Clean up components
            for component in [
                self.network_widget,
                self.growth_visualizer,
                self.system_state,
                self.metrics_dashboard
            ]:
                if hasattr(component, 'cleanup'):
                    component.cleanup()
                    
            # Disconnect signals
            if hasattr(self, 'integrator'):
                self.integrator.system_ready.disconnect()
                self.integrator.state_changed.disconnect()
                self.integrator.error_occurred.disconnect()
                
            # Save window state
            settings = QSettings("Lumina", "NeuralNetworkMonitor")
            settings.setValue("geometry", self.saveGeometry())
            settings.setValue("windowState", self.saveState())
            settings.setValue("splitterState", self.main_splitter.saveState())
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
    def _create_menu_bar(self):
        """Create the application menu bar."""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("File")
        
        save_action = QAction("Save Configuration", self)
        save_action.triggered.connect(self._save_config)
        file_menu.addAction(save_action)
        
        load_action = QAction("Load Configuration", self)
        load_action.triggered.connect(self._load_config_dialog)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menu_bar.addMenu("View")
        
        # Theme submenu
        theme_menu = QMenu("Theme", self)
        theme_group = QActionGroup(self)
        
        dark_action = QAction("Dark", self)
        dark_action.setCheckable(True)
        dark_action.setChecked(True)
        dark_action.triggered.connect(lambda: self._set_theme("dark"))
        theme_group.addAction(dark_action)
        theme_menu.addAction(dark_action)
        
        light_action = QAction("Light", self)
        light_action.setCheckable(True)
        light_action.triggered.connect(lambda: self._set_theme("light"))
        theme_group.addAction(light_action)
        theme_menu.addAction(light_action)
        
        view_menu.addMenu(theme_menu)
        
        # Tools menu
        tools_menu = menu_bar.addMenu("Tools")
        metrics_action = QAction("Metrics Dashboard", this)
        metrics_action.triggered.connect(self._show_metrics)
        tools_menu.addAction(metrics_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        config_path = Path("config.json")
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        return {}
        
    def _save_config(self):
        """Save current configuration to file."""
        try:
            config = {
                'window_state': {
                    'geometry': self.saveGeometry(),
                    'state': self.saveState()
                },
                'components': {
                    name: component.get_config() 
                    for name, component in self.components.items()
                    if hasattr(component, 'get_config')
                }
            }
            
            with open("config.json", "w") as f:
                json.dump(config, f, indent=4)
                
            self.status_label.setText("Configuration saved")
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            self.show_error(f"Failed to save configuration: {str(e)}")
            
    def _load_config_dialog(self):
        """Show dialog to load configuration."""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Configuration",
                "",
                "JSON Files (*.json);;All Files (*)"
            )
            
            if file_name:
                with open(file_name, "r") as f:
                    config = json.load(f)
                    
                # Apply window state
                if 'window_state' in config:
                    self.restoreGeometry(config['window_state']['geometry'])
                    self.restoreState(config['window_state']['state'])
                    
                # Apply component configs
                if 'components' in config:
                    for name, component_config in config['components'].items():
                        if name in self.components and hasattr(self.components[name], 'set_config'):
                            self.components[name].set_config(component_config)
                            
                self.status_label.setText("Configuration loaded")
                
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.show_error(f"Failed to load configuration: {str(e)}")
            
    def _restore_window_state(self):
        """Restore window state from settings."""
        settings = QSettings("Lumina", "NeuralNetworkMonitor")
        self.restoreGeometry(settings.value("geometry"))
        self.restoreState(settings.value("windowState"))
        
    def _update_performance(self):
        """Update performance monitoring data."""
        try:
            # Get current performance metrics
            metrics = {
                'cpu': self.integrator.get_cpu_usage(),
                'memory': self.integrator.get_memory_usage(),
                'timestamp': QTimer.currentTime()
            }
            
            # Add to performance data
            self._performance_data.append(metrics)
            
            # Trim old data
            if len(self._performance_data) > self._max_performance_samples:
                self._performance_data = self._performance_data[-self._max_performance_samples:]
                
            # Update performance indicators
            self._update_performance_indicators()
            
        except Exception as e:
            logger.error(f"Error updating performance data: {e}")
            
    def _update_performance_indicators(self):
        """Update UI performance indicators."""
        try:
            if not self._performance_data:
                return
                
            # Calculate average CPU usage
            avg_cpu = sum(m['cpu'] for m in self._performance_data) / len(self._performance_data)
            
            # Update status bar
            self.status_label.setText(f"Average CPU Usage: {avg_cpu:.1f}%")
            
        except Exception as e:
            logger.error(f"Error updating performance indicators: {e}")
            
    def _set_theme(self, theme: str):
        """Set application theme."""
        try:
            if theme == "dark":
                self._set_dark_theme()
            else:
                self._set_light_theme()
                
            # Save theme preference
            settings = QSettings("Lumina", "NeuralNetworkMonitor")
            settings.setValue("theme", theme)
            
        except Exception as e:
            logger.error(f"Error setting theme: {e}")
            
    def _set_light_theme(self):
        """Set light theme for the application."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                color: #000000;
                padding: 8px 16px;
                border: 1px solid #cccccc;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
            }
            QLabel {
                color: #000000;
            }
            QComboBox {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #cccccc;
                padding: 4px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(resources/icons/down_arrow_light.png);
                width: 12px;
                height: 12px;
            }
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 3px;
                text-align: center;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #4d4d4d;
            }
        """)
        
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Lumina Neural Network Monitor",
            """
            <h3>Lumina Neural Network Monitor</h3>
            <p>Version 1.0.0</p>
            <p>A comprehensive monitoring system for neural network operations.</p>
            <p>© 2024 Lumina Systems</p>
            """
        )
        
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Handle application events."""
        if event.type() == QEvent.Close:
            self._cleanup()
        return super().eventFilter(obj, event)
        
    @Slot()
    def _on_system_ready(self):
        """Handle system ready signal."""
        try:
            self.init_progress.setValue(100)
            self.status_label.setText("System Ready")
        except Exception as e:
            logger.error(f"Error handling system ready signal: {e}")
            
    @Slot(dict)
    def update_state(self, state: Dict[str, Any]):
        """Update UI with new system state."""
        if not self._initialized or self._shutting_down:
            return
            
        try:
            # Update status bar
            status = []
            
            if state['neural_seed']['connected']:
                status.append(f"Neural Seed: {state['neural_seed']['growth_stage']}")
                
            if state['signal_system']['connected']:
                status.append(f"Signals: {state['signal_system']['message_count']}")
                
            if state['spiderweb']['connected']:
                status.append(f"Bridges: {state['spiderweb']['active_bridges']}")
                
            self.status_label.setText(" | ".join(status))
            
            # Update component states
            self._update_components()
            
        except Exception as e:
            logger.error(f"Error updating state: {e}")
            
    @Slot(str)
    def show_error(self, error: str):
        """Show error message."""
        try:
            QMessageBox.critical(self, "Error", error)
        except Exception as e:
            logger.error(f"Error showing error message: {e}")
            
    def _set_dark_theme(self):
        """Set dark theme for the application."""
        try:
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #1e1e1e;
                }
                QTabWidget::pane {
                    border: 1px solid #3d3d3d;
                    background-color: #1e1e1e;
                }
                QTabBar::tab {
                    background-color: #2d2d2d;
                    color: #ffffff;
                    padding: 8px 16px;
                    border: 1px solid #3d3d3d;
                }
                QTabBar::tab:selected {
                    background-color: #3d3d3d;
                }
                QLabel {
                    color: #ffffff;
                }
                QComboBox {
                    background-color: #2d2d2d;
                    color: #ffffff;
                    border: 1px solid #3d3d3d;
                    padding: 4px;
                }
                QComboBox::drop-down {
                    border: none;
                }
                QComboBox::down-arrow {
                    image: url(resources/icons/down_arrow.png);
                    width: 12px;
                    height: 12px;
                }
                QProgressBar {
                    border: 1px solid #3d3d3d;
                    border-radius: 3px;
                    text-align: center;
                    background-color: #2d2d2d;
                }
                QProgressBar::chunk {
                    background-color: #4d4d4d;
                }
            """)
        except Exception as e:
            logger.error(f"Error setting dark theme: {e}")
            
    def _initialize_components(self):
        """Initialize all components."""
        try:
            # Initialize system state
            self.system_state = SystemState()
            self.system_state.set_config(self._config.get('data_sources', {}))
            
            # Connect system state signals
            self.system_state.state_updated.connect(self._handle_state_update)
            self.system_state.health_changed.connect(self._handle_health_change)
            self.system_state.stability_changed.connect(self._handle_stability_change)
            self.system_state.growth_stage_changed.connect(self._handle_growth_stage_change)
            
            # Initialize metrics widget
            self.metrics_dashboard = MetricsDashboard()
            self.metrics_dashboard.set_config(self._config.get('metrics', {}))
            
            # Initialize network visualization
            self.network_widget = Network2DWidget()
            self.network_widget.set_config(self._config.get('network', {}))
            
            # Initialize growth visualizer
            self.growth_visualizer = GrowthVisualizer()
            self.growth_visualizer.set_config(self._config.get('appearance', {}))
            
            # Connect growth visualizer signals
            self.growth_visualizer.stage_changed.connect(self._handle_growth_stage_change)
            self.growth_visualizer.growth_completed.connect(self._handle_growth_completed)
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            
    @Slot(dict)
    def _handle_state_update(self, state: Dict[str, Any]):
        """Handle system state update."""
        try:
            # Update status bar
            self.status_label.setText(f"System State Updated: {state['status']}")
            
            # Emit signal for other components
            self.system_state_changed.emit(state)
            
        except Exception as e:
            logger.error(f"Error handling state update: {e}")
            
    @Slot(dict)
    def _handle_health_change(self, health: Dict[str, Any]):
        """Handle health metrics change."""
        try:
            # Update metrics dashboard
            self.metrics_dashboard.update_health(health)
            
        except Exception as e:
            logger.error(f"Error handling health change: {e}")
            
    @Slot(float)
    def _handle_stability_change(self, stability: float):
        """Handle stability metrics change."""
        try:
            # Update metrics dashboard
            self.metrics_dashboard.update_stability(stability)
            
        except Exception as e:
            logger.error(f"Error handling stability change: {e}")
            
    @Slot(str)
    def _handle_growth_stage_change(self, stage: str):
        """Handle growth stage change."""
        try:
            # Update growth visualizer
            self.growth_visualizer.update_stage(stage)
            
            # Update network visualization
            self.network_widget.update_growth(
                stage=stage,
                nodes=self.system_state.get_active_nodes(),
                connections=self.system_state.get_active_connections()
            )
            
        except Exception as e:
            logger.error(f"Error handling growth stage change: {e}")
            
    def _handle_growth_completed(self):
        """Handle growth completion."""
        try:
            # Update status
            self.status_label.setText("Growth Completed")
            
            # Update metrics
            self.metrics_dashboard.update_growth_completed()
            
        except Exception as e:
            logger.error(f"Error handling growth completion: {e}")
            
    def _show_metrics(self):
        """Show metrics dashboard."""
        try:
            self.metrics_dashboard.show()
            self.metrics_dashboard.activateWindow()
            
        except Exception as e:
            logger.error(f"Error showing metrics dashboard: {e}")
            
    def closeEvent(self, event):
        """Handle window close event."""
        try:
            # Close chat window if open
            if self.chat_window.isVisible():
                self.chat_window.close()
                
            # Clean up resources
            self.metrics_dashboard.cleanup()
            self.system_state.cleanup()
            
            event.accept()
            
        except Exception as e:
            logger.error(f"Error during window close: {e}")
            event.accept() 