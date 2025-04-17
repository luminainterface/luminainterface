#!/usr/bin/env python3
"""
AutoWiki Monitor Component for LUMINA v7.5
Provides monitoring interface for the AutoWikiProcessor
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QFrame
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis

from .auto_wiki_processor import AutoWikiProcessor, LUMINA_COLORS

class ModernMetricsCard(QFrame):
    """Modern metrics card component following Lumina design system"""
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("ModernMetricsCard")
        self.setStyleSheet(f"""
            #ModernMetricsCard {{
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                margin: 10px;
                border: 1px solid {LUMINA_COLORS['accent']};
            }}
            QLabel {{
                color: {LUMINA_COLORS['text']};
            }}
            QLabel[class="title"] {{
                font-size: 16px;
                font-weight: bold;
                color: {LUMINA_COLORS['primary']};
            }}
            QLabel[class="metric"] {{
                font-size: 24px;
                color: {LUMINA_COLORS['accent']};
            }}
            QProgressBar {{
                border: 1px solid {LUMINA_COLORS['accent']};
                border-radius: 4px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {LUMINA_COLORS['accent']};
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        title_label = QLabel(title)
        title_label.setProperty("class", "title")
        layout.addWidget(title_label)

class AutoWikiMonitor(QWidget):
    """Monitoring interface for AutoWikiProcessor"""
    def __init__(self, processor: AutoWikiProcessor, parent=None):
        super().__init__(parent)
        self.processor = processor
        self.setup_ui()
        self.connect_signals()
    
    def setup_ui(self):
        """Set up the monitoring UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        title = QLabel("AutoWiki Monitor")
        title.setStyleSheet(f"""
            font-size: 24px;
            color: {LUMINA_COLORS['primary']};
            font-weight: bold;
        """)
        header_layout.addWidget(title)
        
        # Control buttons
        control_buttons = QWidget()
        control_layout = QHBoxLayout(control_buttons)
        control_layout.setContentsMargins(0, 0, 0, 0)
        
        self.start_button = QPushButton("Start Processing")
        self.stop_button = QPushButton("Stop Processing")
        self.refresh_button = QPushButton("Refresh")
        
        for button in [self.start_button, self.stop_button, self.refresh_button]:
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {LUMINA_COLORS['accent']};
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    background-color: {LUMINA_COLORS['primary']};
                }}
                QPushButton:disabled {{
                    background-color: {LUMINA_COLORS['background']};
                    color: {LUMINA_COLORS['text']};
                }}
            """)
            control_layout.addWidget(button)
        
        header_layout.addWidget(control_buttons)
        layout.addWidget(header)
        
        # Status card
        self.status_card = ModernMetricsCard("System Status")
        self.health_progress = QProgressBar()
        self.health_progress.setRange(0, 100)
        self.status_card.layout().addWidget(self.health_progress)
        
        self.status_label = QLabel()
        self.status_label.setProperty("class", "metric")
        self.status_card.layout().addWidget(self.status_label)
        
        layout.addWidget(self.status_card)
        
        # Performance metrics
        metrics_container = QWidget()
        metrics_layout = QHBoxLayout(metrics_container)
        
        # Response time chart
        self.response_time_card = ModernMetricsCard("Response Time")
        self.response_chart = QChart()
        self.response_series = QLineSeries()
        self.response_chart.addSeries(self.response_series)
        
        axis_x = QValueAxis()
        axis_x.setRange(0, 100)
        axis_x.setLabelFormat("%d")
        axis_x.setTitleText("Samples")
        
        axis_y = QValueAxis()
        axis_y.setRange(0, 5)
        axis_y.setLabelFormat("%.2f s")
        axis_y.setTitleText("Response Time")
        
        self.response_chart.addAxis(axis_x, Qt.AlignBottom)
        self.response_chart.addAxis(axis_y, Qt.AlignLeft)
        self.response_series.attachAxis(axis_x)
        self.response_series.attachAxis(axis_y)
        
        chart_view = QChartView(self.response_chart)
        chart_view.setRenderHint(chart_view.RenderHint.Antialiasing)
        self.response_time_card.layout().addWidget(chart_view)
        
        metrics_layout.addWidget(self.response_time_card)
        
        # Queue metrics
        self.queue_card = ModernMetricsCard("Queue Status")
        self.queue_size_label = QLabel()
        self.queue_size_label.setProperty("class", "metric")
        self.queue_card.layout().addWidget(self.queue_size_label)
        
        self.success_rate_progress = QProgressBar()
        self.success_rate_progress.setRange(0, 100)
        self.queue_card.layout().addWidget(self.success_rate_progress)
        
        metrics_layout.addWidget(self.queue_card)
        
        layout.addWidget(metrics_container)
        
        # Error display
        self.error_card = ModernMetricsCard("Recent Errors")
        self.error_label = QLabel()
        self.error_label.setStyleSheet(f"color: {LUMINA_COLORS['error']};")
        self.error_card.layout().addWidget(self.error_label)
        
        layout.addWidget(self.error_card)
    
    def connect_signals(self):
        """Connect processor signals to UI updates"""
        self.processor.statusChanged.connect(self.update_status)
        self.processor.metricsUpdated.connect(self.update_metrics)
        self.processor.errorOccurred.connect(self.show_error)
        
        self.start_button.clicked.connect(self.processor.start_processing)
        self.stop_button.clicked.connect(self.processor.stop_processing)
        self.refresh_button.clicked.connect(self.refresh_metrics)
    
    @Slot(dict)
    def update_status(self, status: dict):
        """Update status display"""
        state = status['state']
        health = status['health_score']
        
        self.health_progress.setValue(int(health * 100))
        
        state_colors = {
            'active': LUMINA_COLORS['success'],
            'degraded': LUMINA_COLORS['warning'],
            'error': LUMINA_COLORS['error']
        }
        
        self.health_progress.setStyleSheet(f"""
            QProgressBar::chunk {{
                background-color: {state_colors.get(state, LUMINA_COLORS['error'])};
            }}
        """)
        
        self.status_label.setText(f"Status: {state.title()}")
    
    @Slot(dict)
    def update_metrics(self, metrics: dict):
        """Update metrics display"""
        # Update response time chart
        self.response_series.clear()
        for i, time in enumerate(metrics['performance']['response_time'][-100:]):
            self.response_series.append(i, time)
        
        # Update queue status
        self.queue_size_label.setText(f"Queue Size: {metrics['queue_size']}")
        self.success_rate_progress.setValue(int(metrics['success_rate']))
    
    @Slot(str)
    def show_error(self, error: str):
        """Display error message"""
        self.error_label.setText(f"Error: {error}")
    
    def refresh_metrics(self):
        """Manual refresh of metrics"""
        state = self.processor.get_monitoring_state()
        self.update_status({
            'state': state['component_state'],
            'health_score': state['health_score']
        })
        self.update_metrics({
            'performance': state['metrics'],
            'queue_size': state['queue_status']['queue_size'],
            'success_rate': state['queue_status']['success_rate']
        }) 
 
 
"""
AutoWiki Monitor Component for LUMINA v7.5
Provides monitoring interface for the AutoWikiProcessor
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QFrame
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis

from .auto_wiki_processor import AutoWikiProcessor, LUMINA_COLORS

class ModernMetricsCard(QFrame):
    """Modern metrics card component following Lumina design system"""
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("ModernMetricsCard")
        self.setStyleSheet(f"""
            #ModernMetricsCard {{
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                margin: 10px;
                border: 1px solid {LUMINA_COLORS['accent']};
            }}
            QLabel {{
                color: {LUMINA_COLORS['text']};
            }}
            QLabel[class="title"] {{
                font-size: 16px;
                font-weight: bold;
                color: {LUMINA_COLORS['primary']};
            }}
            QLabel[class="metric"] {{
                font-size: 24px;
                color: {LUMINA_COLORS['accent']};
            }}
            QProgressBar {{
                border: 1px solid {LUMINA_COLORS['accent']};
                border-radius: 4px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {LUMINA_COLORS['accent']};
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        title_label = QLabel(title)
        title_label.setProperty("class", "title")
        layout.addWidget(title_label)

class AutoWikiMonitor(QWidget):
    """Monitoring interface for AutoWikiProcessor"""
    def __init__(self, processor: AutoWikiProcessor, parent=None):
        super().__init__(parent)
        self.processor = processor
        self.setup_ui()
        self.connect_signals()
    
    def setup_ui(self):
        """Set up the monitoring UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        title = QLabel("AutoWiki Monitor")
        title.setStyleSheet(f"""
            font-size: 24px;
            color: {LUMINA_COLORS['primary']};
            font-weight: bold;
        """)
        header_layout.addWidget(title)
        
        # Control buttons
        control_buttons = QWidget()
        control_layout = QHBoxLayout(control_buttons)
        control_layout.setContentsMargins(0, 0, 0, 0)
        
        self.start_button = QPushButton("Start Processing")
        self.stop_button = QPushButton("Stop Processing")
        self.refresh_button = QPushButton("Refresh")
        
        for button in [self.start_button, self.stop_button, self.refresh_button]:
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {LUMINA_COLORS['accent']};
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    background-color: {LUMINA_COLORS['primary']};
                }}
                QPushButton:disabled {{
                    background-color: {LUMINA_COLORS['background']};
                    color: {LUMINA_COLORS['text']};
                }}
            """)
            control_layout.addWidget(button)
        
        header_layout.addWidget(control_buttons)
        layout.addWidget(header)
        
        # Status card
        self.status_card = ModernMetricsCard("System Status")
        self.health_progress = QProgressBar()
        self.health_progress.setRange(0, 100)
        self.status_card.layout().addWidget(self.health_progress)
        
        self.status_label = QLabel()
        self.status_label.setProperty("class", "metric")
        self.status_card.layout().addWidget(self.status_label)
        
        layout.addWidget(self.status_card)
        
        # Performance metrics
        metrics_container = QWidget()
        metrics_layout = QHBoxLayout(metrics_container)
        
        # Response time chart
        self.response_time_card = ModernMetricsCard("Response Time")
        self.response_chart = QChart()
        self.response_series = QLineSeries()
        self.response_chart.addSeries(self.response_series)
        
        axis_x = QValueAxis()
        axis_x.setRange(0, 100)
        axis_x.setLabelFormat("%d")
        axis_x.setTitleText("Samples")
        
        axis_y = QValueAxis()
        axis_y.setRange(0, 5)
        axis_y.setLabelFormat("%.2f s")
        axis_y.setTitleText("Response Time")
        
        self.response_chart.addAxis(axis_x, Qt.AlignBottom)
        self.response_chart.addAxis(axis_y, Qt.AlignLeft)
        self.response_series.attachAxis(axis_x)
        self.response_series.attachAxis(axis_y)
        
        chart_view = QChartView(self.response_chart)
        chart_view.setRenderHint(chart_view.RenderHint.Antialiasing)
        self.response_time_card.layout().addWidget(chart_view)
        
        metrics_layout.addWidget(self.response_time_card)
        
        # Queue metrics
        self.queue_card = ModernMetricsCard("Queue Status")
        self.queue_size_label = QLabel()
        self.queue_size_label.setProperty("class", "metric")
        self.queue_card.layout().addWidget(self.queue_size_label)
        
        self.success_rate_progress = QProgressBar()
        self.success_rate_progress.setRange(0, 100)
        self.queue_card.layout().addWidget(self.success_rate_progress)
        
        metrics_layout.addWidget(self.queue_card)
        
        layout.addWidget(metrics_container)
        
        # Error display
        self.error_card = ModernMetricsCard("Recent Errors")
        self.error_label = QLabel()
        self.error_label.setStyleSheet(f"color: {LUMINA_COLORS['error']};")
        self.error_card.layout().addWidget(self.error_label)
        
        layout.addWidget(self.error_card)
    
    def connect_signals(self):
        """Connect processor signals to UI updates"""
        self.processor.statusChanged.connect(self.update_status)
        self.processor.metricsUpdated.connect(self.update_metrics)
        self.processor.errorOccurred.connect(self.show_error)
        
        self.start_button.clicked.connect(self.processor.start_processing)
        self.stop_button.clicked.connect(self.processor.stop_processing)
        self.refresh_button.clicked.connect(self.refresh_metrics)
    
    @Slot(dict)
    def update_status(self, status: dict):
        """Update status display"""
        state = status['state']
        health = status['health_score']
        
        self.health_progress.setValue(int(health * 100))
        
        state_colors = {
            'active': LUMINA_COLORS['success'],
            'degraded': LUMINA_COLORS['warning'],
            'error': LUMINA_COLORS['error']
        }
        
        self.health_progress.setStyleSheet(f"""
            QProgressBar::chunk {{
                background-color: {state_colors.get(state, LUMINA_COLORS['error'])};
            }}
        """)
        
        self.status_label.setText(f"Status: {state.title()}")
    
    @Slot(dict)
    def update_metrics(self, metrics: dict):
        """Update metrics display"""
        # Update response time chart
        self.response_series.clear()
        for i, time in enumerate(metrics['performance']['response_time'][-100:]):
            self.response_series.append(i, time)
        
        # Update queue status
        self.queue_size_label.setText(f"Queue Size: {metrics['queue_size']}")
        self.success_rate_progress.setValue(int(metrics['success_rate']))
    
    @Slot(str)
    def show_error(self, error: str):
        """Display error message"""
        self.error_label.setText(f"Error: {error}")
    
    def refresh_metrics(self):
        """Manual refresh of metrics"""
        state = self.processor.get_monitoring_state()
        self.update_status({
            'state': state['component_state'],
            'health_score': state['health_score']
        })
        self.update_metrics({
            'performance': state['metrics'],
            'queue_size': state['queue_status']['queue_size'],
            'success_rate': state['queue_status']['success_rate']
        }) 
 