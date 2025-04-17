from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QProgressBar, QTabWidget, QGroupBox)
from PySide6.QtCore import Qt, QTimer
import psutil
from frontend.bridge_integration import get_bridge

class MonitorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("System Monitor")
        self.setGeometry(200, 200, 800, 600)
        
        # Initialize version bridge
        self.bridge = get_bridge()
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget for different monitoring views
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # System metrics tab
        self.system_tab = QWidget()
        self.tabs.addTab(self.system_tab, "System Metrics")
        self.setup_system_metrics()
        
        # Version-specific tabs
        for version in range(1, 6):
            version_tab = QWidget()
            self.tabs.addTab(version_tab, f"v{version} Metrics")
            self.setup_version_metrics(version_tab, f"v{version}")
        
        # Set up timer for updating metrics
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_metrics)
        self.timer.start(1000)  # Update every second
    
    def setup_system_metrics(self):
        layout = QVBoxLayout(self.system_tab)
        
        # CPU usage
        cpu_group = QGroupBox("CPU Usage")
        cpu_layout = QVBoxLayout()
        self.cpu_label = QLabel("CPU Usage:")
        self.cpu_bar = QProgressBar()
        cpu_layout.addWidget(self.cpu_label)
        cpu_layout.addWidget(self.cpu_bar)
        cpu_group.setLayout(cpu_layout)
        layout.addWidget(cpu_group)
        
        # Memory usage
        memory_group = QGroupBox("Memory Usage")
        memory_layout = QVBoxLayout()
        self.memory_label = QLabel("Memory Usage:")
        self.memory_bar = QProgressBar()
        memory_layout.addWidget(self.memory_label)
        memory_layout.addWidget(self.memory_bar)
        memory_group.setLayout(memory_layout)
        layout.addWidget(memory_group)
        
        # GPU usage
        gpu_group = QGroupBox("GPU Usage")
        gpu_layout = QVBoxLayout()
        self.gpu_label = QLabel("GPU Usage:")
        self.gpu_bar = QProgressBar()
        gpu_layout.addWidget(self.gpu_label)
        gpu_layout.addWidget(self.gpu_bar)
        gpu_group.setLayout(gpu_layout)
        layout.addWidget(gpu_group)
    
    def setup_version_metrics(self, tab: QWidget, version: str):
        layout = QVBoxLayout(tab)
        
        # Get version-specific metrics from bridge
        metrics = self.bridge.get_visualization_data(version)
        
        # Create metric groups based on available data
        for metric_group, values in metrics.items():
            group = QGroupBox(metric_group)
            group_layout = QVBoxLayout()
            
            for metric_name, value in values.items():
                label = QLabel(f"{metric_name}:")
                progress = QProgressBar()
                progress.setValue(int(value * 100))  # Assuming values are 0-1
                group_layout.addWidget(label)
                group_layout.addWidget(progress)
            
            group.setLayout(group_layout)
            layout.addWidget(group)
    
    def update_metrics(self):
        # Update system metrics
        cpu_percent = psutil.cpu_percent()
        self.cpu_bar.setValue(int(cpu_percent))
        
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        self.memory_bar.setValue(int(memory_percent))
        
        # TODO: Add GPU monitoring
        self.gpu_bar.setValue(0)  # Placeholder for now
        
        # Update version-specific metrics
        current_tab = self.tabs.currentWidget()
        if current_tab != self.system_tab:
            version = self.tabs.tabText(self.tabs.currentIndex()).replace("v", "").replace(" Metrics", "")
            self.setup_version_metrics(current_tab, f"v{version}") 