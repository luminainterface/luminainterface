"""
ModernMetricsCard component for Lumina GUI
"""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PySide6.QtCore import Qt

from .modern_card import ModernCard
from .modern_progress_circle import ModernProgressCircle

class ModernMetricsCard(ModernCard):
    """A card component for displaying metrics with modern styling"""
    
    def __init__(self, title: str = "", parent=None):
        super().__init__(title, parent)
        
        # Initialize metrics container
        self.metrics_container = QWidget()
        self.metrics_layout = QVBoxLayout(self.metrics_container)
        self.metrics_layout.setContentsMargins(0, 0, 0, 0)
        self.metrics_layout.setSpacing(15)
        
        # Add metrics container to card
        self.layout.addWidget(self.metrics_container)
        
        # Store metrics widgets
        self.metrics = {}
        
    def add_metric(self, name: str, value: str, color: str = None):
        """Add a metric to the card
        
        Args:
            name: Name/label of the metric
            value: Current value of the metric
            color: Color for the metric value (optional)
        """
        # Create metric container
        metric = QWidget()
        layout = QHBoxLayout(metric)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # Add label
        label = QLabel(name)
        label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #1A1A1A;
            }
        """)
        layout.addWidget(label)
        
        # Add spacer
        layout.addStretch()
        
        # Add value
        value_label = QLabel(value)
        if color:
            value_label.setStyleSheet(f"""
                QLabel {{
                    font-size: 14px;
                    font-weight: bold;
                    color: {color};
                }}
            """)
        else:
            value_label.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    font-weight: bold;
                    color: #1A1A1A;
                }
            """)
        layout.addWidget(value_label)
        
        # Store metric widgets
        self.metrics[name] = {
            'container': metric,
            'label': label,
            'value': value_label
        }
        
        # Add to layout
        self.metrics_layout.addWidget(metric)
        
    def update_metric(self, name: str, value: str):
        """Update the value of an existing metric
        
        Args:
            name: Name of the metric to update
            value: New value for the metric
        """
        if name in self.metrics:
            self.metrics[name]['value'].setText(str(value))
            
    def remove_metric(self, name: str):
        """Remove a metric from the card
        
        Args:
            name: Name of the metric to remove
        """
        if name in self.metrics:
            metric = self.metrics[name]
            self.metrics_layout.removeWidget(metric['container'])
            metric['container'].deleteLater()
            del self.metrics[name] 