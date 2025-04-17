from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QComboBox, QTableWidget, QTableWidgetItem, QFrame, QSplitter,
                             QGroupBox, QGridLayout, QSpinBox, QSlider, QTabWidget,
                             QProgressBar, QCheckBox, QLineEdit, QTextEdit)
from PySide6.QtCore import Qt, Signal, QTimer, QPointF
from PySide6.QtGui import QPainter, QBrush, QPen, QColor, QFont, QPixmap, QRadialGradient, QPainterPath

import random
import math

class ConnectionMapWidget(QWidget):
    """Widget for visualizing connections between neural networks and glyphs"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(400)
        self.nodes = self.create_sample_nodes()
        self.connections = self.create_sample_connections()
        self.dragging_node = None
        self.hover_node = None
        self.setMouseTracking(True)
    
    def create_sample_nodes(self):
        """Create sample node data for visualization"""
        nodes = []
        
        # Neural network nodes (blue)
        nodes.append({
            'id': 'nn1',
            'type': 'neural',
            'name': 'CNN Feature Extractor',
            'x': 150,
            'y': 100,
            'radius': 30,
            'color': QColor(100, 150, 230)
        })
        
        nodes.append({
            'id': 'nn2',
            'type': 'neural',
            'name': 'RNN Sequence Processor',
            'x': 150,
            'y': 200,
            'radius': 30,
            'color': QColor(100, 150, 230)
        })
        
        nodes.append({
            'id': 'nn3',
            'type': 'neural',
            'name': 'Transformer Encoder',
            'x': 150,
            'y': 300,
            'radius': 30,
            'color': QColor(100, 150, 230)
        })
        
        # Glyph nodes (green)
        nodes.append({
            'id': 'g1',
            'type': 'glyph',
            'name': 'Alpha Circle',
            'x': 450,
            'y': 80,
            'radius': 25,
            'color': QColor(100, 200, 130)
        })
        
        nodes.append({
            'id': 'g2',
            'type': 'glyph',
            'name': 'Triangle',
            'x': 450,
            'y': 180,
            'radius': 25,
            'color': QColor(100, 200, 130)
        })
        
        nodes.append({
            'id': 'g3',
            'type': 'glyph',
            'name': 'Crossed Lines',
            'x': 450,
            'y': 280,
            'radius': 25,
            'color': QColor(100, 200, 130)
        })
        
        nodes.append({
            'id': 'g4',
            'type': 'glyph',
            'name': 'Spiral',
            'x': 450,
            'y': 380,
            'radius': 25,
            'color': QColor(100, 200, 130)
        })
        
        # Integration nodes (purple)
        nodes.append({
            'id': 'i1',
            'type': 'integration',
            'name': 'Feature Mapping',
            'x': 300,
            'y': 150,
            'radius': 20,
            'color': QColor(180, 120, 200)
        })
        
        nodes.append({
            'id': 'i2',
            'type': 'integration',
            'name': 'Pattern Matcher',
            'x': 300,
            'y': 300,
            'radius': 20,
            'color': QColor(180, 120, 200)
        })
        
        return nodes
    
    def create_sample_connections(self):
        """Create sample connection data"""
        connections = []
        
        # Neural to integration connections
        connections.append({
            'source': 'nn1',
            'target': 'i1',
            'strength': 0.85,
            'description': 'Feature extraction'
        })
        
        connections.append({
            'source': 'nn2',
            'target': 'i1',
            'strength': 0.7,
            'description': 'Sequence analysis'
        })
        
        connections.append({
            'source': 'nn2',
            'target': 'i2',
            'strength': 0.65,
            'description': 'Pattern recognition'
        })
        
        connections.append({
            'source': 'nn3',
            'target': 'i2',
            'strength': 0.9,
            'description': 'Context encoding'
        })
        
        # Integration to glyph connections
        connections.append({
            'source': 'i1',
            'target': 'g1',
            'strength': 0.82,
            'description': 'Circle mapping'
        })
        
        connections.append({
            'source': 'i1',
            'target': 'g2',
            'strength': 0.75,
            'description': 'Triangle mapping'
        })
        
        connections.append({
            'source': 'i2',
            'target': 'g3',
            'strength': 0.78,
            'description': 'Cross pattern recognition'
        })
        
        connections.append({
            'source': 'i2',
            'target': 'g4',
            'strength': 0.92,
            'description': 'Spiral recognition'
        })
        
        return connections
    
    def paintEvent(self, event):
        """Paint the connection map"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill background
        painter.fillRect(0, 0, self.width(), self.height(), QColor(245, 245, 245))
        
        # Draw connections first (so they're behind nodes)
        self.draw_connections(painter)
        
        # Draw nodes
        self.draw_nodes(painter)
        
        # Draw legend
        self.draw_legend(painter)
    
    def draw_connections(self, painter):
        """Draw connections between nodes"""
        for conn in self.connections:
            # Get source and target nodes
            source_node = next((n for n in self.nodes if n['id'] == conn['source']), None)
            target_node = next((n for n in self.nodes if n['id'] == conn['target']), None)
            
            if not source_node or not target_node:
                continue
            
            # Calculate connection strength (affects line width)
            strength = conn['strength']
            line_width = 1 + strength * 4  # 1-5 pixels based on strength
            
            # Set pen color based on connection type
            if source_node['type'] == 'neural' and target_node['type'] == 'integration':
                # Neural to integration: blue gradient
                pen_color = QColor(100, 150, 230, int(200 * strength))
            elif source_node['type'] == 'integration' and target_node['type'] == 'glyph':
                # Integration to glyph: green gradient
                pen_color = QColor(100, 200, 130, int(200 * strength))
            else:
                # Default color
                pen_color = QColor(150, 150, 150, int(200 * strength))
            
            # Draw connection line
            pen = QPen(pen_color, line_width)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            
            # Draw a curved path
            path = QPainterPath()
            path.moveTo(source_node['x'], source_node['y'])
            
            # Control points for the curve
            # Calculate a midpoint with some offset
            mid_x = (source_node['x'] + target_node['x']) / 2
            mid_y = (source_node['y'] + target_node['y']) / 2
            
            # Add some random variation to make lines more distinct
            offset_x = (source_node['id'] + target_node['id']).__hash__() % 40 - 20
            offset_y = (source_node['id'] + target_node['id']).__hash__() % 30 - 15
            
            # Draw the curved line
            path.cubicTo(
                mid_x - offset_x, source_node['y'], 
                mid_x + offset_x, target_node['y'],
                target_node['x'], target_node['y']
            )
            painter.drawPath(path)
            
            # Draw a small circle at the midpoint if the connection is highlighted
            if self.hover_node and (self.hover_node['id'] == source_node['id'] or 
                                   self.hover_node['id'] == target_node['id']):
                # Highlight this connection
                painter.setBrush(QBrush(pen_color))
                painter.drawEllipse(mid_x - 3, mid_y - 3, 6, 6)
                
                # Draw the connection description
                painter.setPen(QColor(50, 50, 50))
                painter.setFont(QFont("Arial", 8))
                
                # Position the text along the connection
                text_x = mid_x + 10
                text_y = mid_y - 10
                
                # Draw with a light background for readability
                text_rect = painter.boundingRect(
                    int(text_x), int(text_y), 150, 20, 
                    Qt.AlignLeft | Qt.AlignVCenter, conn['description']
                )
                
                # Background
                painter.fillRect(
                    text_rect.adjusted(-2, -2, 2, 2),
                    QColor(255, 255, 255, 200)
                )
                
                # Text
                painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, conn['description'])
                
                # Draw strength as percentage
                strength_text = f"{int(strength * 100)}%"
                painter.drawText(
                    text_rect.right() + 5, text_rect.top(),
                    40, text_rect.height(),
                    Qt.AlignLeft | Qt.AlignVCenter, strength_text
                )
    
    def draw_nodes(self, painter):
        """Draw all nodes"""
        for node in self.nodes:
            # Determine if node is being hovered or dragged
            is_hover = self.hover_node and self.hover_node['id'] == node['id']
            is_dragging = self.dragging_node and self.dragging_node['id'] == node['id']
            
            # Set up colors
            base_color = node['color']
            border_color = QColor(50, 50, 50)
            
            if is_hover or is_dragging:
                # Lighten the color for hover effect
                base_color = QColor(
                    min(255, base_color.red() + 30),
                    min(255, base_color.green() + 30),
                    min(255, base_color.blue() + 30)
                )
                border_color = QColor(30, 30, 30)
            
            # Create a gradient for the node
            gradient = QRadialGradient(
                node['x'], node['y'],
                node['radius']
            )
            gradient.setColorAt(0, base_color.lighter(120))
            gradient.setColorAt(1, base_color)
            
            # Draw node
            painter.setBrush(QBrush(gradient))
            pen = QPen(border_color, 1.5 if is_hover or is_dragging else 1)
            painter.setPen(pen)
            painter.drawEllipse(
                node['x'] - node['radius'],
                node['y'] - node['radius'],
                node['radius'] * 2,
                node['radius'] * 2
            )
            
            # Draw node label
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            
            # Abbreviate long names
            display_name = node['name']
            if len(display_name) > 15:
                display_name = display_name[:12] + "..."
                
            text_rect = painter.boundingRect(
                int(node['x'] - node['radius']), 
                int(node['y'] - node['radius']), 
                int(node['radius'] * 2), 
                int(node['radius'] * 2), 
                Qt.AlignCenter, 
                display_name
            )
            
            painter.drawText(text_rect, Qt.AlignCenter, display_name)
            
            # If hovering, draw the full name as a tooltip
            if is_hover and len(node['name']) > 15:
                tooltip_rect = painter.boundingRect(
                    int(node['x'] - 100), 
                    int(node['y'] + node['radius'] + 5), 
                    200, 
                    20, 
                    Qt.AlignCenter, 
                    node['name']
                )
                
                # Draw background
                painter.fillRect(
                    tooltip_rect.adjusted(-5, -2, 5, 2),
                    QColor(50, 50, 50, 200)
                )
                
                # Draw text
                painter.drawText(tooltip_rect, Qt.AlignCenter, node['name'])
    
    def draw_legend(self, painter):
        """Draw a legend explaining the node types"""
        legend_x = 10
        legend_y = self.height() - 80
        legend_width = 180
        legend_height = 70
        
        # Draw legend background
        painter.fillRect(
            legend_x, legend_y,
            legend_width, legend_height,
            QColor(255, 255, 255, 200)
        )
        
        painter.setPen(QColor(100, 100, 100))
        painter.drawRect(
            legend_x, legend_y,
            legend_width, legend_height
        )
        
        # Draw legend title
        painter.setPen(QColor(50, 50, 50))
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        painter.drawText(
            legend_x + 10, legend_y + 5,
            legend_width - 20, 20,
            Qt.AlignLeft | Qt.AlignVCenter,
            "Node Types"
        )
        
        # Draw legend items
        painter.setFont(QFont("Arial", 8))
        
        # Neural Network node
        painter.setBrush(QColor(100, 150, 230))
        painter.setPen(QColor(50, 50, 50))
        painter.drawEllipse(legend_x + 15, legend_y + 30, 10, 10)
        painter.drawText(
            legend_x + 30, legend_y + 25,
            100, 20,
            Qt.AlignLeft | Qt.AlignVCenter,
            "Neural Network"
        )
        
        # Integration node
        painter.setBrush(QColor(180, 120, 200))
        painter.drawEllipse(legend_x + 15, legend_y + 50, 10, 10)
        painter.drawText(
            legend_x + 30, legend_y + 45,
            100, 20,
            Qt.AlignLeft | Qt.AlignVCenter,
            "Integration Layer"
        )
        
        # Glyph node
        painter.setBrush(QColor(100, 200, 130))
        painter.drawEllipse(legend_x + 110, legend_y + 30, 10, 10)
        painter.drawText(
            legend_x + 125, legend_y + 25,
            100, 20,
            Qt.AlignLeft | Qt.AlignVCenter,
            "Glyph"
        )
    
    def mousePressEvent(self, event):
        """Handle mouse press to start dragging nodes"""
        pos = event.position()
        x, y = pos.x(), pos.y()
        
        # Check if a node was clicked
        for node in self.nodes:
            # Calculate distance to node center
            distance = math.sqrt((x - node['x'])**2 + (y - node['y'])**2)
            
            if distance <= node['radius']:
                self.dragging_node = node
                break
                
    def mouseReleaseEvent(self, event):
        """Handle mouse release to stop dragging"""
        self.dragging_node = None
    
    def mouseMoveEvent(self, event):
        """Handle mouse movement for dragging and hovering"""
        pos = event.position()
        x, y = pos.x(), pos.y()
        
        # Check for dragging
        if self.dragging_node:
            self.dragging_node['x'] = x
            self.dragging_node['y'] = y
            self.update()
            return
        
        # Check for hovering
        prev_hover = self.hover_node
        self.hover_node = None
        
        for node in self.nodes:
            # Calculate distance to node center
            distance = math.sqrt((x - node['x'])**2 + (y - node['y'])**2)
            
            if distance <= node['radius']:
                self.hover_node = node
                break
        
        # Only update if hover state changed
        if prev_hover != self.hover_node:
            self.update()


class IntegrationMetricsWidget(QWidget):
    """Widget for displaying metrics about neural-glyph integration"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.metrics = self.generate_metrics()
    
    def generate_metrics(self):
        """Generate sample metrics data"""
        return {
            "Overall Integration Score": 78,
            "Neural Coverage": 85,
            "Glyph Recognition Rate": 92,
            "Transformation Accuracy": 81,
            "Processing Latency": 65
        }
    
    def paintEvent(self, event):
        """Draw the metrics display"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Background
        painter.fillRect(0, 0, width, height, QColor(240, 240, 240))
        
        # Calculate metrics layout
        metrics_count = len(self.metrics)
        metrics_per_row = 3
        rows = math.ceil(metrics_count / metrics_per_row)
        
        metric_width = width / metrics_per_row
        metric_height = height / rows
        
        # Draw metrics
        idx = 0
        for metric_name, value in self.metrics.items():
            row = idx // metrics_per_row
            col = idx % metrics_per_row
            
            x = col * metric_width
            y = row * metric_height
            
            # Draw metric box
            painter.setPen(QColor(200, 200, 200))
            painter.drawRect(x + 5, y + 5, metric_width - 10, metric_height - 10)
            
            # Draw metric name
            painter.setPen(QColor(80, 80, 80))
            painter.setFont(QFont("Arial", 9))
            painter.drawText(
                x + 10, y + 10,
                metric_width - 20, 20,
                Qt.AlignLeft, metric_name
            )
            
            # Draw meter background
            painter.fillRect(
                x + 15, y + 35,
                metric_width - 30, 15,
                QColor(220, 220, 220)
            )
            
            # Draw value meter
            # Color based on value
            if value >= 80:
                color = QColor(100, 200, 100)  # Green
            elif value >= 60:
                color = QColor(200, 200, 100)  # Yellow
            else:
                color = QColor(200, 100, 100)  # Red
                
            painter.fillRect(
                x + 15, y + 35,
                int((metric_width - 30) * (value / 100)),
                15,
                color
            )
            
            # Draw value text
            painter.setPen(QColor(50, 50, 50))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.drawText(
                x + 10, y + 55,
                metric_width - 20, 20,
                Qt.AlignCenter, f"{value}%"
            )
            
            idx += 1


class ModelIntegrationPanel(QWidget):
    """Panel for visualizing and configuring neural-glyph integration"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 5)
        
        title = QLabel("Neural-Glyph Integration")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2C3E50;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Add model selector
        model_label = QLabel("Integration Model:")
        header_layout.addWidget(model_label)
        
        self.model_selector = QComboBox()
        self.model_selector.addItems([
            "Basic Feature Mapping",
            "Advanced Semantic Integration",
            "Multi-layered Translation",
            "Experimental Hyperdimensional Bridge"
        ])
        header_layout.addWidget(self.model_selector)
        
        # Add refresh button
        refresh_btn = QPushButton("Refresh View")
        refresh_btn.clicked.connect(self.refresh_view)
        header_layout.addWidget(refresh_btn)
        
        main_layout.addWidget(header)
        
        # Main content with tabs
        self.tabs = QTabWidget()
        
        # Tab 1: Connection Map
        map_tab = QWidget()
        map_layout = QVBoxLayout(map_tab)
        
        self.connection_map = ConnectionMapWidget()
        map_layout.addWidget(self.connection_map)
        
        # Add metrics panel below map
        self.metrics_widget = IntegrationMetricsWidget()
        map_layout.addWidget(self.metrics_widget)
        
        self.tabs.addTab(map_tab, "Connection Map")
        
        # Tab 2: Configuration
        config_tab = QWidget()
        config_layout = QGridLayout(config_tab)
        
        # Neural network configuration
        nn_group = QGroupBox("Neural Network Configuration")
        nn_layout = QGridLayout(nn_group)
        
        nn_layout.addWidget(QLabel("Feature Extractors:"), 0, 0)
        nn_extractors = QComboBox()
        nn_extractors.addItems(["CNN", "ResNet", "Vision Transformer", "Custom"])
        nn_layout.addWidget(nn_extractors, 0, 1)
        
        nn_layout.addWidget(QLabel("Sequence Processing:"), 1, 0)
        nn_sequence = QComboBox()
        nn_sequence.addItems(["LSTM", "GRU", "Transformer", "None"])
        nn_layout.addWidget(nn_sequence, 1, 1)
        
        nn_layout.addWidget(QLabel("Output Dimensions:"), 2, 0)
        nn_dimensions = QSpinBox()
        nn_dimensions.setRange(16, 1024)
        nn_dimensions.setValue(256)
        nn_dimensions.setSingleStep(16)
        nn_layout.addWidget(nn_dimensions, 2, 1)
        
        config_layout.addWidget(nn_group, 0, 0)
        
        # Glyph configuration
        glyph_group = QGroupBox("Glyph Configuration")
        glyph_layout = QGridLayout(glyph_group)
        
        glyph_layout.addWidget(QLabel("Recognition Threshold:"), 0, 0)
        glyph_threshold = QSlider(Qt.Horizontal)
        glyph_threshold.setRange(0, 100)
        glyph_threshold.setValue(70)
        glyph_layout.addWidget(glyph_threshold, 0, 1)
        
        glyph_layout.addWidget(QLabel("Complexity Level:"), 1, 0)
        glyph_complexity = QComboBox()
        glyph_complexity.addItems(["Basic", "Intermediate", "Advanced", "Expert"])
        glyph_layout.addWidget(glyph_complexity, 1, 1)
        
        glyph_layout.addWidget(QLabel("Feedback Mode:"), 2, 0)
        glyph_feedback = QComboBox()
        glyph_feedback.addItems(["Visual Only", "Text + Visual", "Interactive", "Full Sensory"])
        glyph_layout.addWidget(glyph_feedback, 2, 1)
        
        config_layout.addWidget(glyph_group, 0, 1)
        
        # Integration configuration
        integration_group = QGroupBox("Integration Settings")
        integration_layout = QGridLayout(integration_group)
        
        integration_layout.addWidget(QLabel("Mapping Strategy:"), 0, 0)
        integration_strategy = QComboBox()
        integration_strategy.addItems(["Direct Feature Mapping", "Semantic Pooling", "Attention-based", "Hybrid"])
        integration_layout.addWidget(integration_strategy, 0, 1)
        
        integration_layout.addWidget(QLabel("Translation Layers:"), 1, 0)
        integration_layers = QSpinBox()
        integration_layers.setRange(1, 5)
        integration_layers.setValue(2)
        integration_layout.addWidget(integration_layers, 1, 1)
        
        integration_layout.addWidget(QLabel("Feedback Integration:"), 2, 0)
        integration_feedback = QCheckBox("Enable bidirectional feedback")
        integration_feedback.setChecked(True)
        integration_layout.addWidget(integration_feedback, 2, 1)
        
        config_layout.addWidget(integration_group, 1, 0, 1, 2)
        
        # Apply button
        apply_btn = QPushButton("Apply Configuration")
        apply_btn.clicked.connect(self.apply_configuration)
        config_layout.addWidget(apply_btn, 2, 0, 1, 2, Qt.AlignRight)
        
        self.tabs.addTab(config_tab, "Configuration")
        
        # Tab 3: Performance Analysis
        performance_tab = QWidget()
        performance_layout = QVBoxLayout(performance_tab)
        
        performance_label = QLabel("Analyze the performance of neural-glyph integration")
        performance_layout.addWidget(performance_label)
        
        # Placeholder for performance analysis
        performance_layout.addWidget(QLabel("Performance analysis will be implemented here"))
        
        self.tabs.addTab(performance_tab, "Performance Analysis")
        
        main_layout.addWidget(self.tabs)
        
        # Footer with status
        footer = QFrame()
        footer.setFrameShape(QFrame.StyledPanel)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(5, 2, 5, 2)
        
        self.status_label = QLabel("Ready. Viewing default integration model.")
        footer_layout.addWidget(self.status_label)
        
        main_layout.addWidget(footer)
    
    def refresh_view(self):
        """Refresh the integration view"""
        model = self.model_selector.currentText()
        self.status_label.setText(f"Refreshed view using {model} integration model.")
        
        # In a real application, this would update the connection map with real data
        # For this demo, we'll just simulate a refresh
        self.connection_map.update()
    
    def apply_configuration(self):
        """Apply the selected configuration"""
        # In a real application, this would apply the settings to the actual model
        self.status_label.setText("Applied new configuration. Integration model updated.")
        
        # Navigate to the Connection Map tab to show the results
        self.tabs.setCurrentIndex(0) 