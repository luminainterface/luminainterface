from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QComboBox, QTableWidget, QTableWidgetItem, QFrame, QSplitter,
                             QGroupBox, QGridLayout, QSpinBox, QSlider, QTabWidget,
                             QProgressBar, QCheckBox)
from PySide6.QtCore import Qt, Signal, QTimer, QSize
from PySide6.QtGui import QPainter, QBrush, QPen, QColor, QFont, QPixmap, QRadialGradient

import random
import math

class GlyphVisualizerWidget(QWidget):
    """Widget for visualizing glyphs and their neural activation patterns"""
    
    glyph_selected = Signal(int)  # Emitted when a glyph is selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_glyph = None
        self.glyphs = []  # List of glyph data
        self.setMinimumHeight(300)
        self.init_sample_glyphs()
        
    def init_sample_glyphs(self):
        """Initialize sample glyph data for visualization"""
        # Sample glyph data: id, name, pattern (coordinates for drawing), activation_level
        self.glyphs = [
            {
                'id': 1, 
                'name': 'Alpha Circle', 
                'pattern': [
                    {'type': 'circle', 'x': 0.5, 'y': 0.5, 'radius': 0.4}
                ],
                'activation_level': 0.85
            },
            {
                'id': 2, 
                'name': 'Crossed Lines', 
                'pattern': [
                    {'type': 'line', 'x1': 0.2, 'y1': 0.2, 'x2': 0.8, 'y2': 0.8},
                    {'type': 'line', 'x1': 0.2, 'y1': 0.8, 'x2': 0.8, 'y2': 0.2}
                ],
                'activation_level': 0.72
            },
            {
                'id': 3, 
                'name': 'Triangle', 
                'pattern': [
                    {'type': 'line', 'x1': 0.5, 'y1': 0.1, 'x2': 0.1, 'y2': 0.8},
                    {'type': 'line', 'x1': 0.1, 'y1': 0.8, 'x2': 0.9, 'y2': 0.8},
                    {'type': 'line', 'x1': 0.9, 'y1': 0.8, 'x2': 0.5, 'y2': 0.1}
                ],
                'activation_level': 0.91
            },
            {
                'id': 4, 
                'name': 'Square', 
                'pattern': [
                    {'type': 'line', 'x1': 0.2, 'y1': 0.2, 'x2': 0.8, 'y2': 0.2},
                    {'type': 'line', 'x1': 0.8, 'y1': 0.2, 'x2': 0.8, 'y2': 0.8},
                    {'type': 'line', 'x1': 0.8, 'y1': 0.8, 'x2': 0.2, 'y2': 0.8},
                    {'type': 'line', 'x1': 0.2, 'y1': 0.8, 'x2': 0.2, 'y2': 0.2}
                ],
                'activation_level': 0.68
            },
            {
                'id': 5, 
                'name': 'Spiral', 
                'pattern': [
                    {'type': 'spiral', 'x': 0.5, 'y': 0.5, 'radius': 0.4, 'turns': 3}
                ],
                'activation_level': 0.79
            }
        ]
    
    def paintEvent(self, event):
        """Custom paint event to draw the glyphs"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Calculate grid layout
        glyph_count = len(self.glyphs)
        cols = min(5, glyph_count)  # Max 5 glyphs per row
        rows = math.ceil(glyph_count / cols)
        
        cell_width = width / cols
        cell_height = height / rows
        
        # Draw background
        painter.fillRect(0, 0, width, height, QColor(245, 245, 245))
        
        # Draw each glyph
        for i, glyph in enumerate(self.glyphs):
            col = i % cols
            row = i // cols
            
            # Calculate cell position
            x = col * cell_width
            y = row * cell_height
            
            # Draw cell rectangle
            if glyph['id'] == self.selected_glyph:
                # Selected glyph has a highlight
                painter.fillRect(x+2, y+2, cell_width-4, cell_height-4, QColor(220, 230, 255))
                painter.setPen(QPen(QColor(70, 130, 180), 2))
                painter.drawRect(x+2, y+2, cell_width-4, cell_height-4)
            else:
                painter.fillRect(x+2, y+2, cell_width-4, cell_height-4, QColor(255, 255, 255))
                painter.setPen(QPen(QColor(200, 200, 200), 1))
                painter.drawRect(x+2, y+2, cell_width-4, cell_height-4)
            
            # Draw glyph in cell
            self.draw_glyph(painter, glyph, x, y, cell_width, cell_height)
            
            # Draw activation indicator (a bar at the bottom of the cell)
            activation = glyph['activation_level']
            bar_height = 5
            bar_y = y + cell_height - bar_height - 5
            
            # Background bar
            painter.fillRect(x+10, bar_y, cell_width-20, bar_height, QColor(220, 220, 220))
            
            # Activation level bar with color based on level
            if activation >= 0.8:
                color = QColor(100, 200, 100)  # Green for high activation
            elif activation >= 0.6:
                color = QColor(200, 200, 100)  # Yellow for medium activation
            else:
                color = QColor(200, 100, 100)  # Red for low activation
                
            painter.fillRect(x+10, bar_y, int((cell_width-20) * activation), bar_height, color)
            
            # Draw glyph name
            painter.setPen(QColor(50, 50, 50))
            painter.setFont(QFont("Arial", 8))
            name_rect = painter.boundingRect(int(x), int(y+cell_height-25), 
                                         int(cell_width), 20, 
                                         Qt.AlignCenter, glyph['name'])
            painter.drawText(name_rect, Qt.AlignCenter, glyph['name'])
    
    def draw_glyph(self, painter, glyph, x, y, width, height):
        """Draw a single glyph based on its pattern"""
        # Set the coordinate system for the glyph
        # Map glyph coordinates (0-1) to cell coordinates
        def map_x(gx):
            return x + 10 + (width - 20) * gx
            
        def map_y(gy):
            return y + 10 + (height - 40) * gy
        
        # Set up paint style
        painter.setPen(QPen(QColor(40, 40, 40), 2))
        
        # Draw each element in the pattern
        for element in glyph['pattern']:
            if element['type'] == 'circle':
                center_x = map_x(element['x'])
                center_y = map_y(element['y'])
                radius = min(width, height - 20) * element['radius']
                painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
                
            elif element['type'] == 'line':
                painter.drawLine(
                    map_x(element['x1']), map_y(element['y1']),
                    map_x(element['x2']), map_y(element['y2'])
                )
                
            elif element['type'] == 'spiral':
                center_x = map_x(element['x'])
                center_y = map_y(element['y'])
                radius = min(width, height - 20) * element['radius']
                turns = element['turns']
                
                # Draw spiral
                points = []
                steps = 100
                for i in range(steps + 1):
                    t = i / steps * turns * 2 * math.pi
                    r = radius * (i / steps)
                    px = center_x + r * math.cos(t)
                    py = center_y + r * math.sin(t)
                    points.append((px, py))
                
                # Draw the spiral as connected lines
                for i in range(1, len(points)):
                    painter.drawLine(points[i-1][0], points[i-1][1], points[i][0], points[i][1])
    
    def mousePressEvent(self, event):
        """Handle mouse press events to select glyphs"""
        width = self.width()
        height = self.height()
        
        # Calculate grid layout
        glyph_count = len(self.glyphs)
        cols = min(5, glyph_count)
        rows = math.ceil(glyph_count / cols)
        
        cell_width = width / cols
        cell_height = height / rows
        
        # Determine which cell was clicked
        col = int(event.x() / cell_width)
        row = int(event.y() / cell_height)
        
        idx = row * cols + col
        if 0 <= idx < glyph_count:
            self.selected_glyph = self.glyphs[idx]['id']
            self.glyph_selected.emit(self.selected_glyph)
            self.update()


class ActivationHeatmapWidget(QWidget):
    """Widget to display a neural network activation heatmap for a glyph"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.activation_data = None
        self.setMinimumHeight(200)
        self.generate_mock_data()
    
    def generate_mock_data(self):
        """Generate mock activation data for demonstration"""
        # Create a 10x10 grid of activation values
        width, height = 10, 10
        self.activation_data = []
        
        # Generate some patterns
        for y in range(height):
            row = []
            for x in range(width):
                # Create patterns based on position
                if (x + y) % 3 == 0:
                    # Higher activation in a diagonal pattern
                    value = random.uniform(0.7, 0.95)
                elif x % 4 == 0 or y % 4 == 0:
                    # Medium activation in a grid pattern
                    value = random.uniform(0.4, 0.6)
                else:
                    # Low activation elsewhere
                    value = random.uniform(0.1, 0.3)
                row.append(value)
            self.activation_data.append(row)
    
    def update_data(self, glyph_id):
        """Update the activation data based on the selected glyph"""
        # In a real application, this would fetch actual neural activation data
        # For demonstration, we'll generate new random data with patterns
        self.generate_mock_data()
        
        # Add some variation based on glyph_id to simulate different activations
        if glyph_id:
            for y in range(len(self.activation_data)):
                for x in range(len(self.activation_data[0])):
                    # Modify the existing data based on glyph_id
                    modifier = (x + y + glyph_id) % 5 / 10.0
                    self.activation_data[y][x] = max(0, min(1, self.activation_data[y][x] + modifier))
        
        self.update()
    
    def paintEvent(self, event):
        """Paint the activation heatmap"""
        if not self.activation_data:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Get grid dimensions
        grid_height = len(self.activation_data)
        grid_width = len(self.activation_data[0])
        
        # Calculate cell size
        cell_width = width / grid_width
        cell_height = height / grid_height
        
        # Draw each cell in the grid
        for y in range(grid_height):
            for x in range(grid_width):
                value = self.activation_data[y][x]
                
                # Color mapping from activation value (0-1) to color
                # Blue (cold) to red (hot) gradient
                r = int(255 * value)
                g = int(255 * (1.0 - abs(value - 0.5) * 2))
                b = int(255 * (1.0 - value))
                
                color = QColor(r, g, b)
                painter.fillRect(
                    x * cell_width, 
                    y * cell_height, 
                    cell_width, 
                    cell_height, 
                    color
                )
                
                # Draw grid lines
                painter.setPen(QPen(QColor(50, 50, 50, 100), 0.5))
                painter.drawRect(
                    x * cell_width, 
                    y * cell_height, 
                    cell_width, 
                    cell_height
                )
                
                # If cell is large enough, draw the activation value
                if cell_width > 30 and cell_height > 20:
                    painter.setPen(QColor(255, 255, 255))
                    painter.setFont(QFont("Arial", 8))
                    painter.drawText(
                        x * cell_width + 2, 
                        y * cell_height + 2, 
                        cell_width - 4, 
                        cell_height - 4, 
                        Qt.AlignCenter, 
                        f"{value:.2f}"
                    )


class EffectivenessBarChartWidget(QWidget):
    """Widget to display effectiveness metrics for glyphs"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.metrics = {
            "Recognition Rate": 0.82,
            "Training Speed": 0.65,
            "Pattern Completion": 0.74,
            "Cross-Validation": 0.89,
            "Memory Usage": 0.77
        }
        self.selected_glyph = None
        self.setMinimumHeight(200)
    
    def update_glyph(self, glyph_id):
        """Update metrics based on selected glyph"""
        self.selected_glyph = glyph_id
        
        # In a real app, we would fetch actual metrics
        # For demo, we'll generate variations based on glyph_id
        if glyph_id:
            seed = glyph_id * 1234  # Use glyph_id as a seed for deterministic randomness
            random.seed(seed)
            
            # Update each metric with some variation
            for key in self.metrics:
                base = self.metrics[key]
                # Vary by ±15%
                variation = random.uniform(-0.15, 0.15)
                self.metrics[key] = max(0, min(1, base + variation))
                
        self.update()
    
    def paintEvent(self, event):
        """Paint the bar chart"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Background
        painter.fillRect(0, 0, width, height, QColor(245, 245, 245))
        
        # Calculate bar dimensions
        metrics_count = len(self.metrics)
        bar_height = (height - 20) / metrics_count
        label_width = 150  # Width for metric labels
        
        # Draw each metric bar
        y = 10
        for metric, value in self.metrics.items():
            # Draw metric label
            painter.setPen(QColor(50, 50, 50))
            painter.setFont(QFont("Arial", 9))
            painter.drawText(10, y, label_width, bar_height, Qt.AlignVCenter, metric)
            
            # Draw bar background
            bar_bg_rect = QFrame()
            bar_x = label_width + 10
            bar_width = width - bar_x - 10
            painter.fillRect(bar_x, y + 5, bar_width, bar_height - 10, QColor(220, 220, 220))
            
            # Draw value bar
            if value >= 0.8:
                color = QColor(100, 200, 100)  # Green for high values
            elif value >= 0.6:
                color = QColor(200, 200, 100)  # Yellow for medium values
            else:
                color = QColor(200, 100, 100)  # Red for low values
                
            painter.fillRect(
                bar_x, 
                y + 5, 
                int(bar_width * value), 
                bar_height - 10, 
                color
            )
            
            # Draw value text
            painter.setPen(QColor(50, 50, 50))
            painter.drawText(
                bar_x + bar_width + 5, 
                y, 
                40, 
                bar_height, 
                Qt.AlignVCenter, 
                f"{value:.2f}"
            )
            
            y += bar_height


class GlyphAnalysisPanel(QWidget):
    """Panel for analyzing glyph effectiveness and neural connections"""
    
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
        
        title = QLabel("Glyph Analysis")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2C3E50;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Add selector for network model
        model_label = QLabel("Neural Network:")
        header_layout.addWidget(model_label)
        
        self.model_selector = QComboBox()
        self.model_selector.addItems([
            "Standard CNN (trained)",
            "Recurrent GRU Network",
            "Custom Hyperdimensional Model",
            "Transfer Learning Model",
            "Ensemble Model"
        ])
        header_layout.addWidget(self.model_selector)
        
        # Add refresh button
        refresh_btn = QPushButton("Refresh Analysis")
        refresh_btn.clicked.connect(self.refresh_analysis)
        header_layout.addWidget(refresh_btn)
        
        main_layout.addWidget(header)
        
        # Main content area with tabs
        self.tabs = QTabWidget()
        
        # Tab 1: Glyph Overview
        glyph_tab = QWidget()
        glyph_layout = QVBoxLayout(glyph_tab)
        
        # Glyph visualization grid
        self.glyph_visualizer = GlyphVisualizerWidget()
        self.glyph_visualizer.glyph_selected.connect(self.glyph_selected)
        glyph_layout.addWidget(self.glyph_visualizer)
        
        # Splitter for details
        details_splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Activation heatmap
        heatmap_group = QGroupBox("Neural Activation Heatmap")
        heatmap_layout = QVBoxLayout(heatmap_group)
        
        self.heatmap = ActivationHeatmapWidget()
        heatmap_layout.addWidget(self.heatmap)
        
        details_splitter.addWidget(heatmap_group)
        
        # Right side: Effectiveness metrics
        metrics_group = QGroupBox("Effectiveness Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        
        self.effectiveness_chart = EffectivenessBarChartWidget()
        metrics_layout.addWidget(self.effectiveness_chart)
        
        details_splitter.addWidget(metrics_group)
        
        # Set initial splitter sizes
        details_splitter.setSizes([500, 500])
        glyph_layout.addWidget(details_splitter)
        
        self.tabs.addTab(glyph_tab, "Glyph Effectiveness")
        
        # Tab 2: Comparative Analysis
        compare_tab = QWidget()
        compare_layout = QVBoxLayout(compare_tab)
        
        compare_label = QLabel("Compare multiple glyphs and their effectiveness across different metrics.")
        compare_layout.addWidget(compare_label)
        
        # Placeholder for comparative analysis
        compare_layout.addWidget(QLabel("Comparative analysis visualization will be implemented here."))
        
        self.tabs.addTab(compare_tab, "Comparative Analysis")
        
        # Tab 3: Historical Performance
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        
        history_label = QLabel("Track glyph performance improvements over time.")
        history_layout.addWidget(history_label)
        
        # Placeholder for historical performance
        history_layout.addWidget(QLabel("Historical performance tracking will be implemented here."))
        
        self.tabs.addTab(history_tab, "Historical Performance")
        
        main_layout.addWidget(self.tabs)
        
        # Footer with status
        footer = QFrame()
        footer.setFrameShape(QFrame.StyledPanel)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(5, 2, 5, 2)
        
        self.status_label = QLabel("Ready. Select a glyph to analyze its neural activation and effectiveness.")
        footer_layout.addWidget(self.status_label)
        
        main_layout.addWidget(footer)
    
    def glyph_selected(self, glyph_id):
        """Handle glyph selection"""
        # Update the activation heatmap
        self.heatmap.update_data(glyph_id)
        
        # Update the effectiveness metrics
        self.effectiveness_chart.update_glyph(glyph_id)
        
        # Update status
        glyph_name = next((g['name'] for g in self.glyph_visualizer.glyphs if g['id'] == glyph_id), "Unknown")
        self.status_label.setText(f"Analyzing glyph: {glyph_name}")
    
    def refresh_analysis(self):
        """Refresh the analysis with updated data"""
        # In a real application, this would fetch fresh data from the neural network
        # For demonstration, we'll just regenerate the mock data
        
        # Regenerate the glyph visualizer data
        self.glyph_visualizer.init_sample_glyphs()
        self.glyph_visualizer.update()
        
        # Reset the heatmap and effectiveness chart
        self.heatmap.generate_mock_data()
        self.heatmap.update()
        
        selected_model = self.model_selector.currentText()
        self.status_label.setText(f"Analysis refreshed using model: {selected_model}") 