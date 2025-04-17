from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                              QComboBox, QTabWidget, QFrame, QScrollArea, QSplitter,
                              QGridLayout, QGroupBox, QCheckBox, QTableWidget, 
                              QTableWidgetItem, QHeaderView, QSpinBox)
from PySide6.QtCore import Qt, Signal, QTimer, QPointF, QRectF, QSize
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QFont

class PerformanceChart(QWidget):
    """Widget for visualizing neural network performance metrics"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.metrics = []  # List of (accuracy, loss, iteration) tuples
        self.show_accuracy = True
        self.show_loss = True
        self.setStyleSheet("background-color: #FAFAFA;")
        
    def add_metric(self, accuracy, loss, iteration):
        """Add a new performance metric"""
        self.metrics.append((accuracy, loss, iteration))
        # Keep only the last 100 metrics for better visualization
        if len(self.metrics) > 100:
            self.metrics.pop(0)
        self.update()
        
    def paintEvent(self, event):
        if not self.metrics:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        padding = 30  # Space for labels and axes
        
        # Draw coordinate system
        painter.setPen(QPen(QColor("#333333"), 1))
        
        # X-axis
        painter.drawLine(padding, height - padding, width - padding, height - padding)
        
        # Y-axis
        painter.drawLine(padding, padding, padding, height - padding)
        
        # Draw labels
        painter.setFont(QFont("Arial", 8))
        
        # Y-axis label (0.0 to 1.0)
        painter.drawText(5, height - padding, "0.0")
        painter.drawText(5, padding + 10, "1.0")
        
        # X-axis marks (iterations)
        if self.metrics:
            start_iter = self.metrics[0][2]
            end_iter = self.metrics[-1][2]
            
            # Draw some X-axis labels
            for i in range(5):
                x_pos = padding + i * (width - 2 * padding) / 4
                iter_val = start_iter + i * (end_iter - start_iter) / 4
                painter.drawText(x_pos - 15, height - 10, f"{int(iter_val)}")
                
                # Draw grid line
                painter.setPen(QPen(QColor("#DDDDDD"), 1, Qt.DashLine))
                painter.drawLine(x_pos, padding, x_pos, height - padding)
            
            # Reset pen
            painter.setPen(QPen(QColor("#333333"), 1))
                
        # Draw grid lines for Y-axis
        for i in range(5):
            y_pos = padding + i * (height - 2 * padding) / 4
            painter.setPen(QPen(QColor("#DDDDDD"), 1, Qt.DashLine))
            painter.drawLine(padding, y_pos, width - padding, y_pos)
        
        # Draw metrics
        if self.metrics:
            chart_width = width - 2 * padding
            chart_height = height - 2 * padding
            
            if self.show_accuracy:
                # Draw accuracy curve
                painter.setPen(QPen(QColor("#2980B9"), 2))
                
                path = QPainterPath()
                first_x = padding + (self.metrics[0][2] - start_iter) * chart_width / (end_iter - start_iter)
                first_y = height - padding - self.metrics[0][0] * chart_height
                path.moveTo(first_x, first_y)
                
                for accuracy, _, iteration in self.metrics:
                    x = padding + (iteration - start_iter) * chart_width / (end_iter - start_iter)
                    y = height - padding - accuracy * chart_height
                    path.lineTo(x, y)
                    
                painter.drawPath(path)
                
                # Draw legend for accuracy
                painter.fillRect(width - 100, 10, 10, 10, QColor("#2980B9"))
                painter.drawText(width - 80, 20, "Accuracy")
            
            if self.show_loss:
                # Draw loss curve (normalized to 0-1 range)
                painter.setPen(QPen(QColor("#E74C3C"), 2))
                
                # Find max loss for normalization
                max_loss = max(loss for _, loss, _ in self.metrics)
                
                path = QPainterPath()
                first_x = padding + (self.metrics[0][2] - start_iter) * chart_width / (end_iter - start_iter)
                first_y = height - padding - (self.metrics[0][1] / max_loss) * chart_height
                path.moveTo(first_x, first_y)
                
                for _, loss, iteration in self.metrics:
                    x = padding + (iteration - start_iter) * chart_width / (end_iter - start_iter)
                    y = height - padding - (loss / max_loss) * chart_height
                    path.lineTo(x, y)
                    
                painter.drawPath(path)
                
                # Draw legend for loss
                painter.fillRect(width - 100, 30, 10, 10, QColor("#E74C3C"))
                painter.drawText(width - 80, 40, "Loss")

class GlyphCorrelationMatrix(QWidget):
    """Widget for visualizing correlations between glyphs and neural network nodes"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.correlations = {}  # Dictionary of (layer, neuron, glyph) -> correlation value
        self.setStyleSheet("background-color: #FAFAFA;")
        
        self.layer_count = 4  # Number of layers to display
        self.neuron_count = 5  # Neurons per layer to display
        self.glyph_count = 5   # Number of glyphs to correlate with
        
    def set_correlations(self, correlations):
        """Set the correlation values to display"""
        self.correlations = correlations
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Calculate cell size
        total_rows = self.layer_count * self.neuron_count
        total_cols = self.glyph_count
        
        horizontal_padding = 80  # Space for y-axis labels
        vertical_padding = 60    # Space for x-axis labels
        
        cell_width = (width - horizontal_padding) / total_cols
        cell_height = (height - vertical_padding) / total_rows
        
        # Draw grid
        for row in range(total_rows):
            layer = row // self.neuron_count
            neuron = row % self.neuron_count
            
            for col in range(total_cols):
                glyph = col
                
                # Calculate correlation strength (mock data if not available)
                correlation_key = (layer, neuron, glyph)
                correlation = self.correlations.get(correlation_key, 0.0)
                
                # Map correlation to color (blue for positive, red for negative)
                if correlation >= 0:
                    color = QColor(0, 0, int(255 * min(1.0, correlation)))
                else:
                    color = QColor(int(255 * min(1.0, -correlation)), 0, 0)
                
                # Draw cell
                x = horizontal_padding + col * cell_width
                y = row * cell_height
                
                painter.fillRect(x, y, cell_width, cell_height, color)
                painter.setPen(QPen(QColor("#333333"), 1))
                painter.drawRect(x, y, cell_width, cell_height)
                
        # Draw labels
        painter.setFont(QFont("Arial", 8))
        
        # X-axis labels (Glyphs)
        for col in range(total_cols):
            x = horizontal_padding + col * cell_width + cell_width / 2
            y = height - vertical_padding / 2
            painter.drawText(QRectF(x - 20, y - 10, 40, 20), Qt.AlignCenter, f"Glyph {col+1}")
            
        # Y-axis labels (Layers and Neurons)
        for row in range(total_rows):
            layer = row // self.neuron_count
            neuron = row % self.neuron_count
            
            x = horizontal_padding / 2
            y = row * cell_height + cell_height / 2
            
            if neuron == 0:
                # Draw layer label for the first neuron in each layer
                painter.setFont(QFont("Arial", 8, QFont.Bold))
                painter.drawText(QRectF(10, y - 30, horizontal_padding - 20, 20), 
                                Qt.AlignRight, f"Layer {layer+1}")
                painter.setFont(QFont("Arial", 8))
            
            painter.drawText(QRectF(10, y - 10, horizontal_padding - 20, 20), 
                            Qt.AlignRight, f"N{neuron+1}")
            
        # Draw legend
        legend_width = 15
        legend_height = 100
        legend_x = width - 50
        legend_y = 50
        
        # Draw legend gradient
        for i in range(legend_height):
            # Map position to color
            position = 1.0 - i / legend_height  # 1.0 at top, 0.0 at bottom
            
            if position >= 0.5:  # Positive correlation (blue)
                normalized = (position - 0.5) * 2  # Map 0.5-1.0 to 0.0-1.0
                color = QColor(0, 0, int(255 * normalized))
            else:  # Negative correlation (red)
                normalized = (0.5 - position) * 2  # Map 0.0-0.5 to 0.0-1.0
                color = QColor(int(255 * normalized), 0, 0)
                
            painter.fillRect(legend_x, legend_y + i, legend_width, 1, color)
            
        # Draw legend labels
        painter.drawText(legend_x + legend_width + 5, legend_y, "1.0")
        painter.drawText(legend_x + legend_width + 5, legend_y + legend_height / 2, "0.0")
        painter.drawText(legend_x + legend_width + 5, legend_y + legend_height, "-1.0")
        
        # Draw legend border
        painter.setPen(QPen(QColor("#333333"), 1))
        painter.drawRect(legend_x, legend_y, legend_width, legend_height)
        painter.drawText(legend_x - 5, legend_y - 15, "Correlation")

class ActivationDistributionWidget(QWidget):
    """Widget for visualizing the activation distribution of neural network nodes"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.layer_activations = {}  # Dictionary of layer_id -> list of activation values
        self.setStyleSheet("background-color: #FAFAFA;")
        
    def set_activations(self, layer_activations):
        """Set the activation values to display"""
        self.layer_activations = layer_activations
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        padding = 30  # Space for labels and axes
        
        # Draw coordinate system
        painter.setPen(QPen(QColor("#333333"), 1))
        
        # X-axis (activation value from 0.0 to 1.0)
        painter.drawLine(padding, height - padding, width - padding, height - padding)
        
        # Y-axis (frequency)
        painter.drawLine(padding, padding, padding, height - padding)
        
        # Draw labels
        painter.setFont(QFont("Arial", 8))
        
        # X-axis labels
        painter.drawText(padding - 10, height - 10, "0.0")
        painter.drawText(width - padding - 10, height - 10, "1.0")
        painter.drawText(width / 2 - 10, height - 10, "0.5")
        
        # Draw layer histograms
        bin_count = 10  # Number of histogram bins
        bin_width = (width - 2 * padding) / bin_count
        
        if not self.layer_activations:
            return
            
        # Find max frequency across all layers for scaling
        max_freq = 0
        histogram_data = {}
        
        for layer_id, activations in self.layer_activations.items():
            # Create histogram data
            histogram = [0] * bin_count
            for activation in activations:
                bin_index = min(bin_count - 1, int(activation * bin_count))
                histogram[bin_index] += 1
                
            histogram_data[layer_id] = histogram
            max_freq = max(max_freq, max(histogram))
            
        # Draw histograms for each layer with different colors
        layer_colors = [
            QColor("#3498DB"),  # Blue
            QColor("#2ECC71"),  # Green
            QColor("#E74C3C"),  # Red
            QColor("#F39C12")   # Orange
        ]
        
        if max_freq == 0:
            return
            
        for layer_id, histogram in histogram_data.items():
            color = layer_colors[layer_id % len(layer_colors)]
            painter.setPen(QPen(color, 2))
            
            # Draw histogram bars
            for bin_index, freq in enumerate(histogram):
                x = padding + bin_index * bin_width
                bar_height = (freq / max_freq) * (height - 2 * padding)
                painter.drawLine(
                    x + bin_width / 2, 
                    height - padding, 
                    x + bin_width / 2, 
                    height - padding - bar_height
                )
                
            # Draw layer in legend
            legend_y = 20 + layer_id * 20
            painter.fillRect(width - 100, legend_y, 10, 10, color)
            painter.drawText(width - 80, legend_y + 10, f"Layer {layer_id+1}")

class AnalysisPanel(QWidget):
    """Panel for analyzing neural network performance and glyph effectiveness"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        self.setup_mock_data()
        
    def initUI(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 10)
        
        title = QLabel("Neural Network Analysis")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2C3E50;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Network selector
        network_label = QLabel("Network:")
        header_layout.addWidget(network_label)
        
        self.network_selector = QComboBox()
        self.network_selector.addItems(["Primary Network", "Secondary Network"])
        self.network_selector.setMinimumWidth(150)
        header_layout.addWidget(self.network_selector)
        
        # Time range selector
        time_label = QLabel("Time Range:")
        header_layout.addWidget(time_label)
        
        self.time_selector = QComboBox()
        self.time_selector.addItems(["Last Hour", "Last Day", "Last Week", "All Time"])
        header_layout.addWidget(self.time_selector)
        
        # Refresh button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_data)
        header_layout.addWidget(self.refresh_btn)
        
        main_layout.addWidget(header)
        
        # Main content area with tabs
        self.tabs = QTabWidget()
        
        # Performance Tab
        performance_tab = QWidget()
        performance_layout = QVBoxLayout(performance_tab)
        
        # Performance chart
        perf_header = QLabel("Training Performance")
        perf_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #2C3E50;")
        performance_layout.addWidget(perf_header)
        
        self.performance_chart = PerformanceChart()
        performance_layout.addWidget(self.performance_chart, 2)
        
        # Chart controls
        chart_controls = QWidget()
        chart_controls_layout = QHBoxLayout(chart_controls)
        chart_controls_layout.setContentsMargins(0, 0, 0, 0)
        
        self.show_accuracy_cb = QCheckBox("Show Accuracy")
        self.show_accuracy_cb.setChecked(True)
        self.show_accuracy_cb.stateChanged.connect(self.update_chart_visibility)
        chart_controls_layout.addWidget(self.show_accuracy_cb)
        
        self.show_loss_cb = QCheckBox("Show Loss")
        self.show_loss_cb.setChecked(True)
        self.show_loss_cb.stateChanged.connect(self.update_chart_visibility)
        chart_controls_layout.addWidget(self.show_loss_cb)
        
        chart_controls_layout.addStretch()
        
        performance_layout.addWidget(chart_controls)
        
        # Activation distribution
        activation_header = QLabel("Activation Distribution")
        activation_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #2C3E50;")
        performance_layout.addWidget(activation_header)
        
        self.activation_distribution = ActivationDistributionWidget()
        performance_layout.addWidget(self.activation_distribution, 1)
        
        self.tabs.addTab(performance_tab, "Performance")
        
        # Correlation Tab
        correlation_tab = QWidget()
        correlation_layout = QVBoxLayout(correlation_tab)
        
        correlation_header = QLabel("Neuron-Glyph Correlation Analysis")
        correlation_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #2C3E50;")
        correlation_layout.addWidget(correlation_header)
        
        self.correlation_matrix = GlyphCorrelationMatrix()
        correlation_layout.addWidget(self.correlation_matrix, 1)
        
        # Correlation controls
        corr_controls = QWidget()
        corr_controls_layout = QHBoxLayout(corr_controls)
        corr_controls_layout.setContentsMargins(0, 0, 0, 0)
        
        corr_controls_layout.addWidget(QLabel("Layers:"))
        self.layer_spin = QSpinBox()
        self.layer_spin.setRange(1, 10)
        self.layer_spin.setValue(4)
        self.layer_spin.valueChanged.connect(self.update_correlation_settings)
        corr_controls_layout.addWidget(self.layer_spin)
        
        corr_controls_layout.addWidget(QLabel("Neurons per layer:"))
        self.neuron_spin = QSpinBox()
        self.neuron_spin.setRange(1, 20)
        self.neuron_spin.setValue(5)
        self.neuron_spin.valueChanged.connect(self.update_correlation_settings)
        corr_controls_layout.addWidget(self.neuron_spin)
        
        corr_controls_layout.addWidget(QLabel("Glyphs:"))
        self.glyph_spin = QSpinBox()
        self.glyph_spin.setRange(1, 20)
        self.glyph_spin.setValue(5)
        self.glyph_spin.valueChanged.connect(self.update_correlation_settings)
        corr_controls_layout.addWidget(self.glyph_spin)
        
        corr_controls_layout.addStretch()
        
        correlation_layout.addWidget(corr_controls)
        
        # Add correlation explanation text
        explanation = QLabel(
            "This matrix shows the correlation between neural network activations and glyph properties. "
            "Blue cells indicate positive correlation, red cells indicate negative correlation, "
            "and intensity represents the strength of the correlation."
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color: #555; font-style: italic;")
        correlation_layout.addWidget(explanation)
        
        self.tabs.addTab(correlation_tab, "Correlations")
        
        # Insights Tab
        insights_tab = QWidget()
        insights_layout = QVBoxLayout(insights_tab)
        
        insights_header = QLabel("Network Insights")
        insights_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #2C3E50;")
        insights_layout.addWidget(insights_header)
        
        # Insights table
        self.insights_table = QTableWidget(0, 3)
        self.insights_table.setHorizontalHeaderLabels(["Component", "Insight", "Impact"])
        self.insights_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.insights_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.insights_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        insights_layout.addWidget(self.insights_table)
        
        self.tabs.addTab(insights_tab, "Insights")
        
        main_layout.addWidget(self.tabs, 1)
        
        # Footer with status
        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #555;")
        footer_layout.addWidget(self.status_label)
        
        footer_layout.addStretch()
        
        main_layout.addWidget(footer)
        
        # Set up timer for updates
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.refresh_data)
        self.update_timer.start(5000)  # Update every 5000ms
        
    def setup_mock_data(self):
        """Set up mock data for visualization"""
        import random
        
        # Generate training metrics data (accuracy, loss, iteration)
        for i in range(100):
            iteration = i
            # Simulating improving accuracy and decreasing loss over time
            accuracy = 0.5 + 0.5 * (1 - 1 / (1 + 0.05 * i))
            loss = 0.5 * (1 / (1 + 0.05 * i))
            
            # Add some noise
            accuracy += random.uniform(-0.05, 0.05)
            accuracy = max(0, min(1, accuracy))
            
            loss += random.uniform(-0.05, 0.05)
            loss = max(0, loss)
            
            self.performance_chart.add_metric(accuracy, loss, iteration)
            
        # Generate activation distribution data
        layer_activations = {}
        for layer in range(4):
            # Different activation distributions for different layers
            if layer == 0:  # Input layer - uniform distribution
                activations = [random.uniform(0, 1) for _ in range(100)]
            elif layer == 1:  # Hidden layer 1 - normal distribution around 0.3
                activations = [max(0, min(1, random.normalvariate(0.3, 0.2))) for _ in range(100)]
            elif layer == 2:  # Hidden layer 2 - normal distribution around 0.7
                activations = [max(0, min(1, random.normalvariate(0.7, 0.2))) for _ in range(100)]
            else:  # Output layer - bimodal distribution
                activations = []
                for _ in range(100):
                    if random.random() < 0.5:
                        activations.append(max(0, min(1, random.normalvariate(0.2, 0.1))))
                    else:
                        activations.append(max(0, min(1, random.normalvariate(0.8, 0.1))))
                        
            layer_activations[layer] = activations
            
        self.activation_distribution.set_activations(layer_activations)
        
        # Generate correlation data
        correlations = {}
        for layer in range(4):
            for neuron in range(5):
                for glyph in range(5):
                    # Create some patterns in the correlation data
                    if layer == glyph:  # Strong positive correlation when layer and glyph match
                        correlation = random.uniform(0.7, 0.9)
                    elif (layer + neuron) % 5 == glyph:  # Moderate correlation for some patterns
                        correlation = random.uniform(0.3, 0.6)
                    elif (layer * neuron) % 5 == glyph:  # Negative correlation for some patterns
                        correlation = random.uniform(-0.8, -0.2)
                    else:  # Random weak correlation otherwise
                        correlation = random.uniform(-0.2, 0.2)
                        
                    correlations[(layer, neuron, glyph)] = correlation
                    
        self.correlation_matrix.set_correlations(correlations)
        
        # Generate insights data
        insights = [
            ("Neurons", "High activations in the hidden layer 2 correspond to specific glyph patterns", "Medium"),
            ("Connections", "Strong weights between input layer and first hidden layer suggest effective feature extraction", "High"),
            ("Training", "Learning rate decay may improve stability in later epochs", "Medium"),
            ("Glyphs", "Glyph 3 shows strongest correlation with output layer neurons", "High"),
            ("Performance", "Validation accuracy plateaus after 50 epochs", "Medium"),
            ("Architecture", "Consider adding another hidden layer for improved pattern recognition", "Low")
        ]
        
        self.insights_table.setRowCount(len(insights))
        for row, (component, insight, impact) in enumerate(insights):
            self.insights_table.setItem(row, 0, QTableWidgetItem(component))
            self.insights_table.setItem(row, 1, QTableWidgetItem(insight))
            self.insights_table.setItem(row, 2, QTableWidgetItem(impact))
            
            # Color-code impact cells
            impact_item = self.insights_table.item(row, 2)
            if impact == "High":
                impact_item.setBackground(QColor(255, 200, 200))
            elif impact == "Medium":
                impact_item.setBackground(QColor(255, 255, 200))
            else:
                impact_item.setBackground(QColor(200, 255, 200))
    
    def refresh_data(self):
        """Refresh the data displayed in the panel"""
        import random
        
        # Update performance metrics with new data point
        last_metrics = self.performance_chart.metrics[-1] if self.performance_chart.metrics else (0.5, 0.5, 0)
        last_accuracy, last_loss, last_iteration = last_metrics
        
        # Calculate new metrics (simulating ongoing training)
        new_iteration = last_iteration + 1
        accuracy_change = random.uniform(-0.02, 0.03)  # Slight improvement trend
        new_accuracy = max(0, min(1, last_accuracy + accuracy_change))
        
        loss_change = random.uniform(-0.03, 0.02)  # Slight decrease trend
        new_loss = max(0, last_loss + loss_change)
        
        self.performance_chart.add_metric(new_accuracy, new_loss, new_iteration)
        
        # Update status
        self.status_label.setText(f"Last updated: {QTimer.currentTime().toString('hh:mm:ss')}")
        
    def update_chart_visibility(self):
        """Update which lines are visible on the performance chart"""
        self.performance_chart.show_accuracy = self.show_accuracy_cb.isChecked()
        self.performance_chart.show_loss = self.show_loss_cb.isChecked()
        self.performance_chart.update()
        
    def update_correlation_settings(self):
        """Update correlation matrix settings"""
        self.correlation_matrix.layer_count = self.layer_spin.value()
        self.correlation_matrix.neuron_count = self.neuron_spin.value()
        self.correlation_matrix.glyph_count = self.glyph_spin.value()
        self.correlation_matrix.update() 