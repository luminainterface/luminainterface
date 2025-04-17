from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                          QPushButton, QSlider, QComboBox, QToolButton,
                          QFrame, QSplitter, QGroupBox, QGridLayout)
from PyQt5.QtGui import QIcon, QFont, QColor, QPainter, QPen, QBrush, QPainterPath
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPointF, QSize

class NeuronVisualization(QWidget):
    """Widget for visualizing a single neuron with its connections"""
    
    neuron_clicked = pyqtSignal(int, int)  # layer_idx, neuron_idx
    
    def __init__(self, layer_idx, neuron_idx, activation=0.0, weights=None, parent=None):
        super().__init__(parent)
        self.layer_idx = layer_idx
        self.neuron_idx = neuron_idx
        self.activation = activation  # Value between 0.0 and 1.0
        self.weights = weights or []  # Connection weights to neurons in previous layer
        self.hovered = False
        self.selected = False
        
        self.setMinimumSize(50, 50)
        self.setMaximumSize(80, 80)
        self.setMouseTracking(True)
        
    def paintEvent(self, event):
        """Custom paint event for neuron visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate center and radius
        rect = self.rect()
        center_x = rect.width() / 2
        center_y = rect.height() / 2
        radius = min(center_x, center_y) - 5
        
        # Determine colors based on activation and state
        if self.selected:
            border_color = QColor(46, 204, 113)  # Green when selected
            border_width = 3
        elif self.hovered:
            border_color = QColor(52, 152, 219)  # Blue when hovered
            border_width = 2
        else:
            border_color = QColor(44, 62, 80)  # Dark blue-gray normally
            border_width = 1
        
        # Map activation to color (red for high, blue for low)
        if self.activation > 0.5:
            # Interpolate between yellow and red
            r = 255
            g = int(255 - (self.activation - 0.5) * 2 * 255)
            b = 0
        else:
            # Interpolate between blue and yellow
            r = int(self.activation * 2 * 255)
            g = int(self.activation * 2 * 255)
            b = int(255 - self.activation * 2 * 255)
        
        fill_color = QColor(r, g, b, 180)
        
        # Draw neuron as circle
        painter.setPen(QPen(border_color, border_width))
        painter.setBrush(QBrush(fill_color))
        painter.drawEllipse(QPointF(center_x, center_y), radius, radius)
        
        # Draw activation value in center
        painter.setPen(Qt.white)
        painter.setFont(QFont("Arial", 8, QFont.Bold))
        text_rect = QRectF(center_x - radius, center_y - 10, radius * 2, 20)
        painter.drawText(text_rect, Qt.AlignCenter, f"{self.activation:.2f}")
        
    def enterEvent(self, event):
        """Mouse enter event"""
        self.hovered = True
        self.update()
        
    def leaveEvent(self, event):
        """Mouse leave event"""
        self.hovered = False
        self.update()
        
    def mousePressEvent(self, event):
        """Mouse press event"""
        if event.button() == Qt.LeftButton:
            self.selected = not self.selected
            self.update()
            self.neuron_clicked.emit(self.layer_idx, self.neuron_idx)

class LayerVisualization(QWidget):
    """Widget for visualizing a layer of neurons"""
    
    layer_clicked = pyqtSignal(int)  # layer_idx
    neuron_clicked = pyqtSignal(int, int)  # layer_idx, neuron_idx
    
    def __init__(self, layer_idx, layer_size, layer_type="hidden", parent=None):
        super().__init__(parent)
        self.layer_idx = layer_idx
        self.layer_size = layer_size
        self.layer_type = layer_type  # "input", "hidden", or "output"
        self.neurons = []
        
        self.initUI()
        
    def initUI(self):
        """Initialize the layer visualization UI"""
        layout = QVBoxLayout(self)
        
        # Layer label
        if self.layer_type == "input":
            layer_name = "Input Layer"
        elif self.layer_type == "output":
            layer_name = "Output Layer"
        else:
            layer_name = f"Hidden Layer {self.layer_idx}"
            
        label = QLabel(layer_name)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("""
            font-weight: bold;
            color: #3498DB;
        """)
        layout.addWidget(label)
        
        # Neurons container
        neurons_widget = QWidget()
        self.neurons_layout = QVBoxLayout(neurons_widget)
        self.neurons_layout.setSpacing(5)
        self.neurons_layout.setAlignment(Qt.AlignCenter)
        
        # Create neurons
        for i in range(self.layer_size):
            neuron = NeuronVisualization(self.layer_idx, i, activation=0.5)
            neuron.neuron_clicked.connect(self.neuron_clicked)
            self.neurons.append(neuron)
            self.neurons_layout.addWidget(neuron, alignment=Qt.AlignCenter)
            
        layout.addWidget(neurons_widget)
        self.setLayout(layout)

class ConnectionsVisualization(QWidget):
    """Widget for visualizing connections between layers"""
    
    def __init__(self, from_layer_size, to_layer_size, weights=None, parent=None):
        super().__init__(parent)
        self.from_layer_size = from_layer_size
        self.to_layer_size = to_layer_size
        # 2D list of weights, weights[from_idx][to_idx]
        self.weights = weights or [[0.1 for _ in range(to_layer_size)] for _ in range(from_layer_size)]
        
    def paintEvent(self, event):
        """Custom paint event for connections visualization"""
        if not self.isVisible():
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate size info
        rect = self.rect()
        width = rect.width()
        height = rect.height()
        
        # Space for each neuron in each layer
        from_spacing = height / max(1, self.from_layer_size)
        to_spacing = height / max(1, self.to_layer_size)
        
        # Calculate starting y positions
        from_start_y = (height - (self.from_layer_size * from_spacing)) / 2
        to_start_y = (height - (self.to_layer_size * to_spacing)) / 2
        
        # Draw connections between each pair of neurons
        for i in range(self.from_layer_size):
            for j in range(self.to_layer_size):
                # Calculate endpoints
                from_y = from_start_y + i * from_spacing + from_spacing / 2
                to_y = to_start_y + j * to_spacing + to_spacing / 2
                
                # Get weight for this connection
                weight = self.weights[i][j]
                
                # Determine line color and thickness based on weight
                if weight > 0:
                    # Positive weight: Green with opacity based on strength
                    color = QColor(46, 204, 113, int(abs(weight) * 255))
                else:
                    # Negative weight: Red with opacity based on strength
                    color = QColor(231, 76, 60, int(abs(weight) * 255))
                
                line_width = abs(weight) * 3
                
                # Draw line
                painter.setPen(QPen(color, line_width, Qt.SolidLine, Qt.RoundCap))
                painter.drawLine(0, from_y, width, to_y)

class NetworkVisualizationPanel(QWidget):
    """Panel for visualizing neural network architecture and activations"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layers = []  # List of layer sizes
        self.layer_types = []  # List of layer types
        self.layer_visualizations = []  # List of LayerVisualization widgets
        self.connection_visualizations = []  # List of ConnectionsVisualization widgets
        
        # Mock network configuration for demonstration
        self.layers = [4, 8, 6, 3]
        self.layer_types = ["input", "hidden", "hidden", "output"]
        
        self.initUI()
        
    def initUI(self):
        """Initialize the network visualization panel UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        
        # Header area
        header_layout = QHBoxLayout()
        
        # Title
        title = QLabel("Neural Network Visualization")
        title.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #3498DB;
            margin-bottom: 10px;
        """)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Network selector
        self.network_selector = QComboBox()
        self.network_selector.addItems(["Main Network", "Memory Encoder", "Language Processor"])
        self.network_selector.setStyleSheet("""
            background-color: #1E2C3A;
            color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
            min-width: 150px;
        """)
        controls_layout.addWidget(self.network_selector)
        
        # Detail level slider
        controls_layout.addWidget(QLabel("Detail:"))
        self.detail_slider = QSlider(Qt.Horizontal)
        self.detail_slider.setRange(1, 3)
        self.detail_slider.setValue(2)
        self.detail_slider.setMaximumWidth(100)
        self.detail_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #2C3E50;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498DB;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        controls_layout.addWidget(self.detail_slider)
        
        # Snapshot button
        self.snapshot_button = QPushButton("Take Snapshot")
        self.snapshot_button.setStyleSheet("""
            QPushButton {
                background-color: #2980B9;
                color: white;
                border-radius: 5px;
                padding: 8px 15px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3498DB;
            }
            QPushButton:pressed {
                background-color: #1C587F;
            }
        """)
        controls_layout.addWidget(self.snapshot_button)
        
        header_layout.addLayout(controls_layout)
        main_layout.addLayout(header_layout)
        
        # Main visualization area
        viz_container = QSplitter(Qt.Vertical)
        viz_container.setChildrenCollapsible(False)
        
        # Network architecture view
        self.network_view = QWidget()
        self.network_view.setMinimumHeight(300)
        
        network_layout = QHBoxLayout(self.network_view)
        network_layout.setSpacing(0)
        
        # Create layer visualizations and connections
        prev_layer_size = None
        for i, (layer_size, layer_type) in enumerate(zip(self.layers, self.layer_types)):
            # If not the first layer, add connections visualization
            if prev_layer_size is not None:
                connection_viz = ConnectionsVisualization(prev_layer_size, layer_size)
                self.connection_visualizations.append(connection_viz)
                network_layout.addWidget(connection_viz)
            
            # Add layer visualization
            layer_viz = LayerVisualization(i, layer_size, layer_type)
            layer_viz.neuron_clicked.connect(self.on_neuron_clicked)
            self.layer_visualizations.append(layer_viz)
            network_layout.addWidget(layer_viz)
            
            prev_layer_size = layer_size
        
        viz_container.addWidget(self.network_view)
        
        # Details panel
        self.details_panel = QGroupBox("Neuron Details")
        self.details_panel.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 1px solid #2C3E50;
                border-radius: 5px;
                margin-top: 10px;
                padding: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #3498DB;
            }
        """)
        
        details_layout = QGridLayout(self.details_panel)
        
        # Neuron label
        details_layout.addWidget(QLabel("Neuron:"), 0, 0)
        self.neuron_label = QLabel("None selected")
        self.neuron_label.setStyleSheet("color: #ECF0F1; font-weight: bold;")
        details_layout.addWidget(self.neuron_label, 0, 1)
        
        # Activation
        details_layout.addWidget(QLabel("Activation:"), 1, 0)
        self.activation_label = QLabel("0.00")
        self.activation_label.setStyleSheet("color: #ECF0F1;")
        details_layout.addWidget(self.activation_label, 1, 1)
        
        # Bias
        details_layout.addWidget(QLabel("Bias:"), 2, 0)
        self.bias_label = QLabel("0.00")
        self.bias_label.setStyleSheet("color: #ECF0F1;")
        details_layout.addWidget(self.bias_label, 2, 1)
        
        # Input weights
        details_layout.addWidget(QLabel("Input Weights:"), 3, 0, Qt.AlignTop)
        self.weights_label = QLabel("None")
        self.weights_label.setStyleSheet("color: #ECF0F1;")
        details_layout.addWidget(self.weights_label, 3, 1)
        
        # Function
        details_layout.addWidget(QLabel("Activation Function:"), 4, 0)
        self.function_label = QLabel("ReLU")
        self.function_label.setStyleSheet("color: #ECF0F1;")
        details_layout.addWidget(self.function_label, 4, 1)
        
        # Last update
        details_layout.addWidget(QLabel("Last Updated:"), 5, 0)
        self.update_label = QLabel("Never")
        self.update_label.setStyleSheet("color: #ECF0F1;")
        details_layout.addWidget(self.update_label, 5, 1)
        
        viz_container.addWidget(self.details_panel)
        
        # Set initial sizes
        viz_container.setSizes([700, 300])
        
        main_layout.addWidget(viz_container)
        
        # Bottom controls
        bottom_layout = QHBoxLayout()
        
        # Auto-update toggle
        self.auto_update = QPushButton("Auto-Update: OFF")
        self.auto_update.setCheckable(True)
        self.auto_update.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50;
                color: white;
                border-radius: 5px;
                padding: 8px 15px;
                font-size: 14px;
            }
            QPushButton:checked {
                background-color: #27AE60;
            }
            QPushButton:hover {
                background-color: #34495E;
            }
            QPushButton:checked:hover {
                background-color: #2ECC71;
            }
        """)
        self.auto_update.toggled.connect(self.toggle_auto_update)
        bottom_layout.addWidget(self.auto_update)
        
        bottom_layout.addStretch()
        
        # Refresh button
        self.refresh_button = QPushButton("Refresh View")
        self.refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #2980B9;
                color: white;
                border-radius: 5px;
                padding: 8px 15px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3498DB;
            }
            QPushButton:pressed {
                background-color: #1C587F;
            }
        """)
        self.refresh_button.clicked.connect(self.refresh_visualization)
        bottom_layout.addWidget(self.refresh_button)
        
        main_layout.addLayout(bottom_layout)
        
        # Initialize with mock data
        self.update_mock_data()
        
    def toggle_auto_update(self, checked):
        """Toggle auto-update mode"""
        text = "Auto-Update: ON" if checked else "Auto-Update: OFF"
        self.auto_update.setText(text)
        
    def refresh_visualization(self):
        """Refresh the network visualization with current data"""
        self.update_mock_data()
        
    def update_mock_data(self):
        """Update visualization with mock data for demonstration"""
        import random
        
        # Update each neuron with random activation
        for layer_viz in self.layer_visualizations:
            for neuron in layer_viz.neurons:
                neuron.activation = random.random()
                neuron.update()
                
        # Update connection weights with random values
        for conn_viz in self.connection_visualizations:
            for i in range(conn_viz.from_layer_size):
                for j in range(conn_viz.to_layer_size):
                    conn_viz.weights[i][j] = random.uniform(-1, 1)
            conn_viz.update()
        
    def on_neuron_clicked(self, layer_idx, neuron_idx):
        """Handle neuron click event"""
        import random
        from datetime import datetime
        
        # Update neuron details panel
        self.neuron_label.setText(f"Layer {layer_idx}, Neuron {neuron_idx}")
        self.activation_label.setText(f"{random.random():.4f}")
        self.bias_label.setText(f"{random.uniform(-1, 1):.4f}")
        
        # Generate mock weights
        weights_text = ""
        num_weights = random.randint(3, 8)
        for i in range(num_weights):
            weight = random.uniform(-1, 1)
            weights_text += f"W{i}: {weight:.4f}"
            if i < num_weights - 1:
                weights_text += "<br>"
        self.weights_label.setText(weights_text)
        
        # Random activation function
        functions = ["ReLU", "Sigmoid", "Tanh", "LeakyReLU", "Softmax"]
        self.function_label.setText(random.choice(functions))
        
        # Update timestamp
        self.update_label.setText(datetime.now().strftime("%H:%M:%S")) 