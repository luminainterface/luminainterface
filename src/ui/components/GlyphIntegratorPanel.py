from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                              QComboBox, QSlider, QGroupBox, QScrollArea, QFrame, 
                              QGridLayout, QSplitter, QCheckBox, QLineEdit)
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QTimer
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QFont, QPixmap

class GlyphCanvas(QWidget):
    """Canvas for drawing and rendering glyphs that connect to neural network activations"""
    
    glyph_selected = Signal(int)  # Emitted when a glyph is selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.glyphs = []  # List of glyph data
        self.current_glyph = None  # Currently selected glyph
        self.hover_glyph = None  # Glyph being hovered over
        self.neural_activations = {}  # Dictionary mapping neuron_ids to activation values
        
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #1E1E2E;")
        
    def add_glyph(self, glyph_id, path_data, neural_bindings=None):
        """Add a glyph to the canvas
        
        Args:
            glyph_id: Unique identifier for the glyph
            path_data: List of (x, y) coordinates defining the glyph
            neural_bindings: Dict mapping neural layer/node IDs to glyph parameters
        """
        glyph = {
            'id': glyph_id,
            'path': path_data,
            'color': QColor(200, 200, 255, 180),
            'bindings': neural_bindings or {},
            'properties': {
                'size': 1.0,
                'rotation': 0.0,
                'intensity': 0.8,
                'frequency': 0.5
            }
        }
        self.glyphs.append(glyph)
        self.update()
        
    def set_neural_activations(self, activations):
        """Update neural network activations that influence glyphs
        
        Args:
            activations: Dict mapping (layer_id, neuron_id) to activation value
        """
        self.neural_activations = activations
        self.update_glyph_properties()
        self.update()
        
    def update_glyph_properties(self):
        """Update glyph properties based on neural activations"""
        for glyph in self.glyphs:
            bindings = glyph['bindings']
            properties = glyph['properties']
            
            # Apply neural activations to glyph properties
            for (layer_id, neuron_id), prop in bindings.items():
                activation_key = (layer_id, neuron_id)
                if activation_key in self.neural_activations:
                    activation = self.neural_activations[activation_key]
                    
                    # Apply activation to the bound property
                    if prop == 'size':
                        properties['size'] = 0.5 + activation * 1.5
                    elif prop == 'rotation':
                        properties['rotation'] = activation * 360.0
                    elif prop == 'intensity':
                        properties['intensity'] = activation
                    elif prop == 'frequency':
                        properties['frequency'] = activation
            
            # Update color based on intensity
            r = int(200 * properties['intensity'])
            g = int(200 * properties['frequency'])
            b = int(255)
            glyph['color'] = QColor(r, g, b, 180)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw the glyphs
        for glyph in self.glyphs:
            self.paint_glyph(painter, glyph)
    
    def paint_glyph(self, painter, glyph):
        """Paint a single glyph"""
        path = QPainterPath()
        
        # Get glyph properties
        properties = glyph['properties']
        size = properties['size']
        rotation = properties['rotation']
        
        # Center of the widget
        center_x, center_y = self.width() / 2, self.height() / 2
        
        # Create the path
        if glyph['path']:
            # Start the path at the first point
            first_x, first_y = glyph['path'][0]
            path.moveTo(first_x * size + center_x, first_y * size + center_y)
            
            # Add the remaining points
            for x, y in glyph['path'][1:]:
                path.lineTo(x * size + center_x, y * size + center_y)
            
            # Close the path
            path.closeSubpath()
        
        # Apply rotation
        painter.save()
        painter.translate(center_x, center_y)
        painter.rotate(rotation)
        painter.translate(-center_x, -center_y)
        
        # Draw the glyph
        painter.setPen(QPen(QColor(255, 255, 255, 100), 2))
        painter.setBrush(QBrush(glyph['color']))
        
        # Highlight if this is the current or hover glyph
        if glyph is self.current_glyph:
            painter.setPen(QPen(QColor(255, 215, 0), 3))
        elif glyph is self.hover_glyph:
            painter.setPen(QPen(QColor(220, 220, 220), 2))
        
        painter.drawPath(path)
        painter.restore()
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            for glyph in self.glyphs:
                # TODO: Implement hit testing for glyphs
                # For now, just select the first glyph
                self.current_glyph = glyph
                self.glyph_selected.emit(glyph['id'])
                self.update()
                break
        super().mousePressEvent(event)

class NeuralGlyphMapper(QWidget):
    """Widget for mapping neural network nodes to glyph properties"""
    
    mapping_changed = Signal(dict)  # Emitted when mappings are changed
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Neural-Glyph Bindings")
        header.setStyleSheet("font-weight: bold; font-size: 14px; color: #2C3E50;")
        layout.addWidget(header)
        
        # Mapping container
        mapping_container = QScrollArea()
        mapping_container.setWidgetResizable(True)
        mapping_container.setFrameShape(QFrame.NoFrame)
        
        mapping_widget = QWidget()
        self.mapping_layout = QVBoxLayout(mapping_widget)
        self.mapping_layout.setAlignment(Qt.AlignTop)
        
        mapping_container.setWidget(mapping_widget)
        layout.addWidget(mapping_container)
        
        # Add new binding button
        add_btn = QPushButton("Add Binding")
        add_btn.clicked.connect(self.add_binding_row)
        layout.addWidget(add_btn)
        
        # Add a few default bindings
        self.add_binding_row()
        self.add_binding_row()
        
    def add_binding_row(self):
        """Add a new row for neural-glyph binding"""
        binding_row = QWidget()
        row_layout = QHBoxLayout(binding_row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        
        # Neural selector
        layer_combo = QComboBox()
        layer_combo.addItems(["Input Layer", "Hidden Layer 1", "Hidden Layer 2", "Output Layer"])
        row_layout.addWidget(layer_combo)
        
        neuron_combo = QComboBox()
        neuron_combo.addItems([f"Neuron {i}" for i in range(10)])
        row_layout.addWidget(neuron_combo)
        
        # Connection arrow
        arrow_label = QLabel("→")
        arrow_label.setAlignment(Qt.AlignCenter)
        row_layout.addWidget(arrow_label)
        
        # Glyph property selector
        glyph_combo = QComboBox()
        glyph_combo.addItems(["Glyph 1", "Glyph 2", "Glyph 3"])
        row_layout.addWidget(glyph_combo)
        
        property_combo = QComboBox()
        property_combo.addItems(["Size", "Rotation", "Intensity", "Frequency"])
        row_layout.addWidget(property_combo)
        
        # Remove button
        remove_btn = QPushButton("✕")
        remove_btn.setFixedWidth(30)
        remove_btn.clicked.connect(lambda: self.remove_binding_row(binding_row))
        row_layout.addWidget(remove_btn)
        
        self.mapping_layout.addWidget(binding_row)
        
    def remove_binding_row(self, row_widget):
        """Remove a binding row"""
        row_widget.setParent(None)
        
    def get_mappings(self):
        """Get the current neural-glyph mappings"""
        mappings = {}
        for i in range(self.mapping_layout.count()):
            row_widget = self.mapping_layout.itemAt(i).widget()
            if row_widget:
                row_layout = row_widget.layout()
                if row_layout.count() >= 5:
                    layer_combo = row_layout.itemAt(0).widget()
                    neuron_combo = row_layout.itemAt(1).widget()
                    glyph_combo = row_layout.itemAt(3).widget()
                    property_combo = row_layout.itemAt(4).widget()
                    
                    layer = layer_combo.currentText()
                    neuron = neuron_combo.currentText()
                    glyph = glyph_combo.currentText()
                    property_name = property_combo.currentText().lower()
                    
                    # Convert to IDs
                    layer_id = layer_combo.currentIndex()
                    neuron_id = neuron_combo.currentIndex()
                    glyph_id = glyph_combo.currentIndex()
                    
                    key = (layer_id, neuron_id)
                    if key not in mappings:
                        mappings[key] = {}
                    
                    mappings[key][glyph_id] = property_name
        
        return mappings
        
class GlyphIntegratorPanel(QWidget):
    """Panel for integrating neural network activations with the glyph interface"""
    
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
        
        title = QLabel("Neural-Glyph Integration")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2C3E50;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Network selector
        network_label = QLabel("Active Network:")
        header_layout.addWidget(network_label)
        
        self.network_selector = QComboBox()
        self.network_selector.addItems(["Primary Network", "Secondary Network"])
        self.network_selector.setMinimumWidth(150)
        header_layout.addWidget(self.network_selector)
        
        # Auto-update checkbox
        self.auto_update = QCheckBox("Auto-update")
        self.auto_update.setChecked(True)
        header_layout.addWidget(self.auto_update)
        
        # Update button
        self.update_btn = QPushButton("Update Now")
        self.update_btn.clicked.connect(self.update_visualizations)
        header_layout.addWidget(self.update_btn)
        
        main_layout.addWidget(header)
        
        # Main content area with splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Neural-Glyph Mapper
        self.mapper = NeuralGlyphMapper()
        self.mapper.mapping_changed.connect(self.on_mapping_changed)
        splitter.addWidget(self.mapper)
        
        # Right side - Glyph Visualization
        glyph_container = QWidget()
        glyph_layout = QVBoxLayout(glyph_container)
        
        glyph_header = QLabel("Glyph Visualization")
        glyph_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #2C3E50;")
        glyph_layout.addWidget(glyph_header)
        
        self.glyph_canvas = GlyphCanvas()
        glyph_layout.addWidget(self.glyph_canvas, 1)
        
        splitter.addWidget(glyph_container)
        
        # Set initial splitter sizes
        splitter.setSizes([300, 500])
        
        main_layout.addWidget(splitter, 1)
        
        # Footer with status and actions
        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #555;")
        footer_layout.addWidget(self.status_label)
        
        footer_layout.addStretch()
        
        self.save_btn = QPushButton("Save Mapping")
        self.save_btn.clicked.connect(self.save_mapping)
        footer_layout.addWidget(self.save_btn)
        
        self.export_btn = QPushButton("Export Glyphs")
        self.export_btn.clicked.connect(self.export_glyphs)
        footer_layout.addWidget(self.export_btn)
        
        main_layout.addWidget(footer)
        
        # Set up timer for animations/updates
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_visualizations)
        self.update_timer.start(1000)  # Update every 1000ms
        
    def setup_mock_data(self):
        """Set up mock glyph data for visualization"""
        # Create a few sample glyphs
        glyph1_points = [
            (-50, -50), (0, -70), (50, -50),
            (70, 0), (50, 50), (0, 70),
            (-50, 50), (-70, 0)
        ]
        self.glyph_canvas.add_glyph(1, glyph1_points, {
            (0, 2): 'size',
            (1, 4): 'rotation',
            (2, 1): 'intensity'
        })
        
        glyph2_points = [
            (0, -60), (60, 0), (0, 60), (-60, 0)
        ]
        self.glyph_canvas.add_glyph(2, glyph2_points, {
            (1, 3): 'frequency',
            (2, 5): 'intensity',
            (3, 1): 'rotation'
        })
        
        # Set initial activations
        self.update_mock_activations()
    
    def update_mock_activations(self):
        """Update with mock neural activations"""
        import random
        
        # Create random activations for neural network nodes
        activations = {}
        for layer in range(4):  # 4 layers
            for neuron in range(10):  # 10 neurons per layer
                # Generate random activation between 0 and 1
                activation = random.random()
                activations[(layer, neuron)] = activation
        
        self.glyph_canvas.set_neural_activations(activations)
        self.status_label.setText(f"Updated at {QTimer.currentTime().toString('hh:mm:ss')}")
    
    def update_visualizations(self):
        """Update visualizations based on current neural activations"""
        if self.auto_update.isChecked():
            self.update_mock_activations()
    
    def on_mapping_changed(self, mappings):
        """Handle changes to neural-glyph mappings"""
        # Update the glyph bindings
        for glyph in self.glyph_canvas.glyphs:
            # TODO: Update bindings based on mappings
            pass
    
    def save_mapping(self):
        """Save the current neural-glyph mapping configuration"""
        # In a real app, this would save to a file or database
        self.status_label.setText("Mapping saved")
    
    def export_glyphs(self):
        """Export the current glyph visualization"""
        # In a real app, this would export as an image or data file
        self.status_label.setText("Glyphs exported") 