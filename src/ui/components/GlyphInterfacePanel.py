from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QGridLayout, QScrollArea, QFrame,
                            QGroupBox, QLineEdit, QComboBox, QStackedWidget,
                            QSlider)
from PySide6.QtGui import QIcon, QFont, QPainter, QColor, QPen, QBrush, QPainterPath
from PySide6.QtCore import Qt, Signal, Slot, QSize, QPoint, QRect, QTimer

class GlyphCanvas(QWidget):
    """Canvas widget for drawing and displaying glyphs"""
    
    glyph_activated = Signal(int)  # Signal emitted when a glyph is activated
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.glyphs = []
        self.active_glyph = -1
        self.pulsing_glyphs = {}
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animations)
        self.animation_timer.start(50)  # 20 fps animation
        self.init_demo_glyphs()
        
    def init_demo_glyphs(self):
        """Initialize demo glyphs for visualization"""
        # Basic geometric glyphs
        self.glyphs = [
            {"id": 0, "name": "circle", "points": [], "type": "circle", "x": 100, "y": 100, "size": 40, 
             "color": QColor(147, 112, 219, 200), "description": "Unity, wholeness, infinity"},
            {"id": 1, "name": "triangle", "points": [], "type": "triangle", "x": 200, "y": 100, "size": 40, 
             "color": QColor(112, 146, 190, 200), "description": "Transformation, change, direction"},
            {"id": 2, "name": "square", "points": [], "type": "square", "x": 300, "y": 100, "size": 40, 
             "color": QColor(112, 190, 160, 200), "description": "Stability, structure, foundation"},
            {"id": 3, "name": "spiral", "points": [], "type": "spiral", "x": 100, "y": 200, "size": 40, 
             "color": QColor(190, 160, 112, 200), "description": "Growth, evolution, journey"},
            {"id": 4, "name": "wave", "points": [], "type": "wave", "x": 200, "y": 200, "size": 40, 
             "color": QColor(190, 112, 160, 200), "description": "Flow, water, change"},
            {"id": 5, "name": "cross", "points": [], "type": "cross", "x": 300, "y": 200, "size": 40, 
             "color": QColor(160, 112, 190, 200), "description": "Balance, intersection, connection"},
            {"id": 6, "name": "star", "points": [], "type": "star", "x": 150, "y": 300, "size": 40, 
             "color": QColor(147, 112, 219, 200), "description": "Illumination, guidance, direction"},
            {"id": 7, "name": "hexagon", "points": [], "type": "hexagon", "x": 250, "y": 300, "size": 40, 
             "color": QColor(112, 146, 190, 200), "description": "Harmony, balance, perfection"}
        ]
    
    def update_animations(self):
        """Update animations for pulsing glyphs"""
        update_needed = False
        for glyph_id, pulse_data in list(self.pulsing_glyphs.items()):
            pulse_data["time"] += 1
            if pulse_data["time"] > 40:  # 2 seconds (40 * 50ms)
                self.pulsing_glyphs.pop(glyph_id)
            update_needed = True
        
        if update_needed:
            self.update()
    
    def paintEvent(self, event):
        """Draw glyphs on the canvas"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background grid
        self.draw_grid(painter)
        
        # Draw all glyphs
        for glyph in self.glyphs:
            pulse_scale = 1.0
            pulse_opacity = 1.0
            
            # Apply pulsing effect if glyph is pulsing
            if glyph["id"] in self.pulsing_glyphs:
                pulse_data = self.pulsing_glyphs[glyph["id"]]
                pulse_progress = pulse_data["time"] / 40.0  # 0.0 to 1.0
                pulse_scale = 1.0 + 0.3 * (1.0 - abs(pulse_progress * 2 - 1))
                pulse_opacity = max(0.4, 1.0 - pulse_progress)
            
            # Highlight active glyph
            if glyph["id"] == self.active_glyph:
                # Draw glow effect
                glow_color = QColor(glyph["color"])
                for i in range(3):
                    glow_color.setAlpha(50 - i * 15)
                    glow_size = glyph["size"] * (1.0 + 0.15 * (i+1)) * pulse_scale
                    self.draw_glyph(painter, glyph, glyph["x"], glyph["y"], glow_size, glow_color)
            
            # Draw the glyph
            glyph_color = QColor(glyph["color"])
            glyph_color.setAlpha(int(255 * pulse_opacity))
            self.draw_glyph(painter, glyph, glyph["x"], glyph["y"], 
                            glyph["size"] * pulse_scale, glyph_color)
    
    def draw_grid(self, painter):
        """Draw background grid"""
        painter.setPen(QPen(QColor(60, 60, 80, 40), 1, Qt.DotLine))
        
        # Draw horizontal lines
        for y in range(0, self.height(), 20):
            painter.drawLine(0, y, self.width(), y)
        
        # Draw vertical lines
        for x in range(0, self.width(), 20):
            painter.drawLine(x, 0, x, self.height())
    
    def draw_glyph(self, painter, glyph, x, y, size, color):
        """Draw a single glyph"""
        painter.setPen(QPen(color, 2))
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 60)))
        
        if glyph["type"] == "circle":
            painter.drawEllipse(QPoint(x, y), int(size/2), int(size/2))
        
        elif glyph["type"] == "triangle":
            path = QPainterPath()
            path.moveTo(x, y - size/2)
            path.lineTo(x + size/2, y + size/2)
            path.lineTo(x - size/2, y + size/2)
            path.closeSubpath()
            painter.drawPath(path)
        
        elif glyph["type"] == "square":
            painter.drawRect(QRect(int(x - size/2), int(y - size/2), int(size), int(size)))
        
        elif glyph["type"] == "spiral":
            path = QPainterPath()
            path.moveTo(x, y)
            for i in range(0, 360, 10):
                angle = i * 3.14159 / 180.0
                radius = size/2 * (i / 360.0)
                px = x + radius * math.cos(angle)
                py = y + radius * math.sin(angle)
                path.lineTo(px, py)
            painter.drawPath(path)
        
        elif glyph["type"] == "wave":
            path = QPainterPath()
            path.moveTo(x - size/2, y)
            for i in range(int(-size/2), int(size/2)+1, 2):
                path.lineTo(x + i, y + math.sin(i * 0.2) * size/4)
            painter.drawPath(path)
        
        elif glyph["type"] == "cross":
            painter.drawLine(int(x - size/2), int(y), int(x + size/2), int(y))
            painter.drawLine(int(x), int(y - size/2), int(x), int(y + size/2))
        
        elif glyph["type"] == "star":
            path = QPainterPath()
            for i in range(5):
                angle1 = i * 72 * 3.14159 / 180.0 - 3.14159 / 2
                angle2 = (i * 72 + 36) * 3.14159 / 180.0 - 3.14159 / 2
                
                px1 = x + size/2 * math.cos(angle1)
                py1 = y + size/2 * math.sin(angle1)
                px2 = x + size/4 * math.cos(angle2)
                py2 = y + size/4 * math.sin(angle2)
                
                if i == 0:
                    path.moveTo(px1, py1)
                else:
                    path.lineTo(px1, py1)
                path.lineTo(px2, py2)
            
            path.closeSubpath()
            painter.drawPath(path)
        
        elif glyph["type"] == "hexagon":
            path = QPainterPath()
            for i in range(6):
                angle = i * 60 * 3.14159 / 180.0
                px = x + size/2 * math.cos(angle)
                py = y + size/2 * math.sin(angle)
                
                if i == 0:
                    path.moveTo(px, py)
                else:
                    path.lineTo(px, py)
            
            path.closeSubpath()
            painter.drawPath(path)
    
    def mousePressEvent(self, event):
        """Handle mouse press events to select glyphs"""
        pos = event.position()
        x, y = pos.x(), pos.y()
        
        # Check if a glyph was clicked
        for glyph in self.glyphs:
            distance = math.sqrt((x - glyph["x"])**2 + (y - glyph["y"])**2)
            if distance < glyph["size"]/2 + 10:  # Add small margin for easier selection
                self.set_active_glyph(glyph["id"])
                self.glyph_activated.emit(glyph["id"])
                break
    
    def set_active_glyph(self, glyph_id):
        """Set the active glyph and trigger pulsing animation"""
        if glyph_id != self.active_glyph:
            self.active_glyph = glyph_id
            self.pulsing_glyphs[glyph_id] = {"time": 0}
            self.update()
    
    def get_glyph_by_id(self, glyph_id):
        """Get glyph data by ID"""
        for glyph in self.glyphs:
            if glyph["id"] == glyph_id:
                return glyph
        return None


class GlyphDetailsWidget(QWidget):
    """Widget for displaying and editing glyph details"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_glyph = None
        self.initUI()
    
    def initUI(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Glyph title
        self.title_label = QLabel("No Glyph Selected")
        self.title_label.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            color: #9370DB;
        """)
        layout.addWidget(self.title_label)
        
        # Description
        self.description_label = QLabel("Select a glyph to view its details")
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("color: #ECF0F1; font-size: 14px;")
        layout.addWidget(self.description_label)
        
        # Properties group
        properties_group = QGroupBox("Properties")
        properties_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #3A3A3A;
                border-radius: 5px;
                margin-top: 10px;
                font-size: 14px;
                color: #9370DB;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 5px 10px;
                background-color: #1A1A2E;
            }
        """)
        
        properties_layout = QGridLayout(properties_group)
        properties_layout.setColumnStretch(1, 1)
        
        # Color
        properties_layout.addWidget(QLabel("Color:"), 0, 0)
        self.color_combo = QComboBox()
        self.color_combo.addItems(["Purple", "Blue", "Green", "Gold", "Red", "Violet"])
        self.color_combo.setStyleSheet("""
            background-color: #2C3E50;
            color: #ECF0F1;
            border: 1px solid #3A3A3A;
            border-radius: 3px;
            padding: 5px;
        """)
        properties_layout.addWidget(self.color_combo, 0, 1)
        
        # Size
        properties_layout.addWidget(QLabel("Size:"), 1, 0)
        self.size_combo = QComboBox()
        self.size_combo.addItems(["Small", "Medium", "Large"])
        self.size_combo.setCurrentIndex(1)  # Medium by default
        self.size_combo.setStyleSheet("""
            background-color: #2C3E50;
            color: #ECF0F1;
            border: 1px solid #3A3A3A;
            border-radius: 3px;
            padding: 5px;
        """)
        properties_layout.addWidget(self.size_combo, 1, 1)
        
        # Meaning/Association
        properties_layout.addWidget(QLabel("Meaning:"), 2, 0)
        self.meaning_edit = QLineEdit()
        self.meaning_edit.setPlaceholderText("Enter glyph meaning...")
        self.meaning_edit.setStyleSheet("""
            background-color: #2C3E50;
            color: #ECF0F1;
            border: 1px solid #3A3A3A;
            border-radius: 3px;
            padding: 5px;
        """)
        properties_layout.addWidget(self.meaning_edit, 2, 1)
        
        layout.addWidget(properties_group)
        
        # Connections group
        connections_group = QGroupBox("Neural Connections")
        connections_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #3A3A3A;
                border-radius: 5px;
                margin-top: 10px;
                font-size: 14px;
                color: #9370DB;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 5px 10px;
                background-color: #1A1A2E;
            }
        """)
        
        connections_layout = QVBoxLayout(connections_group)
        
        # Add some placeholder connection information
        connections_info = QLabel(
            "This glyph has not been connected to the neural network yet. "
            "Select a glyph and use the 'Connect' button to establish neural pathways."
        )
        connections_info.setWordWrap(True)
        connections_info.setStyleSheet("color: #95A5A6; font-style: italic;")
        connections_layout.addWidget(connections_info)
        
        layout.addWidget(connections_group)
        
        # Actions
        actions_layout = QHBoxLayout()
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setEnabled(False)
        self.connect_btn.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: 1px solid #3A3A3A;
                border-radius: 3px;
                padding: 8px 12px;
            }
            QPushButton:hover {
                background-color: #3E5771;
            }
            QPushButton:disabled {
                background-color: #1E2429;
                color: #5D6D7E;
            }
        """)
        actions_layout.addWidget(self.connect_btn)
        
        self.activate_btn = QPushButton("Activate")
        self.activate_btn.setEnabled(False)
        self.activate_btn.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: 1px solid #3A3A3A;
                border-radius: 3px;
                padding: 8px 12px;
            }
            QPushButton:hover {
                background-color: #3E5771;
            }
            QPushButton:disabled {
                background-color: #1E2429;
                color: #5D6D7E;
            }
        """)
        actions_layout.addWidget(self.activate_btn)
        
        layout.addLayout(actions_layout)
        
        # Add stretch at the end to push everything up
        layout.addStretch()
    
    def update_glyph_details(self, glyph):
        """Update the UI with the selected glyph's details"""
        self.current_glyph = glyph
        
        if glyph:
            self.title_label.setText(f"Glyph: {glyph['name'].title()}")
            self.description_label.setText(glyph["description"])
            
            # Enable buttons
            self.connect_btn.setEnabled(True)
            self.activate_btn.setEnabled(True)
            
            # Update fields
            if "purple" in glyph["color"].name().lower():
                self.color_combo.setCurrentText("Purple")
            elif "blue" in glyph["color"].name().lower():
                self.color_combo.setCurrentText("Blue")
            elif "green" in glyph["color"].name().lower():
                self.color_combo.setCurrentText("Green")
            
            self.meaning_edit.setText(glyph["description"])
        else:
            self.title_label.setText("No Glyph Selected")
            self.description_label.setText("Select a glyph to view its details")
            self.connect_btn.setEnabled(False)
            self.activate_btn.setEnabled(False)


class GlyphSequenceBar(QWidget):
    """Widget to display and manage sequences of glyphs"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sequence = []  # List of glyph IDs in the sequence
        self.initUI()
    
    def initUI(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Glyph Sequence")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #9370DB;")
        layout.addWidget(title)
        
        # Sequence display area
        self.sequence_widget = QWidget()
        self.sequence_layout = QHBoxLayout(self.sequence_widget)
        self.sequence_layout.setSpacing(8)
        self.sequence_layout.setContentsMargins(10, 5, 10, 5)
        self.sequence_layout.addStretch()
        
        sequence_frame = QFrame()
        sequence_frame.setFrameShape(QFrame.StyledPanel)
        sequence_frame.setStyleSheet("""
            QFrame {
                background-color: #1A1A2E;
                border: 1px solid #3A3A3A;
                border-radius: 5px;
            }
        """)
        sequence_frame.setLayout(QVBoxLayout())
        sequence_frame.layout().addWidget(self.sequence_widget)
        
        layout.addWidget(sequence_frame)
        
        # Sequence controls
        controls_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: 1px solid #3A3A3A;
                border-radius: 3px;
                padding: 8px 12px;
            }
            QPushButton:hover {
                background-color: #3E5771;
            }
        """)
        controls_layout.addWidget(self.clear_btn)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: 1px solid #3A3A3A;
                border-radius: 3px;
                padding: 8px 12px;
            }
            QPushButton:hover {
                background-color: #3E5771;
            }
        """)
        controls_layout.addWidget(self.save_btn)
        
        self.execute_btn = QPushButton("Execute")
        self.execute_btn.setStyleSheet("""
            QPushButton {
                background-color: #6C3483;
                color: #ECF0F1;
                border: 1px solid #3A3A3A;
                border-radius: 3px;
                padding: 8px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #8E44AD;
            }
        """)
        controls_layout.addWidget(self.execute_btn)
        
        layout.addLayout(controls_layout)
        
        # Connect signals
        self.clear_btn.clicked.connect(self.clear_sequence)
    
    def add_glyph_to_sequence(self, glyph):
        """Add a glyph to the current sequence"""
        if glyph and len(self.sequence) < 8:  # Limit to 8 glyphs
            self.sequence.append(glyph["id"])
            
            # Create a label to represent the glyph in the sequence
            glyph_label = QLabel()
            glyph_label.setFixedSize(QSize(30, 30))
            glyph_label.setStyleSheet(f"""
                background-color: {glyph["color"].name()};
                border-radius: 15px;
                text-align: center;
            """)
            
            # Remove the stretch if this is the first item
            if len(self.sequence) == 1:
                # Remove the stretch that was added in initUI
                for i in range(self.sequence_layout.count()):
                    item = self.sequence_layout.itemAt(i)
                    if item.spacerItem():
                        self.sequence_layout.removeItem(item)
                        break
            
            # Add the new glyph label
            self.sequence_layout.addWidget(glyph_label)
            
            # Add stretch at the end to keep glyphs aligned to the left
            self.sequence_layout.addStretch()
    
    def clear_sequence(self):
        """Clear the current sequence"""
        self.sequence = []
        
        # Remove all widgets from the layout
        while self.sequence_layout.count():
            item = self.sequence_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add stretch back
        self.sequence_layout.addStretch()


class GlyphInterfacePanel(QWidget):
    """Panel for interacting with glyphs using PySide6"""
    # Signal to notify of glyph state changes
    glyph_state_changed = Signal(dict)
    breath_data_updated = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
        # Initialize breath tracking data
        self.breath_data = {
            "pattern": "normal",
            "phase": 0.0,
            "intensity": 0.5,
            "rhythm": "steady",
            "duration": 4.0,
            "connected_to_network": False
        }
        
        # Initialize breath simulation timer
        self.breath_timer = QTimer(self)
        self.breath_timer.timeout.connect(self.update_breath_phase)
        
        # Set up initial glyph state
        self.current_glyphs = []
        self.network_connected = False
    
    def initUI(self):
        """Initialize the glyph interface UI"""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # Header with title and mode selector
        header_layout = QHBoxLayout()
        
        title = QLabel("Glyph Interface")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #E0E0E0;")
        header_layout.addWidget(title)
        
        # Mode selector
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Edit Mode", "View Mode", "Breath Mode"])
        self.mode_selector.setStyleSheet("""
            QComboBox {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: 1px solid #34495E;
                border-radius: 4px;
                padding: 4px;
                min-width: 100px;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox QAbstractItemView {
                background-color: #2C3E50;
                color: #ECF0F1;
                selection-background-color: #3498DB;
            }
        """)
        self.mode_selector.currentIndexChanged.connect(self.change_mode)
        header_layout.addWidget(self.mode_selector)
        
        main_layout.addLayout(header_layout)
        
        # Glyph canvas frame
        self.glyph_canvas = QFrame()
        self.glyph_canvas.setMinimumHeight(300)
        self.glyph_canvas.setStyleSheet("""
            QFrame {
                background-color: #1E272E;
                border: 1px solid #34495E;
                border-radius: 5px;
            }
        """)
        main_layout.addWidget(self.glyph_canvas)
        
        # Glyph controls
        controls_layout = QHBoxLayout()
        
        # Add glyph button
        self.add_glyph_btn = QPushButton("Add Glyph")
        self.add_glyph_btn.setStyleSheet("""
            QPushButton {
                background-color: #2980B9;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #3498DB;
            }
        """)
        controls_layout.addWidget(self.add_glyph_btn)
        
        # Clear button
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #C0392B;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #E74C3C;
            }
        """)
        controls_layout.addWidget(self.clear_btn)
        
        # Save button
        self.save_btn = QPushButton("Save Pattern")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #2ECC71;
            }
        """)
        controls_layout.addWidget(self.save_btn)
        
        main_layout.addLayout(controls_layout)
        
        # Breath tracking frame - initially hidden, shown in Breath Mode
        self.breath_frame = QFrame()
        self.breath_frame.setVisible(False)
        self.setup_breath_tracking()
        main_layout.addWidget(self.breath_frame)
        
        # Status bar at bottom
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #7F8C8D;")
        main_layout.addWidget(self.status_label)
    
    def change_mode(self, index):
        """Handle mode changes"""
        mode = self.mode_selector.currentText()
        
        # Show/hide relevant controls based on mode
        self.breath_frame.setVisible(mode == "Breath Mode")
        
        # Enable/disable buttons based on mode
        edit_mode = mode == "Edit Mode"
        self.add_glyph_btn.setEnabled(edit_mode)
        self.clear_btn.setEnabled(edit_mode)
        
        # Start/stop breath tracking based on mode
        if mode == "Breath Mode":
            self.start_breath_tracking()
            self.status_label.setText("Breath tracking active")
        else:
            self.stop_breath_tracking()
            self.status_label.setText(f"Switched to {mode}")
    
    def setup_breath_tracking(self):
        """Set up the breath tracking UI components"""
        breath_layout = QVBoxLayout(self.breath_frame)
        breath_layout.setContentsMargins(10, 10, 10, 10)
        
        # Breath tracking header
        breath_header = QLabel("Breath Tracking")
        breath_header.setStyleSheet("font-size: 16px; font-weight: bold; color: #3498DB;")
        breath_layout.addWidget(breath_header)
        
        # Breath controls
        breath_controls = QGridLayout()
        
        # Intensity slider
        intensity_label = QLabel("Breath Intensity:")
        intensity_label.setStyleSheet("color: #ECF0F1;")
        breath_controls.addWidget(intensity_label, 0, 0)
        
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setMinimum(1)
        self.intensity_slider.setMaximum(100)
        self.intensity_slider.setValue(50)
        self.intensity_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #34495E;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498DB;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
        """)
        self.intensity_slider.valueChanged.connect(self.update_breath_intensity)
        breath_controls.addWidget(self.intensity_slider, 0, 1)
        
        self.intensity_value = QLabel("50%")
        self.intensity_value.setStyleSheet("color: #ECF0F1;")
        breath_controls.addWidget(self.intensity_value, 0, 2)
        
        # Duration slider
        duration_label = QLabel("Breath Duration:")
        duration_label.setStyleSheet("color: #ECF0F1;")
        breath_controls.addWidget(duration_label, 1, 0)
        
        self.duration_slider = QSlider(Qt.Horizontal)
        self.duration_slider.setMinimum(20)
        self.duration_slider.setMaximum(120)
        self.duration_slider.setValue(40)
        self.duration_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #34495E;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498DB;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
        """)
        self.duration_slider.valueChanged.connect(self.update_breath_duration)
        breath_controls.addWidget(self.duration_slider, 1, 1)
        
        self.duration_value = QLabel("4.0s")
        self.duration_value.setStyleSheet("color: #ECF0F1;")
        breath_controls.addWidget(self.duration_value, 1, 2)
        
        # Pattern selector
        pattern_label = QLabel("Breath Pattern:")
        pattern_label.setStyleSheet("color: #ECF0F1;")
        breath_controls.addWidget(pattern_label, 2, 0)
        
        self.pattern_selector = QComboBox()
        self.pattern_selector.addItems(["Normal Breathing", "Deep Breathing", "Box Breathing"])
        self.pattern_selector.setStyleSheet("""
            QComboBox {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: 1px solid #34495E;
                border-radius: 4px;
                padding: 4px;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox QAbstractItemView {
                background-color: #2C3E50;
                color: #ECF0F1;
                selection-background-color: #3498DB;
            }
        """)
        self.pattern_selector.currentIndexChanged.connect(self.update_breath_pattern)
        breath_controls.addWidget(self.pattern_selector, 2, 1, 1, 2)
        
        breath_layout.addLayout(breath_controls)
        
        # Network connection
        connection_group = QGroupBox("Neural Network Connection")
        connection_group.setStyleSheet("""
            QGroupBox {
                color: #ECF0F1;
                border: 1px solid #34495E;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        connection_layout = QVBoxLayout(connection_group)
        
        # Connection status
        self.connection_status = QLabel("Status: Disconnected")
        self.connection_status.setStyleSheet("color: #E74C3C;") # Red for disconnected
        connection_layout.addWidget(self.connection_status)
        
        # Connect button
        self.connect_button = QPushButton("Connect to Neural Network")
        self.connect_button.setCheckable(True)
        self.connect_button.setStyleSheet("""
            QPushButton {
                background-color: #8E44AD;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #9B59B6;
            }
            QPushButton:checked {
                background-color: #6C3483;
            }
        """)
        self.connect_button.clicked.connect(self.toggle_network_connection)
        connection_layout.addWidget(self.connect_button)
        
        breath_layout.addWidget(connection_group)
        
        # Visualization area for breath influence on glyphs
        self.visualization_frame = QFrame()
        self.visualization_frame.setMinimumHeight(100)
        self.visualization_frame.setStyleSheet("""
            QFrame {
                background-color: #1E272E;
                border: 1px solid #34495E;
                border-radius: 5px;
            }
        """)
        breath_layout.addWidget(self.visualization_frame)
    
    def start_breath_tracking(self):
        """Start breath tracking simulation"""
        # Reset breath phase
        self.breath_data["phase"] = 0.0
        
        # Start the timer - update 20 times per second
        self.breath_timer.start(50)
    
    def stop_breath_tracking(self):
        """Stop breath tracking simulation"""
        if self.breath_timer.isActive():
            self.breath_timer.stop()
    
    def update_breath_phase(self):
        """Update the breath phase based on timer"""
        # Calculate phase increment based on duration
        # Full cycle should take 'duration' seconds
        increment = 1.0 / (self.breath_data["duration"] * 20)  # 20 updates per second
        
        # Update phase (0.0 to 1.0, then wrap around)
        self.breath_data["phase"] = (self.breath_data["phase"] + increment) % 1.0
        
        # Apply the breath effect to glyphs
        self.apply_breath_to_glyphs()
        
        # Emit breath data if connected to network
        if self.breath_data["connected_to_network"]:
            self.breath_data_updated.emit(self.breath_data)
    
    def update_breath_intensity(self, value):
        """Update breath intensity from slider"""
        intensity = value / 100.0
        self.breath_data["intensity"] = intensity
        self.intensity_value.setText(f"{value}%")
    
    def update_breath_duration(self, value):
        """Update breath cycle duration from slider"""
        duration = value / 10.0
        self.breath_data["duration"] = duration
        self.duration_value.setText(f"{duration:.1f}s")
    
    def update_breath_pattern(self, index):
        """Update breath pattern from selector"""
        pattern_map = {
            0: "normal",
            1: "deep",
            2: "box"
        }
        self.breath_data["pattern"] = pattern_map.get(index, "normal")
    
    def toggle_network_connection(self, checked):
        """Toggle connection to neural network"""
        self.network_connected = checked
        self.breath_data["connected_to_network"] = checked
        
        if checked:
            self.connection_status.setText("Status: Connected")
            self.connection_status.setStyleSheet("color: #2ECC71;")  # Green
            self.connect_button.setText("Disconnect from Neural Network")
        else:
            self.connection_status.setText("Status: Disconnected")
            self.connection_status.setStyleSheet("color: #E74C3C;")  # Red
            self.connect_button.setText("Connect to Neural Network")
    
    def apply_breath_to_glyphs(self):
        """Apply the current breath state to glyph visualization"""
        phase = self.breath_data["phase"]
        intensity = self.breath_data["intensity"]
        pattern = self.breath_data["pattern"]
        
        # This would be implemented to apply different visual effects to glyphs
        # based on the current breath parameters
        
        # For now, just update the visualization frame to represent breathing
        self.visualization_frame.update()
    
    def paintEvent(self, event):
        """Paint the glyph canvas and breath visualization"""
        super().paintEvent(event)
        
        # Paint breath visualization in the visualization frame if visible
        if self.breath_frame.isVisible():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Get the visualization frame geometry
            rect = self.visualization_frame.geometry()
            
            # Draw breath wave
            painter.setPen(Qt.NoPen)
            
            # Calculate wave properties based on breath data
            phase = self.breath_data["phase"]
            intensity = self.breath_data["intensity"]
            pattern = self.breath_data["pattern"]
            
            # Draw different visualizations based on pattern
            if pattern == "box":
                self.draw_box_breath(painter, rect, phase, intensity)
            elif pattern == "deep":
                self.draw_deep_breath(painter, rect, phase, intensity)
            else:
                self.draw_normal_breath(painter, rect, phase, intensity)
    
    def draw_normal_breath(self, painter, rect, phase, intensity):
        """Draw normal breath visualization"""
        import math
        
        # Set color based on connection status
        if self.breath_data["connected_to_network"]:
            breath_color = QColor(52, 152, 219, 200)  # Blue, connected
        else:
            breath_color = QColor(155, 89, 182, 200)  # Purple, disconnected
        
        painter.setBrush(QBrush(breath_color))
        
        # Draw sine wave
        wave_height = rect.height() * 0.4 * intensity
        centerY = rect.y() + rect.height() / 2
        
        path = QPainterPath()
        path.moveTo(rect.x(), centerY)
        
        for x in range(rect.width()):
            # Calculate sine wave with phase offset
            normalizedX = x / rect.width()
            y = centerY - math.sin((normalizedX * 2 * math.pi) + (phase * 2 * math.pi)) * wave_height
            path.lineTo(rect.x() + x, y)
        
        path.lineTo(rect.x() + rect.width(), centerY)
        path.lineTo(rect.x(), centerY)
        
        painter.drawPath(path)
    
    def draw_deep_breath(self, painter, rect, phase, intensity):
        """Draw deep breath visualization"""
        import math
        
        # Set color based on connection status
        if self.breath_data["connected_to_network"]:
            breath_color = QColor(46, 204, 113, 200)  # Green, connected
        else:
            breath_color = QColor(155, 89, 182, 200)  # Purple, disconnected
        
        painter.setBrush(QBrush(breath_color))
        
        # Deep breathing has slower rise and fall
        wave_height = rect.height() * 0.4 * intensity
        centerY = rect.y() + rect.height() / 2
        
        path = QPainterPath()
        path.moveTo(rect.x(), centerY)
        
        for x in range(rect.width()):
            normalizedX = x / rect.width()
            
            # Modified sine wave for deep breathing effect
            if phase < 0.5:  # Inhale
                progress = phase * 2
                if normalizedX < progress:
                    y = centerY - math.pow(normalizedX / progress, 2) * wave_height
                else:
                    y = centerY
            else:  # Exhale
                progress = (phase - 0.5) * 2
                if normalizedX < progress:
                    y = centerY - (1 - math.pow(normalizedX / progress, 2)) * wave_height
                else:
                    y = centerY
            
            path.lineTo(rect.x() + x, y)
        
        path.lineTo(rect.x() + rect.width(), centerY)
        path.lineTo(rect.x(), centerY)
        
        painter.drawPath(path)
    
    def draw_box_breath(self, painter, rect, phase, intensity):
        """Draw box breath visualization"""
        # Set color based on connection status
        if self.breath_data["connected_to_network"]:
            breath_color = QColor(241, 196, 15, 200)  # Yellow, connected
        else:
            breath_color = QColor(155, 89, 182, 200)  # Purple, disconnected
        
        painter.setBrush(QBrush(breath_color))
        
        # Box breathing has 4 distinct phases
        wave_height = rect.height() * 0.4 * intensity
        centerY = rect.y() + rect.height() / 2
        width = rect.width()
        quarter_width = width / 4
        
        path = QPainterPath()
        path.moveTo(rect.x(), centerY)
        
        if phase < 0.25:  # Inhale
            # Phase 1: Inhale (rise)
            progress = phase * 4  # 0 to 1 during this phase
            x1 = rect.x() + (quarter_width * progress)
            y1 = centerY - (wave_height * progress)
            path.lineTo(x1, y1)
            path.lineTo(x1, centerY)
            
        elif phase < 0.5:  # Hold after inhale
            # Phase 2: Hold breath in (plateau)
            progress = (phase - 0.25) * 4  # 0 to 1 during this phase
            x1 = rect.x() + quarter_width
            y1 = centerY - wave_height
            x2 = rect.x() + quarter_width + (quarter_width * progress)
            path.lineTo(x1, y1)
            path.lineTo(x2, y1)
            path.lineTo(x2, centerY)
            
        elif phase < 0.75:  # Exhale
            # Phase 3: Exhale (fall)
            progress = (phase - 0.5) * 4  # 0 to 1 during this phase
            x1 = rect.x() + quarter_width
            y1 = centerY - wave_height
            x2 = rect.x() + (quarter_width * 2)
            x3 = rect.x() + (quarter_width * 2) + (quarter_width * progress)
            y3 = centerY - (wave_height * (1 - progress))
            path.lineTo(x1, y1)
            path.lineTo(x2, y1)
            path.lineTo(x3, y3)
            path.lineTo(x3, centerY)
            
        else:  # Hold after exhale
            # Phase 4: Hold breath out (baseline)
            progress = (phase - 0.75) * 4  # 0 to 1 during this phase
            x1 = rect.x() + quarter_width
            y1 = centerY - wave_height
            x2 = rect.x() + (quarter_width * 2)
            x3 = rect.x() + (quarter_width * 3)
            x4 = rect.x() + (quarter_width * 3) + (quarter_width * progress)
            path.lineTo(x1, y1)
            path.lineTo(x2, y1)
            path.lineTo(x3, centerY)
            path.lineTo(x4, centerY)
        
        path.lineTo(rect.x(), centerY)
        
        painter.drawPath(path)
    
    def process_network_data(self, network_data):
        """
        Process data received from the neural network
        
        Expected format:
        network_data = {
            "available_nodes": {
                "layer0.node0": {"activation": 0.5, "layer": 0, "node": 0},
                ...
            },
            "active_connections": [
                {"from_node": "layer0.node0", "to_node": "layer1.node0", "weight": 0.8},
                ...
            ],
            "network_status": "online",
            "breath_connection": True
        }
        """
        if not isinstance(network_data, dict) or not self.network_connected:
            return
            
        # Update local state based on network data
        if "breath_connection" in network_data:
            # Only update if there's a mismatch - prevents toggle loops
            if network_data["breath_connection"] != self.network_connected:
                self.toggle_network_connection(network_data["breath_connection"])
        
        # Apply network data to glyphs
        self.apply_network_data_to_glyphs(network_data)
    
    def apply_network_data_to_glyphs(self, network_data):
        """Apply neural network data to glyph visualization"""
        # In a real implementation, this would modify glyph properties
        # based on the neural network state
        
        # For this demonstration, we'll just update our status
        if "network_status" in network_data:
            status = network_data["network_status"]
            nodes_count = len(network_data.get("available_nodes", {}))
            connections_count = len(network_data.get("active_connections", []))
            
            status_text = f"Network: {status} | {nodes_count} nodes | {connections_count} connections"
            self.status_label.setText(status_text)
            
            # Trigger a redraw of the visualization
            self.visualization_frame.update()

import math  # Required for glyph drawing 