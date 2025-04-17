from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QComboBox, QScrollArea, QFrame, QLineEdit, QTextEdit,
                             QGroupBox, QGridLayout, QSlider, QTabWidget)
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QTimer
from PySide6.QtGui import QPainter, QBrush, QPen, QColor, QFont, QLinearGradient, QPainterPath, QImage, QRadialGradient

import math
import random

class GlyphCanvas(QWidget):
    """Canvas for displaying and interacting with spiritual glyphs"""
    
    glyph_activated = Signal(dict)  # Signal emitted when a glyph is activated
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.glyphs = self.generate_glyphs()
        self.active_glyph = None
        self.hover_glyph = None
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(50)
        self.animation_phase = 0
        self.energy_points = []
        self.setMouseTracking(True)
    
    def generate_glyphs(self):
        """Generate sample glyphs for the canvas"""
        glyph_types = [
            {"name": "Harmony", "color": QColor(100, 180, 255), "points": 5, "radius": 50},
            {"name": "Insight", "color": QColor(180, 130, 230), "points": 6, "radius": 60},
            {"name": "Balance", "color": QColor(255, 200, 100), "points": 4, "radius": 55},
            {"name": "Transcendence", "color": QColor(140, 230, 140), "points": 7, "radius": 65},
            {"name": "Connection", "color": QColor(255, 150, 150), "points": 3, "radius": 45}
        ]
        
        glyphs = []
        canvas_width = 400
        canvas_height = 300
        
        for i, glyph_type in enumerate(glyph_types):
            # Create glyph at different positions on the canvas
            x = 100 + (i % 3) * 120
            y = 80 + (i // 3) * 130
            
            glyph = {
                "name": glyph_type["name"],
                "color": glyph_type["color"],
                "points": glyph_type["points"],
                "radius": glyph_type["radius"],
                "x": x,
                "y": y,
                "rotation": random.uniform(0, math.pi/4),
                "energy": random.uniform(0.5, 1.0),
                "description": f"Represents spiritual {glyph_type['name'].lower()} and neural integration",
                "state": "inactive"
            }
            glyphs.append(glyph)
        
        return glyphs
    
    def update_animation(self):
        """Update animation state"""
        self.animation_phase += 0.05
        if self.animation_phase > 2 * math.pi:
            self.animation_phase -= 2 * math.pi
        
        # Occasionally add energy points
        if random.random() < 0.1:
            self.add_energy_point()
        
        # Update existing energy points
        for point in self.energy_points:
            point["life"] -= 0.02
            point["x"] += point["vx"]
            point["y"] += point["vy"]
            
            # Apply slight pull toward active glyph if one exists
            if self.active_glyph:
                dx = self.active_glyph["x"] - point["x"]
                dy = self.active_glyph["y"] - point["y"]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 0:
                    point["vx"] += dx / dist * 0.1
                    point["vy"] += dy / dist * 0.1
        
        # Remove dead points
        self.energy_points = [p for p in self.energy_points if p["life"] > 0]
        
        self.update()
    
    def add_energy_point(self):
        """Add a new energy point to the canvas"""
        x = random.uniform(0, self.width())
        y = random.uniform(0, self.height())
        
        self.energy_points.append({
            "x": x,
            "y": y,
            "vx": random.uniform(-0.5, 0.5),
            "vy": random.uniform(-0.5, 0.5),
            "color": QColor(
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(200, 255),
                150
            ),
            "life": 1.0,
            "size": random.uniform(2, 5)
        })
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events to detect hovering over glyphs"""
        x, y = event.x(), event.y()
        
        # Check if hovering over any glyph
        hover_found = False
        for glyph in self.glyphs:
            dx = glyph["x"] - x
            dy = glyph["y"] - y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < glyph["radius"]:
                self.hover_glyph = glyph
                hover_found = True
                self.setCursor(Qt.PointingHandCursor)
                break
        
        if not hover_found:
            self.hover_glyph = None
            self.setCursor(Qt.ArrowCursor)
        
        self.update()
    
    def mousePressEvent(self, event):
        """Handle mouse press events to activate glyphs"""
        if self.hover_glyph:
            # Deactivate previous active glyph
            if self.active_glyph:
                self.active_glyph["state"] = "inactive"
            
            # Set new active glyph
            self.active_glyph = self.hover_glyph
            self.active_glyph["state"] = "active"
            
            # Emit signal
            self.glyph_activated.emit({
                "name": self.active_glyph["name"],
                "description": self.active_glyph["description"]
            })
            
            # Add burst of energy points
            for _ in range(20):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 3)
                self.energy_points.append({
                    "x": self.active_glyph["x"],
                    "y": self.active_glyph["y"],
                    "vx": math.cos(angle) * speed,
                    "vy": math.sin(angle) * speed,
                    "color": QColor(
                        self.active_glyph["color"].red(),
                        self.active_glyph["color"].green(),
                        self.active_glyph["color"].blue(),
                        150
                    ),
                    "life": 1.0,
                    "size": random.uniform(2, 5)
                })
            
            self.update()
    
    def paintEvent(self, event):
        """Paint the glyph canvas"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        width = self.width()
        height = self.height()
        
        # Create background gradient
        gradient = QLinearGradient(0, 0, width, height)
        gradient.setColorAt(0, QColor(20, 20, 30))
        gradient.setColorAt(1, QColor(40, 40, 60))
        painter.fillRect(0, 0, width, height, gradient)
        
        # Draw energy points
        for point in self.energy_points:
            painter.setPen(Qt.NoPen)
            color = point["color"]
            color.setAlpha(int(255 * point["life"]))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(
                QPointF(point["x"], point["y"]),
                point["size"] * point["life"],
                point["size"] * point["life"]
            )
        
        # Draw connection lines between glyphs
        if self.active_glyph:
            for glyph in self.glyphs:
                if glyph != self.active_glyph:
                    # Draw connection line
                    painter.setPen(QPen(QColor(255, 255, 255, 40), 1, Qt.DashLine))
                    painter.drawLine(
                        self.active_glyph["x"],
                        self.active_glyph["y"],
                        glyph["x"],
                        glyph["y"]
                    )
        
        # Draw glyphs
        for glyph in self.glyphs:
            self.draw_glyph(painter, glyph)
    
    def draw_glyph(self, painter, glyph):
        """Draw a single glyph"""
        x, y = glyph["x"], glyph["y"]
        points = glyph["points"]
        radius = glyph["radius"]
        color = glyph["color"]
        rotation = glyph["rotation"]
        energy = glyph["energy"]
        state = glyph["state"]
        
        # Increase energy for active or hovered glyph
        display_energy = energy
        if glyph == self.active_glyph:
            display_energy = min(1.0, energy + 0.3 + 0.1 * math.sin(self.animation_phase * 2))
        elif glyph == self.hover_glyph:
            display_energy = min(1.0, energy + 0.2)
        
        # Create glyph path
        path = QPainterPath()
        
        # Calculate points on a regular polygon
        for i in range(points):
            angle = rotation + i * (2 * math.pi / points)
            px = x + radius * math.cos(angle) * (0.8 + 0.2 * math.sin(self.animation_phase + i))
            py = y + radius * math.sin(angle) * (0.8 + 0.2 * math.sin(self.animation_phase + i))
            
            if i == 0:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)
        
        path.closeSubpath()
        
        # Draw inner glow
        glow_radius = radius * 1.2 * display_energy
        radial_gradient = QRadialGradient(x, y, glow_radius)
        glow_color = QColor(color)
        glow_color.setAlpha(100)
        radial_gradient.setColorAt(0, glow_color)
        glow_color.setAlpha(0)
        radial_gradient.setColorAt(1, glow_color)
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(radial_gradient))
        painter.drawEllipse(x - glow_radius, y - glow_radius, glow_radius * 2, glow_radius * 2)
        
        # Draw glyph outline
        if state == "active":
            painter.setPen(QPen(QColor(255, 255, 255), 2))
        elif glyph == self.hover_glyph:
            painter.setPen(QPen(QColor(255, 255, 255, 180), 1.5))
        else:
            painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
        
        # Draw inner lines
        for i in range(points):
            for j in range(i+2, points):
                if j != i + points//2: # Don't draw diameter lines
                    angle1 = rotation + i * (2 * math.pi / points)
                    angle2 = rotation + j * (2 * math.pi / points)
                    
                    px1 = x + radius * 0.8 * math.cos(angle1)
                    py1 = y + radius * 0.8 * math.sin(angle1)
                    px2 = x + radius * 0.8 * math.cos(angle2)
                    py2 = y + radius * 0.8 * math.sin(angle2)
                    
                    line_color = QColor(color)
                    line_color.setAlpha(int(60 * display_energy))
                    painter.setPen(QPen(line_color, 0.5))
                    painter.drawLine(px1, py1, px2, py2)
        
        # Draw glyph shape with gradient fill
        gradient = QLinearGradient(x - radius, y - radius, x + radius, y + radius)
        fill_color1 = QColor(color)
        fill_color2 = QColor(
            min(255, color.red() + 30),
            min(255, color.green() + 30),
            min(255, color.blue() + 30)
        )
        
        if state == "active":
            fill_color1.setAlpha(150)
            fill_color2.setAlpha(180)
        else:
            fill_color1.setAlpha(100)
            fill_color2.setAlpha(130)
        
        gradient.setColorAt(0, fill_color1)
        gradient.setColorAt(1, fill_color2)
        
        painter.setBrush(QBrush(gradient))
        
        if state == "active":
            painter.setPen(QPen(QColor(255, 255, 255), 2))
        elif glyph == self.hover_glyph:
            painter.setPen(QPen(QColor(255, 255, 255, 180), 1.5))
        else:
            painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
        
        painter.drawPath(path)
        
        # Draw glyph center
        center_color = QColor(
            min(255, color.red() + 50),
            min(255, color.green() + 50),
            min(255, color.blue() + 50),
            int(200 * display_energy)
        )
        painter.setBrush(QBrush(center_color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(x - 5, y - 5, 10, 10)
        
        # Draw glyph name if active or hovered
        if glyph == self.active_glyph or glyph == self.hover_glyph:
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 9, QFont.Bold))
            text_rect = QRectF(x - 80, y + radius + 5, 160, 20)
            painter.drawText(
                text_rect,
                Qt.AlignCenter,
                glyph["name"]
            )


class MeditationGuideWidget(QWidget):
    """Widget for guided meditation and spiritual exercises"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        """Initialize the meditation guide UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("Neural Integration Meditation")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #E0E0E0;")
        layout.addWidget(title)
        
        # Description
        description = QLabel(
            "Connect with your neural network through guided meditation exercises "
            "that enhance pattern recognition and symbolic integration."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #BBBBBB; margin-bottom: 10px;")
        layout.addWidget(description)
        
        # Meditation tabs
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { 
                border: 1px solid #444444; 
                background-color: #303040;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #353545;
                color: #BBBBBB;
                padding: 6px 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #404050;
                color: #FFFFFF;
            }
            QTabBar::tab:hover:!selected {
                background-color: #404050;
            }
        """)
        
        # Guided Meditation Tab
        guided_tab = QWidget()
        guided_layout = QVBoxLayout(guided_tab)
        
        meditation_steps = [
            "1. Find a quiet place and sit comfortably",
            "2. Focus on your breath for 2 minutes",
            "3. Visualize your neural network as a constellation of light",
            "4. Observe patterns forming and dissolving",
            "5. Notice connections between symbolic glyphs",
            "6. Allow insights to emerge naturally",
            "7. When ready, slowly return awareness to your surroundings"
        ]
        
        for step in meditation_steps:
            step_label = QLabel(step)
            step_label.setStyleSheet("color: #D0D0D0; padding: 5px;")
            guided_layout.addWidget(step_label)
        
        start_button = QPushButton("Start Guided Session")
        start_button.setStyleSheet("""
            QPushButton {
                background-color: #4B6EAF;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5680C0;
            }
            QPushButton:pressed {
                background-color: #3A5A8C;
            }
        """)
        guided_layout.addWidget(start_button)
        guided_layout.addStretch()
        
        # Exercises Tab
        exercises_tab = QWidget()
        exercises_layout = QVBoxLayout(exercises_tab)
        
        exercises = [
            {
                "title": "Symbolic Resonance",
                "description": "Focus on a single glyph for 5 minutes, allowing its meaning to resonate with your neural patterns."
            },
            {
                "title": "Pattern Integration",
                "description": "Alternate between visual observation of neural activity and glyph meditation to strengthen connections."
            },
            {
                "title": "Memory Anchoring",
                "description": "Associate learned glyphs with specific neural patterns to create stable memory anchors."
            }
        ]
        
        for exercise in exercises:
            exercise_group = QGroupBox(exercise["title"])
            exercise_group.setStyleSheet("""
                QGroupBox {
                    color: #CCCCCC;
                    font-weight: bold;
                    border: 1px solid #444444;
                    border-radius: 4px;
                    margin-top: 12px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    padding: 0 5px;
                }
            """)
            
            group_layout = QVBoxLayout(exercise_group)
            
            desc = QLabel(exercise["description"])
            desc.setWordWrap(True)
            desc.setStyleSheet("color: #BBBBBB;")
            group_layout.addWidget(desc)
            
            begin_btn = QPushButton("Begin Exercise")
            begin_btn.setStyleSheet("""
                QPushButton {
                    background-color: #406040;
                    color: white;
                    border: none;
                    padding: 6px;
                    border-radius: 3px;
                    margin-top: 5px;
                }
                QPushButton:hover {
                    background-color: #507050;
                }
                QPushButton:pressed {
                    background-color: #305030;
                }
            """)
            group_layout.addWidget(begin_btn)
            
            exercises_layout.addWidget(exercise_group)
        
        exercises_layout.addStretch()
        
        # Journal Tab
        journal_tab = QWidget()
        journal_layout = QVBoxLayout(journal_tab)
        
        journal_label = QLabel("Record your insights and experiences:")
        journal_label.setStyleSheet("color: #D0D0D0;")
        journal_layout.addWidget(journal_label)
        
        self.journal_text = QTextEdit()
        self.journal_text.setStyleSheet("""
            QTextEdit {
                background-color: #353545;
                color: #E0E0E0;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 5px;
            }
        """)
        journal_layout.addWidget(self.journal_text)
        
        save_button = QPushButton("Save Entry")
        save_button.setStyleSheet("""
            QPushButton {
                background-color: #4B6EAF;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5680C0;
            }
            QPushButton:pressed {
                background-color: #3A5A8C;
            }
        """)
        journal_layout.addWidget(save_button)
        
        # Add tabs
        tabs.addTab(guided_tab, "Guided Meditation")
        tabs.addTab(exercises_tab, "Integration Exercises")
        tabs.addTab(journal_tab, "Insight Journal")
        
        layout.addWidget(tabs)


class SpiritualGuidancePanel(QWidget):
    """Panel for exploring spiritual dimensions of neural network learning"""
    
    insight_discovered = Signal(dict)  # Signal emitted when an insight is discovered
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        """Initialize the user interface"""
        # Set dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #252535;
                color: #E0E0E0;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        
        # Header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 5)
        
        title = QLabel("Spiritual Guidance & Integration")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #E0E0E0;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        main_layout.addWidget(header)
        
        # Main content - split into glyph canvas and guidance area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)
        
        # Left side - Glyph Canvas
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Glyph explanation
        glyph_label = QLabel("Sacred Glyphs")
        glyph_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #E0E0E0;")
        left_layout.addWidget(glyph_label)
        
        glyph_desc = QLabel("Interact with symbolic glyphs to discover neural-spiritual connections")
        glyph_desc.setWordWrap(True)
        glyph_desc.setStyleSheet("color: #BBBBBB;")
        left_layout.addWidget(glyph_desc)
        
        # Glyph canvas
        self.glyph_canvas = GlyphCanvas()
        self.glyph_canvas.glyph_activated.connect(self.on_glyph_activated)
        left_layout.addWidget(self.glyph_canvas)
        
        # Current glyph info
        self.glyph_info = QFrame()
        self.glyph_info.setFrameShape(QFrame.StyledPanel)
        self.glyph_info.setStyleSheet("""
            QFrame {
                background-color: #353545;
                border-radius: 4px;
                padding: 10px;
            }
        """)
        
        glyph_info_layout = QVBoxLayout(self.glyph_info)
        glyph_info_layout.setContentsMargins(10, 10, 10, 10)
        
        self.selected_glyph_title = QLabel("Select a glyph above")
        self.selected_glyph_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #E0E0E0;")
        glyph_info_layout.addWidget(self.selected_glyph_title)
        
        self.selected_glyph_desc = QLabel("")
        self.selected_glyph_desc.setWordWrap(True)
        self.selected_glyph_desc.setStyleSheet("color: #BBBBBB;")
        glyph_info_layout.addWidget(self.selected_glyph_desc)
        
        self.insight_label = QLabel("Neural Insight:")
        self.insight_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #E0E0E0; margin-top: 10px;")
        self.insight_label.setVisible(False)
        glyph_info_layout.addWidget(self.insight_label)
        
        self.insight_text = QLabel("")
        self.insight_text.setWordWrap(True)
        self.insight_text.setStyleSheet("color: #D0D0FF; background-color: #33335A; padding: 8px; border-radius: 4px;")
        self.insight_text.setVisible(False)
        glyph_info_layout.addWidget(self.insight_text)
        
        left_layout.addWidget(self.glyph_info)
        
        # Right side - Meditation Guide
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.meditation_guide = MeditationGuideWidget()
        right_layout.addWidget(self.meditation_guide)
        
        # Add panels to content layout
        content_layout.addWidget(left_panel, 1)
        content_layout.addWidget(right_panel, 1)
        
        main_layout.addLayout(content_layout)
        
        # Footer
        footer = QFrame()
        footer.setFrameShape(QFrame.StyledPanel)
        footer.setStyleSheet("""
            QFrame {
                background-color: #303040;
                border-radius: 4px;
            }
        """)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(10, 5, 10, 5)
        
        self.status_label = QLabel("Integrate neural patterns with spiritual understanding")
        self.status_label.setStyleSheet("color: #BBBBBB;")
        footer_layout.addWidget(self.status_label)
        
        main_layout.addWidget(footer)
    
    def on_glyph_activated(self, glyph_data):
        """Handle glyph activation event"""
        # Update glyph info
        self.selected_glyph_title.setText(glyph_data["name"])
        self.selected_glyph_desc.setText(glyph_data["description"])
        
        # Simulate insight after a delay
        QTimer.singleShot(2000, lambda: self.generate_insight(glyph_data))
        
        self.status_label.setText(f"Exploring the {glyph_data['name']} glyph's neural connections...")
    
    def generate_insight(self, glyph_data):
        """Generate an insight based on the activated glyph"""
        insights = {
            "Harmony": "Neural patterns show increased coherence when visualization follows a harmonic structure. Consider implementing resonance optimization in your training algorithm.",
            "Insight": "Metacognitive processing increases when symbolic representation aligns with neural architecture. Try incorporating symbolic pathways in your neural network design.",
            "Balance": "Balanced training sequences show improved retention and generalization. Consider implementing equilibrium-based regularization in your neural network.",
            "Transcendence": "Cross-modal integration creates emergent properties beyond individual pattern recognition. Explore implementing higher-order abstraction layers.",
            "Connection": "Social-symbolic processing areas activate when glyphs are perceived in relational contexts. Try implementing relational inference mechanisms."
        }
        
        insight = insights.get(glyph_data["name"], "Interesting patterns detected in neural activity.")
        
        # Show insight
        self.insight_label.setVisible(True)
        self.insight_text.setText(insight)
        self.insight_text.setVisible(True)
        
        # Emit signal
        self.insight_discovered.emit({
            "glyph": glyph_data["name"],
            "insight": insight,
            "timestamp": "Now"
        })
        
        self.status_label.setText(f"New neural insight discovered through the {glyph_data['name']} glyph!") 