"""
V6 Main Widget

The central widget for the V6 Portal of Contradiction Visualization System,
with a modern holographic appearance and balanced layout.
"""

import os
import sys
import logging
import math  # Add math import for cos and sin
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    # Import Qt compatibility layer from V5
    from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui, Qt, Signal, Slot
    from src.v5.ui.qt_compat import get_widgets, get_gui, get_core
except ImportError:
    logging.warning("V5 Qt compatibility layer not found. Using direct PySide6 imports.")
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        from PySide6.QtCore import Qt, Signal, Slot
        
        # Simple compatibility functions
        def get_widgets():
            return QtWidgets
            
        def get_gui():
            return QtGui
            
        def get_core():
            return QtCore
    except ImportError:
        logging.error("PySide6 not found. Please install PySide6 or configure the V5 Qt compatibility layer.")
        sys.exit(1)

# Get required Qt classes
QSplitter = get_widgets().QSplitter
QPainter = get_gui().QPainter
QLinearGradient = get_gui().QLinearGradient
QRadialGradient = get_gui().QRadialGradient
QColor = get_gui().QColor
QFont = get_gui().QFont
QPen = get_gui().QPen

# In some Qt versions, QGraphicsDropShadowEffect is in QtWidgets not QtGui
try:
    QGraphicsDropShadowEffect = get_gui().QGraphicsDropShadowEffect
except AttributeError:
    QGraphicsDropShadowEffect = get_widgets().QGraphicsDropShadowEffect

# Import the V6 panel base classes
from .panel_base import V6PanelBase, V6PanelContainer

# Import panels
from .panels.chat_panel import ChatPanel
from .panels.duality_processor_panel import DualityProcessorPanel
from .panels.memory_reflection_panel import MemoryReflectionPanel
from .panels.integrated_visualization_panel import IntegratedVisualizationPanel
from .panels.language_module_panel import LanguageModulePanel

# Set up logging
logger = logging.getLogger(__name__)

class HolographicPlaceholder(V6PanelBase):
    """
    A placeholder panel for visualization areas with holographic rendering effects
    """
    def __init__(self, title, placeholder_text="Initializing visualization...", parent=None):
        super().__init__(parent)
        self.title = title
        self.placeholder_text = placeholder_text
        self.particles = []
        self.nodes = []
        self.links = []
        self.generate_particles(50)  # Create 50 particles
        self.generate_nodes(8)  # Create 8 data nodes
        self.hover_node = None
        self.active = False
        
        # Animation timer
        self.animation_timer = QtCore.QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(50)  # 50ms update = ~20fps
        
        # Activate after a delay
        QtCore.QTimer.singleShot(1500, self.activate)
        
        # Install event filter for mouse tracking
        self.setMouseTracking(True)
    
    def activate(self):
        """Activate the visualization with animation"""
        self.active = True
        self.update()
    
    def generate_particles(self, count):
        """Generate random particles for the holographic effect"""
        for _ in range(count):
            particle = {
                'x': QtCore.QRandomGenerator.global_().bounded(self.width() or 300),
                'y': QtCore.QRandomGenerator.global_().bounded(self.height() or 200),
                'size': QtCore.QRandomGenerator.global_().bounded(3, 8),
                'speed_x': (QtCore.QRandomGenerator.global_().bounded(20) - 10) / 10,
                'speed_y': (QtCore.QRandomGenerator.global_().bounded(20) - 10) / 10,
                'opacity': QtCore.QRandomGenerator.global_().bounded(40, 100) / 100,
                'color': QColor(
                    QtCore.QRandomGenerator.global_().bounded(100, 200),
                    QtCore.QRandomGenerator.global_().bounded(150, 255),
                    QtCore.QRandomGenerator.global_().bounded(200, 255),
                    100
                )
            }
            self.particles.append(particle)
    
    def generate_nodes(self, count):
        """Generate data nodes for the visualization"""
        center_x = self.width() / 2 if self.width() else 150
        center_y = self.height() / 2 if self.height() else 100
        
        for i in range(count):
            angle = i * math.pi * 2 / count
            distance = 100 + (i % 3) * 30
            
            node = {
                'x': center_x + distance * math.cos(angle),
                'y': center_y + distance * math.sin(angle),
                'size': 10 + QtCore.QRandomGenerator.global_().bounded(10),
                'color': QColor(
                    52 + QtCore.QRandomGenerator.global_().bounded(50),
                    152 + QtCore.QRandomGenerator.global_().bounded(50), 
                    219 - QtCore.QRandomGenerator.global_().bounded(40),
                    180
                ),
                'pulse': QtCore.QRandomGenerator.global_().bounded(100) / 100,
                'pulse_dir': 1 if QtCore.QRandomGenerator.global_().bounded(2) else -1,
                'label': f"Node-{chr(65+i)}",
                'value': QtCore.QRandomGenerator.global_().bounded(100)
            }
            self.nodes.append(node)
        
        # Create links between nodes (not all nodes are connected)
        for i in range(len(self.nodes)):
            for j in range(i+1, len(self.nodes)):
                # Don't connect all points, only some
                if (i + j) % 3 == 0:
                    self.links.append({
                        'source': i,
                        'target': j,
                        'strength': QtCore.QRandomGenerator.global_().bounded(30, 100) / 100,
                        'active': QtCore.QRandomGenerator.global_().bounded(2) == 1
                    })
    
    def update_animation(self):
        """Update all animated elements"""
        # Update particles
        for particle in self.particles:
            # Update position
            particle['x'] += particle['speed_x']
            particle['y'] += particle['speed_y']
            
            # Bounce off edges
            if particle['x'] <= 0 or particle['x'] >= (self.width() or 300):
                particle['speed_x'] *= -1
            if particle['y'] <= 0 or particle['y'] >= (self.height() or 200):
                particle['speed_y'] *= -1
            
            # Update opacity for pulsing effect
            particle['opacity'] += (QtCore.QRandomGenerator.global_().bounded(20) - 10) / 200
            particle['opacity'] = max(0.4, min(1.0, particle['opacity']))
            
            # Update color alpha
            particle['color'].setAlpha(int(particle['opacity'] * 100))
        
        # Update nodes
        for node in self.nodes:
            # Pulsing effect
            node['pulse'] += 0.03 * node['pulse_dir']
            if node['pulse'] >= 1.0:
                node['pulse'] = 1.0
                node['pulse_dir'] = -1
            elif node['pulse'] <= 0.5:
                node['pulse'] = 0.5
                node['pulse_dir'] = 1
            
            # Occasionally change value 
            if QtCore.QRandomGenerator.global_().bounded(100) < 5:  # 5% chance
                # Fix: Use random bounded value and then shift it to get negative range
                random_change = QtCore.QRandomGenerator.global_().bounded(21) - 10
                node['value'] = max(0, min(100, node['value'] + random_change))
        
        # Update links
        for link in self.links:
            # Occasionally toggle active state
            if QtCore.QRandomGenerator.global_().bounded(100) < 2:  # 2% chance
                link['active'] = not link['active']
        
        # Trigger repaint
        self.update()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for interactivity"""
        # Check if mouse is over any node
        old_hover = self.hover_node
        self.hover_node = None
        
        for i, node in enumerate(self.nodes):
            distance = math.sqrt((event.x() - node['x'])**2 + (event.y() - node['y'])**2)
            if distance < node['size'] + 5:
                self.hover_node = i
                break
        
        if old_hover != self.hover_node:
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse click events"""
        if self.hover_node is not None:
            # Toggle links connected to this node
            for link in self.links:
                if link['source'] == self.hover_node or link['target'] == self.hover_node:
                    link['active'] = not link['active']
            self.update()
    
    def resizeEvent(self, event):
        """Handle resize events to reposition elements"""
        super().resizeEvent(event)
        
        # Reposition nodes on resize
        center_x = self.width() / 2
        center_y = self.height() / 2
        
        for i, node in enumerate(self.nodes):
            angle = i * math.pi * 2 / len(self.nodes)
            distance = 100 + (i % 3) * 30
            node['x'] = center_x + distance * math.cos(angle)
            node['y'] = center_y + distance * math.sin(angle)
    
    def paintEvent(self, event):
        """Custom paint event for holographic visualization"""
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        center_x = self.width() / 2
        center_y = self.height() / 2
        
        # Draw particles
        for particle in self.particles:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QtGui.QBrush(particle['color']))
            painter.drawEllipse(
                particle['x'] - particle['size'] / 2,
                particle['y'] - particle['size'] / 2,
                particle['size'],
                particle['size']
            )
        
        if not self.active:
            # Draw center glow
            radius = min(center_x, center_y) * 0.8
            
            # Create a radial gradient for the center glow
            gradient = QRadialGradient(center_x, center_y, radius)
            gradient.setColorAt(0, QColor(52, 152, 219, 40))  # Semi-transparent blue
            gradient.setColorAt(1, QColor(52, 152, 219, 0))   # Transparent
            
            painter.setBrush(QtGui.QBrush(gradient))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
            
            # Draw title text
            font = QFont("Segoe UI", 24, QFont.Bold)
            painter.setFont(font)
            
            # Create a glow effect for the text
            text_path = QtGui.QPainterPath()
            text_path.addText(center_x - 150, center_y, font, self.title)
            
            # Draw text glow
            glow_pen = QPen(QColor(52, 152, 219, 50), 3)
            painter.setPen(glow_pen)
            painter.drawPath(text_path)
            
            # Draw text
            painter.setPen(QColor(240, 240, 240, 220))
            painter.drawPath(text_path)
            
            # Draw placeholder text
            font = QFont("Segoe UI", 12)
            painter.setFont(font)
            painter.setPen(QColor(200, 200, 200, 180))
            painter.drawText(
                center_x - 150,
                center_y + 40,
                300,
                50,
                Qt.AlignCenter,
                self.placeholder_text
            )
        else:
            # Draw connecting lines first (behind nodes)
            for link in self.links:
                source = self.nodes[link['source']]
                target = self.nodes[link['target']]
                
                if link['active']:
                    # Active link
                    line_color = QColor(52, 152, 219, int(120 * link['strength']))
                    line_width = 2
                else:
                    # Inactive link
                    line_color = QColor(52, 152, 219, int(40 * link['strength']))
                    line_width = 1
                
                pen = QPen(line_color, line_width)
                if link['active']:
                    pen.setStyle(Qt.SolidLine)
                else:
                    pen.setStyle(Qt.DotLine)
                    
                painter.setPen(pen)
                painter.drawLine(source['x'], source['y'], target['x'], target['y'])
            
            # Draw nodes
            for i, node in enumerate(self.nodes):
                # Node glow
                if i == self.hover_node:
                    # Stronger glow for hovered node
                    glow_radius = node['size'] + 10
                    glow = QRadialGradient(node['x'], node['y'], glow_radius)
                    glow.setColorAt(0, QColor(52, 152, 219, 150))
                    glow.setColorAt(1, QColor(52, 152, 219, 0))
                    painter.setBrush(QtGui.QBrush(glow))
                    painter.setPen(Qt.NoPen)
                    painter.drawEllipse(
                        node['x'] - glow_radius, 
                        node['y'] - glow_radius,
                        glow_radius * 2, 
                        glow_radius * 2
                    )
                
                # Node circle
                size = node['size'] * (0.8 + 0.4 * node['pulse'])
                color = QColor(node['color'])
                
                # Make hovered node brighter
                if i == self.hover_node:
                    color = color.lighter(130)
                
                painter.setBrush(QtGui.QBrush(color))
                painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
                painter.drawEllipse(
                    node['x'] - size/2, 
                    node['y'] - size/2,
                    size, 
                    size
                )
                
                # Node label
                if i == self.hover_node:
                    # Draw label for hovered node
                    font = QFont("Segoe UI", 9)
                    painter.setFont(font)
                    painter.setPen(QColor(255, 255, 255, 220))
                    
                    # Draw background for label
                    text_rect = painter.boundingRect(
                        QtCore.QRect(0, 0, 100, 20), 
                        Qt.AlignCenter, 
                        f"{node['label']}: {node['value']}%"
                    )
                    
                    # Position the label above the node
                    text_rect.moveCenter(QtCore.QPoint(int(node['x']), int(node['y'] - size/2 - 15)))
                    
                    # Draw background
                    painter.fillRect(
                        text_rect.adjusted(-5, -3, 5, 3), 
                        QColor(44, 62, 80, 200)
                    )
                    
                    # Draw text
                    painter.drawText(
                        text_rect,
                        Qt.AlignCenter, 
                        f"{node['label']}: {node['value']}%"
                    )
            
            # Draw title in corner instead of center when active
            font = QFont("Segoe UI", 14, QFont.Bold)
            painter.setFont(font)
            painter.setPen(QColor(52, 152, 219, 220))
            painter.drawText(15, 25, self.title)

class V6MainWidget(QtWidgets.QWidget):
    """
    Main widget for the V6 Portal of Contradiction Visualization System.
    
    This widget organizes all the V6 UI components with improved spacing and sizing,
    and a modern holographic appearance.
    """
    
    def __init__(self, socket_manager=None):
        """Initialize the V6 main widget"""
        super().__init__()
        
        # Store socket manager
        self.socket_manager = socket_manager
        
        # Initialize UI
        self.initUI()
    
    def initUI(self):
        """Initialize the user interface"""
        # Set up the main layout with improved spacing
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create and add toolbar
        toolbar = self.createToolbar()
        layout.addWidget(toolbar)
        
        # Main content area
        content_widget = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content_widget)
        content_layout.setContentsMargins(15, 15, 15, 15)  # Slightly reduced margins
        content_layout.setSpacing(15)  # Reduced spacing between panels
        
        # Create the layout containing all the panels
        panels_widget = self.createPanelsLayout()
        content_layout.addWidget(panels_widget)
        
        # Add the content widget to the main layout
        layout.addWidget(content_widget)
        
        # Create and add status bar
        status_bar = self.createStatusBar()
        layout.addWidget(status_bar)
    
    def createToolbar(self):
        """Create the main toolbar with holographic styling"""
        toolbar = QtWidgets.QWidget()
        toolbar.setFixedHeight(60)
        toolbar.setStyleSheet("""
            background-color: rgba(26, 38, 52, 180);
            border-bottom: 1px solid rgba(52, 73, 94, 120);
        """)
        
        layout = QtWidgets.QHBoxLayout(toolbar)
        layout.setContentsMargins(20, 0, 20, 0)
        
        # Add title with glow effect
        title = QtWidgets.QLabel("V6 Portal of Contradiction")
        title.setStyleSheet("""
            color: #3498DB;
            font-weight: bold;
            font-size: 18px;
        """)
        
        # Create glow effect
        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(15)
        glow.setColor(QColor(52, 152, 219, 150))
        glow.setOffset(0, 0)
        title.setGraphicsEffect(glow)
        
        # Add control buttons with improved styling
        refresh_button = QtWidgets.QPushButton("Refresh")
        refresh_button.setMinimumSize(100, 36)
        refresh_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(41, 128, 185, 180);
                color: white;
                border-radius: 4px;
                padding: 8px 15px;
                font-size: 14px;
                border: 1px solid rgba(52, 152, 219, 120);
            }
            QPushButton:hover {
                background-color: rgba(52, 152, 219, 200);
                border: 1px solid rgba(52, 152, 219, 180);
            }
        """)
        refresh_button.clicked.connect(self.refreshAll)
        
        settings_button = QtWidgets.QPushButton("Settings")
        settings_button.setMinimumSize(100, 36)
        settings_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(44, 62, 80, 180);
                color: white;
                border-radius: 4px;
                padding: 8px 15px;
                font-size: 14px;
                border: 1px solid rgba(52, 73, 94, 120);
            }
            QPushButton:hover {
                background-color: rgba(52, 73, 94, 200);
                border: 1px solid rgba(52, 73, 94, 180);
            }
        """)
        settings_button.clicked.connect(self.openSettings)
        
        # Add buttons to layout
        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(refresh_button)
        layout.addWidget(settings_button)
        
        return toolbar
    
    def createStatusBar(self):
        """Create the status bar with holographic styling"""
        status_bar = QtWidgets.QWidget()
        status_bar.setFixedHeight(30)
        status_bar.setStyleSheet("""
            background-color: rgba(26, 38, 52, 180);
            border-top: 1px solid rgba(52, 73, 94, 120);
        """)
        
        layout = QtWidgets.QHBoxLayout(status_bar)
        layout.setContentsMargins(20, 0, 20, 0)
        
        # Status text
        status_text = QtWidgets.QLabel("System Ready | Data Streams Active")
        status_text.setStyleSheet("""
            color: rgba(127, 140, 141, 220);
            font-size: 12px;
        """)
        
        # System info
        system_text = QtWidgets.QLabel("Portal System v6.0.2")
        system_text.setStyleSheet("""
            color: rgba(127, 140, 141, 220);
            font-size: 12px;
        """)
        system_text.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        layout.addWidget(status_text)
        layout.addStretch()
        layout.addWidget(system_text)
        
        return status_bar
    
    def createPanelsLayout(self):
        """Create the layout containing all the panels with holographic styling"""
        # Main widget with horizontal splitter
        panels_widget = QtWidgets.QWidget()
        panels_layout = QtWidgets.QHBoxLayout(panels_widget)
        panels_layout.setContentsMargins(0, 0, 0, 0)
        panels_layout.setSpacing(15)  # Reduced spacing between columns
        
        # Create main horizontal splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setChildrenCollapsible(False)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: rgba(52, 73, 94, 150);
            }
        """)
        
        # Left column - Breath State, Active Glyph, and Analytics
        left_widget = self.createLeftColumn()
        
        # Center column - Mirror Mode and Chat
        center_widget = self.createCenterColumn()
        
        # Right column - Echo Threads, Mythos Generator and Node Embodiment
        right_widget = self.createRightColumn()
        
        # Add columns to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(center_widget)
        splitter.addWidget(right_widget)
        
        # Set initial sizes - more balanced ratio (1:1:1.2)
        splitter.setSizes([450, 450, 540])
        
        # Add splitter to layout
        panels_layout.addWidget(splitter)
        
        return panels_widget
    
    def createLeftColumn(self):
        """Create the left column of panels"""
        panels_container = V6PanelContainer("Left Column")
        
        # Tabs for the left column
        language_panel = LanguageModulePanel(self.socket_manager)
        duality_panel = DualityProcessorPanel(self.socket_manager)
        memory_panel = MemoryReflectionPanel(self.socket_manager)
        
        # Add panels to container
        panels_container.addPanel("Language", language_panel)
        panels_container.addPanel("Duality", duality_panel)
        panels_container.addPanel("Memory", memory_panel)
        
        return panels_container
    
    def createCenterColumn(self):
        """Create the center column of panels with chat at the bottom"""
        # Create widget and layout
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)  # Reduced spacing between panels
        
        # Create vertical splitter
        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(2)
        splitter.setChildrenCollapsible(False)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: rgba(52, 73, 94, 150);
            }
        """)
        
        # Create integrated visualization panel replacing mirror mode
        integrated_viz = IntegratedVisualizationPanel(self.socket_manager)
        viz_container = V6PanelContainer("Integrated Visualization", {"Visualization": integrated_viz})
        
        # Create chat panel 
        self.chat_panel = ChatPanel(self.socket_manager)
        chat_container = V6PanelContainer("Portal Conversation", {"Chat": self.chat_panel})
        
        # Add widgets to main center splitter
        splitter.addWidget(viz_container)
        splitter.addWidget(chat_container)
        
        # Set initial sizes - balanced split for chat (65/35)
        splitter.setSizes([650, 350])
        
        # Add splitter to layout
        layout.addWidget(splitter)
        
        return widget
    
    def createRightColumn(self):
        """Create the right column of panels"""
        # Create widget and layout
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)  # Reduced spacing between panels
        
        # Create vertical splitter
        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(2)
        splitter.setChildrenCollapsible(False)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: rgba(52, 73, 94, 150);
            }
        """)
        
        # Create panels with visualization placeholders
        echo_container = self.createRecursiveEchoPanels()
        mythos_container = self.createMythosPanels()
        embodiment_container = self.createEmbodimentPanels()
        
        # Add panels to splitter
        splitter.addWidget(echo_container)
        splitter.addWidget(mythos_container)
        splitter.addWidget(embodiment_container)
        
        # Set initial sizes - divide space evenly
        splitter.setSizes([333, 333, 334])
        
        # Add splitter to layout
        layout.addWidget(splitter)
        
        return widget
    
    def createBreathStatePanels(self):
        """Create the Breath State Integration panels with visualization placeholders"""
        # Create visualization placeholders for different aspects
        main_panel = HolographicPlaceholder("Breath Visualization", "Initializing breath pattern visualization...")
        rhythm_panel = HolographicPlaceholder("Rhythm Analysis", "Loading breath rhythm analysis module...")
        pattern_panel = HolographicPlaceholder("Pattern Modulation", "Connecting to pattern modulation engine...")
        settings_panel = self.createPlaceholderPanel("Breath Settings")
        
        # Create the container with multiple tabs
        panels = {
            "Visualization": main_panel,
            "Rhythm": rhythm_panel,
            "Pattern Mod": pattern_panel,
            "Settings": settings_panel
        }
        container = V6PanelContainer("Breath-State Integration", panels)
        
        return container
    
    def createGlyphFieldPanels(self):
        """Create the Active Glyph Field panels with visualization placeholders"""
        # Create visualization placeholders for different aspects
        main_panel = HolographicPlaceholder("Glyph Field", "Preparing glyph field visualization...")
        layer_panel = HolographicPlaceholder("Emotional Layers", "Initializing emotional layer rendering...")
        symbol_panel = HolographicPlaceholder("Symbol Activation", "Loading symbol activation patterns...")
        history_panel = self.createPlaceholderPanel("Activation History")
        
        # Create the container with multiple tabs
        panels = {
            "Field": main_panel,
            "Layers": layer_panel,
            "Symbols": symbol_panel,
            "History": history_panel
        }
        container = V6PanelContainer("Active Glyph Field", panels)
        
        return container
    
    def createMirrorModePanels(self):
        """Create the Mirror Mode / Contradiction panels with visualization placeholders"""
        # Create visualization placeholders for different aspects
        main_panel = HolographicPlaceholder("Mirror Mode", "Initializing mirror reflection patterns...")
        contradiction_panel = HolographicPlaceholder("Contradiction", "Loading contradiction analysis engine...")
        # Use actual Duality Processor panel instead of placeholder
        duality_panel = DualityProcessorPanel(self.socket_manager)
        path_panel = self.createPlaceholderPanel("Resolution Pathways")
        
        # Create the container with multiple tabs
        panels = {
            "Mirror": main_panel,
            "Contradictions": contradiction_panel,
            "Duality": duality_panel,
            "Resolutions": path_panel
        }
        container = V6PanelContainer("Mirror Mode / Contradiction", panels)
        
        return container
    
    def createRecursiveEchoPanels(self):
        """Create the Recursive Echo Thread panels with visualization placeholders"""
        # Create visualization placeholders for different aspects
        main_panel = HolographicPlaceholder("Echo Threads", "Preparing recursive echo visualization...")
        # Use actual Memory Reflection panel instead of placeholder
        memory_panel = MemoryReflectionPanel(self.socket_manager)
        resonance_panel = HolographicPlaceholder("Emotional Resonance", "Initializing resonance field mapping...")
        trace_panel = self.createPlaceholderPanel("Thread Trace")
        
        # Create the container with multiple tabs
        panels = {
            "Echo": main_panel,
            "Memory": memory_panel,
            "Resonance": resonance_panel,
            "Trace": trace_panel
        }
        container = V6PanelContainer("Recursive Echo Thread", panels)
        
        return container
    
    def createMythosPanels(self):
        """Create the Mythos Generator panels with visualization placeholders"""
        # Create visualization placeholders for different aspects
        main_panel = HolographicPlaceholder("Mythos Creation", "Initializing mythological pattern synthesis...")
        fragment_panel = HolographicPlaceholder("Session Fragments", "Loading fragment collection manager...")
        archetypal_panel = HolographicPlaceholder("Archetypes", "Preparing archetypal pattern visualization...")
        library_panel = self.createPlaceholderPanel("Narrative Library")
        
        # Create the container with multiple tabs
        panels = {
            "Create": main_panel,
            "Fragments": fragment_panel,
            "Archetypes": archetypal_panel,
            "Library": library_panel
        }
        container = V6PanelContainer("Mythos Generator", panels)
        
        return container
    
    def createEmbodimentPanels(self):
        """Create the Node Embodiment panels with visualization placeholders"""
        # Create visualization placeholders for different aspects
        main_panel = HolographicPlaceholder("Node Transitions", "Preparing node transition visualization...")
        state_panel = HolographicPlaceholder("State Manager", "Initializing state transformation engine...")
        node_panel = HolographicPlaceholder("Node Network", "Loading nodal connection patterns...")
        preference_panel = self.createPlaceholderPanel("User Preferences")
        
        # Create the container with multiple tabs
        panels = {
            "Transitions": main_panel,
            "States": state_panel,
            "Nodes": node_panel,
            "Preferences": preference_panel
        }
        container = V6PanelContainer("Node Embodiment", panels)
        
        return container
    
    def createPlaceholderPanel(self, title):
        """Create a placeholder panel for settings and non-visualization panels"""
        panel = V6PanelBase()
        
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignCenter)
        
        # Create icon
        icon_label = QtWidgets.QLabel()
        icon_label.setFixedSize(70, 70)  # Slightly smaller
        icon_label.setStyleSheet("""
            background-color: rgba(155, 89, 182, 180);
            border-radius: 35px;
            color: white;
            font-size: 32px;
            font-weight: bold;
            text-align: center;
        """)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setText("V6")
        
        # Add glow effect to icon
        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(15)
        glow.setColor(QColor(155, 89, 182, 150))
        glow.setOffset(0, 0)
        icon_label.setGraphicsEffect(glow)
        
        # Create title and description
        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("""
            color: #ECF0F1;
            font-size: 18px;
            font-weight: bold;
            margin-top: 15px;
        """)
        
        desc_label = QtWidgets.QLabel(f"This {title} view is initializing...")
        desc_label.setStyleSheet("""
            color: rgba(127, 140, 141, 220);
            font-size: 14px;
            margin-top: 10px;
        """)
        
        # Add to layout
        layout.addWidget(icon_label, 0, Qt.AlignCenter)
        layout.addWidget(title_label, 0, Qt.AlignCenter)
        layout.addWidget(desc_label, 0, Qt.AlignCenter)
        
        return panel
    
    def createAnalyticsPanels(self):
        """Create the Analytics panels with visualization placeholders"""
        # Create visualization placeholders for different aspects
        main_panel = HolographicPlaceholder("Data Analytics", "Initializing pattern analytics visualization...")
        correlation_panel = HolographicPlaceholder("Correlations", "Processing data correlation patterns...")
        insight_panel = HolographicPlaceholder("Insights", "Generating symbolic insight matrix...")
        metrics_panel = self.createPlaceholderPanel("Metrics Dashboard")
        
        # Create the container with multiple tabs
        panels = {
            "Overview": main_panel,
            "Correlations": correlation_panel,
            "Insights": insight_panel,
            "Metrics": metrics_panel
        }
        container = V6PanelContainer("Symbolic Analytics", panels)
        
        return container
    
    def refreshAll(self):
        """Refresh all panels"""
        logger.info("Refreshing all V6 panels")
        # This would update all panels in a real implementation
    
    def paintEvent(self, event):
        """Custom paint event for holographic background"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill with gradient background
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(22, 33, 51, 255))  # Dark blue at top
        gradient.setColorAt(1, QColor(16, 26, 40, 255))  # Darker blue at bottom
        
        painter.fillRect(self.rect(), gradient)
        
        # Add subtle grid pattern for holographic effect
        self._draw_holographic_grid(painter)
    
    def _draw_holographic_grid(self, painter):
        """Draw a subtle grid pattern for holographic effect"""
        # Set up the pen for grid lines
        pen = QPen(QColor(52, 152, 219, 15))  # Very transparent blue
        pen.setWidth(1)
        painter.setPen(pen)
        
        # Draw horizontal lines
        spacing = 30
        for y in range(0, self.height(), spacing):
            painter.drawLine(0, y, self.width(), y)
        
        # Draw vertical lines
        for x in range(0, self.width(), spacing):
            painter.drawLine(x, 0, x, self.height())
    
    def openSettings(self):
        """Open the settings dialog"""
        logger.info("Opening V6 settings dialog")
        
        # Create a basic settings dialog
        settings_dialog = QtWidgets.QDialog(self.window())
        settings_dialog.setWindowTitle("V6 Portal Settings")
        settings_dialog.setMinimumSize(600, 400)
        settings_dialog.setStyleSheet("""
            background-color: rgba(26, 38, 52, 240);
            color: #ECF0F1;
        """)
        
        # Dialog layout
        layout = QtWidgets.QVBoxLayout(settings_dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title with glow effect
        title = QtWidgets.QLabel("V6 Portal Configuration")
        title.setStyleSheet("""
            color: #3498DB;
            font-weight: bold;
            font-size: 20px;
            margin-bottom: 10px;
        """)
        
        # Add glow effect to title
        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(15)
        glow.setColor(QColor(52, 152, 219, 150))
        glow.setOffset(0, 0)
        title.setGraphicsEffect(glow)
        
        # Create tabs for different settings categories
        tabs = QtWidgets.QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid rgba(52, 73, 94, 120);
                background-color: rgba(44, 62, 80, 180);
                border-radius: 4px;
            }
            
            QTabBar::tab {
                background-color: rgba(44, 62, 80, 150);
                color: #ECF0F1;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            
            QTabBar::tab:selected {
                background-color: rgba(52, 152, 219, 180);
                color: white;
                font-weight: bold;
            }
        """)
        
        # General settings tab
        general_tab = QtWidgets.QWidget()
        general_layout = QtWidgets.QFormLayout(general_tab)
        general_layout.setContentsMargins(15, 15, 15, 15)
        general_layout.setSpacing(10)
        
        # Sample settings
        theme_selector = QtWidgets.QComboBox()
        theme_selector.addItems(["Holographic Blue", "Dark Matter", "Neural Network", "Quantum Field"])
        theme_selector.setStyleSheet("""
            background-color: rgba(52, 73, 94, 180);
            color: white;
            padding: 5px;
            border: 1px solid rgba(52, 152, 219, 120);
            border-radius: 4px;
        """)
        # Connect theme change
        theme_selector.currentTextChanged.connect(self.applyTheme)
        
        animation_checkbox = QtWidgets.QCheckBox("Enable animations")
        animation_checkbox.setChecked(True)
        animation_checkbox.setStyleSheet("color: white;")
        
        debug_checkbox = QtWidgets.QCheckBox("Debug mode")
        debug_checkbox.setChecked(True)
        debug_checkbox.setStyleSheet("color: white;")
        
        # Add fields to layout
        general_layout.addRow("Interface Theme:", theme_selector)
        general_layout.addRow("", animation_checkbox)
        general_layout.addRow("", debug_checkbox)
        
        # Connection settings tab
        connection_tab = QtWidgets.QWidget()
        connection_layout = QtWidgets.QFormLayout(connection_tab)
        connection_layout.setContentsMargins(15, 15, 15, 15)
        connection_layout.setSpacing(10)
        
        # Sample connection settings
        socket_host = QtWidgets.QLineEdit("localhost")
        socket_host.setStyleSheet("""
            background-color: rgba(52, 73, 94, 180);
            color: white;
            padding: 5px;
            border: 1px solid rgba(52, 152, 219, 120);
            border-radius: 4px;
        """)
        
        socket_port = QtWidgets.QLineEdit("8765")
        socket_port.setStyleSheet("""
            background-color: rgba(52, 73, 94, 180);
            color: white;
            padding: 5px;
            border: 1px solid rgba(52, 152, 219, 120);
            border-radius: 4px;
        """)
        
        mock_mode = QtWidgets.QCheckBox("Mock mode")
        mock_mode.setChecked(True)
        mock_mode.setStyleSheet("color: white;")
        
        # Add fields to layout
        connection_layout.addRow("Socket Host:", socket_host)
        connection_layout.addRow("Socket Port:", socket_port)
        connection_layout.addRow("", mock_mode)
        
        # Add tabs
        tabs.addTab(general_tab, "General")
        tabs.addTab(connection_tab, "Connection")
        
        # Add buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.setStyleSheet("""
            QPushButton {
                background-color: rgba(41, 128, 185, 180);
                color: white;
                border-radius: 4px;
                padding: 8px 15px;
                font-size: 14px;
                border: 1px solid rgba(52, 152, 219, 120);
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: rgba(52, 152, 219, 200);
                border: 1px solid rgba(52, 152, 219, 180);
            }
        """)
        
        # Connect signals
        button_box.accepted.connect(settings_dialog.accept)
        button_box.rejected.connect(settings_dialog.reject)
        
        # Add widgets to layout
        layout.addWidget(title)
        layout.addWidget(tabs)
        layout.addWidget(button_box)
        
        # Show dialog
        settings_dialog.exec_()
    
    def applyTheme(self, theme_name):
        """Apply the selected theme to the interface"""
        logger.info(f"Applying theme: {theme_name}")
        
        if theme_name == "Holographic Blue":
            # Default holographic blue theme
            self.setStyleSheet("""
                QWidget {
                    background-color: transparent;
                }
            """)
            self.refreshAll()
        
        elif theme_name == "Dark Matter":
            # Dark theme with purple accents
            self.setStyleSheet("""
                QWidget {
                    background-color: transparent;
                }
            """)
            
            # Update colors for all panels
            for container in self.findChildren(V6PanelContainer):
                container.setStyleSheet("""
                    QTabBar::tab {
                        background-color: rgba(44, 57, 75, 150);
                    }
                    
                    QTabBar::tab:selected {
                        background-color: rgba(155, 89, 182, 180);
                    }
                """)
            
            # Refresh all panels with new color scheme
            self.refreshAll()
            
        elif theme_name == "Neural Network":
            # Green-focused neural theme
            self.setStyleSheet("""
                QWidget {
                    background-color: transparent;
                }
            """)
            
            # Update colors for all panels
            for container in self.findChildren(V6PanelContainer):
                container.setStyleSheet("""
                    QTabBar::tab {
                        background-color: rgba(44, 62, 60, 150);
                    }
                    
                    QTabBar::tab:selected {
                        background-color: rgba(39, 174, 96, 180);
                    }
                """)
            
            # Refresh all panels with new color scheme
            self.refreshAll()
            
        elif theme_name == "Quantum Field":
            # Cyan-orange quantum theme
            self.setStyleSheet("""
                QWidget {
                    background-color: transparent;
                }
            """)
            
            # Update colors for all panels
            for container in self.findChildren(V6PanelContainer):
                container.setStyleSheet("""
                    QTabBar::tab {
                        background-color: rgba(39, 60, 74, 150);
                    }
                    
                    QTabBar::tab:selected {
                        background-color: rgba(22, 160, 133, 180);
                    }
                """)
            
            # Refresh all panels with new color scheme
            self.refreshAll() 