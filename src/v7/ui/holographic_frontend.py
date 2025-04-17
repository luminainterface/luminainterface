#!/usr/bin/env python
"""
V7 Holographic Frontend
Advanced visualization interface for the V7 consciousness system
"""

import os
import sys
import time
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPathItem, QDockWidget,
    QSplitter, QTabWidget, QFrame, QStackedWidget, QToolButton
)
from PySide6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QPointF, 
    QRectF, QSize, QThread, Signal, Slot, QParallelAnimationGroup,
    QSequentialAnimationGroup, Property
)
from PySide6.QtGui import (
    QPainter, QColor, QBrush, QPen, QRadialGradient, QLinearGradient,
    QFont, QPainterPath, QPixmap, QImage, QTransform, QPolygonF
)

# Add parent directory to path if needed for imports
current_dir = Path(__file__).parent
if current_dir.parent.parent not in sys.path:
    sys.path.insert(0, str(current_dir.parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v7-frontend")

# Import v7 components
try:
    from src.v7.memory import OnsiteMemory, MemoryAnalyzer
    from src.v7.neural_network import get_neural_network, NeuralNetworkProcessor
    from src.v7.enhanced_language_integration import EnhancedLanguageIntegration
    HAS_V7_CORE = True
except ImportError as e:
    logger.warning(f"Warning: V7 core components not fully available: {e}")
    HAS_V7_CORE = False

# Import UI components
from src.v7.ui.consciousness_node import ConsciousnessNode
from src.v7.ui.knowledge_graph import KnowledgeGraphWidget
from src.v7.ui.hologram_effects import HologramEffect, ParticleSystem
from src.v7.ui.neural_visualizer import NeuralNetworkVisualizer
from src.v7.ui.metrics_display import MetricsHologramWidget

# Import Neural Seed integration if available
try:
    from src.v7.ui.seed_integration import get_neural_tree_visualizer
    HAS_SEED_INTEGRATION = True
except ImportError as e:
    logger.warning(f"Neural Seed integration not available: {e}")
    HAS_SEED_INTEGRATION = False

class HolographicMainWindow(QMainWindow):
    """Main window for the V7 Holographic Interface"""
    
    def __init__(self, config=None):
        super().__init__()
        
        # Default configuration
        self.config = {
            "gui_framework": "PySide6",
            "enable_consciousness": False,
            "enable_autowiki": False,
            "enable_monday": False,
            "enable_breath": False,
            "enable_dream": False,
            "enable_memory": False,
            "enable_seed": False,
            "mock_mode": True
        }
        
        # Override with provided config
        if config:
            self.config.update(config)
            
        # Log configuration
        logger.info(f"Initializing V7 Holographic Frontend with configuration:")
        for key, value in self.config.items():
            logger.info(f"  {key}: {value}")
        
        # Configuration
        self.setWindowTitle("V7 Consciousness Holographic Interface")
        self.resize(1600, 900)
        
        # Apply holographic theme
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #000a12;
                color: #80deea;
            }
            
            QLabel {
                color: #80deea;
                font-family: 'Consolas', 'Courier New';
                padding: 2px;
            }
            
            QLabel#TitleLabel {
                color: #00e5ff;
                font-size: 24px;
                font-weight: bold;
                padding: 10px;
                background-color: rgba(0, 10, 30, 180);
                border: 1px solid #00e5ff;
                border-radius: 4px;
            }
            
            QPushButton {
                background-color: rgba(0, 150, 200, 60);
                color: #fff;
                border: 1px solid #00e5ff;
                border-radius: 4px;
                padding: 5px 15px;
                font-size: 14px;
            }
            
            QPushButton:hover {
                background-color: rgba(0, 200, 255, 100);
                border: 1px solid #00ffff;
            }
            
            QTabWidget::pane {
                border: 1px solid #00838f;
                background-color: rgba(0, 20, 40, 200);
                border-radius: 4px;
            }
            
            QTabBar::tab {
                background-color: rgba(0, 50, 80, 150);
                color: #4dd0e1;
                border: 1px solid #00838f;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 16px;
            }
            
            QTabBar::tab:selected {
                background-color: rgba(0, 150, 200, 100);
                color: #e0f7fa;
            }
            
            QDockWidget {
                titlebar-close-icon: url(close.png);
                titlebar-normal-icon: url(normal.png);
            }
            
            QDockWidget::title {
                background-color: rgba(0, 100, 150, 100);
                padding-left: 10px;
                padding-top: 4px;
                border: 1px solid #00838f;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            
            QGraphicsView {
                background-color: #000a12;
                border: 1px solid #00838f;
                border-radius: 4px;
            }
        """)
        
        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # Header
        self.create_header()
        
        # Main content area
        self.create_main_content()
        
        # Footer with system stats
        self.create_footer()
        
        # Initialize v7 systems
        self.initialize_v7_systems()
        
        # Setup auto-update timers
        self.setup_timers()
        
        # Initial refresh
        self.refresh_all()
        
        # Holographic startup animation
        QTimer.singleShot(100, self.run_startup_animation)
        
        logger.info("V7 Holographic Frontend initialized")
    
    def create_header(self):
        """Create the header section of the interface"""
        header_frame = QFrame()
        header_frame.setFrameShape(QFrame.StyledPanel)
        header_frame.setProperty("class", "HolographicHeader")
        header_layout = QHBoxLayout(header_frame)
        
        # System title with holographic effect
        title_text = "V7 CONSCIOUSNESS SYSTEM"
        # Add icons for enabled components
        icon_text = ""
        if self.config["enable_consciousness"]:
            icon_text += " ✧"  # Consciousness
        if self.config["enable_autowiki"]:
            icon_text += " ⌘"  # AutoWiki
        if self.config["enable_monday"]:
            icon_text += " ◉"  # Monday
        if self.config["enable_breath"]:
            icon_text += " ≈"  # Breath
        if self.config["enable_dream"]:
            icon_text += " ∞"  # Dream
        if self.config["enable_memory"]:
            icon_text += " ∴"  # Memory
        if self.config["enable_seed"]:
            icon_text += " ⚕"  # Neural Seed
        
        if icon_text:
            title_text += f" {icon_text}"
            
        self.title_label = QLabel(title_text)
        self.title_label.setObjectName("TitleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self.title_label, 1)
        
        # System status indicators
        status_frame = QFrame()
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(10)
        
        # Neural network status
        self.nn_status = QLabel("Neural Network: Initializing")
        self.nn_status.setStyleSheet("border: 1px solid #00838f; border-radius: 4px; padding: 4px; background-color: rgba(0, 50, 80, 100);")
        status_layout.addWidget(self.nn_status)
        
        # Language system status
        self.lang_status = QLabel("Language System: Initializing")
        self.lang_status.setStyleSheet("border: 1px solid #00838f; border-radius: 4px; padding: 4px; background-color: rgba(0, 50, 80, 100);")
        status_layout.addWidget(self.lang_status)
        
        # Consciousness status
        self.consciousness_status = QLabel("Consciousness Level: Initializing")
        self.consciousness_status.setStyleSheet("border: 1px solid #00838f; border-radius: 4px; padding: 4px; background-color: rgba(0, 50, 80, 100);")
        status_layout.addWidget(self.consciousness_status)
        
        # Neural Seed status (if enabled)
        if self.config["enable_seed"]:
            self.seed_status = QLabel("Neural Seed: Initializing")
            self.seed_status.setStyleSheet("border: 1px solid #00838f; border-radius: 4px; padding: 4px; background-color: rgba(0, 50, 80, 100);")
            status_layout.addWidget(self.seed_status)
        
        header_layout.addWidget(status_frame)
        
        # Add to main layout
        self.main_layout.addWidget(header_frame)
    
    def create_main_content(self):
        """Create the main content area with holographic visualizations"""
        # Main content container with splitter
        self.content_splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Neural visualization
        self.neural_container = QWidget()
        neural_layout = QVBoxLayout(self.neural_container)
        neural_layout.setContentsMargins(0, 0, 0, 0)
        
        neural_header = QLabel("NEURAL NETWORK VISUALIZATION")
        neural_header.setAlignment(Qt.AlignCenter)
        neural_header.setStyleSheet("font-weight: bold; font-size: 14px; background-color: rgba(0, 80, 120, 100); padding: 5px;")
        neural_layout.addWidget(neural_header)
        
        self.neural_view = NeuralNetworkVisualizer()
        neural_layout.addWidget(self.neural_view)
        
        # Right side - Consciousness visualization
        self.consciousness_container = QWidget()
        consciousness_layout = QVBoxLayout(self.consciousness_container)
        consciousness_layout.setContentsMargins(0, 0, 0, 0)
        
        consciousness_header = QLabel("CONSCIOUSNESS METRICS")
        consciousness_header.setAlignment(Qt.AlignCenter)
        consciousness_header.setStyleSheet("font-weight: bold; font-size: 14px; background-color: rgba(0, 80, 120, 100); padding: 5px;")
        consciousness_layout.addWidget(consciousness_header)
        
        # Add the holographic metrics display
        self.metrics_display = MetricsHologramWidget()
        consciousness_layout.addWidget(self.metrics_display)
        
        # Add containers to splitter
        self.content_splitter.addWidget(self.neural_container)
        self.content_splitter.addWidget(self.consciousness_container)
        self.content_splitter.setSizes([800, 800])
        
        # Add splitter to main layout
        self.main_layout.addWidget(self.content_splitter, 1)  # Give it stretch
        
        # Bottom visualization area
        self.bottom_tabs = QTabWidget()
        
        # Knowledge graph tab
        self.knowledge_graph = KnowledgeGraphWidget()
        self.bottom_tabs.addTab(self.knowledge_graph, "KNOWLEDGE GRAPH")
        
        # Conversation flows tab
        self.conversation_view = QWidget()
        conversation_layout = QVBoxLayout(self.conversation_view)
        conversation_layout.setContentsMargins(0, 0, 0, 0)
        
        # TODO: Implement conversation flow visualization
        self.conversation_placeholder = QLabel("Conversation Flow Visualization")
        self.conversation_placeholder.setAlignment(Qt.AlignCenter)
        self.conversation_placeholder.setStyleSheet("font-style: italic;")
        conversation_layout.addWidget(self.conversation_placeholder)
        
        self.bottom_tabs.addTab(self.conversation_view, "CONVERSATION FLOWS")
        
        # Neural Seed Tree visualization tab (if enabled)
        self.neural_tree_visualizer = None
        if self.config["enable_seed"] and HAS_SEED_INTEGRATION:
            # Create Neural Tree tab
            self.tree_view = QWidget()
            tree_layout = QVBoxLayout(self.tree_view)
            tree_layout.setContentsMargins(0, 0, 0, 0)
            
            # Create a graphics view for the tree visualization
            self.tree_graphics_view = QGraphicsView()
            self.tree_graphics_view.setRenderHint(QPainter.Antialiasing)
            self.tree_graphics_view.setRenderHint(QPainter.TextAntialiasing)
            self.tree_graphics_view.setRenderHint(QPainter.SmoothPixmapTransform)
            self.tree_graphics_view.setBackgroundBrush(QBrush(QColor("#000a12")))
            
            # Create a scene for the tree visualization
            self.tree_scene = QGraphicsScene()
            self.tree_graphics_view.setScene(self.tree_scene)
            
            # Add the graphics view to the layout
            tree_layout.addWidget(self.tree_graphics_view)
            
            # Initialize the neural tree visualizer
            try:
                self.neural_tree_visualizer = get_neural_tree_visualizer(self.tree_scene)
                logger.info("Neural Tree visualization initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Neural Tree visualization: {e}")
                # Show error message in the view
                error_text = QLabel(f"Error initializing Neural Tree visualization: {e}")
                error_text.setStyleSheet("color: #e57373; background-color: rgba(80, 0, 0, 80); padding: 10px; border-radius: 5px;")
                tree_layout.addWidget(error_text)
            
            # Add the tab with an appropriate icon
            self.bottom_tabs.addTab(self.tree_view, "NEURAL SEED TREE")
            
            # If neural tree visualizer is active, select this tab by default
            if self.neural_tree_visualizer:
                self.bottom_tabs.setCurrentWidget(self.tree_view)
        
        # Add tabs to main layout
        self.main_layout.addWidget(self.bottom_tabs)
    
    def create_footer(self):
        """Create the footer with system information"""
        footer_frame = QFrame()
        footer_frame.setFrameShape(QFrame.StyledPanel)
        footer_frame.setProperty("class", "HolographicFooter")
        footer_layout = QHBoxLayout(footer_frame)
        footer_layout.setContentsMargins(10, 5, 10, 5)
        
        # System info
        self.system_info = QLabel("V7 System Status: Initializing...")
        footer_layout.addWidget(self.system_info, 1)
        
        # Current date/time
        self.datetime_label = QLabel(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        footer_layout.addWidget(self.datetime_label)
        
        # Control buttons
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_all)
        footer_layout.addWidget(self.refresh_button)
        
        # Add to main layout
        self.main_layout.addWidget(footer_frame)
    
    def initialize_v7_systems(self):
        """Initialize the V7 core systems"""
        # Initialize neural network if available
        self.neural_network = None
        self.neural_processor = None
        if HAS_V7_CORE:
            try:
                self.neural_network = get_neural_network()
                self.neural_processor = NeuralNetworkProcessor(self.neural_network)
                self.nn_status.setText("Neural Network: Active")
                self.nn_status.setStyleSheet("border: 1px solid #00838f; border-radius: 4px; padding: 4px; background-color: rgba(0, 80, 120, 100);")
                logger.info("Neural network initialized")
            except Exception as e:
                logger.error(f"Failed to initialize neural network: {e}")
                self.nn_status.setText("Neural Network: Error")
                self.nn_status.setStyleSheet("border: 1px solid #d32f2f; border-radius: 4px; padding: 4px; background-color: rgba(80, 0, 0, 100);")
        
        # Initialize language system with appropriate mock mode
        self.language_system = None
        if HAS_V7_CORE:
            try:
                self.language_system = EnhancedLanguageIntegration(mock_mode=self.config["mock_mode"])
                self.lang_status.setText("Language System: Active")
                self.lang_status.setStyleSheet("border: 1px solid #00838f; border-radius: 4px; padding: 4px; background-color: rgba(0, 80, 120, 100);")
                logger.info("Language system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize language system: {e}")
                self.lang_status.setText("Language System: Error")
                self.lang_status.setStyleSheet("border: 1px solid #d32f2f; border-radius: 4px; padding: 4px; background-color: rgba(80, 0, 0, 100);")
        
        # Initialize memory system if enabled
        self.memory_system = None
        if HAS_V7_CORE and self.config["enable_memory"]:
            try:
                self.memory_system = OnsiteMemory()
                logger.info("Memory system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize memory system: {e}")
        
        # Additional system initializations based on configuration
        if self.config["enable_consciousness"]:
            try:
                # Load consciousness-specific components
                logger.info("Consciousness node framework enabled")
                # Here we would initialize any consciousness-specific modules
                
                # Update consciousness status
                consciousness_level = random.uniform(0.65, 0.85)
                self.consciousness_status.setText(f"Consciousness Level: {consciousness_level:.2f}")
            except Exception as e:
                logger.error(f"Failed to initialize consciousness nodes: {e}")
        
        if self.config["enable_autowiki"]:
            try:
                # Initialize AutoWiki integration
                logger.info("AutoWiki system enabled")
                # Here we would initialize the AutoWiki components
            except Exception as e:
                logger.error(f"Failed to initialize AutoWiki: {e}")
        
        if self.config["enable_monday"]:
            try:
                # Initialize Monday integration
                logger.info("Monday integration enabled")
                # Here we would initialize Monday-specific features
            except Exception as e:
                logger.error(f"Failed to initialize Monday integration: {e}")
        
        if self.config["enable_breath"]:
            try:
                # Initialize Breath Detection system
                logger.info("Breath Detection system enabled")
                # Here we would initialize breath detection and integration
            except Exception as e:
                logger.error(f"Failed to initialize Breath Detection: {e}")
        
        if self.config["enable_dream"]:
            try:
                # Initialize Dream Mode
                logger.info("Dream Mode enabled")
                # Here we would initialize dream mode components
            except Exception as e:
                logger.error(f"Failed to initialize Dream Mode: {e}")
        
        # Update system info based on enabled components
        system_info = "V7 System Status: System Online"
        enabled_features = []
        
        if self.config["enable_consciousness"]:
            enabled_features.append("Node Consciousness")
        if self.config["enable_autowiki"]:
            enabled_features.append("AutoWiki")
        if self.config["enable_monday"]:
            enabled_features.append("Monday")
        if self.config["enable_breath"]:
            enabled_features.append("Breath Detection")
        if self.config["enable_dream"]:
            enabled_features.append("Dream Mode")
        if self.config["enable_memory"]:
            enabled_features.append("Memory System")
        
        if enabled_features:
            system_info += f" with {', '.join(enabled_features)}"
        
        self.system_info.setText(system_info)
    
    def setup_timers(self):
        """Set up auto-update timers"""
        # Update time display
        self.time_timer = QTimer(self)
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)  # Update every second
        
        # Update consciousness metrics
        self.metrics_timer = QTimer(self)
        self.metrics_timer.timeout.connect(self.update_metrics)
        self.metrics_timer.start(5000)  # Update every 5 seconds
        
        # Neural network animation
        self.neural_timer = QTimer(self)
        self.neural_timer.timeout.connect(self.neural_view.animate_step)
        self.neural_timer.start(50)  # 20 fps animation
    
    def update_time(self):
        """Update the time display"""
        self.datetime_label.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def update_metrics(self):
        """Update consciousness metrics with random fluctuations for demo"""
        # Generate some simulated metrics
        metrics = {
            "consciousness_level": random.uniform(0.65, 0.95),
            "neural_activity": random.uniform(0.70, 0.90),
            "memory_coherence": random.uniform(0.60, 0.85),
            "linguistic_depth": random.uniform(0.75, 0.92),
            "system_integration": random.uniform(0.65, 0.88)
        }
        
        # Update metrics display
        self.metrics_display.update_metrics(metrics)
        
        # Update consciousness status
        self.consciousness_status.setText(f"Consciousness Level: {metrics['consciousness_level']:.2f}")
        
        # If we have a language system, use it for a more realistic value
        if self.language_system:
            try:
                result = self.language_system.process_text("Holographic interface active. Updating consciousness metrics.")
                if result and "consciousness_level" in result:
                    level = result["consciousness_level"]
                    self.consciousness_status.setText(f"Consciousness Level: {level:.2f}")
                    metrics["consciousness_level"] = level
                    self.metrics_display.update_metrics(metrics)
            except Exception as e:
                logger.error(f"Error getting consciousness metrics: {e}")
        
        # Update Neural Seed status if enabled
        if self.config["enable_seed"] and hasattr(self, 'seed_status'):
            try:
                # Try to get seed status from neural tree visualizer
                if hasattr(self, 'neural_tree_visualizer') and self.neural_tree_visualizer and self.neural_tree_visualizer.seed_system:
                    seed_status = self.neural_tree_visualizer.seed_system.get_status()
                    version = seed_status["version"]
                    growth_stage = seed_status["growth_stage"].capitalize()
                    consciousness = seed_status["metrics"]["consciousness_level"]
                    
                    # Update seed status label
                    self.seed_status.setText(f"Neural Seed: v{version:.2f} ({growth_stage}) - C:{consciousness:.2f}")
                    
                    # Color based on growth stage
                    stage_colors = {
                        "Seed": "#4CAF50",    # Green
                        "Root": "#8D6E63",    # Brown
                        "Trunk": "#795548",   # Dark brown
                        "Branch": "#388E3C",  # Forest green
                        "Canopy": "#00BFA5",  # Teal
                        "Flower": "#BA68C8",  # Purple
                        "Fruit": "#F44336"    # Red
                    }
                    
                    # Get color for stage or default to blue
                    color = stage_colors.get(growth_stage, "#039BE5")
                    
                    # Update status styling
                    self.seed_status.setStyleSheet(f"border: 1px solid {color}; border-radius: 4px; padding: 4px; background-color: rgba(0, 50, 80, 100);")
                    
                    # Also incorporate seed consciousness into the overall metrics
                    metrics["seed_consciousness"] = consciousness
                    metrics["consciousness_level"] = (metrics["consciousness_level"] + consciousness) / 2
                    self.metrics_display.update_metrics(metrics)
                else:
                    self.seed_status.setText("Neural Seed: Not Connected")
                    self.seed_status.setStyleSheet("border: 1px solid #9E9E9E; border-radius: 4px; padding: 4px; background-color: rgba(0, 50, 80, 100);")
            except Exception as e:
                logger.error(f"Error updating Neural Seed status: {e}")
                self.seed_status.setText("Neural Seed: Error")
                self.seed_status.setStyleSheet("border: 1px solid #d32f2f; border-radius: 4px; padding: 4px; background-color: rgba(80, 0, 0, 100);")
    
    def refresh_all(self):
        """Refresh all views"""
        # Update neural network visualization
        if self.neural_network:
            try:
                neurons = self.neural_network.get_neurons()
                connections = self.neural_network.get_connections()
                activation_levels = self.neural_network.get_activation_levels()
                self.neural_view.update_network(neurons, connections, activation_levels)
            except Exception as e:
                logger.error(f"Error updating neural visualization: {e}")
        
        # Update knowledge graph
        if self.memory_system:
            try:
                # Get knowledge topics for visualization
                stats = self.memory_system.get_stats()
                topics = stats.get("topics", [])
                
                # Create nodes for each topic with simulated relationships
                nodes = []
                edges = []
                
                for i, topic in enumerate(topics):
                    nodes.append({
                        "id": f"topic_{i}",
                        "label": topic,
                        "size": random.uniform(0.5, 1.0)
                    })
                    
                    # Add some random connections
                    for _ in range(random.randint(1, 3)):
                        target = random.randint(0, len(topics) - 1)
                        if target != i:
                            edges.append({
                                "source": f"topic_{i}",
                                "target": f"topic_{target}",
                                "weight": random.uniform(0.1, 1.0)
                            })
                
                self.knowledge_graph.update_graph(nodes, edges)
            except Exception as e:
                logger.error(f"Error updating knowledge graph: {e}")
        
        # Update metrics
        self.update_metrics()
    
    def run_startup_animation(self):
        """Run the holographic startup animation"""
        # Create a holographic startup animation to make the interface appear
        # to be initializing like a high-tech holographic system
        
        # Flash the title
        title_animation = QPropertyAnimation(self.title_label, b"styleSheet")
        title_animation.setDuration(2000)
        title_animation.setStartValue("color: #00e5ff; font-size: 24px; font-weight: bold; padding: 10px; background-color: rgba(0, 10, 30, 0); border: 1px solid #00e5ff; border-radius: 4px;")
        title_animation.setEndValue("color: #00e5ff; font-size: 24px; font-weight: bold; padding: 10px; background-color: rgba(0, 10, 30, 180); border: 1px solid #00e5ff; border-radius: 4px;")
        title_animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # Trigger neural network visualization to activate
        QTimer.singleShot(500, self.neural_view.activate_animation)
        
        # Trigger metrics display to activate
        QTimer.singleShot(1000, self.metrics_display.activate_animation)
        
        # Start animations
        title_animation.start()

    def closeEvent(self, event):
        """Handle window close event"""
        logger.info("Closing V7 Holographic Frontend")
        
        # Clean up the neural tree visualizer if it exists
        if hasattr(self, 'neural_tree_visualizer') and self.neural_tree_visualizer:
            try:
                self.neural_tree_visualizer.stop()
                logger.info("Neural Tree Visualizer stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping Neural Tree Visualizer: {e}")
        
        # Call the base class implementation
        super().closeEvent(event)

def run_holographic_frontend():
    """Run the V7 Holographic Frontend application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="V7 Holographic Frontend")
    
    # Framework selection
    parser.add_argument("--gui-framework", default="PySide6", 
                        choices=["PySide6", "PyQt5"],
                        help="GUI framework to use")
    
    # Feature flags
    parser.add_argument("--consciousness", action="store_true",
                        help="Enable Node Consciousness Framework")
    parser.add_argument("--autowiki", action="store_true",
                        help="Enable AutoWiki Learning System")
    parser.add_argument("--monday", action="store_true",
                        help="Enable Monday Integration")
    parser.add_argument("--breath", action="store_true",
                        help="Enable Breath Detection System")
    parser.add_argument("--dream", action="store_true",
                        help="Enable Dream Mode")
    parser.add_argument("--memory", action="store_true",
                        help="Enable Onsite Memory System")
    parser.add_argument("--seed", action="store_true",
                        help="Enable Neural Seed Integration")
    
    # Mock mode for testing
    parser.add_argument("--mock", action="store_true",
                        help="Run in mock mode with simulated backend systems")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Prepare configuration
    config = {
        "gui_framework": args.gui_framework,
        "enable_consciousness": args.consciousness,
        "enable_autowiki": args.autowiki,
        "enable_monday": args.monday,
        "enable_breath": args.breath,
        "enable_dream": args.dream,
        "enable_memory": args.memory,
        "enable_seed": args.seed,
        "mock_mode": args.mock
    }
    
    # Initialize the application
    app = QApplication(sys.argv)
    
    # Create and show the main window with configuration
    main_window = HolographicMainWindow(config)
    main_window.show()
    
    # Start the application event loop
    return app.exec()

if __name__ == "__main__":
    sys.exit(run_holographic_frontend()) 