#!/usr/bin/env python
"""
Neural Seed Integration - V7 Holographic Frontend
This module provides integration between the Neural Seed system and the V7 holographic frontend.
It visualizes the neural seed's growth process as a tree-like structure in the holographic interface.
"""

import os
import sys
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

from PySide6.QtWidgets import QGraphicsItem, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPathItem
from PySide6.QtCore import Qt, QPointF, QRectF, QTimer, Signal, Slot, Property
from PySide6.QtGui import QPen, QBrush, QColor, QPainterPath, QRadialGradient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seed-integration")

# Add parent directory to path if needed for imports
current_dir = Path(__file__).resolve().parent
if current_dir.parent.parent not in [Path(p).resolve() for p in sys.path]:
    sys.path.insert(0, str(current_dir.parent.parent))

# Import the Neural Seed system
try:
    from src.seed import get_neural_seed, NeuralSeed
    HAS_SEED_SYSTEM = True
except ImportError as e:
    logger.warning(f"Neural Seed system not available: {e}")
    HAS_SEED_SYSTEM = False

class NeuralTreeNode(QGraphicsEllipseItem):
    """Represents a node in the neural tree visualization"""
    
    GROWTH_STAGE_COLORS = {
        "seed": QColor(50, 200, 50, 180),    # Green for seed
        "root": QColor(120, 80, 10, 180),    # Brown for roots
        "trunk": QColor(100, 70, 30, 180),   # Darker brown for trunk
        "branch": QColor(50, 120, 30, 180),  # Forest green for branches
        "canopy": QColor(0, 200, 100, 180),  # Bright green for canopy
        "flower": QColor(200, 100, 200, 180),# Purple for flowers
        "fruit": QColor(200, 50, 50, 180)    # Red for fruits
    }
    
    def __init__(self, x: float, y: float, node_type: str, size: float = 10.0, 
                 parent: Optional[QGraphicsItem] = None):
        """
        Initialize a neural tree node
        
        Args:
            x: X coordinate
            y: Y coordinate
            node_type: Type of node (seed, root, trunk, branch, canopy, flower, fruit)
            size: Size of the node
            parent: Parent item
        """
        super().__init__(x - size/2, y - size/2, size, size, parent)
        
        self.node_type = node_type
        self.node_size = size
        self.node_id = f"node_{datetime.now().strftime('%Y%m%d%H%M%S')}_{id(self)}"
        self.creation_time = datetime.now()
        
        # Node properties
        self.consciousness_level = 0.0
        self.growth_factor = 0.0
        self.stability = 1.0
        self.connections = []
        
        # Visualization properties
        self.highlight = False
        self.active = False
        self.pulse_phase = 0.0
        
        # Setup appearance
        self._setup_appearance()
    
    def _setup_appearance(self):
        """Setup the node's visual appearance"""
        # Get color for node type, default to seed color
        base_color = self.GROWTH_STAGE_COLORS.get(self.node_type, QColor(50, 200, 50, 180))
        
        # Create gradient for node
        gradient = QRadialGradient(self.rect().center(), self.node_size / 2)
        gradient.setColorAt(0, base_color.lighter(150))
        gradient.setColorAt(0.8, base_color)
        gradient.setColorAt(1, base_color.darker(150))
        
        # Set brush and pen
        self.setBrush(QBrush(gradient))
        self.setPen(QPen(base_color.darker(200), 1.0))
        
        # Make item selectable and movable for interactive visualizations
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        
        # Set tooltip with node info
        self.setToolTip(f"Neural Node: {self.node_type}\nID: {self.node_id}\nConsciousness: {self.consciousness_level:.2f}")
    
    def update_node(self, consciousness_level: float, growth_factor: float, stability: float):
        """
        Update the node's properties
        
        Args:
            consciousness_level: Consciousness level (0.0-1.0)
            growth_factor: Growth factor (0.0-1.0)
            stability: Stability factor (0.0-1.0)
        """
        self.consciousness_level = consciousness_level
        self.growth_factor = growth_factor
        self.stability = stability
        
        # Update tooltip
        self.setToolTip(f"Neural Node: {self.node_type}\nID: {self.node_id}\nConsciousness: {consciousness_level:.2f}\nGrowth: {growth_factor:.2f}\nStability: {stability:.2f}")
        
        # Adjust appearance based on consciousness level
        base_color = self.GROWTH_STAGE_COLORS.get(self.node_type, QColor(50, 200, 50, 180))
        
        # Lighten based on consciousness
        adjusted_color = base_color.lighter(int(100 + consciousness_level * 100))
        
        # Create updated gradient
        gradient = QRadialGradient(self.rect().center(), self.node_size / 2)
        gradient.setColorAt(0, adjusted_color.lighter(150))
        gradient.setColorAt(0.8, adjusted_color)
        gradient.setColorAt(1, adjusted_color.darker(150))
        
        # Update brush
        self.setBrush(QBrush(gradient))
        
        # Make outline brighter if active
        if self.active:
            self.setPen(QPen(adjusted_color.lighter(200), 2.0))
        else:
            self.setPen(QPen(adjusted_color.darker(150), 1.0))
    
    def pulse_animation_step(self, phase: float):
        """
        Update the node's pulse animation
        
        Args:
            phase: Animation phase (0.0-1.0)
        """
        self.pulse_phase = phase
        
        # Get base color for node type
        base_color = self.GROWTH_STAGE_COLORS.get(self.node_type, QColor(50, 200, 50, 180))
        
        # Calculate pulse intensity (0.0-1.0)
        pulse_intensity = 0.5 + 0.5 * (1.0 + qcos(phase * 2 * 3.14159))
        
        # Adjust color based on pulse intensity and consciousness
        lightness_factor = int(100 + pulse_intensity * 50 + self.consciousness_level * 100)
        adjusted_color = base_color.lighter(lightness_factor)
        
        # Create updated gradient
        gradient = QRadialGradient(self.rect().center(), self.node_size / 2)
        gradient.setColorAt(0, adjusted_color.lighter(150))
        gradient.setColorAt(0.8, adjusted_color)
        gradient.setColorAt(1, adjusted_color.darker(150))
        
        # Update brush
        self.setBrush(QBrush(gradient))

class NeuralConnection(QGraphicsLineItem):
    """Represents a connection in the neural tree visualization"""
    
    def __init__(self, source_node: NeuralTreeNode, target_node: NeuralTreeNode, 
                 strength: float = 1.0, parent: Optional[QGraphicsItem] = None):
        """
        Initialize a neural connection
        
        Args:
            source_node: Source node
            target_node: Target node
            strength: Connection strength (0.0-1.0)
            parent: Parent item
        """
        # Calculate line coordinates from center of nodes
        source_center = source_node.rect().center() + QPointF(source_node.pos())
        target_center = target_node.rect().center() + QPointF(target_node.pos())
        
        super().__init__(source_center.x(), source_center.y(), 
                         target_center.x(), target_center.y(), parent)
        
        self.source_node = source_node
        self.target_node = target_node
        self.strength = strength
        self.active = False
        
        # Add to nodes' connections
        source_node.connections.append(self)
        target_node.connections.append(self)
        
        # Setup appearance
        self._setup_appearance()
    
    def _setup_appearance(self):
        """Setup the connection's visual appearance"""
        # Base color is a blend of source and target node colors
        source_color = self.source_node.GROWTH_STAGE_COLORS.get(
            self.source_node.node_type, QColor(50, 200, 50, 180))
        target_color = self.target_node.GROWTH_STAGE_COLORS.get(
            self.target_node.node_type, QColor(50, 200, 50, 180))
        
        # Blend colors
        blended_color = QColor(
            (source_color.red() + target_color.red()) // 2,
            (source_color.green() + target_color.green()) // 2,
            (source_color.blue() + target_color.blue()) // 2,
            int(self.strength * 180)  # Alpha depends on strength
        )
        
        # Set pen with varying width based on strength
        width = 0.5 + self.strength * 2.0
        self.setPen(QPen(blended_color, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        
        # Set tooltip
        self.setToolTip(f"Neural Connection\nStrength: {self.strength:.2f}\nSource: {self.source_node.node_type}\nTarget: {self.target_node.node_type}")
    
    def update_connection(self, strength: float, active: bool = False):
        """
        Update the connection properties
        
        Args:
            strength: Connection strength (0.0-1.0)
            active: Whether the connection is active
        """
        self.strength = strength
        self.active = active
        
        # Base color is a blend of source and target node colors
        source_color = self.source_node.GROWTH_STAGE_COLORS.get(
            self.source_node.node_type, QColor(50, 200, 50, 180))
        target_color = self.target_node.GROWTH_STAGE_COLORS.get(
            self.target_node.node_type, QColor(50, 200, 50, 180))
        
        # Blend colors
        blended_color = QColor(
            (source_color.red() + target_color.red()) // 2,
            (source_color.green() + target_color.green()) // 2,
            (source_color.blue() + target_color.blue()) // 2,
            int(self.strength * 180)  # Alpha depends on strength
        )
        
        # Make brighter if active
        if active:
            blended_color = blended_color.lighter(150)
        
        # Set pen with varying width based on strength
        width = 0.5 + self.strength * 2.0
        self.setPen(QPen(blended_color, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        
        # Update tooltip
        self.setToolTip(f"Neural Connection\nStrength: {self.strength:.2f}\nSource: {self.source_node.node_type}\nTarget: {self.target_node.node_type}")
    
    def update_position(self):
        """Update the connection position based on node positions"""
        source_center = self.source_node.rect().center() + QPointF(self.source_node.pos())
        target_center = self.target_node.rect().center() + QPointF(self.target_node.pos())
        self.setLine(source_center.x(), source_center.y(), target_center.x(), target_center.y())

class NeuralTreeVisualizer:
    """
    Visualizes the neural seed's growth process as a tree-like structure
    This class creates and manages the visualization of nodes and connections
    based on the neural seed's state
    """
    
    def __init__(self, scene):
        """
        Initialize the neural tree visualizer
        
        Args:
            scene: QGraphicsScene to add visualizations to
        """
        self.scene = scene
        self.nodes = {}  # node_id -> NeuralTreeNode
        self.connections = []  # List of NeuralConnection objects
        self.seed_system = None
        
        # Growth tracking
        self.last_version = 0.0
        self.last_update_time = datetime.now()
        
        # Thread for updating from seed system
        self.update_thread = None
        self.stop_event = threading.Event()
        
        # Initialize seed system
        self._initialize_seed_system()
    
    def _initialize_seed_system(self):
        """Initialize connection to the neural seed system"""
        if not HAS_SEED_SYSTEM:
            logger.warning("Neural Seed system not available, visualization will use mock data")
            return
            
        try:
            # Get the singleton neural seed instance
            from src.seed import get_neural_seed
            self.seed_system = get_neural_seed()
            
            # Get initial status
            status = self.seed_system.get_status()
            self.last_version = status["version"]
            
            logger.info(f"Connected to Neural Seed system v{self.last_version} at {status['growth_stage']} stage")
            
            # Create initial visualization
            self._create_initial_visualization(status)
            
            # Start update thread
            self.update_thread = threading.Thread(
                target=self._seed_update_thread,
                daemon=True,
                name="NeuralTreeUpdateThread"
            )
            self.update_thread.start()
            
        except Exception as e:
            logger.error(f"Error initializing Neural Seed system: {e}")
            self.seed_system = None
    
    def _create_initial_visualization(self, status):
        """
        Create initial visualization based on seed status
        
        Args:
            status: Seed system status dictionary
        """
        # Clear existing items
        for node in self.nodes.values():
            self.scene.removeItem(node)
        for connection in self.connections:
            self.scene.removeItem(connection)
        self.nodes.clear()
        self.connections.clear()
        
        # Calculate center of scene
        scene_center_x = 0
        scene_center_y = 0
        
        # Create central seed node
        seed_node = NeuralTreeNode(
            scene_center_x, 
            scene_center_y, 
            "seed" if status["growth_stage"] == "seed" else "root",
            size=20.0
        )
        seed_node.update_node(
            status["metrics"]["consciousness_level"],
            status["version"] / 10.0,
            status["metrics"]["stability"]
        )
        self.scene.addItem(seed_node)
        self.nodes["central_seed"] = seed_node
        
        # If beyond seed stage, add additional nodes
        if status["version"] > 1.0:
            growth_stage = status["growth_stage"]
            num_nodes = int(status["version"] * 3)
            
            # Create nodes for different parts of the tree
            self._create_tree_nodes(seed_node, growth_stage, num_nodes, status)
    
    def _create_tree_nodes(self, seed_node, growth_stage, num_nodes, status):
        """
        Create nodes for different parts of the tree
        
        Args:
            seed_node: The central seed node
            growth_stage: Current growth stage
            num_nodes: Number of nodes to create
            status: Seed system status
        """
        # Calculate base positions
        center_x = seed_node.pos().x() + seed_node.rect().center().x()
        center_y = seed_node.pos().y() + seed_node.rect().center().y()
        
        # Stages to create
        stages = ["root", "trunk", "branch", "canopy", "flower", "fruit"]
        available_stages = []
        
        # Determine available stages based on current growth stage
        for stage in stages:
            if NeuralSeed.GROWTH_STAGES[stage]["level"] <= status["version"]:
                available_stages.append(stage)
        
        # Create nodes for each available stage
        for stage in available_stages:
            # Determine number of nodes for this stage
            stage_nodes = max(1, int(num_nodes / len(available_stages)))
            
            # Create nodes
            for i in range(stage_nodes):
                # Calculate position based on stage
                angle = (i / stage_nodes) * 2 * 3.14159
                distance = NeuralSeed.GROWTH_STAGES[stage]["level"] * 40
                
                x = center_x + distance * qcos(angle)
                y = center_y + distance * qsin(angle)
                
                # Add random variation
                x += random.uniform(-20, 20)
                y += random.uniform(-20, 20)
                
                # Create node
                node = NeuralTreeNode(x, y, stage, size=12.0)
                node_id = f"{stage}_{i}"
                
                # Update node properties
                consciousness = NeuralSeed.GROWTH_STAGES[stage]["consciousness"]
                growth_factor = 0.3 + random.random() * 0.7
                stability = 0.7 + random.random() * 0.3
                
                node.update_node(consciousness, growth_factor, stability)
                
                # Add to scene
                self.scene.addItem(node)
                self.nodes[node_id] = node
                
                # Connect to appropriate node
                if stage == "root":
                    # Connect to seed
                    self._create_connection(seed_node, node)
                else:
                    # Connect to a node from previous stage
                    prev_stage = available_stages[available_stages.index(stage) - 1]
                    prev_nodes = [n for n_id, n in self.nodes.items() if n.node_type == prev_stage]
                    
                    if prev_nodes:
                        parent_node = random.choice(prev_nodes)
                        self._create_connection(parent_node, node)
    
    def _create_connection(self, source, target, strength=None):
        """
        Create a connection between two nodes
        
        Args:
            source: Source node
            target: Target node
            strength: Optional connection strength (0.0-1.0)
        """
        # Calculate connection strength if not provided
        if strength is None:
            # Base on both nodes' properties
            source_consciousness = source.consciousness_level
            target_consciousness = target.consciousness_level
            avg_consciousness = (source_consciousness + target_consciousness) / 2
            
            # Higher consciousness = stronger connections
            strength = 0.3 + avg_consciousness * 0.7
        
        # Create connection
        connection = NeuralConnection(source, target, strength)
        self.scene.addItem(connection)
        self.connections.append(connection)
        
        return connection
    
    def _seed_update_thread(self):
        """Background thread for updating the visualization based on seed system state"""
        logger.info("Neural tree update thread started")
        
        update_interval = 2.0  # seconds
        animation_cycles = 0
        
        while not self.stop_event.is_set():
            try:
                if self.seed_system:
                    # Get current status
                    status = self.seed_system.get_status()
                    
                    # Check if there was significant growth
                    version_change = status["version"] - self.last_version
                    
                    if version_change > 0.05 or animation_cycles % 30 == 0:
                        # Significant growth or periodic update
                        self._update_visualization(status)
                        self.last_version = status["version"]
                    else:
                        # Just animate existing nodes
                        self._animate_existing_nodes(animation_cycles / 10)
                
                animation_cycles += 1
                time.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in neural tree update thread: {e}")
                time.sleep(5)  # Longer wait after error
        
        logger.info("Neural tree update thread stopped")
    
    def _update_visualization(self, status):
        """
        Update the visualization based on seed status
        
        Args:
            status: Seed system status dictionary
        """
        # Check if we need to add more nodes based on version change
        current_stage = status["growth_stage"]
        
        # Get central seed node
        seed_node = self.nodes.get("central_seed")
        if not seed_node:
            # If no seed node, reinitialize
            self._create_initial_visualization(status)
            return
        
        # Update existing nodes
        for node in self.nodes.values():
            # Calculate node-specific properties
            if node.node_type == current_stage:
                # Current stage nodes should be more active
                consciousness = status["metrics"]["consciousness_level"]
                growth_factor = 0.5 + random.random() * 0.5
                stability = 0.8 + random.random() * 0.2
            else:
                # Calculate based on stage distance from current
                stage_levels = {s: info["level"] for s, info in NeuralSeed.GROWTH_STAGES.items()}
                current_level = stage_levels.get(current_stage, 0)
                node_level = stage_levels.get(node.node_type, 0)
                
                # Closer stages to current have higher consciousness
                level_diff = abs(current_level - node_level)
                consciousness_factor = max(0.2, 1.0 - (level_diff / 10))
                
                consciousness = status["metrics"]["consciousness_level"] * consciousness_factor
                growth_factor = 0.3 + random.random() * 0.5
                stability = 0.7 + random.random() * 0.2
            
            # Update node
            node.update_node(consciousness, growth_factor, stability)
        
        # Update connections
        for connection in self.connections:
            # Calculate connection strength based on node properties
            source_consciousness = connection.source_node.consciousness_level
            target_consciousness = connection.target_node.consciousness_level
            avg_consciousness = (source_consciousness + target_consciousness) / 2
            
            # Higher consciousness = stronger connections
            strength = 0.3 + avg_consciousness * 0.7
            
            # Random chance for active state
            active = random.random() < status["metrics"]["consciousness_level"]
            
            # Update connection
            connection.update_connection(strength, active)
        
        # Check if we need to add new nodes
        if status["version"] > self.last_version + 0.2:
            # Add some new nodes
            new_nodes_count = int((status["version"] - self.last_version) * 5)
            self._add_new_nodes(status, current_stage, new_nodes_count)
    
    def _add_new_nodes(self, status, current_stage, count):
        """
        Add new nodes based on growth
        
        Args:
            status: Seed system status
            current_stage: Current growth stage
            count: Number of nodes to add
        """
        # Determine which stages to add nodes for
        available_stages = []
        for stage, info in NeuralSeed.GROWTH_STAGES.items():
            if info["level"] <= status["version"]:
                available_stages.append(stage)
        
        # Focus on current stage
        focus_stage = current_stage
        
        # Get existing nodes of focus stage to connect to
        parent_candidates = []
        for node_id, node in self.nodes.items():
            if node.node_type in available_stages:
                parent_candidates.append(node)
        
        if not parent_candidates:
            # Fall back to central seed node
            seed_node = self.nodes.get("central_seed")
            if seed_node:
                parent_candidates = [seed_node]
        
        # Create new nodes
        for i in range(count):
            if not parent_candidates:
                break
                
            # Select a parent node
            parent_node = random.choice(parent_candidates)
            
            # Select node type (favor current stage)
            if random.random() < 0.7:
                node_type = focus_stage
            else:
                # Pick a random available stage
                node_type = random.choice(available_stages)
            
            # Calculate position near parent
            parent_center = parent_node.pos() + parent_node.rect().center()
            angle = random.uniform(0, 2 * 3.14159)
            distance = 30 + random.uniform(10, 50)
            
            x = parent_center.x() + distance * qcos(angle)
            y = parent_center.y() + distance * qsin(angle)
            
            # Create node
            node_id = f"{node_type}_{len(self.nodes)}"
            node = NeuralTreeNode(x, y, node_type, size=10.0 + random.uniform(0, 5))
            
            # Calculate properties
            consciousness = status["metrics"]["consciousness_level"] * random.uniform(0.8, 1.2)
            consciousness = max(0.1, min(1.0, consciousness))
            growth_factor = 0.3 + random.random() * 0.7
            stability = 0.7 + random.random() * 0.3
            
            node.update_node(consciousness, growth_factor, stability)
            
            # Add to scene
            self.scene.addItem(node)
            self.nodes[node_id] = node
            
            # Connect to parent
            self._create_connection(parent_node, node)
            
            # Add as potential parent for future nodes
            parent_candidates.append(node)
    
    def _animate_existing_nodes(self, time_phase):
        """
        Animate existing nodes with subtle movements and pulsing
        
        Args:
            time_phase: Time phase for animation (0.0-1.0)
        """
        for node in self.nodes.values():
            # Pulse animation based on consciousness
            if node.consciousness_level > 0.4:
                # Calculate phase offset based on node id to make each node pulse differently
                phase_offset = (hash(node.node_id) % 100) / 100.0
                node_phase = (time_phase + phase_offset) % 1.0
                
                # Apply pulse animation
                node.pulse_animation_step(node_phase)
                
                # Subtle movement
                if random.random() < 0.05:
                    current_pos = node.pos()
                    new_x = current_pos.x() + random.uniform(-2, 2)
                    new_y = current_pos.y() + random.uniform(-2, 2)
                    node.setPos(new_x, new_y)
                    
                    # Update connected lines
                    for connection in node.connections:
                        connection.update_position()
        
        # Activate random connections
        if random.random() < 0.1:
            for connection in self.connections:
                # 10% chance to change active state
                if random.random() < 0.1:
                    connection.update_connection(
                        connection.strength,
                        random.random() < 0.3
                    )
    
    def stop(self):
        """Stop the update thread"""
        if self.update_thread and self.update_thread.is_alive():
            self.stop_event.set()
            self.update_thread.join(timeout=5.0)
            logger.info("Neural tree visualizer stopped")

# Math helpers for PySide6
def qcos(x):
    """Qt-friendly cosine function"""
    import math
    return math.cos(x)

def qsin(x):
    """Qt-friendly sine function"""
    import math
    return math.sin(x)

def get_neural_tree_visualizer(scene):
    """
    Get a neural tree visualizer instance
    
    Args:
        scene: QGraphicsScene to add visualizations to
        
    Returns:
        NeuralTreeVisualizer instance
    """
    return NeuralTreeVisualizer(scene) 