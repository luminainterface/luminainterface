#!/usr/bin/env python3
"""
Temple to Seed Bridge (v8)

This module implements the bridging mechanism between the Spatial Temple visualization
and the Seed Dispersal system. It allows concepts to flow down through the system
like a tree spreading its fruits. Concepts from the spatial temple's 3D visualization
become seeds that can be dispersed throughout the neural network system.
"""

import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
import uuid
import random

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, 
    QPushButton, QLabel, QFrame, QApplication, QTextEdit,
    QComboBox, QListWidget, QListWidgetItem, QCheckBox
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QSize
from PySide6.QtGui import QFont, QColor, QPalette, QIcon

# Add parent directory to path for imports
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v8.temple_bridge")

# Import modules
from src.v8.spatial_temple_mapper import SpatialTempleMapper, SpatialNode
from src.v8.demo_data_generator import generate_demo_nodes, generate_themed_demo_nodes
from src.v8.seed_dispersal_system import KnowledgeFruit, SeedDispersalWindow

# Try to import auto growth system
try:
    from src.v8.auto_seed_growth import AutoSeedGrowthSystem
    AUTO_GROWTH_AVAILABLE = True
except ImportError:
    AUTO_GROWTH_AVAILABLE = False
    logger.warning("Auto Seed Growth system not available")

class ConceptSeed:
    """
    Represents a concept seed that can be dispersed from the Spatial Temple
    to other parts of the system, like a seed from a fruit.
    """
    def __init__(self, 
                 concept: str,
                 weight: float,
                 node_type: str,
                 connections: Set[str],
                 attributes: Dict[str, Any] = None,
                 source_node_id: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.concept = concept
        self.weight = weight
        self.node_type = node_type
        self.connections = connections.copy() if connections else set()
        self.attributes = attributes.copy() if attributes else {}
        self.source_node_id = source_node_id
        self.created_at = datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission"""
        return {
            "id": self.id,
            "concept": self.concept,
            "weight": self.weight,
            "node_type": self.node_type,
            "connections": list(self.connections),
            "attributes": self.attributes,
            "source_node_id": self.source_node_id,
            "created_at": self.created_at
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConceptSeed':
        """Create from dictionary"""
        seed = cls(
            concept=data["concept"],
            weight=data["weight"],
            node_type=data["node_type"],
            connections=set(data["connections"]),
            attributes=data["attributes"],
            source_node_id=data["source_node_id"]
        )
        seed.id = data["id"]
        seed.created_at = data["created_at"]
        return seed
    
    @classmethod
    def from_spatial_node(cls, node: SpatialNode) -> 'ConceptSeed':
        """Convert a Spatial Temple node to a concept seed"""
        return cls(
            concept=node.concept,
            weight=node.weight,
            node_type=node.node_type,
            connections=node.connections,
            attributes=node.attributes,
            source_node_id=node.id
        )

class TreeFlowPanel(QWidget):
    """
    Panel for visualizing the flow of concepts from the Spatial Temple
    to the Seed system, like a tree with fruits spreading seeds.
    """
    seed_selected = Signal(ConceptSeed)  # Signal when a seed is selected
    seed_dispersed = Signal(ConceptSeed, str)  # Signal when a seed is dispersed
    
    def __init__(self, temple_mapper: Optional[SpatialTempleMapper] = None, parent=None):
        """Initialize the flow panel with an optional temple mapper"""
        super().__init__(parent)
        self.temple_mapper = temple_mapper
        self.concept_seeds = []  # List of available seeds
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("Concept Tree Flow System")
        header_label.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(header_label)
        
        # Temple Visualization section
        temple_frame = QFrame()
        temple_frame.setFrameStyle(QFrame.StyledPanel)
        temple_layout = QVBoxLayout(temple_frame)
        
        temple_layout.addWidget(QLabel("Spatial Temple (Concept Source)"))
        
        # Temple status info
        self.temple_status = QLabel("No temple connection")
        temple_layout.addWidget(self.temple_status)
        
        # Node count and regenerate button
        node_layout = QHBoxLayout()
        node_layout.addWidget(QLabel("Theme:"))
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["ai", "temple", "network", "brain"])
        node_layout.addWidget(self.theme_combo)
        
        generate_btn = QPushButton("Generate Concepts")
        generate_btn.clicked.connect(self.regenerate_temple_nodes)
        node_layout.addWidget(generate_btn)
        
        temple_layout.addLayout(node_layout)
        
        layout.addWidget(temple_frame)
        
        # Concept Seeds section
        seeds_frame = QFrame()
        seeds_frame.setFrameStyle(QFrame.StyledPanel)
        seeds_layout = QVBoxLayout(seeds_frame)
        
        seeds_layout.addWidget(QLabel("Available Concept Seeds"))
        
        # Seeds list
        self.seeds_list = QListWidget()
        self.seeds_list.itemClicked.connect(self.on_seed_selected)
        seeds_layout.addWidget(self.seeds_list)
        
        # Harvest seeds button
        harvest_btn = QPushButton("Harvest Seeds from Temple")
        harvest_btn.clicked.connect(self.harvest_seeds)
        seeds_layout.addWidget(harvest_btn)
        
        layout.addWidget(seeds_frame)
        
        # Seed Dispersal section
        dispersal_frame = QFrame()
        dispersal_frame.setFrameStyle(QFrame.StyledPanel)
        dispersal_layout = QVBoxLayout(dispersal_frame)
        
        dispersal_layout.addWidget(QLabel("Seed Dispersal"))
        
        # Selected seed info
        self.selected_seed_info = QTextEdit()
        self.selected_seed_info.setReadOnly(True)
        self.selected_seed_info.setMaximumHeight(100)
        dispersal_layout.addWidget(self.selected_seed_info)
        
        # Dispersal target
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Target System:"))
        
        self.target_combo = QComboBox()
        self.target_combo.addItems(["Seed System", "Neural Network", "Memory System", "Consciousness Core"])
        target_layout.addWidget(self.target_combo)
        
        dispersal_layout.addLayout(target_layout)
        
        # Dispersal options
        options_layout = QHBoxLayout()
        
        self.include_connections = QCheckBox("Include Connections")
        self.include_connections.setChecked(True)
        options_layout.addWidget(self.include_connections)
        
        self.expand_attributes = QCheckBox("Expand Attributes")
        options_layout.addWidget(self.expand_attributes)
        
        dispersal_layout.addLayout(options_layout)
        
        # Disperse button
        disperse_btn = QPushButton("Disperse Selected Seed")
        disperse_btn.clicked.connect(self.disperse_seed)
        dispersal_layout.addWidget(disperse_btn)
        
        layout.addWidget(dispersal_frame)
        
        # Activity Log
        log_frame = QFrame()
        log_frame.setFrameStyle(QFrame.StyledPanel)
        log_layout = QVBoxLayout(log_frame)
        
        log_layout.addWidget(QLabel("Activity Log"))
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_frame)
    
    def set_temple_mapper(self, temple_mapper: SpatialTempleMapper):
        """Set the temple mapper and update the UI"""
        self.temple_mapper = temple_mapper
        self.update_temple_status()
    
    def update_temple_status(self):
        """Update the temple status display"""
        if not self.temple_mapper:
            self.temple_status.setText("No temple connection")
            return
            
        node_count = len(self.temple_mapper.nodes)
        zone_count = len(self.temple_mapper.zones)
        
        self.temple_status.setText(
            f"Connected: {node_count} nodes in {zone_count} zones"
        )
    
    def regenerate_temple_nodes(self):
        """Regenerate the temple nodes with the selected theme"""
        if not self.temple_mapper:
            self.log_message("No temple mapper available")
            return
            
        theme = self.theme_combo.currentText()
        try:
            # Clear existing nodes
            self.temple_mapper.nodes.clear()
            
            # Generate new nodes
            count = 50
            self.log_message(f"Generating {count} nodes with {theme} theme...")
            
            if theme != 'ai':
                demo_nodes = generate_themed_demo_nodes(count, theme)
            else:
                demo_nodes = generate_demo_nodes(count)
                
            # Add to mapper
            for node in demo_nodes:
                self.temple_mapper.nodes[node.id] = node
                
            # Update status
            self.update_temple_status()
            
            # Clear seeds list since nodes have changed
            self.concept_seeds.clear()
            self.seeds_list.clear()
            
            self.log_message(f"Generated {len(demo_nodes)} concepts in the temple")
            
        except Exception as e:
            self.log_message(f"Error generating nodes: {str(e)}")
    
    def harvest_seeds(self):
        """Harvest concept seeds from the temple nodes"""
        if not self.temple_mapper or not self.temple_mapper.nodes:
            self.log_message("No temple nodes available to harvest")
            return
            
        # Clear existing seeds
        self.concept_seeds.clear()
        self.seeds_list.clear()
        
        # Create seeds from nodes
        node_list = list(self.temple_mapper.nodes.values())
        
        # Sort by weight to prioritize important concepts
        node_list.sort(key=lambda n: n.weight, reverse=True)
        
        # Take top concepts with some randomness
        harvest_count = min(15, len(node_list))
        seed_nodes = node_list[:harvest_count]
        
        # Add some random nodes for diversity
        if len(node_list) > harvest_count:
            seed_nodes.extend(random.sample(node_list[harvest_count:], min(5, len(node_list) - harvest_count)))
            
        # Create seeds
        for node in seed_nodes:
            seed = ConceptSeed.from_spatial_node(node)
            self.concept_seeds.append(seed)
            
            # Add to list widget
            item = QListWidgetItem(f"{seed.concept} ({seed.weight:.2f})")
            item.setData(Qt.UserRole, seed.id)
            self.seeds_list.addItem(item)
            
        self.log_message(f"Harvested {len(self.concept_seeds)} concept seeds from the temple")
    
    def on_seed_selected(self, item: QListWidgetItem):
        """Handle selection of a seed from the list"""
        seed_id = item.data(Qt.UserRole)
        
        # Find the corresponding seed
        seed = next((s for s in self.concept_seeds if s.id == seed_id), None)
        if not seed:
            return
            
        # Update seed info display
        info_text = f"<b>{seed.concept}</b> ({seed.node_type})<br>"
        info_text += f"Weight: {seed.weight:.2f}<br>"
        info_text += f"Connections: {len(seed.connections)}<br>"
        
        if seed.attributes:
            info_text += f"Attributes: {len(seed.attributes)}"
            
        self.selected_seed_info.setHtml(info_text)
        
        # Emit signal
        self.seed_selected.emit(seed)
    
    def disperse_seed(self):
        """Disperse the selected seed to the target system"""
        # Get selected seed
        selected_items = self.seeds_list.selectedItems()
        if not selected_items:
            self.log_message("No seed selected for dispersal")
            return
            
        seed_id = selected_items[0].data(Qt.UserRole)
        seed = next((s for s in self.concept_seeds if s.id == seed_id), None)
        if not seed:
            return
            
        # Get target system
        target_system = self.target_combo.currentText()
        
        # Prepare seed for dispersal
        if not self.include_connections.isChecked():
            seed.connections.clear()
            
        if self.expand_attributes.isChecked() and len(seed.attributes) < 3:
            # Add some generated attributes if none exist
            if not seed.attributes:
                seed.attributes["relevance"] = random.uniform(0.6, 1.0)
                seed.attributes["certainty"] = random.uniform(0.7, 0.9)
            # Add complexity attribute
            seed.attributes["complexity"] = random.uniform(0.3, 0.8)
        
        # Log the dispersal
        self.log_message(f"Dispersing '{seed.concept}' to {target_system}")
        
        # Emit the dispersal signal
        self.seed_dispersed.emit(seed, target_system)
        
        # Highlight in the list to indicate it's been dispersed
        for i in range(self.seeds_list.count()):
            item = self.seeds_list.item(i)
            if item.data(Qt.UserRole) == seed_id:
                item.setText(f"✓ {seed.concept} ({seed.weight:.2f})")
                break
    
    def log_message(self, message: str):
        """Add a message to the activity log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
class TempleSeedBridge(QMainWindow):
    """
    Main window for the Temple to Seed Bridge system, which connects
    the Spatial Temple visualization with the Seed Dispersal system.
    """
    def __init__(self, temple_mapper: Optional[SpatialTempleMapper] = None):
        super().__init__()
        self.setWindowTitle("Lumina v8 - Temple to Seed Bridge")
        self.temple_mapper = temple_mapper
        
        # Initialize auto growth system if available
        self.auto_growth_system = None
        if AUTO_GROWTH_AVAILABLE:
            self.auto_growth_system = AutoSeedGrowthSystem(temple_mapper)
        
        self.setup_ui()
        
        # Start seed system in background
        self.start_seed_system()
        
    def setup_ui(self):
        """Setup the main UI"""
        self.setMinimumSize(800, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel("Temple to Seed Bridge: Concept Flow System")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Explanation
        explanation = QLabel(
            "This system bridges the Spatial Temple visualization with the Seed Dispersal system.\n"
            "Concepts from the temple flow down like fruits from a tree, spreading seeds throughout the system."
        )
        explanation.setAlignment(Qt.AlignCenter)
        layout.addWidget(explanation)
        
        # Main flow panel
        self.flow_panel = TreeFlowPanel(self.temple_mapper)
        self.flow_panel.seed_dispersed.connect(self.handle_seed_dispersal)
        layout.addWidget(self.flow_panel)
        
        # Auto Growth Controls (if available)
        if AUTO_GROWTH_AVAILABLE:
            growth_frame = QFrame()
            growth_frame.setFrameStyle(QFrame.StyledPanel)
            growth_layout = QVBoxLayout(growth_frame)
            
            growth_layout.addWidget(QLabel("<b>Automated Knowledge Growth</b>"))
            
            growth_status = QLabel("Auto growth system ready")
            growth_layout.addWidget(growth_status)
            self.growth_status = growth_status
            
            button_layout = QHBoxLayout()
            
            start_btn = QPushButton("Start Auto Growth")
            start_btn.clicked.connect(self.start_auto_growth)
            button_layout.addWidget(start_btn)
            
            stop_btn = QPushButton("Stop Auto Growth")
            stop_btn.clicked.connect(self.stop_auto_growth)
            button_layout.addWidget(stop_btn)
            
            growth_layout.addLayout(button_layout)
            
            layout.addWidget(growth_frame)
            
            # Start a timer to update growth stats
            self.growth_timer = QTimer(self)
            self.growth_timer.timeout.connect(self.update_growth_stats)
            self.growth_timer.start(5000)  # Update every 5 seconds
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        open_temple_btn = QPushButton("Open Spatial Temple")
        open_temple_btn.clicked.connect(self.open_spatial_temple)
        button_layout.addWidget(open_temple_btn)
        
        open_seed_btn = QPushButton("Open Seed System")
        open_seed_btn.clicked.connect(self.open_seed_system)
        button_layout.addWidget(open_seed_btn)
        
        layout.addLayout(button_layout)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def start_seed_system(self):
        """Initialize connection to the seed system"""
        try:
            from src.seed import get_neural_seed
            self.seed = get_neural_seed()
            self.statusBar().showMessage("Connected to seed system")
            self.flow_panel.log_message("Successfully connected to the seed system")
        except Exception as e:
            self.statusBar().showMessage("Failed to connect to seed system")
            self.flow_panel.log_message(f"Error connecting to seed system: {str(e)}")
            self.seed = None
    
    def set_temple_mapper(self, temple_mapper: SpatialTempleMapper):
        """Set the temple mapper and update the UI"""
        self.temple_mapper = temple_mapper
        self.flow_panel.set_temple_mapper(temple_mapper)
        
        # Update auto growth system if available
        if self.auto_growth_system:
            self.auto_growth_system.set_temple_mapper(temple_mapper)
    
    def handle_seed_dispersal(self, seed: ConceptSeed, target_system: str):
        """Handle a seed being dispersed to a target system"""
        if target_system == "Seed System" and self.seed:
            try:
                # Convert concept seed to knowledge needed by seed system
                seed_dict = seed.to_dict()
                
                # Add to seed system's dictionary with weight-based scaling
                self.seed.dictionary[seed.concept.lower()] = {
                    "weight": seed.weight,
                    "type": seed.node_type,
                    "connections": len(seed.connections),
                    "attributes": seed.attributes
                }
                
                # Log success
                self.flow_panel.log_message(f"Seed '{seed.concept}' added to seed system dictionary")
                self.statusBar().showMessage(f"Seed dispersed to {target_system}")
                
            except Exception as e:
                self.flow_panel.log_message(f"Error dispersing to seed system: {str(e)}")
        else:
            # For other systems, just log the attempt
            self.flow_panel.log_message(f"Seed '{seed.concept}' would be dispersed to {target_system}")
            self.statusBar().showMessage(f"Simulated dispersal to {target_system}")
    
    def start_auto_growth(self):
        """Start the automated growth system"""
        if not self.auto_growth_system:
            self.flow_panel.log_message("Auto growth system not available")
            return
            
        try:
            self.auto_growth_system.start()
            self.flow_panel.log_message("Started automated seed growth")
            self.growth_status.setText("Auto growth system running")
            self.statusBar().showMessage("Auto growth system running")
        except Exception as e:
            self.flow_panel.log_message(f"Error starting auto growth: {str(e)}")
    
    def stop_auto_growth(self):
        """Stop the automated growth system"""
        if not self.auto_growth_system:
            return
            
        try:
            self.auto_growth_system.stop()
            self.flow_panel.log_message("Stopped automated seed growth")
            self.growth_status.setText("Auto growth system stopped")
            self.statusBar().showMessage("Auto growth system stopped")
        except Exception as e:
            self.flow_panel.log_message(f"Error stopping auto growth: {str(e)}")
    
    def update_growth_stats(self):
        """Update the growth statistics display"""
        if not self.auto_growth_system:
            return
            
        try:
            stats = self.auto_growth_system.get_statistics()
            status_text = f"Status: {'Running' if stats['running'] else 'Stopped'} | "
            status_text += f"Seeds: {stats['total_seeds']} | "
            status_text += f"Sources: {stats['knowledge_sources_accessed']} | "
            status_text += f"Concepts added: {stats['concepts_added']}"
            
            self.growth_status.setText(status_text)
            
            # If growth is occurring, update the log occasionally
            if stats['running'] and stats['growth_events'] > 0 and random.random() < 0.3:
                self.flow_panel.log_message(f"Auto growth: Added {stats['concepts_added']} concepts from {stats['knowledge_sources_accessed']} sources")
        except Exception as e:
            logger.error(f"Error updating growth stats: {e}")
    
    def open_spatial_temple(self):
        """Open the spatial temple visualization"""
        self.flow_panel.log_message("Opening Spatial Temple visualization...")
        
        try:
            from src.v8.run_spatial_temple import main
            # This would normally be called in a separate process
            self.flow_panel.log_message("Please run the spatial temple visualization separately using run_spatial_temple.py")
        except Exception as e:
            self.flow_panel.log_message(f"Error opening spatial temple: {str(e)}")
    
    def open_seed_system(self):
        """Open the seed dispersal system"""
        self.flow_panel.log_message("Opening Seed Dispersal system...")
        
        try:
            # This would normally be called in a separate process
            self.flow_panel.log_message("Please run the seed dispersal system separately using seed_dispersal_system.py")
        except Exception as e:
            self.flow_panel.log_message(f"Error opening seed system: {str(e)}")
            
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop auto growth system if running
        if self.auto_growth_system and hasattr(self.auto_growth_system, 'get_statistics'):
            stats = self.auto_growth_system.get_statistics()
            if stats.get('running', False):
                self.stop_auto_growth()
        
        super().closeEvent(event)

def create_demo_temple_mapper() -> SpatialTempleMapper:
    """Create a demo temple mapper with nodes for testing"""
    mapper = SpatialTempleMapper()
    
    # Generate demo nodes with temple theme
    demo_nodes = generate_themed_demo_nodes(50, "temple")
    
    # Add to mapper
    for node in demo_nodes:
        mapper.nodes[node.id] = node
        
    return mapper

def run_temple_bridge():
    """Run the Temple to Seed Bridge system"""
    app = QApplication(sys.argv)
    
    # Create demo temple mapper
    temple_mapper = create_demo_temple_mapper()
    
    # Create and show window
    bridge = TempleSeedBridge(temple_mapper)
    bridge.show()
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(run_temple_bridge()) 