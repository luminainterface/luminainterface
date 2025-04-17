#!/usr/bin/env python3
"""
Concept Save System (v8)

This module extends the Temple-to-Seed Bridge with advanced concept preservation
and connection capabilities. Like a researcher attaching notes and connecting
them with yarn on a concept board, this system allows:

1. Saving new concept sources
2. Adding additional sources to existing concepts
3. Creating connections ("yarn") between concepts
4. Visualizing the connected concept network

This creates a persistent, evolving knowledge web of interconnected ideas.
"""

import os
import sys
import logging
import json
from pathlib import Path
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
import random

from PySide6.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTextEdit, QListWidget, QListWidgetItem,
    QComboBox, QFrame, QSplitter, QLineEdit, QDialog, QDialogButtonBox,
    QTabWidget, QScrollArea, QMenu, QTreeWidget, QTreeWidgetItem,
    QFileDialog
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QColor, QPalette, QFont, QIcon, QAction

# Add parent directory to path for imports
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

# Import temple-to-seed bridge components
from src.v8.temple_to_seed_bridge import ConceptSeed, TempleSeedBridge
from src.v8.spatial_temple_mapper import SpatialNode, SpatialTempleMapper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v8.saveconcept")

class ConceptSource:
    """
    Represents a source of information attached to a concept
    Like a reference or citation that supports an idea
    """
    def __init__(self, 
                title: str, 
                content: str, 
                source_type: str = "note",
                metadata: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())
        self.title = title
        self.content = content
        self.source_type = source_type  # "note", "web", "book", "conversation", etc.
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "source_type": self.source_type,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConceptSource':
        """Create from dictionary"""
        source = cls(
            title=data["title"],
            content=data["content"],
            source_type=data["source_type"],
            metadata=data["metadata"]
        )
        source.id = data["id"]
        source.created_at = data["created_at"]
        source.updated_at = data["updated_at"]
        return source
        
    def update_content(self, new_content: str):
        """Update the source content"""
        self.content = new_content
        self.updated_at = datetime.now().isoformat()

class ConceptYarn:
    """
    Represents a connection between two concepts
    Like a piece of yarn connecting two post-it notes on a concept board
    """
    def __init__(self, 
                from_concept_id: str, 
                to_concept_id: str,
                relationship_type: str = "related",
                strength: float = 1.0,
                notes: str = "",
                metadata: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())
        self.from_concept_id = from_concept_id
        self.to_concept_id = to_concept_id
        self.relationship_type = relationship_type  # "related", "causes", "supports", "contradicts", etc.
        self.strength = strength  # 0.0 to 1.0
        self.notes = notes
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "from_concept_id": self.from_concept_id,
            "to_concept_id": self.to_concept_id,
            "relationship_type": self.relationship_type,
            "strength": self.strength,
            "notes": self.notes,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConceptYarn':
        """Create from dictionary"""
        yarn = cls(
            from_concept_id=data["from_concept_id"],
            to_concept_id=data["to_concept_id"],
            relationship_type=data["relationship_type"],
            strength=data["strength"],
            notes=data["notes"],
            metadata=data["metadata"]
        )
        yarn.id = data["id"]
        yarn.created_at = data["created_at"]
        yarn.updated_at = data["updated_at"]
        return yarn
        
    def update(self, relationship_type: Optional[str] = None, 
              strength: Optional[float] = None, 
              notes: Optional[str] = None):
        """Update the yarn properties"""
        if relationship_type:
            self.relationship_type = relationship_type
        if strength is not None:
            self.strength = strength
        if notes is not None:
            self.notes = notes
        self.updated_at = datetime.now().isoformat()

class EnhancedConcept:
    """
    An enhanced version of a concept with multiple sources and connections
    Like a post-it note on a concept board that can have multiple references
    and yarn connections to other concepts
    """
    def __init__(self, 
                name: str, 
                description: str = "",
                concept_type: str = "general",
                seed: Optional[ConceptSeed] = None,
                metadata: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.concept_type = concept_type
        self.seed_id = seed.id if seed else None
        self.sources: Dict[str, ConceptSource] = {}  # Sources by ID
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.last_accessed = self.created_at
        self.access_count = 0
        
        # Add the seed description as the first source if provided
        if seed:
            self.add_source(
                title=f"Initial concept: {seed.concept}",
                content=f"Type: {seed.node_type}\n"
                        f"Weight: {seed.weight}\n"
                        f"Connections: {', '.join(list(seed.connections)[:5])}\n"
                        f"Attributes: {json.dumps(seed.attributes, indent=2) if seed.attributes else 'None'}",
                source_type="seed"
            )
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "concept_type": self.concept_type,
            "seed_id": self.seed_id,
            "sources": {src_id: src.to_dict() for src_id, src in self.sources.items()},
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedConcept':
        """Create from dictionary"""
        concept = cls(
            name=data["name"],
            description=data["description"],
            concept_type=data["concept_type"],
            metadata=data["metadata"]
        )
        concept.id = data["id"]
        concept.seed_id = data["seed_id"]
        concept.created_at = data["created_at"]
        concept.updated_at = data["updated_at"]
        concept.last_accessed = data["last_accessed"]
        concept.access_count = data["access_count"]
        
        # Add sources
        for src_data in data["sources"].values():
            source = ConceptSource.from_dict(src_data)
            concept.sources[source.id] = source
            
        return concept
        
    def add_source(self, title: str, content: str, source_type: str = "note", metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a new source to this concept and return its ID"""
        source = ConceptSource(title=title, content=content, source_type=source_type, metadata=metadata)
        self.sources[source.id] = source
        self.updated_at = datetime.now().isoformat()
        return source.id
        
    def remove_source(self, source_id: str) -> bool:
        """Remove a source by ID"""
        if source_id in self.sources:
            del self.sources[source_id]
            self.updated_at = datetime.now().isoformat()
            return True
        return False
        
    def access(self):
        """Mark concept as accessed"""
        self.last_accessed = datetime.now().isoformat()
        self.access_count += 1

class ConceptBoard:
    """
    Manages the collection of concepts and their connections
    Like a physical board with post-it notes and yarn connections
    """
    def __init__(self, name: str = "Main Concept Board"):
        self.name = name
        self.concepts: Dict[str, EnhancedConcept] = {}  # Concepts by ID
        self.yarns: Dict[str, ConceptYarn] = {}  # Yarns by ID
        self.data_directory = os.path.join(project_root, "data", "v8", "concepts")
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        
        # Create data directory if needed
        os.makedirs(self.data_directory, exist_ok=True)
        
    def save_to_disk(self, filename: Optional[str] = None):
        """Save board to disk"""
        if not filename:
            filename = f"{self.name.lower().replace(' ', '_')}_board.json"
        
        filepath = os.path.join(self.data_directory, filename)
        
        data = {
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": datetime.now().isoformat(),
            "concepts": {concept_id: concept.to_dict() for concept_id, concept in self.concepts.items()},
            "yarns": {yarn_id: yarn.to_dict() for yarn_id, yarn in self.yarns.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved concept board to {filepath}")
        return filepath
        
    @classmethod
    def load_from_disk(cls, filepath: str) -> 'ConceptBoard':
        """Load board from disk"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        board = cls(name=data["name"])
        board.created_at = data["created_at"]
        board.updated_at = data["updated_at"]
        
        # Load concepts
        for concept_data in data["concepts"].values():
            concept = EnhancedConcept.from_dict(concept_data)
            board.concepts[concept.id] = concept
            
        # Load yarns
        for yarn_data in data["yarns"].values():
            yarn = ConceptYarn.from_dict(yarn_data)
            board.yarns[yarn.id] = yarn
            
        logger.info(f"Loaded concept board from {filepath} with {len(board.concepts)} concepts and {len(board.yarns)} connections")
        return board
        
    def add_concept(self, 
                   name: str, 
                   description: str = "", 
                   concept_type: str = "general",
                   seed: Optional[ConceptSeed] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a new concept to the board and return its ID"""
        concept = EnhancedConcept(
            name=name,
            description=description,
            concept_type=concept_type,
            seed=seed,
            metadata=metadata
        )
        self.concepts[concept.id] = concept
        self.updated_at = datetime.now().isoformat()
        return concept.id
        
    def create_concept_from_seed(self, seed: ConceptSeed) -> str:
        """Create a new concept from a seed and return its ID"""
        return self.add_concept(
            name=seed.concept,
            description=f"Concept created from seed: {seed.concept}",
            concept_type=seed.node_type,
            seed=seed,
            metadata={"weight": seed.weight, "seed_source": "temple"}
        )
        
    def remove_concept(self, concept_id: str) -> bool:
        """Remove a concept and all its connections"""
        if concept_id not in self.concepts:
            return False
            
        # Remove concept
        del self.concepts[concept_id]
        
        # Remove all yarns connected to this concept
        yarns_to_remove = []
        for yarn_id, yarn in self.yarns.items():
            if yarn.from_concept_id == concept_id or yarn.to_concept_id == concept_id:
                yarns_to_remove.append(yarn_id)
                
        for yarn_id in yarns_to_remove:
            del self.yarns[yarn_id]
            
        self.updated_at = datetime.now().isoformat()
        return True
        
    def connect_concepts(self, 
                        from_id: str, 
                        to_id: str, 
                        relationship_type: str = "related",
                        strength: float = 1.0,
                        notes: str = "") -> Optional[str]:
        """Connect two concepts with yarn and return the yarn ID"""
        if from_id not in self.concepts or to_id not in self.concepts:
            logger.error(f"Cannot connect: one or both concepts not found")
            return None
            
        # Check for existing connection
        for yarn in self.yarns.values():
            if ((yarn.from_concept_id == from_id and yarn.to_concept_id == to_id) or
                (yarn.from_concept_id == to_id and yarn.to_concept_id == from_id and relationship_type == "related")):
                logger.info(f"Connection already exists between these concepts")
                return yarn.id
                
        # Create new connection
        yarn = ConceptYarn(
            from_concept_id=from_id,
            to_concept_id=to_id,
            relationship_type=relationship_type,
            strength=strength,
            notes=notes
        )
        self.yarns[yarn.id] = yarn
        self.updated_at = datetime.now().isoformat()
        return yarn.id
        
    def get_connected_concepts(self, concept_id: str) -> List[Tuple[str, str]]:
        """Get IDs and relationship types of all concepts connected to the given concept"""
        if concept_id not in self.concepts:
            return []
            
        connections = []
        for yarn in self.yarns.values():
            if yarn.from_concept_id == concept_id:
                connections.append((yarn.to_concept_id, yarn.relationship_type))
            elif yarn.to_concept_id == concept_id and yarn.relationship_type == "related":
                connections.append((yarn.from_concept_id, "related"))
                
        return connections
        
    def get_all_concept_names(self) -> List[Tuple[str, str]]:
        """Get a list of all concept IDs and names"""
        return [(concept_id, concept.name) for concept_id, concept in self.concepts.items()] 

class AddSourceDialog(QDialog):
    """Dialog for adding a new source to a concept"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Source")
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Title:"))
        self.title_input = QLineEdit()
        title_layout.addWidget(self.title_input)
        layout.addLayout(title_layout)
        
        # Source type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["note", "web", "book", "article", "conversation", "research", "other"])
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)
        
        # Content
        layout.addWidget(QLabel("Content:"))
        self.content_input = QTextEdit()
        self.content_input.setMinimumHeight(200)
        layout.addWidget(self.content_input)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def get_source_data(self) -> Tuple[str, str, str]:
        """Return the source title, content, and type"""
        return (
            self.title_input.text(),
            self.content_input.toPlainText(),
            self.type_combo.currentText()
        )

class AddConnectionDialog(QDialog):
    """Dialog for adding a new connection between concepts"""
    
    def __init__(self, concept_id: str, available_concepts: List[Tuple[str, str]], parent=None):
        super().__init__(parent)
        self.concept_id = concept_id
        self.available_concepts = available_concepts
        self.setWindowTitle("Connect Concepts")
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Target concept
        layout.addWidget(QLabel("Connect to concept:"))
        self.target_combo = QComboBox()
        for concept_id, concept_name in self.available_concepts:
            if concept_id != self.concept_id:  # Don't connect to self
                self.target_combo.addItem(concept_name, concept_id)
        layout.addWidget(self.target_combo)
        
        # Relationship type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Relationship:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["related", "causes", "supports", "contradicts", "expands", "part_of", "example_of"])
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)
        
        # Strength
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(QLabel("Strength:"))
        self.strength_combo = QComboBox()
        self.strength_combo.addItems(["Strong (1.0)", "High (0.75)", "Medium (0.5)", "Low (0.25)", "Weak (0.1)"])
        strength_layout.addWidget(self.strength_combo)
        layout.addLayout(strength_layout)
        
        # Notes
        layout.addWidget(QLabel("Notes:"))
        self.notes_input = QTextEdit()
        self.notes_input.setMaximumHeight(100)
        layout.addWidget(self.notes_input)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def get_connection_data(self) -> Tuple[str, str, float, str]:
        """Return target concept ID, relationship type, strength, and notes"""
        # Get strength value from the combo box text
        strength_text = self.strength_combo.currentText()
        strength = float(strength_text.split("(")[1].split(")")[0])
        
        return (
            self.target_combo.currentData(),
            self.type_combo.currentText(),
            strength,
            self.notes_input.toPlainText()
        )

class ConceptView(QWidget):
    """Widget for viewing and editing a single concept"""
    
    concept_updated = Signal(str)  # Signal when concept is updated
    connection_added = Signal(str, str)  # Signal when a connection is added
    
    def __init__(self, concept: EnhancedConcept, board: ConceptBoard, parent=None):
        super().__init__(parent)
        self.concept = concept
        self.board = board
        self.setup_ui()
        self.update_display()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        self.concept_name = QLabel(self.concept.name)
        self.concept_name.setFont(QFont("Arial", 14, QFont.Bold))
        header_layout.addWidget(self.concept_name)
        
        # Connection button
        connect_btn = QPushButton("Connect...")
        connect_btn.clicked.connect(self.add_connection)
        header_layout.addWidget(connect_btn)
        
        layout.addLayout(header_layout)
        
        # Description
        self.description_display = QLabel(self.concept.description)
        self.description_display.setWordWrap(True)
        layout.addWidget(self.description_display)
        
        # Tab widget for sources and connections
        self.tabs = QTabWidget()
        
        # Sources tab
        sources_widget = QWidget()
        sources_layout = QVBoxLayout(sources_widget)
        
        # Source list
        self.sources_list = QListWidget()
        self.sources_list.setMinimumHeight(150)
        self.sources_list.itemClicked.connect(self.show_source_content)
        sources_layout.addWidget(self.sources_list)
        
        # Source content
        self.source_content = QTextEdit()
        self.source_content.setReadOnly(True)
        sources_layout.addWidget(self.source_content)
        
        # Add source button
        add_source_btn = QPushButton("Add Source")
        add_source_btn.clicked.connect(self.add_source)
        sources_layout.addWidget(add_source_btn)
        
        self.tabs.addTab(sources_widget, "Sources")
        
        # Connections tab
        connections_widget = QWidget()
        connections_layout = QVBoxLayout(connections_widget)
        
        # Connections list
        self.connections_list = QListWidget()
        connections_layout.addWidget(self.connections_list)
        
        self.tabs.addTab(connections_widget, "Connections")
        
        layout.addWidget(self.tabs)
        
    def update_display(self):
        """Update the display with current concept data"""
        # Update header
        self.concept_name.setText(self.concept.name)
        self.description_display.setText(self.concept.description)
        
        # Update sources list
        self.sources_list.clear()
        for source_id, source in self.concept.sources.items():
            item = QListWidgetItem(f"{source.title} ({source.source_type})")
            item.setData(Qt.UserRole, source_id)
            self.sources_list.addItem(item)
            
        # Update connections list
        self.connections_list.clear()
        connections = self.board.get_connected_concepts(self.concept.id)
        for connected_id, relationship in connections:
            if connected_id in self.board.concepts:
                connected_concept = self.board.concepts[connected_id]
                item = QListWidgetItem(f"{connected_concept.name} ({relationship})")
                item.setData(Qt.UserRole, connected_id)
                self.connections_list.addItem(item)
        
    def show_source_content(self, item: QListWidgetItem):
        """Show the content of the selected source"""
        source_id = item.data(Qt.UserRole)
        if source_id in self.concept.sources:
            source = self.concept.sources[source_id]
            self.source_content.setPlainText(source.content)
        
    def add_source(self):
        """Add a new source to the concept"""
        dialog = AddSourceDialog(self)
        if dialog.exec_():
            title, content, source_type = dialog.get_source_data()
            if title and content:
                self.concept.add_source(title, content, source_type)
                self.update_display()
                self.concept_updated.emit(self.concept.id)
                
    def add_connection(self):
        """Add a connection to another concept"""
        available_concepts = self.board.get_all_concept_names()
        dialog = AddConnectionDialog(self.concept.id, available_concepts, self)
        if dialog.exec_():
            target_id, relationship, strength, notes = dialog.get_connection_data()
            if target_id:
                yarn_id = self.board.connect_concepts(
                    from_id=self.concept.id,
                    to_id=target_id,
                    relationship_type=relationship,
                    strength=strength,
                    notes=notes
                )
                if yarn_id:
                    self.update_display()
                    self.connection_added.emit(self.concept.id, target_id)

class ConceptBoardUI(QMainWindow):
    """Main UI for the concept board"""
    
    def __init__(self, board: Optional[ConceptBoard] = None, temple_bridge: Optional[TempleSeedBridge] = None):
        super().__init__()
        self.board = board or ConceptBoard()
        self.temple_bridge = temple_bridge
        self.current_concept_view = None
        
        self.setWindowTitle(f"Concept Board - {self.board.name}")
        self.setMinimumSize(900, 700)
        self.setup_ui()
        self.update_concept_list()
        
        # Auto-save timer
        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start(60000)  # Save every minute
        
    def setup_ui(self):
        # Create central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - concept list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Concept list
        left_layout.addWidget(QLabel("Concepts:"))
        self.concept_list = QListWidget()
        self.concept_list.setMinimumWidth(200)
        self.concept_list.itemClicked.connect(self.show_concept)
        left_layout.addWidget(self.concept_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Add concept button
        add_btn = QPushButton("Add Concept")
        add_btn.clicked.connect(self.add_new_concept)
        button_layout.addWidget(add_btn)
        
        # Import from seed button
        if self.temple_bridge:
            import_btn = QPushButton("Import from Temple")
            import_btn.clicked.connect(self.import_from_temple)
            button_layout.addWidget(import_btn)
        
        left_layout.addLayout(button_layout)
        
        # Right panel - concept view
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.addWidget(QLabel("Select a concept to view details"))
        
        splitter.addWidget(left_panel)
        splitter.addWidget(self.right_panel)
        splitter.setSizes([200, 700])
        
        main_layout.addWidget(splitter)
        
        # Menu
        self.setup_menu()
        
    def setup_menu(self):
        # File menu
        file_menu = self.menuBar().addMenu("File")
        
        # New board
        new_action = QAction("New Board", self)
        new_action.triggered.connect(self.new_board)
        file_menu.addAction(new_action)
        
        # Save board
        save_action = QAction("Save Board", self)
        save_action.triggered.connect(self.save_board)
        file_menu.addAction(save_action)
        
        # Load board
        load_action = QAction("Load Board", self)
        load_action.triggered.connect(self.load_board)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        # Exit
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
    def update_concept_list(self):
        """Update the concept list display"""
        self.concept_list.clear()
        for concept_id, concept in self.board.concepts.items():
            item = QListWidgetItem(concept.name)
            item.setData(Qt.UserRole, concept_id)
            self.concept_list.addItem(item)
            
    def show_concept(self, item: QListWidgetItem):
        """Show the selected concept"""
        concept_id = item.data(Qt.UserRole)
        if concept_id in self.board.concepts:
            concept = self.board.concepts[concept_id]
            
            # Clear right panel
            self.clear_right_panel()
            
            # Create concept view
            self.current_concept_view = ConceptView(concept, self.board)
            self.current_concept_view.concept_updated.connect(self.on_concept_updated)
            self.current_concept_view.connection_added.connect(self.on_connection_added)
            self.right_layout.addWidget(self.current_concept_view)
            
            # Mark concept as accessed
            concept.access()
            
    def clear_right_panel(self):
        """Clear the right panel"""
        if self.current_concept_view:
            self.current_concept_view.deleteLater()
            self.current_concept_view = None
            
        # Remove all widgets from the layout
        while self.right_layout.count():
            item = self.right_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
    def add_new_concept(self):
        """Add a new concept to the board"""
        name, ok = QLineEdit.getText(self, "New Concept", "Enter concept name:")
        if ok and name:
            concept_id = self.board.add_concept(name=name)
            self.update_concept_list()
            
            # Select the new concept
            for i in range(self.concept_list.count()):
                item = self.concept_list.item(i)
                if item.data(Qt.UserRole) == concept_id:
                    self.concept_list.setCurrentItem(item)
                    self.show_concept(item)
                    break
                    
    def import_from_temple(self):
        """Import a concept from the temple"""
        if not self.temple_bridge:
            return
            
        # This would be implemented to get a seed from the temple bridge
        # For now, we'll just create a demo seed
        seed = ConceptSeed(
            concept="Imported Temple Concept",
            weight=0.8,
            node_type="temple_import",
            connections=set(["related_concept_1", "related_concept_2"]),
            attributes={"source": "temple", "importance": "high"}
        )
        
        concept_id = self.board.create_concept_from_seed(seed)
        self.update_concept_list()
        
        # Select the new concept
        for i in range(self.concept_list.count()):
            item = self.concept_list.item(i)
            if item.data(Qt.UserRole) == concept_id:
                self.concept_list.setCurrentItem(item)
                self.show_concept(item)
                break
    
    def on_concept_updated(self, concept_id: str):
        """Handle concept update"""
        # Just update the display for now
        self.update_concept_list()
        
    def on_connection_added(self, from_id: str, to_id: str):
        """Handle new connection between concepts"""
        # Refresh the concept view to show the new connection
        if self.current_concept_view and self.current_concept_view.concept.id == from_id:
            self.current_concept_view.update_display()
            
    def new_board(self):
        """Create a new board"""
        name, ok = QLineEdit.getText(self, "New Board", "Enter board name:")
        if ok and name:
            # Ask to save current board if it has concepts
            if self.board.concepts and self.confirm_save_dialog():
                self.save_board()
                
            # Create new board
            self.board = ConceptBoard(name=name)
            self.setWindowTitle(f"Concept Board - {self.board.name}")
            self.update_concept_list()
            self.clear_right_panel()
            
    def save_board(self):
        """Save the current board"""
        filepath = self.board.save_to_disk()
        self.statusBar().showMessage(f"Saved to {filepath}", 3000)
        
    def load_board(self):
        """Load a board from disk"""
        # Ask to save current board if it has concepts
        if self.board.concepts and self.confirm_save_dialog():
            self.save_board()
            
        # Open file dialog
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Board", self.board.data_directory, "JSON Files (*.json)"
        )
        
        if filepath:
            try:
                self.board = ConceptBoard.load_from_disk(filepath)
                self.setWindowTitle(f"Concept Board - {self.board.name}")
                self.update_concept_list()
                self.clear_right_panel()
                self.statusBar().showMessage(f"Loaded board from {filepath}", 3000)
            except Exception as e:
                logger.error(f"Error loading board: {e}")
                self.statusBar().showMessage(f"Error loading board: {e}", 5000)
                
    def confirm_save_dialog(self) -> bool:
        """Show dialog asking to save current board"""
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, "Save Board?", 
            "Do you want to save the current board?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )
        
        if reply == QMessageBox.Cancel:
            return False
        return reply == QMessageBox.Yes
        
    def autosave(self):
        """Auto-save the board"""
        if self.board.concepts:
            try:
                autosave_dir = os.path.join(self.board.data_directory, "autosave")
                os.makedirs(autosave_dir, exist_ok=True)
                
                filepath = os.path.join(autosave_dir, f"{self.board.name.lower().replace(' ', '_')}_autosave.json")
                
                # Save board data
                data = {
                    "name": self.board.name,
                    "created_at": self.board.created_at,
                    "updated_at": datetime.now().isoformat(),
                    "concepts": {concept_id: concept.to_dict() for concept_id, concept in self.board.concepts.items()},
                    "yarns": {yarn_id: yarn.to_dict() for yarn_id, yarn in self.board.yarns.items()}
                }
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                    
                logger.info(f"Auto-saved board to {filepath}")
            except Exception as e:
                logger.error(f"Error during autosave: {e}")
                
    def closeEvent(self, event):
        """Handle window close event"""
        if self.board.concepts and self.confirm_save_dialog():
            self.save_board()
        event.accept()

class TempleBridgeAdapter:
    """
    Adapter that connects the concept board to the temple bridge system
    for importing concepts directly from the temple
    """
    
    def __init__(self, board: ConceptBoard):
        """Initialize with a concept board"""
        self.board = board
        self.temple_bridge = None
        self.seed_mapping = {}  # Map seed IDs to concept IDs
        self.concept_board_ui = None
        
    def connect_to_temple_bridge(self, temple_bridge: TempleSeedBridge):
        """Connect to a temple bridge instance"""
        self.temple_bridge = temple_bridge
        
        # Connect to temple bridge signals if possible
        if hasattr(self.temple_bridge, 'seed_selected'):
            self.temple_bridge.seed_selected.connect(self.on_seed_selected)
        
    def set_ui(self, ui: ConceptBoardUI):
        """Set the UI instance for updates"""
        self.concept_board_ui = ui
        
    def on_seed_selected(self, seed: ConceptSeed):
        """Handle when a seed is selected in the temple bridge"""
        # Create a concept from the seed if it doesn't exist
        if seed.id not in self.seed_mapping:
            concept_id = self.board.create_concept_from_seed(seed)
            self.seed_mapping[seed.id] = concept_id
            
            # Update UI if available
            if self.concept_board_ui:
                self.concept_board_ui.update_concept_list()
        
    def import_seed(self, seed: ConceptSeed) -> str:
        """Import a seed as a concept and return the concept ID"""
        if seed.id in self.seed_mapping:
            return self.seed_mapping[seed.id]
            
        concept_id = self.board.create_concept_from_seed(seed)
        self.seed_mapping[seed.id] = concept_id
        
        # Update UI if available
        if self.concept_board_ui:
            self.concept_board_ui.update_concept_list()
            
        return concept_id
        
    def get_seeds_from_temple(self) -> List[ConceptSeed]:
        """Get available seeds from the temple bridge"""
        if not self.temple_bridge or not hasattr(self.temple_bridge, 'get_available_seeds'):
            # Create some demo seeds if we can't get real ones
            return [
                ConceptSeed(
                    concept=f"Temple Concept {i}",
                    weight=random.random(),
                    node_type="temple_concept",
                    connections=set([f"related_{j}" for j in range(3)])
                )
                for i in range(5)
            ]
            
        return self.temple_bridge.get_available_seeds()

class ImportSeedDialog(QDialog):
    """Dialog for importing seeds from the temple"""
    
    def __init__(self, adapter: TempleBridgeAdapter, parent=None):
        super().__init__(parent)
        self.adapter = adapter
        self.selected_seed = None
        self.setWindowTitle("Import from Temple")
        self.setup_ui()
        self.populate_seeds()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        layout.addWidget(QLabel("Available Temple Seeds:"))
        
        # Seed list
        self.seed_list = QListWidget()
        self.seed_list.setMinimumHeight(200)
        self.seed_list.itemClicked.connect(self.on_seed_selected)
        layout.addWidget(self.seed_list)
        
        # Seed details
        layout.addWidget(QLabel("Seed Details:"))
        self.seed_details = QTextEdit()
        self.seed_details.setReadOnly(True)
        layout.addWidget(self.seed_details)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Disable OK button until a seed is selected
        self.ok_button = button_box.button(QDialogButtonBox.Ok)
        self.ok_button.setEnabled(False)
        
    def populate_seeds(self):
        """Populate the seed list"""
        seeds = self.adapter.get_seeds_from_temple()
        
        for seed in seeds:
            item = QListWidgetItem(seed.concept)
            item.setData(Qt.UserRole, seed)
            self.seed_list.addItem(item)
            
    def on_seed_selected(self, item: QListWidgetItem):
        """Handle seed selection"""
        seed = item.data(Qt.UserRole)
        self.selected_seed = seed
        
        # Update details
        details = f"Concept: {seed.concept}\n"
        details += f"Type: {seed.node_type}\n"
        details += f"Weight: {seed.weight:.2f}\n"
        details += f"Connections: {', '.join(list(seed.connections)[:5])}\n"
        
        if seed.attributes:
            details += "\nAttributes:\n"
            for key, value in seed.attributes.items():
                details += f"  {key}: {value}\n"
                
        self.seed_details.setPlainText(details)
        
        # Enable OK button
        self.ok_button.setEnabled(True)

class EnhancedConceptBoardUI(ConceptBoardUI):
    """Enhanced concept board UI with temple bridge integration"""
    
    def __init__(self, board: Optional[ConceptBoard] = None, temple_bridge: Optional[TempleSeedBridge] = None):
        super().__init__(board, temple_bridge)
        
        # Create adapter for temple bridge
        self.adapter = TempleBridgeAdapter(self.board)
        if temple_bridge:
            self.adapter.connect_to_temple_bridge(temple_bridge)
        self.adapter.set_ui(self)
        
    def import_from_temple(self):
        """Import a concept from the temple with dialog"""
        dialog = ImportSeedDialog(self.adapter, self)
        if dialog.exec_() and dialog.selected_seed:
            seed = dialog.selected_seed
            concept_id = self.adapter.import_seed(seed)
            
            # Update display and select the new concept
            self.update_concept_list()
            
            # Select the new concept
            for i in range(self.concept_list.count()):
                item = self.concept_list.item(i)
                if item.data(Qt.UserRole) == concept_id:
                    self.concept_list.setCurrentItem(item)
                    self.show_concept(item)
                    break

def run_concept_board_with_temple():
    """Run the concept board integrated with the temple bridge"""
    app = QApplication(sys.argv)
    
    # Try to import temple bridge components
    try:
        # First try to create a temple bridge directly
        from src.v8.temple_to_seed_bridge import create_demo_temple_mapper, TempleSeedBridge
        temple_mapper = create_demo_temple_mapper()
        temple_bridge = TempleSeedBridge(temple_mapper)
    except Exception as e:
        logger.warning(f"Could not create temple bridge: {e}")
        temple_bridge = None
    
    # Create concept board
    board = ConceptBoard(name="Temple Concept Collection")
    
    # Create enhanced UI with temple integration
    window = EnhancedConceptBoardUI(board, temple_bridge)
    window.show()
    
    return app.exec()

def main():
    """Main entry point with command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Concept Save System for Temple-to-Seed Bridge")
    parser.add_argument('--standalone', action='store_true', help='Run in standalone mode without temple integration')
    parser.add_argument('--load', type=str, help='Load a saved concept board file')
    args = parser.parse_args()
    
    if args.standalone:
        return run_concept_board()
    else:
        return run_concept_board_with_temple()

if __name__ == "__main__":
    main() 