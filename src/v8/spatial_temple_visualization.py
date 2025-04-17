#!/usr/bin/env python3
"""
Spatial Temple Visualization Module (v8)

This module provides a 3D visualization interface for the Spatial Temple,
allowing users to navigate the temple in a three-dimensional environment and
interact with the concepts and chambers.
"""

import sys
import math
import logging
from typing import Dict, List, Tuple, Any, Optional, Set

from PySide6.QtCore import Qt, Signal, Slot, QObject, QTimer, Property, QPropertyAnimation
from PySide6.QtGui import QVector3D, QColor, QFont, QQuaternion, QPainter, QSurfaceFormat
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFrame, QSlider, QSplitter, QTabWidget, QTextEdit, QLineEdit,
    QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem
)

# Import Qt3D modules
try:
    from PySide6.Qt3DCore import Qt3DCore
    from PySide6.Qt3DRender import Qt3DRender
    from PySide6.Qt3DExtras import Qt3DExtras
    from PySide6.Qt3DInput import Qt3DInput
    HAS_QT3D = True
except ImportError:
    logging.warning("Qt3D modules not available. Will use fallback visualization.")
    HAS_QT3D = False

# Add parent directory to path for imports
import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import spatial temple mapper
from src.v8.spatial_temple_mapper import (
    SpatialTempleMapper, SpatialNode, SpatialConnection, TempleZone
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v8.spatial_temple_visualization")

class SpatialTempleWidget(QWidget):
    """
    Widget for displaying a 3D visualization of the spatial temple
    
    This widget provides a complete interface for navigating and
    interacting with the spatial temple in 3D.
    """
    # Signals for navigation and interaction
    concept_selected = Signal(str)  # Emitted when a concept is selected
    zone_entered = Signal(str)      # Emitted when entering a zone
    
    def __init__(self, mapper: Optional[SpatialTempleMapper] = None, mode: str = '3d', parent=None):
        """Initialize the widget with an optional spatial temple mapper"""
        super().__init__(parent)
        self.mapper = mapper
        self.mode = mode.lower()  # '2d' or '3d'
        
        # Temple state
        self.current_position = QVector3D(0, 0, 0)
        self.current_direction = QVector3D(0, 0, -1)  # Looking forward
        self.current_zone_id = None
        self.selected_node_id = None
        
        # Initialize UI
        self.initUI()
        
        # Set up animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_visualization)
        self.animation_timer.start(30)  # ~33 fps
        
    def initUI(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Create a splitter for resizable areas
        splitter = QSplitter(Qt.Horizontal)
        
        # 3D view widget (main area)
        if self.mode == '2d':
            # Force 2D visualization mode
            logger.info("Using 2D visualization mode as requested")
            self.view_area = self.create_fallback_view()
        else:
            # Try 3D visualization or fall back to 2D
            self.view_area = self.create_3d_view()
        
        splitter.addWidget(self.view_area)
        
        # Info and controls panel
        info_panel = QFrame()
        info_panel.setFrameStyle(QFrame.StyledPanel)
        info_panel.setMaximumWidth(300)
        info_layout = QVBoxLayout(info_panel)
        
        # Position info
        pos_frame = QFrame()
        pos_layout = QVBoxLayout(pos_frame)
        self.position_label = QLabel("Position: (0, 0, 0)")
        self.zone_label = QLabel("Current Zone: None")
        pos_layout.addWidget(self.position_label)
        pos_layout.addWidget(self.zone_label)
        
        # Navigation controls
        nav_frame = QFrame()
        nav_layout = QVBoxLayout(nav_frame)
        nav_layout.addWidget(QLabel("<b>Navigation Controls</b>"))
        
        # Move buttons
        move_layout = QHBoxLayout()
        self.btn_forward = QPushButton("Forward")
        self.btn_back = QPushButton("Back")
        self.btn_left = QPushButton("Left")
        self.btn_right = QPushButton("Right")
        self.btn_up = QPushButton("Up")
        self.btn_down = QPushButton("Down")
        
        # Connect move buttons
        self.btn_forward.clicked.connect(lambda: self.move_direction(0, 0, -1))
        self.btn_back.clicked.connect(lambda: self.move_direction(0, 0, 1))
        self.btn_left.clicked.connect(lambda: self.move_direction(-1, 0, 0))
        self.btn_right.clicked.connect(lambda: self.move_direction(1, 0, 0))
        self.btn_up.clicked.connect(lambda: self.move_direction(0, 1, 0))
        self.btn_down.clicked.connect(lambda: self.move_direction(0, -1, 0))
        
        # Arrange buttons
        move_grid = QVBoxLayout()
        row1 = QHBoxLayout()
        row1.addWidget(self.btn_forward)
        row2 = QHBoxLayout()
        row2.addWidget(self.btn_left)
        row2.addWidget(self.btn_right)
        row3 = QHBoxLayout()
        row3.addWidget(self.btn_back)
        row4 = QHBoxLayout()
        row4.addWidget(self.btn_up)
        row4.addWidget(self.btn_down)
        
        move_grid.addLayout(row1)
        move_grid.addLayout(row2)
        move_grid.addLayout(row3)
        move_grid.addLayout(row4)
        nav_layout.addLayout(move_grid)
        
        # Selected concept info
        concept_frame = QFrame()
        concept_layout = QVBoxLayout(concept_frame)
        concept_layout.addWidget(QLabel("<b>Selected Concept</b>"))
        self.concept_label = QLabel("None selected")
        self.concept_info = QLabel("")
        concept_layout.addWidget(self.concept_label)
        concept_layout.addWidget(self.concept_info)
        
        # Add all frames to info panel
        info_layout.addWidget(pos_frame)
        info_layout.addWidget(nav_frame)
        info_layout.addWidget(concept_frame)
        info_layout.addStretch()
        
        # Add components to splitter
        splitter.addWidget(info_panel)
        splitter.setSizes([700, 300])
        
        # Add splitter to main layout
        layout.addWidget(splitter)
    
    def create_3d_view(self) -> QWidget:
        """Create the 3D view widget using Qt3D if available, or a fallback"""
        # Check if Qt3D modules are available
        if not HAS_QT3D:
            logger.warning("Qt3D modules not available. Using 2D fallback visualization.")
            return self.create_fallback_view()
        
        try:
            # Configure surface format for compatibility with integrated GPUs
            # This is important for Intel GPUs which may have OpenGL limitations
            from PySide6.QtGui import QSurfaceFormat
            logger.info("Configuring OpenGL surface format for integrated GPU compatibility")
            
            format = QSurfaceFormat()
            # Try compatibility profile for better support with Intel GPUs
            format.setRenderableType(QSurfaceFormat.OpenGL)
            format.setProfile(QSurfaceFormat.CompatibilityProfile)
            # Use OpenGL 2.1 for maximum compatibility
            format.setVersion(2, 1)
            format.setSamples(4)  # Antialiasing
            QSurfaceFormat.setDefaultFormat(format)
            
            logger.info(f"OpenGL format - Version: {format.majorVersion()}.{format.minorVersion()}, "
                      f"Profile: {format.profile()}, Samples: {format.samples()}")
            
            # Test OpenGL context creation before proceeding
            try:
                from PySide6.QtGui import QOpenGLContext
                test_context = QOpenGLContext()
                context_created = test_context.create()
                logger.info(f"OpenGL context creation test: {'SUCCESS' if context_created else 'FAILED'}")
                if context_created and test_context.isValid():
                    surface_format = test_context.format()
                    logger.info(f"Created context - Version: {surface_format.majorVersion()}.{surface_format.minorVersion()}, "
                              f"Profile: {surface_format.profile()}")
                else:
                    logger.warning("OpenGL context is not valid, may have issues with 3D rendering")
            except Exception as e:
                logger.error(f"OpenGL context test error: {e}")
            
            # Create 3D window with enhanced error handling
            logger.info("Creating Qt3D window")
            view = Qt3DExtras.Qt3DWindow()
            container = QWidget.createWindowContainer(view)
            container.setMinimumSize(640, 480)
            container.setFocusPolicy(Qt.StrongFocus)
            
            # Create root entity
            logger.info("Creating root entity")
            self.root_entity = Qt3DCore.QEntity()
            
            # Create camera
            logger.info("Setting up camera")
            self.camera = view.camera()
            self.camera.setPosition(self.current_position)
            self.camera.setViewCenter(self.current_position + self.current_direction * 10)
            self.camera.setUpVector(QVector3D(0, 1, 0))
            self.camera.setAspectRatio(16.0/9.0)
            self.camera.setFieldOfView(45)
            self.camera.setNearPlane(0.1)
            self.camera.setFarPlane(1000.0)
            
            # Configure lights
            logger.info("Setting up lighting")
            light_entity = Qt3DCore.QEntity(self.root_entity)
            light = Qt3DRender.QPointLight(light_entity)
            light.setColor(QColor(255, 255, 255))
            light.setIntensity(1.5)
            light_transform = Qt3DCore.QTransform(light_entity)
            light_transform.setTranslation(QVector3D(0, 50, 50))
            light_entity.addComponent(light)
            light_entity.addComponent(light_transform)
            
            # Add a background for better visibility
            scene_background = Qt3DExtras.QCuboidMesh()
            background_entity = Qt3DCore.QEntity(self.root_entity)
            background_material = Qt3DExtras.QPhongMaterial(background_entity)
            background_material.setAmbient(QColor(30, 30, 45))
            background_transform = Qt3DCore.QTransform(background_entity)
            background_transform.setScale3D(QVector3D(1000, 1000, 1000))
            background_entity.addComponent(scene_background)
            background_entity.addComponent(background_material)
            background_entity.addComponent(background_transform)
            
            # Test entity - add a simple sphere to verify rendering works
            logger.info("Adding test sphere to verify 3D rendering")
            test_entity = Qt3DCore.QEntity(self.root_entity)
            test_mesh = Qt3DExtras.QSphereMesh()
            test_mesh.setRadius(10.0)
            test_material = Qt3DExtras.QPhongMaterial()
            test_material.setDiffuse(QColor(255, 100, 100))
            test_transform = Qt3DCore.QTransform()
            test_transform.setTranslation(QVector3D(0, 0, 0))
            test_entity.addComponent(test_mesh)
            test_entity.addComponent(test_material)
            test_entity.addComponent(test_transform)
            
            # Create temple entities
            logger.info("Creating temple entities")
            self.create_temple_entities()
            
            # Set root entity
            logger.info("Setting root entity")
            view.setRootEntity(self.root_entity)
            
            # Add a warning message for users with integrated GPUs
            logger.info("3D view created successfully with integrated GPU support")
            print("NOTE: If the 3D visualization appears black or empty, try the following:")
            print("1. Update your graphics drivers to the latest version")
            print("2. Run with --mode=2d flag for the 2D fallback visualization")
            
            return container
            
        except Exception as e:
            logger.error(f"Error creating 3D view: {e}")
            logger.error(f"Exception details: {type(e).__name__} - {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.warning("Falling back to 2D visualization due to error")
            return self.create_fallback_view()
    
    def create_fallback_view(self) -> QWidget:
        """Create a fallback view when Qt3D is not available"""
        from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem
        from PySide6.QtGui import QPen, QBrush
        from PySide6.QtCore import QRectF
        
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(frame)
        
        # Create 2D visualization components
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setMinimumSize(600, 500)
        
        # Add the view to the layout
        layout.addWidget(self.view)
        
        # Create graphics items for all nodes and connections
        if self.mapper:
            self.create_2d_visualization()
        
        return frame
    
    def create_2d_visualization(self):
        """Create a 2D visualization of the nodes and connections"""
        if not self.mapper:
            return
            
        from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem
        from PySide6.QtGui import QPen, QBrush, QColor, QFont
        from PySide6.QtCore import QRectF, QPointF
        
        # Clear the scene
        self.scene.clear()
        
        # Set scene size
        self.scene.setSceneRect(-300, -300, 600, 600)
        
        # Dictionary to store node positions for connection lines
        node_positions = {}
        node_graphics_items = {}
        
        # Create items for each node
        for node_id, node in self.mapper.nodes.items():
            # Get 3D position and map to 2D
            x, y, z = node.position
            # Project from 3D to 2D - using X and Z for horizontal plane
            # Scale positions to fit the view
            pos_x = x * 2
            pos_y = z * 2
            
            # Create node circle
            radius = 4 + (node.weight * 5)  # Size based on weight
            ellipse = NodeGraphicsItem(pos_x - radius, pos_y - radius, radius * 2, radius * 2, node)
            
            # Color based on node_type
            color_map = {
                "concept": QColor(50, 150, 255),  # Blue
                "entity": QColor(50, 255, 150),   # Green
                "process": QColor(255, 150, 50),  # Orange
                "attribute": QColor(200, 100, 200), # Purple
                "action": QColor(255, 100, 100),  # Red
                "event": QColor(255, 255, 100),   # Yellow
                "relation": QColor(100, 200, 255), # Light blue
                "property": QColor(150, 255, 200) # Light green
            }
            color = color_map.get(node.node_type, QColor(150, 150, 150))
            
            # Set up node appearance
            ellipse.setBrush(QBrush(color))
            ellipse.setPen(QPen(color.darker(), 1))
            
            # Add tooltip with node information
            ellipse.setToolTip(f"{node.concept} (Weight: {node.weight:.2f})")
            
            # Make nodes selectable and hoverable
            ellipse.setAcceptHoverEvents(True)
            ellipse.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable, True)
            
            # Connect to the selection signal
            ellipse.node_clicked.connect(self.on_node_clicked)
            
            # Add to scene
            self.scene.addItem(ellipse)
            
            # Store position for connection lines
            node_positions[node_id] = (pos_x, pos_y)
            node_graphics_items[node_id] = ellipse
            
            # Add text label for important nodes (based on weight)
            if node.weight > 1.0:
                text = QGraphicsTextItem(node.concept)
                text.setPos(pos_x + radius + 2, pos_y - 8)
                text.setFont(QFont("Arial", 8))
                self.scene.addItem(text)
        
        # Create connection lines
        processed_connections = set()
        for node_id, node in self.mapper.nodes.items():
            for target_id in node.connections:
                # Create a unique key for this connection to avoid duplicates
                conn_key = tuple(sorted([node_id, target_id]))
                
                if conn_key in processed_connections:
                    continue
                
                processed_connections.add(conn_key)
                
                if target_id in node_positions:
                    x1, y1 = node_positions[node_id]
                    x2, y2 = node_positions[target_id]
                    
                    # Create line for connection
                    line = QGraphicsLineItem(x1, y1, x2, y2)
                    
                    # Set appearance based on connection
                    # For now, use a simple semi-transparent line
                    line_pen = QPen(QColor(100, 100, 100, 100), 0.5)
                    line.setPen(line_pen)
                    
                    # Add to scene (behind nodes)
                    line.setZValue(-1)
                    self.scene.addItem(line)
            
        # Set view to show the whole scene
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        
    def create_temple_entities(self):
        """Create 3D entities for the temple visualization"""
        if not HAS_QT3D or not self.mapper:
            return
        
        # Clear existing entities except base components
        self._clear_temple_entities()
        
        # Create entities for zones
        self._create_zone_entities()
        
        # Create entities for nodes (concepts)
        self._create_node_entities()
        
        # Create entities for connections
        self._create_connection_entities()
    
    def _clear_temple_entities(self):
        """Clear existing temple entities"""
        # Implementation depends on how entities are stored
        # This is a placeholder for the entity cleanup logic
        pass
    
    def _create_zone_entities(self):
        """Create entities for temple zones"""
        if not HAS_QT3D or not self.mapper:
            return
        
        # For each zone in the mapper, create a sphere entity
        for zone in self.mapper.get_all_zones():
            # Create zone entity
            zone_entity = Qt3DCore.QEntity(self.root_entity)
            
            # Create mesh
            sphere_mesh = Qt3DExtras.QSphereMesh()
            sphere_mesh.setRadius(zone.radius)
            sphere_mesh.setRings(32)
            sphere_mesh.setSlices(32)
            
            # Create material
            zone_material = Qt3DExtras.QPhongMaterial()
            
            # Set color based on zone type
            color = self._get_zone_color(zone.zone_type)
            zone_material.setDiffuse(color)
            zone_material.setAmbient(color.darker())
            zone_material.setShininess(0)
            zone_material.setSpecular(QColor(10, 10, 10))
            
            # Make it semi-transparent
            effect = zone_material.effect()
            technique = effect.techniques()[0]
            render_pass = technique.renderPasses()[0]
            render_state = Qt3DRender.QRenderState()
            blend_state = Qt3DRender.QBlendEquation()
            blend_state.setBlendFunction(Qt3DRender.QBlendEquation.Add)
            render_state.addChild(blend_state)
            render_pass.addRenderState(render_state)
            
            # Set transform
            transform = Qt3DCore.QTransform()
            transform.setTranslation(QVector3D(*zone.center))
            
            # Add components to entity
            zone_entity.addComponent(sphere_mesh)
            zone_entity.addComponent(zone_material)
            zone_entity.addComponent(transform)
    
    def _create_node_entities(self):
        """Create entities for concept nodes"""
        if not HAS_QT3D or not self.mapper:
            return
        
        # For each node in the mapper, create a sphere entity
        for node in self.mapper.get_all_nodes():
            # Create node entity
            node_entity = Qt3DCore.QEntity(self.root_entity)
            
            # Create mesh
            sphere_mesh = Qt3DExtras.QSphereMesh()
            sphere_mesh.setRadius(1.0 + node.weight * 1.5)  # Size based on weight
            sphere_mesh.setRings(16)
            sphere_mesh.setSlices(16)
            
            # Create material
            node_material = Qt3DExtras.QPhongMaterial()
            node_material.setDiffuse(QColor(100, 180, 255))
            
            # Set transform
            transform = Qt3DCore.QTransform()
            transform.setTranslation(QVector3D(*node.position))
            
            # Add components to entity
            node_entity.addComponent(sphere_mesh)
            node_entity.addComponent(node_material)
            node_entity.addComponent(transform)
            
            # TODO: Add text label for concept name
    
    def _create_connection_entities(self):
        """Create entities for connections between nodes"""
        if not HAS_QT3D or not self.mapper:
            return
        
        # For each connection in the mapper, create a cylinder entity
        for connection in self.mapper.get_all_connections():
            # Get the connected nodes
            source_node = self.mapper.get_node(connection.source_id)
            target_node = self.mapper.get_node(connection.target_id)
            
            if not source_node or not target_node:
                continue
            
            # Create connection entity
            connection_entity = Qt3DCore.QEntity(self.root_entity)
            
            # Calculate cylinder parameters
            source_pos = QVector3D(*source_node.position)
            target_pos = QVector3D(*target_node.position)
            
            # Calculate midpoint
            midpoint = (source_pos + target_pos) / 2
            
            # Calculate distance between points
            direction = target_pos - source_pos
            distance = direction.length()
            
            # Create mesh
            cylinder = Qt3DExtras.QCylinderMesh()
            cylinder.setRadius(0.3 * connection.strength)  # Thickness based on strength
            cylinder.setLength(distance)
            cylinder.setRings(8)
            cylinder.setSlices(16)
            
            # Create material
            material = Qt3DExtras.QPhongMaterial()
            material.setDiffuse(QColor(200, 200, 200, 100 + int(155 * connection.strength)))
            
            # Calculate rotation to align cylinder with the connection direction
            up = QVector3D(0, 1, 0)
            normal_direction = direction.normalized()
            
            # Use QQuaternion to handle the rotation
            rotation = QQuaternion.rotationTo(up, normal_direction)
            
            # Create transform
            transform = Qt3DCore.QTransform()
            transform.setTranslation(midpoint)
            transform.setRotation(rotation)
            
            # Add components to entity
            connection_entity.addComponent(cylinder)
            connection_entity.addComponent(material)
            connection_entity.addComponent(transform)
    
    def _get_zone_color(self, zone_type: str) -> QColor:
        """Get a color based on zone type"""
        colors = {
            "knowledge": QColor(80, 120, 255, 60),  # Blue, semi-transparent
            "reflection": QColor(180, 80, 200, 60),  # Purple, semi-transparent
            "integration": QColor(80, 200, 120, 60),  # Green, semi-transparent
            "contradiction": QColor(220, 100, 80, 60),  # Red, semi-transparent
            "synthesis": QColor(200, 180, 80, 60),  # Yellow, semi-transparent
            "memory": QColor(80, 180, 200, 60),  # Cyan, semi-transparent
            "consciousness": QColor(220, 180, 255, 60),  # Pink, semi-transparent
            "query": QColor(240, 240, 240, 60),  # White, semi-transparent
            "ritual": QColor(100, 100, 100, 60)  # Gray, semi-transparent
        }
        return colors.get(zone_type, QColor(150, 150, 150, 60))
    
    def move_direction(self, x: float, y: float, z: float):
        """Move in the specified direction"""
        # Update position
        self.current_position += QVector3D(x * 5, y * 5, z * 5)
        
        # Update camera position if using Qt3D
        if HAS_QT3D and hasattr(self, 'camera'):
            self.camera.setPosition(self.current_position)
            self.camera.setViewCenter(self.current_position + self.current_direction * 10)
        # For 2D fallback view, adjust the scene view
        elif hasattr(self, 'scene') and hasattr(self, 'view'):
            # Translate the scene view to simulate movement
            self.view.centerOn(self.current_position.x() * 2, self.current_position.z() * 2)
        
        # Update position label
        px, py, pz = self.current_position.x(), self.current_position.y(), self.current_position.z()
        self.position_label.setText(f"Position: ({px:.1f}, {py:.1f}, {pz:.1f})")
        
        # Check if we've entered a new zone
        self._check_current_zone()
    
    def _check_current_zone(self):
        """Check if the current position is in a zone"""
        if not self.mapper:
            return
        
        # Convert current position to tuple for mapper functions
        position = (self.current_position.x(), self.current_position.y(), self.current_position.z())
        
        # Check each zone
        for zone in self.mapper.get_all_zones():
            if zone.contains_point(position):
                # If we entered a new zone
                if self.current_zone_id != zone.id:
                    self.current_zone_id = zone.id
                    self.zone_label.setText(f"Current Zone: {zone.name}")
                    self.zone_entered.emit(zone.id)
                return
        
        # If not in any zone
        if self.current_zone_id is not None:
            self.current_zone_id = None
            self.zone_label.setText("Current Zone: None")
    
    def select_concept(self, node_id: str):
        """Select a concept node"""
        if not self.mapper:
            return
        
        node = self.mapper.get_node(node_id)
        if not node:
            return
        
        self.selected_node_id = node_id
        self.concept_label.setText(f"Selected: {node.concept}")
        
        # Get additional info about the node
        connections = len(node.connections)
        self.concept_info.setText(f"Weight: {node.weight:.2f}\nConnections: {connections}")
        
        # Emit signal
        self.concept_selected.emit(node.concept)
    
    def update_visualization(self):
        """Update the visualization (called by timer)"""
        # Update 2D visualization if using fallback
        if not HAS_QT3D and hasattr(self, 'scene') and hasattr(self, 'view'):
            # Recreate the visualization to reflect any changes
            self.create_2d_visualization()
    
    def update_from_mapper(self):
        """Update the visualization based on the current mapper state"""
        if HAS_QT3D and self.mapper:
            self.create_temple_entities()
    
    def set_mapper(self, mapper: SpatialTempleMapper):
        """Set a new mapper and update the visualization"""
        self.mapper = mapper
        if HAS_QT3D:
            self.create_temple_entities()

    def on_node_clicked(self, node):
        """Handle a node being clicked in the visualization"""
        # Show information about the node
        self.selected_node_id = node.id
        
        # Emit signal with node concept
        self.concept_selected.emit(node.concept)

class NodeGraphicsItem(QGraphicsEllipseItem):
    """Custom QGraphicsEllipseItem that knows about its associated node"""
    
    # Define signal - need to use a intermediate QObject for this
    class SignalEmitter(QObject):
        node_clicked = Signal(object)  # Signal that passes the node object
        
    def __init__(self, x, y, w, h, node):
        super().__init__(x, y, w, h)
        self.node = node
        self._signal_emitter = self.SignalEmitter()
        self.node_clicked = self._signal_emitter.node_clicked
        self.setAcceptHoverEvents(True)
        
    def mousePressEvent(self, event):
        """Handle mouse press on the node"""
        super().mousePressEvent(event)
        self.node_clicked.emit(self.node)
        
        # Visually indicate selection
        self.setScale(1.2)
        
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        super().mouseReleaseEvent(event)
        self.setScale(1.0)
        
    def hoverEnterEvent(self, event):
        """Handle mouse hover enter"""
        self.setCursor(Qt.PointingHandCursor)
        # Make the node slightly larger on hover
        self.setScale(1.1)
        super().hoverEnterEvent(event)
        
    def hoverLeaveEvent(self, event):
        """Handle mouse hover leave"""
        self.setCursor(Qt.ArrowCursor)
        # Reset scale
        self.setScale(1.0)
        super().hoverLeaveEvent(event)

class SpatialTempleVisualizationWindow(QMainWindow):
    """
    Main window for the Spatial Temple Visualization
    
    This provides a standalone interface for navigating and exploring
    the spatial temple in 3D.
    """
    
    def __init__(self, mapper: Optional[SpatialTempleMapper] = None, mode: str = '3d'):
        """Initialize the window with an optional spatial temple mapper"""
        super().__init__()
        self.mapper = mapper
        self.mode = mode.lower()  # '2d' or '3d'
        
        self.setWindowTitle("Lumina v8 - Spatial Temple")
        self.setMinimumSize(1000, 700)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # Create visualization widget with specified mode
        self.temple_widget = SpatialTempleWidget(mapper, mode=self.mode)
        layout.addWidget(self.temple_widget)
        
        # Status bar
        self.statusBar().showMessage(f"Ready | Spatial Temple Visualization ({self.mode.upper()} mode)")
    
    def set_mapper(self, mapper: SpatialTempleMapper):
        """Set a new mapper and update the visualization"""
        self.mapper = mapper
        self.temple_widget.set_mapper(mapper)

def run_visualization(mapper: Optional[SpatialTempleMapper] = None):
    """Run the spatial temple visualization as a standalone application"""
    from PySide6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create the visualization window
    window = SpatialTempleVisualizationWindow(mapper)
    window.show()
    
    return app.exec()

if __name__ == "__main__":
    # Create a sample mapper for testing
    mapper = SpatialTempleMapper()
    
    # Generate some test data
    text = "The spatial temple provides a three-dimensional navigation experience for concepts, using metaphorical spaces for knowledge organization."
    mapper.map_concepts(text)
    
    # Run the visualization
    sys.exit(run_visualization(mapper)) 