#!/usr/bin/env python3
"""
Memory Visualizer for V7 Node Consciousness.

This module provides visualization components for the Memory Node, displaying
stored memories, their relationships, strengths, and decay patterns in real-time.
"""

import sys
import time
import math
import logging
import random
from typing import Dict, List, Optional, Tuple, Any, Set

# Import Qt components
try:
    from src.v5.qt_compat import QtWidgets, QtCore, QtGui
except ImportError:
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
    except ImportError:
        logging.error("Failed to import Qt components. UI visualization will not be available.")
        QtWidgets = QtCore = QtGui = None

# Configure logging
logger = logging.getLogger(__name__)

class MemoryVisualizer(QtWidgets.QWidget):
    """
    Memory node visualization component for the V7 system.
    
    Displays memory nodes as interactive nodes in a graph, with strength represented
    by opacity and size, relationships as edges, and memory types as colors.
    """
    
    def __init__(self, parent=None, config=None):
        """
        Initialize the Memory Visualizer.
        
        Args:
            parent: Parent widget
            config: Configuration dictionary with visualization settings
        """
        if QtWidgets is None:
            raise ImportError("Qt components are not available. Cannot create Memory Visualizer.")
            
        super().__init__(parent)
        self.setMinimumSize(500, 400)
        
        # Default configuration
        self._config = {
            "max_visible_memories": 50,
            "node_size_range": (5, 30),
            "edge_thickness_range": (1, 5),
            "animation_speed": 1.0,
            "layout_algorithm": "force_directed",
            "color_by_type": True,
            "show_labels": True,
            "decay_animation": True,
            "highlight_active": True,
            "background_color": "#1E1E1E",
            "text_color": "#FFFFFF",
            "grid_color": "#333333",
            "connection_distance": 150,
        }
        
        # Override defaults with provided config
        if config:
            self._config.update(config)
            
        # Memory visualization data
        self._memories = {}  # id -> memory data
        self._memory_positions = {}  # id -> (x, y) position
        self._memory_velocities = {}  # id -> (vx, vy) velocity
        self._memory_targets = {}  # id -> (tx, ty) target position
        self._memory_types = set()  # Set of all memory types
        self._memory_connections = {}  # id -> [connected_id, ...]
        self._highlighted_memories = set()  # Currently highlighted memories
        self._selected_memory = None  # Currently selected memory
        self._zoom_level = 1.0
        self._pan_offset = (0, 0)
        self._hover_memory = None  # Memory being hovered over
        self._simulation_active = False
        self._last_update_time = time.time()
        self._color_map = {}  # memory_type -> color
        
        # UI setup
        self._setup_ui()
        
        # Start visualization updates
        self._update_timer = QtCore.QTimer(self)
        self._update_timer.timeout.connect(self._update_visualization)
        self._update_timer.start(30)  # ~33 fps
        
        # Start simulation
        self._simulation_active = True
        
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        
        # Canvas for memory visualization
        self._canvas = QtWidgets.QWidget(self)
        self._canvas.setStyleSheet(f"background-color: {self._config['background_color']};")
        self._canvas.paintEvent = self._paint_canvas
        self._canvas.mousePressEvent = self._canvas_mouse_press
        self._canvas.mouseMoveEvent = self._canvas_mouse_move
        self._canvas.mouseReleaseEvent = self._canvas_mouse_release
        self._canvas.wheelEvent = self._canvas_wheel
        
        # Controls layout
        controls_layout = QtWidgets.QHBoxLayout()
        
        # Filter by type combobox
        self._type_filter = QtWidgets.QComboBox(self)
        self._type_filter.addItem("All Types")
        self._type_filter.currentIndexChanged.connect(self._filter_changed)
        
        # Search input
        self._search_input = QtWidgets.QLineEdit(self)
        self._search_input.setPlaceholderText("Search memories...")
        self._search_input.textChanged.connect(self._search_changed)
        
        # Strength threshold slider
        strength_layout = QtWidgets.QHBoxLayout()
        strength_layout.addWidget(QtWidgets.QLabel("Min Strength:"))
        self._strength_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self._strength_slider.setRange(0, 100)
        self._strength_slider.setValue(20)  # 0.2 default
        self._strength_slider.valueChanged.connect(self._filter_changed)
        strength_layout.addWidget(self._strength_slider)
        
        # Add controls to layout
        controls_layout.addWidget(QtWidgets.QLabel("Type:"))
        controls_layout.addWidget(self._type_filter)
        controls_layout.addWidget(self._search_input)
        controls_layout.addLayout(strength_layout)
        
        # Add items to main layout
        layout.addWidget(self._canvas)
        layout.addLayout(controls_layout)
        
        # Detail panel for selected memory
        self._detail_panel = QtWidgets.QTextEdit(self)
        self._detail_panel.setReadOnly(True)
        self._detail_panel.setMaximumHeight(150)
        self._detail_panel.setStyleSheet("background-color: #2D2D2D; color: #FFFFFF;")
        layout.addWidget(self._detail_panel)
        
        # Set main layout
        self.setLayout(layout)
        
    def set_memories(self, memories: List[Dict]):
        """
        Update the visualization with new memory data.
        
        Args:
            memories: List of memory dictionaries from the Memory Node
        """
        # Update memory types list
        new_types = {memory["memory_type"] for memory in memories}
        if new_types != self._memory_types:
            self._memory_types = new_types
            current_text = self._type_filter.currentText()
            self._type_filter.clear()
            self._type_filter.addItem("All Types")
            for memory_type in sorted(self._memory_types):
                self._type_filter.addItem(memory_type)
            
            # Restore previous selection or default to "All Types"
            index = self._type_filter.findText(current_text)
            if index >= 0:
                self._type_filter.setCurrentIndex(index)
            else:
                self._type_filter.setCurrentIndex(0)
        
        # Assign colors to memory types if needed
        for memory_type in self._memory_types:
            if memory_type not in self._color_map:
                hue = hash(memory_type) % 360
                self._color_map[memory_type] = QtGui.QColor.fromHsv(hue, 200, 240)
        
        # Update memory data
        old_ids = set(self._memories.keys())
        new_ids = {memory["id"] for memory in memories}
        
        # Remove memories that no longer exist
        for memory_id in old_ids - new_ids:
            self._memories.pop(memory_id, None)
            self._memory_positions.pop(memory_id, None)
            self._memory_velocities.pop(memory_id, None)
            self._memory_targets.pop(memory_id, None)
            self._memory_connections.pop(memory_id, None)
            
            if memory_id in self._highlighted_memories:
                self._highlighted_memories.remove(memory_id)
                
            if self._selected_memory == memory_id:
                self._selected_memory = None
                self._update_detail_panel(None)
        
        # Update existing memories and add new ones
        for memory in memories:
            memory_id = memory["id"]
            
            # New memory - assign initial position
            if memory_id not in self._memories:
                # Place new memories at random positions
                width, height = self.width(), self.height()
                x = random.uniform(width * 0.1, width * 0.9)
                y = random.uniform(height * 0.1, height * 0.9)
                self._memory_positions[memory_id] = (x, y)
                self._memory_velocities[memory_id] = (0, 0)
                self._memory_targets[memory_id] = (x, y)
                self._memory_connections[memory_id] = set()
            
            # Update memory data
            self._memories[memory_id] = memory
            
        # Update connections between memories
        # Here's a simple implementation: connect memories with similar tags or related content
        for memory_id, memory in self._memories.items():
            connections = set()
            memory_tags = set(memory.get("tags", []))
            content_words = set(memory.get("content", "").lower().split())
            
            # Find connections to other memories
            for other_id, other_memory in self._memories.items():
                if memory_id == other_id:
                    continue
                    
                # Check for tag overlap
                other_tags = set(other_memory.get("tags", []))
                if memory_tags & other_tags:  # Intersection of tags
                    connections.add(other_id)
                    continue
                    
                # Check for content similarity
                other_content_words = set(other_memory.get("content", "").lower().split())
                common_words = content_words & other_content_words
                if len(common_words) >= 2:  # At least 2 common words
                    connections.add(other_id)
                    
                # Check for related metadata
                if memory.get("memory_type") == other_memory.get("memory_type"):
                    relation_score = 0
                    # Check metadata for relationships
                    if "related_to" in memory.get("metadata", {}) and memory["metadata"]["related_to"] == other_id:
                        relation_score += 1
                    if "related_to" in other_memory.get("metadata", {}) and other_memory["metadata"]["related_to"] == memory_id:
                        relation_score += 1
                    
                    if relation_score > 0:
                        connections.add(other_id)
            
            self._memory_connections[memory_id] = connections
        
        # Update the visualization
        self._canvas.update()
        
    def highlight_memories(self, memory_ids: List[str]):
        """Highlight specific memories in the visualization."""
        self._highlighted_memories = set(memory_ids)
        self._canvas.update()
        
    def select_memory(self, memory_id: Optional[str]):
        """Select a specific memory to show details."""
        self._selected_memory = memory_id
        self._update_detail_panel(self._memories.get(memory_id) if memory_id else None)
        self._canvas.update()
        
    def _update_detail_panel(self, memory: Optional[Dict]):
        """Update the detail panel with information about the selected memory."""
        if memory is None:
            self._detail_panel.setText("No memory selected")
            return
            
        # Format memory details as HTML
        html = f"""
        <h3>{memory.get('content', 'No content')}</h3>
        <p><b>Type:</b> {memory.get('memory_type', 'Unknown')}</p>
        <p><b>Strength:</b> {memory.get('strength', 0):.2f}</p>
        <p><b>Tags:</b> {', '.join(memory.get('tags', []))}</p>
        <p><b>Created:</b> {memory.get('created_at', 'Unknown')}</p>
        <p><b>Last accessed:</b> {memory.get('last_accessed', 'Unknown')}</p>
        """
        
        # Add metadata if available
        if metadata := memory.get('metadata', {}):
            html += "<p><b>Metadata:</b></p><ul>"
            for key, value in metadata.items():
                html += f"<li>{key}: {value}</li>"
            html += "</ul>"
            
        self._detail_panel.setHtml(html)
        
    def _filter_changed(self):
        """Handle filter changes and update the visualization."""
        self._canvas.update()
        
    def _search_changed(self):
        """Handle search text changes and update the visualization."""
        search_text = self._search_input.text().lower()
        
        if not search_text:
            # Clear highlights when search is empty
            self._highlighted_memories = set()
        else:
            # Highlight memories matching the search
            self._highlighted_memories = set()
            for memory_id, memory in self._memories.items():
                content = memory.get("content", "").lower()
                tags = " ".join(memory.get("tags", [])).lower()
                
                if search_text in content or search_text in tags:
                    self._highlighted_memories.add(memory_id)
                    
        self._canvas.update()
        
    def _get_filtered_memories(self) -> Dict[str, Dict]:
        """Get memories filtered by current filter settings."""
        filtered = {}
        
        # Get filter values
        filter_type = self._type_filter.currentText()
        min_strength = self._strength_slider.value() / 100.0
        
        for memory_id, memory in self._memories.items():
            # Filter by type
            if filter_type != "All Types" and memory.get("memory_type") != filter_type:
                continue
                
            # Filter by strength
            if memory.get("strength", 0) < min_strength:
                continue
                
            filtered[memory_id] = memory
            
        return filtered
        
    def _update_visualization(self):
        """Update memory positions based on force-directed layout."""
        if not self._simulation_active or not self._memories:
            return
            
        current_time = time.time()
        elapsed = current_time - self._last_update_time
        self._last_update_time = current_time
        
        # Limit the elapsed time to avoid large jumps
        elapsed = min(elapsed, 0.1)
        
        # Apply force-directed layout for visible memories
        filtered_memories = self._get_filtered_memories()
        memory_ids = list(filtered_memories.keys())
        
        # Only process the top N memories by strength for performance
        if len(memory_ids) > self._config["max_visible_memories"]:
            memory_ids.sort(
                key=lambda mid: filtered_memories[mid].get("strength", 0), 
                reverse=True
            )
            memory_ids = memory_ids[:self._config["max_visible_memories"]]
        
        # Calculate forces and update positions
        for i, memory_id in enumerate(memory_ids):
            if memory_id not in self._memory_positions:
                continue
                
            # Get current position
            x, y = self._memory_positions[memory_id]
            vx, vy = self._memory_velocities[memory_id]
            
            # Reset forces
            fx, fy = 0, 0
            
            # Repulsive forces from other memories
            for other_id in memory_ids:
                if memory_id == other_id:
                    continue
                    
                if other_id not in self._memory_positions:
                    continue
                    
                ox, oy = self._memory_positions[other_id]
                dx, dy = x - ox, y - oy
                distance = math.sqrt(dx*dx + dy*dy) + 0.1  # Avoid division by zero
                
                # Stronger repulsion for closer memories
                if distance < 50:
                    repulsion_force = 1.0 / (distance * distance) * 50
                    fx += dx * repulsion_force
                    fy += dy * repulsion_force
            
            # Attractive forces for connected memories
            for other_id in self._memory_connections.get(memory_id, set()):
                if other_id not in memory_ids or other_id not in self._memory_positions:
                    continue
                    
                ox, oy = self._memory_positions[other_id]
                dx, dy = ox - x, oy - y
                distance = math.sqrt(dx*dx + dy*dy) + 0.1  # Avoid division by zero
                
                # Only apply attraction if distance is too large
                if distance > self._config["connection_distance"]:
                    strength = filtered_memories[memory_id].get("strength", 0.5)
                    attraction_force = (distance - self._config["connection_distance"]) * 0.005 * strength
                    fx += dx * attraction_force
                    fy += dy * attraction_force
            
            # Boundary forces to keep memories in view
            width, height = self.width(), self.height()
            margin = 50
            
            if x < margin:
                fx += (margin - x) * 0.1
            elif x > width - margin:
                fx += (width - margin - x) * 0.1
                
            if y < margin:
                fy += (margin - y) * 0.1
            elif y > height - margin:
                fy += (height - margin - y) * 0.1
            
            # Update velocity with forces
            vx = vx * 0.9 + fx * elapsed * 10  # Damping factor
            vy = vy * 0.9 + fy * elapsed * 10
            
            # Limit velocity
            speed = math.sqrt(vx*vx + vy*vy)
            if speed > 100:
                vx = vx / speed * 100
                vy = vy / speed * 100
                
            # Update position
            x += vx * elapsed
            y += vy * elapsed
            
            # Store updated values
            self._memory_positions[memory_id] = (x, y)
            self._memory_velocities[memory_id] = (vx, vy)
        
        # Request a repaint
        self._canvas.update()
        
    def _paint_canvas(self, event):
        """Paint the memory visualization on the canvas."""
        if not self._memories:
            return
            
        # Set up painter
        painter = QtGui.QPainter(self._canvas)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Apply zoom and pan transformations
        painter.translate(self._pan_offset[0], self._pan_offset[1])
        painter.scale(self._zoom_level, self._zoom_level)
        
        # Draw background grid
        self._draw_grid(painter)
        
        # Get filtered memories
        filtered_memories = self._get_filtered_memories()
        
        # Draw connections first (so they're behind the nodes)
        self._draw_connections(painter, filtered_memories)
        
        # Draw memory nodes
        self._draw_memory_nodes(painter, filtered_memories)
        
        # End painting
        painter.end()
        
    def _draw_grid(self, painter):
        """Draw a grid in the background."""
        width, height = self.width(), self.height()
        grid_size = 50
        grid_pen = QtGui.QPen(QtGui.QColor(self._config["grid_color"]))
        grid_pen.setWidth(1)
        grid_pen.setStyle(QtCore.Qt.DotLine)
        painter.setPen(grid_pen)
        
        # Draw horizontal grid lines
        for y in range(0, height, grid_size):
            painter.drawLine(0, y, width, y)
            
        # Draw vertical grid lines
        for x in range(0, width, grid_size):
            painter.drawLine(x, 0, x, height)
            
    def _draw_connections(self, painter, filtered_memories):
        """Draw connections between memory nodes."""
        # Set the pen for connections
        connection_pen = QtGui.QPen(QtGui.QColor(100, 100, 100, 100))
        connection_pen.setWidth(1)
        painter.setPen(connection_pen)
        
        # Draw lines between connected memories
        for memory_id, connections in self._memory_connections.items():
            if memory_id not in filtered_memories or memory_id not in self._memory_positions:
                continue
                
            x1, y1 = self._memory_positions[memory_id]
            
            for other_id in connections:
                if other_id not in filtered_memories or other_id not in self._memory_positions:
                    continue
                    
                x2, y2 = self._memory_positions[other_id]
                
                # Calculate connection strength
                memory_strength = filtered_memories[memory_id].get("strength", 0.5)
                other_strength = filtered_memories[other_id].get("strength", 0.5)
                conn_strength = (memory_strength + other_strength) / 2.0
                
                # Adjust line thickness based on connection strength
                min_width, max_width = self._config["edge_thickness_range"]
                width = min_width + conn_strength * (max_width - min_width)
                connection_pen.setWidth(int(width))
                
                # Adjust opacity based on connection strength
                alpha = int(100 + conn_strength * 155)  # 100 to 255
                
                # Highlight connections for selected or highlighted memories
                if (self._selected_memory in (memory_id, other_id) or 
                    memory_id in self._highlighted_memories or 
                    other_id in self._highlighted_memories):
                    # Use a distinct color for highlighted connections
                    connection_pen.setColor(QtGui.QColor(200, 200, 255, alpha))
                else:
                    connection_pen.setColor(QtGui.QColor(100, 100, 100, alpha))
                    
                painter.setPen(connection_pen)
                painter.drawLine(x1, y1, x2, y2)
                
    def _draw_memory_nodes(self, painter, filtered_memories):
        """Draw memory nodes as circles with labels."""
        # First determine draw order based on strength
        memory_ids = list(filtered_memories.keys())
        memory_ids.sort(key=lambda mid: filtered_memories[mid].get("strength", 0))
        
        for memory_id in memory_ids:
            if memory_id not in self._memory_positions:
                continue
                
            memory = filtered_memories[memory_id]
            x, y = self._memory_positions[memory_id]
            
            # Determine node size based on memory strength
            strength = memory.get("strength", 0.5)
            min_size, max_size = self._config["node_size_range"]
            size = min_size + strength * (max_size - min_size)
            
            # Determine node color based on memory type
            if self._config["color_by_type"] and memory.get("memory_type") in self._color_map:
                color = self._color_map[memory.get("memory_type")]
            else:
                # Default color by strength (green to red)
                hue = int(120 * (1.0 - strength))  # 120 (green) to 0 (red)
                color = QtGui.QColor.fromHsv(hue, 240, 200)
                
            # Adjust opacity based on strength
            alpha = int(100 + strength * 155)  # 100 to 255
            color.setAlpha(alpha)
            
            # Draw highlight for selected or highlighted memory
            if (memory_id == self._selected_memory or
                memory_id in self._highlighted_memories or
                memory_id == self._hover_memory):
                # Draw highlight circle
                highlight_color = QtGui.QColor(255, 255, 200, 150)
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(highlight_color)
                painter.drawEllipse(QtCore.QPointF(x, y), size + 5, size + 5)
                
            # Draw memory node
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(QtCore.QPointF(x, y), size, size)
            
            # Draw node border
            border_pen = QtGui.QPen(QtGui.QColor(0, 0, 0, alpha))
            border_pen.setWidth(1)
            painter.setPen(border_pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(QtCore.QPointF(x, y), size, size)
            
            # Draw labels if enabled
            if self._config["show_labels"]:
                # Only show labels for sufficiently large nodes or highlighted ones
                if (size >= 15 or 
                    memory_id == self._selected_memory or 
                    memory_id in self._highlighted_memories or 
                    memory_id == self._hover_memory):
                    
                    # Set text color and font
                    painter.setPen(QtGui.QColor(self._config["text_color"]))
                    font = painter.font()
                    font.setPointSize(8)
                    painter.setFont(font)
                    
                    # Get label text - use first few words of content
                    content = memory.get("content", "")
                    words = content.split()
                    if len(words) > 3:
                        label = " ".join(words[:3]) + "..."
                    else:
                        label = content
                        
                    # Draw text below the node
                    text_rect = QtCore.QRectF(x - 100, y + size + 5, 200, 30)
                    painter.drawText(text_rect, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop, label)
            
    def _canvas_mouse_press(self, event):
        """Handle mouse press events on the canvas."""
        pos = event.position() if hasattr(event, 'position') else event.pos()
        x, y = pos.x(), pos.y()
        
        # Apply inverse zoom and pan to get the transformed coordinates
        x = (x - self._pan_offset[0]) / self._zoom_level
        y = (y - self._pan_offset[1]) / self._zoom_level
        
        # Check if a memory node was clicked
        clicked_memory = None
        min_distance = float('inf')
        
        filtered_memories = self._get_filtered_memories()
        for memory_id, memory_pos in self._memory_positions.items():
            if memory_id not in filtered_memories:
                continue
                
            memory = filtered_memories[memory_id]
            strength = memory.get("strength", 0.5)
            min_size, max_size = self._config["node_size_range"]
            size = min_size + strength * (max_size - min_size)
            
            # Calculate distance to node center
            node_x, node_y = memory_pos
            distance = math.sqrt((x - node_x)**2 + (y - node_y)**2)
            
            # Check if click is within node radius
            if distance <= size and distance < min_distance:
                clicked_memory = memory_id
                min_distance = distance
        
        if clicked_memory is not None:
            # If a node was clicked, select it
            self.select_memory(clicked_memory)
        else:
            # If background was clicked, deselect current node
            self.select_memory(None)
            
            # Store start position for panning
            self._pan_start_pos = (x, y)
            self._pan_start_offset = self._pan_offset
            
    def _canvas_mouse_move(self, event):
        """Handle mouse move events on the canvas."""
        pos = event.position() if hasattr(event, 'position') else event.pos()
        x, y = pos.x(), pos.y()
        
        # Apply inverse zoom and pan to get the transformed coordinates
        tx = (x - self._pan_offset[0]) / self._zoom_level
        ty = (y - self._pan_offset[1]) / self._zoom_level
        
        # Handle panning if right button is pressed
        if hasattr(event, 'buttons') and event.buttons() & QtCore.Qt.RightButton:
            # Calculate new pan offset
            dx = x - self._pan_start_pos[0]
            dy = y - self._pan_start_pos[1]
            self._pan_offset = (self._pan_start_offset[0] + dx, self._pan_start_offset[1] + dy)
            self._canvas.update()
            return
        
        # Check if hovering over a memory node
        hover_memory = None
        min_distance = float('inf')
        
        filtered_memories = self._get_filtered_memories()
        for memory_id, memory_pos in self._memory_positions.items():
            if memory_id not in filtered_memories:
                continue
                
            memory = filtered_memories[memory_id]
            strength = memory.get("strength", 0.5)
            min_size, max_size = self._config["node_size_range"]
            size = min_size + strength * (max_size - min_size)
            
            # Calculate distance to node center
            node_x, node_y = memory_pos
            distance = math.sqrt((tx - node_x)**2 + (ty - node_y)**2)
            
            # Check if cursor is within node radius
            if distance <= size and distance < min_distance:
                hover_memory = memory_id
                min_distance = distance
        
        # Update hover state if changed
        if hover_memory != self._hover_memory:
            self._hover_memory = hover_memory
            self._canvas.update()
            
            # Update cursor
            if hover_memory is not None:
                self._canvas.setCursor(QtCore.Qt.PointingHandCursor)
            else:
                self._canvas.setCursor(QtCore.Qt.ArrowCursor)
                
    def _canvas_mouse_release(self, event):
        """Handle mouse release events on the canvas."""
        # Reset panning state
        self._pan_start_pos = None
        self._pan_start_offset = None
        
    def _canvas_wheel(self, event):
        """Handle mouse wheel events for zooming."""
        delta = event.angleDelta().y() if hasattr(event, 'angleDelta') else event.delta()
        
        # Determine zoom direction and factor
        zoom_factor = 1.1 if delta > 0 else 0.9
        
        # Get mouse position
        pos = event.position() if hasattr(event, 'position') else event.pos()
        x, y = pos.x(), pos.y()
        
        # Calculate new zoom level
        new_zoom = self._zoom_level * zoom_factor
        
        # Limit zoom range
        if 0.2 <= new_zoom <= 5.0:
            # Calculate adjustment to keep point under mouse constant
            dx = x - self._pan_offset[0]
            dy = y - self._pan_offset[1]
            
            # Apply new zoom
            self._zoom_level = new_zoom
            
            # Adjust pan offset to maintain mouse position
            new_dx = dx * zoom_factor
            new_dy = dy * zoom_factor
            self._pan_offset = (x - new_dx, y - new_dy)
            
            # Update canvas
            self._canvas.update()
            
    def reset_view(self):
        """Reset zoom and pan to default values."""
        self._zoom_level = 1.0
        self._pan_offset = (0, 0)
        self._canvas.update()
        
    def set_simulation_active(self, active: bool):
        """Enable or disable the force-directed simulation."""
        self._simulation_active = active
        if active:
            self._last_update_time = time.time()
        
    def keyPressEvent(self, event):
        """Handle key press events."""
        # Reset view on 'R' key
        if event.key() == QtCore.Qt.Key_R:
            self.reset_view()
        elif event.key() == QtCore.Qt.Key_Space:
            # Toggle simulation on space bar
            self.set_simulation_active(not self._simulation_active)
        else:
            super().keyPressEvent(event)


class MemoryVisualizerPlugin:
    """
    Plugin for integrating the Memory Visualizer with the V7 system.
    
    This class handles connecting to the memory node, retrieving memory data,
    and updating the visualization in real-time.
    """
    
    def __init__(self, v7_connector=None, memory_node=None, config=None):
        """
        Initialize the Memory Visualizer Plugin.
        
        Args:
            v7_connector: V7 backend connector for communication
            memory_node: Direct reference to the Memory Node (optional)
            config: Configuration dictionary
        """
        self._v7_connector = v7_connector
        self._memory_node = memory_node
        self._visualizer = None
        self._config = config or {}
        self._update_interval = self._config.get("update_interval", 5.0)  # seconds
        self._memory_limit = self._config.get("memory_limit", 200)  # max memories to retrieve
        self._last_update = 0
        self._next_update_timer = None
        
        logger.info("Memory Visualizer Plugin initialized")
        
    def get_widget(self):
        """Get the memory visualizer widget."""
        if QtWidgets is None:
            logger.error("Qt components not available. Cannot create visualization widget.")
            return None
            
        if self._visualizer is None:
            try:
                self._visualizer = MemoryVisualizer(config=self._config)
                # Schedule first update
                self._schedule_next_update(0.5)  # First update after 0.5s
            except Exception as e:
                logger.error(f"Failed to create Memory Visualizer: {e}")
                return None
                
        return self._visualizer
        
    def _schedule_next_update(self, delay=None):
        """Schedule the next memory data update."""
        if QtCore is None or self._visualizer is None:
            return
            
        if self._next_update_timer is not None:
            self._next_update_timer.stop()
            
        delay = delay or self._update_interval
        self._next_update_timer = QtCore.QTimer()
        self._next_update_timer.setSingleShot(True)
        self._next_update_timer.timeout.connect(self._update_memory_data)
        self._next_update_timer.start(int(delay * 1000))
        
    def _update_memory_data(self):
        """Retrieve and update memory data from the memory node."""
        if self._visualizer is None:
            return
            
        try:
            memories = []
            
            # Try to get memories from direct reference first
            if self._memory_node is not None and hasattr(self._memory_node, "list_memories"):
                try:
                    memories = self._memory_node.list_memories(limit=self._memory_limit)
                    logger.debug(f"Retrieved {len(memories)} memories from memory node")
                except Exception as e:
                    logger.warning(f"Failed to retrieve memories from memory node: {e}")
            
            # If direct access failed or no memories found, try via connector
            if not memories and self._v7_connector is not None:
                try:
                    # Check if connector has memory_node attribute
                    if hasattr(self._v7_connector, "memory_node") and self._v7_connector.memory_node is not None:
                        memories = self._v7_connector.memory_node.list_memories(limit=self._memory_limit)
                        logger.debug(f"Retrieved {len(memories)} memories via connector")
                    # Or try an event-based approach
                    else:
                        event_data = {
                            "action": "get_memories",
                            "limit": self._memory_limit
                        }
                        result = self._v7_connector.send_event("memory_request", event_data)
                        if isinstance(result, dict) and "memories" in result:
                            memories = result["memories"]
                            logger.debug(f"Retrieved {len(memories)} memories via event")
                except Exception as e:
                    logger.warning(f"Failed to retrieve memories via connector: {e}")
            
            # Update the visualizer with the memory data
            if memories:
                self._visualizer.set_memories(memories)
                
            # Schedule next update
            self._schedule_next_update()
            
        except Exception as e:
            logger.error(f"Error updating memory data: {e}")
            # Try again after a delay
            self._schedule_next_update(max(1.0, self._update_interval / 2))


def get_memory_visualizer(v7_connector=None, memory_node=None, config=None):
    """
    Factory function to create and return a memory visualizer.
    
    Args:
        v7_connector: V7 backend connector
        memory_node: Direct reference to memory node (optional)
        config: Configuration dictionary
        
    Returns:
        Memory visualizer widget or None if unavailable
    """
    try:
        plugin = MemoryVisualizerPlugin(
            v7_connector=v7_connector,
            memory_node=memory_node,
            config=config
        )
        return plugin.get_widget()
    except Exception as e:
        logger.error(f"Failed to create memory visualizer: {e}")
        return None 