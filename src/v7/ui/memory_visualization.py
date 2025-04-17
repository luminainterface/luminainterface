#!/usr/bin/env python3
"""
Memory Visualization Component for V7 Node Consciousness.

This module provides visualization for the Memory Node's state, including:
- Memory strength visualization
- Memory type distribution
- Memory access patterns
- Memory decay visualization
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Any, Union
import time

try:
    from PySide6.QtCore import Qt, QTimer, Signal, QSize, QRectF, Property, QPointF
    from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QPainterPath, QLinearGradient
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
        QGraphicsView, QGraphicsScene, QGraphicsItem,
        QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsTextItem,
        QComboBox, QSlider, QCheckBox
    )
    QT_AVAILABLE = True
except ImportError:
    logging.warning("PySide6 not available, memory visualization will be limited")
    QT_AVAILABLE = False

class MemoryNode:
    """Represents a memory node for visualization purposes."""
    
    def __init__(self, memory_id: str, content: str, memory_type: str, 
                 strength: float, created_at: float, last_accessed: float,
                 tags: List[str], metadata: Dict[str, Any]):
        self.memory_id = memory_id
        self.content = content
        self.memory_type = memory_type
        self.strength = strength
        self.created_at = created_at
        self.last_accessed = last_accessed
        self.tags = tags
        self.metadata = metadata
        self.position = (0, 0)
        self.velocity = (0, 0)
        self.size = max(10, min(30, int(strength * 30)))
        
        # Map memory types to colors
        self.type_colors = {
            'fact': QColor(52, 152, 219),      # Blue
            'experience': QColor(155, 89, 182), # Purple
            'relation': QColor(46, 204, 113),   # Green
            'procedure': QColor(241, 196, 15),  # Yellow
            'default': QColor(149, 165, 166)    # Gray
        }
        
    def get_color(self) -> QColor:
        """Get the color based on memory type with opacity based on strength."""
        color = self.type_colors.get(self.memory_type, self.type_colors['default'])
        return QColor(color.red(), color.green(), color.blue(), 
                      int(max(50, min(255, self.strength * 255))))
        
    def get_border_color(self) -> QColor:
        """Get a darker border color based on the memory type."""
        color = self.get_color()
        return QColor(max(0, color.red() - 30), 
                     max(0, color.green() - 30),
                     max(0, color.blue() - 30),
                     255)
    
    def is_recently_accessed(self, current_time: float, threshold: float = 10.0) -> bool:
        """Check if the memory was accessed recently."""
        return (current_time - self.last_accessed) < threshold


class MemoryGraphItem(QGraphicsEllipseItem):
    """Graphics item representing a memory node in the visualization."""
    
    def __init__(self, memory: MemoryNode):
        super().__init__(0, 0, memory.size, memory.size)
        self.memory = memory
        self.setPos(memory.position[0], memory.position[1])
        self.setBrush(QBrush(memory.get_color()))
        self.setPen(QPen(memory.get_border_color(), 2))
        self.setToolTip(self._create_tooltip())
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        
    def _create_tooltip(self) -> str:
        """Create a rich tooltip with memory details."""
        tooltip = f"<b>{self.memory.content[:50]}{'...' if len(self.memory.content) > 50 else ''}</b><br>"
        tooltip += f"Type: <b>{self.memory.memory_type}</b><br>"
        tooltip += f"Strength: <b>{self.memory.strength:.2f}</b><br>"
        tooltip += f"Tags: <b>{', '.join(self.memory.tags)}</b><br>"
        tooltip += f"Created: <b>{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.memory.created_at))}</b><br>"
        tooltip += f"Last Accessed: <b>{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.memory.last_accessed))}</b>"
        return tooltip
        
    def update_appearance(self, current_time: float):
        """Update the appearance based on current state."""
        self.memory.size = max(10, min(30, int(self.memory.strength * 30)))
        self.setRect(0, 0, self.memory.size, self.memory.size)
        
        # Highlight recently accessed memories with a glow effect
        if self.memory.is_recently_accessed(current_time):
            glow_color = QColor(255, 255, 255, 100)
            self.setPen(QPen(glow_color, 3))
        else:
            self.setPen(QPen(self.memory.get_border_color(), 2))
            
        self.setBrush(QBrush(self.memory.get_color()))
        self.setPos(self.memory.position[0], self.memory.position[1])


class MemoryGraphView(QGraphicsView):
    """Custom graphics view for memory visualization."""
    
    memory_selected = Signal(str)  # Signal emitted when a memory is selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.TextAntialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        
        # Setup scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Memory items
        self.memory_items = {}
        
        # Physics simulation parameters
        self.repulsion = 5000
        self.attraction = 0.05
        self.damping = 0.8
        self.time_step = 0.1
        
        # Setup animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._update_physics)
        self.animation_timer.start(50)  # Update at 20 FPS
        
    def add_memory(self, memory: MemoryNode):
        """Add a memory node to the visualization."""
        if memory.memory_id in self.memory_items:
            # Update existing memory
            self.memory_items[memory.memory_id].memory = memory
            self.memory_items[memory.memory_id].update_appearance(time.time())
        else:
            # Place new memories in a circle formation
            angle = len(self.memory_items) * (2 * math.pi / max(1, len(self.memory_items)))
            radius = 150
            memory.position = (
                self.scene.width() / 2 + radius * math.cos(angle),
                self.scene.height() / 2 + radius * math.sin(angle)
            )
            memory_item = MemoryGraphItem(memory)
            self.memory_items[memory.memory_id] = memory_item
            self.scene.addItem(memory_item)
    
    def remove_memory(self, memory_id: str):
        """Remove a memory node from the visualization."""
        if memory_id in self.memory_items:
            self.scene.removeItem(self.memory_items[memory_id])
            del self.memory_items[memory_id]
    
    def update_memories(self, memories: List[MemoryNode]):
        """Update the visualization with a new list of memories."""
        # Remove memories that are no longer present
        current_ids = {memory.memory_id for memory in memories}
        for memory_id in list(self.memory_items.keys()):
            if memory_id not in current_ids:
                self.remove_memory(memory_id)
        
        # Add or update memories
        for memory in memories:
            self.add_memory(memory)
    
    def _update_physics(self):
        """Update the physics simulation for memory node positions."""
        if not self.memory_items:
            return
            
        current_time = time.time()
        memory_list = list(self.memory_items.values())
        
        # Calculate forces between nodes
        for i, item1 in enumerate(memory_list):
            memory1 = item1.memory
            fx, fy = 0, 0
            
            # Repulsive forces between nodes
            for j, item2 in enumerate(memory_list):
                if i == j:
                    continue
                    
                memory2 = item2.memory
                dx = memory1.position[0] - memory2.position[0]
                dy = memory1.position[1] - memory2.position[1]
                distance = max(1, math.sqrt(dx*dx + dy*dy))
                
                # Repulsive force
                force = self.repulsion / (distance * distance)
                fx += force * dx / distance
                fy += force * dy / distance
            
            # Attractive force toward center
            dx = self.scene.width() / 2 - memory1.position[0]
            dy = self.scene.height() / 2 - memory1.position[1]
            distance = max(1, math.sqrt(dx*dx + dy*dy))
            
            fx += self.attraction * dx
            fy += self.attraction * dy
            
            # Update velocity with damping
            memory1.velocity = (
                memory1.velocity[0] * self.damping + fx * self.time_step,
                memory1.velocity[1] * self.damping + fy * self.time_step
            )
            
            # Update position
            memory1.position = (
                memory1.position[0] + memory1.velocity[0] * self.time_step,
                memory1.position[1] + memory1.velocity[1] * self.time_step
            )
            
            # Ensure memory stays within bounds
            padding = 50
            min_x, min_y = padding, padding
            max_x = max(self.scene.width() - padding, min_x + 1)
            max_y = max(self.scene.height() - padding, min_y + 1)
            
            memory1.position = (
                max(min_x, min(max_x, memory1.position[0])),
                max(min_y, min(max_y, memory1.position[1]))
            )
            
            # Update the item appearance
            item1.update_appearance(current_time)
    
    def resizeEvent(self, event):
        """Handle resize events to adjust the scene size."""
        super().resizeEvent(event)
        self.scene.setSceneRect(0, 0, self.width(), self.height())
        
    def mousePressEvent(self, event):
        """Handle mouse press events for memory selection."""
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.pos())
            if isinstance(item, MemoryGraphItem):
                self.memory_selected.emit(item.memory.memory_id)


class MemoryTypeDistribution(QWidget):
    """Widget showing distribution of memories by type."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.type_counts = {'fact': 0, 'experience': 0, 'relation': 0, 'procedure': 0}
        self.setMinimumHeight(100)
        
        # Colors for different memory types
        self.type_colors = {
            'fact': QColor(52, 152, 219),      # Blue
            'experience': QColor(155, 89, 182), # Purple
            'relation': QColor(46, 204, 113),   # Green
            'procedure': QColor(241, 196, 15),  # Yellow
        }
    
    def update_distribution(self, memories: List[MemoryNode]):
        """Update the distribution data based on memories."""
        # Reset counts
        self.type_counts = {'fact': 0, 'experience': 0, 'relation': 0, 'procedure': 0}
        
        # Count memories by type
        for memory in memories:
            if memory.memory_type in self.type_counts:
                self.type_counts[memory.memory_type] += 1
        
        self.update()
    
    def paintEvent(self, event):
        """Paint the bar chart."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        
        # Calculate total count and bar width
        total_count = sum(self.type_counts.values())
        if total_count == 0:
            return
            
        bar_width = width / len(self.type_counts)
        max_count = max(self.type_counts.values())
        
        # Draw bars
        x = 0
        for memory_type, count in self.type_counts.items():
            # Calculate bar height proportional to count
            bar_height = (count / max_count) * (height - 40) if max_count > 0 else 0
            
            # Draw bar
            color = self.type_colors.get(memory_type, QColor(149, 165, 166))
            painter.fillRect(x, height - bar_height - 20, bar_width - 4, bar_height, color)
            
            # Draw label
            painter.drawText(x, height - 2, bar_width, 20, 
                            Qt.AlignCenter, f"{memory_type}\n({count})")
            
            x += bar_width


class MemoryStrengthHistogram(QWidget):
    """Widget showing histogram of memory strengths."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.strength_bins = [0] * 10  # 10 bins for strengths 0.0-1.0
        self.setMinimumHeight(100)
    
    def update_histogram(self, memories: List[MemoryNode]):
        """Update the histogram data based on memories."""
        # Reset bins
        self.strength_bins = [0] * 10
        
        # Count memories by strength bin
        for memory in memories:
            bin_index = min(9, int(memory.strength * 10))
            self.strength_bins[bin_index] += 1
        
        self.update()
    
    def paintEvent(self, event):
        """Paint the histogram."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        
        # Calculate bar width
        bar_width = width / len(self.strength_bins)
        max_count = max(self.strength_bins) if any(self.strength_bins) else 1
        
        # Create gradient for strength visualization
        gradient = QLinearGradient(0, 0, width, 0)
        gradient.setColorAt(0, QColor(231, 76, 60))    # Red (weak)
        gradient.setColorAt(0.5, QColor(241, 196, 15)) # Yellow (medium)
        gradient.setColorAt(1, QColor(46, 204, 113))   # Green (strong)
        
        # Draw bars
        for i, count in enumerate(self.strength_bins):
            # Calculate bar height proportional to count
            bar_height = (count / max_count) * (height - 40) if max_count > 0 else 0
            
            # Calculate bar color based on strength
            strength = (i + 0.5) / 10
            color = QColor(
                int((1 - strength) * 231 + strength * 46),
                int((1 - strength) * 76 + strength * 204),
                int((1 - strength) * 60 + strength * 113)
            )
            
            # Draw bar
            painter.fillRect(i * bar_width, height - bar_height - 20, 
                            bar_width - 2, bar_height, color)
            
            # Draw label
            label = f"{i/10:.1f}-{(i+1)/10:.1f}"
            painter.drawText(i * bar_width, height - 20, bar_width, 20, 
                            Qt.AlignCenter, label)


class MemoryVisualizationWidget(QWidget):
    """Main widget for memory visualization."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Memory Visualization")
        self.resize(800, 600)
        
        if not QT_AVAILABLE:
            self._setup_fallback_ui()
            return
            
        self._setup_ui()
        
        # Data
        self.memories = []
        
        # Update timer for decay visualization
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_decay_visualization)
        self.update_timer.start(1000)  # Update every second
    
    def _setup_ui(self):
        """Set up the UI components."""
        main_layout = QVBoxLayout(self)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Filter by type
        self.type_filter = QComboBox(self)
        self.type_filter.addItem("All Types")
        self.type_filter.addItems(["fact", "experience", "relation", "procedure"])
        self.type_filter.currentTextChanged.connect(self._apply_filters)
        controls_layout.addWidget(QLabel("Filter by Type:"))
        controls_layout.addWidget(self.type_filter)
        
        # Filter by minimum strength
        self.strength_filter = QSlider(Qt.Horizontal, self)
        self.strength_filter.setRange(0, 100)
        self.strength_filter.setValue(0)
        self.strength_filter.setTickPosition(QSlider.TicksBelow)
        self.strength_filter.setTickInterval(10)
        self.strength_filter.valueChanged.connect(self._apply_filters)
        controls_layout.addWidget(QLabel("Min Strength:"))
        controls_layout.addWidget(self.strength_filter)
        
        # Show decay checkbox
        self.show_decay = QCheckBox("Show Decay Simulation", self)
        self.show_decay.setChecked(True)
        self.show_decay.stateChanged.connect(self._toggle_decay_simulation)
        controls_layout.addWidget(self.show_decay)
        
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)
        
        # Graph view
        self.graph_view = MemoryGraphView(self)
        main_layout.addWidget(self.graph_view, 3)
        
        # Distribution charts layout
        charts_layout = QHBoxLayout()
        
        # Memory type distribution
        self.type_distribution = MemoryTypeDistribution(self)
        charts_layout.addWidget(self.type_distribution)
        
        # Memory strength histogram
        self.strength_histogram = MemoryStrengthHistogram(self)
        charts_layout.addWidget(self.strength_histogram)
        
        main_layout.addLayout(charts_layout, 1)
        
        # Details panel
        self.details_label = QLabel("Select a memory node to view details", self)
        self.details_label.setWordWrap(True)
        self.details_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.details_label.setMinimumHeight(100)
        main_layout.addWidget(self.details_label)
        
        # Connect signals
        self.graph_view.memory_selected.connect(self._show_memory_details)
    
    def _setup_fallback_ui(self):
        """Set up a fallback UI when Qt is not available."""
        layout = QVBoxLayout(self)
        label = QLabel("PySide6 is not available. Memory visualization is limited.", self)
        layout.addWidget(label)
    
    def update_memories(self, memory_data: List[Dict[str, Any]]):
        """Update the visualization with new memory data."""
        if not QT_AVAILABLE:
            return
            
        # Convert raw data to MemoryNode objects
        self.memories = []
        for data in memory_data:
            memory = MemoryNode(
                memory_id=data.get('id', ''),
                content=data.get('content', ''),
                memory_type=data.get('memory_type', 'default'),
                strength=data.get('strength', 0.5),
                created_at=data.get('created_at', time.time()),
                last_accessed=data.get('last_accessed', time.time()),
                tags=data.get('tags', []),
                metadata=data.get('metadata', {})
            )
            self.memories.append(memory)
        
        # Apply current filters
        self._apply_filters()
        
        # Update distribution charts
        self.type_distribution.update_distribution(self.memories)
        self.strength_histogram.update_histogram(self.memories)
    
    def _apply_filters(self):
        """Apply filters to the memories."""
        if not QT_AVAILABLE or not self.memories:
            return
            
        filtered_memories = self.memories.copy()
        
        # Filter by type
        selected_type = self.type_filter.currentText()
        if selected_type != "All Types":
            filtered_memories = [m for m in filtered_memories if m.memory_type == selected_type]
        
        # Filter by strength
        min_strength = self.strength_filter.value() / 100.0
        filtered_memories = [m for m in filtered_memories if m.strength >= min_strength]
        
        # Update graph view with filtered memories
        self.graph_view.update_memories(filtered_memories)
    
    def _show_memory_details(self, memory_id: str):
        """Show details for the selected memory."""
        if not QT_AVAILABLE:
            return
            
        for memory in self.memories:
            if memory.memory_id == memory_id:
                details = f"<h3>{memory.content}</h3>"
                details += f"<p><b>ID:</b> {memory.memory_id}</p>"
                details += f"<p><b>Type:</b> {memory.memory_type}</p>"
                details += f"<p><b>Strength:</b> {memory.strength:.2f}</p>"
                details += f"<p><b>Created:</b> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(memory.created_at))}</p>"
                details += f"<p><b>Last Accessed:</b> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(memory.last_accessed))}</p>"
                details += f"<p><b>Tags:</b> {', '.join(memory.tags)}</p>"
                
                if memory.metadata:
                    details += "<p><b>Metadata:</b></p><ul>"
                    for key, value in memory.metadata.items():
                        details += f"<li>{key}: {value}</li>"
                    details += "</ul>"
                
                self.details_label.setText(details)
                break
    
    def _update_decay_visualization(self):
        """Update visualization to simulate memory decay over time."""
        if not QT_AVAILABLE or not self.show_decay.isChecked():
            return
            
        # Simulate decay for visualization purposes only
        # This doesn't affect the actual memory data
        for memory in self.memories:
            # Simulate very slight decay for visualization
            memory.strength = max(0.1, memory.strength * 0.9999)
        
        # Re-apply filters to update the visualization
        self._apply_filters()
        
        # Update distribution charts
        self.strength_histogram.update_histogram(self.memories)
    
    def _toggle_decay_simulation(self, state):
        """Toggle the decay simulation on/off."""
        if state == Qt.Checked:
            self.update_timer.start(1000)
        else:
            self.update_timer.stop()
            # Reset memories to their original state
            self._apply_filters()


class MemoryVisualizer:
    """Main class for memory visualization, providing a facade to the UI components."""
    
    def __init__(self):
        self.widget = None if not QT_AVAILABLE else MemoryVisualizationWidget()
        
    def show(self):
        """Show the visualization widget."""
        if self.widget:
            self.widget.show()
        else:
            logging.warning("Cannot show memory visualization: Qt not available")
    
    def update(self, memory_data: List[Dict[str, Any]]):
        """Update the visualization with new memory data."""
        if self.widget:
            self.widget.update_memories(memory_data)
    
    def is_available(self) -> bool:
        """Check if visualization is available."""
        return QT_AVAILABLE and self.widget is not None 