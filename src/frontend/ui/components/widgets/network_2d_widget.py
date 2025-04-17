from PySide6.QtWidgets import QWidget, QToolTip
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QPainterPath, QLinearGradient
from PySide6.QtCore import Qt, QPointF, Signal, QRectF, QTime, QElapsedTimer, QTimer
from typing import Dict, Any, List, Tuple, Optional
import logging
import math
import random

from .visualization_widget import VisualizationWidget
from .growth_visualizer import GrowthVisualizer

class Network2DWidget(VisualizationWidget):
    """2D neural network visualization widget."""
    
    # Signals for testing and monitoring
    layout_generated = Signal(int, int)  # Emitted after layout generation
    node_updated = Signal(int, float)  # Emitted when a node's activation changes
    connection_updated = Signal(int, float)  # Emitted when a connection's weight changes
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Initialize growth visualizer
        self.growth_visualizer = GrowthVisualizer(self)
        self.growth_visualizer.setMinimumSize(400, 400)
        
        # Growth state tracking
        self.last_growth_stage = None
        self.health_metrics = {}
        self.stability_score = 1.0
        self.gate_states = {}
        self.consciousness_level = 0.0
        self.energy_level = 1.0
        
        # Backend connection
        self.backend_connector = None
        self.backend_connected = False
        self.backend_path = None
        
        # Timers
        self.growth_timer = QTimer(self)
        self.growth_timer.timeout.connect(self.update_growth)
        self.growth_timer.start(100)  # Update growth every 100ms
        
        # Signals
        self.new_nodes_added = Signal(list)
        self.growth_stage_changed = Signal(str)
        
        # Initialize other attributes
        self.nodes = []
        self.connections = []
        self.signals = []
        self.node_radius = 15
        self.connection_width = 2
        self._num_layers = 0
        self._nodes_per_layer = 0
        self.hovered_node = -1
        
        # Animation variables
        self.animation_speed = 1.0
        self.last_update_time = QTime.currentTime()
        self.elapsed_timer = QElapsedTimer()
        self.elapsed_timer.start()
        
        # Node and connection animation parameters
        self.node_oscillation_speed = 0.5
        self.connection_oscillation_speed = 0.25
        self.node_phase_offset = 0.0
        self.connection_phase_offset = 0.0
        
        # Node complexity parameters
        self.min_nodes = 1
        self.max_nodes = 300
        self.current_node_complexity = 0.0
        self.target_node_complexity = 0.0
        self.complexity_transition_speed = 0.1
        
        # Signal transfer animation
        self.signal_speed = 0.5
        self.signals = []
        self.signal_radius = 5
        self.signal_frequency = 0.1
        
        # Bidirectional signal transfer variables
        self.bidirectional_enabled = True
        self.reverse_signal_probability = 0.3
        self.reverse_signal_speed_multiplier = 0.8
        self.reverse_signal_strength_multiplier = 0.7
        
        # Diagonal signal transfer variables
        self.diagonal_signal_enabled = True
        self.diagonal_signal_probability = 0.3
        self.diagonal_signal_speed_multiplier = 1.2
        self.diagonal_signal_strength_multiplier = 0.8
        
        self.setMouseTracking(True)
        
    def initialize(self, params: Dict[str, Any]) -> bool:
        """Initialize the 2D network visualization."""
        try:
            if not super().initialize(params):
                return False
                
            self.logger.info("Initializing 2D network visualization")
            
            # Initialize growth visualizer
            if 'growth' in params:
                self.growth_visualizer.set_config(params['growth'])
            
            # Generate initial network layout
            num_layers = params.get("num_layers", 3)
            nodes_per_layer = params.get("nodes_per_layer", 4)
            
            # Set animation parameters from params if provided
            self.animation_speed = params.get("animation_speed", 1.0)
            self.node_oscillation_speed = params.get("node_oscillation_speed", 0.5)
            self.connection_oscillation_speed = params.get("connection_oscillation_speed", 0.25)
            self.signal_speed = params.get("signal_speed", 0.5)
            self.signal_frequency = params.get("signal_frequency", 0.1)
            
            # Set node complexity parameters
            self.min_nodes = params.get("min_nodes", 1)
            self.max_nodes = params.get("max_nodes", 300)
            self.current_node_complexity = params.get("initial_complexity", 0.0)
            self.target_node_complexity = self.current_node_complexity
            self.complexity_transition_speed = params.get("complexity_transition_speed", 0.1)
            
            # Set bidirectional signal parameters
            self.bidirectional_enabled = params.get("bidirectional_enabled", True)
            self.reverse_signal_probability = params.get("reverse_signal_probability", 0.3)
            self.reverse_signal_speed_multiplier = params.get("reverse_signal_speed_multiplier", 0.8)
            self.reverse_signal_strength_multiplier = params.get("reverse_signal_strength_multiplier", 0.7)
            
            # Set diagonal signal parameters
            self.diagonal_signal_enabled = params.get("diagonal_signal_enabled", True)
            self.diagonal_signal_probability = params.get("diagonal_signal_probability", 0.3)
            self.diagonal_signal_speed_multiplier = params.get("diagonal_signal_speed_multiplier", 1.2)
            self.diagonal_signal_strength_multiplier = params.get("diagonal_signal_strength_multiplier", 0.8)
            
            # Start growth timer
            self.growth_timer.start()
            
            return self._generate_network_layout(num_layers, nodes_per_layer)
        except Exception as e:
            self.logger.error(f"Failed to initialize 2D network: {str(e)}")
            self.error_occurred.emit(str(e))
            return False
            
    def _generate_network_layout(self, num_layers: int, nodes_per_layer: int) -> bool:
        """Generate the initial network layout."""
        try:
            self.nodes.clear()
            self.connections.clear()
            
            # Handle single node case
            if num_layers == 1 and nodes_per_layer == 1:
                # Place single node in center
                self.nodes.append((self.width() / 2, self.height() / 2, 0.0))
                self._num_layers = 1
                self._nodes_per_layer = 1
                self.layout_generated.emit(1, 1)
                return True
            
            # Calculate node positions
            width = self.width()
            height = self.height()
            layer_spacing = width / (num_layers + 1)
            node_spacing = height / (nodes_per_layer + 1)
            
            # Create nodes
            for layer in range(num_layers):
                for node in range(nodes_per_layer):
                    x = (layer + 1) * layer_spacing
                    y = (node + 1) * node_spacing
                    self.nodes.append((x, y, 0.0))  # Initial activation of 0
                    
            # Create connections
            for layer in range(num_layers - 1):
                for from_node in range(nodes_per_layer):
                    for to_node in range(nodes_per_layer):
                        from_idx = layer * nodes_per_layer + from_node
                        to_idx = (layer + 1) * nodes_per_layer + to_node
                        self.connections.append((from_idx, to_idx, 0.5))  # Initial weight of 0.5
            
            self._num_layers = num_layers
            self._nodes_per_layer = nodes_per_layer
            self.layout_generated.emit(num_layers, nodes_per_layer)
            return True
                        
        except Exception as e:
            self.logger.error(f"Failed to generate network layout: {str(e)}")
            self.error_occurred.emit(str(e))
            return False
            
    def _get_node_color(self, activation: float) -> QColor:
        """Get a color based on node activation using a gradient."""
        if activation < 0.5:
            # Blue (cold) to white gradient
            intensity = int(255 * (activation * 2))
            return QColor(intensity, intensity, 255)
        else:
            # White to red (hot) gradient
            intensity = int(255 * (2 - activation * 2))
            return QColor(255, intensity, intensity)
            
    def _get_connection_color(self, weight: float) -> QColor:
        """Get a color based on connection weight using a gradient."""
        if weight < 0.5:
            # Negative weights: blue gradient
            intensity = int(255 * (weight * 2))
            return QColor(0, 0, intensity)
        else:
            # Positive weights: red gradient
            intensity = int(255 * (weight * 2 - 1))
            return QColor(intensity, 0, 0)
            
    def update_visualization(self):
        """Update the 2D network visualization based on elapsed time."""
        try:
            if not self.is_initialized():
                return
                
            # Calculate time since last update
            current_time = self.elapsed_timer.elapsed() / 1000.0  # Convert to seconds
            time_delta = current_time - self.last_update_time
            self.last_update_time = current_time
            
            # Update node complexity
            if abs(self.current_node_complexity - self.target_node_complexity) > 0.001:
                # Smoothly transition to target complexity
                direction = 1.0 if self.target_node_complexity > self.current_node_complexity else -1.0
                self.current_node_complexity += direction * self.complexity_transition_speed * time_delta
                self.current_node_complexity = max(0.0, min(1.0, self.current_node_complexity))
                
                # Calculate new number of nodes and layers
                total_nodes = int(self.min_nodes + (self.max_nodes - self.min_nodes) * self.current_node_complexity)
                
                # Special case for single node
                if total_nodes == 1:
                    num_layers = 1
                    nodes_per_layer = 1
                else:
                    num_layers = max(2, int(math.sqrt(total_nodes / 2)))
                    nodes_per_layer = max(2, int(total_nodes / num_layers))
                
                # Regenerate layout if needed
                if num_layers != self._num_layers or nodes_per_layer != self._nodes_per_layer:
                    self._generate_network_layout(num_layers, nodes_per_layer)
            
            # Update node activations based on time
            for i in range(len(self.nodes)):
                x, y, _ = self.nodes[i]
                # Calculate activation using sine wave based on time
                phase = (current_time * self.node_oscillation_speed * self.animation_speed + 
                        i * self.node_phase_offset)
                activation = (math.sin(phase) + 1) / 2  # Normalize to [0, 1]
                self.nodes[i] = (x, y, activation)
                self.node_updated.emit(i, activation)
                
            # Update connection weights based on time
            for i in range(len(self.connections)):
                from_idx, to_idx, _ = self.connections[i]
                # Calculate weight using sine wave based on time
                phase = (current_time * self.connection_oscillation_speed * self.animation_speed + 
                        i * self.connection_phase_offset)
                weight = (math.sin(phase) + 1) / 2  # Normalize to [0, 1]
                self.connections[i] = (from_idx, to_idx, weight)
                self.connection_updated.emit(i, weight)
                
            # Update signal transfers
            self._update_signals(time_delta)
            
            # Generate new signals
            if current_time % (1.0 / self.signal_frequency) < time_delta:
                self._generate_new_signals()
                
            super().update_visualization()
        except Exception as e:
            self.logger.error(f"Failed to update 2D network: {str(e)}")
            self.error_occurred.emit(str(e))
            
    def _update_signals(self, time_delta: float):
        """Update the position and state of all signals."""
        # Update existing signals
        new_signals = []
        for from_idx, to_idx, progress, strength, is_reverse in self.signals:
            # Determine if this is a diagonal signal
            is_diagonal = (from_idx % self._nodes_per_layer != to_idx % self._nodes_per_layer)
            
            # Calculate speed multipliers
            diagonal_multiplier = self.diagonal_signal_speed_multiplier if is_diagonal else 1.0
            reverse_multiplier = self.reverse_signal_speed_multiplier if is_reverse else 1.0
            
            # Move signal along connection
            progress += time_delta * self.signal_speed * self.animation_speed * diagonal_multiplier * reverse_multiplier
            
            # Remove signal if it reached the destination
            if progress < 1.0:
                new_signals.append((from_idx, to_idx, progress, strength, is_reverse))
                
        self.signals = new_signals
        
    def _generate_new_signals(self):
        """Generate new signals between nodes."""
        for from_idx, to_idx, weight in self.connections:
            # Only generate signals for strong enough connections
            if weight > 0.3:
                # Determine if this should be a diagonal signal
                is_diagonal = (self.diagonal_signal_enabled and 
                             from_idx % self._nodes_per_layer != to_idx % self._nodes_per_layer)
                
                # Determine if this should be a reverse signal
                is_reverse = (self.bidirectional_enabled and 
                            random.random() < self.reverse_signal_probability)
                
                # Calculate signal strength
                base_strength = weight * (0.5 + 0.5 * math.sin(self.elapsed_timer.elapsed() / 1000.0))
                diagonal_multiplier = self.diagonal_signal_strength_multiplier if is_diagonal else 1.0
                reverse_multiplier = self.reverse_signal_strength_multiplier if is_reverse else 1.0
                strength = base_strength * diagonal_multiplier * reverse_multiplier
                
                # Add new signal
                self.signals.append((from_idx, to_idx, 0.0, strength, is_reverse))
                
    def paintEvent(self, event):
        """Paint the network visualization."""
        try:
            if not self.is_initialized():
                return
                
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Draw growth visualization first
            if self.growth_visualizer:
                self.growth_visualizer.paintEvent(event)
            
            # Draw connections
            for from_idx, to_idx, weight in self.connections:
                from_x, from_y, _ = self.nodes[from_idx]
                to_x, to_y, _ = self.nodes[to_idx]
                
                # Set pen color based on weight
                color = self._get_connection_color(weight)
                painter.setPen(QPen(color, self.connection_width))
                painter.drawLine(QPointF(from_x, from_y), QPointF(to_x, to_y))
                
            # Draw signals
            for from_idx, to_idx, progress, strength, is_reverse in self.signals:
                from_x, from_y, _ = self.nodes[from_idx]
                to_x, to_y, _ = self.nodes[to_idx]
                
                # Calculate signal position
                if is_reverse:
                    signal_x = to_x + (from_x - to_x) * progress
                    signal_y = to_y + (from_y - to_y) * progress
                else:
                    signal_x = from_x + (to_x - from_x) * progress
                    signal_y = from_y + (to_y - from_y) * progress
                
                # Draw signal pulse
                color = self._get_connection_color(strength)
                painter.setPen(QPen(color, 1))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(QPointF(signal_x, signal_y), 
                                  self.signal_radius, self.signal_radius)
                
            # Draw nodes
            for i, (x, y, activation) in enumerate(self.nodes):
                # Set colors based on activation
                color = self._get_node_color(activation)
                painter.setBrush(QBrush(color))
                
                # Highlight hovered node
                if i == self.hovered_node:
                    painter.setPen(QPen(Qt.yellow, 2))
                else:
                    painter.setPen(QPen(Qt.black, 1))
                    
                # Draw node
                painter.drawEllipse(QPointF(x, y), self.node_radius, self.node_radius)
                
                # Draw node label
                layer = i // self._nodes_per_layer
                node = i % self._nodes_per_layer
                label = f"L{layer}N{node}"
                rect = QRectF(x - self.node_radius, y - self.node_radius/2, 
                            2 * self.node_radius, self.node_radius)
                painter.drawText(rect, Qt.AlignCenter, label)
                
        except Exception as e:
            self.logger.error(f"Failed to paint 2D network: {str(e)}")
            self.error_occurred.emit(str(e))
            
    def start_animation(self, interval_ms: int = 16):
        """Start the visualization animation."""
        super().start_animation(interval_ms)
        self.elapsed_timer.restart()
        self.last_update_time = 0
        self.signals.clear()
        
    def stop_animation(self):
        """Stop the visualization animation."""
        super().stop_animation()
        
    def resizeEvent(self, event):
        """Handle widget resize events."""
        try:
            super().resizeEvent(event)
            if self.is_initialized():
                # Update growth visualizer size
                if self.growth_visualizer:
                    self.growth_visualizer.setGeometry(self.rect())
                
                # Regenerate layout on resize
                self._generate_network_layout(self._num_layers, self._nodes_per_layer)
        except Exception as e:
            self.logger.error(f"Error during 2D network resize: {str(e)}")
            self.error_occurred.emit(str(e))
            
    def get_node_count(self) -> int:
        """Get the total number of nodes in the network."""
        return len(self.nodes)
        
    def get_connection_count(self) -> int:
        """Get the total number of connections in the network."""
        return len(self.connections)
        
    def get_node_activation(self, index: int) -> Optional[float]:
        """Get the activation value of a specific node."""
        try:
            return self.nodes[index][2]
        except IndexError:
            self.logger.error(f"Invalid node index: {index}")
            return None
            
    def get_connection_weight(self, index: int) -> Optional[float]:
        """Get the weight value of a specific connection."""
        try:
            return self.connections[index][2]
        except IndexError:
            self.logger.error(f"Invalid connection index: {index}")
            return None
            
    def set_animation_speed(self, speed: float):
        """Set the animation speed multiplier."""
        self.animation_speed = max(0.1, min(5.0, speed))  # Clamp between 0.1 and 5.0
        
    def cleanup(self):
        """Clean up visualization resources."""
        try:
            # Stop growth timer
            self.growth_timer.stop()
            
            # Clean up growth visualizer
            if self.growth_visualizer:
                self.growth_visualizer.cleanup()
                
            # Clean up network resources
            self.nodes.clear()
            self.connections.clear()
            self.signals.clear()
            
            super().cleanup()
        except Exception as e:
            self.logger.error(f"Error during 2D network cleanup: {str(e)}")
            self.error_occurred.emit(str(e))
            
    def set_signal_speed(self, speed: float):
        """Set the signal transfer speed."""
        self.signal_speed = max(0.1, min(2.0, speed))
        
    def set_signal_frequency(self, frequency: float):
        """Set how often new signals are generated."""
        self.signal_frequency = max(0.01, min(1.0, frequency))
        
    def mouseMoveEvent(self, event):
        """Handle mouse move events for hover effects."""
        try:
            pos = event.pos()
            old_hovered = self.hovered_node
            self.hovered_node = -1
            
            # Check if mouse is over any node
            for i, (x, y, _) in enumerate(self.nodes):
                if math.sqrt((pos.x() - x)**2 + (pos.y() - y)**2) <= self.node_radius:
                    self.hovered_node = i
                    activation = self.get_node_activation(i)
                    tooltip = f"Node {i}\nLayer: {i // self._nodes_per_layer}\nActivation: {activation:.3f}"
                    QToolTip.showText(event.globalPos(), tooltip)
                    break
                    
            if old_hovered != self.hovered_node:
                self.update()
                
        except Exception as e:
            self.logger.error(f"Error handling mouse move: {str(e)}")
            
    def set_diagonal_signal_enabled(self, enabled: bool):
        """Enable or disable diagonal signal transfer."""
        self.diagonal_signal_enabled = enabled
        
    def set_diagonal_signal_probability(self, probability: float):
        """Set the probability of generating diagonal signals."""
        self.diagonal_signal_probability = max(0.0, min(1.0, probability))
        
    def set_diagonal_signal_speed_multiplier(self, multiplier: float):
        """Set the speed multiplier for diagonal signals."""
        self.diagonal_signal_speed_multiplier = max(0.1, min(2.0, multiplier))
        
    def set_diagonal_signal_strength_multiplier(self, multiplier: float):
        """Set the strength multiplier for diagonal signals."""
        self.diagonal_signal_strength_multiplier = max(0.1, min(1.0, multiplier))
        
    def set_bidirectional_enabled(self, enabled: bool):
        """Enable or disable bidirectional signal transfer."""
        self.bidirectional_enabled = enabled
        
    def set_reverse_signal_probability(self, probability: float):
        """Set the probability of generating reverse signals."""
        self.reverse_signal_probability = max(0.0, min(1.0, probability))
        
    def set_reverse_signal_speed_multiplier(self, multiplier: float):
        """Set the speed multiplier for reverse signals."""
        self.reverse_signal_speed_multiplier = max(0.1, min(1.0, multiplier))
        
    def set_reverse_signal_strength_multiplier(self, multiplier: float):
        """Set the strength multiplier for reverse signals."""
        self.reverse_signal_strength_multiplier = max(0.1, min(1.0, multiplier))
        
    def set_node_complexity(self, complexity: float):
        """Set the target node complexity level (0.0 to 1.0)."""
        self.target_node_complexity = max(0.0, min(1.0, complexity))
        
    def set_complexity_transition_speed(self, speed: float):
        """Set the speed of complexity transitions."""
        self.complexity_transition_speed = max(0.01, min(1.0, speed))
        
    def get_current_node_count(self) -> int:
        """Get the current number of nodes in the network."""
        return len(self.nodes)
        
    def get_current_complexity(self) -> float:
        """Get the current complexity level (0.0 to 1.0)."""
        return self.current_node_complexity 

    def update_growth(self):
        """Update growth visualization based on current state."""
        if not self.growth_visualizer:
            return
            
        # Get new nodes and stage changes
        new_nodes = self._get_new_nodes()
        stage_change = self._get_stage_change()
        
        # Update growth visualizer
        self.growth_visualizer.update_from_network(
            nodes=self.nodes,
            connections=self.connections,
            signals=self.signals
        )
        
        # Update system state
        system_state = {
            'health': self.health_metrics,
            'stability': self.stability_score,
            'gate_states': self.gate_states,
            'consciousness': self.consciousness_level,
            'energy': self.energy_level
        }
        self.growth_visualizer.update_from_system_state(system_state)
        
        # Update backend state
        if self.backend_connector and self.backend_connector.is_connected():
            backend_info = self.backend_connector.get_backend_info()
            self.growth_visualizer.update_from_backend(backend_info)
            
        # Emit signals for new nodes and stage changes
        if new_nodes:
            self.new_nodes_added.emit(new_nodes)
        if stage_change:
            self.growth_stage_changed.emit(stage_change)
            
        # Update visualization
        self.update()

    def _get_new_nodes(self) -> List[Dict]:
        """Get newly added nodes since last update."""
        new_nodes = []
        for node in self.nodes:
            if node.get('new', False):
                new_nodes.append(node)
                node['new'] = False
        return new_nodes

    def _get_stage_change(self) -> Optional[str]:
        """Check if growth stage has changed."""
        if not self.growth_visualizer:
            return None
            
        current_stage = self.growth_visualizer.growth_stage.name
        if current_stage != self.last_growth_stage:
            self.last_growth_stage = current_stage
            return current_stage
        return None