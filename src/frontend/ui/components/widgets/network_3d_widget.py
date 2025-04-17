from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtOpenGL import QOpenGLShaderProgram, QOpenGLShader
from PySide6.QtGui import QMatrix4x4, QVector3D
from PySide6.QtCore import Qt, QTimer, Signal
from typing import Dict, Any, List, Tuple, Optional
import logging
import numpy as np
from OpenGL import GL

from .visualization_widget import VisualizationWidget

class Network3DWidget(QOpenGLWidget, VisualizationWidget):
    """3D neural network visualization widget using OpenGL."""
    
    # Signals for testing and monitoring
    layout_generated = Signal(int, int)  # Emitted after layout generation
    node_updated = Signal(int, float)  # Emitted when a node's activation changes
    connection_updated = Signal(int, float)  # Emitted when a connection's weight changes
    camera_rotated = Signal(float)  # Emitted when camera angle changes
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.nodes: List[Tuple[float, float, float, float]] = []  # (x, y, z, activation)
        self.connections: List[Tuple[int, int, float]] = []  # (from_idx, to_idx, weight)
        self.node_radius = 0.1
        self.connection_width = 0.05
        
        # OpenGL resources
        self.program = None
        self.vertex_buffer = None
        self.index_buffer = None
        self.mvp_matrix = QMatrix4x4()
        
        # Camera parameters
        self.camera_distance = 5.0
        self.camera_angle = 0.0
        self.camera_height = 2.0
        
        # State tracking
        self._num_layers = 0
        self._nodes_per_layer = 0
        self._gl_initialized = False
        
    def initialize(self, params: Dict[str, Any]) -> bool:
        """Initialize the 3D network visualization."""
        try:
            if not super().initialize(params):
                return False
                
            self.logger.info("Initializing 3D network visualization")
            # Generate initial network layout
            num_layers = params.get("num_layers", 3)
            nodes_per_layer = params.get("nodes_per_layer", 4)
            return self._generate_network_layout(num_layers, nodes_per_layer)
        except Exception as e:
            self.logger.error(f"Failed to initialize 3D network: {str(e)}")
            self.error_occurred.emit(str(e))
            return False
            
    def _generate_network_layout(self, num_layers: int, nodes_per_layer: int) -> bool:
        """Generate the initial network layout."""
        try:
            self.nodes.clear()
            self.connections.clear()
            
            # Calculate node positions in 3D space
            layer_spacing = 2.0 / (num_layers + 1)
            node_spacing = 2.0 / (nodes_per_layer + 1)
            
            # Create nodes
            for layer in range(num_layers):
                for node in range(nodes_per_layer):
                    x = (layer + 1) * layer_spacing - 1.0
                    y = (node + 1) * node_spacing - 1.0
                    z = 0.0  # Initial z position
                    self.nodes.append((x, y, z, 0.0))  # Initial activation of 0
                    
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
            
    def initializeGL(self):
        """Initialize OpenGL resources."""
        try:
            if self._gl_initialized:
                return
                
            # Create shader program
            self.program = QOpenGLShaderProgram()
            self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, """
                attribute vec3 position;
                attribute vec3 color;
                uniform mat4 mvp;
                varying vec3 vColor;
                void main() {
                    gl_Position = mvp * vec4(position, 1.0);
                    vColor = color;
                }
            """)
            self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, """
                varying vec3 vColor;
                void main() {
                    gl_FragColor = vec4(vColor, 1.0);
                }
            """)
            self.program.link()
            
            # Enable depth testing
            GL.glEnable(GL.GL_DEPTH_TEST)
            
            self._gl_initialized = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenGL: {str(e)}")
            self.error_occurred.emit(str(e))
            
    def resizeGL(self, width: int, height: int):
        """Handle OpenGL viewport resize."""
        try:
            GL.glViewport(0, 0, width, height)
            self._update_projection_matrix()
        except Exception as e:
            self.logger.error(f"Error during OpenGL resize: {str(e)}")
            self.error_occurred.emit(str(e))
            
    def _update_projection_matrix(self):
        """Update the projection matrix based on current viewport."""
        try:
            self.mvp_matrix.setToIdentity()
            self.mvp_matrix.perspective(45.0, self.width() / self.height(), 0.1, 100.0)
            self.mvp_matrix.lookAt(
                QVector3D(0, self.camera_height, self.camera_distance),
                QVector3D(0, 0, 0),
                QVector3D(0, 1, 0)
            )
            self.mvp_matrix.rotate(self.camera_angle, 0, 1, 0)
        except Exception as e:
            self.logger.error(f"Failed to update projection matrix: {str(e)}")
            self.error_occurred.emit(str(e))
            
    def paintGL(self):
        """Render the 3D network visualization."""
        try:
            if not self.is_initialized() or not self._gl_initialized:
                return
                
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            
            self.program.bind()
            self.program.setUniformValue("mvp", self.mvp_matrix)
            
            # Draw connections
            for from_idx, to_idx, weight in self.connections:
                from_x, from_y, from_z, _ = self.nodes[from_idx]
                to_x, to_y, to_z, _ = self.nodes[to_idx]
                
                # Set color based on weight
                intensity = weight
                self.program.setAttributeValue("color", intensity, intensity, intensity)
                
                # Draw line
                GL.glBegin(GL.GL_LINES)
                GL.glVertex3f(from_x, from_y, from_z)
                GL.glVertex3f(to_x, to_y, to_z)
                GL.glEnd()
                
            # Draw nodes
            for x, y, z, activation in self.nodes:
                # Set color based on activation
                intensity = activation
                self.program.setAttributeValue("color", intensity, intensity, intensity)
                
                # Draw sphere
                self._draw_sphere(x, y, z, self.node_radius)
                
            self.program.release()
            
        except Exception as e:
            self.logger.error(f"Failed to paint 3D network: {str(e)}")
            self.error_occurred.emit(str(e))
            
    def _draw_sphere(self, x: float, y: float, z: float, radius: float):
        """Draw a sphere at the given position."""
        try:
            # Simple sphere approximation using triangles
            for i in range(0, 360, 10):
                for j in range(-90, 90, 10):
                    theta1 = np.radians(i)
                    theta2 = np.radians(i + 10)
                    phi1 = np.radians(j)
                    phi2 = np.radians(j + 10)
                    
                    # Calculate vertices
                    v1 = self._sphere_point(theta1, phi1, radius)
                    v2 = self._sphere_point(theta1, phi2, radius)
                    v3 = self._sphere_point(theta2, phi1, radius)
                    v4 = self._sphere_point(theta2, phi2, radius)
                    
                    # Draw two triangles
                    GL.glBegin(GL.GL_TRIANGLES)
                    GL.glVertex3f(x + v1[0], y + v1[1], z + v1[2])
                    GL.glVertex3f(x + v2[0], y + v2[1], z + v2[2])
                    GL.glVertex3f(x + v3[0], y + v3[1], z + v3[2])
                    
                    GL.glVertex3f(x + v2[0], y + v2[1], z + v2[2])
                    GL.glVertex3f(x + v4[0], y + v4[1], z + v4[2])
                    GL.glVertex3f(x + v3[0], y + v3[1], z + v3[2])
                    GL.glEnd()
                    
        except Exception as e:
            self.logger.error(f"Failed to draw sphere: {str(e)}")
            self.error_occurred.emit(str(e))
            
    def _sphere_point(self, theta: float, phi: float, radius: float) -> Tuple[float, float, float]:
        """Calculate a point on a sphere."""
        x = radius * np.cos(phi) * np.cos(theta)
        y = radius * np.cos(phi) * np.sin(theta)
        z = radius * np.sin(phi)
        return (x, y, z)
        
    def update_visualization(self):
        """Update the 3D network visualization."""
        try:
            if not self.is_initialized():
                return
                
            # Update node activations and connection weights
            for i in range(len(self.nodes)):
                x, y, z, activation = self.nodes[i]
                # Simulate some activation changes
                activation = (activation + 0.01) % 1.0
                self.nodes[i] = (x, y, z, activation)
                self.node_updated.emit(i, activation)
                
            for i in range(len(self.connections)):
                from_idx, to_idx, weight = self.connections[i]
                # Simulate some weight changes
                weight = (weight + 0.005) % 1.0
                self.connections[i] = (from_idx, to_idx, weight)
                self.connection_updated.emit(i, weight)
                
            # Rotate camera
            self.camera_angle = (self.camera_angle + 1) % 360
            self.camera_rotated.emit(self.camera_angle)
            self._update_projection_matrix()
            
            super().update_visualization()
        except Exception as e:
            self.logger.error(f"Failed to update 3D network: {str(e)}")
            self.error_occurred.emit(str(e))
            
    def resizeEvent(self, event):
        """Handle widget resize events."""
        try:
            super().resizeEvent(event)
            if self.is_initialized():
                # Regenerate layout on resize
                self._generate_network_layout(self._num_layers, self._nodes_per_layer)
        except Exception as e:
            self.logger.error(f"Error during 3D network resize: {str(e)}")
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
            return self.nodes[index][3]
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
            
    def get_camera_angle(self) -> float:
        """Get the current camera angle."""
        return self.camera_angle
        
    def cleanup(self):
        """Clean up visualization resources."""
        try:
            if self.program:
                self.program.release()
                self.program = None
                
            self.nodes.clear()
            self.connections.clear()
            self._gl_initialized = False
            super().cleanup()
        except Exception as e:
            self.logger.error(f"Error during 3D network cleanup: {str(e)}")
            self.error_occurred.emit(str(e)) 