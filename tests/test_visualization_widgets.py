import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from src.frontend.ui.components.widgets.network_2d_widget import Network2DWidget
from src.frontend.ui.components.widgets.network_3d_widget import Network3DWidget

@pytest.fixture
def app():
    """Create a QApplication instance for testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

@pytest.fixture
def network_2d_widget(app):
    """Create a Network2DWidget instance for testing."""
    widget = Network2DWidget()
    widget.show()
    return widget

@pytest.fixture
def network_3d_widget(app):
    """Create a Network3DWidget instance for testing."""
    widget = Network3DWidget()
    widget.show()
    return widget

def test_2d_widget_initialization(network_2d_widget):
    """Test initialization of 2D network widget."""
    # Test initialization with default parameters
    assert network_2d_widget.initialize({})
    assert network_2d_widget.is_initialized()
    assert network_2d_widget.get_node_count() > 0
    assert network_2d_widget.get_connection_count() > 0
    
    # Test initialization with custom parameters
    assert network_2d_widget.initialize({
        "num_layers": 4,
        "nodes_per_layer": 5
    })
    assert network_2d_widget.get_node_count() == 20  # 4 layers * 5 nodes
    assert network_2d_widget.get_connection_count() == 75  # 3 connections * 5 * 5

def test_3d_widget_initialization(network_3d_widget):
    """Test initialization of 3D network widget."""
    # Test initialization with default parameters
    assert network_3d_widget.initialize({})
    assert network_3d_widget.is_initialized()
    assert network_3d_widget.get_node_count() > 0
    assert network_3d_widget.get_connection_count() > 0
    
    # Test initialization with custom parameters
    assert network_3d_widget.initialize({
        "num_layers": 4,
        "nodes_per_layer": 5
    })
    assert network_3d_widget.get_node_count() == 20  # 4 layers * 5 nodes
    assert network_3d_widget.get_connection_count() == 75  # 3 connections * 5 * 5

def test_2d_widget_animation(network_2d_widget):
    """Test animation functionality of 2D network widget."""
    # Initialize widget
    assert network_2d_widget.initialize({})
    
    # Start animation
    network_2d_widget.start_animation()
    assert network_2d_widget.is_animating()
    
    # Wait for a few updates
    QTimer.singleShot(100, lambda: network_2d_widget.stop_animation())
    
    # Verify node activations changed
    initial_activation = network_2d_widget.get_node_activation(0)
    def check_activation():
        assert network_2d_widget.get_node_activation(0) != initial_activation
    QTimer.singleShot(200, check_activation)

def test_3d_widget_animation(network_3d_widget):
    """Test animation functionality of 3D network widget."""
    # Initialize widget
    assert network_3d_widget.initialize({})
    
    # Start animation
    network_3d_widget.start_animation()
    assert network_3d_widget.is_animating()
    
    # Wait for a few updates
    QTimer.singleShot(100, lambda: network_3d_widget.stop_animation())
    
    # Verify node activations and camera angle changed
    initial_activation = network_3d_widget.get_node_activation(0)
    initial_angle = network_3d_widget.get_camera_angle()
    def check_changes():
        assert network_3d_widget.get_node_activation(0) != initial_activation
        assert network_3d_widget.get_camera_angle() != initial_angle
    QTimer.singleShot(200, check_changes)

def test_2d_widget_resize(network_2d_widget):
    """Test resize functionality of 2D network widget."""
    # Initialize widget
    assert network_2d_widget.initialize({})
    
    # Store initial node positions
    initial_positions = [network_2d_widget.nodes[i][:2] for i in range(len(network_2d_widget.nodes))]
    
    # Resize widget
    network_2d_widget.resize(800, 600)
    
    # Verify node positions updated
    for i in range(len(network_2d_widget.nodes)):
        assert network_2d_widget.nodes[i][:2] != initial_positions[i]

def test_3d_widget_resize(network_3d_widget):
    """Test resize functionality of 3D network widget."""
    # Initialize widget
    assert network_3d_widget.initialize({})
    
    # Store initial node positions
    initial_positions = [network_3d_widget.nodes[i][:3] for i in range(len(network_3d_widget.nodes))]
    
    # Resize widget
    network_3d_widget.resize(800, 600)
    
    # Verify node positions updated
    for i in range(len(network_3d_widget.nodes)):
        assert network_3d_widget.nodes[i][:3] != initial_positions[i]

def test_2d_widget_cleanup(network_2d_widget):
    """Test cleanup functionality of 2D network widget."""
    # Initialize and start animation
    assert network_2d_widget.initialize({})
    network_2d_widget.start_animation()
    
    # Cleanup
    network_2d_widget.cleanup()
    
    # Verify cleanup
    assert not network_2d_widget.is_initialized()
    assert not network_2d_widget.is_animating()
    assert len(network_2d_widget.nodes) == 0
    assert len(network_2d_widget.connections) == 0

def test_3d_widget_cleanup(network_3d_widget):
    """Test cleanup functionality of 3D network widget."""
    # Initialize and start animation
    assert network_3d_widget.initialize({})
    network_3d_widget.start_animation()
    
    # Cleanup
    network_3d_widget.cleanup()
    
    # Verify cleanup
    assert not network_3d_widget.is_initialized()
    assert not network_3d_widget.is_animating()
    assert not network_3d_widget._gl_initialized
    assert network_3d_widget.program is None
    assert len(network_3d_widget.nodes) == 0
    assert len(network_3d_widget.connections) == 0

def test_2d_widget_error_handling(network_2d_widget):
    """Test error handling of 2D network widget."""
    # Test invalid initialization
    assert not network_2d_widget.initialize({"invalid_param": "value"})
    
    # Test invalid node access
    assert network_2d_widget.get_node_activation(-1) is None
    assert network_2d_widget.get_node_activation(1000) is None
    
    # Test invalid connection access
    assert network_2d_widget.get_connection_weight(-1) is None
    assert network_2d_widget.get_connection_weight(1000) is None

def test_3d_widget_error_handling(network_3d_widget):
    """Test error handling of 3D network widget."""
    # Test invalid initialization
    assert not network_3d_widget.initialize({"invalid_param": "value"})
    
    # Test invalid node access
    assert network_3d_widget.get_node_activation(-1) is None
    assert network_3d_widget.get_node_activation(1000) is None
    
    # Test invalid connection access
    assert network_3d_widget.get_connection_weight(-1) is None
    assert network_3d_widget.get_connection_weight(1000) is None 