import sys
import os
import json
import logging
import asyncio
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer
from components.widgets.network_2d_widget import Network2DWidget
from test_window import TestWindow
from config.system_profiler import generate_config, get_system_info, calculate_optimal_settings
from components.backend_connector import BackendConnector
from components.system_state import SystemState

def save_config(config: dict, path: str):
    """Save configuration to file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)

def load_config():
    """Load configuration from JSON file or generate new one"""
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'visualizer_config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Generating new configuration based on system capabilities...")
        config = generate_config()
        save_config(config, config_path)
        return config

def show_system_info(system_info: dict):
    """Display system information to user"""
    info_text = f"""
    System Information:
    - CPU: {system_info['cpu']['cores']} cores, {system_info['cpu']['threads']} threads
    - Memory: {system_info['memory']['total'] / (1024**3):.1f} GB total
    - GPU: {system_info['gpu']['name'] if system_info['gpu']['count'] > 0 else 'Not detected'}
    - Display: {system_info['display']['primary_resolution']['width']}x{system_info['display']['primary_resolution']['height']}
    
    Generated optimal settings based on your system capabilities.
    """
    
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText("System Profile Generated")
    msg.setInformativeText(info_text)
    msg.setWindowTitle("System Information")
    msg.exec_()

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('visualizer.log')
        ]
    )

class VisualizerLauncher:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.network_widget = Network2DWidget()
        self.system_state = SystemState()
        self.backend_path = os.path.join(os.path.dirname(__file__), '..', '..', 'integration', 'backend.py')
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize system profiler
        self.system_info = get_system_info()
        self.optimal_settings = calculate_optimal_settings(self.system_info)
        
        # Set up update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_visualization)
        self.update_timer.start(int(self.optimal_settings['animation']['update_interval'] * 1000))
        
        # Register backend callbacks
        self.network_widget.backend_connector.register_state_change_callback(self._on_backend_state_change)
        self.network_widget.backend_connector.register_metrics_change_callback(self._on_metrics_change)
        self.network_widget.backend_connector.register_gate_change_callback(self._on_gate_change)
        self.network_widget.backend_connector.register_health_change_callback(self._on_health_change)
        
    def _load_config(self) -> dict:
        """Load configuration from JSON file."""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'visualizer_config.json')
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            return {}
            
    async def initialize(self):
        """Initialize the visualizer and connect to backend."""
        try:
            # Connect to backend
            backend_connected = await self.network_widget.connect_backend(self.backend_path)
            if not backend_connected:
                logging.error("Failed to connect to backend system")
                return False
                
            # Apply optimal settings
            self._apply_settings(self.optimal_settings)
            
            # Start update timer
            self.update_timer.start()
            
            return True
        except Exception as e:
            logging.error(f"Error during initialization: {e}")
            return False
            
    def _apply_settings(self, settings: dict):
        """Apply settings to the visualizer."""
        try:
            # Apply network settings
            self.network_widget.set_node_radius(settings['network']['node_radius'])
            self.network_widget.set_connection_width(settings['network']['connection_width'])
            
            # Apply appearance settings
            self.network_widget.set_background_color(settings['appearance']['background_color'])
            self.network_widget.set_node_color(settings['appearance']['node_color'])
            self.network_widget.set_connection_color(settings['appearance']['connection_color'])
            
            # Apply animation settings
            self.update_timer.setInterval(int(settings['animation']['update_interval'] * 1000))
            
        except Exception as e:
            logging.error(f"Error applying settings: {e}")
            
    def _on_backend_state_change(self, state: dict):
        """Handle backend state changes."""
        try:
            # Update system state
            self.system_state.update_from_backend(state)
            
            # Adjust visualization parameters based on state
            if 'gate_states' in state:
                active_gates = sum(1 for gate in state['gate_states'].values() 
                                 if gate['state'] == 'OPEN')
                # Adjust node count based on active gates
                self.network_widget.target_node_count = max(
                    self.network_widget.min_nodes,
                    min(self.network_widget.max_nodes, active_gates * 2)
                )
                
        except Exception as e:
            logging.error(f"Error handling backend state change: {e}")
            
    def _on_metrics_change(self, metrics: dict):
        """Handle metrics changes."""
        try:
            # Adjust animation speed based on system load
            if 'cpu_usage' in metrics:
                load_factor = metrics['cpu_usage'] / 100.0
                self.network_widget.animation_speed = max(0.5, min(2.0, 1.0 + load_factor))
                
        except Exception as e:
            logging.error(f"Error handling metrics change: {e}")
            
    def _on_gate_change(self, gate_states: dict):
        """Handle gate state changes."""
        try:
            # Update connection strengths based on gate states
            for gate_id, state in gate_states.items():
                gate_idx = int(gate_id)
                if gate_idx < len(self.network_widget.nodes):
                    # Update node activation
                    x, y, _, node_type, _ = self.network_widget.nodes[gate_idx]
                    activation = state['output']
                    self.network_widget.nodes[gate_idx] = (x, y, activation, node_type, None)
                    
        except Exception as e:
            logging.error(f"Error handling gate change: {e}")
            
    def _on_health_change(self, health_status: dict):
        """Handle health status changes."""
        try:
            # Adjust visualization based on system health
            if 'stability' in health_status:
                stability = health_status['stability']
                # Adjust node radius based on stability
                self.network_widget.node_radius = max(10, min(20, 15 * stability))
                # Adjust connection width based on stability
                self.network_widget.connection_width = max(1, min(3, 2 * stability))
                
        except Exception as e:
            logging.error(f"Error handling health change: {e}")
            
    def _update_visualization(self):
        """Update the visualization with current data."""
        try:
            # Get backend info
            backend_info = self.network_widget.get_backend_info()
            
            # Update system state
            self.system_state.update_from_backend(backend_info)
            
            # Update visualization
            self.network_widget.update()
            
        except Exception as e:
            logging.error(f"Error updating visualization: {e}")
            
    def run(self):
        """Run the visualizer application."""
        try:
            # Initialize the visualizer
            asyncio.run(self.initialize())
            
            # Show the network widget
            self.network_widget.show()
            
            # Start the application event loop
            sys.exit(self.app.exec_())
            
        except Exception as e:
            logging.error(f"Error running visualizer: {e}")
            sys.exit(1)
            
    def cleanup(self):
        """Clean up resources."""
        try:
            self.update_timer.stop()
            self.network_widget.cleanup()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    launcher = VisualizerLauncher()
    launcher.run() 