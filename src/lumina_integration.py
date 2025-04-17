import logging
import socket
import json
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional
from queue import Queue, Empty
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PySide6.QtCore import Qt, QTimer

class NodeVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LUMINA Node Visualization")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Create graph
        self.graph = nx.DiGraph()
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.pos = nx.spring_layout(self.graph)
        
        # Setup animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start(1000)  # Update every second
        
    def update_visualization(self, node_data: Dict[str, Any] = None):
        """Update the node visualization"""
        if node_data:
            # Update graph with new data
            self.graph.clear()
            
            # Add central node
            self.graph.add_node("Central", type="central", status="active")
            
            # Add connected nodes
            for node_name, node_info in node_data.items():
                self.graph.add_node(node_name, **node_info)
                self.graph.add_edge(node_name, "Central")
            
            # Update positions
            self.pos = nx.spring_layout(self.graph)
            
            # Clear and redraw
            self.ax.clear()
            
            # Draw nodes
            node_colors = []
            for node in self.graph.nodes():
                if self.graph.nodes[node].get('type') == 'central':
                    node_colors.append('red')
                elif self.graph.nodes[node].get('status') == 'active':
                    node_colors.append('green')
                else:
                    node_colors.append('gray')
            
            nx.draw_networkx_nodes(self.graph, self.pos, node_color=node_colors, ax=self.ax)
            nx.draw_networkx_edges(self.graph, self.pos, ax=self.ax)
            nx.draw_networkx_labels(self.graph, self.pos, ax=self.ax)
            
            # Update status
            active_nodes = sum(1 for node in self.graph.nodes() 
                             if self.graph.nodes[node].get('status') == 'active')
            self.status_label.setText(f"Active Nodes: {active_nodes}")
            
            plt.draw()
            
    def closeEvent(self, event):
        """Handle window close event"""
        self.timer.stop()
        plt.close()
        event.accept()

class LUMINAIntegration:
    def __init__(self, host: str = 'localhost', port: int = 5678):
        self.logger = self._setup_logging()
        self.host = host
        self.port = port
        self.central_socket: Optional[socket.socket] = None
        self.hybrid_socket: Optional[socket.socket] = None
        self.running = True
        self.message_queue = Queue()
        self.processor_thread = None
        self.connections = {}
        self.system_state = {}
        self.last_heartbeat = {}
        self.connection_lock = threading.Lock()
        self.node_name = "LUMINA"
        self.heartbeat_interval = 10
        self.connection_timeout = 30
        self.initial_heartbeat_sent = False
        
        # Initialize visualization
        self.app = QApplication.instance() or QApplication([])
        self.visualizer = NodeVisualizer()
        self.visualizer.show()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('LUMINAIntegration')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
    def connect_to_nodes(self) -> bool:
        """Connect to central and hybrid nodes"""
        try:
            with self.connection_lock:
                # Connect to central node
                self.central_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.central_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.central_socket.settimeout(5.0)
                self.central_socket.connect((self.host, self.port))
                
                # Register as LUMINA node
                registration = {
                    'name': self.node_name,
                    'version': '7.5',
                    'type': 'processor',
                    'capabilities': ['data_processing', 'state_management', 'hybrid_integration'],
                    'timestamp': datetime.now().isoformat()
                }
                self.central_socket.send(json.dumps(registration).encode())
                
                # Wait for response with timeout
                self.central_socket.settimeout(5.0)
                response = self.central_socket.recv(4096)
                response_data = json.loads(response.decode())
                
                if response_data.get('status') != 'success':
                    self.logger.error("Failed to register with central node")
                    return False
                    
                self.logger.info("Successfully connected to central node")
                self.last_heartbeat['central'] = time.time()
                
                # Send initial heartbeat immediately
                self._send_initial_heartbeat()
                
                # Start message processor thread
                self.processor_thread = threading.Thread(target=self._process_messages, daemon=True)
                self.processor_thread.start()
                
                # Start heartbeat thread
                threading.Thread(target=self._send_heartbeats, daemon=True).start()
                
                # Start connection monitor thread
                threading.Thread(target=self._monitor_connections, daemon=True).start()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to connect to nodes: {str(e)}")
            return False
            
    def _send_initial_heartbeat(self):
        """Send initial heartbeat immediately after connection"""
        try:
            heartbeat = {
                'type': 'heartbeat',
                'name': self.node_name,
                'timestamp': datetime.now().isoformat(),
                'initial': True
            }
            
            with self.connection_lock:
                if self.central_socket:
                    self.central_socket.send(json.dumps(heartbeat).encode())
                    self.initial_heartbeat_sent = True
                    self.logger.debug("Sent initial heartbeat")
                    
        except Exception as e:
            self.logger.error(f"Error sending initial heartbeat: {str(e)}")
            
    def _process_messages(self):
        """Process incoming messages from nodes"""
        while self.running:
            try:
                with self.connection_lock:
                    if self.central_socket:
                        # Check for messages from central node
                        self.central_socket.settimeout(0.1)
                        try:
                            data = self.central_socket.recv(4096)
                            if data:
                                message = json.loads(data.decode())
                                self._handle_message(message, 'central')
                        except socket.timeout:
                            pass
                            
                # Process queued messages
                try:
                    message = self.message_queue.get_nowait()
                    self._handle_message(message['data'], message['source'])
                except Empty:
                    pass
                    
                time.sleep(0.01)  # Prevent CPU overuse
                
            except Exception as e:
                self.logger.error(f"Error processing messages: {str(e)}")
                
    def _monitor_connections(self):
        """Monitor connection health"""
        while self.running:
            try:
                current_time = time.time()
                
                with self.connection_lock:
                    # Check central node connection
                    if self.central_socket:
                        last_heartbeat = self.last_heartbeat.get('central', 0)
                        if current_time - last_heartbeat > self.connection_timeout:
                            self.logger.warning("Central node connection timeout, attempting reconnect...")
                            self._reconnect_central()
                            
                    # Check hybrid node connection
                    if self.hybrid_socket:
                        last_heartbeat = self.last_heartbeat.get('hybrid', 0)
                        if current_time - last_heartbeat > self.connection_timeout:
                            self.logger.warning("Hybrid node connection timeout, attempting reconnect...")
                            self._reconnect_hybrid()
                            
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring connections: {str(e)}")
                
    def _reconnect_central(self):
        """Reconnect to central node"""
        try:
            with self.connection_lock:
                if self.central_socket:
                    try:
                        self.central_socket.close()
                    except:
                        pass
                self.central_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.central_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.central_socket.settimeout(5.0)
                self.central_socket.connect((self.host, self.port))
                
                # Re-register after reconnection
                registration = {
                    'name': self.node_name,
                    'version': '7.5',
                    'type': 'processor',
                    'capabilities': ['data_processing', 'state_management', 'hybrid_integration'],
                    'timestamp': datetime.now().isoformat()
                }
                self.central_socket.send(json.dumps(registration).encode())
                
                # Wait for response with timeout
                self.central_socket.settimeout(5.0)
                response = self.central_socket.recv(4096)
                response_data = json.loads(response.decode())
                
                if response_data.get('status') == 'success':
                    self.last_heartbeat['central'] = time.time()
                    self._send_initial_heartbeat()  # Send initial heartbeat after reconnection
                    self.logger.info("Reconnected to central node")
                else:
                    self.logger.error("Failed to re-register with central node")
                    
        except Exception as e:
            self.logger.error(f"Failed to reconnect to central node: {str(e)}")
            
    def _reconnect_hybrid(self):
        """Reconnect to hybrid node"""
        try:
            with self.connection_lock:
                if self.hybrid_socket:
                    try:
                        self.hybrid_socket.close()
                    except:
                        pass
                self.hybrid_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.hybrid_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.hybrid_socket.settimeout(5.0)
                self.hybrid_socket.connect((self.host, self.port + 1))  # Hybrid node on next port
                self.last_heartbeat['hybrid'] = time.time()
                self.logger.info("Reconnected to hybrid node")
        except Exception as e:
            self.logger.error(f"Failed to reconnect to hybrid node: {str(e)}")
                
    def _handle_message(self, message: Dict[str, Any], source: str):
        """Handle incoming messages"""
        try:
            message_type = message.get('type')
            
            if message_type == 'data':
                # Process data based on source
                if source == 'central':
                    self._process_central_data(message.get('data', {}))
                elif source == 'hybrid':
                    self._process_hybrid_data(message.get('data', {}))
                    
            elif message_type == 'heartbeat_ack':
                self.last_heartbeat[source] = time.time()
                self.logger.debug(f"Received heartbeat acknowledgment from {source}")
                
            elif message_type == 'node_status':
                # Update visualization with node status
                self.visualizer.update_visualization(message.get('nodes', {}))
                
        except Exception as e:
            self.logger.error(f"Error handling message from {source}: {str(e)}")
            
    def _process_central_data(self, data: Dict[str, Any]):
        """Process data from central node"""
        try:
            # Extract relevant data
            node_data = data.get('node_data', {})
            system_state = data.get('system_state', {})
            
            # Update local state
            self._update_system_state(system_state)
            
            # Process node data
            processed_data = self._process_node_data(node_data)
            
            # Send processed data back to central node
            self.send_to_central({
                'type': 'processed_data',
                'data': processed_data,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Error processing central data: {str(e)}")
            
    def _process_hybrid_data(self, data: Dict[str, Any]):
        """Process data from hybrid node"""
        try:
            # Process hybrid-specific data
            hybrid_data = data.get('hybrid_data', {})
            
            # Perform hybrid data processing
            processed_data = self._process_hybrid_specific(hybrid_data)
            
            # Send processed data back
            self.send_to_hybrid({
                'type': 'processed_data',
                'data': processed_data,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Error processing hybrid data: {str(e)}")
            
    def _process_node_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process node data with LUMINA logic"""
        # Implement LUMINA-specific processing here
        return {
            'processed': True,
            'result': data,
            'lumina_timestamp': datetime.now().isoformat()
        }
        
    def _process_hybrid_specific(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process hybrid-specific data with LUMINA logic"""
        # Implement hybrid-specific processing here
        return {
            'processed': True,
            'hybrid_result': data,
            'lumina_timestamp': datetime.now().isoformat()
        }
        
    def _update_system_state(self, state: Dict[str, Any]):
        """Update local system state"""
        self.system_state = state
        self.logger.debug(f"Updated system state: {state}")
        
    def send_to_central(self, data: Dict[str, Any]):
        """Send data to central node"""
        try:
            with self.connection_lock:
                if self.central_socket:
                    self.central_socket.send(json.dumps(data).encode())
        except Exception as e:
            self.logger.error(f"Error sending to central node: {str(e)}")
            
    def send_to_hybrid(self, data: Dict[str, Any]):
        """Send data to hybrid node"""
        try:
            with self.connection_lock:
                if self.hybrid_socket:
                    self.hybrid_socket.send(json.dumps(data).encode())
        except Exception as e:
            self.logger.error(f"Error sending to hybrid node: {str(e)}")
            
    def _send_heartbeats(self):
        """Send periodic heartbeats to nodes"""
        while self.running:
            try:
                heartbeat = {
                    'type': 'heartbeat',
                    'name': self.node_name,
                    'timestamp': datetime.now().isoformat()
                }
                
                with self.connection_lock:
                    if self.central_socket:
                        self.central_socket.send(json.dumps(heartbeat).encode())
                        
                    if self.hybrid_socket:
                        self.hybrid_socket.send(json.dumps(heartbeat).encode())
                        
                time.sleep(self.heartbeat_interval)  # Use configurable interval
                
            except Exception as e:
                self.logger.error(f"Error sending heartbeats: {str(e)}")
                
    def stop(self):
        """Stop the integration"""
        self.running = False
        
        with self.connection_lock:
            if self.central_socket:
                try:
                    self.central_socket.close()
                except:
                    pass
                    
            if self.hybrid_socket:
                try:
                    self.hybrid_socket.close()
                except:
                    pass
                    
        self.logger.info("LUMINA integration stopped")

def main():
    """Main entry point for LUMINA integration"""
    integration = LUMINAIntegration()
    
    try:
        if integration.connect_to_nodes():
            # Keep the main thread alive
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        integration.logger.info("Received shutdown signal")
    finally:
        integration.stop()
        
if __name__ == "__main__":
    main() 