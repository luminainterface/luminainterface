import sys
import socket
import json
import time
import threading
import logging
from typing import Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/node_connections.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NodeConnector")

class NodeConnector:
    def __init__(self, host: str = 'localhost', port: int = 5678):
        self.host = host
        self.port = port
        self.connections: Dict[str, socket.socket] = {}
        self.running = True
        
    def connect_node(self, node_name: str, node_version: str = '7.5') -> bool:
        """Connect a single node to the central node"""
        try:
            logger.info(f"Attempting to connect {node_name} to {self.host}:{self.port}")
            
            # Create socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)  # Set timeout for connection attempts
            
            # Try to connect
            try:
                sock.connect((self.host, self.port))
            except ConnectionRefusedError:
                logger.error(f"Connection refused for {node_name}. Is the central node running?")
                return False
            except socket.timeout:
                logger.error(f"Connection timeout for {node_name}")
                return False
                
            logger.info(f"Socket connected for {node_name}")
            
            # Register node
            registration = {
                'name': node_name,
                'version': node_version,
                'type': 'node',
                'timestamp': datetime.now().isoformat()
            }
            
            registration_data = json.dumps(registration).encode()
            logger.info(f"Sending registration data: {registration_data}")
            
            try:
                sock.send(registration_data)
            except Exception as e:
                logger.error(f"Failed to send registration data for {node_name}: {str(e)}")
                return False
                
            # Wait for response
            try:
                response = sock.recv(4096)
                if not response:
                    logger.error(f"No response received for {node_name}")
                    return False
                    
                logger.info(f"Received response for {node_name}: {response}")
                response_data = json.loads(response.decode())
                
                if response_data.get('status') == 'success':
                    self.connections[node_name] = sock
                    logger.info(f"Successfully connected {node_name} (v{node_version})")
                    return True
                else:
                    logger.error(f"Failed to connect {node_name}: {response_data.get('message')}")
                    sock.close()
                    return False
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response for {node_name}: {str(e)}")
                return False
            except socket.timeout:
                logger.error(f"Response timeout for {node_name}")
                return False
                
        except Exception as e:
            logger.error(f"Unexpected error connecting {node_name}: {str(e)}")
            return False
            
    def start_heartbeat(self, node_name: str):
        """Start sending heartbeats for a node"""
        while self.running and node_name in self.connections:
            try:
                heartbeat = {
                    'type': 'heartbeat',
                    'node': node_name,
                    'timestamp': datetime.now().isoformat()
                }
                self.connections[node_name].send(json.dumps(heartbeat).encode())
                logger.debug(f"Sent heartbeat for {node_name}")
                time.sleep(30)  # Send heartbeat every 30 seconds
            except Exception as e:
                logger.error(f"Heartbeat error for {node_name}: {str(e)}")
                break
                
    def connect_all_nodes(self):
        """Connect all supported node types"""
        nodes = [
            ('RSEN', '7.5'),
            ('HybridNode', '7.5'),
            ('NodeZero', '7.5'),
            ('PortalNode', '7.5'),
            ('WormholeNode', '7.5'),
            ('ZPENode', '7.5'),
            ('NeutrinoNode', '7.5'),
            ('GameTheoryNode', '7.5'),
            ('ConsciousnessNode', '7.5'),
            ('GaugeTheoryNode', '7.5'),
            ('FractalNodes', '7.5'),
            ('InfiniteMindsNode', '7.5'),
            ('VoidInfinityNode', '7.5')
        ]
        
        successful_connections = 0
        
        # Connect all nodes
        for node_name, version in nodes:
            if self.connect_node(node_name, version):
                successful_connections += 1
                # Start heartbeat thread for each successful connection
                threading.Thread(
                    target=self.start_heartbeat,
                    args=(node_name,),
                    daemon=True
                ).start()
                
        logger.info(f"Connected {successful_connections}/{len(nodes)} nodes to central node")
        
    def stop(self):
        """Stop all connections"""
        self.running = False
        for node_name, sock in self.connections.items():
            try:
                sock.close()
                logger.info(f"Closed connection for {node_name}")
            except Exception as e:
                logger.error(f"Error closing connection for {node_name}: {str(e)}")
                
def main():
    connector = NodeConnector()
    try:
        connector.connect_all_nodes()
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        connector.stop()
        
if __name__ == "__main__":
    main() 