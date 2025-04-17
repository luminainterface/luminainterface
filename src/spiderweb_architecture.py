"""
Spiderweb Architecture for Lumina Neural Network

This module implements the Spiderweb architecture for connecting different versions
of the Lumina Neural Network system. It provides a flexible, decentralized way to
manage connections and data flow between versions.
"""

from typing import Dict, Any, List, Optional, Callable
import logging
import threading
from queue import Queue
from datetime import datetime

class SpiderwebNode:
    """Represents a node in the Spiderweb architecture"""
    
    def __init__(self, version_id: str, system: Any):
        self.version_id = version_id
        self.system = system
        self.connections: List['SpiderwebNode'] = []
        self.message_queue = Queue()
        self.handlers: Dict[str, Callable] = {}
        self.active = True
        
    def connect(self, node: 'SpiderwebNode') -> bool:
        """Connect to another node"""
        if node not in self.connections:
            self.connections.append(node)
            return True
        return False
        
    def disconnect(self, node: 'SpiderwebNode') -> bool:
        """Disconnect from a node"""
        if node in self.connections:
            self.connections.remove(node)
            return True
        return False
        
    def send_message(self, message: Dict[str, Any]) -> bool:
        """Send a message to all connected nodes"""
        try:
            for node in self.connections:
                node.message_queue.put({
                    **message,
                    'source': self.version_id,
                    'timestamp': datetime.now().isoformat()
                })
            return True
        except Exception as e:
            logging.error(f"Error sending message from {self.version_id}: {str(e)}")
            return False
            
    def process_messages(self):
        """Process messages in the queue"""
        while self.active:
            try:
                message = self.message_queue.get(timeout=1)
                handler = self.handlers.get(message.get('type'))
                if handler:
                    handler(message)
            except Exception as e:
                logging.error(f"Error processing message in {self.version_id}: {str(e)}")

class SpiderwebArchitecture:
    """Main class implementing the Spiderweb architecture"""
    
    def __init__(self):
        self.nodes: Dict[str, SpiderwebNode] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.logger = logging.getLogger('SpiderwebArchitecture')
        
    def add_node(self, version_id: str, system: Any) -> bool:
        """Add a new node to the architecture"""
        if version_id in self.nodes:
            self.logger.warning(f"Node {version_id} already exists")
            return False
            
        node = SpiderwebNode(version_id, system)
        self.nodes[version_id] = node
        
        # Start message processing thread
        thread = threading.Thread(target=node.process_messages, daemon=True)
        thread.start()
        self.threads[version_id] = thread
        
        self.logger.info(f"Added node {version_id}")
        return True
        
    def remove_node(self, version_id: str) -> bool:
        """Remove a node from the architecture"""
        if version_id not in self.nodes:
            return False
            
        node = self.nodes[version_id]
        node.active = False
        
        # Disconnect from all other nodes
        for other_node in list(node.connections):
            node.disconnect(other_node)
            other_node.disconnect(node)
            
        # Stop processing thread
        if version_id in self.threads:
            self.threads[version_id].join(timeout=1)
            del self.threads[version_id]
            
        del self.nodes[version_id]
        self.logger.info(f"Removed node {version_id}")
        return True
        
    def connect_nodes(self, version_id1: str, version_id2: str) -> bool:
        """Connect two nodes"""
        if version_id1 not in self.nodes or version_id2 not in self.nodes:
            self.logger.error(f"One or both nodes not found: {version_id1}, {version_id2}")
            return False
            
        node1 = self.nodes[version_id1]
        node2 = self.nodes[version_id2]
        
        success1 = node1.connect(node2)
        success2 = node2.connect(node1)
        
        if success1 and success2:
            self.logger.info(f"Connected nodes {version_id1} and {version_id2}")
            return True
        return False
        
    def register_handler(self, version_id: str, message_type: str, handler: Callable) -> bool:
        """Register a message handler for a node"""
        if version_id not in self.nodes:
            return False
            
        self.nodes[version_id].handlers[message_type] = handler
        self.logger.info(f"Registered handler for {message_type} in {version_id}")
        return True
        
    def broadcast_message(self, source_version: str, message: Dict[str, Any]) -> bool:
        """Broadcast a message from a node"""
        if source_version not in self.nodes:
            return False
            
        return self.nodes[source_version].send_message(message)
        
    def get_node_status(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a node"""
        if version_id not in self.nodes:
            return None
            
        node = self.nodes[version_id]
        return {
            'version_id': version_id,
            'connections': [n.version_id for n in node.connections],
            'active': node.active,
            'message_queue_size': node.message_queue.qsize(),
            'handlers': list(node.handlers.keys())
        }
        
    def get_architecture_status(self) -> Dict[str, Any]:
        """Get status of the entire architecture"""
        return {
            'nodes': {version_id: self.get_node_status(version_id) 
                     for version_id in self.nodes},
            'total_nodes': len(self.nodes),
            'total_connections': sum(len(node.connections) for node in self.nodes.values()) // 2
        } 