import os
import sys
import logging
import argparse
import signal
import time
from typing import Dict, Any, Optional
from pathlib import Path
import json
import socket
import threading
from datetime import datetime
import sqlite3
from queue import Queue
from threading import Lock

# Add src directory to path
src_path = os.path.dirname(os.path.abspath(__file__))
if src_path not in sys.path:
    sys.path.append(src_path)

from central_node import CentralNode
from version_bridge_integration import VersionBridgeIntegration

class CentralNodeLauncher:
    def __init__(self, config_path: str = "config/central_node_config.json"):
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.central_node = None
        self.running = True
        self.version_bridge = None
        self.connections = {}
        self.node_registry = {}
        self.database_connections = {}
        self.database_lock = Lock()
        self.database_queue = Queue()
        self.database_thread = threading.Thread(target=self._database_worker, daemon=True)
        self.database_thread.start()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        logger = logging.getLogger('CentralNodeLauncher')
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(
            f"logs/central_node_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            return {
                'port': 5678,
                'max_connections': 100,
                'version': '7.5',
                'database': {
                    'type': 'sqlite',
                    'path': 'data/central_node.db'
                },
                'nodes': {
                    'allowed_versions': ['1', '2', '5', '6', '7', '7.5', '8', '9', '10'],
                    'max_retries': 3,
                    'timeout': 30
                }
            }
            
    def _database_worker(self):
        """Worker thread for handling database operations"""
        conn = sqlite3.connect('data/central_node.db')
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                name TEXT PRIMARY KEY,
                version TEXT,
                status TEXT,
                last_heartbeat TIMESTAMP
            )
        ''')
        conn.commit()
        
        while self.running:
            try:
                # Get operation from queue
                operation = self.database_queue.get(timeout=1)
                if operation is None:
                    continue
                    
                query = operation['query']
                params = operation.get('params', ())
                callback = operation.get('callback')
                
                # Execute query
                cursor.execute(query, params)
                conn.commit()
                
                # Call callback if provided
                if callback:
                    callback(cursor.fetchall())
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Database worker error: {str(e)}")
                
        conn.close()
            
    def initialize_central_node(self):
        """Initialize the central node and its components"""
        try:
            self.central_node = CentralNode()
            self.version_bridge = VersionBridgeIntegration(self.central_node)
            self.logger.info("Central node initialized successfully")
            
            # Initialize database connections
            self._initialize_databases()
            
            # Start connection listener
            self._start_connection_listener()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize central node: {str(e)}")
            raise
            
    def _initialize_databases(self):
        """Initialize database connections"""
        try:
            db_config = self.config.get('database', {})
            db_type = db_config.get('type', 'sqlite')
            
            if db_type == 'sqlite':
                import sqlite3
                db_path = db_config.get('path', 'data/central_node.db')
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                self.database_connections['main'] = sqlite3.connect(db_path)
                self.logger.info(f"Connected to SQLite database at {db_path}")
                
                # Initialize database schema
                self._initialize_database_schema()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize databases: {str(e)}")
            raise
            
    def _initialize_database_schema(self):
        """Initialize database schema"""
        try:
            conn = self.database_connections['main']
            cursor = conn.cursor()
            
            # Create nodes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    last_heartbeat TIMESTAMP,
                    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create data_logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id INTEGER,
                    data_type TEXT NOT NULL,
                    data_content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (node_id) REFERENCES nodes (id)
                )
            ''')
            
            # Create version_logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS version_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_version TEXT NOT NULL,
                    target_version TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            self.logger.info("Database schema initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database schema: {str(e)}")
            raise
            
    def _start_connection_listener(self):
        """Start listening for node connections"""
        try:
            port = self.config.get('port', 5678)
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.bind(('0.0.0.0', port))
            self.server_socket.listen(self.config.get('max_connections', 100))
            
            self.logger.info(f"Listening for node connections on port {port}")
            
            # Start connection handler thread
            self.connection_thread = threading.Thread(
                target=self._handle_connections,
                daemon=True
            )
            self.connection_thread.start()
            
        except Exception as e:
            self.logger.error(f"Failed to start connection listener: {str(e)}")
            raise
            
    def _handle_connections(self):
        """Handle incoming node connections"""
        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                self.logger.info(f"New connection from {address}")
                
                # Start client handler thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
                
            except Exception as e:
                if self.running:  # Only log if we're still running
                    self.logger.error(f"Error handling connection: {str(e)}")
                    
    def _handle_client(self, client_socket: socket.socket, address: tuple):
        """Handle individual client connection"""
        try:
            # Receive node registration
            data = client_socket.recv(4096)
            if not data:
                return
                
            registration = json.loads(data.decode())
            node_name = registration.get('name')
            node_version = registration.get('version')
            
            # Validate node version
            if not self.version_bridge.validate_version_compatibility(node_version):
                client_socket.send(json.dumps({
                    'status': 'error',
                    'message': f'Unsupported version: {node_version}'
                }).encode())
                return
                
            # Register node
            if self._register_node(node_name, node_version, client_socket, address):
                # Send registration confirmation
                client_socket.send(json.dumps({
                    'status': 'success',
                    'message': 'Node registered successfully',
                    'central_node_version': '7.5'
                }).encode())
                
                # Handle node communication
                while self.running:
                    data = client_socket.recv(4096)
                    if not data:
                        break
                        
                    message = json.loads(data.decode())
                    self._process_node_message(node_name, message)
            else:
                client_socket.send(json.dumps({
                    'status': 'error',
                    'message': 'Failed to register node'
                }).encode())
                
        except Exception as e:
            self.logger.error(f"Error handling client {address}: {str(e)}")
        finally:
            self._unregister_node(node_name)
            client_socket.close()
            
    def _register_node(self, node_name: str, node_version: str, 
                      socket: socket.socket, address: tuple):
        """Register a new node"""
        try:
            # Store connection
            self.connections[node_name] = {
                'socket': socket,
                'address': address,
                'version': node_version,
                'last_heartbeat': datetime.now()
            }
            
            # Queue database operation
            self.database_queue.put({
                'query': '''
                    INSERT OR REPLACE INTO nodes (name, version, status, last_heartbeat)
                    VALUES (?, ?, ?, datetime('now'))
                ''',
                'params': (node_name, node_version, 'active')
            })
            
            self.logger.info(f"Registered node {node_name} (v{node_version})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register node {node_name}: {str(e)}")
            return False
            
    def _unregister_node(self, node_name: str):
        """Unregister a node"""
        try:
            if node_name in self.connections:
                del self.connections[node_name]
                
            # Queue database operation
            self.database_queue.put({
                'query': '''
                    UPDATE nodes
                    SET status = 'inactive'
                    WHERE name = ?
                ''',
                'params': (node_name,)
            })
            
            self.logger.info(f"Unregistered node {node_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister node {node_name}: {str(e)}")
            return False
            
    def _process_node_message(self, node_name: str, message: Dict[str, Any]):
        """Process message from a node"""
        try:
            message_type = message.get('type')
            data = message.get('data', {})
            
            if message_type == 'data':
                # Process data through version bridge
                node_version = self.connections[node_name]['version']
                processed_data = self.version_bridge.process_with_version(data, node_version)
                
                # Log data
                self._log_data(node_name, processed_data)
                
                # Send response
                self.connections[node_name]['socket'].send(json.dumps({
                    'status': 'success',
                    'data': processed_data
                }).encode())
                
            elif message_type == 'heartbeat':
                # Update last heartbeat
                self.connections[node_name]['last_heartbeat'] = datetime.now()
                
                # Queue database operation
                self.database_queue.put({
                    'query': '''
                        UPDATE nodes
                        SET last_heartbeat = datetime('now')
                        WHERE name = ?
                    ''',
                    'params': (node_name,)
                })
                
                # Send acknowledgment
                self.connections[node_name]['socket'].send(json.dumps({
                    'status': 'success',
                    'message': 'heartbeat received'
                }).encode())
                
        except Exception as e:
            self.logger.error(f"Error processing message from {node_name}: {str(e)}")
            
    def _log_data(self, node_name: str, data: Dict[str, Any]):
        """Log data to database"""
        try:
            conn = self.database_connections['main']
            cursor = conn.cursor()
            
            # Get node ID
            cursor.execute('SELECT id FROM nodes WHERE name = ?', (node_name,))
            node_id = cursor.fetchone()[0]
            
            # Log data
            cursor.execute('''
                INSERT INTO data_logs (node_id, data_type, data_content)
                VALUES (?, ?, ?)
            ''', (node_id, 'processed', json.dumps(data)))
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to log data from {node_name}: {str(e)}")
            
    def start(self):
        """Start the central node launcher"""
        try:
            self.running = True
            self.initialize_central_node()
            
            self.logger.info("Central node launcher started successfully")
            
            # Keep main thread alive
            while self.running:
                time.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Error starting central node launcher: {str(e)}")
            self.stop()
            
    def stop(self):
        """Stop the central node launcher"""
        try:
            self.running = False
            
            # Close all connections
            for node_name in list(self.connections.keys()):
                self._unregister_node(node_name)
                
            # Close database connections
            for conn in self.database_connections.values():
                conn.close()
                
            self.logger.info("Central node launcher stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping central node launcher: {str(e)}")
            
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received shutdown signal {signum}")
        self.stop()
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='Central Node Launcher')
    parser.add_argument('--config', type=str, default='config/central_node_config.json',
                      help='Path to configuration file')
    parser.add_argument('--port', type=int, help='Override port from config')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Create config directory if it doesn't exist
    os.makedirs(os.path.dirname(args.config), exist_ok=True)
    
    # Initialize launcher
    launcher = CentralNodeLauncher(args.config)
    
    # Override port if specified
    if args.port:
        launcher.config['port'] = args.port
        
    # Set debug logging if specified
    if args.debug:
        launcher.logger.setLevel(logging.DEBUG)
        
    try:
        launcher.start()
    except KeyboardInterrupt:
        launcher.stop()
    except Exception as e:
        launcher.logger.error(f"Fatal error: {str(e)}")
        launcher.stop()
        sys.exit(1)

if __name__ == "__main__":
    main() 