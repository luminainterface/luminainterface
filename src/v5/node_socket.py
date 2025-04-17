"""
NodeSocket - Core socket interface for node communication

This module provides the socket interface used by V5 plugins for
node-to-node communication and message passing.
"""

import threading
import time
import uuid
import json
import logging
from queue import Queue

# Import Qt compatibility layer
from .ui.qt_compat import QtCore, Signal, Slot, QtCompat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NodeSocket:
    """Core socket interface for node communication"""
    
    def __init__(self, node_id, socket_type="service"):
        """
        Initialize a node socket
        
        Args:
            node_id: Unique identifier for this socket
            socket_type: Type of socket (service, ui, integration)
        """
        self.node_id = node_id
        self.socket_type = socket_type
        self.message_queue = Queue()
        self.subscribers = []
        self.message_handlers = {}
        self.response_handlers = {}
        self._lock = threading.RLock()
        
        logger.info(f"Initialized NodeSocket: {node_id} (type: {socket_type})")
        
    def connect_to(self, target_socket):
        """
        Connect to another socket bidirectionally
        
        Args:
            target_socket: The target NodeSocket to connect to
        """
        with self._lock:
            if target_socket not in self.subscribers:
                self.subscribers.append(target_socket)
                logger.info(f"Socket {self.node_id} connected to {target_socket.node_id}")
                
                # Establish bidirectional connection if not already connected
                if self not in target_socket.subscribers:
                    target_socket.connect_to(self)
    
    def send_message(self, message):
        """
        Send message to all subscribers
        
        Args:
            message: Dictionary containing the message data
        """
        if not isinstance(message, dict):
            raise ValueError("Message must be a dictionary")
            
        # Add default fields
        message.setdefault("source", self.node_id)
        message.setdefault("timestamp", time.time())
        message.setdefault("id", str(uuid.uuid4()))
        
        # Log message details
        logger.info(f"Socket {self.node_id} sending message type: {message.get('type')} to {len(self.subscribers)} subscribers")
        logger.debug(f"Message contents: {message}")
        
        # Send to all subscribers
        with self._lock:
            sent_count = 0
            for subscriber in self.subscribers:
                subscriber.receive_message(message)
                sent_count += 1
                
            logger.info(f"Socket {self.node_id} sent message to {sent_count} subscribers")
    
    def receive_message(self, message):
        """
        Receive message from another socket
        
        Args:
            message: Dictionary containing the message data
        """
        # Log received message
        logger.info(f"Socket {self.node_id} received message type: {message.get('type')} from {message.get('source')}")
        
        # Add to queue
        self.message_queue.put(message)
        
        # Process message if handler exists
        message_type = message.get("type")
        if message_type in self.message_handlers:
            try:
                logger.info(f"Socket {self.node_id} handling message type: {message_type}")
                self.message_handlers[message_type](message)
                logger.debug(f"Socket {self.node_id} handled message type: {message_type}")
            except Exception as e:
                logger.error(f"Error handling message of type {message_type}: {str(e)}", exc_info=True)
        else:
            logger.warning(f"Socket {self.node_id} has no handler for message type: {message_type}")
        
        # Check if this is a response to a request
        request_id = message.get("request_id")
        if request_id and request_id in self.response_handlers:
            handler = self.response_handlers.pop(request_id)
            try:
                logger.info(f"Socket {self.node_id} handling response for request: {request_id}")
                handler(message)
                logger.debug(f"Socket {self.node_id} handled response for request: {request_id}")
            except Exception as e:
                logger.error(f"Error handling response for request {request_id}: {str(e)}", exc_info=True)
    
    def register_response_handler(self, request_id, handler):
        """
        Register a handler for a specific response
        
        Args:
            request_id: The ID of the request to handle the response for
            handler: Callback function to handle the response
        """
        self.response_handlers[request_id] = handler
        logger.debug(f"Socket {self.node_id} registered handler for request: {request_id}")
    
    def register_message_handler(self, message_type, handler):
        """
        Register a handler for a specific message type
        
        Args:
            message_type: The type of message to handle
            handler: Callback function to handle the message
        """
        self.message_handlers[message_type] = handler
        logger.debug(f"Socket {self.node_id} registered handler for message type: {message_type}")
    
    def get_message_count(self):
        """Return the number of messages in the queue"""
        return self.message_queue.qsize()
    
    def get_next_message(self, block=False, timeout=None):
        """
        Get the next message from the queue
        
        Args:
            block: Whether to block until a message is available
            timeout: Timeout in seconds if blocking
            
        Returns:
            Message dictionary or None if no message is available
        """
        try:
            return self.message_queue.get(block=block, timeout=timeout)
        except Queue.Empty:
            return None


class QtSocketAdapter:
    """Adapter to connect NodeSocket to Qt signals/slots"""
    
    def __init__(self, socket):
        """
        Initialize the adapter
        
        Args:
            socket: The NodeSocket to adapt
        """
        from PySide6.QtCore import QObject, Signal, Slot, QTimer
        
        # Create a QObject to emit signals
        class SignalEmitter(QObject):
            messageReceived = Signal(dict)
            
        self.socket = socket
        self.emitter = SignalEmitter()
        self.messageReceived = self.emitter.messageReceived
        
        # Set up a timer to check for messages
        self.timer = QTimer()
        self.timer.timeout.connect(self._check_messages)
        self.timer.start(50)  # Check every 50ms
        
    def _check_messages(self):
        """Check for new messages and emit signals"""
        while self.socket.get_message_count() > 0:
            message = self.socket.get_next_message()
            if message:
                self.emitter.messageReceived.emit(message)


class WebSocketServer:
    """WebSocket server for real-time updates"""
    
    def __init__(self, host="localhost", port=8765):
        """
        Initialize the WebSocket server
        
        Args:
            host: Host to bind to
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self.clients = set()
        self.server = None
        self.thread = None
        self.running = False
        self._lock = threading.RLock()
        
    def start(self):
        """Start the WebSocket server in a background thread"""
        if self.running:
            return
            
        try:
            import asyncio
            import websockets
            
            # Define the server coroutine
            async def server_coroutine():
                async def handler(websocket, path):
                    # Register client
                    with self._lock:
                        self.clients.add(websocket)
                    
                    try:
                        # Keep the connection open
                        while True:
                            await asyncio.sleep(1)
                    except websockets.exceptions.ConnectionClosed:
                        pass
                    finally:
                        # Unregister client
                        with self._lock:
                            self.clients.remove(websocket)
                
                # Start the server
                self.server = await websockets.serve(handler, self.host, self.port)
                logger.info(f"WebSocket server started at ws://{self.host}:{self.port}")
                
                # Keep the server running
                while self.running:
                    await asyncio.sleep(1)
            
            # Set up the event loop in a new thread
            def run_server():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self.running = True
                loop.run_until_complete(server_coroutine())
                loop.close()
            
            # Start the thread
            self.thread = threading.Thread(target=run_server)
            self.thread.daemon = True
            self.thread.start()
            
        except ImportError:
            logger.error("WebSocket server requires 'websockets' package. Install with: pip install websockets")
            raise
    
    def stop(self):
        """Stop the WebSocket server"""
        self.running = False
        if self.server:
            self.server.close()
        if self.thread:
            self.thread.join(timeout=5)
    
    def broadcast(self, message):
        """
        Broadcast a message to all connected clients
        
        Args:
            message: String or JSON-serializable object to broadcast
        """
        if not self.running:
            logger.warning("Cannot broadcast: WebSocket server not running")
            return
        
        # Convert to string if necessary
        if not isinstance(message, str):
            message = json.dumps(message)
        
        # Send to all clients
        with self._lock:
            if not self.clients:
                logger.debug("No WebSocket clients connected")
                return
                
            import asyncio
            import websockets
            
            # Create tasks to send to each client
            tasks = []
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            for client in self.clients:
                tasks.append(asyncio.ensure_future(client.send(message)))
            
            # Run the tasks
            if tasks:
                try:
                    loop.run_until_complete(asyncio.gather(*tasks))
                    logger.debug(f"Broadcast message to {len(tasks)} WebSocket clients")
                except websockets.exceptions.ConnectionClosed:
                    logger.debug("Some clients disconnected during broadcast")
                finally:
                    loop.close()


class WebSocketClient:
    """WebSocket client for real-time updates"""
    
    def __init__(self, endpoint):
        """
        Initialize the WebSocket client
        
        Args:
            endpoint: WebSocket endpoint URL
        """
        from PySide6.QtCore import QObject, Signal, Slot
        from PySide6.QtWebSockets import QWebSocket
        
        # Create a QObject to emit signals
        class SignalEmitter(QObject):
            messageReceived = Signal(str)
            connected = Signal()
            disconnected = Signal()
            
        self.endpoint = endpoint
        self.websocket = QWebSocket()
        self.emitter = SignalEmitter()
        self.messageReceived = self.emitter.messageReceived
        self.connected = self.emitter.connected
        self.disconnected = self.emitter.disconnected
        
        # Connect signals
        self.websocket.textMessageReceived.connect(self.emitter.messageReceived)
        self.websocket.connected.connect(self.emitter.connected)
        self.websocket.disconnected.connect(self.emitter.disconnected)
        
    def connect(self):
        """Connect to the WebSocket endpoint"""
        from PySide6.QtCore import QUrl
        self.websocket.open(QUrl(self.endpoint))
        
    def disconnect(self):
        """Disconnect from the WebSocket endpoint"""
        self.websocket.close()
        
    def send_message(self, message):
        """
        Send a message to the WebSocket server
        
        Args:
            message: String or JSON-serializable object to send
        """
        if not isinstance(message, str):
            message = json.dumps(message)
        self.websocket.sendTextMessage(message) 