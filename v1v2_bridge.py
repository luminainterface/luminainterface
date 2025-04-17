#!/usr/bin/env python
"""
V1V2 Bridge - Connects v1 text interface with v2 graphical interface
for the Lumina Neural Network System
"""

import os
import sys
import time
import json
import logging
import threading
from queue import Queue
from typing import Dict, Any, List, Callable, Optional

# Configure logging
logger = logging.getLogger("V1V2-Bridge")

class V1V2Bridge:
    """
    Bridge between v1 text interface and v2 graphical interface.
    
    This class provides seamless communication between the text-based interface (v1)
    and the graphical interface (v2) of the Lumina Neural Network System.
    """
    
    def __init__(self, mock_mode=False):
        """
        Initialize the V1V2 Bridge
        
        Args:
            mock_mode (bool): Whether to use simulated/mock data
        """
        self.mock_mode = mock_mode
        self.v1_connected = False
        self.v2_connected = False
        self.message_handlers = {}
        self.message_queue = Queue()
        self.running = False
        self.processing_thread = None
        
        logger.info(f"Initialized V1V2 Bridge (mock_mode={mock_mode})")
    
    def initialize(self) -> bool:
        """
        Initialize the bridge components
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Load configuration
            self._load_config()
            
            # Start message processing thread
            self._start_message_processing()
            
            if self.mock_mode:
                logger.info("Running in mock mode with simulated interfaces")
                self.v1_connected = True
                self.v2_connected = True
            else:
                # Connect to v1 text interface
                self.v1_connected = self._connect_to_v1()
                
                # Connect to v2 graphical interface
                self.v2_connected = self._connect_to_v2()
            
            return self.v1_connected and self.v2_connected
            
        except Exception as e:
            logger.error(f"Error initializing V1V2 Bridge: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def _load_config(self):
        """Load bridge configuration"""
        self.config = {
            "v1_host": "localhost",
            "v1_port": 5001,
            "v2_host": "localhost",
            "v2_port": 5002,
            "message_timeout": 5.0,  # seconds
            "reconnect_interval": 30.0,  # seconds
            "max_reconnect_attempts": 5
        }
        
        # Try to load from file
        config_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "config",
            "v1v2_bridge.json"
        )
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    self.config.update(file_config)
                    logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.warning(f"Error loading configuration from {config_file}: {e}")
    
    def _connect_to_v1(self) -> bool:
        """
        Connect to v1 text interface
        
        Returns:
            bool: True if connection successful
        """
        if self.mock_mode:
            return True
            
        logger.info("Connecting to v1 text interface...")
        
        try:
            # In a real implementation, this would establish a connection
            # to the v1 text interface (e.g., socket, pipe, etc.)
            
            # For now, simulate connection
            time.sleep(0.5)
            
            logger.info("Connected to v1 text interface")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to v1 text interface: {e}")
            return False
    
    def _connect_to_v2(self) -> bool:
        """
        Connect to v2 graphical interface
        
        Returns:
            bool: True if connection successful
        """
        if self.mock_mode:
            return True
            
        logger.info("Connecting to v2 graphical interface...")
        
        try:
            # In a real implementation, this would establish a connection
            # to the v2 graphical interface (e.g., socket, IPC, etc.)
            
            # For now, simulate connection
            time.sleep(0.5)
            
            logger.info("Connected to v2 graphical interface")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to v2 graphical interface: {e}")
            return False
    
    def _start_message_processing(self):
        """Start the message processing thread"""
        if self.processing_thread is not None and self.processing_thread.is_alive():
            logger.warning("Message processing thread already running")
            return
            
        self.running = True
        self.processing_thread = threading.Thread(
            target=self._process_messages,
            daemon=True,
            name="V1V2-Bridge-Thread"
        )
        self.processing_thread.start()
        logger.info("Started message processing thread")
    
    def _stop_message_processing(self):
        """Stop the message processing thread"""
        self.running = False
        if self.processing_thread is not None:
            try:
                self.processing_thread.join(timeout=2.0)
                logger.info("Stopped message processing thread")
            except:
                logger.warning("Failed to join message processing thread")
    
    def _process_messages(self):
        """Process messages from the queue"""
        while self.running:
            try:
                if self.message_queue.empty():
                    time.sleep(0.05)  # Prevent CPU spinning
                    continue
                    
                message = self.message_queue.get(block=False)
                self._handle_message(message)
                self.message_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    def _handle_message(self, message: Dict[str, Any]):
        """
        Handle a message from the queue
        
        Args:
            message: Message dictionary
        """
        source = message.get("source", "unknown")
        msg_type = message.get("type", "unknown")
        data = message.get("data", {})
        
        logger.debug(f"Handling message: {msg_type} from {source}")
        
        # Check for registered handlers
        handler_key = f"{source}:{msg_type}"
        if handler_key in self.message_handlers:
            try:
                handler = self.message_handlers[handler_key]
                handler(message)
            except Exception as e:
                logger.error(f"Error in message handler for {handler_key}: {e}")
        else:
            # Default processing based on source and type
            if source == "v1":
                # Forward v1 message to v2
                self._forward_to_v2(msg_type, data)
            elif source == "v2":
                # Forward v2 message to v1
                self._forward_to_v1(msg_type, data)
            else:
                logger.warning(f"Unhandled message from {source}: {msg_type}")
    
    def _forward_to_v1(self, msg_type: str, data: Dict[str, Any]):
        """
        Forward a message to v1 text interface
        
        Args:
            msg_type: Message type
            data: Message data
        """
        if not self.v1_connected and not self.mock_mode:
            logger.warning(f"Cannot forward {msg_type} to v1: Not connected")
            return
            
        logger.debug(f"Forwarding {msg_type} to v1")
        
        if self.mock_mode:
            logger.info(f"[MOCK] Forwarded {msg_type} to v1: {data}")
            return
            
        # In a real implementation, this would send the message to v1
        # Example: self.v1_socket.send(json.dumps({"type": msg_type, "data": data}))
    
    def _forward_to_v2(self, msg_type: str, data: Dict[str, Any]):
        """
        Forward a message to v2 graphical interface
        
        Args:
            msg_type: Message type
            data: Message data
        """
        if not self.v2_connected and not self.mock_mode:
            logger.warning(f"Cannot forward {msg_type} to v2: Not connected")
            return
            
        logger.debug(f"Forwarding {msg_type} to v2")
        
        if self.mock_mode:
            logger.info(f"[MOCK] Forwarded {msg_type} to v2: {data}")
            return
            
        # In a real implementation, this would send the message to v2
        # Example: self.v2_socket.send(json.dumps({"type": msg_type, "data": data}))
    
    def register_message_handler(self, source: str, msg_type: str, handler: Callable):
        """
        Register a handler for a specific message type from a specific source
        
        Args:
            source: Source of the message (e.g., "v1", "v2")
            msg_type: Type of the message
            handler: Handler function
        """
        handler_key = f"{source}:{msg_type}"
        self.message_handlers[handler_key] = handler
        logger.debug(f"Registered handler for {handler_key}")
    
    def send_to_v1v2(self, msg_type: str, data: Dict[str, Any]):
        """
        Send a message to both v1 and v2 interfaces
        
        Args:
            msg_type: Message type
            data: Message data
        """
        logger.debug(f"Sending {msg_type} to v1v2")
        
        # Send to v1
        self._forward_to_v1(msg_type, data)
        
        # Send to v2
        self._forward_to_v2(msg_type, data)
        
        return True
    
    def send_to_v1(self, msg_type: str, data: Dict[str, Any]):
        """
        Send a message to v1 text interface
        
        Args:
            msg_type: Message type
            data: Message data
        """
        return self._forward_to_v1(msg_type, data)
    
    def send_to_v2(self, msg_type: str, data: Dict[str, Any]):
        """
        Send a message to v2 graphical interface
        
        Args:
            msg_type: Message type
            data: Message data
        """
        return self._forward_to_v2(msg_type, data)
    
    def reconnect(self):
        """Attempt to reconnect if disconnected"""
        if not self.v1_connected:
            self.v1_connected = self._connect_to_v1()
        
        if not self.v2_connected:
            self.v2_connected = self._connect_to_v2()
        
        return self.v1_connected and self.v2_connected
    
    def stop(self):
        """Stop the bridge and clean up resources"""
        logger.info("Stopping V1V2 Bridge")
        
        self._stop_message_processing()
        
        # Close connections
        self.v1_connected = False
        self.v2_connected = False
        
        logger.info("V1V2 Bridge stopped")
    
    def get_status(self):
        """
        Get the status of the bridge
        
        Returns:
            dict: Status information
        """
        return {
            "connected": self.v1_connected and self.v2_connected,
            "v1_connected": self.v1_connected,
            "v2_connected": self.v2_connected,
            "mock_mode": self.mock_mode,
            "message_handlers": len(self.message_handlers),
            "queue_size": self.message_queue.qsize()
        }

# Test code
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create bridge
    bridge = V1V2Bridge(mock_mode=True)
    
    # Initialize
    success = bridge.initialize()
    print(f"Bridge initialized: {success}")
    
    # Test sending messages
    bridge.send_to_v1("test_message", {"text": "Hello from v2"})
    bridge.send_to_v2("test_message", {"text": "Hello from v1"})
    
    # Get status
    status = bridge.get_status()
    print(f"Bridge status: {json.dumps(status, indent=2)}")
    
    # Sleep briefly to allow messages to be processed
    time.sleep(1)
    
    # Stop
    bridge.stop()
    print("Test complete") 