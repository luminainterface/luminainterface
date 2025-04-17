#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Connection Recovery Manager for LUMINA V7 Dashboard
==================================================

Simplified implementation of connection recovery mechanisms.
"""

import socket
import logging
import threading
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any, Callable

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ConnectionRecovery")

class ConnectionRecoveryManager:
    """
    Manages connection recovery between dashboard and LUMINA V7 system
    
    Features:
    - Automatic reconnection with exponential backoff
    - Connection health monitoring
    - Fallback to mock mode
    - Connection event callbacks
    """
    
    # Connection states
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"
    MOCK_MODE = "mock_mode"
    
    def __init__(self, host: str = "localhost", port: int = 5678,
                auto_reconnect: bool = True, max_retries: int = 5,
                initial_retry_delay: float = 1.0, max_retry_delay: float = 30.0,
                health_check_interval: float = 30.0, fallback_to_mock: bool = True):
        """Initialize the connection recovery manager"""
        # Connection parameters
        self.host = host
        self.port = port
        self.socket = None
        self.state = self.DISCONNECTED
        self.last_connection_time = None
        self.last_error = None
        self.attempt_count = 0
        
        # Recovery settings
        self.auto_reconnect = auto_reconnect
        self.max_retries = max_retries
        self.retry_delay = initial_retry_delay
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay
        self.health_check_interval = health_check_interval
        self.fallback_to_mock = fallback_to_mock
        
        # Threading
        self.recovery_thread = None
        self.health_check_thread = None
        self.stop_threads = threading.Event()
        
        # Event callbacks
        self.on_connected = None
        self.on_disconnected = None
        self.on_connection_failed = None
        self.on_mock_mode = None
        
        logger.info(f"ConnectionRecoveryManager initialized: {host}:{port}")
    
    def connect(self) -> bool:
        """
        Establish connection to the LUMINA V7 system
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if self.state == self.CONNECTED:
            return True
        
        self.state = self.CONNECTING
        logger.info(f"Connecting to {self.host}:{self.port}...")
        
        try:
            # Create new socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)  # 5 second connection timeout
            self.socket.connect((self.host, self.port))
            
            # Update state
            self.state = self.CONNECTED
            self.last_connection_time = datetime.now()
            self.attempt_count = 0
            self.retry_delay = self.initial_retry_delay  # Reset backoff
            
            # Start health check
            self._start_health_check()
            
            # Trigger callback
            if self.on_connected:
                self.on_connected()
                
            logger.info(f"Successfully connected to {self.host}:{self.port}")
            return True
            
        except Exception as e:
            self.state = self.FAILED
            self.last_error = str(e)
            self.attempt_count += 1
            
            logger.error(f"Connection failed: {e}")
            
            # Start recovery if auto-reconnect enabled
            if self.auto_reconnect:
                self._start_recovery()
            
            # Fallback to mock mode if enabled
            if self.fallback_to_mock and self.on_mock_mode:
                self.state = self.MOCK_MODE
                self.on_mock_mode()
                logger.warning("Entered mock mode due to connection failure")
            
            # Trigger callback
            if self.on_connection_failed:
                self.on_connection_failed(str(e))
                
            return False
    
    def disconnect(self):
        """Close connection to the server"""
        # Stop background threads
        self._stop_threads()
        
        # Close socket
        if self.socket:
            try:
                self.socket.close()
            except Exception as e:
                logger.error(f"Error closing socket: {e}")
            finally:
                self.socket = None
        
        # Update state (only if previously connected)
        if self.state == self.CONNECTED:
            self.state = self.DISCONNECTED
            
            # Trigger callback
            if self.on_disconnected:
                self.on_disconnected()
        
        logger.info("Disconnected from server")
    
    def send_request(self, request_data: Dict) -> tuple:
        """
        Send request to server with automatic recovery
        
        Args:
            request_data: Dictionary of request data
            
        Returns:
            (success, response): Tuple with success flag and response or error message
        """
        # Check if in mock mode
        if self.state == self.MOCK_MODE:
            return False, "In mock mode"
        
        # Connect if not connected
        if self.state != self.CONNECTED:
            if not self.connect():
                return False, f"Not connected: {self.last_error}"
        
        try:
            # Encode and send request
            request_json = json.dumps(request_data)
            self.socket.sendall(request_json.encode('utf-8') + b'\n')
            
            # Receive and decode response
            response = self._receive_response()
            
            # Update last successful connection time
            self.last_connection_time = datetime.now()
            
            return True, response
            
        except Exception as e:
            logger.error(f"Error in send_request: {e}")
            self.last_error = str(e)
            
            # Disconnect and start recovery
            self.disconnect()
            if self.auto_reconnect:
                self._start_recovery()
            
            return False, str(e)
    
    def _receive_response(self) -> Dict:
        """Receive JSON response from server"""
        buffer = b''
        
        while True:
            chunk = self.socket.recv(4096)
            if not chunk:
                raise ConnectionError("Connection closed by server")
                
            buffer += chunk
            
            # Check if we have a complete response (ends with newline)
            if b'\n' in buffer:
                break
        
        # Decode JSON response
        response_str = buffer.decode('utf-8').strip()
        response = json.loads(response_str)
        
        return response
    
    def _start_recovery(self):
        """Start connection recovery thread"""
        if self.recovery_thread and self.recovery_thread.is_alive():
            return  # Already running
            
        # Create and start recovery thread
        self.stop_threads.clear()
        self.recovery_thread = threading.Thread(
            target=self._recovery_worker,
            daemon=True,
            name="ConnectionRecoveryThread"
        )
        self.recovery_thread.start()
        
        logger.info("Started connection recovery thread")
    
    def _recovery_worker(self):
        """Worker thread for connection recovery with exponential backoff"""
        # Keep trying until max retries reached or connection successful
        while (self.max_retries <= 0 or self.attempt_count <= self.max_retries) and not self.stop_threads.is_set():
            # Wait before retrying with exponential backoff
            logger.info(f"Waiting {self.retry_delay:.1f}s before retry attempt {self.attempt_count}")
            
            # Sleep with ability to be interrupted
            if self.stop_threads.wait(self.retry_delay):
                break  # Thread stop requested
            
            # Try to connect
            if self.connect():
                break  # Connected successfully
            
            # Increase retry delay with exponential backoff (max 30s)
            self.retry_delay = min(self.retry_delay * 1.5, self.max_retry_delay)
        
        # Log if max retries reached
        if self.max_retries > 0 and self.attempt_count > self.max_retries:
            logger.error(f"Maximum retry attempts ({self.max_retries}) reached")
    
    def _start_health_check(self):
        """Start health check thread"""
        if self.health_check_thread and self.health_check_thread.is_alive():
            return  # Already running
            
        # Create and start health check thread
        self.stop_threads.clear()
        self.health_check_thread = threading.Thread(
            target=self._health_check_worker,
            daemon=True,
            name="ConnectionHealthCheckThread"
        )
        self.health_check_thread.start()
        
        logger.debug("Started health check thread")
    
    def _health_check_worker(self):
        """Worker thread for periodic connection health checks"""
        while self.state == self.CONNECTED and not self.stop_threads.is_set():
            # Sleep for health check interval
            if self.stop_threads.wait(self.health_check_interval):
                break  # Thread stop requested
            
            # Send health check ping
            try:
                ping_request = {
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                }
                
                success, response = self.send_request(ping_request)
                if not success:
                    logger.warning(f"Health check failed: {response}")
                else:
                    logger.debug("Health check successful")
                    
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                # Connection will be recovered by send_request
    
    def _stop_threads(self):
        """Stop all background threads"""
        self.stop_threads.set()
        
        # Wait for threads to terminate
        threads = []
        if self.recovery_thread and self.recovery_thread.is_alive():
            threads.append(self.recovery_thread)
            
        if self.health_check_thread and self.health_check_thread.is_alive():
            threads.append(self.health_check_thread)
            
        # Join all threads with timeout
        for thread in threads:
            thread.join(timeout=1.0)
    
    def set_callbacks(self, on_connected: Optional[Callable] = None,
                     on_disconnected: Optional[Callable] = None,
                     on_connection_failed: Optional[Callable] = None,
                     on_mock_mode: Optional[Callable] = None):
        """Register callback functions for connection events"""
        if on_connected:
            self.on_connected = on_connected
        
        if on_disconnected:
            self.on_disconnected = on_disconnected
        
        if on_connection_failed:
            self.on_connection_failed = on_connection_failed
        
        if on_mock_mode:
            self.on_mock_mode = on_mock_mode
    
    def get_status(self) -> Dict[str, Any]:
        """Get current connection status information"""
        return {
            "state": self.state,
            "host": self.host,
            "port": self.port,
            "last_connection_time": self.last_connection_time.isoformat() if self.last_connection_time else None,
            "attempt_count": self.attempt_count,
            "last_error": self.last_error,
            "retry_delay": self.retry_delay
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.disconnect()
        logger.info("ConnectionRecoveryManager cleaned up") 