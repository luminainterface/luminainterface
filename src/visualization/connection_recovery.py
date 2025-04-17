#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Connection Recovery Module for LUMINA V7 Dashboard
=================================================

Implements automated connection recovery mechanisms for dashboard-system communication.
"""

import os
import sys
import time
import socket
import logging
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Callable

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/connection_recovery.log")
    ]
)
logger = logging.getLogger("ConnectionRecovery")

class ConnectionRecovery:
    """
    Implements connection recovery mechanisms for dashboard-system communication
    
    This class handles:
    - Automatic reconnection with exponential backoff
    - Connection health monitoring
    - Fallback to mock data when connection fails
    - Connection state logging and reporting
    """
    
    # Connection states
    STATE_DISCONNECTED = "disconnected"
    STATE_CONNECTING = "connecting"
    STATE_CONNECTED = "connected"
    STATE_FAILED = "failed"
    STATE_MOCK = "mock"
    
    def __init__(self, host="localhost", port=5678, 
                 auto_reconnect=True, max_retries=10, 
                 initial_retry_delay=1.0, max_retry_delay=60.0,
                 fallback_to_mock=True, health_check_interval=30.0):
        """
        Initialize the connection recovery system
        
        Args:
            host: Host to connect to
            port: Port to connect to
            auto_reconnect: Whether to automatically reconnect when connection fails
            max_retries: Maximum number of retry attempts before giving up
            initial_retry_delay: Initial delay between retries in seconds
            max_retry_delay: Maximum delay between retries in seconds
            fallback_to_mock: Whether to fall back to mock data when connection fails
            health_check_interval: Interval between connection health checks in seconds
        """
        # Connection parameters
        self.host = host
        self.port = port
        self.socket = None
        self.state = self.STATE_DISCONNECTED
        self.last_successful_connection = None
        self.last_error = None
        self.connection_attempts = 0
        
        # Recovery parameters
        self.auto_reconnect = auto_reconnect
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay
        self.current_retry_delay = initial_retry_delay
        self.fallback_to_mock = fallback_to_mock
        
        # Health check
        self.health_check_interval = health_check_interval
        self.health_check_thread = None
        self.health_check_event = threading.Event()
        
        # Callback functions
        self.on_connected_callback = None
        self.on_disconnected_callback = None
        self.on_failed_callback = None
        self.on_mock_mode_callback = None
        
        # Recovery thread
        self.recovery_thread = None
        self.recovery_event = threading.Event()
        
        # Log initialization
        logger.info(f"Initialized ConnectionRecovery: host={host}, port={port}, auto_reconnect={auto_reconnect}")
    
    def connect(self):
        """
        Establish connection to the server
        
        Returns:
            bool: Whether connection was successful
        """
        if self.state == self.STATE_CONNECTED:
            logger.debug("Already connected")
            return True
        
        # Update state
        self.state = self.STATE_CONNECTING
        
        try:
            # Create new socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)  # 10 second timeout
            
            # Connect to server
            logger.info(f"Connecting to {self.host}:{self.port}...")
            self.socket.connect((self.host, self.port))
            
            # Update state
            self.state = self.STATE_CONNECTED
            self.last_successful_connection = datetime.now()
            self.connection_attempts = 0
            self.current_retry_delay = self.initial_retry_delay  # Reset backoff delay
            
            # Start health check if not already running
            self._start_health_check()
            
            # Call connected callback
            if self.on_connected_callback:
                self.on_connected_callback()
            
            logger.info(f"Successfully connected to {self.host}:{self.port}")
            return True
            
        except Exception as e:
            # Update state
            self.state = self.STATE_FAILED
            self.last_error = str(e)
            self.connection_attempts += 1
            
            logger.error(f"Connection failed: {e}")
            
            # Start recovery if auto-reconnect is enabled
            if self.auto_reconnect:
                self._start_recovery()
            
            # Fall back to mock mode if enabled
            if self.fallback_to_mock:
                self._enter_mock_mode()
            
            # Call failed callback
            if self.on_failed_callback:
                self.on_failed_callback(str(e))
            
            return False
    
    def disconnect(self):
        """Disconnect from the server"""
        # Stop health check and recovery threads
        self._stop_health_check()
        self._stop_recovery()
        
        if self.socket:
            try:
                self.socket.close()
            except Exception as e:
                logger.error(f"Error closing socket: {e}")
            finally:
                self.socket = None
        
        # Update state
        prev_state = self.state
        self.state = self.STATE_DISCONNECTED
        
        # Call disconnected callback if was previously connected
        if prev_state == self.STATE_CONNECTED and self.on_disconnected_callback:
            self.on_disconnected_callback()
        
        logger.info("Disconnected from server")
    
    def send_request(self, request_data):
        """
        Send request to server with automatic recovery
        
        Args:
            request_data: Data to send to server
            
        Returns:
            Tuple[bool, Any]: (success, response_data or error message)
        """
        # Check if in mock mode
        if self.state == self.STATE_MOCK:
            logger.debug("In mock mode, generating mock response")
            return False, "In mock mode"
        
        # Check if connected
        if self.state != self.STATE_CONNECTED:
            # Try to connect if not connected
            if not self.connect():
                return False, f"Not connected: {self.last_error}"
        
        try:
            # Convert request data to JSON string
            request_json = json.dumps(request_data)
            
            # Send request
            self.socket.sendall(request_json.encode('utf-8') + b'\n')
            
            # Receive response
            response_data = self._receive_response()
            
            # Update last successful connection time
            self.last_successful_connection = datetime.now()
            
            return True, response_data
            
        except Exception as e:
            # Connection failed, update state
            logger.error(f"Error sending request: {e}")
            self.last_error = str(e)
            
            # Disconnect and start recovery
            self.disconnect()
            if self.auto_reconnect:
                self._start_recovery()
            
            # Fall back to mock mode if enabled
            if self.fallback_to_mock:
                self._enter_mock_mode()
            
            return False, str(e)
    
    def _receive_response(self):
        """
        Receive response from server
        
        Returns:
            Dict: Parsed JSON response
        """
        # Initialize buffer
        response_buffer = b''
        
        # Read until newline or timeout
        while True:
            chunk = self.socket.recv(4096)
            if not chunk:
                # Connection closed
                raise ConnectionError("Connection closed by server")
            
            response_buffer += chunk
            
            # Check if response is complete (ends with newline)
            if b'\n' in response_buffer:
                break
        
        # Decode and parse response
        response_json = response_buffer.decode('utf-8').strip()
        response_data = json.loads(response_json)
        
        return response_data
    
    def _start_recovery(self):
        """Start recovery thread to reconnect automatically"""
        # Stop existing recovery thread if running
        self._stop_recovery()
        
        # Create and start new recovery thread
        self.recovery_event.clear()
        self.recovery_thread = threading.Thread(
            target=self._recovery_worker,
            daemon=True,
            name="ConnectionRecoveryThread"
        )
        self.recovery_thread.start()
        
        logger.info("Started connection recovery thread")
    
    def _stop_recovery(self):
        """Stop recovery thread if running"""
        if self.recovery_thread and self.recovery_thread.is_alive():
            self.recovery_event.set()
            self.recovery_thread.join(timeout=1.0)
            self.recovery_thread = None
            logger.debug("Stopped recovery thread")
    
    def _recovery_worker(self):
        """Worker thread for connection recovery with exponential backoff"""
        # Keep trying until max retries or successful connection
        while (self.connection_attempts <= self.max_retries or self.max_retries <= 0) and not self.recovery_event.is_set():
            # Wait with exponential backoff
            logger.info(f"Retry attempt {self.connection_attempts}/{self.max_retries}, waiting {self.current_retry_delay:.1f}s")
            
            # Sleep with ability to interrupt
            if self.recovery_event.wait(self.current_retry_delay):
                # Event was set, exit thread
                break
            
            # Try to connect
            if self.connect():
                # Connected successfully, exit thread
                break
            
            # Calculate next retry delay with exponential backoff
            self.current_retry_delay = min(self.current_retry_delay * 1.5, self.max_retry_delay)
        
        # If we've reached max retries, log and give up
        if self.connection_attempts > self.max_retries and self.max_retries > 0:
            logger.error(f"Max retries ({self.max_retries}) exceeded, giving up")
            
            # Fall back to mock mode if enabled
            if self.fallback_to_mock:
                self._enter_mock_mode()
    
    def _start_health_check(self):
        """Start health check thread if not already running"""
        if not self.health_check_thread or not self.health_check_thread.is_alive():
            self.health_check_event.clear()
            self.health_check_thread = threading.Thread(
                target=self._health_check_worker,
                daemon=True,
                name="ConnectionHealthCheckThread"
            )
            self.health_check_thread.start()
            logger.debug("Started health check thread")
    
    def _stop_health_check(self):
        """Stop health check thread if running"""
        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_event.set()
            self.health_check_thread.join(timeout=1.0)
            self.health_check_thread = None
            logger.debug("Stopped health check thread")
    
    def _health_check_worker(self):
        """Worker thread for periodic connection health checks"""
        while self.state == self.STATE_CONNECTED and not self.health_check_event.is_set():
            # Wait for health check interval or until interrupted
            if self.health_check_event.wait(self.health_check_interval):
                # Event was set, exit thread
                break
            
            # Check if we've been connected recently
            if self.last_successful_connection:
                time_since_last_connection = datetime.now() - self.last_successful_connection
                
                # If no successful connection in too long, assume connection is dead
                if time_since_last_connection > timedelta(seconds=self.health_check_interval * 2):
                    logger.warning(f"No successful connection in {time_since_last_connection.total_seconds():.1f}s, reconnecting")
                    self.disconnect()
                    self.connect()
                    continue
            
            # Perform health check by sending ping
            try:
                # Create health check request
                ping_request = {
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Send request and get response
                success, response = self.send_request(ping_request)
                
                if not success:
                    logger.warning(f"Health check failed: {response}")
                    # Connection will be recovered in send_request if auto_reconnect is True
                else:
                    logger.debug("Health check successful")
                
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                self.last_error = str(e)
                
                # Disconnect and reconnect
                self.disconnect()
                self.connect()
    
    def _enter_mock_mode(self):
        """Enter mock mode when connection fails"""
        # Update state
        self.state = self.STATE_MOCK
        
        # Call mock mode callback
        if self.on_mock_mode_callback:
            self.on_mock_mode_callback()
        
        logger.warning("Entered mock mode due to connection failure")
    
    def get_status(self):
        """
        Get current connection status
        
        Returns:
            Dict: Status information
        """
        return {
            "state": self.state,
            "host": self.host,
            "port": self.port,
            "last_successful_connection": self.last_successful_connection.isoformat() if self.last_successful_connection else None,
            "connection_attempts": self.connection_attempts,
            "last_error": self.last_error,
            "auto_reconnect": self.auto_reconnect,
            "max_retries": self.max_retries,
            "current_retry_delay": self.current_retry_delay,
            "fallback_to_mock": self.fallback_to_mock
        }
    
    def register_callbacks(self, on_connected=None, on_disconnected=None, 
                          on_failed=None, on_mock_mode=None):
        """
        Register callback functions for connection events
        
        Args:
            on_connected: Callback when connection is established
            on_disconnected: Callback when connection is lost
            on_failed: Callback when connection fails
            on_mock_mode: Callback when entering mock mode
        """
        if on_connected:
            self.on_connected_callback = on_connected
        
        if on_disconnected:
            self.on_disconnected_callback = on_disconnected
        
        if on_failed:
            self.on_failed_callback = on_failed
        
        if on_mock_mode:
            self.on_mock_mode_callback = on_mock_mode
        
        logger.debug("Registered connection callbacks")
    
    def cleanup(self):
        """Clean up resources when no longer needed"""
        self.disconnect()
        logger.info("ConnectionRecovery cleaned up") 