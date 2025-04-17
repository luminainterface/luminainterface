#!/usr/bin/env python3
"""
Mistral PySide6 Integration

This module provides integration between the Mistral LLM system and the PySide6 frontend
through the V7 socket manager architecture.
"""

import sys
import os
import json
import logging
import asyncio
import threading
import time
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

# Add parent directory to path to allow imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import PySide6
from PySide6.QtCore import QObject, Signal, Slot, QTimer

# Import Mistral components
from src.api.mistral_integration_fixed import MistralIntegration

# Import socket manager
from src.v7.ui.v7_socket_manager import V7SocketManager

# Configure logging
logger = logging.getLogger("MistralPySideIntegration")

class MistralSignals(QObject):
    """Signal wrapper for thread-safe communication with the UI"""
    message_received = Signal(dict)
    status_update = Signal(dict)
    response_received = Signal(str, dict)
    error_occurred = Signal(str)
    weights_updated = Signal(float, float)
    processing_started = Signal()
    processing_finished = Signal()

class MistralPySideIntegration:
    """
    Integration between Mistral LLM system and PySide6 through V7 socket manager
    
    This class provides the bridge between the Mistral Enhanced System and the
    PySide6 UI components, using the V7 socket manager for communication.
    """
    
    def __init__(self, socket_manager: V7SocketManager, api_key: Optional[str] = None):
        """
        Initialize the Mistral PySide6 integration
        
        Args:
            socket_manager: V7 socket manager instance
            api_key: Optional Mistral API key (can also be set via environment variable)
        """
        self.socket_manager = socket_manager
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        self.mistral_system = None
        self.signals = MistralSignals()
        self.conversation_history = []
        self.is_processing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_status)
        
        # Register with socket manager
        self._register_with_socket_manager()
        
        # Initialize Mistral system
        self._initialize_mistral_system()
        
        logger.info("Mistral PySide6 integration initialized")
    
    def _register_with_socket_manager(self):
        """Register with the V7 socket manager"""
        # Create and register a plugin for the socket manager
        class MistralPlugin:
            def __init__(self, integration):
                self.integration = integration
                self.plugin_id = "mistral_llm_plugin"
            
            def get_plugin_id(self):
                return self.plugin_id
            
            def process_message(self, message):
                if message.get('type') == 'mistral_request':
                    self.integration.process_message(message)
        
        # Create and register plugin
        plugin = MistralPlugin(self)
        self.socket_manager.register_plugin(plugin)
        
        # Subscribe to relevant message types
        self.socket_manager.subscribe("mistral_request", self._handle_mistral_request)
        self.socket_manager.subscribe("mistral_weights_update", self._handle_weights_update)
        
        logger.info("Registered with socket manager")
    
    def _initialize_mistral_system(self):
        """Initialize the Mistral Enhanced System"""
        try:
            self.mistral_system = MistralIntegration(
                api_key=self.api_key,
                model="mistral-large-latest",
                llm_weight=0.7,
                nn_weight=0.3
            )
            logger.info("Mistral System initialized")
        except Exception as e:
            logger.error(f"Error initializing Mistral system: {e}")
            self.signals.error_occurred.emit(f"Error initializing Mistral system: {e}")
    
    @Slot(dict)
    def _handle_mistral_request(self, message):
        """
        Handle a Mistral request from the socket manager
        
        Args:
            message: Request message
        """
        request_type = message.get('request_type', '')
        request_id = message.get('request_id', '')
        
        if request_type == 'chat':
            user_message = message.get('message', '')
            self.process_message_async(user_message, request_id)
        
        elif request_type == 'set_weights':
            llm_weight = message.get('llm_weight', 0.7)
            nn_weight = message.get('nn_weight', 0.3)
            self.set_weights(llm_weight, nn_weight)
        
        elif request_type == 'get_conversation':
            self._send_conversation_history(request_id)
        
        elif request_type == 'get_stats':
            self._send_system_stats(request_id)
    
    @Slot(dict)
    def _handle_weights_update(self, message):
        """
        Handle a weights update message from the socket manager
        
        Args:
            message: Weights update message
        """
        llm_weight = message.get('llm_weight', 0.7)
        nn_weight = message.get('nn_weight', 0.3)
        self.set_weights(llm_weight, nn_weight)
    
    def process_message_async(self, user_message: str, request_id: str = ''):
        """
        Process a user message asynchronously
        
        Args:
            user_message: User message
            request_id: Optional request ID for tracking
        """
        if not self.mistral_system:
            error_msg = "Mistral system not initialized"
            logger.error(error_msg)
            self.signals.error_occurred.emit(error_msg)
            return
        
        if self.is_processing:
            error_msg = "Already processing a message"
            logger.warning(error_msg)
            self.signals.error_occurred.emit(error_msg)
            return
        
        # Create and start processing thread
        def process_thread():
            self.is_processing = True
            self.signals.processing_started.emit()
            
            try:
                # Add message to conversation history
                self.conversation_history.append({
                    "role": "user",
                    "content": user_message,
                    "timestamp": int(time.time())
                })
                
                # Process message
                result = self.mistral_system.process_message(user_message)
                response = result.get('combined_response', '')
                
                # Add response to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": int(time.time())
                })
                
                # Send response to UI
                response_data = {
                    "type": "mistral_response",
                    "request_id": request_id,
                    "message": response,
                    "success": True
                }
                
                # Emit signal for UI update
                self.signals.response_received.emit(request_id, response_data)
                
                # Send response through socket manager
                self.socket_manager.send_message(response_data)
            
            except Exception as e:
                error_msg = f"Error processing message: {e}"
                logger.error(error_msg)
                
                # Send error response
                error_data = {
                    "type": "mistral_response",
                    "request_id": request_id,
                    "error": error_msg,
                    "success": False
                }
                
                # Emit signal for UI update
                self.signals.error_occurred.emit(error_msg)
                
                # Send error through socket manager
                self.socket_manager.send_message(error_data)
            
            finally:
                self.is_processing = False
                self.signals.processing_finished.emit()
        
        # Start processing in a separate thread
        thread = threading.Thread(target=process_thread)
        thread.daemon = True
        thread.start()
    
    def set_weights(self, llm_weight: float, nn_weight: float):
        """
        Set the LLM and neural network weights
        
        Args:
            llm_weight: LLM weight (0.0-1.0)
            nn_weight: Neural network weight (0.0-1.0)
        """
        if not self.mistral_system:
            error_msg = "Mistral system not initialized"
            logger.error(error_msg)
            self.signals.error_occurred.emit(error_msg)
            return
        
        try:
            # Update weights in the Mistral system
            weights = self.mistral_system.adjust_weights(llm_weight, nn_weight)
            
            # Extract actual weights (might be normalized)
            llm_weight = weights.get('llm_weight', llm_weight)
            nn_weight = weights.get('nn_weight', nn_weight)
            
            # Emit signal for UI update
            self.signals.weights_updated.emit(llm_weight, nn_weight)
            
            # Send update through socket manager
            self.socket_manager.send_message({
                "type": "mistral_weights_updated",
                "llm_weight": llm_weight,
                "nn_weight": nn_weight
            })
            
            logger.info(f"Weights updated: LLM={llm_weight}, NN={nn_weight}")
        except Exception as e:
            error_msg = f"Error updating weights: {e}"
            logger.error(error_msg)
            self.signals.error_occurred.emit(error_msg)
    
    def _send_conversation_history(self, request_id: str):
        """
        Send the conversation history through the socket manager
        
        Args:
            request_id: Request ID for tracking
        """
        history_data = {
            "type": "mistral_conversation_history",
            "request_id": request_id,
            "history": self.conversation_history
        }
        self.socket_manager.send_message(history_data)
    
    def _send_system_stats(self, request_id: str):
        """
        Send system statistics through the socket manager
        
        Args:
            request_id: Request ID for tracking
        """
        if not self.mistral_system:
            error_msg = "Mistral system not initialized"
            logger.error(error_msg)
            self.signals.error_occurred.emit(error_msg)
            return
        
        try:
            stats = self.mistral_system.get_system_stats()
            
            stats_data = {
                "type": "mistral_system_stats",
                "request_id": request_id,
                "stats": stats
            }
            self.socket_manager.send_message(stats_data)
        
        except Exception as e:
            error_msg = f"Error getting system stats: {e}"
            logger.error(error_msg)
            self.signals.error_occurred.emit(error_msg)
    
    @Slot()
    def check_status(self):
        """Check the status of the Mistral system"""
        if not self.mistral_system:
            return
        
        # Send status update
        status_data = {
            "type": "mistral_status_update",
            "is_processing": self.is_processing,
            "is_initialized": self.mistral_system is not None,
            "conversation_length": len(self.conversation_history) // 2,  # User-assistant pairs
            "llm_weight": self.mistral_system.central_node.llm_weight,
            "nn_weight": self.mistral_system.central_node.nn_weight
        }
        
        # Emit signal for UI update
        self.signals.status_update.emit(status_data)
        
        # Send status through socket manager
        self.socket_manager.send_message(status_data)
    
    def start_status_timer(self, interval_ms: int = 5000):
        """
        Start the status update timer
        
        Args:
            interval_ms: Update interval in milliseconds
        """
        self.timer.start(interval_ms)
        logger.debug(f"Status timer started with interval {interval_ms}ms")
    
    def stop_status_timer(self):
        """Stop the status update timer"""
        self.timer.stop()
        logger.debug("Status timer stopped")
    
    def shutdown(self):
        """Shut down the Mistral system and clean up resources"""
        # Stop timer
        self.stop_status_timer()
        
        # Close Mistral system
        if self.mistral_system:
            try:
                self.mistral_system.close()
                self.mistral_system = None
                logger.info("Mistral system closed")
            except Exception as e:
                logger.error(f"Error closing Mistral system: {e}")
        
        logger.info("Mistral PySide6 integration shut down")

# For testing
if __name__ == "__main__":
    import time
    from PySide6.QtWidgets import QApplication
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create socket manager
    socket_manager = V7SocketManager()
    
    # Create integration
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    integration = MistralPySideIntegration(socket_manager, api_key)
    
    # Start status timer
    integration.start_status_timer()
    
    # Test message processing
    def on_response(request_id, response_data):
        print(f"Response received for request {request_id}:")
        print(response_data["message"])
    
    def on_error(error_msg):
        print(f"Error: {error_msg}")
    
    def on_status_update(status):
        print(f"Status update: {status}")
    
    # Connect signals
    integration.signals.response_received.connect(on_response)
    integration.signals.error_occurred.connect(on_error)
    integration.signals.status_update.connect(on_status_update)
    
    # Test message
    integration.process_message_async("Hello, what can you tell me about neural networks?", "test_request_1")
    
    # Run application
    timer = QTimer()
    timer.timeout.connect(app.quit)
    timer.start(30000)  # Run for 30 seconds
    
    app.exec()
    
    # Shutdown
    integration.shutdown() 