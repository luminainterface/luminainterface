#!/usr/bin/env python3
"""
Signal Processor

This module processes signals from the backend system,
handling data transformation and routing to appropriate components.
"""

import logging
from typing import Dict, Any, Optional, List
from collections import deque
from datetime import datetime

from PySide6.QtCore import QObject, Signal, Slot

from .config import SIGNAL_CONFIG

logger = logging.getLogger(__name__)

class SignalProcessor(QObject):
    """Processes signals from the backend system."""
    
    # Signals
    signal_processed = Signal(dict)  # Emitted when a signal is processed
    error_occurred = Signal(str)  # Emitted when an error occurs
    
    def __init__(self):
        """Initialize the signal processor."""
        super().__init__()
        self.signal_buffer = deque(maxlen=SIGNAL_CONFIG['buffer_size'])
        self.handlers = {}
        self.processing = False
        
    def register_handler(self, signal_type: str, handler: callable) -> None:
        """Register a handler for a specific signal type."""
        self.handlers[signal_type] = handler
        
    def unregister_handler(self, signal_type: str) -> None:
        """Unregister a handler for a specific signal type."""
        if signal_type in self.handlers:
            del self.handlers[signal_type]
            
    @Slot(dict)
    def process_signal(self, signal_data: Dict[str, Any]) -> None:
        """Process an incoming signal."""
        try:
            # Add signal to buffer
            self.signal_buffer.append({
                'data': signal_data,
                'timestamp': datetime.now(),
                'retries': 0
            })
            
            # Start processing if not already running
            if not self.processing:
                self._process_buffer()
                
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            self.error_occurred.emit(str(e))
            
    def _process_buffer(self) -> None:
        """Process signals in the buffer."""
        self.processing = True
        
        while self.signal_buffer:
            try:
                # Get next signal
                signal = self.signal_buffer.popleft()
                
                # Process signal
                signal_type = signal['data'].get('type')
                if signal_type in self.handlers:
                    # Call handler
                    self.handlers[signal_type](signal['data'])
                    
                    # Emit processed signal
                    self.signal_processed.emit(signal['data'])
                    
                else:
                    # Retry or discard
                    if signal['retries'] < SIGNAL_CONFIG['max_retries']:
                        signal['retries'] += 1
                        self.signal_buffer.append(signal)
                    else:
                        logger.warning(f"Discarding signal after {SIGNAL_CONFIG['max_retries']} retries")
                        
            except Exception as e:
                logger.error(f"Error processing signal from buffer: {e}")
                self.error_occurred.emit(str(e))
                
        self.processing = False
        
    def get_pending_signals(self) -> List[Dict[str, Any]]:
        """Get list of pending signals."""
        return list(self.signal_buffer)
        
    def clear_buffer(self) -> None:
        """Clear the signal buffer."""
        self.signal_buffer.clear()
        
    def get_handler_count(self) -> int:
        """Get number of registered handlers."""
        return len(self.handlers) 