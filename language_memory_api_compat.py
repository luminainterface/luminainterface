"""
Language Memory API Compatibility Layer

This module provides a compatibility layer for the Language Memory API
that works seamlessly with both PySide6 and PyQt5-based interfaces.
"""

import os
import sys
import logging
import json
import threading
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Qt compatibility utilities
try:
    from src.v5.ui.qt_compat import QtCompat, QtCore, Signal
    HAS_QT = True
except ImportError:
    logger.warning("Qt compatibility layer not available. GUI features will be disabled.")
    HAS_QT = False

# Try to import the Language Memory API
try:
    from src.language_memory_synthesis_integration import LanguageMemoryAPI
    HAS_LANGUAGE_MEMORY_API = True
except ImportError:
    logger.warning("Language Memory API not available. Using mock functionality.")
    HAS_LANGUAGE_MEMORY_API = False


class LanguageMemoryAPICompat:
    """
    Compatibility layer for the Language Memory API providing interfaces
    for both PySide6 and PyQt5 based applications.
    """
    
    # Signal definitions for Qt integration
    if HAS_QT:
        class Signals(QtCore.QObject):
            memory_updated = Signal(dict)
            topics_updated = Signal(dict)
            error_occurred = Signal(str)
            status_changed = Signal(str)
            
        signals = Signals()
    
    def __init__(self):
        """Initialize the Language Memory API compatibility layer."""
        self.api = None
        self.mock_mode = not HAS_LANGUAGE_MEMORY_API
        self.topic_cache = {}
        self.context_cache = {}
        self.last_status = "initialized"
        
        # Initialize real API if available
        if HAS_LANGUAGE_MEMORY_API:
            try:
                self.api = LanguageMemoryAPI()
                logger.info("Successfully initialized Language Memory API")
            except Exception as e:
                logger.error(f"Failed to initialize Language Memory API: {str(e)}")
                self.mock_mode = True
        
        # Set up background threads
        self.active = True
        self.status_thread = threading.Thread(target=self._status_check_loop, daemon=True)
        self.status_thread.start()
        
        logger.info(f"Language Memory API Compatibility Layer initialized (mock: {self.mock_mode})")
    
    def process_message(self, message: str, user_id: str = "default", 
                       session_id: str = None, async_mode: bool = False) -> Dict:
        """
        Process a message through the Language Memory system.
        
        Args:
            message: The message to process
            user_id: Identifier for the user
            session_id: Identifier for the session
            async_mode: Whether to process asynchronously
            
        Returns:
            Dictionary containing the processed message and context
        """
        if async_mode and HAS_QT:
            # Start processing in a background thread
            thread = threading.Thread(
                target=self._process_message_thread,
                args=(message, user_id, session_id),
                daemon=True
            )
            thread.start()
            return {"status": "processing", "message": message}
        
        try:
            if self.mock_mode or not self.api:
                response = self._generate_mock_process_response(message, user_id, session_id)
            else:
                response = self.api.process_message(message, user_id, session_id)
            
            # Cache the response
            cache_key = f"{user_id}_{session_id}_{hash(message)}"
            self.context_cache[cache_key] = response
            
            # Emit signal if Qt is available
            if HAS_QT:
                self.signals.memory_updated.emit(response)
            
            return response
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            
            # Emit error signal if Qt is available
            if HAS_QT:
                self.signals.error_occurred.emit(error_msg)
                
            return {"error": error_msg, "message": message}
    
    def _process_message_thread(self, message, user_id, session_id):
        """Background thread for processing messages asynchronously."""
        try:
            result = self.process_message(message, user_id, session_id, async_mode=False)
            
            # Emit signal with the result if Qt is available
            if HAS_QT:
                self.signals.memory_updated.emit(result)
                
        except Exception as e:
            error_msg = f"Error in async message processing: {str(e)}"
            logger.error(error_msg)
            
            # Emit error signal if Qt is available
            if HAS_QT:
                self.signals.error_occurred.emit(error_msg)
    
    def synthesize_topic(self, topic: str, depth: int = 3, 
                        async_mode: bool = False) -> Dict:
        """
        Synthesize memories around a topic.
        
        Args:
            topic: The topic to synthesize
            depth: How deep to search for related memories
            async_mode: Whether to process asynchronously
            
        Returns:
            Dictionary containing synthesized memories
        """
        if async_mode and HAS_QT:
            # Start synthesis in a background thread
            thread = threading.Thread(
                target=self._synthesize_topic_thread,
                args=(topic, depth),
                daemon=True
            )
            thread.start()
            return {"status": "processing", "topic": topic}
        
        try:
            # Check cache first
            cache_key = f"{topic}_{depth}"
            if cache_key in self.topic_cache:
                logger.info(f"Using cached synthesis for topic: {topic}")
                return self.topic_cache[cache_key]
            
            if self.mock_mode or not self.api:
                response = self._generate_mock_synthesis_response(topic, depth)
            else:
                response = self.api.get_topic_details(topic, depth=depth)
            
            # Cache the response
            self.topic_cache[cache_key] = response
            
            # Emit signal if Qt is available
            if HAS_QT:
                self.signals.topics_updated.emit(response)
            
            return response
        except Exception as e:
            error_msg = f"Error synthesizing topic: {str(e)}"
            logger.error(error_msg)
            
            # Emit error signal if Qt is available
            if HAS_QT:
                self.signals.error_occurred.emit(error_msg)
                
            return {"error": error_msg, "topic": topic}
    
    def _synthesize_topic_thread(self, topic, depth):
        """Background thread for synthesizing topics asynchronously."""
        try:
            result = self.synthesize_topic(topic, depth, async_mode=False)
            
            # Emit signal with the result if Qt is available
            if HAS_QT:
                self.signals.topics_updated.emit(result)
                
        except Exception as e:
            error_msg = f"Error in async topic synthesis: {str(e)}"
            logger.error(error_msg)
            
            # Emit error signal if Qt is available
            if HAS_QT:
                self.signals.error_occurred.emit(error_msg)
    
    def store_memory(self, content: str, metadata: Dict = None) -> Dict:
        """
        Store a memory in the Language Memory system.
        
        Args:
            content: The content to store
            metadata: Additional metadata for the memory
            
        Returns:
            Dictionary containing the result of the storage operation
        """
        try:
            if self.mock_mode or not self.api:
                return {"status": "success", "message": "Memory stored (mock mode)"}
            
            return self.api.store_external_memory(content, metadata or {})
        except Exception as e:
            error_msg = f"Error storing memory: {str(e)}"
            logger.error(error_msg)
            
            # Emit error signal if Qt is available
            if HAS_QT:
                self.signals.error_occurred.emit(error_msg)
                
            return {"error": error_msg}
    
    def get_topics(self, limit: int = 10) -> List:
        """
        Get a list of synthesized topics.
        
        Args:
            limit: Maximum number of topics to return
            
        Returns:
            List of topics
        """
        try:
            if self.mock_mode or not self.api:
                return self._generate_mock_topics(limit)
            
            return self.api.get_topics(limit=limit)
        except Exception as e:
            error_msg = f"Error getting topics: {str(e)}"
            logger.error(error_msg)
            
            # Emit error signal if Qt is available
            if HAS_QT:
                self.signals.error_occurred.emit(error_msg)
                
            return []
    
    def health_check(self) -> Dict:
        """
        Check the health of the Language Memory system.
        
        Returns:
            Dictionary containing health status
        """
        try:
            if self.mock_mode or not self.api:
                return {
                    "status": "healthy",
                    "mock": True,
                    "components": {
                        "memory_storage": "healthy",
                        "synthesis_engine": "healthy",
                        "api_interface": "healthy"
                    }
                }
            
            return self.api.health_check()
        except Exception as e:
            error_msg = f"Error checking health: {str(e)}"
            logger.error(error_msg)
            
            # Emit error signal if Qt is available
            if HAS_QT:
                self.signals.error_occurred.emit(error_msg)
                
            return {"status": "error", "error": str(e)}
    
    def clear_cache(self):
        """Clear all caches."""
        self.topic_cache.clear()
        self.context_cache.clear()
        logger.info("Cleared all caches")
    
    def _status_check_loop(self):
        """Background thread for checking system status."""
        while self.active:
            try:
                # Check health
                health = self.health_check()
                status = health.get("status", "unknown")
                
                # If status changed, emit signal
                if status != self.last_status and HAS_QT:
                    self.signals.status_changed.emit(status)
                    
                self.last_status = status
                
            except Exception as e:
                logger.error(f"Error in status check loop: {str(e)}")
            
            # Sleep for a while
            time.sleep(30)
    
    def shutdown(self):
        """Shut down the compatibility layer."""
        logger.info("Shutting down Language Memory API Compatibility Layer")
        self.active = False
        
        if self.status_thread and self.status_thread.is_alive():
            self.status_thread.join(timeout=1.0)
    
    def _generate_mock_process_response(self, message, user_id, session_id):
        """Generate a mock response for message processing."""
        import random
        
        return {
            "status": "success",
            "system_prompt": f"You are an AI assistant with memory about previous conversations with the user {user_id}.",
            "context": {
                "user_message": message,
                "user_id": user_id,
                "session_id": session_id or f"session_{int(time.time())}",
                "memory_context": [
                    f"Earlier you discussed neural networks and their architecture.",
                    f"You were interested in the integration of language systems.",
                    f"The concept of '{random.choice(message.split())}' has appeared in {random.randint(1, 5)} previous conversations."
                ]
            }
        }
    
    def _generate_mock_synthesis_response(self, topic, depth):
        """Generate a mock response for topic synthesis."""
        import random
        
        # Generate some related topics based on the main topic
        related_topics = []
        for i in range(3 + depth):
            related_topics.append({
                "topic": f"{topic}_related_{i}",
                "relevance": round(random.uniform(0.3, 0.95), 2)
            })
        
        # Generate some insights
        insights = [
            f"The concept of {topic} appears frequently in neural processing discussions",
            f"{topic} demonstrates connections to language understanding",
            f"Pattern recognition related to {topic} shows high coherence",
            f"Processing {topic} activates both logical and intuitive pathways"
        ]
        
        return {
            "status": "success",
            "topic": topic,
            "depth": depth,
            "synthesized_memory": {
                "core_understanding": f"{topic} is a fundamental concept in neural processing",
                "core_insights": insights,
                "source_count": random.randint(5, 20)
            },
            "related_topics": related_topics,
            "mock": True
        }
    
    def _generate_mock_topics(self, limit):
        """Generate mock topics."""
        topics = [
            "neural_networks",
            "consciousness",
            "language_processing",
            "memory_systems",
            "pattern_recognition",
            "resonance",
            "fractal_patterns",
            "self_awareness",
            "emergence",
            "integration",
            "echo_systems",
            "reflection",
            "knowledge_graphs",
            "recursion",
            "perception"
        ]
        
        # Return a slice of the topics list
        return topics[:limit]


# Create a singleton instance
memory_api = LanguageMemoryAPICompat() 