"""
Language Memory Bridge for V5 PySide6 Client

This module provides the bridge between the V5 PySide6 client and the Language Memory System.
"""

import os
import sys
import json
import time
import logging
import threading
import random
import uuid
from pathlib import Path

# Add parent directory to path if needed
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

logger = logging.getLogger(__name__)

# Try to import from PySide6
try:
    from PySide6.QtCore import QObject, Signal
    USING_PYSIDE6 = True
except ImportError:
    from PyQt5.QtCore import QObject, pyqtSignal as Signal
    USING_PYSIDE6 = False

class LanguageMemoryBridge(QObject):
    """Bridge between V5 PySide6 client and Language Memory System"""
    
    # Signals
    connection_status_changed = Signal(bool, str)
    memory_topics_updated = Signal(list)
    memory_search_results = Signal(dict)
    memory_stats_updated = Signal(dict)
    
    def __init__(self, mock_mode=False):
        """
        Initialize the bridge
        
        Args:
            mock_mode: Use mock mode for testing without backend services
        """
        super().__init__()
        
        self.mock_mode = mock_mode
        self.connected = False
        self.memory_api = None
        self.memory_api_socket = None
        self.cache = {
            "topics": [],
            "stats": {},
            "memories": {},
            "fractal_params": {}
        }
        
        logger.info(f"Language Memory Bridge initialized (mock_mode={mock_mode})")
    
    def connect(self):
        """
        Connect to the Language Memory System
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.mock_mode:
            # Simulate connection delay
            time.sleep(random.uniform(0.5, 1.5))
            
            # Set connected flag
            self.connected = True
            
            # Emit signal
            self.connection_status_changed.emit(True, "Connected to Language Memory System (mock mode)")
            
            # Start mock data thread
            threading.Thread(
                target=self._mock_data_thread,
                daemon=True
            ).start()
            
            logger.info("Connected to Language Memory System (mock mode)")
            return True
        
        # Try to connect to the real Language Memory System
        try:
            # Try to import Memory API
            try:
                from src.memory_api import MemoryAPI
                self.memory_api = MemoryAPI()
                logger.info("Successfully imported Memory API")
            except ImportError as e:
                logger.warning(f"Memory API not available: {str(e)}")
                
                # Try to import from language_memory_v5_bridge.py if available
                try:
                    from language_memory_v5_bridge import MemoryAPISocketProvider
                    self.memory_api_socket = MemoryAPISocketProvider()
                    logger.info("Successfully imported Memory API Socket Provider")
                except ImportError as e:
                    logger.error(f"Memory API Socket Provider not available: {str(e)}")
                    logger.error("Cannot connect to Language Memory System")
                    return False
            
            # Test connection
            if self.memory_api:
                test_result = self.memory_api.get_memory_stats()
                if not test_result:
                    logger.error("Memory API connection test failed")
                    return False
            
            elif self.memory_api_socket:
                test_result = self.memory_api_socket.connect()
                if not test_result:
                    logger.error("Memory API Socket connection test failed")
                    return False
            
            # Set connected flag
            self.connected = True
            
            # Emit signal
            self.connection_status_changed.emit(True, "Connected to Language Memory System")
            
            logger.info("Connected to Language Memory System")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to Language Memory System: {str(e)}")
            return False
    
    def disconnect(self):
        """
        Disconnect from the Language Memory System
        
        Returns:
            True if disconnection successful, False otherwise
        """
        # Set disconnected flag
        self.connected = False
        
        # Emit signal
        self.connection_status_changed.emit(False, "Disconnected from Language Memory System")
        
        if self.memory_api_socket:
            try:
                self.memory_api_socket.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting from Memory API Socket: {str(e)}")
        
        logger.info("Disconnected from Language Memory System")
        return True
    
    def get_available_topics(self):
        """
        Get available topics from Language Memory System
        
        Returns:
            List of topic dictionaries
        """
        if not self.connected:
            logger.warning("Not connected to Language Memory System")
            return []
        
        if self.mock_mode:
            # Use cached data or generate if not available
            if not self.cache["topics"]:
                self.cache["topics"] = self._generate_mock_topics()
            
            return self.cache["topics"]
        
        try:
            if self.memory_api:
                topics = self.memory_api.get_topics()
                if topics:
                    self.cache["topics"] = topics.get("topics", [])
                    return self.cache["topics"]
            
            elif self.memory_api_socket:
                message = {
                    "type": "get_available_topics",
                    "request_id": str(uuid.uuid4())
                }
                response = self.memory_api_socket.send_message(message)
                if response and response.get("status") == "success":
                    self.cache["topics"] = response.get("data", {}).get("topics", [])
                    return self.cache["topics"]
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting available topics: {str(e)}")
            return []
    
    def search_memories(self, query, max_results=10):
        """
        Search for memories in the Language Memory System
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with search results
        """
        if not self.connected:
            logger.warning("Not connected to Language Memory System")
            return {"memories": [], "query": query, "total_results": 0}
        
        if self.mock_mode:
            # Generate mock search results
            search_id = f"search_{query}_{max_results}"
            if search_id not in self.cache["memories"]:
                mock_results = self._generate_mock_search_results(query, max_results)
                self.cache["memories"][search_id] = mock_results
            
            result = self.cache["memories"][search_id]
            
            # Emit signal
            self.memory_search_results.emit(result)
            
            return result
        
        try:
            if self.memory_api:
                result = self.memory_api.retrieve_relevant_memories(query, max_results)
                if result and result.get("status") == "success":
                    # Emit signal
                    self.memory_search_results.emit(result)
                    return result
            
            elif self.memory_api_socket:
                message = {
                    "type": "search_memories",
                    "request_id": str(uuid.uuid4()),
                    "content": {
                        "query": query,
                        "max_results": max_results
                    }
                }
                response = self.memory_api_socket.send_message(message)
                if response and response.get("status") == "success":
                    # Emit signal
                    self.memory_search_results.emit(response.get("data", {}))
                    return response.get("data", {})
            
            return {"memories": [], "query": query, "total_results": 0}
            
        except Exception as e:
            logger.error(f"Error searching memories: {str(e)}")
            return {"memories": [], "query": query, "total_results": 0}
    
    def store_memory(self, text, metadata=None):
        """
        Store a memory in the Language Memory System
        
        Args:
            text: Memory text
            metadata: Optional metadata dictionary
            
        Returns:
            True if storage successful, False otherwise
        """
        if not self.connected:
            logger.warning("Not connected to Language Memory System")
            return False
        
        if metadata is None:
            metadata = {}
        
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = time.time()
        
        if self.mock_mode:
            # Simulate storage delay
            time.sleep(random.uniform(0.1, 0.5))
            logger.debug(f"Stored memory (mock): {text[:50]}...")
            return True
        
        try:
            if self.memory_api:
                result = self.memory_api.store_memory(text, metadata)
                return result and result.get("status") == "success"
            
            elif self.memory_api_socket:
                message = {
                    "type": "store_memory",
                    "request_id": str(uuid.uuid4()),
                    "content": {
                        "text": text,
                        "metadata": metadata
                    }
                }
                response = self.memory_api_socket.send_message(message)
                return response and response.get("status") == "success"
            
            return False
            
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            return False
    
    def store_conversation(self, message, metadata=None):
        """
        Store a conversation message in the Language Memory System
        
        Args:
            message: Message text
            metadata: Optional metadata dictionary
            
        Returns:
            True if storage successful, False otherwise
        """
        if not self.connected:
            logger.warning("Not connected to Language Memory System")
            return False
        
        if metadata is None:
            metadata = {}
        
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = time.time()
        
        # Add message type if not present
        if "type" not in metadata:
            metadata["type"] = "conversation"
        
        if self.mock_mode:
            # Simulate storage delay
            time.sleep(random.uniform(0.1, 0.3))
            logger.debug(f"Stored conversation (mock): {message[:50]}...")
            return True
        
        try:
            if self.memory_api:
                result = self.memory_api.store_conversation(message, metadata)
                return result and result.get("status") == "success"
            
            elif self.memory_api_socket:
                message_obj = {
                    "type": "store_conversation",
                    "request_id": str(uuid.uuid4()),
                    "content": {
                        "message": message,
                        "metadata": metadata
                    }
                }
                response = self.memory_api_socket.send_message(message_obj)
                return response and response.get("status") == "success"
            
            return False
            
        except Exception as e:
            logger.error(f"Error storing conversation: {str(e)}")
            return False
    
    def get_memory_stats(self):
        """
        Get memory statistics from the Language Memory System
        
        Returns:
            Dictionary with memory statistics
        """
        if not self.connected:
            logger.warning("Not connected to Language Memory System")
            return {}
        
        if self.mock_mode:
            # Use cached data or generate if not available
            if not self.cache["stats"]:
                self.cache["stats"] = self._generate_mock_stats()
            
            # Emit signal
            self.memory_stats_updated.emit(self.cache["stats"])
            
            return self.cache["stats"]
        
        try:
            if self.memory_api:
                stats = self.memory_api.get_memory_stats()
                if stats:
                    # Update cache
                    self.cache["stats"] = stats
                    
                    # Emit signal
                    self.memory_stats_updated.emit(stats)
                    
                    return stats
            
            elif self.memory_api_socket:
                message = {
                    "type": "get_stats",
                    "request_id": str(uuid.uuid4())
                }
                response = self.memory_api_socket.send_message(message)
                if response and response.get("status") == "success":
                    stats = response.get("data", {}).get("stats", {})
                    
                    # Update cache
                    self.cache["stats"] = stats
                    
                    # Emit signal
                    self.memory_stats_updated.emit(stats)
                    
                    return stats
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {str(e)}")
            return {}
    
    def generate_fractal_parameters(self, topic, complexity=0.5):
        """
        Generate fractal visualization parameters based on a memory topic
        
        Args:
            topic: Memory topic name or ID
            complexity: Complexity factor (0.0-1.0)
            
        Returns:
            Dictionary with fractal parameters
        """
        if not self.connected:
            logger.warning("Not connected to Language Memory System")
            return {}
        
        if self.mock_mode:
            # Check cache first
            cache_key = f"{topic}_{complexity}"
            if cache_key in self.cache["fractal_params"]:
                return self.cache["fractal_params"][cache_key]
            
            # Generate mock parameters
            params = self._generate_mock_fractal_parameters(topic, complexity)
            
            # Cache the result
            self.cache["fractal_params"][cache_key] = params
            
            return params
        
        try:
            if self.memory_api:
                # Try to use memory API if it has this method
                if hasattr(self.memory_api, "generate_fractal_parameters"):
                    return self.memory_api.generate_fractal_parameters(topic, complexity)
            
            elif self.memory_api_socket:
                message = {
                    "type": "generate_fractal_parameters",
                    "request_id": str(uuid.uuid4()),
                    "content": {
                        "topic": topic,
                        "complexity": complexity
                    }
                }
                response = self.memory_api_socket.send_message(message)
                if response and response.get("status") == "success":
                    return response.get("data", {})
            
            # Fallback to generating them locally
            return self._generate_mock_fractal_parameters(topic, complexity)
            
        except Exception as e:
            logger.error(f"Error generating fractal parameters: {str(e)}")
            return {}
    
    def enhance_message_with_memory(self, message, enhance_mode="combined"):
        """
        Enhance a message with memory context
        
        Args:
            message: Message text
            enhance_mode: Enhancement mode ("contextual", "combined", "synthesized")
            
        Returns:
            Dictionary with enhanced message
        """
        if not self.connected:
            logger.warning("Not connected to Language Memory System")
            return {"enhanced_message": message, "enhanced_context": ""}
        
        if self.mock_mode:
            # Simulate processing delay
            time.sleep(random.uniform(0.2, 0.8))
            
            # Generate mock enhanced context
            words = message.split()
            if len(words) > 3:
                selected_words = random.sample(words, min(3, len(words)))
                context = f"Previous discussions mentioned {', '.join(selected_words)}. "
                context += "This relates to neural networks and pattern recognition."
            else:
                context = "This relates to previous discussions about neural networks and pattern recognition."
            
            return {
                "enhanced_message": message,
                "enhanced_context": context,
                "enhance_mode": enhance_mode
            }
        
        try:
            if self.memory_api:
                # Use memory API if it has this method
                if hasattr(self.memory_api, "enhance_message_with_memory"):
                    result = self.memory_api.enhance_message_with_memory(
                        message=message,
                        enhance_mode=enhance_mode
                    )
                    return result
            
            elif self.memory_api_socket:
                message_obj = {
                    "type": "enhance_message",
                    "request_id": str(uuid.uuid4()),
                    "content": {
                        "message": message,
                        "enhance_mode": enhance_mode
                    }
                }
                response = self.memory_api_socket.send_message(message_obj)
                if response and response.get("status") == "success":
                    return response.get("data", {})
            
            # Fallback to simple response
            return {"enhanced_message": message, "enhanced_context": ""}
            
        except Exception as e:
            logger.error(f"Error enhancing message: {str(e)}")
            return {"enhanced_message": message, "enhanced_context": ""}
    
    def _mock_data_thread(self):
        """Thread for generating mock data updates"""
        while self.connected:
            try:
                # Update cached stats occasionally
                if random.random() < 0.2:  # 20% chance each cycle
                    stats = self._generate_mock_stats()
                    self.cache["stats"] = stats
                    self.memory_stats_updated.emit(stats)
                    logger.debug("Updated mock memory stats")
                
                # Update cached topics occasionally
                if random.random() < 0.1:  # 10% chance each cycle
                    topics = self._generate_mock_topics()
                    self.cache["topics"] = topics
                    self.memory_topics_updated.emit(topics)
                    logger.debug("Updated mock memory topics")
                
            except Exception as e:
                logger.error(f"Error in mock data thread: {str(e)}")
            
            # Sleep for a bit
            time.sleep(random.uniform(5, 15))
    
    def _generate_mock_topics(self):
        """Generate mock memory topics"""
        topics = [
            {"id": "neural_networks", "name": "Neural Networks", "count": random.randint(20, 50)},
            {"id": "consciousness", "name": "Consciousness", "count": random.randint(15, 40)},
            {"id": "pattern_recognition", "name": "Pattern Recognition", "count": random.randint(10, 30)},
            {"id": "language_processing", "name": "Language Processing", "count": random.randint(25, 45)},
            {"id": "memory_systems", "name": "Memory Systems", "count": random.randint(15, 35)},
            {"id": "learning_algorithms", "name": "Learning Algorithms", "count": random.randint(10, 25)},
            {"id": "fractal_patterns", "name": "Fractal Patterns", "count": random.randint(5, 20)}
        ]
        
        return topics
    
    def _generate_mock_search_results(self, query, max_results):
        """Generate mock memory search results"""
        # Generate memories based on query
        memories = []
        for i in range(max_results):
            memories.append({
                "id": f"memory_{i}",
                "text": f"This is a memory about {query} with some additional context and details "
                        f"for testing purposes. Item {i+1} of {max_results}.",
                "timestamp": time.time() - random.randint(0, 86400 * 30),  # Random time in last 30 days
                "relevance": round(random.uniform(0.5, 0.95), 2),
                "topic": random.choice(["Neural Networks", "Consciousness", "Pattern Recognition"]),
                "source": random.choice(["conversation", "system", "user"])
            })
        
        return {
            "memories": memories,
            "query": query,
            "total_results": max_results
        }
    
    def _generate_mock_stats(self):
        """Generate mock memory statistics"""
        return {
            "total_memories": random.randint(1000, 5000),
            "total_conversations": random.randint(50, 200),
            "total_topics": random.randint(20, 50),
            "top_topics": [
                {"topic": "neural networks", "count": random.randint(40, 100)},
                {"topic": "consciousness", "count": random.randint(30, 80)},
                {"topic": "machine learning", "count": random.randint(20, 60)},
                {"topic": "pattern recognition", "count": random.randint(15, 50)},
                {"topic": "language processing", "count": random.randint(10, 40)}
            ],
            "memory_growth": {
                "daily": random.randint(10, 50),
                "weekly": random.randint(50, 300),
                "monthly": random.randint(200, 1000)
            },
            "last_updated": time.time()
        }
    
    def _generate_mock_fractal_parameters(self, topic, complexity):
        """Generate mock fractal parameters for visualization"""
        # Create a deterministic but varied set of parameters based on the topic and complexity
        topic_hash = hash(topic) % 10000
        random.seed(topic_hash)
        
        # Base fractal parameters
        params = {
            "topic": topic,
            "complexity": complexity,
            "fractal_type": random.choice(["julia", "mandelbrot", "tree", "neural"]),
            "recursion_depth": max(3, min(8, int(3 + (complexity * 5)))),
            "color_scheme": random.choice(["blue", "green", "rainbow", "psychedelic", "monochrome"]),
            "parameters": {
                "scale": 0.5 + (complexity * 0.5),
                "rotation": random.uniform(0, 360),
                "symmetry": random.randint(2, 8),
                "density": 0.2 + (complexity * 0.8),
            },
            "animation": {
                "speed": 0.5 + (complexity * 0.5),
                "wave_factor": random.uniform(0.1, 0.5),
                "pulse_rate": random.uniform(0.2, 2.0)
            }
        }
        
        # Reset random seed
        random.seed()
        
        return params 