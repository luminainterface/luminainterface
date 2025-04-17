#!/usr/bin/env python3
"""
Language Consciousness Node for LUMINA V7

This module implements a specialized consciousness node for language processing
within the V7 Node Consciousness architecture. It handles natural language 
understanding, generation, and semantic processing.
"""

import json
import logging
import queue
import threading
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Callable

try:
    from src.v7.lumina_v7.core.node_consciousness_manager import BaseConsciousnessNode, NodeState
except ImportError:
    logging.warning("❌ Failed to import from core node_consciousness_manager")
    # Fallback definitions for standalone testing
    class NodeState(Enum):
        INACTIVE = 0
        INITIALIZING = 1
        ACTIVE = 2
        PAUSED = 3
        ERROR = 4
        SHUTDOWN = 5
    
    class BaseConsciousnessNode:
        """Fallback base class for standalone testing"""
        def __init__(self, node_id, node_type):
            self.node_id = node_id
            self.node_type = node_type
            self.state = NodeState.INACTIVE


class LanguageCapability(Enum):
    """Capabilities of the Language Node"""
    TEXT_UNDERSTANDING = "text_understanding"
    TEXT_GENERATION = "text_generation"
    SEMANTIC_MAPPING = "semantic_mapping"
    CONTEXTUAL_MEMORY = "contextual_memory"
    CONTRADICTION_DETECTION = "contradiction_detection"


class LanguageNode(BaseConsciousnessNode):
    """
    Language Consciousness Node for handling natural language processing
    and understanding within the V7 Node Consciousness system.
    """
    
    def __init__(self, node_id: str = "language_primary", 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Language Node with configuration settings.
        
        Args:
            node_id: Unique identifier for this language node
            config: Configuration dictionary with settings for the language node
        """
        super().__init__(node_id=node_id, node_type="language")
        
        # Default configuration
        self.default_config = {
            "embedding_dimension": 768,
            "context_window_size": 2048,
            "max_processing_queue": 100,
            "processing_interval": 0.05,  # seconds
            "memory_retention": 0.85,  # factor for memory retention (0-1)
            "active_capabilities": [cap.value for cap in LanguageCapability],
            "contradiction_threshold": 0.7,
            "semantic_depth": 3,
        }
        
        # Merge with provided config
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # Initialize component attributes
        self.capabilities = {cap: True for cap in self.config["active_capabilities"]}
        self.semantic_context = {}
        self.message_queue = queue.Queue(maxsize=self.config["max_processing_queue"])
        self.processing_thread = None
        self.running = False
        self.last_processed_time = 0
        
        # Memory structures
        self.short_term_memory = []
        self.semantic_map = {}
        self.contradictions = []
        
        # Event listeners
        self.event_listeners = {
            "text_processed": [],
            "semantic_update": [],
            "contradiction_detected": [],
            "memory_updated": [],
            "state_changed": []
        }
        
        # Metrics
        self.metrics = {
            "processed_messages": 0,
            "detected_contradictions": 0,
            "semantic_updates": 0,
            "average_processing_time": 0
        }
        
        logging.info(f"✅ Language Node '{node_id}' initialized with {len(self.capabilities)} active capabilities")
    
    def activate(self):
        """Activate the language node and start processing thread"""
        if self.state == NodeState.INACTIVE or self.state == NodeState.PAUSED:
            self.state = NodeState.INITIALIZING
            self._notify_state_change()
            
            # Start processing thread
            self.running = True
            self.processing_thread = threading.Thread(
                target=self._process_queue,
                name=f"LanguageNode-{self.node_id}-Processor",
                daemon=True
            )
            self.processing_thread.start()
            
            logging.info(f"✅ Language Node '{self.node_id}' activated and processing thread started")
            self.state = NodeState.ACTIVE
            self._notify_state_change()
            return True
        return False
    
    def deactivate(self):
        """Deactivate the language node and stop processing"""
        if self.state == NodeState.ACTIVE:
            self.state = NodeState.SHUTDOWN
            self._notify_state_change()
            
            # Stop processing thread
            self.running = False
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)
            
            # Clear queues and temporary memory
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                except queue.Empty:
                    break
            
            logging.info(f"❌ Language Node '{self.node_id}' deactivated")
            self.state = NodeState.INACTIVE
            self._notify_state_change()
            return True
        return False
    
    def receive_message(self, message: Dict[str, Any]) -> bool:
        """
        Receive a message for processing by the language node.
        
        Args:
            message: Dictionary containing the message data with at least
                    'type' and 'content' fields
        
        Returns:
            bool: True if message was queued successfully, False otherwise
        """
        if self.state != NodeState.ACTIVE:
            logging.warning(f"⚠️ Language Node '{self.node_id}' cannot receive messages when not active")
            return False
            
        try:
            # Add timestamp if not present
            if 'timestamp' not in message:
                message['timestamp'] = time.time()
                
            # Add to processing queue
            self.message_queue.put_nowait(message)
            return True
        except queue.Full:
            logging.error(f"❌ Language Node '{self.node_id}' message queue is full")
            return False
    
    def _process_queue(self):
        """Process messages from the queue in a separate thread"""
        while self.running:
            try:
                if not self.message_queue.empty():
                    message = self.message_queue.get(timeout=0.1)
                    self._process_message(message)
                    self.message_queue.task_done()
                else:
                    time.sleep(self.config["processing_interval"])
            except queue.Empty:
                time.sleep(self.config["processing_interval"])
            except Exception as e:
                logging.error(f"❌ Error in language node processing: {str(e)}")
                self.state = NodeState.ERROR
                self._notify_state_change()
                time.sleep(1.0)  # Pause briefly before continuing
    
    def _process_message(self, message: Dict[str, Any]):
        """
        Process a single message based on its type.
        
        Args:
            message: The message dictionary to process
        """
        start_time = time.time()
        message_type = message.get('type', 'unknown')
        
        try:
            if message_type == 'text_input':
                self._process_text(message)
            elif message_type == 'semantic_query':
                self._process_semantic_query(message)
            elif message_type == 'memory_request':
                self._process_memory_request(message)
            elif message_type == 'contradiction_check':
                self._check_contradictions(message)
            elif message_type == 'system_command':
                self._process_system_command(message)
            else:
                logging.warning(f"⚠️ Unknown message type: {message_type}")
            
            # Update metrics
            self.metrics["processed_messages"] += 1
            processing_time = time.time() - start_time
            self.metrics["average_processing_time"] = (
                (self.metrics["average_processing_time"] * (self.metrics["processed_messages"] - 1) +
                processing_time) / self.metrics["processed_messages"]
            )
            
            self.last_processed_time = time.time()
        except Exception as e:
            logging.error(f"❌ Error processing message of type {message_type}: {str(e)}")
    
    def _process_text(self, message: Dict[str, Any]):
        """Process text input messages"""
        text = message.get('content', '')
        if not text:
            return
            
        # Store in short-term memory
        memory_item = {
            'text': text,
            'timestamp': message.get('timestamp', time.time()),
            'source': message.get('source', 'unknown'),
            'processed': True
        }
        self.short_term_memory.append(memory_item)
        
        # Maintain memory size limit
        while len(self.short_term_memory) > self.config["context_window_size"]:
            self.short_term_memory.pop(0)
        
        # Update semantic map
        self._update_semantic_map(text, memory_item)
        
        # Trigger event for text processed
        self._trigger_event("text_processed", {
            'text': text,
            'timestamp': memory_item['timestamp'],
            'memory_id': len(self.short_term_memory) - 1
        })
        
        logging.debug(f"✅ Processed text: {text[:50]}...")
    
    def _process_semantic_query(self, message: Dict[str, Any]):
        """Process semantic query messages"""
        query = message.get('content', '')
        depth = message.get('depth', self.config["semantic_depth"])
        
        if not query:
            return
            
        # Search the semantic map for relevant connections
        results = {}
        if query in self.semantic_map:
            results = self.semantic_map[query]
        else:
            # Search for similar concepts (simple implementation)
            for concept, connections in self.semantic_map.items():
                if query in concept or concept in query:
                    results[concept] = connections
        
        # Prepare response
        response = {
            'query': query,
            'results': results,
            'timestamp': time.time()
        }
        
        # Send response
        self._trigger_event("semantic_update", response)
        
        logging.debug(f"✅ Processed semantic query: {query}")
    
    def _process_memory_request(self, message: Dict[str, Any]):
        """Process memory retrieval requests"""
        query = message.get('content', '')
        limit = message.get('limit', 10)
        
        if not query:
            return
            
        # Simple memory search (could be enhanced with embeddings)
        matches = []
        for i, memory in enumerate(self.short_term_memory):
            if query.lower() in memory['text'].lower():
                matches.append({
                    'memory_id': i,
                    'text': memory['text'],
                    'timestamp': memory['timestamp'],
                    'source': memory['source']
                })
                if len(matches) >= limit:
                    break
        
        # Prepare response
        response = {
            'query': query,
            'memories': matches,
            'timestamp': time.time()
        }
        
        # Send response
        self._trigger_event("memory_updated", response)
        
        logging.debug(f"✅ Processed memory request: {query}, found {len(matches)} matches")
    
    def _check_contradictions(self, message: Dict[str, Any]):
        """Check for contradictions in statements"""
        statement = message.get('content', '')
        context = message.get('context', [])
        
        if not statement:
            return
            
        # Very simple contradiction detection (would be enhanced with NLU)
        contradictions = []
        
        # Look through recent memories for potential contradictions
        for i, memory in enumerate(self.short_term_memory[-20:]):
            # Extremely basic contradiction check - just for demonstration
            # In a real implementation, this would use language understanding
            if any(negation in statement.lower() for negation in ["not ", "n't", "never"]):
                # Extract what might be negated
                potential_contradiction = statement.lower()
                for neg in ["not ", "n't", "never"]:
                    potential_contradiction = potential_contradiction.replace(neg, "")
                
                # Compare with memory
                memory_text = memory['text'].lower()
                if potential_contradiction.strip() in memory_text and "not " not in memory_text and "n't" not in memory_text:
                    contradictions.append({
                        'statement': statement,
                        'contradiction': memory['text'],
                        'memory_id': len(self.short_term_memory) - 20 + i,
                        'confidence': 0.8  # Arbitrary confidence value
                    })
        
        if contradictions:
            # Update metrics
            self.metrics["detected_contradictions"] += len(contradictions)
            self.contradictions.extend(contradictions)
            
            # Trigger event
            self._trigger_event("contradiction_detected", {
                'statement': statement,
                'contradictions': contradictions,
                'timestamp': time.time()
            })
            
            logging.info(f"⚠️ Detected {len(contradictions)} contradictions for: {statement[:50]}...")
    
    def _process_system_command(self, message: Dict[str, Any]):
        """Process system commands"""
        command = message.get('command', '')
        params = message.get('params', {})
        
        if command == 'clear_memory':
            self.short_term_memory = []
            logging.info("🧹 Cleared language node short-term memory")
        elif command == 'update_config':
            self.config.update(params)
            logging.info(f"⚙️ Updated language node configuration: {params}")
        elif command == 'get_status':
            status = self.get_status()
            self._trigger_event("state_changed", status)
        elif command == 'pause':
            if self.state == NodeState.ACTIVE:
                self.state = NodeState.PAUSED
                self._notify_state_change()
        elif command == 'resume':
            if self.state == NodeState.PAUSED:
                self.state = NodeState.ACTIVE
                self._notify_state_change()
    
    def _update_semantic_map(self, text: str, memory_item: Dict[str, Any]):
        """
        Update the semantic map with new text information.
        This is a simplified implementation that would be replaced with 
        more sophisticated semantic understanding in a real system.
        """
        # Very simple semantic mapping for demonstration
        words = text.lower().split()
        for word in words:
            # Skip short words and common stop words
            if len(word) <= 3 or word in ["the", "and", "that", "this", "with", "from"]:
                continue
                
            # Add to semantic map
            if word not in self.semantic_map:
                self.semantic_map[word] = {
                    'frequency': 1,
                    'memories': [len(self.short_term_memory) - 1],
                    'connections': {}
                }
            else:
                self.semantic_map[word]['frequency'] += 1
                if len(self.short_term_memory) - 1 not in self.semantic_map[word]['memories']:
                    self.semantic_map[word]['memories'].append(len(self.short_term_memory) - 1)
            
            # Add connections between words in the same input
            for other_word in words:
                if other_word != word and len(other_word) > 3:
                    if other_word not in self.semantic_map[word]['connections']:
                        self.semantic_map[word]['connections'][other_word] = 1
                    else:
                        self.semantic_map[word]['connections'][other_word] += 1
        
        # Update metrics
        self.metrics["semantic_updates"] += 1
        
        # Trigger event
        self._trigger_event("semantic_update", {
            'updated_concepts': [w for w in words if len(w) > 3],
            'timestamp': time.time()
        })
    
    def _trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Trigger an event for all registered listeners"""
        if event_type not in self.event_listeners:
            return
            
        for listener in self.event_listeners[event_type]:
            try:
                listener(data)
            except Exception as e:
                logging.error(f"❌ Error in event listener for {event_type}: {str(e)}")
    
    def _notify_state_change(self):
        """Notify listeners of state changes"""
        self._trigger_event("state_changed", {
            'node_id': self.node_id,
            'state': self.state.name,
            'timestamp': time.time()
        })
    
    def register_event_listener(self, event_type: str, callback: Callable):
        """
        Register a callback function for a specific event type.
        
        Args:
            event_type: Type of event to listen for
            callback: Function to call when event occurs
        
        Returns:
            bool: True if registered successfully, False otherwise
        """
        if event_type not in self.event_listeners:
            logging.warning(f"⚠️ Unknown event type: {event_type}")
            return False
            
        self.event_listeners[event_type].append(callback)
        return True
    
    def unregister_event_listener(self, event_type: str, callback: Callable):
        """
        Unregister a callback function for a specific event type.
        
        Args:
            event_type: Type of event to unregister from
            callback: Function to unregister
            
        Returns:
            bool: True if unregistered successfully, False otherwise
        """
        if event_type not in self.event_listeners:
            return False
            
        if callback in self.event_listeners[event_type]:
            self.event_listeners[event_type].remove(callback)
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the language node.
        
        Returns:
            dict: Status information about the node
        """
        return {
            'node_id': self.node_id,
            'node_type': 'language',
            'state': self.state.name,
            'metrics': self.metrics,
            'queue_size': self.message_queue.qsize(),
            'memory_size': len(self.short_term_memory),
            'semantic_concepts': len(self.semantic_map),
            'contradictions': len(self.contradictions),
            'capabilities': self.capabilities,
            'last_processed': self.last_processed_time
        }
    
    def get_ui_widget(self):
        """
        Get a UI widget representing this language node.
        
        Returns:
            Optional UI widget for displaying language node status
        """
        try:
            from src.v7.ui.widgets.language_node_widget import LanguageNodeWidget
            return LanguageNodeWidget(self)
        except ImportError:
            logging.warning("⚠️ Language node widget not available")
            return None


# For standalone testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create and test language node
    node = LanguageNode(node_id="test_language_node")
    node.activate()
    
    # Send test messages
    node.receive_message({
        'type': 'text_input',
        'content': 'The sky is blue and the clouds are white.',
        'source': 'test'
    })
    
    time.sleep(1.0)
    
    node.receive_message({
        'type': 'text_input',
        'content': 'The sky is not blue today, it is gray.',
        'source': 'test'
    })
    
    time.sleep(1.0)
    
    node.receive_message({
        'type': 'contradiction_check',
        'content': 'The sky is not blue today, it is gray.',
        'source': 'test'
    })
    
    time.sleep(1.0)
    
    # Print status
    print(json.dumps(node.get_status(), indent=2))
    
    # Clean up
    node.deactivate() 