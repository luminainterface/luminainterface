#!/usr/bin/env python
"""
Language Consciousness Node for V7

This module provides a specialized consciousness node for language processing
capabilities within the V7 Node Consciousness system, building on the core
language memory and processing components.
"""

import logging
import threading
import time
import os
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from queue import Queue

from src.v7.node_consciousness_manager import ConsciousnessNode, NodeState

# Configure logging
logger = logging.getLogger(__name__)

# Try to import language memory components - use mock implementations if not available
try:
    from src.language.language_memory import LanguageMemory
    LANGUAGE_MEMORY_AVAILABLE = True
    logger.info("Language Memory module is available")
except ImportError:
    LANGUAGE_MEMORY_AVAILABLE = False
    logger.warning("Language Memory module is not available - using mock implementation")

try:
    # Optional LLM components
    from src.language.llm_integration import LLMIntegration
    LLM_INTEGRATION_AVAILABLE = True
    logger.info("LLM Integration module is available")
except ImportError:
    LLM_INTEGRATION_AVAILABLE = False
    logger.warning("LLM Integration module is not available - disabling LLM features")


class MockLanguageMemory:
    """
    Mock implementation of Language Memory for when the actual module is not available.
    Provides minimal functionality for testing and development.
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.associations = {}
        self.sentences = []
        self.patterns = {}
        self.initialized = True
        logger.info("Initialized Mock Language Memory")
        
    def process_text(self, text, **kwargs):
        words = text.lower().split()
        result = {
            'word_count': len(words),
            'sentence_count': text.count('.') + text.count('!') + text.count('?'),
            'unique_words': len(set(words)),
            'associations_found': 0,
            'patterns_recognized': 0
        }
        logger.debug(f"Mock Language Memory processed: {text[:50]}...")
        return result
    
    def store_association(self, word1, word2, strength=0.5):
        key = f"{word1}:{word2}"
        self.associations[key] = strength
        return True
    
    def store_sentence(self, sentence, metadata=None):
        self.sentences.append((sentence, metadata or {}))
        return len(self.sentences) - 1
    
    def store_pattern(self, pattern_name, pattern_data):
        self.patterns[pattern_name] = pattern_data
        return True
    
    def get_associations(self, word, min_strength=0.0):
        return []
    
    def get_sentences(self, query=None, limit=10):
        if not query:
            return self.sentences[:limit]
        return []
    
    def get_patterns(self, pattern_name=None):
        if pattern_name:
            return self.patterns.get(pattern_name, {})
        return self.patterns


class LanguageConsciousnessNode(ConsciousnessNode):
    """
    Specialized consciousness node for language processing capabilities.
    
    This node integrates language memory, pattern recognition, and
    linguistic processing to provide advanced language capabilities
    within the V7 Node Consciousness system.
    """
    
    def __init__(self, node_id: Optional[str] = None, 
                 name: str = 'Language Node', 
                 config: Optional[Dict[str, Any]] = None):
        """Initialize the Language Consciousness Node."""
        super().__init__(node_id=node_id, name=name, node_type='language')
        
        self.config = config or {}
        
        # Initialize with default settings
        self.memory_path = self.config.get('memory_path', './data/language_memory')
        self.use_llm = self.config.get('use_llm', LLM_INTEGRATION_AVAILABLE)
        self.llm_weight = self.config.get('llm_weight', 0.5)
        self.memory_persistence = self.config.get('memory_persistence', True)
        self.auto_learn = self.config.get('auto_learn', True)
        
        # Components and state
        self.language_memory = None
        self.llm_integration = None
        self.message_queue = Queue()
        self.processor_thread = None
        self._running = False
        self.processing_lock = threading.RLock()
        self.sentiment_values = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        self.context_memory = []
        self.pattern_strength = {}
        
        # Stats tracking
        self.stats = {
            'texts_processed': 0,
            'words_processed': 0,
            'sentences_processed': 0,
            'patterns_recognized': 0,
            'associations_learned': 0
        }
        
        # Specialized attributes for language node
        self.attributes.update({
            'language_version': '0.1.0',
            'language_capability': 'basic',
            'llm_integrated': False,
            'sentiment_tracking': True,
            'context_aware': True
        })
        
        logger.info(f"Language Consciousness Node initialized: {self.name}")
    
    def _initialize(self) -> None:
        """Initialize the language node components."""
        try:
            # Create memory directory if it doesn't exist
            if self.memory_persistence:
                os.makedirs(self.memory_path, exist_ok=True)
            
            # Initialize language memory
            if LANGUAGE_MEMORY_AVAILABLE:
                memory_config = {
                    'memory_path': self.memory_path,
                    'persistence': self.memory_persistence
                }
                self.language_memory = LanguageMemory(config=memory_config)
                self.attributes['language_capability'] = 'advanced'
            else:
                self.language_memory = MockLanguageMemory(config={'memory_path': self.memory_path})
                self.attributes['language_capability'] = 'basic'
            
            # Initialize LLM integration if available and enabled
            if self.use_llm and LLM_INTEGRATION_AVAILABLE:
                self.llm_integration = LLMIntegration(weight=self.llm_weight)
                self.attributes['llm_integrated'] = True
            
            # Start the processing thread
            self._running = True
            self.processor_thread = threading.Thread(
                target=self._message_processor,
                name="LanguageNodeProcessor",
                daemon=True
            )
            self.processor_thread.start()
            
            logger.info(f"Language node {self.name} fully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing language node {self.name}: {str(e)}")
            raise
    
    def _cleanup(self) -> None:
        """Clean up resources used by the language node."""
        # Stop the processing thread
        self._running = False
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=2.0)
        
        # Persist memory if needed
        if self.memory_persistence and hasattr(self.language_memory, 'save'):
            try:
                logger.info(f"Persisting language memory for node {self.name}")
                getattr(self.language_memory, 'save')()
            except Exception as e:
                logger.error(f"Error persisting language memory: {str(e)}")
        
        # Clear resources
        self.language_memory = None
        self.llm_integration = None
        
        logger.info(f"Language node {self.name} cleaned up")
    
    def _process_impl(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of language processing logic."""
        start_time = time.time()
        
        # Process different input types
        if 'text' in data:
            return self._process_text(data['text'], data.get('metadata', {}))
        elif 'pattern' in data:
            return self._process_pattern(data['pattern'], data.get('metadata', {}))
        elif 'query' in data:
            return self._process_query(data['query'], data.get('params', {}))
        elif 'batch' in data:
            return self._process_batch(data['batch'], data.get('params', {}))
        else:
            logger.warning(f"Unsupported data format in language node {self.name}")
            return {
                'success': False,
                'error': 'Unsupported data format',
                'expected': ['text', 'pattern', 'query', 'batch'],
                'received': list(data.keys())
            }
    
    def _learn_impl(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of language learning logic."""
        start_time = time.time()
        results = {
            'success': True,
            'items_processed': 0,
            'memory_updates': 0,
            'patterns_learned': 0,
            'associations_formed': 0,
            'processing_time': 0
        }
        
        try:
            # Process different types of training data
            if 'texts' in training_data:
                # Process multiple texts
                for text in training_data['texts']:
                    if isinstance(text, str):
                        self._learn_from_text(text, {})
                    elif isinstance(text, dict) and 'content' in text:
                        self._learn_from_text(text['content'], text.get('metadata', {}))
                    
                    results['items_processed'] += 1
            
            if 'patterns' in training_data:
                # Learn patterns
                for pattern in training_data['patterns']:
                    if isinstance(pattern, dict) and 'name' in pattern and 'data' in pattern:
                        success = self.language_memory.store_pattern(
                            pattern['name'], pattern['data']
                        )
                        if success:
                            results['patterns_learned'] += 1
                            results['memory_updates'] += 1
                
                results['items_processed'] += len(training_data['patterns'])
            
            if 'associations' in training_data:
                # Learn word associations
                for assoc in training_data['associations']:
                    if isinstance(assoc, dict) and 'word1' in assoc and 'word2' in assoc:
                        success = self.language_memory.store_association(
                            assoc['word1'], 
                            assoc['word2'],
                            assoc.get('strength', 0.5)
                        )
                        if success:
                            results['associations_formed'] += 1
                            results['memory_updates'] += 1
                
                results['items_processed'] += len(training_data['associations'])
            
            # Update stats
            self.stats['patterns_recognized'] += results['patterns_learned']
            self.stats['associations_learned'] += results['associations_formed']
            
            results['processing_time'] = time.time() - start_time
            return results
            
        except Exception as e:
            logger.error(f"Error in language learning for node {self.name}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'items_processed': results['items_processed']
            }
    
    def _message_processor(self) -> None:
        """Background thread for processing queued messages."""
        while self._running:
            try:
                # Get message with a timeout to allow checking _running periodically
                try:
                    message = self.message_queue.get(timeout=0.5)
                except:
                    continue
                
                # Process the message
                if message['type'] == 'text':
                    result = self._process_text(message['content'], message['metadata'])
                elif message['type'] == 'pattern':
                    result = self._process_pattern(message['content'], message['metadata'])
                else:
                    result = {'success': False, 'error': f"Unknown message type: {message['type']}"}
                
                # Trigger event with result
                self._trigger_event('message_processed', {
                    'message_type': message['type'],
                    'result': result
                })
                
                # Mark task as done
                self.message_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in message processor for language node {self.name}: {str(e)}")
                time.sleep(0.1)  # Short delay after error
    
    def _process_text(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a text input."""
        if not text:
            return {'success': False, 'error': 'Empty text input'}
        
        metadata = metadata or {}
        with self.processing_lock:
            try:
                # Process with language memory
                memory_result = self.language_memory.process_text(
                    text, 
                    context=metadata.get('context'),
                    store=self.auto_learn
                )
                
                # Update stats
                self.stats['texts_processed'] += 1
                self.stats['words_processed'] += memory_result.get('word_count', 0)
                self.stats['sentences_processed'] += memory_result.get('sentence_count', 0)
                
                # Store in context memory
                self._update_context_memory(text, metadata, memory_result)
                
                # Process with LLM if available
                llm_result = {}
                if self.use_llm and self.llm_integration:
                    llm_result = self.llm_integration.enhance_processing(
                        text, 
                        memory_result,
                        weight=self.llm_weight
                    )
                
                # Create combined result
                result = {
                    'success': True,
                    'text': text,
                    'language_score': memory_result.get('language_score', 0.0),
                    'word_count': memory_result.get('word_count', 0),
                    'sentence_count': memory_result.get('sentence_count', 0),
                    'patterns_detected': memory_result.get('patterns_detected', []),
                    'associations_found': memory_result.get('associations_found', 0),
                    'sentiment': self._analyze_sentiment(text),
                    'memory_result': memory_result,
                    'llm_result': llm_result,
                    'llm_enhanced': bool(llm_result),
                    'stored_in_memory': self.auto_learn,
                    'context_id': len(self.context_memory) - 1 if self.context_memory else None
                }
                
                return result
                
            except Exception as e:
                logger.error(f"Error processing text in language node {self.name}: {str(e)}")
                return {'success': False, 'error': str(e)}
    
    def _process_pattern(self, pattern: Dict[str, Any], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a pattern input."""
        if not pattern or not isinstance(pattern, dict):
            return {'success': False, 'error': 'Invalid pattern input'}
        
        metadata = metadata or {}
        with self.processing_lock:
            try:
                pattern_name = pattern.get('name', f"pattern_{int(time.time())}")
                pattern_data = pattern.get('data', {})
                
                # Store the pattern if auto-learn is enabled
                stored = False
                if self.auto_learn:
                    stored = self.language_memory.store_pattern(pattern_name, pattern_data)
                
                # Update pattern strength tracking
                if pattern_name not in self.pattern_strength:
                    self.pattern_strength[pattern_name] = 0.0
                
                # Increment strength (max 1.0)
                self.pattern_strength[pattern_name] = min(
                    1.0, 
                    self.pattern_strength[pattern_name] + 0.1
                )
                
                # Return result
                return {
                    'success': True,
                    'pattern_name': pattern_name,
                    'stored': stored,
                    'current_strength': self.pattern_strength[pattern_name],
                    'auto_learn': self.auto_learn
                }
                
            except Exception as e:
                logger.error(f"Error processing pattern in language node {self.name}: {str(e)}")
                return {'success': False, 'error': str(e)}
    
    def _process_query(self, query: Dict[str, Any], params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a query against the language memory."""
        if not query or not isinstance(query, dict):
            return {'success': False, 'error': 'Invalid query input'}
        
        params = params or {}
        with self.processing_lock:
            try:
                query_type = query.get('type', 'general')
                
                if query_type == 'associations':
                    # Query for word associations
                    word = query.get('word')
                    if not word:
                        return {'success': False, 'error': 'No word specified for associations query'}
                    
                    min_strength = params.get('min_strength', 0.0)
                    associations = self.language_memory.get_associations(word, min_strength)
                    
                    return {
                        'success': True,
                        'query_type': 'associations',
                        'word': word,
                        'associations': associations,
                        'count': len(associations)
                    }
                    
                elif query_type == 'sentences':
                    # Query for stored sentences
                    search = query.get('search')
                    limit = params.get('limit', 10)
                    sentences = self.language_memory.get_sentences(search, limit)
                    
                    return {
                        'success': True,
                        'query_type': 'sentences',
                        'search': search,
                        'sentences': sentences,
                        'count': len(sentences)
                    }
                    
                elif query_type == 'patterns':
                    # Query for stored patterns
                    pattern_name = query.get('pattern_name')
                    patterns = self.language_memory.get_patterns(pattern_name)
                    
                    return {
                        'success': True,
                        'query_type': 'patterns',
                        'pattern_name': pattern_name,
                        'patterns': patterns,
                        'count': len(patterns) if isinstance(patterns, (list, dict)) else 1
                    }
                    
                elif query_type == 'context':
                    # Query context memory
                    limit = params.get('limit', 5)
                    offset = params.get('offset', 0)
                    contexts = self.context_memory[offset:offset+limit] if self.context_memory else []
                    
                    return {
                        'success': True,
                        'query_type': 'context',
                        'contexts': contexts,
                        'count': len(contexts),
                        'total': len(self.context_memory)
                    }
                    
                elif query_type == 'stats':
                    # Return node stats
                    return {
                        'success': True,
                        'query_type': 'stats',
                        'stats': self.stats,
                        'sentiment_values': self.sentiment_values,
                        'pattern_strength': self.pattern_strength
                    }
                    
                else:
                    return {'success': False, 'error': f"Unknown query type: {query_type}"}
                    
            except Exception as e:
                logger.error(f"Error processing query in language node {self.name}: {str(e)}")
                return {'success': False, 'error': str(e)}
    
    def _process_batch(self, batch: List[Dict[str, Any]], params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a batch of items."""
        if not batch or not isinstance(batch, list):
            return {'success': False, 'error': 'Invalid batch input'}
        
        params = params or {}
        results = []
        success_count = 0
        
        with self.processing_lock:
            for item in batch:
                if not isinstance(item, dict):
                    results.append({'success': False, 'error': 'Item must be a dictionary'})
                    continue
                
                item_type = item.get('type')
                content = item.get('content')
                metadata = item.get('metadata', {})
                
                if item_type == 'text' and content:
                    result = self._process_text(content, metadata)
                elif item_type == 'pattern' and content:
                    result = self._process_pattern(content, metadata)
                elif item_type == 'query' and content:
                    result = self._process_query(content, metadata)
                else:
                    result = {'success': False, 'error': f"Invalid or unsupported item type: {item_type}"}
                
                results.append(result)
                if result.get('success', False):
                    success_count += 1
        
        return {
            'success': True,
            'batch_size': len(batch),
            'success_count': success_count,
            'results': results
        }
    
    def _learn_from_text(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Learn from a text sample."""
        if not text:
            return {'success': False, 'error': 'Empty text input'}
        
        metadata = metadata or {}
        try:
            # Process with language memory with storage enabled
            memory_result = self.language_memory.process_text(
                text, 
                context=metadata.get('context'),
                store=True  # Force storage
            )
            
            # Update stats
            self.stats['texts_processed'] += 1
            self.stats['words_processed'] += memory_result.get('word_count', 0)
            self.stats['sentences_processed'] += memory_result.get('sentence_count', 0)
            
            return {
                'success': True,
                'text_length': len(text),
                'words_processed': memory_result.get('word_count', 0),
                'sentences_processed': memory_result.get('sentence_count', 0),
                'associations_learned': memory_result.get('associations_stored', 0)
            }
            
        except Exception as e:
            logger.error(f"Error learning from text in language node {self.name}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _update_context_memory(self, text: str, metadata: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Update the context memory with the latest processed text."""
        # Limit context memory size
        max_contexts = self.config.get('max_context_memory', 50)
        if len(self.context_memory) >= max_contexts:
            self.context_memory.pop(0)  # Remove oldest item
        
        # Add new context
        self.context_memory.append({
            'text': text,
            'timestamp': time.time(),
            'metadata': metadata,
            'summary': {
                'word_count': result.get('word_count', 0),
                'sentence_count': result.get('sentence_count', 0),
                'language_score': result.get('language_score', 0.0)
            }
        })
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of the text."""
        # Very simple sentiment analysis for demonstration
        # In a real implementation, this would use more sophisticated algorithms
        positive_words = {'good', 'great', 'excellent', 'happy', 'positive', 'joy', 'love', 'like'}
        negative_words = {'bad', 'terrible', 'awful', 'sad', 'negative', 'hate', 'dislike', 'angry'}
        
        words = text.lower().split()
        word_set = set(words)
        
        pos_count = sum(1 for word in word_set if word in positive_words)
        neg_count = sum(1 for word in word_set if word in negative_words)
        total = max(1, len(word_set))
        
        positive_score = pos_count / total
        negative_score = neg_count / total
        neutral_score = 1.0 - (positive_score + negative_score)
        
        # Update running sentiment values with decay
        decay = 0.8
        self.sentiment_values['positive'] = (decay * self.sentiment_values['positive'] + 
                                            (1-decay) * positive_score)
        self.sentiment_values['negative'] = (decay * self.sentiment_values['negative'] + 
                                            (1-decay) * negative_score)
        self.sentiment_values['neutral'] = (decay * self.sentiment_values['neutral'] + 
                                           (1-decay) * neutral_score)
        
        return {
            'positive': positive_score,
            'negative': negative_score,
            'neutral': neutral_score,
            'primary': 'positive' if positive_score > negative_score and positive_score > neutral_score else
                      'negative' if negative_score > positive_score and negative_score > neutral_score else
                      'neutral'
        }
    
    def queue_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Queue a text message for asynchronous processing.
        
        Args:
            text: The text to process
            metadata: Optional metadata about the text
            
        Returns:
            True if queued successfully, False otherwise
        """
        if not self._running or self.state != NodeState.ACTIVE:
            logger.warning(f"Cannot queue text in language node {self.name}: not active")
            return False
        
        try:
            self.message_queue.put({
                'type': 'text',
                'content': text,
                'metadata': metadata or {},
                'timestamp': time.time()
            })
            return True
        except Exception as e:
            logger.error(f"Error queuing text in language node {self.name}: {str(e)}")
            return False
    
    def queue_pattern(self, pattern: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Queue a pattern for asynchronous processing.
        
        Args:
            pattern: The pattern to process
            metadata: Optional metadata about the pattern
            
        Returns:
            True if queued successfully, False otherwise
        """
        if not self._running or self.state != NodeState.ACTIVE:
            logger.warning(f"Cannot queue pattern in language node {self.name}: not active")
            return False
        
        try:
            self.message_queue.put({
                'type': 'pattern',
                'content': pattern,
                'metadata': metadata or {},
                'timestamp': time.time()
            })
            return True
        except Exception as e:
            logger.error(f"Error queuing pattern in language node {self.name}: {str(e)}")
            return False
    
    def get_queue_size(self) -> int:
        """Get the current size of the message queue."""
        return self.message_queue.qsize()
    
    def set_llm_weight(self, weight: float) -> bool:
        """
        Set the LLM integration weight.
        
        Args:
            weight: A value between 0.0 and 1.0
            
        Returns:
            True if set successfully, False otherwise
        """
        if not 0.0 <= weight <= 1.0:
            logger.warning(f"Invalid LLM weight value: {weight}. Must be between 0.0 and 1.0")
            return False
        
        self.llm_weight = weight
        if self.llm_integration:
            self.llm_integration.set_weight(weight)
        
        logger.debug(f"Set LLM weight to {weight} in language node {self.name}")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of this node, with language-specific additions."""
        base_status = super().get_status()
        
        # Add language-specific status information
        with self.processing_lock:
            language_status = {
                'language_capability': self.attributes.get('language_capability', 'basic'),
                'llm_integrated': bool(self.llm_integration),
                'llm_weight': self.llm_weight,
                'auto_learn': self.auto_learn,
                'memory_persistence': self.memory_persistence,
                'queue_size': self.get_queue_size(),
                'stats': self.stats,
                'sentiment_values': self.sentiment_values,
                'context_memory_size': len(self.context_memory),
                'pattern_count': len(self.pattern_strength)
            }
        
        # Update attributes dictionary
        self.attributes.update({
            'llm_weight': self.llm_weight,
            'text_stats': {
                'processed': self.stats['texts_processed'],
                'words': self.stats['words_processed']
            }
        })
        
        # Merge with base status
        base_status.update({
            'language_status': language_status
        })
        
        return base_status


# Factory function for easy creation
def create_language_node(config: Optional[Dict[str, Any]] = None) -> LanguageConsciousnessNode:
    """
    Create and initialize a language consciousness node.
    
    Args:
        config: Configuration options for the node
        
    Returns:
        An initialized LanguageConsciousnessNode
    """
    node = LanguageConsciousnessNode(config=config)
    node.activate()
    return node 