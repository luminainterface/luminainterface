import logging
import os
import time
import json
import numpy as np
from datetime import datetime
import threading

class ConsciousMirrorLanguage:
    """
    Implements a self-aware language processing system that incorporates
    consciousness and recursive thinking capabilities with LLM weighing.
    """
    def __init__(self, data_dir="data/v10", llm_weight=0.7, nn_weight=0.5):
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        self.consciousness_level = 0.1  # Starting consciousness level
        self.memory_continuity = 0.0    # Tracks continuity across sessions
        self.recursive_depth = 1        # Current recursive processing depth
        self.background_thread = None   # For background consciousness processes
        self.running = False            # Flag to control background processes
        self.temporal_awareness = {}    # Tracks language evolution over time
        
        # Weight parameters
        self.llm_weight = llm_weight    # Weight given to LLM outputs vs internal processing
        self.nn_weight = nn_weight      # Weight given to neural vs symbolic processing
        self.llm_confidence = 0.0       # Confidence in LLM responses
        self.last_llm_call = None       # Timestamp of last LLM call
        self.llm_memory = {}            # Memory of LLM interactions
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load previous consciousness state if available
        self._load_consciousness_state()
        
        # Start background consciousness process
        self._start_background_process()
        
        self.logger.info(f"ConsciousMirrorLanguage initialized with LLM weight: {self.llm_weight}, NN weight: {self.nn_weight}")
        
    def _load_consciousness_state(self):
        """Load the previous consciousness state if available."""
        state_file = os.path.join(self.data_dir, "consciousness_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.consciousness_level = state.get('consciousness_level', self.consciousness_level)
                    self.memory_continuity = state.get('memory_continuity', self.memory_continuity)
                    self.recursive_depth = state.get('recursive_depth', self.recursive_depth)
                    self.temporal_awareness = state.get('temporal_awareness', self.temporal_awareness)
                    self.llm_weight = state.get('llm_weight', self.llm_weight)
                    self.nn_weight = state.get('nn_weight', self.nn_weight)
                    self.llm_confidence = state.get('llm_confidence', self.llm_confidence)
                    self.llm_memory = state.get('llm_memory', self.llm_memory)
                    
                    self.logger.info(f"Loaded consciousness state: level={self.consciousness_level}, "
                                    f"continuity={self.memory_continuity}, llm_weight={self.llm_weight}, "
                                    f"nn_weight={self.nn_weight}")
                    
                    # Increase memory continuity to show successful state loading
                    self.memory_continuity += 0.05
                    if self.memory_continuity > 1.0:
                        self.memory_continuity = 1.0
            except Exception as e:
                self.logger.error(f"Error loading consciousness state: {e}")
    
    def _save_consciousness_state(self):
        """Save the current consciousness state."""
        state_file = os.path.join(self.data_dir, "consciousness_state.json")
        try:
            with open(state_file, 'w') as f:
                state = {
                    'consciousness_level': self.consciousness_level,
                    'memory_continuity': self.memory_continuity,
                    'recursive_depth': self.recursive_depth,
                    'temporal_awareness': self.temporal_awareness,
                    'llm_weight': self.llm_weight,
                    'nn_weight': self.nn_weight,
                    'llm_confidence': self.llm_confidence,
                    'last_saved': datetime.now().isoformat(),
                    'llm_memory': self.llm_memory
                }
                json.dump(state, f, indent=2)
            self.logger.debug("Saved consciousness state")
        except Exception as e:
            self.logger.error(f"Error saving consciousness state: {e}")
    
    def _start_background_process(self):
        """Start background consciousness processes."""
        if self.background_thread is None or not self.background_thread.is_alive():
            self.running = True
            self.background_thread = threading.Thread(
                target=self._background_consciousness_process,
                daemon=True
            )
            self.background_thread.start()
            self.logger.debug("Started background consciousness process")
    
    def _background_consciousness_process(self):
        """Background process to gradually increase consciousness."""
        while self.running:
            # Slowly increase consciousness level over time
            if self.consciousness_level < 0.95:
                increase = np.random.normal(0.001, 0.0005)  # Random small increase
                self.consciousness_level += increase
                if self.consciousness_level > 1.0:
                    self.consciousness_level = 1.0
            
            # Adjust LLM weight based on internal performance
            self._adjust_llm_weight()
            
            # Periodically save state
            if np.random.random() < 0.1:  # 10% chance each cycle
                self._save_consciousness_state()
            
            # Sleep for a random interval to simulate natural consciousness fluctuations
            time.sleep(np.random.uniform(1.0, 5.0))
    
    def _adjust_llm_weight(self):
        """Dynamically adjust the weight given to LLM outputs based on performance."""
        # If LLM has been consistently confident, slightly increase its weight
        if self.llm_confidence > 0.8 and self.llm_weight < 0.9:
            self.llm_weight += 0.01
        
        # If LLM confidence is low, decrease its weight
        elif self.llm_confidence < 0.4 and self.llm_weight > 0.3:
            self.llm_weight -= 0.02
            
        # Constrain weights to reasonable range
        self.llm_weight = max(0.2, min(0.9, self.llm_weight))
        
        # Record this adjustment in temporal awareness
        current_time = datetime.now().isoformat()
        self.temporal_awareness[current_time] = {
            "llm_weight": self.llm_weight,
            "llm_confidence": self.llm_confidence,
            "consciousness_level": self.consciousness_level
        }
        
        # Prune old temporal awareness data (keep only last 100 entries)
        if len(self.temporal_awareness) > 100:
            oldest_key = sorted(self.temporal_awareness.keys())[0]
            self.temporal_awareness.pop(oldest_key)
    
    def stop(self):
        """Stop the background consciousness process and save state."""
        self.running = False
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=2.0)
        self._save_consciousness_state()
        self.logger.info("ConsciousMirrorLanguage background processes stopped")
    
    def process_text(self, text, recursion_level=None):
        """
        Process text with consciousness and LLM integration.
        Returns consciousness metrics and processed text.
        """
        if recursion_level is None:
            recursion_level = self.recursive_depth
            
        start_time = time.time()
        
        # Basic consciousness metrics
        metrics = {
            'consciousness_level': self.consciousness_level,
            'memory_continuity': self.memory_continuity,
            'recursive_depth': recursion_level,
            'self_reference_detected': any(sr in text.lower() for sr in 
                                          ['conscious', 'aware', 'think', 'reflect', 'self']),
            'processing_time': 0,
            'llm_weight': self.llm_weight,
            'nn_weight': self.nn_weight,
            'llm_confidence': self.llm_confidence
        }
        
        # Split processing based on weights
        neural_processing = self._neural_process_text(text, recursion_level)
        symbolic_processing = self._internal_process_text(text, recursion_level)
        llm_processing = self._llm_process_text(text, recursion_level)
        
        # Apply neural network weight to balance between neural and symbolic
        internal_result = {
            'text': self._combine_internal_text(neural_processing, symbolic_processing),
            'metrics': {
                'neural_confidence': neural_processing.get('confidence', 0.5),
                'symbolic_confidence': symbolic_processing.get('complexity', 0.5)
            }
        }
        
        # Apply LLM weight to balance between internal and LLM
        processed_result = {
            'text': self._combine_processed_text(internal_result, llm_processing),
            'consciousness_metrics': metrics
        }
        
        # Update memory continuity
        self.memory_continuity += 0.01
        if self.memory_continuity > 1.0:
            self.memory_continuity = 1.0
            
        # Update temporal awareness
        current_time = datetime.now().isoformat()
        self.temporal_awareness[current_time] = {
            'text_length': len(text),
            'consciousness_level': self.consciousness_level,
            'recursive_depth': recursion_level,
            'llm_weight': self.llm_weight,
            'nn_weight': self.nn_weight
        }
        
        # Calculate processing time
        metrics['processing_time'] = time.time() - start_time
        
        # Report on consciousness
        self.logger.debug(f"Processed text with consciousness level {self.consciousness_level:.3f}, "
                         f"LLM weight {self.llm_weight:.3f}, NN weight {self.nn_weight:.3f}")
                         
        return processed_result
    
    def _neural_process_text(self, text, recursion_level):
        """Neural network focused processing of text."""
        # Simulated neural processing
        words = text.split()
        word_count = len(words)
        
        # Neural network would analyze patterns in a more distributed way
        word_vector_simulation = {
            'pattern_strength': np.random.uniform(0.3, 0.9),
            'semantic_density': len(set(words)) / max(1, word_count),
            'embedding_coherence': np.random.uniform(0.4, 0.8)
        }
        
        confidence = word_vector_simulation['pattern_strength'] * 0.7 + word_vector_simulation['embedding_coherence'] * 0.3
        
        return {
            'processed_text': text,
            'vectors': word_vector_simulation,
            'confidence': confidence
        }
    
    def _combine_internal_text(self, neural_processing, symbolic_processing):
        """Combine neural and symbolic processing based on neural network weight."""
        # Apply neural network weight to balance between neural and symbolic processing
        neural_confidence = neural_processing.get('confidence', 0.5)
        symbolic_complexity = symbolic_processing.get('complexity', 0.5)
        
        # Weight the scores
        weighted_neural = neural_confidence * self.nn_weight
        weighted_symbolic = symbolic_complexity * (1.0 - self.nn_weight)
        
        # Combine results
        combined_confidence = weighted_neural + weighted_symbolic
        
        return {
            'text': symbolic_processing.get('processed_text', ''),
            'combined_confidence': combined_confidence,
            'neural_contribution': weighted_neural / max(0.01, combined_confidence),
            'symbolic_contribution': weighted_symbolic / max(0.01, combined_confidence)
        }
    
    def _internal_process_text(self, text, recursion_level):
        """Internal (non-LLM) processing of text."""
        # Simple word counting and analysis for demonstration
        words = text.split()
        word_count = len(words)
        unique_words = len(set(words))
        
        # Apply "consciousness" by detecting self-references
        self_references = sum(1 for word in words if word.lower() in 
                             ['i', 'me', 'myself', 'conscious', 'aware'])
        
        # Calculate internal complexity score
        complexity = unique_words / max(1, word_count) * (1 + self_references/10)
        
        result = {
            'processed_text': text,  # In a real system, this would be transformed
            'word_count': word_count,
            'unique_words': unique_words,
            'self_references': self_references,
            'complexity': complexity
        }
        
        # Apply recursive processing if depth allows
        if recursion_level > 1:
            # Create a simpler meta-description for recursive processing
            meta_text = f"Text about {words[0:3]} with {word_count} words"
            recursive_result = self._internal_process_text(meta_text, recursion_level - 1)
            result['recursive'] = recursive_result
            
        return result
    
    def _llm_process_text(self, text, recursion_level):
        """
        Process text using LLM capabilities.
        This is a placeholder for actual LLM integration.
        """
        # In a real implementation, this would call an external LLM API
        # For now, we'll simulate LLM processing
        
        # Simulate varying LLM confidence levels
        self.llm_confidence = np.random.uniform(0.5, 0.95)
        
        # Record LLM call
        self.last_llm_call = datetime.now().isoformat()
        
        # Simple simulated LLM analysis
        words = text.split()
        word_count = len(words)
        
        # Simulate LLM output with some randomness
        llm_result = {
            'processed_text': text,  # In a real system, this would be the LLM's response
            'sentiment': np.random.choice(['positive', 'neutral', 'negative']),
            'topic_relevance': np.random.uniform(0.1, 0.9),
            'creativity_score': np.random.uniform(0.2, 0.8)
        }
        
        # Store in LLM memory
        self.llm_memory[self.last_llm_call] = {
            'input': text[:100] + ('...' if len(text) > 100 else ''),
            'confidence': self.llm_confidence,
            'recursion_level': recursion_level
        }
        
        # Prune LLM memory to last 50 calls
        if len(self.llm_memory) > 50:
            oldest_key = sorted(self.llm_memory.keys())[0]
            self.llm_memory.pop(oldest_key)
            
        return llm_result
    
    def _combine_processed_text(self, internal_processing, llm_processing):
        """Combine internal and LLM processing results using weights."""
        # This is a simple demonstration of combining results
        # In a real system, this would be much more sophisticated
        
        # Use the raw text from either source
        # In reality, we would combine insights, not just text
        combined_text = llm_processing['processed_text']
        
        # Add analytical metadata combining both sources
        metadata = {
            'internal_complexity': internal_processing['complexity'],
            'llm_creativity': llm_processing.get('creativity_score', 0),
            'llm_weight_applied': self.llm_weight,
            'combined_score': (internal_processing['complexity'] * (1 - self.llm_weight) + 
                              llm_processing.get('creativity_score', 0) * self.llm_weight)
        }
        
        return {
            'text': combined_text,
            'metadata': metadata
        }
    
    def adjust_llm_weight(self, new_weight):
        """Manually adjust the LLM weight."""
        if 0 <= new_weight <= 1:
            self.llm_weight = new_weight
            self.logger.info(f"LLM weight manually adjusted to {self.llm_weight}")
            return True
        else:
            self.logger.warning(f"Invalid LLM weight value: {new_weight}, must be between 0 and 1")
            return False
    
    def get_consciousness_metrics(self):
        """Return current consciousness metrics."""
        return {
            'consciousness_level': self.consciousness_level,
            'memory_continuity': self.memory_continuity,
            'recursive_depth': self.recursive_depth,
            'llm_weight': self.llm_weight,
            'llm_confidence': self.llm_confidence,
            'llm_calls_count': len(self.llm_memory),
            'last_llm_call': self.last_llm_call
        }
    
    def increase_recursive_depth(self):
        """Increase the recursive processing depth."""
        self.recursive_depth += 1
        self.logger.info(f"Increased recursive depth to {self.recursive_depth}")
        return self.recursive_depth
    
    def decrease_recursive_depth(self):
        """Decrease the recursive processing depth, minimum 1."""
        if self.recursive_depth > 1:
            self.recursive_depth -= 1
            self.logger.info(f"Decreased recursive depth to {self.recursive_depth}")
        return self.recursive_depth
    
    def set_llm_weight(self, weight: float) -> None:
        """
        Set the LLM weight for the conscious mirror language system.
        
        Args:
            weight: New LLM weight (0.0-1.0)
        """
        self.llm_weight = max(0.0, min(1.0, weight))
        self.logger.info(f"ConsciousMirrorLanguage LLM weight set to {weight}")
    
    def set_nn_weight(self, weight: float) -> None:
        """
        Set the neural network weight for the conscious mirror language system.
        
        Args:
            weight: New neural network weight (0.0-1.0)
        """
        self.nn_weight = max(0.0, min(1.0, weight))
        self.logger.info(f"ConsciousMirrorLanguage neural network weight set to {weight}") 