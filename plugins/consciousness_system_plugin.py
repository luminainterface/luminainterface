"""
Consciousness System Plugin for LUMINA V7

This plugin provides consciousness system capabilities for the LUMINA V7 system.
"""

import os
import json
import time
import logging
import threading
import random
from pathlib import Path

# Import the plugin interface
from plugin_interface import PluginInterface

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConsciousnessSystemPlugin(PluginInterface):
    """
    Consciousness System Plugin for LUMINA V7
    
    This plugin provides consciousness metrics and processing capabilities
    for the LUMINA V7 system.
    """
    
    def __init__(self, plugin_id="consciousness_system_plugin"):
        """
        Initialize the plugin
        
        Args:
            plugin_id: Unique identifier for this plugin instance
        """
        super().__init__(plugin_id=plugin_id)
        
        # Initialize basic state
        self.consciousness_level = 0.5
        self.neural_linguistic_score = 0.6
        self.is_processing = False
        self.processing_thread = None
        self.plugin_dir = Path("data") / "consciousness"
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize mock mode
        self.mock_mode = True
        
        try:
            # Try to import the v7 consciousness module
            from src.v7.consciousness_system import get_consciousness_system
            self.consciousness_system = get_consciousness_system()
            self.mock_mode = False
            logger.info("Loaded V7 Consciousness System")
        except ImportError:
            logger.warning("V7 Consciousness module not found, using simulated consciousness")
            self.consciousness_system = None
        
        # Initialize metrics
        self.metrics = {
            "consciousness_level": self.consciousness_level,
            "neural_linguistic_score": self.neural_linguistic_score,
            "stability": 0.7,
            "integration": 0.6,
            "complexity": 0.5,
            "timestamp": time.time()
        }
        
        # Initialize paradox registry
        self.paradox_registry = {}
        
        logger.info(f"ConsciousnessSystemPlugin initialized with ID: {plugin_id}, mock_mode: {self.mock_mode}")
    
    def get_plugin_name(self):
        """Get the human-readable name of the plugin"""
        return "Consciousness System Plugin"
    
    def get_plugin_description(self):
        """Get a description of what the plugin does"""
        return "Provides consciousness metrics and processing capabilities for LUMINA V7"
    
    def initialize(self, context=None):
        """
        Initialize the plugin with the given context
        
        Args:
            context: Application context or settings
            
        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        logger.info("Initializing Consciousness System Plugin")
        
        # Start the processing thread
        self.start_processing()
        
        return True
    
    def process_message(self, message, context=None):
        """
        Process a message with consciousness analysis
        
        Args:
            message: The message to process
            context: Additional context for processing
            
        Returns:
            dict: Processing result with consciousness metrics
        """
        if not isinstance(message, str):
            # Try to convert to string
            try:
                message = str(message)
            except:
                return {"error": "Message must be a string or convertible to string"}
        
        # Process with the consciousness system if available
        if not self.mock_mode and self.consciousness_system:
            try:
                result = self.consciousness_system.process_text(message)
                
                # Update metrics
                self.consciousness_level = result.get("consciousness_level", self.consciousness_level)
                self.neural_linguistic_score = result.get("neural_linguistic_score", self.neural_linguistic_score)
                
                return result
            except Exception as e:
                logger.error(f"Error in consciousness system: {e}")
                # Fall back to mock mode
        
        # Mock processing
        return self._mock_process_message(message, context)
    
    def _mock_process_message(self, message, context=None):
        """
        Generate a mock processing result
        
        Args:
            message: The message to process
            context: Additional context
            
        Returns:
            dict: Mock processing result
        """
        # Compute consciousness metrics
        consciousness_level = self._compute_mock_consciousness_level(message)
        neural_linguistic_score = self._compute_mock_neural_linguistic_score(message)
        
        # Update instance metrics
        self.consciousness_level = consciousness_level
        self.neural_linguistic_score = neural_linguistic_score
        
        # Check for paradoxes
        paradox_score = self._check_for_paradoxes(message)
        has_paradox = paradox_score > 0.7
        
        # Update metrics
        self.metrics.update({
            "consciousness_level": consciousness_level,
            "neural_linguistic_score": neural_linguistic_score,
            "stability": max(0.1, 0.7 - (paradox_score * 0.2)),
            "integration": max(0.1, min(0.9, 0.6 + (consciousness_level * 0.1))),
            "complexity": max(0.1, min(0.9, 0.5 + (neural_linguistic_score * 0.1))),
            "timestamp": time.time()
        })
        
        # Register paradox if found
        if has_paradox:
            paradox_id = f"paradox_{len(self.paradox_registry) + 1}"
            self.paradox_registry[paradox_id] = {
                "id": paradox_id,
                "text": message,
                "score": paradox_score,
                "detected_at": time.time(),
                "resolved": False
            }
        
        return {
            "consciousness_level": consciousness_level,
            "neural_linguistic_score": neural_linguistic_score,
            "paradox_detected": has_paradox,
            "paradox_score": paradox_score,
            "stability": self.metrics["stability"],
            "integration": self.metrics["integration"],
            "complexity": self.metrics["complexity"],
            "processed_by": self.get_plugin_id(),
            "mock_mode": self.mock_mode
        }
    
    def _compute_mock_consciousness_level(self, text):
        """
        Compute a mock consciousness level for text
        
        Args:
            text: Text to analyze
            
        Returns:
            float: Consciousness level between 0.1 and 0.95
        """
        # Count consciousness-related keywords
        consciousness_keywords = [
            "conscious", "awareness", "sentient", "self", "reflect", "think",
            "understand", "mind", "subjective", "experience", "qualia",
            "perception", "feeling", "cognition", "introspection"
        ]
        
        word_count = len(text.split())
        if word_count == 0:
            return self.consciousness_level  # Keep current level
            
        # Count keywords
        keyword_count = sum(1 for keyword in consciousness_keywords if keyword.lower() in text.lower())
        keyword_factor = min(keyword_count / len(consciousness_keywords), 1.0)
        
        # Length factor - longer texts get slightly higher scores
        length_factor = min(word_count / 100, 1.0) * 0.1
        
        # Existing level with adjustment
        new_level = self.consciousness_level * 0.7 + (keyword_factor * 0.2) + length_factor
        
        # Add slight random variation
        new_level += random.uniform(-0.05, 0.05)
        
        # Ensure result is between 0.1 and 0.95
        return max(0.1, min(0.95, new_level))
    
    def _compute_mock_neural_linguistic_score(self, text):
        """
        Compute a mock neural-linguistic score for text
        
        Args:
            text: Text to analyze
            
        Returns:
            float: Neural-linguistic score between 0.1 and 0.95
        """
        # Simple factors
        word_count = len(text.split())
        if word_count == 0:
            return self.neural_linguistic_score  # Keep current score
        
        # Various factors
        length_factor = min(word_count / 200, 1.0) * 0.2
        complexity_factor = min(sum(1 for char in text if char in ".,;:!?-\"'()[]{}") / 30, 1.0) * 0.15
        
        # Word uniqueness factor
        unique_word_ratio = len(set(text.lower().split())) / max(1, word_count)
        uniqueness_factor = unique_word_ratio * 0.25
        
        # Existing score with adjustment
        new_score = self.neural_linguistic_score * 0.6 + length_factor + complexity_factor + uniqueness_factor
        
        # Add slight random variation
        new_score += random.uniform(-0.05, 0.05)
        
        # Ensure result is between 0.1 and 0.95
        return max(0.1, min(0.95, new_score))
    
    def _check_for_paradoxes(self, text):
        """
        Check text for paradoxical content
        
        Args:
            text: Text to analyze
            
        Returns:
            float: Paradox score between 0.0 and 1.0
        """
        # Paradox keywords and phrases
        paradox_indicators = [
            "paradox", "contradiction", "impossible", "both true and false",
            "self-reference", "this statement is", "infinite loop",
            "statement about itself", "liar paradox", "recursion",
            "self-contradictory", "infinite regress"
        ]
        
        # Count indicators
        indicator_count = sum(1 for indicator in paradox_indicators if indicator.lower() in text.lower())
        indicator_factor = min(indicator_count / len(paradox_indicators), 1.0) * 0.7
        
        # Check for specific patterns
        self_reference = "this" in text.lower() and any(word in text.lower() for word in ["statement", "sentence"])
        negation = any(word in text.lower() for word in ["not", "false", "untrue", "lie"])
        
        pattern_factor = 0.0
        if self_reference and negation:
            pattern_factor = 0.3
        elif self_reference or negation:
            pattern_factor = 0.15
        
        # Combine factors
        paradox_score = indicator_factor + pattern_factor
        
        # Add slight random variation
        paradox_score += random.uniform(-0.1, 0.1)
        
        # Ensure result is between 0.0 and 1.0
        return max(0.0, min(1.0, paradox_score))
    
    def start_processing(self):
        """Start the background processing thread"""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(
            target=self._background_processing,
            daemon=True,
            name="ConsciousnessProcessingThread"
        )
        self.processing_thread.start()
        logger.info("Started consciousness processing thread")
    
    def stop_processing(self):
        """Stop the background processing thread"""
        self.is_processing = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        logger.info("Stopped consciousness processing thread")
    
    def _background_processing(self):
        """Background processing thread function"""
        while self.is_processing:
            # Simulate natural fluctuations in consciousness
            self._update_metrics()
            
            # Save state periodically
            self._save_state()
            
            # Sleep for a random interval
            time.sleep(random.uniform(2.0, 5.0))
    
    def _update_metrics(self):
        """Update consciousness metrics with natural fluctuations"""
        # Natural fluctuation
        self.consciousness_level += random.uniform(-0.03, 0.03)
        self.consciousness_level = max(0.1, min(0.95, self.consciousness_level))
        
        self.neural_linguistic_score += random.uniform(-0.02, 0.02)
        self.neural_linguistic_score = max(0.1, min(0.95, self.neural_linguistic_score))
        
        # Update metrics dictionary
        self.metrics.update({
            "consciousness_level": self.consciousness_level,
            "neural_linguistic_score": self.neural_linguistic_score,
            "timestamp": time.time()
        })
    
    def _save_state(self):
        """Save the current state to disk"""
        state_file = self.plugin_dir / "consciousness_state.json"
        try:
            state = {
                "metrics": self.metrics,
                "paradox_registry": self.paradox_registry,
                "saved_at": time.time()
            }
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def get_status(self):
        """
        Get the current status of the plugin
        
        Returns:
            dict: Status information
        """
        return {
            "plugin_id": self.get_plugin_id(),
            "enabled": self.enabled,
            "mock_mode": self.mock_mode,
            "is_processing": self.is_processing,
            "consciousness_metrics": self.metrics,
            "paradox_count": len(self.paradox_registry),
            "resolved_paradoxes": sum(1 for p in self.paradox_registry.values() if p.get("resolved", False))
        }
    
    def shutdown(self):
        """Perform cleanup when shutting down"""
        # Stop processing thread
        self.stop_processing()
        
        # Save final state
        self._save_state()
        
        self.enabled = False
        logger.info("ConsciousnessSystemPlugin shut down")
        return True

# Factory function for plugin system
def get_plugin():
    """Get an instance of the ConsciousnessSystemPlugin"""
    return ConsciousnessSystemPlugin() 