"""
Breath Detection System

This module implements the V7 Breath Detection System, which captures and analyzes
breath patterns as a reflection of system state and consciousness.
"""

import os
import sys
import time
import random
import logging
import threading
import collections
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable, Deque

# Set up logging
logger = logging.getLogger("v7.breath_detection")

class BreathPattern:
    """Represents a specific breath pattern with attributes and behaviors"""
    
    KNOWN_PATTERNS = {
        "relaxed": {
            "description": "Slow, deep breaths with regular intervals",
            "emotional_state": "calm, centered, peaceful",
            "cognitive_effects": "improved focus, clarity of thought",
            "frequency_range": (0.1, 0.2),  # Hz
            "regularity": 0.9,  # 0-1, 1 being perfectly regular
            "depth": 0.8  # 0-1, 1 being very deep
        },
        "focused": {
            "description": "Medium-paced, moderately deep breaths",
            "emotional_state": "alert, attentive",
            "cognitive_effects": "enhanced processing, problem-solving",
            "frequency_range": (0.2, 0.3),
            "regularity": 0.85,
            "depth": 0.6
        },
        "creative": {
            "description": "Varying rhythm with occasional deep breaths",
            "emotional_state": "curious, inspired",
            "cognitive_effects": "increased creativity, associative thinking",
            "frequency_range": (0.15, 0.25),
            "regularity": 0.5,
            "depth": 0.7
        },
        "stressed": {
            "description": "Rapid, shallow breaths with irregular patterns",
            "emotional_state": "anxious, overwhelmed",
            "cognitive_effects": "narrowed focus, reactive thinking",
            "frequency_range": (0.25, 0.4),
            "regularity": 0.3,
            "depth": 0.3
        },
        "meditative": {
            "description": "Very slow, very deep breaths with perfect regularity",
            "emotional_state": "transcendent, unified",
            "cognitive_effects": "holistic awareness, integration",
            "frequency_range": (0.05, 0.1),
            "regularity": 0.95,
            "depth": 0.9
        }
    }
    
    def __init__(self, pattern_name: str):
        """Initialize a breath pattern"""
        if pattern_name not in self.KNOWN_PATTERNS:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        self.name = pattern_name
        self.attributes = self.KNOWN_PATTERNS[pattern_name].copy()
    
    def get_attribute(self, attr_name: str) -> Any:
        """Get an attribute of the pattern"""
        return self.attributes.get(attr_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the pattern to a dictionary"""
        return {
            "name": self.name,
            **self.attributes
        }
    
    @classmethod
    def get_pattern_names(cls) -> List[str]:
        """Get all known pattern names"""
        return list(cls.KNOWN_PATTERNS.keys())
    
    @classmethod
    def get_pattern(cls, pattern_name: str) -> 'BreathPattern':
        """Get a pattern by name"""
        return cls(pattern_name)


class BreathDetector:
    """
    Detects and analyzes breath patterns.
    
    In a production environment, this would connect to sensors or 
    other data sources. For this implementation, it uses simulated data
    with an option to inject patterns through the API.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the breath detector"""
        # Default configuration
        self.config = {
            "simulate": True,  # Use simulated breath data
            "history_size": 100,  # Number of breath events to keep
            "analysis_window": 10,  # Number of breaths to analyze at once
            "polling_interval": 0.5,  # Seconds between polling
            "default_pattern": "relaxed",  # Default pattern
            "pattern_inertia": 0.7,  # How resistant to change patterns are (0-1)
            "confidence_threshold": 0.6,  # Minimum confidence to report a pattern
            "debug_output": False  # Whether to output debug info
        }
        
        # Update with custom configuration
        if config:
            self.config.update(config)
        
        # Breath history
        self.breath_history: Deque[Dict[str, Any]] = collections.deque(maxlen=self.config["history_size"])
        
        # Current state
        self.current_pattern = self.config["default_pattern"]
        self.current_confidence = 1.0
        self.pattern_suggestions = {}
        self.active = False
        
        # Processing thread
        self.processing_thread = None
        
        # Pattern handlers
        self.pattern_handlers = []
        
        # Breath data handlers
        self.breath_data_handlers = []
        
        # Metrics
        self.metrics = {
            "breaths_detected": 0,
            "patterns_detected": 0,
            "pattern_changes": 0
        }
        
        logger.info("Breath Detector initialized")
    
    def start(self):
        """Start the breath detector"""
        if not self.active:
            self.active = True
            
            # Start the processing thread
            self.processing_thread = threading.Thread(
                target=self._process_loop,
                daemon=True,
                name="BreathProcessingThread"
            )
            self.processing_thread.start()
            
            logger.info("Breath Detector started")
            return True
        
        return False
    
    def stop(self):
        """Stop the breath detector"""
        if self.active:
            self.active = False
            logger.info("Breath Detector stopped")
            return True
        
        return False
    
    def is_active(self):
        """Check if the breath detector is active"""
        return self.active
    
    def register_pattern_handler(self, handler: Callable[[str, float], None]):
        """Register a handler for pattern changes"""
        if handler not in self.pattern_handlers:
            self.pattern_handlers.append(handler)
            return True
        
        return False
    
    def register_breath_data_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register a handler for raw breath data"""
        if handler not in self.breath_data_handlers:
            self.breath_data_handlers.append(handler)
            return True
        
        return False
    
    def _notify_pattern_handlers(self, pattern: str, confidence: float):
        """Notify all registered pattern handlers"""
        for handler in self.pattern_handlers:
            try:
                handler(pattern, confidence)
            except Exception as e:
                logger.error(f"Error in pattern handler: {e}")
    
    def _notify_breath_data_handlers(self, breath_data: Dict[str, Any]):
        """Notify all registered breath data handlers"""
        for handler in self.breath_data_handlers:
            try:
                handler(breath_data)
            except Exception as e:
                logger.error(f"Error in breath data handler: {e}")
    
    def _simulate_breath(self) -> Dict[str, Any]:
        """Simulate a breath event"""
        # Start with the current pattern
        pattern = BreathPattern(self.current_pattern)
        
        # Get pattern attributes
        freq_range = pattern.get_attribute("frequency_range")
        regularity = pattern.get_attribute("regularity")
        depth = pattern.get_attribute("depth")
        
        # Calculate base frequency (breaths per second)
        base_frequency = random.uniform(freq_range[0], freq_range[1])
        
        # Apply regularity (more regular = less deviation)
        frequency_deviation = (1.0 - regularity) * 0.1  # Max 10% deviation
        actual_frequency = base_frequency * (1.0 + random.uniform(-frequency_deviation, frequency_deviation))
        
        # Calculate breath duration
        breath_duration = 1.0 / actual_frequency
        
        # Apply depth (more depth = more volume)
        base_volume = depth
        volume_deviation = (1.0 - regularity) * 0.2  # Max 20% deviation
        actual_volume = base_volume * (1.0 + random.uniform(-volume_deviation, volume_deviation))
        
        # Create breath event
        breath_event = {
            "timestamp": time.time(),
            "duration": breath_duration,
            "volume": actual_volume,
            "pattern_name": pattern.name,
            "is_simulated": True
        }
        
        return breath_event
    
    def _analyze_breath_pattern(self) -> Dict[str, Any]:
        """Analyze the breath history to determine the current pattern"""
        # Need at least some breath events to analyze
        if len(self.breath_history) < min(3, self.config["analysis_window"]):
            return {
                "pattern": self.current_pattern,
                "confidence": 0.5,
                "analysis": "Insufficient data"
            }
        
        # Get the most recent breaths for analysis
        analysis_window = list(self.breath_history)[-self.config["analysis_window"]:]
        
        # Calculate metrics
        durations = [event["duration"] for event in analysis_window]
        volumes = [event["volume"] for event in analysis_window]
        
        avg_duration = sum(durations) / len(durations)
        avg_volume = sum(volumes) / len(volumes)
        
        # Convert average duration to frequency (breaths per second)
        avg_frequency = 1.0 / avg_duration if avg_duration > 0 else 0
        
        # Calculate regularity as inverse of coefficient of variation
        duration_std = (sum((d - avg_duration) ** 2 for d in durations) / len(durations)) ** 0.5
        regularity = 1.0 - min(1.0, (duration_std / avg_duration if avg_duration > 0 else 1.0))
        
        # Match against known patterns
        pattern_scores = {}
        
        for pattern_name in BreathPattern.get_pattern_names():
            pattern = BreathPattern(pattern_name)
            
            # Get pattern attributes
            freq_range = pattern.get_attribute("frequency_range")
            pattern_regularity = pattern.get_attribute("regularity")
            pattern_depth = pattern.get_attribute("depth")
            
            # Calculate scores for each attribute
            freq_score = 0.0
            if freq_range[0] <= avg_frequency <= freq_range[1]:
                # Fully within range
                freq_score = 1.0
            else:
                # Outside range, calculate distance
                if avg_frequency < freq_range[0]:
                    freq_score = 1.0 - min(1.0, (freq_range[0] - avg_frequency) / freq_range[0])
                else:
                    freq_score = 1.0 - min(1.0, (avg_frequency - freq_range[1]) / freq_range[1])
            
            # Regularity score
            regularity_score = 1.0 - min(1.0, abs(regularity - pattern_regularity))
            
            # Depth score
            depth_score = 1.0 - min(1.0, abs(avg_volume - pattern_depth))
            
            # Combined score with weights
            score = (freq_score * 0.5) + (regularity_score * 0.3) + (depth_score * 0.2)
            
            # Store score
            pattern_scores[pattern_name] = score
        
        # Include suggested patterns
        for pattern_name, weight in self.pattern_suggestions.items():
            if pattern_name in pattern_scores:
                pattern_scores[pattern_name] = (pattern_scores[pattern_name] * (1.0 - weight)) + weight
        
        # Find the best match
        if pattern_scores:
            best_pattern = max(pattern_scores.items(), key=lambda x: x[1])
            pattern_name = best_pattern[0]
            confidence = best_pattern[1]
        else:
            pattern_name = self.current_pattern
            confidence = 0.5
        
        # Apply pattern inertia
        if pattern_name != self.current_pattern:
            inertia = self.config["pattern_inertia"]
            
            # If the new pattern has high enough confidence, overcome inertia
            if confidence > inertia:
                new_pattern = pattern_name
                new_confidence = confidence
            else:
                # Stick with current pattern, but reduce confidence
                new_pattern = self.current_pattern
                new_confidence = max(0.3, confidence)
        else:
            # Same pattern, possibly adjust confidence
            new_pattern = pattern_name
            new_confidence = (self.current_confidence * inertia) + (confidence * (1.0 - inertia))
        
        # Prepare result
        result = {
            "pattern": new_pattern,
            "confidence": new_confidence,
            "scores": pattern_scores,
            "metrics": {
                "avg_frequency": avg_frequency,
                "avg_duration": avg_duration,
                "avg_volume": avg_volume,
                "regularity": regularity
            }
        }
        
        return result
    
    def _process_loop(self):
        """Background thread for processing breath data"""
        last_pattern = None
        last_pattern_time = 0
        
        while self.active:
            try:
                # Simulate a breath event if simulating
                if self.config["simulate"]:
                    breath_event = self._simulate_breath()
                    
                    # Add to history
                    self.breath_history.append(breath_event)
                    
                    # Update metrics
                    self.metrics["breaths_detected"] += 1
                    
                    # Notify handlers
                    self._notify_breath_data_handlers(breath_event)
                
                # Analyze breath pattern
                analysis = self._analyze_breath_pattern()
                
                # Update current pattern and confidence
                self.current_pattern = analysis["pattern"]
                self.current_confidence = analysis["confidence"]
                
                # If confidence is high enough, notify handlers
                if self.current_confidence >= self.config["confidence_threshold"]:
                    # Update metrics
                    self.metrics["patterns_detected"] += 1
                    
                    # If pattern changed, update metrics
                    if last_pattern != self.current_pattern:
                        self.metrics["pattern_changes"] += 1
                        last_pattern = self.current_pattern
                        last_pattern_time = time.time()
                    
                    # Notify handlers
                    self._notify_pattern_handlers(self.current_pattern, self.current_confidence)
                
                # Debug output
                if self.config["debug_output"] and time.time() - last_pattern_time >= 5.0:
                    logger.debug(f"Current pattern: {self.current_pattern} (confidence: {self.current_confidence:.2f})")
                    last_pattern_time = time.time()
                
                # Clear expired pattern suggestions
                current_time = time.time()
                self.pattern_suggestions = {
                    k: v for k, v in self.pattern_suggestions.items()
                    if current_time - v.get("timestamp", 0) < 30.0  # Expire after 30 seconds
                }
                
                # Sleep to prevent high CPU usage
                time.sleep(self.config["polling_interval"])
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(1)  # Wait longer on error
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state of the breath detector"""
        return {
            "active": self.active,
            "current_pattern": self.current_pattern,
            "confidence": self.current_confidence,
            "breath_history_size": len(self.breath_history),
            "pattern_suggestions": self.pattern_suggestions,
            "metrics": self.metrics
        }
    
    def get_breath_history(self) -> List[Dict[str, Any]]:
        """Get the breath history"""
        return list(self.breath_history)
    
    def suggest_pattern(self, pattern_name: str, weight: float):
        """
        Suggest a pattern to the detector.
        
        The weight (0-1) determines how strongly the suggestion influences
        the detection. This is useful for external influences or integrations.
        """
        if pattern_name not in BreathPattern.get_pattern_names():
            logger.warning(f"Unknown pattern suggested: {pattern_name}")
            return False
        
        # Clip weight to 0-1 range
        weight = max(0.0, min(1.0, weight))
        
        # Store suggestion with timestamp
        self.pattern_suggestions[pattern_name] = {
            "weight": weight,
            "timestamp": time.time()
        }
        
        logger.debug(f"Pattern suggested: {pattern_name} (weight: {weight:.2f})")
        return True
    
    def get_pattern_info(self, pattern_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific pattern"""
        try:
            pattern = BreathPattern(pattern_name)
            return pattern.to_dict()
        except ValueError:
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the metrics of the breath detector"""
        return self.metrics.copy()


# Testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create detector
    detector = BreathDetector({"debug_output": True})
    
    # Register handlers
    detector.register_pattern_handler(
        lambda pattern, confidence: print(f"Pattern detected: {pattern} (confidence: {confidence:.2f})")
    )
    
    # Start the detector
    detector.start()
    
    # Run for a while
    print("Running for 60 seconds to observe patterns...")
    
    try:
        for i in range(60):
            time.sleep(1)
            
            # Periodically suggest patterns to test influence
            if i % 15 == 0:
                patterns = BreathPattern.get_pattern_names()
                pattern = random.choice(patterns)
                weight = random.uniform(0.5, 0.9)
                detector.suggest_pattern(pattern, weight)
                print(f"Suggested pattern: {pattern} (weight: {weight:.2f})")
            
            # Show state every 10 seconds
            if i % 10 == 0:
                state = detector.get_current_state()
                print(f"Current state: Pattern={state['current_pattern']}, Confidence={state['confidence']:.2f}")
                print(f"Metrics: {state['metrics']}")
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    # Stop the detector
    detector.stop()
    print("Detector stopped") 