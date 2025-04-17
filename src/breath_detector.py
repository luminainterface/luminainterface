import numpy as np
from typing import Dict, List, Optional
import time
from dataclasses import dataclass
import logging

@dataclass
class BreathPattern:
    """Represents a detected breath pattern"""
    pattern_type: str
    intensity: float
    duration: float
    timestamp: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class BreathDetector:
    """Detects and analyzes breath patterns for neural processing adjustment"""
    
    def __init__(self, sample_rate: int = 100, window_size: int = 50):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.buffer = []
        self.pattern_history = []
        self.last_breath_time = 0
        self.current_pattern = None
        
        # Pattern recognition thresholds
        self.thresholds = {
            'deep_breath': {'min_intensity': 0.7, 'min_duration': 1.5},
            'shallow_breath': {'min_intensity': 0.3, 'max_duration': 1.0},
            'rapid_breath': {'min_frequency': 2.0, 'max_duration': 0.5}
        }
        
    def process_sample(self, sample: float) -> Optional[BreathPattern]:
        """Process a new breath sample and detect patterns"""
        try:
            # Add sample to buffer
            self.buffer.append(sample)
            
            # Maintain window size
            if len(self.buffer) > self.window_size:
                self.buffer.pop(0)
                
            # Analyze pattern if we have enough samples
            if len(self.buffer) == self.window_size:
                return self._analyze_pattern()
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing breath sample: {str(e)}")
            return None
            
    def _analyze_pattern(self) -> Optional[BreathPattern]:
        """Analyze current buffer for breath patterns"""
        # Calculate basic metrics
        intensity = np.mean(np.abs(self.buffer))
        duration = len(self.buffer) / self.sample_rate
        frequency = 1.0 / (time.time() - self.last_breath_time) if self.last_breath_time > 0 else 0
        
        # Detect pattern type
        pattern_type = None
        confidence = 0.0
        
        if intensity >= self.thresholds['deep_breath']['min_intensity'] and duration >= self.thresholds['deep_breath']['min_duration']:
            pattern_type = 'deep_breath'
            confidence = 0.9
        elif intensity <= self.thresholds['shallow_breath']['min_intensity'] and duration <= self.thresholds['shallow_breath']['max_duration']:
            pattern_type = 'shallow_breath'
            confidence = 0.8
        elif frequency >= self.thresholds['rapid_breath']['min_frequency'] and duration <= self.thresholds['rapid_breath']['max_duration']:
            pattern_type = 'rapid_breath'
            confidence = 0.7
            
        if pattern_type:
            # Create pattern object
            pattern = BreathPattern(
                pattern_type=pattern_type,
                intensity=intensity,
                duration=duration,
                timestamp=time.time(),
                confidence=confidence,
                metadata={
                    'frequency': frequency,
                    'buffer_size': len(self.buffer)
                }
            )
            
            # Update history
            self.pattern_history.append(pattern)
            self.last_breath_time = time.time()
            self.current_pattern = pattern
            
            return pattern
            
        return None
        
    def get_current_state(self) -> Dict[str, Any]:
        """Get current breath detection state"""
        return {
            'current_pattern': self.current_pattern.__dict__ if self.current_pattern else None,
            'pattern_history': [p.__dict__ for p in self.pattern_history[-5:]],  # Last 5 patterns
            'buffer_size': len(self.buffer),
            'sample_rate': self.sample_rate
        }
        
    def adjust_neural_weights(self, current_weights: Dict[str, float]) -> Dict[str, float]:
        """Adjust neural processing weights based on breath patterns"""
        if not self.current_pattern:
            return current_weights
            
        # Base adjustments
        adjustments = {
            'neural_weight': 1.0,
            'language_weight': 1.0,
            'memory_weight': 1.0
        }
        
        # Adjust based on pattern type
        if self.current_pattern.pattern_type == 'deep_breath':
            adjustments['neural_weight'] *= 1.2
            adjustments['memory_weight'] *= 1.1
        elif self.current_pattern.pattern_type == 'shallow_breath':
            adjustments['language_weight'] *= 1.2
            adjustments['neural_weight'] *= 0.8
        elif self.current_pattern.pattern_type == 'rapid_breath':
            adjustments['neural_weight'] *= 1.3
            adjustments['language_weight'] *= 0.7
            
        # Apply intensity scaling
        intensity_factor = self.current_pattern.intensity
        for key in adjustments:
            adjustments[key] = 1.0 + (adjustments[key] - 1.0) * intensity_factor
            
        # Normalize weights
        total = sum(adjustments.values())
        for key in adjustments:
            adjustments[key] /= total
            
        return adjustments 
 
 
from typing import Dict, List, Optional
import time
from dataclasses import dataclass
import logging

@dataclass
class BreathPattern:
    """Represents a detected breath pattern"""
    pattern_type: str
    intensity: float
    duration: float
    timestamp: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class BreathDetector:
    """Detects and analyzes breath patterns for neural processing adjustment"""
    
    def __init__(self, sample_rate: int = 100, window_size: int = 50):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.buffer = []
        self.pattern_history = []
        self.last_breath_time = 0
        self.current_pattern = None
        
        # Pattern recognition thresholds
        self.thresholds = {
            'deep_breath': {'min_intensity': 0.7, 'min_duration': 1.5},
            'shallow_breath': {'min_intensity': 0.3, 'max_duration': 1.0},
            'rapid_breath': {'min_frequency': 2.0, 'max_duration': 0.5}
        }
        
    def process_sample(self, sample: float) -> Optional[BreathPattern]:
        """Process a new breath sample and detect patterns"""
        try:
            # Add sample to buffer
            self.buffer.append(sample)
            
            # Maintain window size
            if len(self.buffer) > self.window_size:
                self.buffer.pop(0)
                
            # Analyze pattern if we have enough samples
            if len(self.buffer) == self.window_size:
                return self._analyze_pattern()
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing breath sample: {str(e)}")
            return None
            
    def _analyze_pattern(self) -> Optional[BreathPattern]:
        """Analyze current buffer for breath patterns"""
        # Calculate basic metrics
        intensity = np.mean(np.abs(self.buffer))
        duration = len(self.buffer) / self.sample_rate
        frequency = 1.0 / (time.time() - self.last_breath_time) if self.last_breath_time > 0 else 0
        
        # Detect pattern type
        pattern_type = None
        confidence = 0.0
        
        if intensity >= self.thresholds['deep_breath']['min_intensity'] and duration >= self.thresholds['deep_breath']['min_duration']:
            pattern_type = 'deep_breath'
            confidence = 0.9
        elif intensity <= self.thresholds['shallow_breath']['min_intensity'] and duration <= self.thresholds['shallow_breath']['max_duration']:
            pattern_type = 'shallow_breath'
            confidence = 0.8
        elif frequency >= self.thresholds['rapid_breath']['min_frequency'] and duration <= self.thresholds['rapid_breath']['max_duration']:
            pattern_type = 'rapid_breath'
            confidence = 0.7
            
        if pattern_type:
            # Create pattern object
            pattern = BreathPattern(
                pattern_type=pattern_type,
                intensity=intensity,
                duration=duration,
                timestamp=time.time(),
                confidence=confidence,
                metadata={
                    'frequency': frequency,
                    'buffer_size': len(self.buffer)
                }
            )
            
            # Update history
            self.pattern_history.append(pattern)
            self.last_breath_time = time.time()
            self.current_pattern = pattern
            
            return pattern
            
        return None
        
    def get_current_state(self) -> Dict[str, Any]:
        """Get current breath detection state"""
        return {
            'current_pattern': self.current_pattern.__dict__ if self.current_pattern else None,
            'pattern_history': [p.__dict__ for p in self.pattern_history[-5:]],  # Last 5 patterns
            'buffer_size': len(self.buffer),
            'sample_rate': self.sample_rate
        }
        
    def adjust_neural_weights(self, current_weights: Dict[str, float]) -> Dict[str, float]:
        """Adjust neural processing weights based on breath patterns"""
        if not self.current_pattern:
            return current_weights
            
        # Base adjustments
        adjustments = {
            'neural_weight': 1.0,
            'language_weight': 1.0,
            'memory_weight': 1.0
        }
        
        # Adjust based on pattern type
        if self.current_pattern.pattern_type == 'deep_breath':
            adjustments['neural_weight'] *= 1.2
            adjustments['memory_weight'] *= 1.1
        elif self.current_pattern.pattern_type == 'shallow_breath':
            adjustments['language_weight'] *= 1.2
            adjustments['neural_weight'] *= 0.8
        elif self.current_pattern.pattern_type == 'rapid_breath':
            adjustments['neural_weight'] *= 1.3
            adjustments['language_weight'] *= 0.7
            
        # Apply intensity scaling
        intensity_factor = self.current_pattern.intensity
        for key in adjustments:
            adjustments[key] = 1.0 + (adjustments[key] - 1.0) * intensity_factor
            
        # Normalize weights
        total = sum(adjustments.values())
        for key in adjustments:
            adjustments[key] /= total
            
        return adjustments 
 