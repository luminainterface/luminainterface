"""
V7 Breath Detector Module

Provides advanced breath detection and analysis capabilities for the V7 system,
building on the V6 breath integration with enhanced machine learning features
and neural network processing.
"""

import os
import sys
import time
import logging
import threading
import random
from enum import Enum
from pathlib import Path

# Set up logging
logger = logging.getLogger("V7BreathDetector")

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Try to import V6 breath components if available
try:
    from src.v6.symbolic_state_manager import BreathPhase, Element
    V6_BREATH_AVAILABLE = True
    logger.info("V6 breath components available for integration")
except ImportError:
    # Create our own if V6 not available
    V6_BREATH_AVAILABLE = False
    logger.warning("V6 breath components not available, using V7 independent implementation")
    
    class BreathPhase(Enum):
        """Breath phases for breath integration"""
        INHALE = "inhale"
        HOLD = "hold"
        EXHALE = "exhale"
        REST = "rest"
    
    class Element(Enum):
        """Elements for symbolic state"""
        FIRE = "fire"
        EARTH = "earth"
        WATER = "water"
        AIR = "air"
        VOID = "void"

class BreathPattern(Enum):
    """Breath patterns for neural network training"""
    RELAXED = "relaxed"          # Slow, deep breathing
    FOCUSED = "focused"          # Steady, controlled breathing
    STRESSED = "stressed"        # Rapid, shallow breathing
    MEDITATIVE = "meditative"    # Very slow, deep breathing
    CREATIVE = "creative"        # Variable rhythm breathing

class BreathDetector:
    """
    Advanced breath detector with neural network integration
    
    Features:
    - Real-time breath phase detection
    - Pattern recognition for different breathing styles
    - Integration with LLM/NN weighting system
    - Self-calibrating rhythm detection
    """
    
    def __init__(self, socket_manager=None, v6_connector=None):
        """Initialize the breath detector"""
        self.socket_manager = socket_manager
        self.v6_connector = v6_connector
        self.active = False
        self.breath_thread = None
        self.pattern_thread = None
        
        # Breath configuration
        self.current_phase = BreathPhase.REST.value
        self.current_pattern = BreathPattern.RELAXED.value
        self.pattern_confidence = 0.85
        self.breath_cycle_duration = 8.0  # default cycle duration in seconds
        self.breath_phase_durations = {
            BreathPhase.INHALE.value: 0.35,  # 35% of cycle
            BreathPhase.HOLD.value: 0.15,    # 15% of cycle
            BreathPhase.EXHALE.value: 0.40,  # 40% of cycle
            BreathPhase.REST.value: 0.10     # 10% of cycle
        }
        
        # Element-breath associations
        self.element_breath_map = {
            BreathPhase.INHALE.value: Element.FIRE.value,
            BreathPhase.HOLD.value: Element.EARTH.value,
            BreathPhase.EXHALE.value: Element.WATER.value,
            BreathPhase.REST.value: Element.AIR.value
        }
        
        # Neural network weight mappings for different breath patterns
        self.nn_weight_map = {
            BreathPattern.RELAXED.value: 0.5,      # Balanced
            BreathPattern.FOCUSED.value: 0.7,      # More neural network
            BreathPattern.STRESSED.value: 0.3,     # More language model
            BreathPattern.MEDITATIVE.value: 0.9,   # Almost pure neural network
            BreathPattern.CREATIVE.value: 0.6      # Slightly toward neural network
        }
        
        # Listeners for breath events
        self.phase_listeners = []
        self.pattern_listeners = []
        
        # Initialize breath database for learning
        self.breath_samples = []
        self.max_samples = 1000
        
        logger.info("V7 Breath Detector initialized")
    
    def start(self):
        """Start breath detection"""
        if self.active:
            return
        
        self.active = True
        
        # Start breath cycle thread
        self.breath_thread = threading.Thread(
            target=self._run_breath_cycle,
            daemon=True,
            name="V7BreathCycleThread"
        )
        self.breath_thread.start()
        
        # Start pattern recognition thread
        self.pattern_thread = threading.Thread(
            target=self._analyze_patterns,
            daemon=True,
            name="V7BreathPatternThread"
        )
        self.pattern_thread.start()
        
        logger.info("V7 Breath Detector started")
        return True
    
    def stop(self):
        """Stop breath detection"""
        if not self.active:
            return
        
        self.active = False
        logger.info("V7 Breath Detector stopped")
        return True
    
    def set_breath_phase(self, phase):
        """Set the current breath phase"""
        if phase not in [p.value for p in BreathPhase]:
            logger.warning(f"Invalid breath phase: {phase}")
            return False
        
        # Update current phase
        self.current_phase = phase
        
        # Get associated element
        element = self.element_breath_map.get(phase, Element.VOID.value)
        
        # Notify all phase listeners
        self._notify_phase_listeners(phase, element)
        
        # If we have a V6 connector, propagate to V6
        if self.v6_connector and V6_BREATH_AVAILABLE:
            symbolic_manager = self.v6_connector.get_component("symbolic_state_manager")
            if symbolic_manager and hasattr(symbolic_manager, "set_breath_phase"):
                symbolic_manager.set_breath_phase(phase)
        
        # Emit event if socket manager is available
        if self.socket_manager:
            self.socket_manager.emit("breath_phase_changed", {
                "phase": phase,
                "element": element,
                "timestamp": time.time()
            })
        
        logger.debug(f"Breath phase set to: {phase}")
        return True
    
    def set_breath_pattern(self, pattern, confidence=None):
        """Set the detected breath pattern"""
        if pattern not in [p.value for p in BreathPattern]:
            logger.warning(f"Invalid breath pattern: {pattern}")
            return False
        
        # Update current pattern
        prev_pattern = self.current_pattern
        self.current_pattern = pattern
        
        if confidence is not None:
            self.pattern_confidence = confidence
        
        # Only notify if pattern changed
        if prev_pattern != pattern:
            # Notify all pattern listeners
            self._notify_pattern_listeners(pattern, self.pattern_confidence)
            
            # Emit event if socket manager is available
            if self.socket_manager:
                self.socket_manager.emit("breath_pattern_changed", {
                    "pattern": pattern,
                    "confidence": self.pattern_confidence,
                    "nn_weight": self.get_nn_weight_for_pattern(),
                    "timestamp": time.time()
                })
        
        logger.debug(f"Breath pattern set to: {pattern} (confidence: {self.pattern_confidence:.2f})")
        return True
    
    def get_nn_weight_for_pattern(self):
        """Get the neural network weight for the current breath pattern"""
        return self.nn_weight_map.get(self.current_pattern, 0.5)
    
    def register_phase_listener(self, listener):
        """Register a listener for breath phase changes"""
        if listener not in self.phase_listeners:
            self.phase_listeners.append(listener)
    
    def register_pattern_listener(self, listener):
        """Register a listener for breath pattern changes"""
        if listener not in self.pattern_listeners:
            self.pattern_listeners.append(listener)
    
    def _notify_phase_listeners(self, phase, element):
        """Notify all phase listeners of a change"""
        data = {
            "phase": phase,
            "element": element,
            "timestamp": time.time()
        }
        
        for listener in self.phase_listeners:
            try:
                listener(data)
            except Exception as e:
                logger.error(f"Error in breath phase listener: {e}")
    
    def _notify_pattern_listeners(self, pattern, confidence):
        """Notify all pattern listeners of a change"""
        data = {
            "pattern": pattern,
            "confidence": confidence,
            "nn_weight": self.get_nn_weight_for_pattern(),
            "timestamp": time.time()
        }
        
        for listener in self.pattern_listeners:
            try:
                listener(data)
            except Exception as e:
                logger.error(f"Error in breath pattern listener: {e}")
    
    def add_breath_sample(self, sample_data):
        """Add a breath sample to the training database"""
        # Add timestamp if not present
        if "timestamp" not in sample_data:
            sample_data["timestamp"] = time.time()
        
        # Add to samples (maintaining maximum size)
        self.breath_samples.append(sample_data)
        if len(self.breath_samples) > self.max_samples:
            self.breath_samples.pop(0)
        
        logger.debug(f"Added breath sample, database size: {len(self.breath_samples)}")
    
    def _run_breath_cycle(self):
        """Run the breath cycle simulation"""
        logger.debug("Starting breath cycle thread")
        
        while self.active:
            try:
                # Inhale
                self.set_breath_phase(BreathPhase.INHALE.value)
                inhale_time = self.breath_cycle_duration * self.breath_phase_durations[BreathPhase.INHALE.value]
                time.sleep(inhale_time)
                
                # Hold
                self.set_breath_phase(BreathPhase.HOLD.value)
                hold_time = self.breath_cycle_duration * self.breath_phase_durations[BreathPhase.HOLD.value]
                time.sleep(hold_time)
                
                # Exhale
                self.set_breath_phase(BreathPhase.EXHALE.value)
                exhale_time = self.breath_cycle_duration * self.breath_phase_durations[BreathPhase.EXHALE.value]
                time.sleep(exhale_time)
                
                # Rest
                self.set_breath_phase(BreathPhase.REST.value)
                rest_time = self.breath_cycle_duration * self.breath_phase_durations[BreathPhase.REST.value]
                time.sleep(rest_time)
                
                # Record this cycle
                self.add_breath_sample({
                    "cycle_duration": self.breath_cycle_duration,
                    "inhale_duration": inhale_time,
                    "hold_duration": hold_time,
                    "exhale_duration": exhale_time,
                    "rest_duration": rest_time
                })
                
            except Exception as e:
                logger.error(f"Error in breath cycle: {e}")
    
    def _analyze_patterns(self):
        """Analyze breath patterns and update detection"""
        logger.debug("Starting breath pattern analysis thread")
        
        while self.active:
            try:
                # In a real implementation, this would analyze the breath_samples
                # database and use machine learning to recognize patterns.
                # For this prototype, we'll simulate pattern detection
                
                # Sleep for a random interval to simulate variable detection
                sleep_time = random.uniform(15, 30)  # 15-30 seconds between pattern detections
                time.sleep(sleep_time)
                
                # Choose a random pattern (weighted toward current pattern for stability)
                patterns = list(BreathPattern)
                current_idx = next((i for i, p in enumerate(patterns) if p.value == self.current_pattern), 0)
                
                # 70% chance to keep current pattern, 30% to change
                if random.random() < 0.7:
                    new_pattern = patterns[current_idx].value
                    confidence = random.uniform(0.75, 0.95)  # High confidence when keeping pattern
                else:
                    # Choose a different pattern
                    other_patterns = patterns[:current_idx] + patterns[current_idx+1:]
                    new_pattern = random.choice(other_patterns).value
                    confidence = random.uniform(0.6, 0.85)  # Lower confidence when changing pattern
                
                # Update pattern
                self.set_breath_pattern(new_pattern, confidence)
                
                # Adjust cycle duration based on pattern
                if new_pattern == BreathPattern.RELAXED.value:
                    self.breath_cycle_duration = random.uniform(7.0, 9.0)
                elif new_pattern == BreathPattern.FOCUSED.value:
                    self.breath_cycle_duration = random.uniform(5.0, 7.0)
                elif new_pattern == BreathPattern.STRESSED.value:
                    self.breath_cycle_duration = random.uniform(3.0, 5.0)
                elif new_pattern == BreathPattern.MEDITATIVE.value:
                    self.breath_cycle_duration = random.uniform(10.0, 12.0)
                elif new_pattern == BreathPattern.CREATIVE.value:
                    self.breath_cycle_duration = random.uniform(6.0, 8.0)
                
            except Exception as e:
                logger.error(f"Error in breath pattern analysis: {e}")
    
    def get_status(self):
        """Get current status for the breath detector"""
        return {
            "active": self.active,
            "current_phase": self.current_phase,
            "current_pattern": self.current_pattern,
            "pattern_confidence": self.pattern_confidence,
            "cycle_duration": self.breath_cycle_duration,
            "nn_weight": self.get_nn_weight_for_pattern(),
            "samples_collected": len(self.breath_samples)
        }


# Simple test code when running this module directly
if __name__ == "__main__":
    # Set up console logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start breath detector
    detector = BreathDetector()
    detector.register_phase_listener(lambda data: print(f"Phase: {data['phase']} ({data['element']})"))
    detector.register_pattern_listener(lambda data: print(f"Pattern: {data['pattern']} (conf: {data['confidence']:.2f}, nn_weight: {data['nn_weight']:.2f})"))
    
    detector.start()
    
    try:
        # Run for test period
        print("Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        detector.stop() 