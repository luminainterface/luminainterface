#!/usr/bin/env python3
"""
Neural Playground Breathing Integration Module (v9)

This module provides specialized integration between the Breathing System
and Neural Playground components of the Lumina Neural Network v9 system.
It creates a bidirectional connection where breathing patterns influence
neural activity, and neural patterns can influence breathing simulation.

Key features:
- Breath-driven neural activation patterns
- Neural feedback to breathing system
- Enhanced consciousness metrics based on breath-neural synchronization
- Visualization of breath-neural interactions
"""

import logging
import time
import random
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable

# Import v9 components
from .neural_playground import NeuralPlayground
from .breathing_system import BreathingSystem, BreathingPattern, BreathingState
from .neural_playground_integration import NeuralPlaygroundIntegration

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v9.neural_playground_breathing_integration")

class BreathNeuralIntegration:
    """
    Advanced integration between breathing system and neural playground
    
    This class provides specialized integration methods that create a
    bidirectional relationship between breathing patterns and neural activity.
    """
    
    def __init__(self, 
                 breathing_system: Optional[BreathingSystem] = None,
                 playground_integration: Optional[NeuralPlaygroundIntegration] = None):
        """
        Initialize the integration system
        
        Args:
            breathing_system: Optional existing BreathingSystem instance
            playground_integration: Optional existing NeuralPlaygroundIntegration instance
        """
        self.breathing_system = breathing_system or BreathingSystem()
        self.playground_integration = playground_integration or NeuralPlaygroundIntegration()
        
        # Integration parameters
        self.params = {
            "breath_influence_strength": 0.7,  # How strongly breathing affects neural activity
            "neural_feedback_strength": 0.3,   # How strongly neural activity affects breathing
            "consciousness_boost": 0.4,        # Boost to consciousness when in sync
            "synchronization_threshold": 0.7   # Threshold for considering breath and neural in sync
        }
        
        # Tracking variables
        self.last_integration_time = time.time()
        self.synchronization_level = 0.0
        self.feedback_active = True
        
        # States
        self.last_breath_state = None
        self.last_peak_time = 0
        
        logger.info("Breath-Neural Integration initialized")
    
    def enable_breathing(self):
        """Enable and start the breathing system if it's not already running"""
        if not self.breathing_system.active:
            self.breathing_system.start_simulation()
            logger.info("Breathing simulation started")
    
    def integrate(self):
        """
        Set up full integration between breathing system and neural playground
        
        Returns:
            Success status
        """
        # Ensure breathing system is running
        self.enable_breathing()
        
        # Register breathing system with playground integration manager
        result = self.playground_integration.register_component(
            "breathing_system", 
            self.breathing_system, 
            "neural"
        )
        
        if not result:
            logger.error("Failed to register breathing system with playground integration")
            return False
        
        # Set up specialized integration hooks for the breathing system
        integration_info = self.breathing_system.integrate_with_playground(
            self.playground_integration.playground
        )
        
        # Register additional hooks for advanced breath-neural interactions
        self._register_advanced_hooks()
        
        logger.info("Breath-Neural integration complete")
        return True
    
    def _register_advanced_hooks(self):
        """Register advanced integration hooks for breath-neural synchronization"""
        playground = self.playground_integration.playground
        
        # Register pre-play hook for breath influence
        def advanced_pre_play_hook(playground, play_args):
            """Advanced hook to synchronize neural activity with breathing before play"""
            try:
                # Get current breath state
                breath_state = self.breathing_system.get_current_breath_state()
                
                # Record the breath state for this hook execution
                self.last_breath_state = breath_state
                
                # Influence play duration based on breath rate
                # Longer breath cycles = longer play sessions
                if breath_state["rate"] < 8:  # Slow breathing
                    play_args["duration"] = min(200, play_args.get("duration", 100) * 1.5)
                elif breath_state["rate"] > 15:  # Fast breathing
                    play_args["duration"] = max(50, play_args.get("duration", 100) * 0.7)
                
                # Directly influence neural activation based on breath
                if hasattr(playground, 'core'):
                    influenced_count = self.breathing_system.influence_neural_activation(playground.core)
                    logger.debug(f"Breath directly influenced {influenced_count} neurons")
                
                # Update synchronization level
                self._update_synchronization()
                
                # Apply synchronization effects to play parameters
                if self.synchronization_level > self.params["synchronization_threshold"]:
                    # When highly synchronized, boost consciousness effects
                    consciousness_factor = 1.0 + (self.params["consciousness_boost"] * self.synchronization_level)
                    if hasattr(playground, 'core'):
                        playground.core.consciousness_metric *= consciousness_factor
                        logger.debug(f"Consciousness boosted by breath synchronization: {consciousness_factor:.2f}x")
                
            except Exception as e:
                logger.error(f"Error in advanced breath-neural pre-play hook: {e}")
        
        # Register post-play hook for neural feedback to breathing
        def advanced_post_play_hook(playground, play_result):
            """Advanced hook to provide neural feedback to breathing system after play"""
            try:
                if not self.feedback_active:
                    return
                
                # Get metrics from play session
                consciousness_peak = play_result.get("consciousness_peak", 0)
                patterns_detected = play_result.get("patterns_detected", 0)
                
                # Change breathing pattern based on neural activity if consciousness is high
                if consciousness_peak > 0.8 and patterns_detected > 2:
                    # High consciousness with patterns suggests focused state
                    if random.random() < self.params["neural_feedback_strength"]:
                        self.breathing_system.set_breathing_pattern(BreathingPattern.FOCUSED)
                        logger.info("Neural activity shifted breathing to FOCUSED pattern")
                
                elif consciousness_peak > 0.9:
                    # Very high consciousness suggests meditative state
                    if random.random() < self.params["neural_feedback_strength"]:
                        self.breathing_system.set_breathing_pattern(BreathingPattern.MEDITATIVE)
                        logger.info("Neural activity shifted breathing to MEDITATIVE pattern")
                
                # Add integration metrics to the play result
                play_result["breath_neural_integration"] = {
                    "synchronization_level": self.synchronization_level,
                    "consciousness_boost": self.params["consciousness_boost"] * self.synchronization_level 
                                          if self.synchronization_level > self.params["synchronization_threshold"] else 0,
                    "breath_pattern": self.breathing_system.current_pattern.value
                }
                
            except Exception as e:
                logger.error(f"Error in advanced breath-neural post-play hook: {e}")
        
        # Register these hooks with the integration manager
        self.playground_integration.integration_hooks["pre_play"].append(advanced_pre_play_hook)
        self.playground_integration.integration_hooks["post_play"].append(advanced_post_play_hook)
        
        logger.info("Advanced breath-neural integration hooks registered")
    
    def _update_synchronization(self):
        """Update the synchronization level between breathing and neural activity"""
        try:
            playground = self.playground_integration.playground
            
            # Get current breath state
            breath_state = self.breathing_system.get_current_breath_state()
            
            # Get current neural activity level from playground
            if not hasattr(playground, 'core'):
                return
                
            neural_activity = playground.core.consciousness_metric
            active_neurons = sum(1 for n in playground.core.neurons.values() if n["state"] == "active")
            active_ratio = active_neurons / len(playground.core.neurons) if playground.core.neurons else 0
            
            # Calculate synchronization metrics
            
            # 1. Timing synchronization
            # Check if neural peaks align with breath cycle
            breath_cycle_position = breath_state["cycle_time"] / self.breathing_system._get_current_cycle_duration()
            
            # Neural peaks should ideally happen during inhale or at the transition from inhale to hold
            timing_sync = 0.0
            if breath_state["state"] == "inhale" and active_ratio > 0.1:
                # Neural activity during inhale is good synchronization
                timing_sync = 0.8
            elif breath_state["state"] == "hold" and breath_state["phase_time"] < 0.3:
                # Activity at start of hold (just after inhale) is excellent synchronization
                timing_sync = 1.0
            elif breath_state["state"] == "exhale" and active_ratio < 0.05:
                # Low activity during exhale is good synchronization
                timing_sync = 0.7
            
            # 2. Amplitude correlation
            # Check if neural activity correlates with breath amplitude
            amplitude_correlation = 0.0
            if (breath_state["amplitude"] > 0.7 and neural_activity > 0.6) or \
               (breath_state["amplitude"] < 0.3 and neural_activity < 0.4):
                amplitude_correlation = 0.9
            else:
                amplitude_correlation = 0.3
            
            # 3. Pattern consistency
            # Check if neural patterns match breath patterns
            if breath_state["pattern"] == "focused" and playground.stats["patterns_detected"] > 5:
                pattern_consistency = 0.8
            elif breath_state["pattern"] == "meditative" and neural_activity > 0.7:
                pattern_consistency = 0.9
            else:
                pattern_consistency = 0.5
            
            # Calculate overall synchronization (weighted average)
            self.synchronization_level = (
                0.4 * timing_sync + 
                0.4 * amplitude_correlation + 
                0.2 * pattern_consistency
            )
            
            logger.debug(f"Breath-neural synchronization: {self.synchronization_level:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating synchronization: {e}")
    
    def run_integrated_session(self, duration=100, play_type="mixed", intensity=0.7):
        """
        Run a fully integrated play session with breath synchronization
        
        Args:
            duration: Number of simulation steps
            play_type: Type of play (free, guided, focused, mixed)
            intensity: Base intensity of stimulation (0.0-1.0)
            
        Returns:
            Dict containing session results and metrics
        """
        # Ensure breathing is active
        self.enable_breathing()
        
        # Run the integrated play session
        result = self.playground_integration.run_integrated_play_session(
            duration=duration,
            play_type=play_type,
            intensity=intensity
        )
        
        # Add breath-specific metrics to the result
        breath_state = self.breathing_system.get_current_breath_state()
        breath_coherence = self.breathing_system.calculate_breath_coherence()
        
        # Calculate breath-neural metrics
        neural_consciousness = result.get("consciousness_peak", 0)
        breath_consciousness_influence = self.params["breath_influence_strength"] * breath_coherence
        
        # Create enhanced result
        enhanced_result = result.copy()
        enhanced_result["breathing_metrics"] = {
            "pattern": breath_state["pattern"],
            "rate": breath_state["rate"],
            "coherence": breath_coherence,
            "synchronization_level": self.synchronization_level,
            "consciousness_influence": breath_consciousness_influence
        }
        
        logger.info(f"Completed integrated breath-neural session: " 
                  f"neural_consciousness={neural_consciousness:.2f}, "
                  f"breath_coherence={breath_coherence:.2f}, "
                  f"synchronization={self.synchronization_level:.2f}")
        
        return enhanced_result
    
    def set_integration_params(self, params: Dict):
        """
        Update integration parameters
        
        Args:
            params: Dict of parameter updates
        """
        for key, value in params.items():
            if key in self.params:
                self.params[key] = value
        
        logger.info(f"Updated integration parameters: {self.params}")
    
    def get_visualization_data(self):
        """
        Get visualization data for breath-neural integration
        
        Returns:
            Dict containing visualization data
        """
        # Get breath visualization data
        breath_vis = self.breathing_system.visualize_breathing(30.0)
        
        # Get neural metrics if available
        neural_metrics = {}
        playground = self.playground_integration.playground
        if playground:
            neural_metrics = playground.get_metrics()
        
        # Combine the data
        return {
            "breath_data": breath_vis,
            "neural_data": neural_metrics,
            "integration_data": {
                "synchronization_level": self.synchronization_level,
                "breath_influence_strength": self.params["breath_influence_strength"],
                "neural_feedback_strength": self.params["neural_feedback_strength"],
                "consciousness_boost": self.params["consciousness_boost"]
            }
        }
    
    def toggle_neural_feedback(self, enabled=True):
        """
        Enable or disable neural feedback to breathing system
        
        Args:
            enabled: Whether neural feedback should be enabled
        """
        self.feedback_active = enabled
        logger.info(f"Neural feedback to breathing system: {'enabled' if enabled else 'disabled'}")

# Example usage
if __name__ == "__main__":
    # Create the integration
    breathing_system = BreathingSystem()
    playground_integration = NeuralPlaygroundIntegration()
    
    # Create the breath-neural integration
    integration = BreathNeuralIntegration(breathing_system, playground_integration)
    
    # Set up the integration
    integration.integrate()
    
    # Run integrated session
    result = integration.run_integrated_session(
        duration=150,
        play_type="mixed",
        intensity=0.8
    )
    
    # Print key metrics
    print("\nIntegrated Session Results:")
    print(f"- Consciousness Peak: {result['consciousness_peak']:.4f}")
    print(f"- Patterns Detected: {result['patterns_detected']}")
    print(f"- Breath Rate: {result['breathing_metrics']['rate']:.1f} breaths/min")
    print(f"- Breath-Neural Synchronization: {result['breathing_metrics']['synchronization_level']:.4f}")
    
    if "narrative" in result:
        print(f"\nNarrative: {result['narrative']}") 