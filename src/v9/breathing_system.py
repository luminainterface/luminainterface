#!/usr/bin/env python3
"""
Breathing System Module (v9)

This module provides simulated breathing patterns that can influence
the Lumina Neural Network system. It includes simulation of various
breathing patterns and is designed to be extended with real breathing
input captured from a microphone.

Key features:
- Simulated breathing patterns (calm, excited, focused, etc.)
- Breathing-neural integration capabilities
- Framework for future microphone breathing capture
- Breath visualization
"""

import logging
import time
import random
import math
import numpy as np
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from datetime import datetime

# Define component type for auto-discovery
LUMINA_COMPONENT_TYPE = "neural"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v9.breathing_system")

class BreathingState(Enum):
    """Enum for different breathing states"""
    INHALE = "inhale"
    EXHALE = "exhale"
    HOLD = "hold"
    PAUSE = "pause"

class BreathingPattern(Enum):
    """Enum for different breathing patterns"""
    CALM = "calm"           # Slow, deep breathing
    EXCITED = "excited"     # Rapid, shallow breathing
    FOCUSED = "focused"     # Controlled, steady breathing
    MEDITATIVE = "meditative"  # Very slow, deep breathing with holds
    RANDOM = "random"       # Unpredictable pattern (for testing)
    CUSTOM = "custom"       # User-defined pattern

class BreathingSystem:
    """
    Simulated breathing system that can influence neural playground dynamics
    
    This class simulates different breathing patterns and provides
    mechanisms to influence neural activity based on breath state.
    It is designed to be extended with real breathing data in the future.
    """
    
    def __init__(self, 
                 simulation_rate: float = 10.0,  # Hz
                 default_pattern: BreathingPattern = BreathingPattern.CALM):
        """
        Initialize the breathing system
        
        Args:
            simulation_rate: Rate at which breathing is simulated (Hz)
            default_pattern: Default breathing pattern to use
        """
        self.simulation_rate = simulation_rate
        self.time_step = 1.0 / simulation_rate
        self.current_pattern = default_pattern
        self.current_state = BreathingState.PAUSE
        
        # Timing parameters (in seconds)
        self.pattern_params = {
            BreathingPattern.CALM: {
                "inhale_duration": 4.0,
                "exhale_duration": 6.0,
                "hold_duration": 0.0,
                "pause_duration": 0.5,
                "amplitude": 0.8,
                "variability": 0.1,
            },
            BreathingPattern.EXCITED: {
                "inhale_duration": 1.5,
                "exhale_duration": 1.5,
                "hold_duration": 0.0,
                "pause_duration": 0.2,
                "amplitude": 0.9,
                "variability": 0.3,
            },
            BreathingPattern.FOCUSED: {
                "inhale_duration": 4.0,
                "exhale_duration": 4.0,
                "hold_duration": 1.0,
                "pause_duration": 0.2,
                "amplitude": 0.7,
                "variability": 0.05,
            },
            BreathingPattern.MEDITATIVE: {
                "inhale_duration": 6.0,
                "exhale_duration": 8.0,
                "hold_duration": 4.0,
                "pause_duration": 0.0,
                "amplitude": 0.9,
                "variability": 0.05,
            },
            BreathingPattern.RANDOM: {
                "inhale_duration": 2.0,
                "exhale_duration": 2.0,
                "hold_duration": 0.5,
                "pause_duration": 0.5,
                "amplitude": 0.7,
                "variability": 0.5,
            },
            BreathingPattern.CUSTOM: {
                "inhale_duration": 3.0,
                "exhale_duration": 3.0,
                "hold_duration": 0.0,
                "pause_duration": 0.5,
                "amplitude": 0.8,
                "variability": 0.1,
            }
        }
        
        # Current breathing metrics
        self.breath_amplitude = 0.0
        self.breath_rate = 0.0
        self.breath_coherence = 0.95  # How consistent the breathing is
        self.breath_depth = 0.0
        self.breath_hold_duration = 0.0
        
        # Breath history (for analysis and visualization)
        self.history_max_length = int(60 * simulation_rate)  # Store 60 seconds of data
        self.breath_history = []
        
        # State tracking
        self.current_cycle_time = 0.0
        self.current_phase_time = 0.0
        self.active = False
        self.thread = None
        self.last_real_breath_time = time.time()
        
        # Microphone integration placeholders
        self.microphone_enabled = False
        self.mic_data_buffer = []
        self.mic_calibration = {
            "baseline": 0.0,
            "max_amplitude": 1.0
        }
        
        # Neural influence parameters
        self.neural_influence = {
            "consciousness_factor": 0.2,  # How much breathing affects consciousness
            "activation_threshold": 0.6,  # Breath amplitude needed to trigger neural activations
            "coherence_influence": 0.5    # How much breath coherence affects neural stability
        }
        
        # Initialize with some history data
        self._initialize_history()
        
        logger.info(f"Breathing system initialized with {default_pattern.value} pattern")
    
    def _initialize_history(self):
        """Initialize the breath history with some data matching the current pattern"""
        cycle_duration = self._get_current_cycle_duration()
        steps = min(100, self.history_max_length)
        
        for i in range(steps):
            t = (i / steps) * cycle_duration
            amplitude = self._calculate_breath_amplitude(t, cycle_duration)
            self.breath_history.append({
                "timestamp": time.time() - (steps - i) * self.time_step,
                "amplitude": amplitude,
                "state": self._determine_breath_state(t, cycle_duration),
                "pattern": self.current_pattern.value
            })
    
    def _get_current_cycle_duration(self) -> float:
        """Calculate the total duration of the current breathing cycle"""
        params = self.pattern_params[self.current_pattern]
        return (params["inhale_duration"] + 
                params["exhale_duration"] + 
                params["hold_duration"] + 
                params["pause_duration"])
    
    def _determine_breath_state(self, t: float, cycle_duration: float) -> BreathingState:
        """Determine the breath state at time t within a cycle"""
        params = self.pattern_params[self.current_pattern]
        
        # Normalize t to be within cycle_duration
        t = t % cycle_duration
        
        # Determine the state based on where we are in the cycle
        if t < params["inhale_duration"]:
            return BreathingState.INHALE
        t -= params["inhale_duration"]
        
        if t < params["hold_duration"]:
            return BreathingState.HOLD
        t -= params["hold_duration"]
        
        if t < params["exhale_duration"]:
            return BreathingState.EXHALE
        t -= params["exhale_duration"]
        
        return BreathingState.PAUSE
    
    def _calculate_breath_amplitude(self, t: float, cycle_duration: float) -> float:
        """Calculate the breath amplitude at time t within a cycle"""
        params = self.pattern_params[self.current_pattern]
        state = self._determine_breath_state(t, cycle_duration)
        
        # Normalize t to be within cycle_duration
        t = t % cycle_duration
        
        # Apply slight variability to amplitude
        base_amplitude = params["amplitude"]
        variability = params["variability"]
        amplitude_variation = base_amplitude * (1.0 + random.uniform(-variability, variability))
        
        # Calculate amplitude based on breath state
        if state == BreathingState.INHALE:
            # Increasing from 0 to max during inhale
            phase_progress = t / params["inhale_duration"]
            return amplitude_variation * self._smooth_curve(phase_progress)
        
        t -= params["inhale_duration"]
        
        if state == BreathingState.HOLD:
            # Maintain max amplitude during hold
            return amplitude_variation
        
        t -= params["hold_duration"]
        
        if state == BreathingState.EXHALE:
            # Decreasing from max to 0 during exhale
            phase_progress = t / params["exhale_duration"]
            return amplitude_variation * (1.0 - self._smooth_curve(phase_progress))
        
        # In PAUSE state, amplitude is 0
        return 0.0
    
    def _smooth_curve(self, t: float) -> float:
        """Create a smooth curve between 0 and 1 using sine function"""
        # Map t from [0,1] to [0,pi/2] for a quarter sine wave
        return math.sin(t * math.pi / 2)
    
    def start_simulation(self):
        """Start the breathing simulation thread"""
        if self.active:
            logger.warning("Breathing simulation already running")
            return
        
        self.active = True
        self.thread = threading.Thread(target=self._simulation_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Started breathing simulation with {self.current_pattern.value} pattern")
    
    def stop_simulation(self):
        """Stop the breathing simulation thread"""
        self.active = False
        if self.thread:
            self.thread.join(timeout=1.0)
            logger.info("Stopped breathing simulation")
    
    def _simulation_loop(self):
        """Main simulation loop running in a separate thread"""
        last_time = time.time()
        
        while self.active:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Update simulation state
            self._update_simulation(dt)
            
            # Sleep to maintain simulation rate
            sleep_time = max(0.0, self.time_step - (time.time() - current_time))
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _update_simulation(self, dt: float):
        """Update the breathing simulation by the specified time step"""
        # Update cycle time
        self.current_cycle_time += dt
        self.current_phase_time += dt
        
        # Get current pattern parameters
        cycle_duration = self._get_current_cycle_duration()
        
        # Check if we've moved to a new breath state
        new_state = self._determine_breath_state(self.current_cycle_time, cycle_duration)
        if new_state != self.current_state:
            self.current_phase_time = 0.0
            self.current_state = new_state
            logger.debug(f"Breath state changed to {self.current_state.value}")
        
        # Calculate current breath amplitude
        self.breath_amplitude = self._calculate_breath_amplitude(
            self.current_cycle_time, cycle_duration)
        
        # Calculate breath rate (breaths per minute)
        self.breath_rate = 60.0 / cycle_duration
        
        # Calculate breath depth based on recent history
        if len(self.breath_history) > 0:
            recent_amplitudes = [entry["amplitude"] for entry in self.breath_history[-int(cycle_duration * self.simulation_rate):]]
            if recent_amplitudes:
                self.breath_depth = max(0.1, max(recent_amplitudes) - min(recent_amplitudes))
        
        # Record current breath state
        self.breath_history.append({
            "timestamp": time.time(),
            "amplitude": self.breath_amplitude,
            "state": self.current_state.value,
            "pattern": self.current_pattern.value
        })
        
        # Trim history if it's too long
        if len(self.breath_history) > self.history_max_length:
            self.breath_history = self.breath_history[-self.history_max_length:]
    
    def set_breathing_pattern(self, pattern: BreathingPattern):
        """
        Change the current breathing pattern
        
        Args:
            pattern: The new breathing pattern to use
        """
        self.current_pattern = pattern
        logger.info(f"Changed breathing pattern to {pattern.value}")
    
    def get_custom_pattern_params(self) -> Dict:
        """Get the current custom pattern parameters"""
        return self.pattern_params[BreathingPattern.CUSTOM].copy()
    
    def set_custom_pattern_params(self, params: Dict):
        """
        Set custom breathing pattern parameters
        
        Args:
            params: Dict containing custom pattern parameters
        """
        # Validate parameters
        required_keys = ["inhale_duration", "exhale_duration", 
                        "hold_duration", "pause_duration", 
                        "amplitude", "variability"]
        
        for key in required_keys:
            if key not in params:
                logger.error(f"Missing required key in custom pattern params: {key}")
                return
        
        # Update the parameters
        self.pattern_params[BreathingPattern.CUSTOM] = params.copy()
        logger.info("Updated custom breathing pattern parameters")
    
    def get_current_breath_state(self) -> Dict:
        """
        Get the current breathing state
        
        Returns:
            Dict containing current breathing information
        """
        return {
            "state": self.current_state.value,
            "pattern": self.current_pattern.value,
            "amplitude": self.breath_amplitude,
            "rate": self.breath_rate,
            "coherence": self.breath_coherence,
            "depth": self.breath_depth,
            "cycle_time": self.current_cycle_time,
            "phase_time": self.current_phase_time
        }
    
    def get_breath_history(self, seconds: float = None) -> List[Dict]:
        """
        Get breath history for the specified number of seconds
        
        Args:
            seconds: Number of seconds of history to return (None for all)
            
        Returns:
            List of breath history entries
        """
        if seconds is None:
            return self.breath_history.copy()
        
        # Calculate how many entries to return
        entries = int(seconds * self.simulation_rate)
        return self.breath_history[-entries:].copy() if entries > 0 else []
    
    def calculate_breath_coherence(self, window_seconds: float = 30.0) -> float:
        """
        Calculate breathing coherence (regularity) over a time window
        
        Args:
            window_seconds: Window size in seconds
            
        Returns:
            Coherence score (0.0-1.0)
        """
        # Get relevant history entries
        history = self.get_breath_history(window_seconds)
        if not history or len(history) < 10:
            return 0.0
        
        # Calculate cycle times
        cycle_starts = []
        for i in range(1, len(history)):
            if (history[i-1]["state"] != BreathingState.INHALE.value and 
                history[i]["state"] == BreathingState.INHALE.value):
                cycle_starts.append(history[i]["timestamp"])
        
        if len(cycle_starts) < 2:
            return 0.0
        
        # Calculate interval between cycles
        intervals = [cycle_starts[i] - cycle_starts[i-1] for i in range(1, len(cycle_starts))]
        if not intervals:
            return 0.0
        
        # Calculate standard deviation and mean of intervals
        mean_interval = sum(intervals) / len(intervals)
        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0
        
        # Calculate coefficient of variation (lower is more coherent)
        cv = std_dev / mean_interval if mean_interval > 0 else 1.0
        
        # Map CV to coherence score (0.0-1.0)
        # A CV of 0 means perfect regularity (coherence of 1.0)
        # A CV of 0.5 or higher means poor regularity (coherence of 0.0)
        coherence = max(0.0, min(1.0, 1.0 - 2.0 * cv))
        
        self.breath_coherence = coherence
        return coherence
    
    # ===== Microphone Integration Methods =====
    
    def enable_microphone(self, device_index: int = None):
        """
        Enable microphone input for breath detection
        
        Args:
            device_index: Optional microphone device index
        """
        # This is a placeholder for actual microphone setup
        # A real implementation would initialize audio capture here
        logger.info("Microphone integration is a placeholder in this version")
        
        try:
            # Simulate microphone initialization
            self.microphone_enabled = True
            logger.info(f"Microphone enabled (placeholder) with device index: {device_index}")
            return True
        except Exception as e:
            logger.error(f"Error enabling microphone: {e}")
            return False
    
    def disable_microphone(self):
        """Disable microphone input"""
        # Placeholder for stopping audio capture
        self.microphone_enabled = False
        logger.info("Microphone disabled")
    
    def calibrate_microphone(self):
        """
        Calibrate microphone for breath detection
        
        Returns:
            Success status
        """
        # Placeholder for real calibration
        logger.info("Calibrating microphone (placeholder implementation)")
        
        # Simulate calibration process
        self.mic_calibration = {
            "baseline": 0.1,
            "max_amplitude": 0.8,
            "noise_floor": 0.05
        }
        
        logger.info("Microphone calibration completed")
        return True
    
    def detect_real_breath(self, audio_data):
        """
        Process audio data to detect breathing
        
        Args:
            audio_data: Audio data from microphone
            
        Returns:
            Detected breath amplitude
        """
        # This is a placeholder for real breath detection from audio
        # A real implementation would analyze audio to detect breathing patterns
        
        # For now, just return the simulated breath
        return self.breath_amplitude
    
    # ===== Neural Integration Methods =====
    
    def integrate_with_playground(self, playground):
        """
        Integrate with Neural Playground
        
        Args:
            playground: NeuralPlayground instance
            
        Returns:
            Dict containing integration information and hooks
        """
        logger.info("Integrating breathing system with Neural Playground")
        
        # Start the breathing simulation if not already running
        if not self.active:
            self.start_simulation()
        
        # Define hooks
        def pre_play_hook(playground, play_args):
            """Hook called before play session"""
            try:
                # Get current breath state
                breath_state = self.get_current_breath_state()
                
                # Modify play parameters based on breathing
                # Breath amplitude affects play intensity
                intensity_factor = 0.7 + (breath_state["amplitude"] * 0.5)
                play_args["intensity"] = min(1.0, max(0.1, 
                                           play_args.get("intensity", 0.5) * intensity_factor))
                
                # Breath pattern affects play type
                if self.current_pattern == BreathingPattern.CALM:
                    # Calm breathing favors free play
                    if play_args.get("play_type") == "mixed":
                        play_args["play_type"] = "mixed"  # Keep mixed
                    elif random.random() < 0.7:  # 70% chance to switch to free
                        play_args["play_type"] = "free"
                
                elif self.current_pattern == BreathingPattern.FOCUSED:
                    # Focused breathing favors guided play
                    if play_args.get("play_type") == "mixed":
                        play_args["play_type"] = "mixed"  # Keep mixed
                    elif random.random() < 0.7:  # 70% chance to switch to guided
                        play_args["play_type"] = "guided"
                
                elif self.current_pattern == BreathingPattern.MEDITATIVE:
                    # Meditative breathing favors focused play
                    if play_args.get("play_type") == "mixed":
                        play_args["play_type"] = "mixed"  # Keep mixed
                    elif random.random() < 0.7:  # 70% chance to switch to focused
                        play_args["play_type"] = "focused"
                
                # Log the influence
                logger.debug(f"Breathing influence on play: " 
                           f"intensity={play_args['intensity']:.2f}, "
                           f"play_type={play_args['play_type']}")
                
            except Exception as e:
                logger.error(f"Error in breathing system pre-play hook: {e}")
        
        def post_play_hook(playground, play_result):
            """Hook called after play session"""
            try:
                # Get current breath state
                breath_state = self.get_current_breath_state()
                
                # Calculate how much the breathing system influenced the play session
                # This is for reporting purposes
                coherence = self.calculate_breath_coherence()
                
                # Add breathing data to the play result for downstream components
                play_result["breathing_data"] = {
                    "pattern": self.current_pattern.value,
                    "state": self.current_state.value,
                    "amplitude": self.breath_amplitude,
                    "rate": self.breath_rate,
                    "coherence": coherence,
                    "influence_factor": self.neural_influence["consciousness_factor"]
                }
                
                logger.debug(f"Breathing data added to play result: "
                           f"pattern={self.current_pattern.value}, "
                           f"coherence={coherence:.2f}")
                
            except Exception as e:
                logger.error(f"Error in breathing system post-play hook: {e}")
        
        # Define a pattern detected hook
        def pattern_detected_hook(playground, pattern):
            """Hook called when a pattern is detected"""
            try:
                # Correlate pattern with breathing state
                breath_state = self.get_current_breath_state()
                
                # Add breathing context to the pattern
                pattern["breathing_context"] = {
                    "state": self.current_state.value,
                    "amplitude": self.breath_amplitude,
                    "pattern": self.current_pattern.value
                }
                
                logger.debug(f"Pattern detection correlated with breathing state: "
                           f"{self.current_state.value}")
                
            except Exception as e:
                logger.error(f"Error in breathing system pattern hook: {e}")
        
        # Return integration information and hooks
        return {
            "component_type": "breathing_system",
            "hooks": {
                "pre_play": pre_play_hook,
                "post_play": post_play_hook,
                "pattern_detected": pattern_detected_hook
            }
        }
    
    def influence_neural_activation(self, playground_core):
        """
        Directly influence neural activation based on current breath
        
        Args:
            playground_core: NeuralPlaygroundCore instance
            
        Returns:
            Number of influenced neurons
        """
        if not hasattr(playground_core, 'activate') or not callable(playground_core.activate):
            logger.error("Invalid playground core provided")
            return 0
        
        # Only activate neurons when breath amplitude is above threshold
        if self.breath_amplitude < self.neural_influence["activation_threshold"]:
            return 0
        
        # Get list of neurons
        neuron_ids = list(playground_core.neurons.keys())
        if not neuron_ids:
            return 0
        
        # Number of neurons to activate based on breath amplitude
        activation_count = max(1, int(len(neuron_ids) * self.breath_amplitude * 0.1))
        
        # Activate neurons
        influenced_count = 0
        for _ in range(activation_count):
            # Select random neuron
            neuron_id = random.choice(neuron_ids)
            
            # Influence based on breath state
            if self.current_state == BreathingState.INHALE:
                # Inhaling tends to excite
                activation_value = self.breath_amplitude
            elif self.current_state == BreathingState.EXHALE:
                # Exhaling tends to inhibit
                activation_value = self.breath_amplitude * 0.5
            else:
                # Other states have moderate influence
                activation_value = self.breath_amplitude * 0.25
            
            # Apply coherence factor (more coherent breathing has stronger effect)
            activation_value *= (0.5 + 0.5 * self.breath_coherence)
            
            # Activate the neuron
            if playground_core.activate(neuron_id, activation_value):
                influenced_count += 1
        
        return influenced_count
    
    def set_neural_influence_params(self, params: Dict):
        """
        Set parameters controlling how breathing influences neural activity
        
        Args:
            params: Dict of influence parameters
        """
        for key, value in params.items():
            if key in self.neural_influence:
                self.neural_influence[key] = value
        
        logger.info(f"Updated neural influence parameters: {self.neural_influence}")
    
    def get_neural_influence_params(self) -> Dict:
        """Get the current neural influence parameters"""
        return self.neural_influence.copy()
    
    def visualize_breathing(self, duration_seconds: float = 30.0) -> Dict:
        """
        Create a visualization of breathing data
        
        Args:
            duration_seconds: Amount of history to visualize
            
        Returns:
            Dict with visualization data
        """
        # Get breath history for the specified duration
        history = self.get_breath_history(duration_seconds)
        
        # Extract timestamps and amplitudes
        timestamps = [entry["timestamp"] for entry in history]
        amplitudes = [entry["amplitude"] for entry in history]
        states = [entry["state"] for entry in history]
        
        # Normalize timestamps to start from 0
        if timestamps:
            start_time = timestamps[0]
            timestamps = [t - start_time for t in timestamps]
        
        # Calculate key metrics
        coherence = self.calculate_breath_coherence(duration_seconds)
        mean_amplitude = sum(amplitudes) / len(amplitudes) if amplitudes else 0
        max_amplitude = max(amplitudes) if amplitudes else 0
        
        # Create visualization data
        return {
            "type": "breathing_visualization",
            "timestamps": timestamps,
            "amplitudes": amplitudes,
            "states": states,
            "pattern": self.current_pattern.value,
            "coherence": coherence,
            "mean_amplitude": mean_amplitude,
            "max_amplitude": max_amplitude,
            "breath_rate": self.breath_rate
        }
        
    def __del__(self):
        """Cleanup when the object is deleted"""
        self.stop_simulation()

# Example usage
if __name__ == "__main__":
    # Create breathing system
    breathing = BreathingSystem(simulation_rate=20.0)
    
    # Start simulation
    breathing.start_simulation()
    
    try:
        # Print some stats for each breath pattern
        patterns = list(BreathingPattern)
        
        for pattern in patterns:
            print(f"\nTesting pattern: {pattern.value}")
            breathing.set_breathing_pattern(pattern)
            
            # Let the pattern run for a few seconds
            pattern_duration = breathing._get_current_cycle_duration() * 2
            time.sleep(pattern_duration)
            
            # Get and print current state
            state = breathing.get_current_breath_state()
            coherence = breathing.calculate_breath_coherence(5.0)
            
            print(f"  Rate: {state['rate']:.1f} breaths/min")
            print(f"  Amplitude: {state['amplitude']:.2f}")
            print(f"  Coherence: {coherence:.2f}")
            print(f"  Current state: {state['state']}")
    
    finally:
        # Stop simulation
        breathing.stop_simulation()
        print("\nBreathing simulation stopped") 