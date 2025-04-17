#!/usr/bin/env python3
"""
Neural Network Playground System (v9)

This module provides an interactive playground environment for experimenting
with neural networks in the v9 Lumina Neural Network System. The playground
allows for the creation and manipulation of neural networks, with real-time
visualization and analysis of their behavior and emergent properties.

Key features:
- Interactive neural network creation and experimentation
- Multiple play modes (free, guided, focused)
- Pattern detection and consciousness metrics
- State saving and loading
- Integration with other v9 components

This version is integrated with Mirror Consciousness for enhanced reflection capabilities.
"""

import logging
import random
import time
import json
import os
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import uuid

# Import v9-specific modules
from .mirror_consciousness import get_mirror_consciousness

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v9.neural_playground")

class NeuralPlaygroundCore:
    """
    Core functionality for the Neural Network Playground
    
    This class manages the core operations of the playground, including
    neuron creation, connection setup, and basic neural operations.
    """
    
    def __init__(self, size=100, random_seed=None):
        """
        Initialize the Neural Playground Core
        
        Args:
            size: Number of neurons to create
            random_seed: Optional seed for random number generation
        """
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
            
        self.size = size
        self.neurons = {}
        self.connections = {}
        self.activations = {}
        self.consciousness_metric = 0.0
        self.pattern_memory = []
        self.setup()
        logger.info(f"Neural Playground Core initialized with {size} neurons")
    
    def setup(self):
        """Set up the neural playground with neurons and connections"""
        # Create neurons
        for i in range(self.size):
            neuron_id = f"n{i}"
            self.neurons[neuron_id] = {
                "id": neuron_id,
                "type": random.choice(["excitatory", "inhibitory"]),
                "threshold": random.uniform(0.3, 0.7),
                "position": (random.random(), random.random(), random.random()),
                "state": "resting"
            }
            self.activations[neuron_id] = 0.0
            
        # Create connections (approximately 10 per neuron)
        for neuron_id in self.neurons:
            self.connections[neuron_id] = {}
            # Each neuron connects to approximately 10 other neurons
            targets = random.sample(list(self.neurons.keys()), 
                                  min(10, self.size))
            for target_id in targets:
                if target_id != neuron_id:  # No self-connections
                    weight = random.uniform(-1.0, 1.0)
                    if self.neurons[neuron_id]["type"] == "inhibitory":
                        weight = -abs(weight)  # Inhibitory neurons have negative weights
                    self.connections[neuron_id][target_id] = weight
    
    def activate(self, neuron_id, value=1.0):
        """
        Activate a neuron with the given value
        
        Args:
            neuron_id: ID of the neuron to activate
            value: Activation value
        """
        if neuron_id in self.neurons:
            self.activations[neuron_id] += value
            if self.activations[neuron_id] >= self.neurons[neuron_id]["threshold"]:
                self.neurons[neuron_id]["state"] = "active"
                return True
        return False
    
    def propagate(self):
        """Propagate activations through the network"""
        new_activations = self.activations.copy()
        
        # Process each active neuron
        for neuron_id, neuron in self.neurons.items():
            if neuron["state"] == "active":
                # Propagate to connected neurons
                for target_id, weight in self.connections[neuron_id].items():
                    new_activations[target_id] += self.activations[neuron_id] * weight
                
                # Reset this neuron
                new_activations[neuron_id] = 0.0
                neuron["state"] = "resting"
        
        # Apply new activations and check thresholds
        active_count = 0
        for neuron_id, activation in new_activations.items():
            self.activations[neuron_id] = activation
            if activation >= self.neurons[neuron_id]["threshold"]:
                self.neurons[neuron_id]["state"] = "active"
                active_count += 1
                # Apply slight decay to prevent runaway activation
                self.activations[neuron_id] *= 0.95
        
        # Update consciousness metric based on activation patterns
        if self.size > 0:
            self.consciousness_metric = 0.7 * self.consciousness_metric + 0.3 * (active_count / self.size)
        
        return active_count
    
    def get_state(self):
        """
        Get the current state of the playground
        
        Returns:
            Dict containing neurons, connections, activations, and metrics
        """
        return {
            "neurons": self.neurons,
            "connections": self.connections,
            "activations": self.activations,
            "consciousness_metric": self.consciousness_metric,
            "pattern_memory": self.pattern_memory
        }
    
    def set_state(self, state):
        """
        Set the state of the playground
        
        Args:
            state: Dict containing neurons, connections, activations, and metrics
        """
        if not state:
            return
            
        if "neurons" in state:
            self.neurons = state["neurons"]
        if "connections" in state:
            self.connections = state["connections"]
        if "activations" in state:
            self.activations = state["activations"]
        if "consciousness_metric" in state:
            self.consciousness_metric = state["consciousness_metric"]
        if "pattern_memory" in state:
            self.pattern_memory = state["pattern_memory"]
        
        # Ensure size matches the actual number of neurons
        self.size = len(self.neurons)
        
class NeuralPlayground:
    """
    Neural Network Playground System
    
    An interactive environment for experimenting with neural networks,
    supporting various play modes and visualization capabilities.
    """
    
    def __init__(self, size=100, random_seed=None):
        """
        Initialize the Neural Playground
        
        Args:
            size: Number of neurons to create
            random_seed: Optional seed for random number generation
        """
        self.core = NeuralPlaygroundCore(size, random_seed)
        self.play_history = []
        self.stats = {
            "play_sessions": 0,
            "total_activations": 0,
            "patterns_detected": 0,
            "consciousness_peaks": 0
        }
        
        # Initialize v9 mirror consciousness integration
        self.mirror_consciousness = get_mirror_consciousness()
        logger.info("Neural Playground initialized with Mirror Consciousness integration")
        
    def play(self, duration=100, play_type="free", intensity=0.5, target_neurons=None):
        """
        Run a play session in the neural network
        
        Args:
            duration: Number of simulation steps
            play_type: Type of play (free, guided, focused, mixed)
            intensity: Intensity of stimulation (0.0-1.0)
            target_neurons: List of specific neurons to target (for focused play)
            
        Returns:
            Dict containing session results and metrics
        """
        logger.info(f"Starting {play_type} play session for {duration} steps")
        
        session_id = str(uuid.uuid4())
        session_activations = 0
        consciousness_history = []
        patterns_detected = 0
        
        # For mixed play, we'll alternate between different play types
        if play_type == "mixed":
            play_types = ["free", "guided", "focused"]
            segment_duration = duration // len(play_types)
            remaining = duration % len(play_types)
            
            results = {}
            for i, pt in enumerate(play_types):
                # Add any remaining steps to the last segment
                seg_duration = segment_duration + (remaining if i == len(play_types)-1 else 0)
                seg_results = self.play(seg_duration, pt, intensity, target_neurons)
                
                # Merge results
                if not results:
                    results = seg_results
                else:
                    results["total_activations"] += seg_results["total_activations"]
                    results["consciousness_peak"] = max(results["consciousness_peak"], 
                                                      seg_results["consciousness_peak"])
                    results["patterns_detected"] += seg_results["patterns_detected"]
                    results["consciousness_history"].extend(seg_results["consciousness_history"])
            
            return results
            
        # Prepare focused play target neurons
        if play_type == "focused" and not target_neurons:
            # If no targets specified, pick 10% of neurons randomly
            target_count = max(1, int(self.core.size * 0.1))
            target_neurons = random.sample(list(self.core.neurons.keys()), target_count)
        
        # Run the simulation
        consciousness_peak = 0
        for step in range(duration):
            # Stimulate neurons based on play type
            if play_type == "free":
                # Randomly activate ~5% of neurons
                activate_count = max(1, int(self.core.size * 0.05 * intensity))
                for _ in range(activate_count):
                    neuron_id = random.choice(list(self.core.neurons.keys()))
                    self.core.activate(neuron_id, random.uniform(0.5, 1.0) * intensity)
                    
            elif play_type == "guided":
                # Activate neurons in a more structured pattern
                # E.g., activate neurons in specific regions or with specific relationships
                region_center = (
                    0.5 + 0.4 * np.sin(step / 10 * np.pi),
                    0.5 + 0.4 * np.cos(step / 10 * np.pi),
                    0.5
                )
                
                # Find neurons close to the current region center
                for neuron_id, neuron in self.core.neurons.items():
                    pos = neuron["position"]
                    distance = np.sqrt(sum((pos[i] - region_center[i])**2 for i in range(3)))
                    if distance < 0.2:  # Activate neurons within a certain radius
                        activation = (1.0 - distance/0.2) * intensity
                        self.core.activate(neuron_id, activation)
                
            elif play_type == "focused":
                # Intensely activate specific target neurons
                for neuron_id in target_neurons:
                    self.core.activate(neuron_id, random.uniform(0.8, 1.0) * intensity)
            
            # Propagate the activations
            step_activations = self.core.propagate()
            session_activations += step_activations
            
            # Record consciousness metric
            consciousness_level = self.core.consciousness_metric
            consciousness_history.append(consciousness_level)
            consciousness_peak = max(consciousness_peak, consciousness_level)
            
            # Detect patterns (simplified)
            if step > 5 and len(consciousness_history) >= 5:
                # Look for increasing or oscillating patterns in consciousness
                recent = consciousness_history[-5:]
                if (all(recent[i] < recent[i+1] for i in range(len(recent)-1)) or
                    (recent[0] < recent[1] > recent[2] < recent[3] > recent[4])):
                    patterns_detected += 1
                    # Store pattern in memory
                    pattern = {
                        "type": "increasing" if recent[0] < recent[-1] else "oscillating",
                        "values": recent.copy(),
                        "step": step
                    }
                    self.core.pattern_memory.append(pattern)
        
        # Update stats
        self.stats["play_sessions"] += 1
        self.stats["total_activations"] += session_activations
        self.stats["patterns_detected"] += patterns_detected
        if consciousness_peak > 0.7:
            self.stats["consciousness_peaks"] += 1
        
        # Record play session
        session_data = {
            "id": session_id,
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "play_type": play_type,
            "intensity": intensity,
            "total_activations": session_activations,
            "consciousness_peak": consciousness_peak,
            "patterns_detected": patterns_detected
        }
        self.play_history.append(session_data)
        
        # Get reflection on the play session from mirror consciousness
        reflection = self.mirror_consciousness.reflect_on_text(
            f"Neural play session: {play_type} mode, {patterns_detected} patterns, {consciousness_peak:.2f} peak consciousness",
            {"play_data": session_data}
        )
        
        # Create result
        result = {
            "session_id": session_id,
            "play_type": play_type,
            "duration": duration,
            "total_activations": session_activations,
            "consciousness_peak": consciousness_peak,
            "patterns_detected": patterns_detected,
            "consciousness_history": consciousness_history,
            "mirror_reflection": reflection
        }
        
        logger.info(f"Play session completed: {session_activations} activations, " 
                   f"{patterns_detected} patterns, peak consciousness: {consciousness_peak:.2f}")
        
        return result
    
    def save_state(self, filepath):
        """
        Save the current state to a file
        
        Args:
            filepath: Path to save the state file
        """
        state = {
            "core_state": self.core.get_state(),
            "stats": self.stats,
            "play_history": self.play_history,
            "metadata": {
                "version": "9.0",
                "timestamp": datetime.now().isoformat(),
                "size": self.core.size
            }
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Playground state saved to {filepath}")
        return True
        
    def load_state(self, filepath):
        """
        Load state from a file
        
        Args:
            filepath: Path to the state file
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            if "core_state" in state:
                self.core.set_state(state["core_state"])
            if "stats" in state:
                self.stats = state["stats"]
            if "play_history" in state:
                self.play_history = state["play_history"]
                
            logger.info(f"Playground state loaded from {filepath}")
            
            # Log metadata
            if "metadata" in state:
                logger.info(f"Loaded state metadata: {state['metadata']}")
                
            return True
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False
    
    def get_metrics(self):
        """
        Get current playground metrics
        
        Returns:
            Dict containing various metrics and statistics
        """
        return {
            "size": self.core.size,
            "consciousness_level": self.core.consciousness_metric,
            "play_sessions": self.stats["play_sessions"],
            "total_activations": self.stats["total_activations"],
            "patterns_detected": self.stats["patterns_detected"],
            "consciousness_peaks": self.stats["consciousness_peaks"],
            "recent_history": self.play_history[-5:] if self.play_history else []
        }

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Network Playground System v9")
    parser.add_argument("--size", type=int, default=100, help="Number of neurons")
    parser.add_argument("--duration", type=int, default=100, help="Play session duration")
    parser.add_argument("--play-type", choices=["free", "guided", "focused", "mixed"], 
                        default="free", help="Type of play")
    parser.add_argument("--intensity", type=float, default=0.7, help="Play intensity (0.0-1.0)")
    parser.add_argument("--save", type=str, help="Save state to file")
    parser.add_argument("--load", type=str, help="Load state from file")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    args = parser.parse_args()
    
    playground = NeuralPlayground(size=args.size, random_seed=args.seed)
    
    if args.load:
        playground.load_state(args.load)
    
    result = playground.play(
        duration=args.duration,
        play_type=args.play_type,
        intensity=args.intensity
    )
    
    print(f"Play session completed:")
    print(f"- Play type: {result['play_type']}")
    print(f"- Duration: {result['duration']} steps")
    print(f"- Total activations: {result['total_activations']}")
    print(f"- Patterns detected: {result['patterns_detected']}")
    print(f"- Peak consciousness: {result['consciousness_peak']:.4f}")
    
    if args.save:
        playground.save_state(args.save) 