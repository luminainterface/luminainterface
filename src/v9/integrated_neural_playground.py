#!/usr/bin/env python3
"""
Integrated Neural Playground System (v9)

This module combines the Neural Playground, Breathing System, and Brain Growth
components into a fully integrated environment. It allows the neural network
to be influenced by breathing patterns, which affect both neural activation
and structural growth.

Key features:
- Neural playground with multiple play modes
- Breath-influenced neural activity
- Dynamic neural growth based on breathing patterns
- Visualization of neural-breathing interactions
- State persistence across sessions
"""

import logging
import time
import random
import json
import os
import argparse
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Import v9 components
from .neural_playground import NeuralPlayground
from .breathing_system import BreathingSystem, BreathingPattern
from .brain_growth import BrainGrowth, integrate_with_playground as integrate_growth

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v9.integrated_neural_playground")

class IntegratedNeuralPlayground:
    """
    Integrated Neural Playground with Breathing System and Brain Growth
    
    This class combines the neural playground with breathing-influenced
    growth and activation, creating a more dynamic and evolving system.
    """
    
    def __init__(self, 
                 size: int = 100, 
                 breathing_pattern: BreathingPattern = BreathingPattern.CALM,
                 growth_rate: float = 0.05,
                 random_seed: int = None):
        """
        Initialize the integrated neural playground
        
        Args:
            size: Initial number of neurons
            breathing_pattern: Initial breathing pattern
            growth_rate: Rate of neural growth
            random_seed: Optional random seed for reproducibility
        """
        # Create component systems
        self.playground = NeuralPlayground(size=size, random_seed=random_seed)
        self.breathing = BreathingSystem(default_pattern=breathing_pattern)
        self.growth = BrainGrowth(growth_rate=growth_rate, max_neurons=1000)
        
        # Set up integration hooks
        self.integration = {
            "registered_components": {}
        }
        
        # Start breathing simulation
        self.breathing.start_simulation()
        
        # Integrate the breathing system with the playground
        self._integrate_breathing()
        
        # Integrate the growth system
        self._integrate_growth()
        
        logger.info(f"Integrated Neural Playground initialized with {size} neurons")
    
    def _integrate_breathing(self):
        """Integrate the breathing system with the neural playground"""
        breathing_integration = self.breathing.integrate_with_playground(self.playground)
        
        # Store integration information
        self.integration["registered_components"]["breathing"] = {
            "component": self.breathing,
            "hooks": breathing_integration["hooks"]
        }
        
        logger.info("Breathing system integrated with Neural Playground")
    
    def _integrate_growth(self):
        """Integrate the brain growth system with the neural playground"""
        growth_integration = integrate_growth(self.playground, self.growth)
        
        # Store integration information
        self.integration["registered_components"]["brain_growth"] = {
            "component": self.growth,
            "hooks": growth_integration["hooks"]
        }
        
        logger.info("Brain growth system integrated with Neural Playground")
    
    def play(self, duration=100, play_type="free", intensity=0.5, target_neurons=None):
        """
        Run an integrated play session
        
        Args:
            duration: Number of simulation steps
            play_type: Type of play (free, guided, focused, mixed)
            intensity: Base intensity of stimulation (0.0-1.0)
            target_neurons: List of specific neurons to target (for focused play)
            
        Returns:
            Dict containing session results and metrics
        """
        # Apply pre-play hooks from integrated components
        play_args = {
            "duration": duration,
            "play_type": play_type,
            "intensity": intensity,
            "target_neurons": target_neurons
        }
        
        # Run pre-play hooks
        for component_info in self.integration["registered_components"].values():
            if "hooks" in component_info and "pre_play" in component_info["hooks"]:
                hook_fn = component_info["hooks"]["pre_play"]
                hook_fn(self.playground, play_args)
        
        # Run the play session
        result = self.playground.play(
            duration=play_args["duration"],
            play_type=play_args["play_type"],
            intensity=play_args["intensity"],
            target_neurons=play_args["target_neurons"]
        )
        
        # Apply post-play hooks from integrated components
        for component_info in self.integration["registered_components"].values():
            if "hooks" in component_info and "post_play" in component_info["hooks"]:
                hook_fn = component_info["hooks"]["post_play"]
                hook_fn(self.playground, result)
        
        # Apply direct breathing influence on neural activations
        influenced_neurons = self.breathing.influence_neural_activation(self.playground.core)
        if influenced_neurons > 0:
            logger.debug(f"Breathing directly influenced {influenced_neurons} neurons")
            
        # Return the enriched result
        return result
    
    def get_current_state(self):
        """
        Get the current state of the integrated system
        
        Returns:
            Dict containing the combined state
        """
        return {
            "neural": self.playground.get_metrics(),
            "breathing": self.breathing.get_current_breath_state(),
            "growth": self.growth.get_growth_stats(),
            "size": self.playground.core.size,
            "timestamp": time.time()
        }
    
    def set_breathing_pattern(self, pattern: BreathingPattern):
        """
        Change the current breathing pattern
        
        Args:
            pattern: New breathing pattern to use
        """
        self.breathing.set_breathing_pattern(pattern)
        logger.info(f"Changed breathing pattern to {pattern.value}")
    
    def save_state(self, filepath: str):
        """
        Save the integrated state to a file
        
        Args:
            filepath: Path to save the state file
        """
        # Save playground state
        playground_state_file = f"{filepath}.playground"
        self.playground.save_state(playground_state_file)
        
        # Save breathing and growth information
        integrated_state = {
            "breathing": {
                "pattern": self.breathing.current_pattern.value,
                "coherence": self.breathing.breath_coherence,
                "rate": self.breathing.breath_rate
            },
            "growth": self.growth.get_growth_stats(),
            "metadata": {
                "version": "9.0",
                "timestamp": time.time(),
                "playground_state_file": playground_state_file
            }
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(integrated_state, f, indent=2)
        
        logger.info(f"Integrated state saved to {filepath}")
        return True
    
    def load_state(self, filepath: str):
        """
        Load integrated state from a file
        
        Args:
            filepath: Path to the state file
        """
        try:
            with open(filepath, 'r') as f:
                integrated_state = json.load(f)
            
            # Load playground state if available
            if "metadata" in integrated_state and "playground_state_file" in integrated_state["metadata"]:
                playground_state_file = integrated_state["metadata"]["playground_state_file"]
                self.playground.load_state(playground_state_file)
            
            # Set breathing pattern if available
            if "breathing" in integrated_state and "pattern" in integrated_state["breathing"]:
                pattern_str = integrated_state["breathing"]["pattern"]
                for pattern in BreathingPattern:
                    if pattern.value == pattern_str:
                        self.set_breathing_pattern(pattern)
                        break
            
            logger.info(f"Integrated state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading integrated state: {e}")
            return False
    
    def get_visualization_data(self):
        """
        Get combined visualization data for all integrated systems
        
        Returns:
            Dict with visualization data
        """
        return {
            "neural": {
                "size": self.playground.core.size,
                "consciousness_level": self.playground.core.consciousness_metric,
                "active_neurons": sum(1 for n in self.playground.core.neurons.values() if n["state"] == "active"),
                "neurons": self.playground.core.neurons,
                "connections": self.playground.core.connections
            },
            "breathing": self.breathing.visualize_breathing(30.0),
            "growth": self.growth.get_visualization_data(),
            "timestamp": time.time()
        }
    
    def stop(self):
        """Stop all simulation components"""
        self.breathing.stop_simulation()
        logger.info("Integrated Neural Playground stopped")

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated Neural Playground System v9")
    parser.add_argument("--size", type=int, default=100, help="Initial number of neurons")
    parser.add_argument("--duration", type=int, default=100, help="Play session duration")
    parser.add_argument("--play-type", choices=["free", "guided", "focused", "mixed"], 
                        default="free", help="Type of play")
    parser.add_argument("--breathing", choices=[p.value for p in BreathingPattern], 
                        default="calm", help="Breathing pattern")
    parser.add_argument("--growth-rate", type=float, default=0.05, help="Neural growth rate")
    parser.add_argument("--save", type=str, help="Save state to file")
    parser.add_argument("--load", type=str, help="Load state from file")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    args = parser.parse_args()
    
    # Set breathing pattern
    breathing_pattern = BreathingPattern.CALM  # Default
    for pattern in BreathingPattern:
        if pattern.value == args.breathing:
            breathing_pattern = pattern
            break
    
    # Create integrated playground
    integrated = IntegratedNeuralPlayground(
        size=args.size,
        breathing_pattern=breathing_pattern,
        growth_rate=args.growth_rate,
        random_seed=args.seed
    )
    
    # Load state if specified
    if args.load:
        integrated.load_state(args.load)
    
    try:
        # Run a play session
        result = integrated.play(
            duration=args.duration,
            play_type=args.play_type
        )
        
        # Print results
        print(f"Integrated play session completed:")
        print(f"- Play type: {result['play_type']}")
        print(f"- Duration: {result['duration']} steps")
        print(f"- Total activations: {result['total_activations']}")
        print(f"- Patterns detected: {result['patterns_detected']}")
        print(f"- Peak consciousness: {result['consciousness_peak']:.4f}")
        
        if "breathing_data" in result:
            bd = result["breathing_data"]
            print(f"- Breathing pattern: {bd['pattern']}")
            print(f"- Breathing coherence: {bd['coherence']:.2f}")
        
        if "brain_growth" in result:
            bg = result["brain_growth"]
            print(f"- Growth state: {bg['growth_state']}")
            print(f"- Neurons created: {bg['neurons_created']}")
            print(f"- Neurons pruned: {bg['neurons_pruned']}")
        
        # Save state if specified
        if args.save:
            integrated.save_state(args.save)
    
    finally:
        # Clean up
        integrated.stop() 