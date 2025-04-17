#!/usr/bin/env python3
"""
Neuroplasticity Module (v9)

This module provides a neuroplasticity system for the Neural Playground
that enables dynamic adjustment of neural connections based on activity
patterns and breathing state. It implements mechanisms for synaptic
strengthening/weakening, new connection formation, neural pruning,
and memory consolidation.

Key features:
- Breath-influenced synaptic strength changes
- Adaptive connection formation based on activation patterns
- Automated neural pruning for inactive connections
- Memory consolidation during meditative breathing states
- Hebbian and homeostatic plasticity mechanisms
"""

import logging
import random
import time
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum, auto

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v9.neuroplasticity")

class PlasticityMode(Enum):
    """Different modes for neuroplasticity operation"""
    HEBBIAN = auto()          # Strengthen connections between co-active neurons
    HOMEOSTATIC = auto()      # Balance activity across the network
    STDP = auto()             # Spike-timing-dependent plasticity
    BREATH_ENHANCED = auto()  # Breath-synchronized plasticity
    CONSOLIDATION = auto()    # Memory consolidation (strengthening important patterns)
    PRUNING = auto()          # Remove weak connections

class Neuroplasticity:
    """
    Neuroplasticity system for dynamic neural connections
    
    This class provides mechanisms for neural networks to adapt their
    structure based on activity patterns and breathing state.
    """
    
    def __init__(self, 
                plasticity_strength: float = 0.1, 
                default_mode: PlasticityMode = PlasticityMode.HEBBIAN):
        """
        Initialize the neuroplasticity system
        
        Args:
            plasticity_strength: How strongly connections are modified (0.0-1.0)
            default_mode: Default plasticity mode
        """
        self.plasticity_strength = plasticity_strength
        self.current_mode = default_mode
        
        # Activity history tracking
        self.activity_history = {}
        self.co_activation_counts = {}
        
        # Connection history tracking
        self.connection_usage = {}
        self.connection_strength_history = {}
        
        # Integration parameters
        self.breath_influence = {
            "hebbian_multiplier": 1.5,    # Strengthen Hebbian during inhale
            "homeostatic_multiplier": 1.2, # Strengthen homeostatic during exhale
            "pruning_threshold": 0.2,     # Connection strength threshold for pruning
            "consolidation_threshold": 0.7 # Breath coherence threshold for consolidation
        }
        
        # Statistics
        self.stats = {
            "connections_strengthened": 0,
            "connections_weakened": 0,
            "connections_created": 0,
            "connections_pruned": 0,
            "consolidation_events": 0
        }
        
        logger.info(f"Neuroplasticity system initialized with {default_mode.name} mode")
    
    def process_network(self, neural_network, breath_state=None):
        """
        Apply neuroplasticity to a neural network
        
        Args:
            neural_network: Neural network to modify (must have neurons and connections)
            breath_state: Optional current breathing state
            
        Returns:
            Dict with neuroplasticity statistics
        """
        if not hasattr(neural_network, 'neurons') or not hasattr(neural_network, 'connections'):
            logger.error("Invalid neural network provided")
            return self.stats
        
        # Update activity history
        self._update_activity_history(neural_network)
        
        # Apply different plasticity mechanisms based on mode
        if self.current_mode == PlasticityMode.HEBBIAN:
            self._apply_hebbian_plasticity(neural_network, breath_state)
        elif self.current_mode == PlasticityMode.HOMEOSTATIC:
            self._apply_homeostatic_plasticity(neural_network, breath_state)
        elif self.current_mode == PlasticityMode.BREATH_ENHANCED:
            self._apply_breath_enhanced_plasticity(neural_network, breath_state)
        elif self.current_mode == PlasticityMode.CONSOLIDATION:
            self._apply_consolidation(neural_network, breath_state)
        elif self.current_mode == PlasticityMode.PRUNING:
            self._apply_pruning(neural_network)
        
        # Return updated stats
        return self.stats
    
    def _update_activity_history(self, neural_network):
        """Update activity history for all neurons"""
        # Get currently active neurons
        active_neurons = [n_id for n_id, neuron in neural_network.neurons.items() 
                        if isinstance(neuron, dict) and neuron.get("state") == "active"]
        
        # Update activation history for all neurons
        for neuron_id in neural_network.neurons:
            if neuron_id not in self.activity_history:
                self.activity_history[neuron_id] = [0] * 10  # Keep last 10 activations
            
            # Shift history and add new activation (1 for active, 0 for inactive)
            self.activity_history[neuron_id].pop(0)
            self.activity_history[neuron_id].append(1 if neuron_id in active_neurons else 0)
        
        # Update co-activation counts for active neurons
        for i, n1 in enumerate(active_neurons):
            for j, n2 in enumerate(active_neurons[i+1:], i+1):
                pair_key = tuple(sorted([n1, n2]))
                if pair_key not in self.co_activation_counts:
                    self.co_activation_counts[pair_key] = 0
                self.co_activation_counts[pair_key] += 1
    
    def _apply_hebbian_plasticity(self, neural_network, breath_state=None):
        """Apply Hebbian plasticity - 'neurons that fire together, wire together'"""
        # Get hebbian multiplier from breathing state
        hebbian_multiplier = 1.0
        if breath_state and "state" in breath_state:
            if breath_state["state"] == "inhale":
                hebbian_multiplier = self.breath_influence["hebbian_multiplier"]
            elif breath_state["state"] == "exhale":
                hebbian_multiplier = 0.8
        
        # Calculate effective strength
        effective_strength = self.plasticity_strength * hebbian_multiplier
        
        # Strengthen connections between co-active neurons
        connections_modified = 0
        for neuron_id, connections in neural_network.connections.items():
            # Skip if neuron has been inactive
            if neuron_id not in self.activity_history or sum(self.activity_history[neuron_id][-3:]) == 0:
                continue
                
            for target_id in connections:
                # Skip if target has been inactive
                if target_id not in self.activity_history or sum(self.activity_history[target_id][-3:]) == 0:
                    continue
                
                # Get co-activation count
                pair_key = tuple(sorted([neuron_id, target_id]))
                co_activation = self.co_activation_counts.get(pair_key, 0)
                
                # Strengthen connection based on co-activation
                if co_activation > 0:
                    # Get current weight
                    current_weight = neural_network.connections[neuron_id][target_id]
                    
                    # Calculate weight increase based on co-activation and breathing
                    weight_increase = effective_strength * min(co_activation * 0.1, 0.5)
                    
                    # Apply increase with ceiling at 2.0
                    new_weight = min(2.0, current_weight + weight_increase)
                    neural_network.connections[neuron_id][target_id] = new_weight
                    
                    if new_weight > current_weight:
                        self.stats["connections_strengthened"] += 1
                        connections_modified += 1
        
        logger.debug(f"Hebbian plasticity: modified {connections_modified} connections")
        return connections_modified
    
    def _apply_homeostatic_plasticity(self, neural_network, breath_state=None):
        """Apply homeostatic plasticity - balance activity across network"""
        # Get homeostatic multiplier from breathing state
        homeostatic_multiplier = 1.0
        if breath_state and "state" in breath_state:
            if breath_state["state"] == "exhale":
                homeostatic_multiplier = self.breath_influence["homeostatic_multiplier"]
        
        # Calculate effective strength
        effective_strength = self.plasticity_strength * homeostatic_multiplier
        
        # Calculate average activity for all neurons
        avg_activity = {}
        for neuron_id in neural_network.neurons:
            if neuron_id in self.activity_history:
                avg_activity[neuron_id] = sum(self.activity_history[neuron_id]) / len(self.activity_history[neuron_id])
        
        # Calculate network-wide average activity
        network_avg = sum(avg_activity.values()) / max(1, len(avg_activity))
        
        # Modify connections to balance activity
        connections_modified = 0
        for neuron_id, connections in neural_network.connections.items():
            if neuron_id not in avg_activity:
                continue
                
            # If this neuron is more active than average, weaken its outgoing connections
            if avg_activity[neuron_id] > network_avg * 1.5:
                for target_id in connections:
                    current_weight = neural_network.connections[neuron_id][target_id]
                    new_weight = max(-2.0, current_weight - effective_strength * 0.2)
                    neural_network.connections[neuron_id][target_id] = new_weight
                    
                    if new_weight < current_weight:
                        self.stats["connections_weakened"] += 1
                        connections_modified += 1
            
            # If this neuron is less active than average, strengthen its incoming connections
            elif avg_activity[neuron_id] < network_avg * 0.5:
                for source_id, source_conns in neural_network.connections.items():
                    if neuron_id in source_conns:
                        current_weight = neural_network.connections[source_id][neuron_id]
                        new_weight = min(2.0, current_weight + effective_strength * 0.2)
                        neural_network.connections[source_id][neuron_id] = new_weight
                        
                        if new_weight > current_weight:
                            self.stats["connections_strengthened"] += 1
                            connections_modified += 1
        
        logger.debug(f"Homeostatic plasticity: modified {connections_modified} connections")
        return connections_modified
    
    def _apply_breath_enhanced_plasticity(self, neural_network, breath_state=None):
        """Apply specialized breathing-enhanced plasticity"""
        if not breath_state:
            # Fall back to Hebbian if no breath state
            return self._apply_hebbian_plasticity(neural_network)
        
        # Choose plasticity mode based on breath state
        if breath_state["state"] == "inhale":
            # During inhale, emphasize Hebbian (strengthening)
            return self._apply_hebbian_plasticity(neural_network, breath_state)
        elif breath_state["state"] == "exhale":
            # During exhale, emphasize homeostatic (balancing)
            return self._apply_homeostatic_plasticity(neural_network, breath_state)
        elif breath_state["state"] == "hold" and breath_state["amplitude"] > 0.7:
            # During high-amplitude hold, try to create new connections
            return self._create_new_connections(neural_network, breath_state)
        elif breath_state["pattern"] == "meditative":
            # During meditative breathing, consolidate memory
            return self._apply_consolidation(neural_network, breath_state)
        else:
            # Default to Hebbian
            return self._apply_hebbian_plasticity(neural_network, breath_state)
    
    def _create_new_connections(self, neural_network, breath_state=None):
        """Create new connections between neurons that aren't connected yet"""
        # Get list of neurons and their activations
        neuron_ids = list(neural_network.neurons.keys())
        if not neuron_ids or len(neuron_ids) < 2:
            return 0
        
        # Find neurons with high recent activity
        active_neurons = []
        for neuron_id in neuron_ids:
            if neuron_id in self.activity_history and sum(self.activity_history[neuron_id][-5:]) >= 3:
                active_neurons.append(neuron_id)
        
        if len(active_neurons) < 2:
            return 0
        
        # Calculate how many connections to create
        # More during inhale, fewer during exhale
        connection_factor = 0.05  # Default 5% of possible new connections
        if breath_state and "state" in breath_state:
            if breath_state["state"] == "inhale":
                connection_factor = 0.08
            elif breath_state["state"] == "exhale":
                connection_factor = 0.03
        
        max_new_connections = int(len(active_neurons) * (len(active_neurons) - 1) / 2 * connection_factor)
        max_new_connections = min(max_new_connections, 10)  # Cap at 10 new connections
        
        # Create new connections
        connections_created = 0
        attempts = 0
        
        while connections_created < max_new_connections and attempts < max_new_connections * 3:
            attempts += 1
            
            # Select two random active neurons
            source_id = random.choice(active_neurons)
            target_id = random.choice(active_neurons)
            
            # Skip self-connections
            if source_id == target_id:
                continue
            
            # Check if connection already exists
            if (source_id in neural_network.connections and 
                target_id in neural_network.connections[source_id]):
                continue
            
            # Create new connection
            if source_id not in neural_network.connections:
                neural_network.connections[source_id] = {}
            
            # Initialize with a moderate weight
            initial_weight = 0.3 + random.random() * 0.2
            neural_network.connections[source_id][target_id] = initial_weight
            
            # Update stats
            self.stats["connections_created"] += 1
            connections_created += 1
        
        logger.debug(f"Created {connections_created} new connections")
        return connections_created
    
    def _apply_pruning(self, neural_network):
        """Prune weak connections that aren't used frequently"""
        pruning_threshold = self.breath_influence["pruning_threshold"]
        connections_pruned = 0
        
        for source_id, connections in list(neural_network.connections.items()):
            for target_id in list(connections.keys()):
                weight = neural_network.connections[source_id][target_id]
                
                # Check if connection is weak
                if abs(weight) < pruning_threshold:
                    # Check if both neurons have been inactive recently
                    source_inactive = (source_id in self.activity_history and 
                                      sum(self.activity_history[source_id][-10:]) <= 1)
                    target_inactive = (target_id in self.activity_history and 
                                      sum(self.activity_history[target_id][-10:]) <= 1)
                    
                    # Prune if both neurons are inactive or weight is very low
                    if (source_inactive and target_inactive) or abs(weight) < pruning_threshold / 2:
                        del neural_network.connections[source_id][target_id]
                        connections_pruned += 1
                        self.stats["connections_pruned"] += 1
        
        logger.debug(f"Pruned {connections_pruned} weak connections")
        return connections_pruned
    
    def _apply_consolidation(self, neural_network, breath_state=None):
        """Consolidate important connections during meditative states"""
        # Only consolidate during coherent, meditative breathing
        if not breath_state or breath_state.get("pattern") != "meditative":
            return 0
            
        coherence = breath_state.get("coherence", 0)
        if coherence < self.breath_influence["consolidation_threshold"]:
            return 0
            
        # Find patterns (clusters of neurons that are frequently co-active)
        patterns = self._identify_patterns(neural_network)
        if not patterns:
            return 0
            
        # Strengthen connections within patterns
        consolidation_multiplier = min(1.5, 1.0 + coherence * 0.5)
        connections_consolidated = 0
        
        for pattern in patterns:
            # Strengthen internal connections of the pattern
            for i, n1 in enumerate(pattern):
                for n2 in pattern[i+1:]:
                    # Check if connection exists in either direction
                    if (n1 in neural_network.connections and 
                        n2 in neural_network.connections[n1]):
                        # Strengthen connection
                        current_weight = neural_network.connections[n1][n2]
                        neural_network.connections[n1][n2] = min(2.0, current_weight * consolidation_multiplier)
                        connections_consolidated += 1
                    
                    if (n2 in neural_network.connections and 
                        n1 in neural_network.connections[n2]):
                        # Strengthen connection
                        current_weight = neural_network.connections[n2][n1]
                        neural_network.connections[n2][n1] = min(2.0, current_weight * consolidation_multiplier)
                        connections_consolidated += 1
        
        # Update stats
        self.stats["consolidation_events"] += 1
        self.stats["connections_strengthened"] += connections_consolidated
        
        logger.debug(f"Consolidated {connections_consolidated} connections in {len(patterns)} patterns")
        return connections_consolidated
    
    def _identify_patterns(self, neural_network):
        """Identify patterns (neuron clusters) based on co-activation history"""
        # Sort co-activation counts from highest to lowest
        sorted_co_activations = sorted(
            self.co_activation_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Take top 20% of co-activations
        top_count = max(5, len(sorted_co_activations) // 5)
        top_co_activations = sorted_co_activations[:top_count]
        
        # Build patterns from co-activated neuron pairs
        patterns = []
        used_neurons = set()
        
        for (n1, n2), count in top_co_activations:
            # Skip if count is too low
            if count < 3:
                continue
                
            # Find if either neuron is already in a pattern
            existing_pattern = None
            for pattern in patterns:
                if n1 in pattern or n2 in pattern:
                    existing_pattern = pattern
                    break
            
            if existing_pattern:
                # Add to existing pattern if not already in it
                if n1 not in existing_pattern:
                    existing_pattern.append(n1)
                if n2 not in existing_pattern:
                    existing_pattern.append(n2)
            else:
                # Create new pattern
                patterns.append([n1, n2])
            
            used_neurons.add(n1)
            used_neurons.add(n2)
        
        return patterns
    
    def set_mode(self, mode: PlasticityMode):
        """Set the plasticity mode"""
        self.current_mode = mode
        logger.info(f"Set plasticity mode to {mode.name}")
    
    def set_breath_influence(self, params: Dict):
        """Update breath influence parameters"""
        for key, value in params.items():
            if key in self.breath_influence:
                self.breath_influence[key] = value
        
        logger.info(f"Updated breath influence parameters: {self.breath_influence}")
    
    def reset_stats(self):
        """Reset neuroplasticity statistics"""
        for key in self.stats:
            self.stats[key] = 0
    
    def get_stats(self):
        """Get neuroplasticity statistics"""
        return self.stats.copy()

def integrate_with_playground(playground, neuroplasticity=None):
    """
    Integrate neuroplasticity system with a neural playground
    
    Args:
        playground: Neural playground instance
        neuroplasticity: Optional existing neuroplasticity instance
        
    Returns:
        Integration info and hooks
    """
    # Create neuroplasticity if not provided
    if not neuroplasticity:
        neuroplasticity = Neuroplasticity()
    
    # Define hooks
    def post_play_hook(playground, play_result):
        """Apply neuroplasticity after play session"""
        try:
            # Get breath state if available
            breath_state = None
            if "breathing_data" in play_result:
                breath_state = play_result["breathing_data"]
            
            # Apply neuroplasticity
            stats = neuroplasticity.process_network(playground.core, breath_state)
            
            # Add neuroplasticity stats to play result
            play_result["neuroplasticity_stats"] = stats
            
        except Exception as e:
            logger.error(f"Error in neuroplasticity post-play hook: {e}")
    
    # Define hooks dictionary
    hooks = {
        "post_play": post_play_hook
    }
    
    # Return integration info
    return {
        "component_type": "neuroplasticity",
        "hooks": hooks,
        "neuroplasticity": neuroplasticity
    }

# Example usage
if __name__ == "__main__":
    from .neural_playground import NeuralPlayground
    
    # Create playground and neuroplasticity
    playground = NeuralPlayground(size=100)
    neuroplasticity = Neuroplasticity(plasticity_strength=0.2)
    
    # Run a play session
    result = playground.play(duration=100, play_type="mixed", intensity=0.7)
    
    # Apply neuroplasticity
    stats = neuroplasticity.process_network(playground.core)
    
    print(f"Neuroplasticity results:")
    print(f"- Connections strengthened: {stats['connections_strengthened']}")
    print(f"- Connections weakened: {stats['connections_weakened']}")
    print(f"- Connections created: {stats['connections_created']}")
    print(f"- Connections pruned: {stats['connections_pruned']}") 