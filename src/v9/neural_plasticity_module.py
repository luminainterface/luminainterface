#!/usr/bin/env python3
"""
Neural Plasticity Module (v11)

An enhanced implementation of neuroplasticity for the Lumina Neural Network system
with advanced biologically-inspired learning mechanisms. This module extends the v9
neuroplasticity system with more sophisticated Hebbian learning, synaptic weight decay,
and homeostatic scaling mechanisms.

Key features:
- Enhanced Hebbian learning with spike-timing-dependent plasticity
- Temporal sensitivity for sequence learning
- Automated synaptic weight decay for unused connections
- Competitive learning among neural populations
- Self-organizing neural maps
- Homeostatic scaling with excitation-inhibition balancing
- Activity normalization across neural regions
- Integration with breathing patterns for enhanced learning

This module implements the v11 roadmap capabilities while maintaining
backward compatibility with the v9 neuroplasticity system.
"""

import logging
import random
import time
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from enum import Enum, auto
from datetime import datetime

# Import existing neuroplasticity system for backward compatibility
from v9.neuroplasticity import Neuroplasticity as BaseNeuroplasticity
from v9.neuroplasticity import PlasticityMode as BasePlasticityMode

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v11.neural_plasticity")

class PlasticityMode(BasePlasticityMode):
    """Extended modes for neuroplasticity operation"""
    COMPETITIVE = auto()      # Competitive learning
    TEMPORAL = auto()         # Temporal sequence learning
    SELF_ORGANIZING = auto()  # Self-organizing map formation

class NeuralPlasticityModule:
    """
    Enhanced Neural Plasticity Module for dynamic neural network adaptation
    
    This class extends the v9 Neuroplasticity system with more sophisticated
    biologically-inspired learning mechanisms, including enhanced Hebbian learning,
    synaptic weight decay, and homeostatic regulation.
    """
    
    def __init__(self, 
                 plasticity_strength: float = 0.15, 
                 default_mode: PlasticityMode = PlasticityMode.HEBBIAN,
                 decay_rate: float = 0.05,
                 competitive_strength: float = 0.2,
                 temporal_window: int = 5,
                 integration_level: float = 1.0):
        """
        Initialize the enhanced neural plasticity module
        
        Args:
            plasticity_strength: Base strength for connection modifications (0.0-1.0)
            default_mode: Default plasticity mode
            decay_rate: Rate at which unused connections decay
            competitive_strength: Strength of competitive inhibition
            temporal_window: Window size for temporal sequence learning
            integration_level: Level of integration with breathing system (0.0-1.0)
        """
        # Initialize base neuroplasticity system for compatibility
        self.base_system = BaseNeuroplasticity(plasticity_strength, default_mode)
        
        # Enhanced parameters
        self.plasticity_strength = plasticity_strength
        self.current_mode = default_mode
        self.decay_rate = decay_rate
        self.competitive_strength = competitive_strength
        self.temporal_window = temporal_window
        self.integration_level = integration_level
        
        # Activity tracking with temporal information
        self.activity_history = {}  # neuron_id -> list of timestamps and activation values
        self.temporal_patterns = {} # sequence -> count
        
        # Connection tracking with timestamps
        self.connection_usage = {}  # (source, target) -> last_use_time
        self.connection_age = {}    # (source, target) -> creation_time
        
        # Neuron grouping for competitive learning
        self.neuron_groups = {}     # group_id -> set of neuron_ids
        
        # Homeostatic parameters
        self.target_activity = 0.1  # Target activity level (fraction of time active)
        self.activity_scaling = {}  # neuron_id -> scaling factor
        
        # Enhanced statistics
        self.stats = {
            "connections_strengthened": 0,
            "connections_weakened": 0,
            "connections_created": 0,
            "connections_pruned": 0,
            "hebbian_updates": 0,
            "stdp_updates": 0,
            "temporal_sequences_learned": 0,
            "competitive_adjustments": 0,
            "homeostatic_scaling_events": 0,
            "weight_decay_events": 0
        }
        
        logger.info(f"Neural Plasticity Module initialized with {default_mode.name} mode")
    
    def process_network(self, neural_network, breath_state=None, timestep=None):
        """
        Apply enhanced neural plasticity to a neural network
        
        Args:
            neural_network: Neural network to modify (must have neurons and connections)
            breath_state: Optional current breathing state
            timestep: Current timestep for temporal tracking
            
        Returns:
            Dict with neural plasticity statistics
        """
        if not hasattr(neural_network, 'neurons') or not hasattr(neural_network, 'connections'):
            logger.error("Invalid neural network provided")
            return self.stats
        
        # Use current time if no timestep provided
        if timestep is None:
            timestep = time.time()
        
        # Update activity history with temporal information
        self._update_activity_history(neural_network, timestep)
        
        # Apply hebbian learning (always active as base mechanism)
        self._apply_enhanced_hebbian(neural_network, breath_state)
        
        # Apply synaptic weight decay
        self._apply_weight_decay(neural_network, timestep)
        
        # Apply specific plasticity mechanisms based on mode
        if self.current_mode == PlasticityMode.STDP:
            self._apply_stdp(neural_network, timestep)
        elif self.current_mode == PlasticityMode.TEMPORAL:
            self._apply_temporal_learning(neural_network, timestep)
        elif self.current_mode == PlasticityMode.COMPETITIVE:
            self._apply_competitive_learning(neural_network)
        elif self.current_mode == PlasticityMode.SELF_ORGANIZING:
            self._apply_self_organizing(neural_network)
        elif self.current_mode == PlasticityMode.HOMEOSTATIC:
            self._apply_enhanced_homeostatic(neural_network, breath_state)
        elif self.current_mode == PlasticityMode.BREATH_ENHANCED:
            self._apply_enhanced_breath_plasticity(neural_network, breath_state, timestep)
        
        # Return updated stats
        return self.stats
    
    def _update_activity_history(self, neural_network, timestep):
        """Update activity history with temporal information"""
        # Get currently active neurons
        active_neurons = [n_id for n_id, neuron in neural_network.neurons.items() 
                         if isinstance(neuron, dict) and neuron.get("state") == "active"]
        
        # Update activation history for all neurons with timestamp
        for neuron_id in neural_network.neurons:
            if neuron_id not in self.activity_history:
                self.activity_history[neuron_id] = []
            
            # Add new activation with timestamp (1 for active, 0 for inactive)
            is_active = 1 if neuron_id in active_neurons else 0
            self.activity_history[neuron_id].append((timestep, is_active))
            
            # Keep history limited to reasonable size
            while len(self.activity_history[neuron_id]) > 100:
                self.activity_history[neuron_id].pop(0)
        
        # Update temporal patterns
        if len(active_neurons) > 0:
            # Convert active pattern to a hashable sequence
            pattern = tuple(sorted(active_neurons))
            if pattern not in self.temporal_patterns:
                self.temporal_patterns[pattern] = 0
            self.temporal_patterns[pattern] += 1
    
    def _apply_enhanced_hebbian(self, neural_network, breath_state=None):
        """Apply enhanced Hebbian plasticity with additional factors"""
        # Get breathing influence
        breath_factor = self._get_breath_factor(breath_state, "hebbian")
        
        # Calculate effective strength
        effective_strength = self.plasticity_strength * breath_factor
        
        # Get currently active neurons
        active_neurons = [n_id for n_id, neuron in neural_network.neurons.items() 
                         if isinstance(neuron, dict) and neuron.get("state") == "active"]
        
        # Apply to all connections between active neurons
        connections_modified = 0
        for source_id in active_neurons:
            if source_id not in neural_network.connections:
                continue
                
            for target_id in active_neurons:
                if target_id not in neural_network.connections[source_id]:
                    continue
                
                # Get current weight
                current_weight = neural_network.connections[source_id][target_id]
                
                # Calculate weight increase with diminishing returns for already strong connections
                weight_increase = effective_strength * (1.0 - current_weight/2.0)
                
                # Apply increase with ceiling
                new_weight = min(2.0, current_weight + weight_increase)
                neural_network.connections[source_id][target_id] = new_weight
                
                # Update connection usage timestamp
                self.connection_usage[(source_id, target_id)] = time.time()
                
                # Record connection age if new
                if (source_id, target_id) not in self.connection_age:
                    self.connection_age[(source_id, target_id)] = time.time()
                
                if new_weight > current_weight:
                    self.stats["connections_strengthened"] += 1
                    self.stats["hebbian_updates"] += 1
                    connections_modified += 1
        
        logger.debug(f"Enhanced Hebbian: modified {connections_modified} connections")
        return connections_modified
    
    def _apply_weight_decay(self, neural_network, current_time):
        """Apply synaptic weight decay to unused connections"""
        # Set minimum time before decay starts (10 seconds)
        min_unused_time = 10
        connections_decayed = 0
        
        for source_id, connections in neural_network.connections.items():
            for target_id, weight in list(connections.items()):
                # Get last usage time
                last_use = self.connection_usage.get((source_id, target_id), 0)
                
                # Skip recently used connections
                if current_time - last_use < min_unused_time:
                    continue
                
                # Calculate decay based on time since last use
                time_factor = min(1.0, (current_time - last_use) / 100.0)  # Max effect after 100 seconds
                decay_amount = self.decay_rate * time_factor
                
                # Apply decay
                new_weight = max(0.01, weight - decay_amount)
                
                # Update connection weight
                neural_network.connections[source_id][target_id] = new_weight
                
                # Track statistics
                if new_weight < weight:
                    self.stats["connections_weakened"] += 1
                    self.stats["weight_decay_events"] += 1
                    connections_decayed += 1
                
                # Prune if below threshold
                if new_weight <= 0.05:
                    del neural_network.connections[source_id][target_id]
                    self.stats["connections_pruned"] += 1
        
        logger.debug(f"Weight decay: modified {connections_decayed} connections")
        return connections_decayed
    
    def _apply_stdp(self, neural_network, current_time):
        """Apply spike-timing-dependent plasticity"""
        # STDP window in seconds
        stdp_window = 1.0  # 1 second
        stdp_updates = 0
        
        # Get recent activity for all neurons
        recent_activity = {}
        for neuron_id, history in self.activity_history.items():
            # Get activations within the STDP window
            recent = [(t, a) for t, a in history if current_time - t <= stdp_window]
            if recent:
                recent_activity[neuron_id] = recent
        
        # Process all connections
        for source_id, connections in neural_network.connections.items():
            if source_id not in recent_activity:
                continue
                
            source_activations = [(t, a) for t, a in recent_activity[source_id] if a > 0]
            if not source_activations:
                continue
                
            for target_id in connections:
                if target_id not in recent_activity:
                    continue
                    
                target_activations = [(t, a) for t, a in recent_activity[target_id] if a > 0]
                if not target_activations:
                    continue
                
                # Calculate STDP effect based on relative timing
                stdp_effect = 0
                for src_time, _ in source_activations:
                    for tgt_time, _ in target_activations:
                        # Calculate time difference (negative if source before target)
                        time_diff = tgt_time - src_time
                        
                        # Skip if too far apart
                        if abs(time_diff) > stdp_window:
                            continue
                        
                        # STDP curve: strengthening if source before target, weakening otherwise
                        if time_diff > 0:  # Target fires after source (strengthen)
                            effect = math.exp(-time_diff / (stdp_window/3)) * 0.1
                            stdp_effect += effect
                        else:  # Target fires before source (weaken)
                            effect = math.exp(time_diff / (stdp_window/3)) * 0.05
                            stdp_effect -= effect
                
                # Apply STDP effect
                if stdp_effect != 0:
                    current_weight = neural_network.connections[source_id][target_id]
                    new_weight = max(0.01, min(2.0, current_weight + stdp_effect))
                    neural_network.connections[source_id][target_id] = new_weight
                    
                    # Update stats
                    if new_weight > current_weight:
                        self.stats["connections_strengthened"] += 1
                    elif new_weight < current_weight:
                        self.stats["connections_weakened"] += 1
                    self.stats["stdp_updates"] += 1
                    stdp_updates += 1
        
        logger.debug(f"STDP: modified {stdp_updates} connections")
        return stdp_updates
    
    def _apply_temporal_learning(self, neural_network, current_time):
        """Learn temporal sequences of neural activations"""
        # Get recent activity sequences
        sequences_learned = 0
        
        # Find neurons that activated in sequence
        sequence_window = self.temporal_window  # Number of timesteps to consider
        
        # Group recent activations by time
        timestep_activations = {}
        for neuron_id, history in self.activity_history.items():
            for t, active in history:
                if active and current_time - t <= sequence_window:
                    if t not in timestep_activations:
                        timestep_activations[t] = []
                    timestep_activations[t].append(neuron_id)
        
        # Sort timesteps
        sorted_times = sorted(timestep_activations.keys())
        
        # Skip if not enough timesteps
        if len(sorted_times) < 2:
            return 0
        
        # Process sequential activations
        for i in range(len(sorted_times) - 1):
            current_time = sorted_times[i]
            next_time = sorted_times[i + 1]
            
            # Skip if too far apart
            if next_time - current_time > 1.0:  # 1 second max gap
                continue
                
            # Get neurons active at these times
            current_neurons = timestep_activations[current_time]
            next_neurons = timestep_activations[next_time]
            
            # Strengthen connections from current to next
            for source_id in current_neurons:
                if source_id not in neural_network.connections:
                    neural_network.connections[source_id] = {}
                    
                for target_id in next_neurons:
                    # Skip self-connections
                    if source_id == target_id:
                        continue
                    
                    # Create connection if it doesn't exist
                    if target_id not in neural_network.connections[source_id]:
                        neural_network.connections[source_id][target_id] = 0.3
                        self.stats["connections_created"] += 1
                        self.connection_age[(source_id, target_id)] = current_time
                    
                    # Strengthen connection
                    current_weight = neural_network.connections[source_id][target_id]
                    new_weight = min(2.0, current_weight + 0.1)
                    neural_network.connections[source_id][target_id] = new_weight
                    
                    # Update usage
                    self.connection_usage[(source_id, target_id)] = current_time
                    
                    if new_weight > current_weight:
                        self.stats["connections_strengthened"] += 1
                        sequences_learned += 1
        
        if sequences_learned > 0:
            self.stats["temporal_sequences_learned"] += sequences_learned
            logger.debug(f"Temporal learning: strengthened {sequences_learned} sequential connections")
        
        return sequences_learned
    
    def _apply_competitive_learning(self, neural_network):
        """Apply competitive learning within neural groups"""
        # Initialize neuron groups if not done
        if not self.neuron_groups:
            self._initialize_neuron_groups(neural_network)
        
        connections_modified = 0
        
        # Process each group
        for group_id, neurons in self.neuron_groups.items():
            # Find most active neuron in group
            most_active = None
            highest_activity = -1
            
            for neuron_id in neurons:
                if neuron_id not in self.activity_history:
                    continue
                
                # Calculate recent activity
                recent_activity = sum(a for _, a in self.activity_history[neuron_id][-10:])
                
                if recent_activity > highest_activity:
                    highest_activity = recent_activity
                    most_active = neuron_id
            
            # Skip if no active neurons or just one neuron
            if most_active is None or len(neurons) <= 1:
                continue
            
            # Strengthen connections to most active neuron
            for source_id in neural_network.connections:
                if most_active in neural_network.connections[source_id]:
                    current_weight = neural_network.connections[source_id][most_active]
                    new_weight = min(2.0, current_weight + self.competitive_strength * 0.1)
                    neural_network.connections[source_id][most_active] = new_weight
                    
                    if new_weight > current_weight:
                        self.stats["connections_strengthened"] += 1
                        connections_modified += 1
            
            # Weaken connections to other neurons in group
            for neuron_id in neurons:
                if neuron_id == most_active:
                    continue
                
                for source_id in neural_network.connections:
                    if neuron_id in neural_network.connections[source_id]:
                        current_weight = neural_network.connections[source_id][neuron_id]
                        new_weight = max(0.1, current_weight - self.competitive_strength * 0.05)
                        neural_network.connections[source_id][neuron_id] = new_weight
                        
                        if new_weight < current_weight:
                            self.stats["connections_weakened"] += 1
                            connections_modified += 1
        
        if connections_modified > 0:
            self.stats["competitive_adjustments"] += connections_modified
            logger.debug(f"Competitive learning: modified {connections_modified} connections")
        
        return connections_modified
    
    def _initialize_neuron_groups(self, neural_network):
        """Initialize competitive neuron groups based on connectivity"""
        # Simple grouping based on connectivity patterns
        # In a more advanced implementation, this could use community detection algorithms
        
        # Reset groups
        self.neuron_groups = {}
        
        # Group size
        group_size = 5
        
        # Assign neurons to groups
        neuron_ids = list(neural_network.neurons.keys())
        random.shuffle(neuron_ids)
        
        for i, neuron_id in enumerate(neuron_ids):
            group_id = i // group_size
            if group_id not in self.neuron_groups:
                self.neuron_groups[group_id] = set()
            self.neuron_groups[group_id].add(neuron_id)
        
        logger.info(f"Initialized {len(self.neuron_groups)} competitive neuron groups")
    
    def _apply_enhanced_homeostatic(self, neural_network, breath_state=None):
        """Apply enhanced homeostatic plasticity with excitation-inhibition balance"""
        # Get breath influence
        breath_factor = self._get_breath_factor(breath_state, "homeostatic")
        
        # Calculate effective strength
        effective_strength = self.plasticity_strength * breath_factor
        
        # Calculate average activity for all neurons over time window
        avg_activity = {}
        for neuron_id, history in self.activity_history.items():
            # Use recent history (last 20 activations)
            recent = history[-20:]
            if recent:
                avg_activity[neuron_id] = sum(a for _, a in recent) / len(recent)
        
        # Calculate network-wide average activity
        network_avg = sum(avg_activity.values()) / max(1, len(avg_activity))
        
        # Initialize or update activity scaling factors
        for neuron_id in neural_network.neurons:
            if neuron_id not in self.activity_scaling:
                self.activity_scaling[neuron_id] = 1.0
        
        # Update activity scaling factors based on target activity
        for neuron_id, activity in avg_activity.items():
            if activity > self.target_activity * 1.5:
                # Too active - decrease scaling
                self.activity_scaling[neuron_id] *= (1.0 - effective_strength * 0.1)
            elif activity < self.target_activity * 0.5:
                # Too inactive - increase scaling
                self.activity_scaling[neuron_id] *= (1.0 + effective_strength * 0.1)
            
            # Keep scaling factor in reasonable range
            self.activity_scaling[neuron_id] = max(0.5, min(2.0, self.activity_scaling[neuron_id]))
        
        # Apply homeostatic adjustments based on scaling factors
        connections_modified = 0
        for source_id, connections in neural_network.connections.items():
            for target_id, weight in list(connections.items()):
                # Skip if target doesn't have scaling factor
                if target_id not in self.activity_scaling:
                    continue
                
                # Get target's activity scaling
                scaling = self.activity_scaling[target_id]
                
                # Apply scaling to weight
                if scaling != 1.0:
                    # If scaling < 1.0, reduce weight (neuron too active)
                    # If scaling > 1.0, increase weight (neuron too inactive)
                    adjustment = (scaling - 1.0) * effective_strength
                    new_weight = max(0.01, min(2.0, weight + adjustment))
                    
                    # Apply if changed
                    if new_weight != weight:
                        neural_network.connections[source_id][target_id] = new_weight
                        
                        if new_weight > weight:
                            self.stats["connections_strengthened"] += 1
                        else:
                            self.stats["connections_weakened"] += 1
                            
                        self.stats["homeostatic_scaling_events"] += 1
                        connections_modified += 1
        
        logger.debug(f"Enhanced homeostatic: modified {connections_modified} connections")
        return connections_modified
    
    def _apply_self_organizing(self, neural_network):
        """Apply self-organizing map principles to neural network"""
        # This is a simplified implementation of self-organizing maps
        # A full implementation would include spatial relationships and neighborhood functions
        
        # Identify neural groups if not done
        if not self.neuron_groups:
            self._initialize_neuron_groups(neural_network)
        
        connections_modified = 0
        
        # For each input pattern, find the best matching unit (BMU)
        # and strengthen its connections
        
        # Get active pattern
        active_neurons = [n_id for n_id, neuron in neural_network.neurons.items() 
                         if isinstance(neuron, dict) and neuron.get("state") == "active"]
        
        if not active_neurons:
            return 0
        
        # Find best matching unit for this pattern
        best_match = None
        best_match_score = -1
        
        for neuron_id in neural_network.neurons:
            match_score = 0
            
            # Calculate how well this neuron's connections match the active pattern
            if neuron_id in neural_network.connections:
                for target_id, weight in neural_network.connections[neuron_id].items():
                    if target_id in active_neurons:
                        match_score += weight
            
            if match_score > best_match_score:
                best_match_score = match_score
                best_match = neuron_id
        
        if not best_match:
            return 0
        
        # Strengthen connections from BMU to active pattern
        for target_id in active_neurons:
            if best_match not in neural_network.connections:
                neural_network.connections[best_match] = {}
                
            if target_id not in neural_network.connections[best_match]:
                neural_network.connections[best_match][target_id] = 0.3
                self.stats["connections_created"] += 1
            else:
                current_weight = neural_network.connections[best_match][target_id]
                new_weight = min(2.0, current_weight + 0.1)
                neural_network.connections[best_match][target_id] = new_weight
                
                if new_weight > current_weight:
                    self.stats["connections_strengthened"] += 1
                    connections_modified += 1
        
        logger.debug(f"Self-organizing map: modified {connections_modified} connections")
        return connections_modified
    
    def _apply_enhanced_breath_plasticity(self, neural_network, breath_state, timestep):
        """Apply breathing-enhanced plasticity with all mechanisms"""
        if not breath_state:
            return self._apply_enhanced_hebbian(neural_network, None)
        
        mechanisms_applied = 0
        
        # Apply mechanisms based on breath state
        if breath_state["state"] == "inhale":
            # During inhale, apply hebbian and STDP
            if self._apply_enhanced_hebbian(neural_network, breath_state) > 0:
                mechanisms_applied += 1
            if self._apply_stdp(neural_network, timestep) > 0:
                mechanisms_applied += 1
                
        elif breath_state["state"] == "exhale":
            # During exhale, apply homeostatic and weight decay
            if self._apply_enhanced_homeostatic(neural_network, breath_state) > 0:
                mechanisms_applied += 1
            if self._apply_weight_decay(neural_network, timestep) > 0:
                mechanisms_applied += 1
                
        elif breath_state["state"] == "hold" and breath_state.get("amplitude", 0) > 0.7:
            # During high-amplitude hold, apply temporal learning
            if self._apply_temporal_learning(neural_network, timestep) > 0:
                mechanisms_applied += 1
                
        elif breath_state.get("pattern") == "meditative":
            # During meditative breathing, apply self-organizing
            if self._apply_self_organizing(neural_network) > 0:
                mechanisms_applied += 1
        
        # Apply competitive learning if other mechanisms were successful
        if mechanisms_applied > 0:
            self._apply_competitive_learning(neural_network)
        
        return mechanisms_applied
    
    def _get_breath_factor(self, breath_state, mechanism_type):
        """Calculate breath influence factor for different mechanism types"""
        if not breath_state or self.integration_level <= 0:
            return 1.0
            
        # Default factor
        factor = 1.0
        
        # Apply breath state influences
        if "state" in breath_state:
            if mechanism_type == "hebbian":
                # Hebbian is enhanced during inhale
                if breath_state["state"] == "inhale":
                    factor = 1.0 + (0.5 * self.integration_level)
                elif breath_state["state"] == "exhale":
                    factor = 1.0 - (0.2 * self.integration_level)
            elif mechanism_type == "homeostatic":
                # Homeostatic is enhanced during exhale
                if breath_state["state"] == "exhale":
                    factor = 1.0 + (0.5 * self.integration_level)
                elif breath_state["state"] == "inhale":
                    factor = 1.0 - (0.2 * self.integration_level)
        
        # Apply amplitude influence
        if "amplitude" in breath_state:
            # Higher amplitude increases effect
            amplitude_factor = 1.0 + ((breath_state["amplitude"] - 0.5) * self.integration_level)
            factor *= max(0.5, min(1.5, amplitude_factor))
        
        # Apply coherence influence
        if "coherence" in breath_state:
            # Higher coherence increases effect
            coherence_factor = 1.0 + ((breath_state["coherence"] - 0.5) * self.integration_level)
            factor *= max(0.5, min(1.5, coherence_factor))
        
        return factor
    
    def create_new_connections(self, neural_network, breath_state=None):
        """Create new connections based on activity patterns"""
        # Forward to base implementation for compatibility
        return self.base_system._create_new_connections(neural_network, breath_state)
    
    def prune_connections(self, neural_network, prune_threshold=0.1):
        """Prune weak connections below threshold"""
        connections_pruned = 0
        
        for source_id, connections in list(neural_network.connections.items()):
            for target_id, weight in list(connections.items()):
                if weight < prune_threshold:
                    del neural_network.connections[source_id][target_id]
                    self.stats["connections_pruned"] += 1
                    connections_pruned += 1
        
        logger.debug(f"Pruned {connections_pruned} weak connections")
        return connections_pruned
    
    def get_stats(self):
        """Get detailed statistics about neural plasticity operations"""
        return self.stats
    
    def set_mode(self, mode):
        """Set the current plasticity mode"""
        if isinstance(mode, PlasticityMode):
            self.current_mode = mode
            logger.info(f"Neural plasticity mode set to {mode.name}")
            return True
        return False


# For direct module testing
if __name__ == "__main__":
    # Simple test network
    class TestNetwork:
        def __init__(self):
            self.neurons = {f"n{i}": {"state": "inactive"} for i in range(10)}
            self.connections = {}
            
            # Add some connections
            for i in range(9):
                self.connections[f"n{i}"] = {f"n{i+1}": 0.5}
    
    network = TestNetwork()
    plasticity = NeuralPlasticityModule()
    
    # Activate some neurons
    network.neurons["n1"]["state"] = "active"
    network.neurons["n2"]["state"] = "active"
    
    # Process with plasticity
    plasticity.process_network(network)
    
    # Print results
    print("Neural Plasticity Module Test:")
    print(f"Stats: {plasticity.get_stats()}")
    print(f"Connections: {network.connections}") 