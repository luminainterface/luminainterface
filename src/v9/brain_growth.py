#!/usr/bin/env python3
"""
Brain Growth Module (v9)

This module provides mechanisms for neural network growth influenced by breathing patterns.
It serves as a bridge between the Breathing System and Neural Playground, enabling
the creation of new neurons and neural structures based on breathing coherence and patterns.

Key features:
- Breathing-influenced neuron generation
- Dynamic growth of neural regions based on breathing coherence
- Neural pathway development during specific breathing states
- Structural evolution of the neural network over time
- Neural pruning guided by breathing patterns
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
logger = logging.getLogger("v9.brain_growth")

class GrowthState(Enum):
    """Different states for the brain growth system"""
    IDLE = auto()          # No growth occurring
    EXPANSION = auto()     # Creating new neurons
    ORGANIZATION = auto()  # Organizing neural structures
    CONSOLIDATION = auto() # Strengthening existing structures
    PRUNING = auto()       # Removing unnecessary structures

class BrainGrowth:
    """
    Brain Growth system for neural network structural development
    
    This class bridges breathing patterns with neural growth mechanisms,
    enabling the neural playground to expand and evolve its structure
    based on breathing coherence and patterns.
    """
    
    def __init__(self, 
                 growth_rate: float = 0.05,
                 max_neurons: int = 500):
        """
        Initialize the brain growth system
        
        Args:
            growth_rate: Base rate of neuron creation (0.0-1.0)
            max_neurons: Maximum number of neurons that can be created
        """
        self.growth_rate = growth_rate
        self.max_neurons = max_neurons
        self.current_state = GrowthState.IDLE
        
        # Track growth metrics
        self.neurons_created = 0
        self.neurons_pruned = 0
        self.regions_formed = 0
        self.growth_cycles = 0
        
        # Neural region tracking
        self.regions = {}  # Neural regions that have formed
        self.region_neurons = {}  # Mapping of neurons to regions
        
        # Growth parameters influenced by breathing
        self.breath_influence = {
            "creation_multiplier": 1.5,    # Multiplier for neuron creation during deep inhale
            "pruning_threshold": 0.3,      # Activity threshold below which neurons can be pruned
            "organization_coherence": 0.6,  # Breath coherence needed for neural organization
            "consolidation_duration": 3.0   # Time (in seconds) of consistent breathing for consolidation
        }
        
        # Session tracking
        self.last_growth_time = time.time()
        self.consistent_breathing_duration = 0.0
        self.previous_breath_pattern = None
        
        # Statistics
        self.stats = {
            "neurons_created_total": 0,
            "neurons_pruned_total": 0,
            "regions_formed_total": 0,
            "growth_cycles_total": 0,
            "neuron_growth_by_pattern": {
                "calm": 0,
                "focused": 0,
                "meditative": 0,
                "excited": 0
            }
        }
        
        logger.info(f"Brain Growth system initialized with growth rate {growth_rate}")
    
    def process_cycle(self, neural_network, breath_state=None, neuroplasticity=None):
        """
        Process a growth cycle based on current breathing state
        
        Args:
            neural_network: Neural network to modify (playground core)
            breath_state: Current breathing state
            neuroplasticity: Optional neuroplasticity system for coordination
            
        Returns:
            Dict with growth statistics
        """
        if not hasattr(neural_network, 'neurons') or not hasattr(neural_network, 'connections'):
            logger.error("Invalid neural network provided")
            return self.stats
        
        # Initialize metrics for this cycle
        self.neurons_created = 0
        self.neurons_pruned = 0
        self.regions_formed = 0
        self.growth_cycles += 1
        self.stats["growth_cycles_total"] += 1
        
        # Determine growth state based on breathing
        self._update_growth_state(breath_state)
        
        # Apply different growth mechanisms based on state
        if self.current_state == GrowthState.EXPANSION:
            self._expand_network(neural_network, breath_state)
        elif self.current_state == GrowthState.ORGANIZATION:
            self._organize_network(neural_network, breath_state)
        elif self.current_state == GrowthState.CONSOLIDATION:
            self._consolidate_network(neural_network, breath_state, neuroplasticity)
        elif self.current_state == GrowthState.PRUNING:
            self._prune_network(neural_network, breath_state)
        
        # Update statistics
        self.stats["neurons_created_total"] += self.neurons_created
        self.stats["neurons_pruned_total"] += self.neurons_pruned
        self.stats["regions_formed_total"] += self.regions_formed
        
        # Update pattern-specific stats
        if breath_state and "pattern" in breath_state:
            pattern = breath_state["pattern"].lower()
            if pattern in self.stats["neuron_growth_by_pattern"]:
                self.stats["neuron_growth_by_pattern"][pattern] += self.neurons_created
        
        # Record this processing time
        self.last_growth_time = time.time()
        
        return {
            "growth_state": self.current_state.name,
            "neurons_created": self.neurons_created,
            "neurons_pruned": self.neurons_pruned,
            "regions_formed": self.regions_formed
        }
    
    def _update_growth_state(self, breath_state):
        """Update growth state based on breathing pattern and coherence"""
        if not breath_state:
            self.current_state = GrowthState.IDLE
            self.consistent_breathing_duration = 0.0
            self.previous_breath_pattern = None
            return
        
        # Get current pattern and coherence
        pattern = breath_state.get("pattern", "").lower()
        coherence = breath_state.get("coherence", 0.0)
        amplitude = breath_state.get("amplitude", 0.5)
        
        # Check for consistent breathing
        if self.previous_breath_pattern == pattern:
            self.consistent_breathing_duration += time.time() - self.last_growth_time
        else:
            self.consistent_breathing_duration = 0.0
            self.previous_breath_pattern = pattern
        
        # Determine state based on pattern, coherence, and breath state
        if pattern == "meditative" and coherence > self.breath_influence["organization_coherence"]:
            if self.consistent_breathing_duration > self.breath_influence["consolidation_duration"]:
                self.current_state = GrowthState.CONSOLIDATION
            else:
                self.current_state = GrowthState.ORGANIZATION
        
        elif pattern == "focused" and coherence > 0.5:
            self.current_state = GrowthState.ORGANIZATION
        
        elif pattern == "calm" and breath_state.get("state") == "inhale" and amplitude > 0.7:
            self.current_state = GrowthState.EXPANSION
        
        elif pattern == "excited" and breath_state.get("state") == "exhale":
            self.current_state = GrowthState.PRUNING
        
        else:
            # Default based on breath state
            if breath_state.get("state") == "inhale" and amplitude > 0.6:
                self.current_state = GrowthState.EXPANSION
            elif breath_state.get("state") == "exhale" and coherence < 0.4:
                self.current_state = GrowthState.PRUNING
            else:
                self.current_state = GrowthState.IDLE
        
        logger.debug(f"Brain growth state: {self.current_state.name}")
    
    def _expand_network(self, neural_network, breath_state):
        """Create new neurons based on breathing state"""
        # Check if we've reached the maximum number of neurons
        current_neuron_count = len(neural_network.neurons)
        if current_neuron_count >= self.max_neurons:
            logger.debug(f"Maximum neuron count reached ({self.max_neurons})")
            return 0
        
        # Calculate how many neurons to create based on breathing
        growth_multiplier = 1.0
        if breath_state and "amplitude" in breath_state:
            if breath_state["state"] == "inhale":
                growth_multiplier = min(2.0, breath_state["amplitude"] * self.breath_influence["creation_multiplier"])
            elif breath_state["state"] == "hold":
                growth_multiplier = 0.7  # Lower growth during hold
        
        # Calculate base number of neurons to create
        base_count = int(self.growth_rate * current_neuron_count * growth_multiplier)
        create_count = min(base_count, self.max_neurons - current_neuron_count)
        create_count = max(1, create_count)  # At least 1 neuron
        
        # Create new neurons
        for i in range(create_count):
            # Generate neuron ID
            neuron_id = f"n{len(neural_network.neurons)}"
            
            # Determine neuron type based on breathing pattern
            if breath_state and "pattern" in breath_state:
                if breath_state["pattern"].lower() == "focused":
                    neuron_type = "excitatory"  # More excitatory neurons during focused breathing
                elif breath_state["pattern"].lower() == "meditative":
                    neuron_type = random.choice(["excitatory", "inhibitory", "modulatory"])
                else:
                    neuron_type = random.choice(["excitatory", "inhibitory"])
            else:
                neuron_type = random.choice(["excitatory", "inhibitory"])
            
            # Create the neuron
            neural_network.neurons[neuron_id] = {
                "id": neuron_id,
                "type": neuron_type,
                "threshold": random.uniform(0.3, 0.7),
                "position": self._generate_position(neural_network, breath_state),
                "state": "resting",
                "created_by": "brain_growth",
                "breath_pattern": breath_state["pattern"] if breath_state else "unknown",
                "creation_time": time.time()
            }
            
            # Initialize activations
            neural_network.activations[neuron_id] = 0.0
            
            # Create connections to existing neurons
            self._create_initial_connections(neural_network, neuron_id, breath_state)
            
            self.neurons_created += 1
        
        logger.debug(f"Created {create_count} new neurons")
        return create_count
    
    def _generate_position(self, neural_network, breath_state=None):
        """Generate a position for a new neuron based on breathing state"""
        # If we have regions, try to place near an existing region center
        if self.regions and random.random() < 0.7:
            # Select a region
            region_id = random.choice(list(self.regions.keys()))
            region_center = self.regions[region_id]["center"]
            
            # Create near the region with some variability
            variability = 0.2
            if breath_state and "coherence" in breath_state:
                # Lower variability with higher coherence
                variability = max(0.05, 0.3 - breath_state["coherence"] * 0.25)
            
            return (
                region_center[0] + random.uniform(-variability, variability),
                region_center[1] + random.uniform(-variability, variability),
                region_center[2] + random.uniform(-variability, variability)
            )
        
        # Create in a new position with a tendency toward certain areas based on breathing
        if breath_state and "pattern" in breath_state:
            pattern = breath_state["pattern"].lower()
            
            if pattern == "focused":
                # More organized, frontal positioning for focused breathing
                return (
                    random.uniform(0.6, 0.9),
                    random.uniform(0.4, 0.7),
                    random.uniform(0.4, 0.8)
                )
            elif pattern == "meditative":
                # Deeper central positioning for meditative breathing
                return (
                    random.uniform(0.3, 0.7),
                    random.uniform(0.3, 0.7),
                    random.uniform(0.3, 0.7)
                )
            elif pattern == "excited":
                # More dispersed positioning for excited breathing
                return (
                    random.uniform(0.1, 0.9),
                    random.uniform(0.1, 0.9),
                    random.uniform(0.1, 0.9)
                )
        
        # Default random position
        return (
            random.uniform(0.1, 0.9),
            random.uniform(0.1, 0.9),
            random.uniform(0.1, 0.9)
        )
    
    def _create_initial_connections(self, neural_network, neuron_id, breath_state=None):
        """Create initial connections for a new neuron"""
        # Determine number of connections based on breathing pattern
        base_connections = min(10, max(3, len(neural_network.neurons) // 10))
        
        if breath_state and "pattern" in breath_state:
            pattern = breath_state["pattern"].lower()
            
            if pattern == "focused":
                # More targeted connections for focused breathing
                connection_count = max(2, int(base_connections * 0.7))
            elif pattern == "meditative":
                # More extensive connections for meditative breathing
                connection_count = max(5, int(base_connections * 1.3))
            elif pattern == "excited":
                # More varied connections for excited breathing
                connection_count = max(3, int(base_connections * 1.1))
            else:
                connection_count = base_connections
        else:
            connection_count = base_connections
        
        # Get list of existing neurons
        existing_neurons = list(neural_network.neurons.keys())
        existing_neurons.remove(neuron_id)  # Don't connect to self
        
        if not existing_neurons:
            return
        
        # Initialize connections for this neuron
        if neuron_id not in neural_network.connections:
            neural_network.connections[neuron_id] = {}
        
        # Create outgoing connections
        neurons_to_connect = min(connection_count, len(existing_neurons))
        target_neurons = random.sample(existing_neurons, neurons_to_connect)
        
        for target_id in target_neurons:
            # Determine connection weight based on neuron type and breathing
            if neural_network.neurons[neuron_id]["type"] == "excitatory":
                weight = random.uniform(0.3, 0.8)
            elif neural_network.neurons[neuron_id]["type"] == "inhibitory":
                weight = -random.uniform(0.3, 0.8)
            else:  # modulatory
                weight = random.uniform(-0.5, 0.5)
            
            # Adjust weight based on breathing
            if breath_state and "amplitude" in breath_state:
                weight *= (0.8 + breath_state["amplitude"] * 0.4)
            
            # Add connection
            neural_network.connections[neuron_id][target_id] = weight
        
        # Create incoming connections (some existing neurons connect to this one)
        incoming_count = max(1, connection_count // 2)
        source_neurons = random.sample(existing_neurons, min(incoming_count, len(existing_neurons)))
        
        for source_id in source_neurons:
            if source_id not in neural_network.connections:
                neural_network.connections[source_id] = {}
            
            # Determine connection weight
            if neural_network.neurons[source_id]["type"] == "excitatory":
                weight = random.uniform(0.3, 0.8)
            elif neural_network.neurons[source_id]["type"] == "inhibitory":
                weight = -random.uniform(0.3, 0.8)
            else:  # modulatory
                weight = random.uniform(-0.5, 0.5)
            
            # Add connection
            neural_network.connections[source_id][neuron_id] = weight
    
    def _organize_network(self, neural_network, breath_state):
        """Organize neurons into functional regions based on breathing state"""
        if not breath_state or "coherence" not in breath_state:
            return 0
        
        # Only organize with sufficient coherence
        if breath_state["coherence"] < self.breath_influence["organization_coherence"]:
            return 0
        
        # Identify potential regions based on neuron positions
        self._identify_regions(neural_network, breath_state)
        
        # Enhance connections within regions
        regions_enhanced = 0
        for region_id, region in self.regions.items():
            if self._strengthen_region(neural_network, region_id, breath_state):
                regions_enhanced += 1
        
        return regions_enhanced
    
    def _identify_regions(self, neural_network, breath_state):
        """Identify and create neural regions based on spatial clustering"""
        # Skip if too few neurons
        if len(neural_network.neurons) < 10:
            return 0
        
        # Parameters for region identification
        region_radius = 0.2  # Base radius for considering neurons in same region
        if breath_state and "pattern" in breath_state:
            if breath_state["pattern"].lower() == "focused":
                region_radius = 0.15  # Tighter regions for focused breathing
            elif breath_state["pattern"].lower() == "meditative":
                region_radius = 0.25  # Larger regions for meditative breathing
        
        # Get positions of all neurons
        neuron_positions = {}
        for neuron_id, neuron in neural_network.neurons.items():
            if isinstance(neuron, dict) and "position" in neuron:
                neuron_positions[neuron_id] = neuron["position"]
        
        # Find dense areas that could form regions
        potential_centers = []
        for neuron_id, position in neuron_positions.items():
            # Count nearby neurons
            nearby_count = 0
            for other_pos in neuron_positions.values():
                dist = math.sqrt(sum((position[i] - other_pos[i])**2 for i in range(3)))
                if dist < region_radius:
                    nearby_count += 1
            
            # If enough nearby neurons, this could be a region center
            if nearby_count >= 5:  # At least 5 neurons including self
                potential_centers.append((neuron_id, position, nearby_count))
        
        # Sort by density (number of nearby neurons)
        potential_centers.sort(key=lambda x: x[2], reverse=True)
        
        # Create new regions, avoiding overlap with existing regions
        regions_created = 0
        for center_id, center_pos, _ in potential_centers:
            # Skip if already part of a region
            if center_id in self.region_neurons:
                continue
                
            # Skip if too close to existing region
            too_close = False
            for existing_region in self.regions.values():
                existing_center = existing_region["center"]
                dist = math.sqrt(sum((center_pos[i] - existing_center[i])**2 for i in range(3)))
                if dist < region_radius * 1.5:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # Create a new region
            region_id = f"region_{len(self.regions)}"
            self.regions[region_id] = {
                "id": region_id,
                "center": center_pos,
                "radius": region_radius,
                "formation_time": time.time(),
                "breath_pattern": breath_state["pattern"] if breath_state else "unknown",
                "neurons": []
            }
            
            # Assign neurons to this region
            for neuron_id, position in neuron_positions.items():
                if neuron_id in self.region_neurons:
                    continue  # Already in a region
                    
                dist = math.sqrt(sum((position[i] - center_pos[i])**2 for i in range(3)))
                if dist < region_radius:
                    self.regions[region_id]["neurons"].append(neuron_id)
                    self.region_neurons[neuron_id] = region_id
            
            regions_created += 1
            self.regions_formed += 1
            
            # Limit number of regions created per cycle
            if regions_created >= 2:
                break
        
        logger.debug(f"Created {regions_created} neural regions")
        return regions_created
    
    def _strengthen_region(self, neural_network, region_id, breath_state):
        """Strengthen connections within a neural region"""
        if region_id not in self.regions:
            return False
            
        region = self.regions[region_id]
        region_neurons = region["neurons"]
        
        if len(region_neurons) < 3:
            return False
            
        # Determine strengthening factor based on breathing
        strengthen_factor = 0.2  # Base strengthening
        if breath_state and "coherence" in breath_state:
            strengthen_factor *= (1.0 + breath_state["coherence"])
        
        # Strengthen connections between neurons in this region
        connections_strengthened = 0
        
        for i, n1 in enumerate(region_neurons):
            if n1 not in neural_network.connections:
                neural_network.connections[n1] = {}
                
            for n2 in region_neurons[i+1:]:
                # Strengthen existing connection
                if n2 in neural_network.connections[n1]:
                    current_weight = neural_network.connections[n1][n2]
                    # For excitatory connections, strengthen, for inhibitory, weaken slightly
                    if current_weight > 0:
                        neural_network.connections[n1][n2] = min(2.0, current_weight * (1.0 + strengthen_factor))
                    else:
                        neural_network.connections[n1][n2] = max(-2.0, current_weight * (1.0 - strengthen_factor * 0.5))
                    connections_strengthened += 1
                
                # Create new connection if none exists with some probability
                elif random.random() < 0.3:
                    weight = random.uniform(0.2, 0.5)
                    neural_network.connections[n1][n2] = weight
                    connections_strengthened += 1
                
                # Check reverse direction
                if n1 not in neural_network.connections.get(n2, {}):
                    if random.random() < 0.3:
                        if n2 not in neural_network.connections:
                            neural_network.connections[n2] = {}
                        weight = random.uniform(0.2, 0.5)
                        neural_network.connections[n2][n1] = weight
                        connections_strengthened += 1
        
        logger.debug(f"Strengthened {connections_strengthened} connections in region {region_id}")
        return connections_strengthened > 0
    
    def _consolidate_network(self, neural_network, breath_state, neuroplasticity=None):
        """Consolidate neural pathways during meditative states"""
        if not breath_state or breath_state.get("pattern", "").lower() != "meditative":
            return 0
        
        # Identify active neural pathways
        pathways = self._identify_active_pathways(neural_network)
        if not pathways:
            return 0
        
        # Consolidate the pathways
        consolidated_count = 0
        
        for pathway in pathways:
            # Strengthen connections along the pathway
            for i in range(len(pathway)-1):
                source = pathway[i]
                target = pathway[i+1]
                
                # Skip if connection doesn't exist
                if source not in neural_network.connections or target not in neural_network.connections[source]:
                    continue
                
                # Strengthen the connection
                current_weight = neural_network.connections[source][target]
                consolidation_factor = 1.2
                
                if breath_state.get("coherence", 0) > 0.8:
                    consolidation_factor = 1.4  # Stronger consolidation with high coherence
                
                new_weight = min(2.0, current_weight * consolidation_factor)
                neural_network.connections[source][target] = new_weight
                consolidated_count += 1
        
        # If we have neuroplasticity, switch it to consolidation mode
        if neuroplasticity and hasattr(neuroplasticity, 'set_mode'):
            if hasattr(neuroplasticity, 'PlasticityMode') and hasattr(neuroplasticity.PlasticityMode, 'CONSOLIDATION'):
                neuroplasticity.set_mode(neuroplasticity.PlasticityMode.CONSOLIDATION)
        
        logger.debug(f"Consolidated {consolidated_count} connections in {len(pathways)} pathways")
        return consolidated_count
    
    def _identify_active_pathways(self, neural_network):
        """Identify active neural pathways based on activation patterns"""
        # Find neurons that have been active recently
        active_neurons = []
        
        for neuron_id, neuron in neural_network.neurons.items():
            if neuron.get("state") == "active" or neural_network.activations.get(neuron_id, 0) > 0.5:
                active_neurons.append(neuron_id)
        
        if len(active_neurons) < 3:
            return []
        
        # Identify potential pathways (sequences of connected neurons)
        pathways = []
        
        for start_neuron in active_neurons:
            # Start a new pathway
            current_pathway = [start_neuron]
            visited = {start_neuron}
            
            # Try to extend the pathway
            self._extend_pathway(neural_network, current_pathway, visited, active_neurons, 0)
            
            # Add pathway if it's long enough
            if len(current_pathway) >= 3:
                pathways.append(current_pathway)
        
        # Sort by length and return the top 5
        pathways.sort(key=len, reverse=True)
        return pathways[:5]
    
    def _extend_pathway(self, neural_network, pathway, visited, active_neurons, depth):
        """Recursively extend a pathway with connected neurons"""
        if depth >= 5:  # Limit pathway depth
            return
            
        current_neuron = pathway[-1]
        
        # Find connected neurons that are active
        if current_neuron not in neural_network.connections:
            return
            
        connected = []
        for target, weight in neural_network.connections[current_neuron].items():
            if target in active_neurons and target not in visited and weight > 0.3:
                connected.append((target, weight))
        
        # Sort by connection strength
        connected.sort(key=lambda x: x[1], reverse=True)
        
        # Add the strongest connection to the pathway
        if connected:
            next_neuron = connected[0][0]
            pathway.append(next_neuron)
            visited.add(next_neuron)
            
            # Continue extending
            self._extend_pathway(neural_network, pathway, visited, active_neurons, depth + 1)
    
    def _prune_network(self, neural_network, breath_state):
        """Prune inactive neurons based on breathing state"""
        # Skip if network is too small
        if len(neural_network.neurons) < 20:
            return 0
        
        # Determine pruning threshold based on breathing
        pruning_threshold = self.breath_influence["pruning_threshold"]
        
        if breath_state:
            if breath_state.get("pattern") == "excited":
                pruning_threshold *= 0.8  # More aggressive pruning during excited breathing
            elif breath_state.get("pattern") == "meditative":
                pruning_threshold *= 1.5  # Less pruning during meditative breathing
        
        # Find inactive neurons
        inactive_neurons = []
        
        for neuron_id, neuron in neural_network.neurons.items():
            # Skip if neuron is part of a region
            if neuron_id in self.region_neurons:
                continue
                
            # Check activation level
            activation = neural_network.activations.get(neuron_id, 0)
            
            if activation < pruning_threshold:
                # Check connection activity
                incoming_connections = 0
                outgoing_connections = 0
                
                # Check outgoing
                outgoing_connections = len(neural_network.connections.get(neuron_id, {}))
                
                # Check incoming
                for source, targets in neural_network.connections.items():
                    if neuron_id in targets:
                        incoming_connections += 1
                
                # Consider for pruning if poorly connected
                if incoming_connections <= 1 and outgoing_connections <= 1:
                    inactive_neurons.append(neuron_id)
        
        # Limit number of neurons to prune per cycle
        max_prune = min(len(inactive_neurons), max(1, int(len(neural_network.neurons) * 0.05)))
        neurons_to_prune = random.sample(inactive_neurons, max_prune) if len(inactive_neurons) > max_prune else inactive_neurons
        
        # Prune the selected neurons
        for neuron_id in neurons_to_prune:
            # Remove from neurons
            del neural_network.neurons[neuron_id]
            
            # Remove from activations
            if neuron_id in neural_network.activations:
                del neural_network.activations[neuron_id]
            
            # Remove outgoing connections
            if neuron_id in neural_network.connections:
                del neural_network.connections[neuron_id]
            
            # Remove incoming connections
            for source, targets in neural_network.connections.items():
                if neuron_id in targets:
                    del neural_network.connections[source][neuron_id]
            
            self.neurons_pruned += 1
        
        logger.debug(f"Pruned {len(neurons_to_prune)} inactive neurons")
        return len(neurons_to_prune)
    
    def get_growth_stats(self):
        """Get statistics for the growth system"""
        return self.stats.copy()
    
    def get_regions(self):
        """Get information about neural regions"""
        return self.regions.copy()
    
    def get_visualization_data(self):
        """
        Get visualization data for brain growth
        
        Returns:
            Dict with visualization data
        """
        return {
            "type": "brain_growth_visualization",
            "growth_state": self.current_state.name,
            "neurons_created": self.neurons_created,
            "neurons_pruned": self.neurons_pruned,
            "regions": [
                {
                    "id": region["id"],
                    "center": region["center"],
                    "radius": region["radius"],
                    "neuron_count": len(region["neurons"]),
                    "pattern": region["breath_pattern"]
                }
                for region in self.regions.values()
            ],
            "stats": self.stats,
            "consistent_breathing_duration": self.consistent_breathing_duration
        }
    
    def set_breath_influence(self, params: Dict):
        """Update parameters for breath influence on growth"""
        for key, value in params.items():
            if key in self.breath_influence:
                self.breath_influence[key] = value
        
        logger.info(f"Updated breath influence parameters: {self.breath_influence}")

def integrate_with_playground(playground, brain_growth=None):
    """
    Integrate brain growth system with neural playground
    
    Args:
        playground: Neural playground instance
        brain_growth: Optional existing brain growth instance
        
    Returns:
        Integration info and hooks
    """
    # Create brain growth if not provided
    if not brain_growth:
        brain_growth = BrainGrowth()
    
    # Define hooks
    def post_play_hook(playground, play_result):
        """Apply brain growth after play session"""
        try:
            # Get breath state if available
            breath_state = None
            if "breathing_data" in play_result:
                breath_state = play_result["breathing_data"]
            
            # Get neuroplasticity if available
            neuroplasticity = None
            if hasattr(playground, 'integration') and hasattr(playground.integration, 'registered_components'):
                components = playground.integration.registered_components
                if "neuroplasticity" in components:
                    neuroplasticity = components["neuroplasticity"].get("component")
            
            # Process growth cycle
            growth_result = brain_growth.process_cycle(playground.core, breath_state, neuroplasticity)
            
            # Add growth result to play result
            play_result["brain_growth"] = growth_result
            
            # Update neural network size if needed
            if hasattr(playground.core, 'size'):
                playground.core.size = len(playground.core.neurons)
            
        except Exception as e:
            logger.error(f"Error in brain growth post-play hook: {e}")
    
    # Define hooks dictionary
    hooks = {
        "post_play": post_play_hook
    }
    
    # Return integration info
    return {
        "component_type": "brain_growth",
        "hooks": hooks,
        "brain_growth": brain_growth
    }

# Example usage
if __name__ == "__main__":
    from .neural_playground import NeuralPlayground
    from .breathing_system import BreathingSystem
    
    # Create playground
    playground = NeuralPlayground(size=50)
    
    # Create brain growth system
    brain_growth = BrainGrowth(growth_rate=0.1)
    
    # Create breathing system
    breathing = BreathingSystem()
    breathing.start_simulation()
    
    # Wait for breathing to stabilize
    time.sleep(2)
    
    # Get breath state
    breath_state = breathing.get_current_breath_state()
    
    # Run a play session
    playground.play(duration=100, play_type="mixed", intensity=0.7)
    
    # Process growth cycle
    growth_result = brain_growth.process_cycle(playground.core, breath_state)
    
    print(f"Brain Growth results:")
    print(f"- Growth state: {growth_result['growth_state']}")
    print(f"- Neurons created: {growth_result['neurons_created']}")
    print(f"- Neurons pruned: {growth_result['neurons_pruned']}")
    print(f"- Regions formed: {growth_result['regions_formed']}")
    
    # Clean up
    breathing.stop_simulation() 