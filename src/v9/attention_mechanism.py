#!/usr/bin/env python3
"""
Attention Mechanism Module (v12)

An implementation of neural attention mechanisms for the Lumina Neural Network system,
enabling selective focus on relevant information while suppressing distracting inputs.
This module provides both bottom-up (stimulus-driven) and top-down (goal-directed)
attention processing.

Key features:
- Focal attention for directed processing of specific inputs
- Distributed attention for parallel processing of multiple inputs
- Bottom-up (salience-based) attention for detecting novel or important stimuli
- Top-down (goal-directed) attention for task-specific processing
- Attention shifting based on relevance and priority
- Integration with breathing patterns for enhanced focus
- Dynamic allocation of processing resources

This module implements the v12 roadmap capabilities as part of the
advanced cognitive architecture of the Lumina Neural Network system.
"""

import logging
import numpy as np
import time
import math
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v12.attention_mechanism")

class AttentionMode(Enum):
    """Modes for attention mechanism operation"""
    FOCAL = auto()        # Focus on a single region or item
    DISTRIBUTED = auto()  # Distribute across multiple regions or items
    SCANNING = auto()     # Constantly shift between potential targets
    VIGILANT = auto()     # Alert state monitoring for important changes
    RELAXED = auto()      # Low-priority general monitoring

@dataclass
class AttentionTarget:
    """Representation of an attention target"""
    id: str                                   # Unique identifier
    priority: float = 0.5                     # Current priority (0.0-1.0)
    salience: float = 0.5                     # Inherent salience/importance (0.0-1.0)
    position: Optional[Tuple[float, ...]] = None  # Position in feature space
    features: Dict[str, Any] = field(default_factory=dict)  # Features of this target
    last_attended: float = 0.0                # Timestamp of last attendance
    attention_duration: float = 0.0           # Duration of current attention
    habituation: float = 0.0                  # Reduction in priority due to habituation

class AttentionMechanism:
    """
    Neural attention mechanism for selective information processing
    
    This class implements various forms of attention, including focal and distributed
    attention, with bottom-up and top-down processing. It dynamically allocates
    processing resources based on relevance and priority.
    """
    
    def __init__(self, 
                 default_mode: AttentionMode = AttentionMode.FOCAL,
                 max_focal_targets: int = 1,
                 max_distributed_targets: int = 7,
                 habituation_rate: float = 0.05,
                 recovery_rate: float = 0.02,
                 attention_span: float = 5.0,
                 salience_threshold: float = 0.3,
                 breathing_integration: float = 0.5):
        """
        Initialize the attention mechanism
        
        Args:
            default_mode: Default attention mode
            max_focal_targets: Maximum number of targets in focal attention
            max_distributed_targets: Maximum number of targets in distributed attention
            habituation_rate: Rate at which attention habituates to stimuli
            recovery_rate: Rate at which habituation recovers
            attention_span: Typical duration of sustained attention (seconds)
            salience_threshold: Minimum salience to trigger bottom-up attention
            breathing_integration: Level of integration with breathing system (0.0-1.0)
        """
        self.current_mode = default_mode
        self.max_focal_targets = max_focal_targets
        self.max_distributed_targets = max_distributed_targets
        self.habituation_rate = habituation_rate
        self.recovery_rate = recovery_rate
        self.attention_span = attention_span
        self.salience_threshold = salience_threshold
        self.breathing_integration = breathing_integration
        
        # Attention targets and focus
        self.attention_targets: Dict[str, AttentionTarget] = {}
        self.focused_targets: List[str] = []
        self.excluded_targets: Set[str] = set()
        
        # Current priority weights for different attention dimensions
        self.priority_weights = {
            "salience": 1.0,            # Bottom-up importance
            "relevance": 1.0,           # Top-down task relevance
            "novelty": 0.8,             # Newness/unexpectedness
            "urgency": 0.7,             # Time sensitivity
            "emotional_value": 0.6,     # Emotional significance
            "effort_cost": 0.4,         # Processing effort required
            "expected_value": 0.5       # Expected information gain
        }
        
        # Task-related modulation
        self.current_task = None
        self.task_relevance_fn: Optional[Callable] = None
        
        # Attention history for analysis
        self.attention_history = deque(maxlen=100)
        
        # Statistics
        self.stats = {
            "attention_shifts": 0,          # Count of attention focus changes
            "bottom_up_captures": 0,        # Count of salience-based attention shifts
            "top_down_selections": 0,       # Count of goal-directed attention shifts
            "sustained_focus_time": 0.0,    # Total time of sustained attention
            "distraction_events": 0,        # Count of off-task attention shifts
            "targets_processed": 0,         # Count of targets processed
            "mean_target_duration": 0.0     # Average time spent on each target
        }
        
        logger.info(f"Attention Mechanism initialized in {default_mode.name} mode")
    
    def register_target(self, 
                        target_id: str, 
                        initial_salience: float = 0.5,
                        initial_priority: Optional[float] = None,
                        position: Optional[Tuple[float, ...]] = None,
                        features: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a new attention target
        
        Args:
            target_id: Unique identifier for the target
            initial_salience: Initial salience value (0.0-1.0)
            initial_priority: Initial priority value (0.0-1.0), defaults to salience
            position: Optional position in feature space
            features: Optional dictionary of target features
            
        Returns:
            True if target was registered, False if it already existed
        """
        if target_id in self.attention_targets:
            return False
        
        if initial_priority is None:
            initial_priority = initial_salience
        
        self.attention_targets[target_id] = AttentionTarget(
            id=target_id,
            priority=initial_priority,
            salience=initial_salience,
            position=position,
            features=features or {},
            last_attended=0.0,
            attention_duration=0.0,
            habituation=0.0
        )
        
        logger.debug(f"Registered attention target: {target_id} (salience: {initial_salience:.2f})")
        return True
    
    def update_target(self, 
                      target_id: str, 
                      salience: Optional[float] = None,
                      priority: Optional[float] = None,
                      position: Optional[Tuple[float, ...]] = None,
                      features: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing attention target
        
        Args:
            target_id: Unique identifier for the target
            salience: New salience value (0.0-1.0)
            priority: New priority value (0.0-1.0)
            position: New position in feature space
            features: New or updated features
            
        Returns:
            True if target was updated, False if it doesn't exist
        """
        if target_id not in self.attention_targets:
            return False
        
        target = self.attention_targets[target_id]
        
        if salience is not None:
            target.salience = max(0.0, min(1.0, salience))
        
        if priority is not None:
            target.priority = max(0.0, min(1.0, priority))
        
        if position is not None:
            target.position = position
        
        if features is not None:
            target.features.update(features)
        
        return True
    
    def remove_target(self, target_id: str) -> bool:
        """Remove an attention target"""
        if target_id not in self.attention_targets:
            return False
        
        if target_id in self.focused_targets:
            self.focused_targets.remove(target_id)
        
        if target_id in self.excluded_targets:
            self.excluded_targets.remove(target_id)
        
        del self.attention_targets[target_id]
        return True
    
    def process_attention_cycle(self, 
                               neural_network, 
                               breath_state=None, 
                               timestep=None) -> Dict[str, Any]:
        """
        Process a complete attention cycle
        
        Args:
            neural_network: Neural network to apply attention to
            breath_state: Optional current breathing state
            timestep: Current timestep for timing
            
        Returns:
            Dict with attention processing statistics
        """
        if timestep is None:
            timestep = time.time()
        
        # Update habituation and recovery for all targets
        self._update_habituation(timestep)
        
        # Apply breathing influence on attention
        breath_factor = self._get_breath_factor(breath_state)
        
        # Process bottom-up attention (salience-based)
        bottom_up_targets = self._process_bottom_up_attention(neural_network, breath_factor)
        
        # Process top-down attention (goal-directed)
        top_down_targets = self._process_top_down_attention(neural_network, breath_factor)
        
        # Integrate bottom-up and top-down
        selected_targets = self._integrate_attention_signals(
            bottom_up_targets, 
            top_down_targets, 
            breath_factor
        )
        
        # Determine final focus based on attention mode
        self._update_focus(selected_targets, timestep)
        
        # Apply attention effects to neural network
        self._apply_attention_effects(neural_network, timestep)
        
        # Update attention history
        self._update_attention_history(timestep)
        
        # Update statistics
        self._update_statistics(timestep)
        
        # Return current attention state
        return {
            "mode": self.current_mode.name,
            "focused_targets": self.focused_targets.copy(),
            "stats": self.stats.copy(),
            "breath_factor": breath_factor
        }
    
    def _update_habituation(self, current_time: float) -> None:
        """Update habituation levels for all targets"""
        time_since_last_update = 0.1  # Assume ~10Hz processing frequency
        
        for target_id, target in self.attention_targets.items():
            # Increase habituation for attended targets
            if target_id in self.focused_targets:
                # More habituation for longer continuous attention
                duration_factor = min(1.0, target.attention_duration / (2 * self.attention_span))
                target.habituation = min(1.0, target.habituation + 
                                        (self.habituation_rate * time_since_last_update * (1 + duration_factor)))
                target.attention_duration += time_since_last_update
            else:
                # Recovery for non-attended targets
                target.habituation = max(0.0, target.habituation - 
                                       (self.recovery_rate * time_since_last_update))
                target.attention_duration = 0
    
    def _get_breath_factor(self, breath_state) -> float:
        """Calculate breath influence factor on attention"""
        if not breath_state or self.breathing_integration <= 0:
            return 1.0
            
        # Default factor
        factor = 1.0
        
        # Apply breath state influences
        if "state" in breath_state:
            if breath_state["state"] == "inhale":
                # Inhale enhances distributed attention (broader awareness)
                if self.current_mode == AttentionMode.DISTRIBUTED:
                    factor = 1.0 + (0.3 * self.breathing_integration)
                elif self.current_mode == AttentionMode.FOCAL:
                    factor = 1.0 - (0.1 * self.breathing_integration)
            elif breath_state["state"] == "exhale":
                # Exhale enhances focal attention (narrower focus)
                if self.current_mode == AttentionMode.FOCAL:
                    factor = 1.0 + (0.3 * self.breathing_integration)
                elif self.current_mode == AttentionMode.DISTRIBUTED:
                    factor = 1.0 - (0.1 * self.breathing_integration)
            elif breath_state["state"] == "hold":
                # Hold enhances sustained attention
                factor = 1.0 + (0.2 * self.breathing_integration)
        
        # Apply coherence influence
        if "coherence" in breath_state:
            # Higher coherence increases focused attention
            coherence_factor = 1.0 + ((breath_state["coherence"] - 0.5) * self.breathing_integration)
            factor *= max(0.7, min(1.3, coherence_factor))
        
        return factor
    
    def _process_bottom_up_attention(self, neural_network, breath_factor: float) -> Dict[str, float]:
        """Process bottom-up (salience-based) attention"""
        bottom_up_scores = {}
        
        # Calculate bottom-up attention scores based on salience
        for target_id, target in self.attention_targets.items():
            # Skip excluded targets
            if target_id in self.excluded_targets:
                continue
            
            # Calculate effective salience (reduced by habituation)
            effective_salience = target.salience * (1.0 - target.habituation)
            
            # Skip targets below threshold
            if effective_salience < self.salience_threshold:
                continue
            
            # Calculate novelty based on neural network activity (simplified)
            novelty = 0.0
            if hasattr(neural_network, 'get_region_activity') and 'region' in target.features:
                region_activity = neural_network.get_region_activity(target.features['region'])
                if region_activity is not None:
                    # Higher novelty for unexpected activity patterns
                    expected_activity = target.features.get('expected_activity', 0.5)
                    novelty = abs(region_activity - expected_activity)
            
            # Integrate salience and novelty
            bottom_up_scores[target_id] = (
                effective_salience * self.priority_weights["salience"] +
                novelty * self.priority_weights["novelty"]
            )
        
        # Check for strong enough stimulus to trigger bottom-up capture
        max_score = max(bottom_up_scores.values()) if bottom_up_scores else 0
        if max_score > 0.7:  # High salience threshold for capture
            self.stats["bottom_up_captures"] += 1
        
        return bottom_up_scores
    
    def _process_top_down_attention(self, neural_network, breath_factor: float) -> Dict[str, float]:
        """Process top-down (goal-directed) attention"""
        top_down_scores = {}
        
        # Skip if no current task
        if self.current_task is None:
            return top_down_scores
        
        # Calculate top-down scores based on task relevance
        for target_id, target in self.attention_targets.items():
            # Skip excluded targets
            if target_id in self.excluded_targets:
                continue
            
            # Calculate task relevance
            task_relevance = 0.5  # Default medium relevance
            
            # Use custom relevance function if provided
            if self.task_relevance_fn is not None:
                task_relevance = self.task_relevance_fn(target, self.current_task)
            # Otherwise calculate based on feature match
            elif isinstance(self.current_task, dict) and 'relevant_features' in self.current_task:
                relevance_score = 0
                relevant_features = self.current_task['relevant_features']
                
                for feature, value in relevant_features.items():
                    if feature in target.features:
                        # Increase relevance for matching features
                        similarity = 1.0 - min(1.0, abs(target.features[feature] - value) 
                                             if isinstance(value, (int, float)) else 0.0)
                        relevance_score += similarity
                
                # Normalize relevance score
                task_relevance = min(1.0, relevance_score / max(1, len(relevant_features)))
            
            # Calculate urgency if defined
            urgency = self.current_task.get('urgency', 0.5) if isinstance(self.current_task, dict) else 0.5
            
            # Calculate expected value
            expected_value = target.features.get('expected_value', 0.5)
            
            # Calculate processing cost
            effort_cost = 1.0 - target.features.get('complexity', 0.5)  # Invert complexity
            
            # Integrate factors with weights
            top_down_scores[target_id] = (
                task_relevance * self.priority_weights["relevance"] +
                urgency * self.priority_weights["urgency"] +
                expected_value * self.priority_weights["expected_value"] +
                effort_cost * self.priority_weights["effort_cost"]
            )
        
        return top_down_scores
    
    def _integrate_attention_signals(self, 
                                    bottom_up: Dict[str, float],
                                    top_down: Dict[str, float],
                                    breath_factor: float) -> Dict[str, float]:
        """Integrate bottom-up and top-down attention signals"""
        integrated_scores = {}
        
        # Get all target IDs from both signals
        all_targets = set(bottom_up.keys()).union(set(top_down.keys()))
        
        for target_id in all_targets:
            # Get scores, default to 0 if not present
            bottom_up_score = bottom_up.get(target_id, 0.0)
            top_down_score = top_down.get(target_id, 0.0)
            
            # Adjust balance based on breath state
            # Higher breath factor emphasizes top-down control
            bottom_up_weight = 1.0
            top_down_weight = breath_factor
            
            # Calculate integrated score
            total_weight = bottom_up_weight + top_down_weight
            integrated_scores[target_id] = (
                (bottom_up_score * bottom_up_weight) +
                (top_down_score * top_down_weight)
            ) / total_weight
        
        return integrated_scores
    
    def _update_focus(self, attention_scores: Dict[str, float], timestep: float) -> None:
        """Update the attention focus based on integrated scores and mode"""
        previously_focused = set(self.focused_targets)
        
        # Sort targets by score
        sorted_targets = sorted(
            attention_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Clear current focus
        self.focused_targets = []
        
        # Apply different selection strategy based on mode
        if self.current_mode == AttentionMode.FOCAL:
            # Select top N targets for focal attention
            for target_id, score in sorted_targets[:self.max_focal_targets]:
                if score > 0:
                    self.focused_targets.append(target_id)
                    
        elif self.current_mode == AttentionMode.DISTRIBUTED:
            # Select up to max_distributed_targets with score above threshold
            for target_id, score in sorted_targets[:self.max_distributed_targets]:
                if score > 0.3:  # Minimum threshold for distributed attention
                    self.focused_targets.append(target_id)
                    
        elif self.current_mode == AttentionMode.SCANNING:
            # Implement time-based scanning through targets
            scan_period = 2.0  # Time to cycle through all targets
            scan_index = int((timestep % scan_period) / scan_period * len(sorted_targets))
            if sorted_targets:
                self.focused_targets = [sorted_targets[scan_index % len(sorted_targets)][0]]
                
        elif self.current_mode == AttentionMode.VIGILANT:
            # Focus on high-salience targets plus some recent changes
            high_salience = [t_id for t_id, score in sorted_targets if score > 0.7]
            recent_changes = [t_id for t_id, target in self.attention_targets.items()
                             if target.features.get('recently_changed', False)]
            self.focused_targets = list(set(high_salience + recent_changes))
            
        elif self.current_mode == AttentionMode.RELAXED:
            # Lower threshold, broader focus
            for target_id, score in sorted_targets[:self.max_distributed_targets * 2]:
                if score > 0.2:  # Lower threshold for relaxed attention
                    self.focused_targets.append(target_id)
        
        # Update timestamps for newly attended targets
        currently_focused = set(self.focused_targets)
        
        # Newly focused targets
        for target_id in currently_focused - previously_focused:
            if target_id in self.attention_targets:
                self.attention_targets[target_id].last_attended = timestep
                self.attention_targets[target_id].attention_duration = 0
                self.stats["attention_shifts"] += 1
        
        # Continuously focused targets
        for target_id in currently_focused.intersection(previously_focused):
            target = self.attention_targets.get(target_id)
            if target:
                # Update duration for sustained attention
                duration = timestep - target.last_attended
                target.attention_duration += duration
                target.last_attended = timestep
        
        # If focus changed, update stats
        if previously_focused != currently_focused:
            logger.debug(f"Attention shifted to: {self.focused_targets}")
            
            # Check if shift was off-task
            if self.current_task and isinstance(self.current_task, dict):
                task_targets = self.current_task.get('relevant_targets', [])
                if task_targets:
                    # Consider off-task if no task-relevant targets are in focus
                    if not set(task_targets).intersection(set(self.focused_targets)):
                        self.stats["distraction_events"] += 1
    
    def _apply_attention_effects(self, neural_network, timestep: float) -> None:
        """Apply attention effects to the neural network"""
        if not hasattr(neural_network, 'neurons') or not self.focused_targets:
            return
        
        # Track processing
        processed = 0
        
        # Apply attention effects based on mode
        if self.current_mode == AttentionMode.FOCAL:
            # Strong boost to focused targets, suppression of others
            for target_id in self.focused_targets:
                if 'region' in self.attention_targets[target_id].features:
                    region = self.attention_targets[target_id].features['region']
                    self._boost_region_activity(neural_network, region, 0.5)
                    processed += 1
            
            # Suppress non-attended regions (global inhibition)
            self._apply_global_inhibition(neural_network, 0.2)
                
        elif self.current_mode == AttentionMode.DISTRIBUTED:
            # Moderate boost to multiple targets
            for target_id in self.focused_targets:
                if 'region' in self.attention_targets[target_id].features:
                    region = self.attention_targets[target_id].features['region']
                    self._boost_region_activity(neural_network, region, 0.3)
                    processed += 1
        
        # Other modes have more specialized effects...
        
        self.stats["targets_processed"] += processed
    
    def _boost_region_activity(self, neural_network, region, boost_amount: float) -> None:
        """Boost activity in a neural network region"""
        # Implementation depends on the neural network structure
        # Here's a simplified example:
        if hasattr(neural_network, 'boost_region'):
            neural_network.boost_region(region, boost_amount)
        elif isinstance(region, (list, tuple)) and hasattr(neural_network, 'neurons'):
            # Apply to specific neurons
            for neuron_id in region:
                if neuron_id in neural_network.neurons:
                    # Increase activation for this neuron
                    if isinstance(neural_network.neurons[neuron_id], dict):
                        if 'activation' in neural_network.neurons[neuron_id]:
                            neural_network.neurons[neuron_id]['activation'] = min(
                                1.0, 
                                neural_network.neurons[neuron_id]['activation'] + boost_amount
                            )
    
    def _apply_global_inhibition(self, neural_network, inhibition_amount: float) -> None:
        """Apply global inhibition to non-attended regions"""
        # Implementation depends on the neural network structure
        if hasattr(neural_network, 'apply_global_inhibition'):
            # Exclude focused regions
            excluded_regions = []
            for target_id in self.focused_targets:
                if 'region' in self.attention_targets[target_id].features:
                    excluded_regions.append(self.attention_targets[target_id].features['region'])
            
            neural_network.apply_global_inhibition(inhibition_amount, excluded_regions)
    
    def _update_attention_history(self, timestep: float) -> None:
        """Update attention history for analysis"""
        self.attention_history.append({
            'timestep': timestep,
            'mode': self.current_mode,
            'focused_targets': self.focused_targets.copy(),
            'current_task': self.current_task
        })
    
    def _update_statistics(self, timestep: float) -> None:
        """Update attention statistics"""
        # Calculate sustained focus time
        total_duration = 0
        for target_id in self.focused_targets:
            if target_id in self.attention_targets:
                total_duration += self.attention_targets[target_id].attention_duration
        
        self.stats["sustained_focus_time"] = total_duration
        
        # Calculate mean target duration
        if self.stats["targets_processed"] > 0:
            self.stats["mean_target_duration"] = self.stats["sustained_focus_time"] / self.stats["targets_processed"]
    
    def set_mode(self, mode: AttentionMode) -> bool:
        """Set the current attention mode"""
        if isinstance(mode, AttentionMode):
            self.current_mode = mode
            logger.info(f"Attention mode set to {mode.name}")
            return True
        return False
    
    def set_task(self, task: Any) -> None:
        """Set the current task for top-down attention"""
        self.current_task = task
        logger.info(f"Attention task updated: {task}")
    
    def set_task_relevance_function(self, relevance_fn: Callable) -> None:
        """Set a custom function for calculating task relevance"""
        self.task_relevance_fn = relevance_fn
    
    def exclude_target(self, target_id: str) -> bool:
        """Exclude a target from attention"""
        if target_id not in self.attention_targets:
            return False
        
        self.excluded_targets.add(target_id)
        
        if target_id in self.focused_targets:
            self.focused_targets.remove(target_id)
            
        return True
    
    def include_target(self, target_id: str) -> bool:
        """Re-include a previously excluded target"""
        if target_id not in self.attention_targets:
            return False
        
        if target_id in self.excluded_targets:
            self.excluded_targets.remove(target_id)
            return True
            
        return False
    
    def get_focus_distribution(self) -> Dict[str, float]:
        """Get the current distribution of attention focus"""
        result = {}
        
        total_time = 0
        for target in self.attention_targets.values():
            total_time += target.attention_duration
        
        if total_time > 0:
            for target_id, target in self.attention_targets.items():
                result[target_id] = target.attention_duration / total_time
        
        return result
    
    def get_current_focus(self) -> List[str]:
        """Get currently focused targets"""
        return self.focused_targets.copy()
    
    def get_target_info(self, target_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific target"""
        if target_id not in self.attention_targets:
            return None
        
        target = self.attention_targets[target_id]
        return {
            'id': target.id,
            'priority': target.priority,
            'salience': target.salience,
            'effective_salience': target.salience * (1.0 - target.habituation),
            'position': target.position,
            'features': target.features,
            'last_attended': target.last_attended,
            'attention_duration': target.attention_duration,
            'habituation': target.habituation,
            'is_focused': target_id in self.focused_targets,
            'is_excluded': target_id in self.excluded_targets
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get attention mechanism statistics"""
        return self.stats.copy()


# For direct module testing
if __name__ == "__main__":
    # Simple test network
    class TestNetwork:
        def __init__(self):
            self.neurons = {f"n{i}": {"activation": 0.0, "state": "inactive"} for i in range(20)}
            
        def boost_region(self, region, amount):
            if isinstance(region, list):
                for neuron_id in region:
                    if neuron_id in self.neurons:
                        self.neurons[neuron_id]["activation"] += amount
                        if self.neurons[neuron_id]["activation"] > 0.5:
                            self.neurons[neuron_id]["state"] = "active"
        
        def apply_global_inhibition(self, amount, excluded_regions):
            excluded_neurons = []
            for region in excluded_regions:
                if isinstance(region, list):
                    excluded_neurons.extend(region)
            
            for neuron_id, neuron in self.neurons.items():
                if neuron_id not in excluded_neurons:
                    neuron["activation"] = max(0.0, neuron["activation"] - amount)
                    if neuron["activation"] < 0.5:
                        neuron["state"] = "inactive"
    
    # Create test environment
    network = TestNetwork()
    attention = AttentionMechanism()
    
    # Register some targets
    attention.register_target("target1", initial_salience=0.8, 
                             features={"region": ["n1", "n2", "n3"], "complexity": 0.3})
    attention.register_target("target2", initial_salience=0.6, 
                             features={"region": ["n5", "n6", "n7"], "complexity": 0.5})
    attention.register_target("target3", initial_salience=0.4, 
                             features={"region": ["n10", "n11", "n12"], "complexity": 0.7})
    
    # Set a task
    attention.set_task({
        "name": "find_simple_patterns",
        "relevant_features": {"complexity": 0.3},
        "urgency": 0.7
    })
    
    # Process a few attention cycles
    for i in range(5):
        print(f"\nProcessing attention cycle {i+1}")
        
        # Simulate changing salience
        if i == 2:
            attention.update_target("target3", salience=0.9)
            print("Updated target3 salience to 0.9")
        
        # Process attention
        result = attention.process_attention_cycle(network)
        
        # Print results
        print(f"Mode: {result['mode']}")
        print(f"Focused targets: {result['focused_targets']}")
        
        # Print neuron states after attention
        active_neurons = [n_id for n_id, n in network.neurons.items() if n["state"] == "active"]
        print(f"Active neurons: {active_neurons}")
        
        time.sleep(0.1) 