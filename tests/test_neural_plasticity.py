"""
Test suite for the Neural Plasticity Module

This module tests the key components and functionality of the neural plasticity
module, including STDP learning, homeostatic regulation, and structural plasticity.
"""

import sys
import os
import unittest
import numpy as np
import time
import threading
from unittest.mock import MagicMock, patch

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.v9.neural_plasticity import (
    PlasticityState,
    SpikeEvent,
    SynapticConnection,
    HebbianLearningRule,
    HomeostasisController,
    StructuralPlasticityEngine,
    PlasticityManager
)
from src.v9.breathing_system import BreathingSystem, BreathState

class TestSynapticConnection(unittest.TestCase):
    """Test the SynapticConnection class"""
    
    def test_update_traces(self):
        """Test that synaptic traces decay correctly"""
        # Create a connection with initial traces
        conn = SynapticConnection(
            pre_id=1,
            post_id=2,
            weight=0.5,
            creation_time=100.0,
            last_update=100.0,
            trace_pre=1.0,
            trace_post=1.0,
            eligibility_trace=1.0
        )
        
        # Create config with decay times
        config = {
            "trace_decay_time": 20.0,
            "eligibility_decay_time": 100.0
        }
        
        # Update traces after 20ms (one time constant)
        conn.update_traces(120.0, config)
        
        # After one time constant, traces should decay to ~0.368 of original value
        self.assertAlmostEqual(conn.trace_pre, 1.0 * np.exp(-1), places=3)
        self.assertAlmostEqual(conn.trace_post, 1.0 * np.exp(-1), places=3)
        self.assertAlmostEqual(conn.eligibility_trace, 1.0 * np.exp(-20/100), places=3)
        self.assertEqual(conn.last_update, 120.0)

class TestHebbianLearningRule(unittest.TestCase):
    """Test the HebbianLearningRule class"""
    
    def setUp(self):
        """Set up common test fixtures"""
        self.config = {
            "stdp_window_potentiation": 20.0,
            "stdp_window_depression": 20.0,
            "stdp_potentiation_rate": 0.01,
            "stdp_depression_rate": 0.012,
            "trace_decay_time": 20.0,
            "eligibility_decay_time": 100.0,
            "covariance_learning_rate": 0.005,
            "max_weight": 1.0
        }
        self.hebbian = HebbianLearningRule(self.config)
    
    def test_compute_weight_change_pre_spike(self):
        """Test weight change calculation for pre-synaptic spike"""
        # Create a connection with post trace already set
        conn = SynapticConnection(
            pre_id=1,
            post_id=2,
            weight=0.5,
            creation_time=100.0,
            last_update=100.0,
            trace_pre=0.0,
            trace_post=0.8  # High post trace should lead to depression
        )
        
        # Compute weight change for pre-synaptic spike
        weight_change = self.hebbian.compute_weight_change(conn, True, 110.0)
        
        # Pre spike with high post trace should cause depression
        self.assertTrue(weight_change < 0)
        # Post trace should remain (with decay), pre trace should be set to 1.0
        self.assertAlmostEqual(conn.trace_pre, 1.0)
        self.assertLess(conn.trace_post, 0.8)  # Should have decayed
    
    def test_compute_weight_change_post_spike(self):
        """Test weight change calculation for post-synaptic spike"""
        # Create a connection with pre trace already set
        conn = SynapticConnection(
            pre_id=1,
            post_id=2,
            weight=0.5,
            creation_time=100.0,
            last_update=100.0,
            trace_pre=0.7,  # High pre trace should lead to potentiation
            trace_post=0.0
        )
        
        # Compute weight change for post-synaptic spike
        weight_change = self.hebbian.compute_weight_change(conn, False, 110.0)
        
        # Post spike with high pre trace should cause potentiation
        self.assertTrue(weight_change > 0)
        # Pre trace should remain (with decay), post trace should be set to 1.0
        self.assertLess(conn.trace_pre, 0.7)  # Should have decayed
        self.assertAlmostEqual(conn.trace_post, 1.0)
        # Eligibility trace should have increased
        self.assertGreater(conn.eligibility_trace, 0.0)
    
    def test_apply_covariance_learning(self):
        """Test covariance-based Hebbian learning"""
        # Create test connections
        conn1 = SynapticConnection(
            pre_id=1, post_id=2, weight=0.5, 
            creation_time=100.0, last_update=100.0
        )
        conn2 = SynapticConnection(
            pre_id=3, post_id=4, weight=0.5, 
            creation_time=100.0, last_update=100.0
        )
        
        # Create activity vectors (correlated for conn1, anti-correlated for conn2)
        activity_vectors = {
            1: np.array([0.1, 0.2, 0.3, 0.4, 0.5]),  # Positively correlated with 2
            2: np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
            3: np.array([0.1, 0.2, 0.3, 0.4, 0.5]),  # Negatively correlated with 4
            4: np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        }
        
        # Apply covariance learning
        self.hebbian.apply_covariance_learning([conn1, conn2], activity_vectors)
        
        # Positive correlation should strengthen weight
        self.assertGreater(conn1.weight, 0.5)
        # Negative correlation should weaken weight
        self.assertLess(conn2.weight, 0.5)

class TestHomeostasisController(unittest.TestCase):
    """Test the HomeostasisController class"""
    
    def setUp(self):
        """Set up common test fixtures"""
        self.config = {
            "target_activity": 0.1,
            "homeostatic_time_constant": 3600,
            "scaling_rate": 0.1,  # Higher rate for testing
            "scaling_interval": 0.0,  # No delay for testing
            "excitability_adjustment_rate": 0.1,
            "min_threshold": 0.1,
            "max_threshold": 10.0,
            "min_weight": 0.0,
            "max_weight": 1.0
        }
        self.homeostasis = HomeostasisController(self.config)
    
    def test_update_activity_tracker(self):
        """Test that activity tracker correctly updates with spike events"""
        # Create spike events for neurons
        spikes = [
            SpikeEvent(neuron_id=1, timestamp=100.0),
            SpikeEvent(neuron_id=1, timestamp=101.0),
            SpikeEvent(neuron_id=2, timestamp=102.0)
        ]
        
        # Update activity tracker
        self.homeostasis.update_activity_tracker(spikes)
        
        # Check activity levels
        self.assertIn(1, self.homeostasis.neuron_activity)
        self.assertIn(2, self.homeostasis.neuron_activity)
        self.assertEqual(self.homeostasis.neuron_activity[1], 2.0)  # Two spikes
        self.assertEqual(self.homeostasis.neuron_activity[2], 1.0)  # One spike
    
    def test_apply_synaptic_scaling(self):
        """Test that synaptic scaling correctly adjusts weights"""
        # Create connections for a neuron
        connections = {
            1: [
                SynapticConnection(pre_id=2, post_id=1, weight=0.5, 
                                  creation_time=100.0, last_update=100.0),
                SynapticConnection(pre_id=3, post_id=1, weight=0.7, 
                                  creation_time=100.0, last_update=100.0)
            ]
        }
        
        # Set activity for the neuron (too high)
        self.homeostasis.neuron_activity = {1: 0.3}  # 3x target
        
        # Apply scaling (should decrease weights)
        self.homeostasis.apply_synaptic_scaling(connections)
        
        # Weights should have decreased
        self.assertLess(connections[1][0].weight, 0.5)
        self.assertLess(connections[1][1].weight, 0.7)
        
        # Set activity for the neuron (too low)
        self.homeostasis.neuron_activity = {1: 0.05}  # 0.5x target
        
        # Apply scaling (should increase weights)
        self.homeostasis.apply_synaptic_scaling(connections)
        
        # Get the weights after scaling up
        weight1 = connections[1][0].weight
        weight2 = connections[1][1].weight
        
        # Weights should have increased
        self.assertGreater(weight1, connections[1][0].weight)
        self.assertGreater(weight2, connections[1][1].weight)
    
    def test_adjust_neuron_excitability(self):
        """Test that neuron excitability is adjusted correctly"""
        # Create neuron properties
        neuron_properties = {
            1: {"threshold": 1.0},
            2: {"threshold": 1.0},
            3: {"threshold": 1.0}
        }
        
        # Set activities (too high, too low, and just right)
        self.homeostasis.neuron_activity = {
            1: 0.2,   # Too active (2x target)
            2: 0.05,  # Too inactive (0.5x target)
            3: 0.1    # Just right
        }
        
        # Adjust excitability
        self.homeostasis.adjust_neuron_excitability(neuron_properties)
        
        # Thresholds should have changed accordingly
        self.assertGreater(neuron_properties[1]["threshold"], 1.0)  # Increased
        self.assertLess(neuron_properties[2]["threshold"], 1.0)     # Decreased
        self.assertAlmostEqual(neuron_properties[3]["threshold"], 1.0)  # Unchanged

class TestStructuralPlasticityEngine(unittest.TestCase):
    """Test the StructuralPlasticityEngine class"""
    
    def setUp(self):
        """Set up common test fixtures"""
        self.config = {
            "creation_probability": 1.0,  # Always create for testing
            "pruning_threshold": 0.2,
            "pruning_interval": 0.0,  # No delay for testing
            "connection_radius": 3.0,
            "inactivity_pruning_time": 10.0,
            "inactivity_pruning_probability": 1.0  # Always prune inactive for testing
        }
        self.structural = StructuralPlasticityEngine(self.config)
    
    def test_generate_connection_candidates(self):
        """Test that connection candidates are generated based on proximity"""
        # Create neuron positions
        neuron_positions = {
            1: np.array([0.0, 0.0, 0.0]),
            2: np.array([1.0, 0.0, 0.0]),  # Close to 1
            3: np.array([10.0, 0.0, 0.0])  # Far from 1 and 2
        }
        
        # No existing connections
        existing_connections = set()
        
        # Generate candidates
        self.structural.generate_connection_candidates(neuron_positions, existing_connections)
        
        # Should have candidates between 1 and 2 (both directions)
        self.assertIn((1, 2), self.structural.connection_candidates)
        self.assertIn((2, 1), self.structural.connection_candidates)
        
        # Should not have candidates between 1 and 3 or 2 and 3 (too far)
        self.assertNotIn((1, 3), self.structural.connection_candidates)
        self.assertNotIn((3, 1), self.structural.connection_candidates)
        self.assertNotIn((2, 3), self.structural.connection_candidates)
        self.assertNotIn((3, 2), self.structural.connection_candidates)
    
    def test_try_form_connections(self):
        """Test that connections are formed based on activity correlation"""
        # Add connection candidates
        self.structural.connection_candidates = {(1, 2), (3, 4), (5, 6)}
        
        # Create activity history (correlated for 1-2, not for others)
        activity_history = {
            1: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            2: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # Correlated with 1
            3: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            4: [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]   # Anti-correlated with 3
            # 5-6 have no activity history
        }
        
        # No existing connections
        existing_connections = set()
        
        # Try to form connections
        new_connections = self.structural.try_form_connections(
            activity_history, existing_connections)
        
        # With creation_probability=1.0, all candidates should form connections
        self.assertEqual(len(new_connections), 3)
        self.assertIn((1, 2), new_connections)
        self.assertIn((3, 4), new_connections)
        self.assertIn((5, 6), new_connections)
        
        # Candidates should have been removed
        self.assertEqual(len(self.structural.connection_candidates), 0)
    
    def test_prune_connections(self):
        """Test that weak and inactive connections are pruned"""
        # Create connections with different weights and update times
        current_time = time.time()
        connections = [
            SynapticConnection(pre_id=1, post_id=2, weight=0.1,  # Below threshold
                              creation_time=current_time-20, last_update=current_time-20),
            SynapticConnection(pre_id=3, post_id=4, weight=0.3,  # Above threshold
                              creation_time=current_time-20, last_update=current_time-20),
            SynapticConnection(pre_id=5, post_id=6, weight=0.5,  # Above threshold but inactive
                              creation_time=current_time-20, last_update=current_time-15)
        ]
        
        # Set inactivity pruning time to 5 seconds for testing
        self.config["inactivity_pruning_time"] = 5.0
        
        # Prune connections
        to_prune = self.structural.prune_connections(connections)
        
        # Should prune the weak connection and the inactive one
        self.assertEqual(len(to_prune), 2)
        self.assertIn(connections[0], to_prune)  # Weak
        self.assertIn(connections[2], to_prune)  # Inactive

class TestPlasticityManager(unittest.TestCase):
    """Test the PlasticityManager class"""
    
    def setUp(self):
        """Set up common test fixtures"""
        # Create mock breathing system
        self.mock_breathing = MagicMock(spec=BreathingSystem)
        self.mock_breathing.get_current_state.return_value = BreathState.INHALE
        self.mock_breathing.get_current_amplitude.return_value = 1.0
        
        # Create plasticity manager with testing config
        self.config = {
            "stdp_window_potentiation": 20.0,
            "stdp_window_depression": 20.0,
            "stdp_potentiation_rate": 0.1,  # Higher for testing
            "stdp_depression_rate": 0.12,   # Higher for testing
            "trace_decay_time": 20.0,
            "eligibility_decay_time": 100.0,
            "target_activity": 0.1,
            "homeostatic_time_constant": 3600,
            "scaling_rate": 0.1,            # Higher for testing
            "scaling_interval": 0.0,        # No delay for testing
            "excitability_adjustment_rate": 0.1,
            "min_threshold": 0.1,
            "max_threshold": 10.0,
            "min_weight": 0.0,
            "max_weight": 1.0,
            "weight_init_mean": 0.5,
            "weight_init_std": 0.1,
            "creation_probability": 0.5,
            "pruning_threshold": 0.2,
            "pruning_interval": 0.0,        # No delay for testing
            "connection_radius": 3.0,
            "inactivity_pruning_time": 3600.0,
            "inactivity_pruning_probability": 0.3,
            "max_connections_per_neuron": 1000,
            "metaplasticity_time_constant": 900.0,
            "metaplasticity_strength": 0.1,
            "covariance_learning_rate": 0.01,
            "breathing_modulation_strength": 0.5
        }
        
        self.plasticity = PlasticityManager(
            config=self.config,
            breathing_system=self.mock_breathing
        )
    
    def test_create_connection(self):
        """Test that connections are correctly created"""
        # Create a connection
        conn = self.plasticity.create_connection(pre_id=1, post_id=2, initial_weight=0.6)
        
        # Check connection properties
        self.assertEqual(conn.pre_id, 1)
        self.assertEqual(conn.post_id, 2)
        self.assertEqual(conn.weight, 0.6)
        
        # Check that connection is stored correctly
        self.assertIn(2, self.plasticity.connections)
        self.assertIn(conn, self.plasticity.connections[2])
        self.assertIn((1, 2), self.plasticity.connections_by_id)
        self.assertEqual(self.plasticity.connections_by_id[(1, 2)], conn)
        
        # Create another connection with random weight
        conn2 = self.plasticity.create_connection(pre_id=3, post_id=4)
        
        # Weight should be within bounds
        self.assertGreaterEqual(conn2.weight, self.config["min_weight"])
        self.assertLessEqual(conn2.weight, self.config["max_weight"])
    
    def test_remove_connection(self):
        """Test that connections are correctly removed"""
        # Create a connection
        conn = self.plasticity.create_connection(pre_id=1, post_id=2, initial_weight=0.6)
        
        # Remove the connection
        result = self.plasticity.remove_connection(pre_id=1, post_id=2)
        
        # Should return True for successful removal
        self.assertTrue(result)
        
        # Connection should be removed from storage
        self.assertNotIn((1, 2), self.plasticity.connections_by_id)
        self.assertEqual(len(self.plasticity.connections[2]), 0)
        
        # Trying to remove a non-existent connection should return False
        result = self.plasticity.remove_connection(pre_id=5, post_id=6)
        self.assertFalse(result)
    
    def test_register_and_process_spikes(self):
        """Test spike registration and processing"""
        # Set up connections for testing
        self.plasticity.create_connection(pre_id=1, post_id=2, initial_weight=0.5)
        self.plasticity.create_connection(pre_id=2, post_id=3, initial_weight=0.5)
        
        # Register a pre-synaptic spike
        self.plasticity.register_spike(neuron_id=1, timestamp=100.0)
        self.assertEqual(len(self.plasticity.spike_buffer), 1)
        
        # Process spikes manually (normally done by the processing thread)
        self.plasticity._process_spikes()
        
        # Buffer should be empty now
        self.assertEqual(len(self.plasticity.spike_buffer), 0)
        
        # Register a post-synaptic spike (which should cause potentiation)
        self.plasticity.register_spike(neuron_id=2, timestamp=110.0)
        self.plasticity._process_spikes()
        
        # Get the updated weight
        weight = self.plasticity.get_connection_weight(pre_id=1, post_id=2)
        
        # Weight should have increased due to STDP potentiation
        self.assertGreater(weight, 0.5)
    
    def test_start_stop(self):
        """Test that the processing thread can be started and stopped"""
        # Start the processing thread
        self.plasticity.start()
        
        # Check that thread is running
        self.assertTrue(self.plasticity.running)
        self.assertIsNotNone(self.plasticity.thread)
        self.assertTrue(self.plasticity.thread.is_alive())
        
        # Stop the processing thread
        self.plasticity.stop()
        
        # Check that thread is stopped
        self.assertFalse(self.plasticity.running)
        self.assertFalse(self.plasticity.thread.is_alive())
    
    def test_get_statistics(self):
        """Test that statistics are correctly reported"""
        # Create some connections
        self.plasticity.create_connection(pre_id=1, post_id=2, initial_weight=0.4)
        self.plasticity.create_connection(pre_id=3, post_id=2, initial_weight=0.6)
        
        # Get statistics
        stats = self.plasticity.get_statistics()
        
        # Check statistics
        self.assertEqual(stats["state"], "INACTIVE")
        self.assertEqual(stats["total_connections"], 2)
        self.assertEqual(stats["neurons_with_connections"], 1)
        self.assertAlmostEqual(stats["average_weight"], 0.5)
        self.assertEqual(stats["spikes_in_buffer"], 0)
    
    def test_breathing_modulation(self):
        """Test that breathing modulates plasticity"""
        # Create a connection
        self.plasticity.create_connection(pre_id=1, post_id=2, initial_weight=0.5)
        
        # Test with inhale state - should enhance potentiation
        self.mock_breathing.get_current_state.return_value = BreathState.INHALE
        self.plasticity.register_spike(neuron_id=1, timestamp=100.0)
        self.plasticity.register_spike(neuron_id=2, timestamp=110.0)  # Post follows pre - LTP
        self.plasticity._process_spikes()
        
        # Get weight after potentiation during inhale
        weight_inhale = self.plasticity.get_connection_weight(pre_id=1, post_id=2)
        
        # Reset connection
        self.plasticity.remove_connection(pre_id=1, post_id=2)
        conn = self.plasticity.create_connection(pre_id=1, post_id=2, initial_weight=0.5)
        
        # Test with exhale state - should reduce potentiation
        self.mock_breathing.get_current_state.return_value = BreathState.EXHALE
        self.plasticity.register_spike(neuron_id=1, timestamp=100.0)
        self.plasticity.register_spike(neuron_id=2, timestamp=110.0)  # Post follows pre - LTP
        self.plasticity._process_spikes()
        
        # Get weight after potentiation during exhale
        weight_exhale = self.plasticity.get_connection_weight(pre_id=1, post_id=2)
        
        # Potentiation during inhale should be stronger than during exhale
        self.assertGreater(weight_inhale, weight_exhale)

if __name__ == "__main__":
    unittest.main() 