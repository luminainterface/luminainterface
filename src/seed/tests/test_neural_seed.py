"""
Tests for the Neural Seed module
"""

import unittest
from unittest.mock import patch
import math
from datetime import datetime, timedelta
import time

from src.seed.neural_seed import NeuralSeed, ConnectionSocket

class TestNeuralSeed(unittest.TestCase):
    """Test cases for NeuralSeed"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.seed = NeuralSeed()
        
    def test_initialization(self):
        """Test initialization of NeuralSeed"""
        self.assertIsNotNone(self.seed.id)
        self.assertEqual(self.seed.state['growth_rate'], 0.0)
        self.assertEqual(self.seed.state['stability'], 1.0)
        self.assertEqual(self.seed.state['complexity'], 0.0)
        self.assertEqual(self.seed.state['consciousness_level'], 0.0)
        self.assertEqual(self.seed.state['stage'], 'seed')
        self.assertEqual(self.seed.state['age'], 0)
        self.assertEqual(self.seed.state['dictionary_size'], 100)
        self.assertEqual(self.seed.state['connection_stability'], 1.0)
        self.assertEqual(self.seed.state['bridge_count'], 0)
        self.assertEqual(len(self.seed.sockets), 0)
        self.assertEqual(len(self.seed.bridges), 0)
        
    def test_growth_stages(self):
        """Test growth stage transitions"""
        stages = ['seed', 'sprout', 'sapling', 'mature']
        thresholds = [0.0, 0.3, 0.6, 0.9]
        
        for stage, threshold in zip(stages, thresholds):
            self.seed.state['consciousness_level'] = threshold
            self.seed._check_stage_transition()
            self.assertEqual(self.seed.state['stage'], stage)
            
    def test_stability_calculations(self):
        """Test stability calculation methods"""
        # Test component stability
        self.assertEqual(self.seed._calculate_component_stability(), 1.0)
        
        # Add some test components
        self.seed.state['active_components'].add('test_component')
        self.assertEqual(self.seed._calculate_component_stability(), 1.0)
        
        # Test growth stability
        self.assertEqual(self.seed._calculate_growth_stability(), 1.0)
        
        # Add growth history
        self.seed.metrics['growth_history'] = [0.1, 0.2, 0.3]
        stability = self.seed._calculate_growth_stability()
        self.assertGreater(stability, 0.0)
        self.assertLessEqual(stability, 1.0)
        
        # Test complexity stability
        self.assertEqual(self.seed._calculate_complexity_stability(), 1.0)
        
        # Add complexity history
        self.seed.metrics['complexity_history'] = [0.1, 0.15, 0.2]
        stability = self.seed._calculate_complexity_stability()
        self.assertGreater(stability, 0.0)
        self.assertLessEqual(stability, 1.0)
        
    def test_growth_factor_calculation(self):
        """Test growth factor calculation"""
        # Test initial growth factor
        factor = self.seed._calculate_growth_factor()
        self.assertGreater(factor, 0.0)
        self.assertLessEqual(factor, 1.0)
        
        # Test with increased age
        self.seed.state['age'] = 10
        factor_with_age = self.seed._calculate_growth_factor()
        self.assertGreater(factor_with_age, factor)
        
        # Test with increased complexity
        self.seed.state['complexity'] = 0.5
        factor_with_complexity = self.seed._calculate_growth_factor()
        self.assertLess(factor_with_complexity, factor_with_age)
        
    def test_dictionary_adaptation(self):
        """Test dictionary size adaptation"""
        initial_size = self.seed.state['dictionary_size']
        
        # Test growth increases dictionary size
        self.seed.state['consciousness_level'] = 0.5
        self.seed.state['complexity'] = 0.3
        self.seed._adapt_dictionary_size()
        
        self.assertGreater(self.seed.state['dictionary_size'], initial_size)
        
    def test_component_activation(self):
        """Test component activation and deactivation"""
        component = "test_component"
        
        # Test activation with high stability
        self.seed.state['stability'] = 0.8
        self.assertTrue(self.seed.activate_component(component))
        self.assertIn(component, self.seed.state['active_components'])
        self.assertNotIn(component, self.seed.state['dormant_components'])
        
        # Test activation with low stability
        self.seed.state['stability'] = 0.6
        self.assertFalse(self.seed.activate_component("another_component"))
        
        # Test deactivation
        self.seed.deactivate_component(component)
        self.assertNotIn(component, self.seed.state['active_components'])
        self.assertIn(component, self.seed.state['dormant_components'])
        
    def test_word_dictionary(self):
        """Test dictionary word management"""
        word = "test"
        embedding = [0.1, 0.2, 0.3]
        
        # Test adding word within size limit
        self.assertTrue(self.seed.add_word(word, embedding))
        self.assertIn(word, self.seed.dictionary)
        self.assertEqual(self.seed.dictionary[word], embedding)
        
        # Test adding words beyond size limit
        self.seed.state['dictionary_size'] = 1  # Set size limit to 1
        self.assertFalse(self.seed.add_word("another_word", embedding))
        
    def test_get_state(self):
        """Test getting neural seed state"""
        # Add some test data
        self.seed.metrics['growth_history'] = [0.1, 0.2, 0.3]
        self.seed.metrics['stability_history'] = [1.0, 0.9, 0.8]
        self.seed.metrics['complexity_history'] = [0.1, 0.15, 0.2]
        self.seed.metrics['data_transferred'] = 5
        self.seed.metrics['last_transfer'] = time.time()
        
        state = self.seed.get_state()
        
        self.assertEqual(state['id'], self.seed.id)
        self.assertEqual(state['state'], self.seed.state)
        
        # Check metrics
        for metric_name, metric_value in state['metrics'].items():
            if isinstance(metric_value, list):
                self.assertLessEqual(len(metric_value), 10)
            else:
                self.assertIsInstance(metric_value, (int, float))
            
    @patch('src.seed.neural_seed.datetime')
    def test_growth_loop(self, mock_datetime):
        """Test the growth loop execution"""
        # Mock datetime to control loop timing
        mock_times = [
            datetime.now() + timedelta(seconds=i)
            for i in range(5)
        ]
        mock_datetime.now.side_effect = mock_times
        
        # Start growth process
        self.seed.start_growth()
        
        # Wait for a short time to allow growth to occur
        time.sleep(0.5)
        
        # Stop growth
        self.seed.stop_growth()
        
        # Verify metrics were updated
        self.assertGreater(len(self.seed.metrics['growth_history']), 0)
        self.assertGreater(len(self.seed.metrics['stability_history']), 0)
        self.assertGreater(len(self.seed.metrics['complexity_history']), 0)
        
        # Verify state changes
        self.assertGreater(self.seed.state['age'], 0)
        self.assertGreater(self.seed.state['consciousness_level'], 0)

    def test_socket_creation(self):
        """Test socket creation and management"""
        # Create input socket
        input_socket_id = self.seed.create_socket("input")
        self.assertIsNotNone(input_socket_id)
        self.assertIn(input_socket_id, self.seed.sockets)
        self.assertEqual(self.seed.sockets[input_socket_id].type, "input")
        
        # Create output socket
        output_socket_id = self.seed.create_socket("output")
        self.assertIsNotNone(output_socket_id)
        self.assertIn(output_socket_id, self.seed.sockets)
        self.assertEqual(self.seed.sockets[output_socket_id].type, "output")
        
        # Remove socket
        self.assertTrue(self.seed.remove_socket(input_socket_id))
        self.assertNotIn(input_socket_id, self.seed.sockets)
        
    def test_socket_connections(self):
        """Test socket connection management"""
        # Create sockets
        input_socket_id = self.seed.create_socket("input")
        output_socket_id = self.seed.create_socket("output")
        
        # Connect sockets
        self.assertTrue(self.seed.connect_sockets(output_socket_id, input_socket_id))
        self.assertIn(input_socket_id, self.seed.sockets[output_socket_id].connections)
        
        # Disconnect sockets
        self.assertTrue(self.seed.disconnect_sockets(output_socket_id, input_socket_id))
        self.assertNotIn(input_socket_id, self.seed.sockets[output_socket_id].connections)
        
    def test_bridge_creation(self):
        """Test bridge creation and management"""
        # Create sockets
        output_socket_id = self.seed.create_socket("output")
        
        # Create bridge
        bridge_id = self.seed.create_bridge(
            output_socket_id,
            "target_seed_id",
            "target_socket_id"
        )
        self.assertIsNotNone(bridge_id)
        self.assertIn(bridge_id, self.seed.bridges)
        self.assertEqual(self.seed.state['bridge_count'], 1)
        
        # Remove bridge
        self.assertTrue(self.seed.remove_bridge(bridge_id))
        self.assertNotIn(bridge_id, self.seed.bridges)
        self.assertEqual(self.seed.state['bridge_count'], 0)
        
    def test_bridge_stability(self):
        """Test bridge stability calculation"""
        # Create sockets and bridge
        output_socket_id = self.seed.create_socket("output")
        bridge_id = self.seed.create_bridge(
            output_socket_id,
            "target_seed_id",
            "target_socket_id"
        )
        
        # Test stability calculation
        stability = self.seed._calculate_bridge_stability(self.seed.bridges[bridge_id])
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 1.0)
        
    def test_socket_data_transfer(self):
        """Test socket data transfer functionality"""
        # Create sockets and connect them
        input_socket_id = self.seed.create_socket("input")
        output_socket_id = self.seed.create_socket("output")
        self.seed.connect_sockets(output_socket_id, input_socket_id)
        
        # Activate sockets
        self.seed.sockets[input_socket_id].active = True
        self.seed.sockets[output_socket_id].active = True
        
        # Start growth process to activate socket processing
        self.seed.start_growth()
        
        # Send and receive data
        test_data = {"test": "data"}
        self.assertTrue(self.seed.sockets[output_socket_id].send(test_data))
        
        # Wait for data to be processed
        time.sleep(0.5)
        
        # Try to receive data multiple times
        received_data = None
        for _ in range(5):
            received_data = self.seed.sockets[input_socket_id].receive()
            if received_data:
                break
            time.sleep(0.1)
            
        self.assertEqual(received_data, test_data)
        
        # Stop growth
        self.seed.stop_growth()
        
    def test_thread_management(self):
        """Test thread management in start/stop growth"""
        # Start growth
        self.seed.start_growth()
        self.assertTrue(self.seed.running)
        self.assertIsNotNone(self.seed._socket_thread)
        self.assertIsNotNone(self.seed._bridge_thread)
        
        # Stop growth
        self.seed.stop_growth()
        self.assertFalse(self.seed.running)
        
    def test_get_state_with_connections(self):
        """Test get_state with socket and bridge information"""
        # Create sockets and bridge
        input_socket_id = self.seed.create_socket("input")
        output_socket_id = self.seed.create_socket("output")
        self.seed.connect_sockets(output_socket_id, input_socket_id)
        bridge_id = self.seed.create_bridge(
            output_socket_id,
            "target_seed_id",
            "target_socket_id"
        )
        
        # Get state
        state = self.seed.get_state()
        
        # Check socket information
        self.assertIn('sockets', state)
        self.assertIn(input_socket_id, state['sockets'])
        self.assertIn(output_socket_id, state['sockets'])
        
        # Check bridge information
        self.assertIn('bridges', state)
        self.assertIn(bridge_id, state['bridges'])
        
    def test_invalid_socket_operations(self):
        """Test invalid socket operations"""
        # Try to connect non-existent sockets
        self.assertFalse(self.seed.connect_sockets("nonexistent1", "nonexistent2"))
        
        # Try to create bridge with invalid socket
        self.assertIsNone(self.seed.create_bridge(
            "nonexistent",
            "target_seed_id",
            "target_socket_id"
        ))
        
    def test_socket_cleanup(self):
        """Test socket cleanup when removing bridge"""
        # Create sockets and bridge
        output_socket_id = self.seed.create_socket("output")
        bridge_id = self.seed.create_bridge(
            output_socket_id,
            "target_seed_id",
            "target_socket_id"
        )
        
        # Remove bridge
        self.assertTrue(self.seed.remove_bridge(bridge_id))
        self.assertEqual(self.seed.state['bridge_count'], 0)
        
        # Socket should still exist
        self.assertIn(output_socket_id, self.seed.sockets)

if __name__ == '__main__':
    unittest.main() 