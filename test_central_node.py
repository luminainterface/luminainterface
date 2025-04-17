import unittest
from central_node import CentralNode
import logging

class TestCentralNode(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.central_node = CentralNode()
        # Set logging level to ERROR to reduce output during tests
        logging.getLogger('CentralNode').setLevel(logging.ERROR)

    def test_component_initialization(self):
        """Test if all components are properly initialized"""
        # Test nodes
        self.assertIn('RSEN', self.central_node.nodes)
        self.assertIn('HybridNode', self.central_node.nodes)
        self.assertIn('NodeZero', self.central_node.nodes)
        
        # Test processors
        self.assertIn('NeuralProcessor', self.central_node.processors)
        self.assertIn('LanguageProcessor', self.central_node.processors)
        self.assertIn('NodeManager', self.central_node.processors)
        
        # Test data directories
        self.assertIn('training_data', self.central_node.data_dirs)
        self.assertIn('model_output', self.central_node.data_dirs)

    def test_get_node(self):
        """Test node retrieval"""
        node = self.central_node.get_node('RSEN')
        self.assertIsNotNone(node)
        self.assertIsNone(self.central_node.get_node('NonExistentNode'))

    def test_get_processor(self):
        """Test processor retrieval"""
        processor = self.central_node.get_processor('NeuralProcessor')
        self.assertIsNotNone(processor)
        self.assertIsNone(self.central_node.get_processor('NonExistentProcessor'))

    def test_list_available_components(self):
        """Test component listing"""
        components = self.central_node.list_available_components()
        self.assertIn('nodes', components)
        self.assertIn('processors', components)
        self.assertIn('data_directories', components)
        
        # Verify some expected components are listed
        self.assertIn('RSEN', components['nodes'])
        self.assertIn('NeuralProcessor', components['processors'])

    def test_system_status(self):
        """Test system status reporting"""
        status = self.central_node.get_system_status()
        self.assertIn('active_nodes', status)
        self.assertIn('active_processors', status)
        self.assertIn('data_directories', status)
        self.assertIn('total_components', status)
        
        # Verify total components calculation
        self.assertEqual(
            status['total_components'],
            status['active_nodes'] + status['active_processors']
        )

    def test_component_dependencies(self):
        """Test component dependency mapping"""
        dependencies = self.central_node.get_component_dependencies()
        self.assertIn('RSEN', dependencies)
        self.assertIn('HybridNode', dependencies)
        
        # Verify some expected dependencies
        self.assertIn('NeuralProcessor', dependencies['RSEN'])
        self.assertIn('NodeManager', dependencies['RSEN'])

    def test_invalid_operations(self):
        """Test error handling for invalid operations"""
        # Test invalid node operation
        with self.assertRaises(ValueError):
            self.central_node.execute_node_operation('NonExistentNode', 'process')
        
        # Test invalid processor operation
        with self.assertRaises(ValueError):
            self.central_node.process_data('NonExistentProcessor', {})

def run_tests():
    """Run all tests and print results"""
    print("Starting Central Node Tests...")
    unittest.main(verbosity=2)
    print("\nAll tests completed!")

if __name__ == '__main__':
    run_tests() 