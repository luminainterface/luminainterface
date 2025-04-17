import unittest
import numpy as np
from src.neural_linguistic_bridge import (
    NeuralLinguisticBridge,
    NeuralPattern,
    LinguisticPattern
)

class TestNeuralLinguisticBridge(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.bridge = NeuralLinguisticBridge(dimension=64)
        
    def test_neural_to_linguistic_conversion(self):
        """Test conversion from neural to linguistic pattern"""
        # Create test neural pattern
        neural_pattern = NeuralPattern(
            activation_pattern=np.random.rand(64),
            temporal_sequence=[np.random.rand(64) for _ in range(3)],
            strength=0.8,
            frequency=0.5,
            connections=[(0, 1), (1, 2), (2, 0)],
            metadata={'pattern_type': 'test'}
        )
        
        # Convert to linguistic pattern
        linguistic_pattern = self.bridge.neural_to_linguistic(neural_pattern)
        
        # Verify conversion
        self.assertIsInstance(linguistic_pattern, LinguisticPattern)
        self.assertEqual(linguistic_pattern.semantic_vector.shape, (64,))
        self.assertGreater(len(linguistic_pattern.text), 0)
        self.assertGreater(len(linguistic_pattern.associations), 0)
        self.assertTrue(0 <= linguistic_pattern.confidence <= 1)
        self.assertEqual(linguistic_pattern.metadata['original_pattern_type'], 'neural')
        
    def test_linguistic_to_neural_conversion(self):
        """Test conversion from linguistic to neural pattern"""
        # Create test linguistic pattern
        linguistic_pattern = LinguisticPattern(
            text="Test pattern with moderate activity",
            semantic_vector=np.random.rand(64),
            context={'strength': 0.7, 'complexity': 0.5},
            associations=['test', 'moderate'],
            confidence=0.8,
            metadata={'pattern_type': 'test'}
        )
        
        # Convert to neural pattern
        neural_pattern = self.bridge.linguistic_to_neural(linguistic_pattern)
        
        # Verify conversion
        self.assertIsInstance(neural_pattern, NeuralPattern)
        self.assertEqual(neural_pattern.activation_pattern.shape, (64,))
        self.assertGreater(len(neural_pattern.temporal_sequence), 0)
        self.assertTrue(0 <= neural_pattern.strength <= 1)
        self.assertTrue(0 <= neural_pattern.frequency <= 1)
        self.assertGreater(len(neural_pattern.connections), 0)
        self.assertEqual(neural_pattern.metadata['original_pattern_type'], 'linguistic')
        
    def test_bidirectional_conversion(self):
        """Test pattern preservation through bidirectional conversion"""
        # Create initial neural pattern
        original_neural = NeuralPattern(
            activation_pattern=np.random.rand(64),
            temporal_sequence=[np.random.rand(64) for _ in range(3)],
            strength=0.8,
            frequency=0.5,
            connections=[(0, 1), (1, 2), (2, 0)],
            metadata={'pattern_type': 'test'}
        )
        
        # Convert neural -> linguistic -> neural
        linguistic = self.bridge.neural_to_linguistic(original_neural)
        converted_neural = self.bridge.linguistic_to_neural(linguistic)
        
        # Verify pattern preservation
        np.testing.assert_array_almost_equal(
            original_neural.activation_pattern,
            converted_neural.activation_pattern,
            decimal=5
        )
        self.assertAlmostEqual(original_neural.strength, converted_neural.strength, places=5)
        
    def test_pattern_description(self):
        """Test generation of pattern descriptions"""
        # Create test neural pattern with strong activation
        neural_pattern = NeuralPattern(
            activation_pattern=np.ones(64) * 0.9,  # Strong activation
            temporal_sequence=[np.random.rand(64) for _ in range(3)],
            strength=0.9,
            frequency=0.5,
            connections=[(i, i+1) for i in range(30)],  # Dense connections
            metadata={'pattern_type': 'test', 'confidence': 0.9}
        )
        
        # Convert to linguistic pattern
        linguistic_pattern = self.bridge.neural_to_linguistic(neural_pattern)
        
        # Verify description contains expected elements
        self.assertIn("Strong activation pattern", linguistic_pattern.text)
        self.assertIn("dense neural connections", linguistic_pattern.text)
        self.assertIn("high confidence", linguistic_pattern.text)
        
    def test_pattern_associations(self):
        """Test extraction of pattern associations"""
        # Create test neural pattern with specific characteristics
        neural_pattern = NeuralPattern(
            activation_pattern=np.ones(64) * 0.9,  # High intensity
            temporal_sequence=[np.random.rand(64) for _ in range(6)],  # Extended sequence
            strength=0.8,
            frequency=0.5,
            connections=[(i, i+1) for i in range(40)],  # Highly connected
            metadata={'pattern_type': 'test_type'}
        )
        
        # Convert to linguistic pattern
        linguistic_pattern = self.bridge.neural_to_linguistic(neural_pattern)
        
        # Verify expected associations
        self.assertIn("high_intensity", linguistic_pattern.associations)
        self.assertIn("extended_sequence", linguistic_pattern.associations)
        self.assertIn("highly_connected", linguistic_pattern.associations)
        self.assertIn("type_test_type", linguistic_pattern.associations)
        
    def test_error_handling(self):
        """Test error handling in pattern conversion"""
        # Test with invalid neural pattern
        with self.assertRaises(Exception):
            invalid_neural = NeuralPattern(
                activation_pattern=np.array([]),  # Empty pattern
                temporal_sequence=[],
                strength=0.5,
                frequency=0.5,
                connections=[],
                metadata={}
            )
            self.bridge.neural_to_linguistic(invalid_neural)
            
        # Test with invalid linguistic pattern
        with self.assertRaises(Exception):
            invalid_linguistic = LinguisticPattern(
                text="",  # Empty text
                semantic_vector=np.array([]),  # Empty vector
                context={},
                associations=[],
                confidence=0.5,
                metadata={}
            )
            self.bridge.linguistic_to_neural(invalid_linguistic)
            
    def test_pattern_memory(self):
        """Test pattern memory functionality"""
        # Create and convert a pattern
        neural_pattern = NeuralPattern(
            activation_pattern=np.random.rand(64),
            temporal_sequence=[np.random.rand(64) for _ in range(3)],
            strength=0.8,
            frequency=0.5,
            connections=[(0, 1), (1, 2), (2, 0)],
            metadata={'pattern_type': 'test'}
        )
        
        linguistic_pattern = self.bridge.neural_to_linguistic(neural_pattern)
        
        # Verify pattern is stored in memory
        pattern_id = self.bridge._generate_pattern_id(neural_pattern)
        stored_neural, stored_linguistic = self.bridge.pattern_memory[pattern_id]
        
        # Verify stored patterns match
        np.testing.assert_array_equal(stored_neural.activation_pattern, neural_pattern.activation_pattern)
        np.testing.assert_array_equal(stored_linguistic.semantic_vector, linguistic_pattern.semantic_vector)

if __name__ == '__main__':
    unittest.main() 
 
 
import numpy as np
from src.neural_linguistic_bridge import (
    NeuralLinguisticBridge,
    NeuralPattern,
    LinguisticPattern
)

class TestNeuralLinguisticBridge(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.bridge = NeuralLinguisticBridge(dimension=64)
        
    def test_neural_to_linguistic_conversion(self):
        """Test conversion from neural to linguistic pattern"""
        # Create test neural pattern
        neural_pattern = NeuralPattern(
            activation_pattern=np.random.rand(64),
            temporal_sequence=[np.random.rand(64) for _ in range(3)],
            strength=0.8,
            frequency=0.5,
            connections=[(0, 1), (1, 2), (2, 0)],
            metadata={'pattern_type': 'test'}
        )
        
        # Convert to linguistic pattern
        linguistic_pattern = self.bridge.neural_to_linguistic(neural_pattern)
        
        # Verify conversion
        self.assertIsInstance(linguistic_pattern, LinguisticPattern)
        self.assertEqual(linguistic_pattern.semantic_vector.shape, (64,))
        self.assertGreater(len(linguistic_pattern.text), 0)
        self.assertGreater(len(linguistic_pattern.associations), 0)
        self.assertTrue(0 <= linguistic_pattern.confidence <= 1)
        self.assertEqual(linguistic_pattern.metadata['original_pattern_type'], 'neural')
        
    def test_linguistic_to_neural_conversion(self):
        """Test conversion from linguistic to neural pattern"""
        # Create test linguistic pattern
        linguistic_pattern = LinguisticPattern(
            text="Test pattern with moderate activity",
            semantic_vector=np.random.rand(64),
            context={'strength': 0.7, 'complexity': 0.5},
            associations=['test', 'moderate'],
            confidence=0.8,
            metadata={'pattern_type': 'test'}
        )
        
        # Convert to neural pattern
        neural_pattern = self.bridge.linguistic_to_neural(linguistic_pattern)
        
        # Verify conversion
        self.assertIsInstance(neural_pattern, NeuralPattern)
        self.assertEqual(neural_pattern.activation_pattern.shape, (64,))
        self.assertGreater(len(neural_pattern.temporal_sequence), 0)
        self.assertTrue(0 <= neural_pattern.strength <= 1)
        self.assertTrue(0 <= neural_pattern.frequency <= 1)
        self.assertGreater(len(neural_pattern.connections), 0)
        self.assertEqual(neural_pattern.metadata['original_pattern_type'], 'linguistic')
        
    def test_bidirectional_conversion(self):
        """Test pattern preservation through bidirectional conversion"""
        # Create initial neural pattern
        original_neural = NeuralPattern(
            activation_pattern=np.random.rand(64),
            temporal_sequence=[np.random.rand(64) for _ in range(3)],
            strength=0.8,
            frequency=0.5,
            connections=[(0, 1), (1, 2), (2, 0)],
            metadata={'pattern_type': 'test'}
        )
        
        # Convert neural -> linguistic -> neural
        linguistic = self.bridge.neural_to_linguistic(original_neural)
        converted_neural = self.bridge.linguistic_to_neural(linguistic)
        
        # Verify pattern preservation
        np.testing.assert_array_almost_equal(
            original_neural.activation_pattern,
            converted_neural.activation_pattern,
            decimal=5
        )
        self.assertAlmostEqual(original_neural.strength, converted_neural.strength, places=5)
        
    def test_pattern_description(self):
        """Test generation of pattern descriptions"""
        # Create test neural pattern with strong activation
        neural_pattern = NeuralPattern(
            activation_pattern=np.ones(64) * 0.9,  # Strong activation
            temporal_sequence=[np.random.rand(64) for _ in range(3)],
            strength=0.9,
            frequency=0.5,
            connections=[(i, i+1) for i in range(30)],  # Dense connections
            metadata={'pattern_type': 'test', 'confidence': 0.9}
        )
        
        # Convert to linguistic pattern
        linguistic_pattern = self.bridge.neural_to_linguistic(neural_pattern)
        
        # Verify description contains expected elements
        self.assertIn("Strong activation pattern", linguistic_pattern.text)
        self.assertIn("dense neural connections", linguistic_pattern.text)
        self.assertIn("high confidence", linguistic_pattern.text)
        
    def test_pattern_associations(self):
        """Test extraction of pattern associations"""
        # Create test neural pattern with specific characteristics
        neural_pattern = NeuralPattern(
            activation_pattern=np.ones(64) * 0.9,  # High intensity
            temporal_sequence=[np.random.rand(64) for _ in range(6)],  # Extended sequence
            strength=0.8,
            frequency=0.5,
            connections=[(i, i+1) for i in range(40)],  # Highly connected
            metadata={'pattern_type': 'test_type'}
        )
        
        # Convert to linguistic pattern
        linguistic_pattern = self.bridge.neural_to_linguistic(neural_pattern)
        
        # Verify expected associations
        self.assertIn("high_intensity", linguistic_pattern.associations)
        self.assertIn("extended_sequence", linguistic_pattern.associations)
        self.assertIn("highly_connected", linguistic_pattern.associations)
        self.assertIn("type_test_type", linguistic_pattern.associations)
        
    def test_error_handling(self):
        """Test error handling in pattern conversion"""
        # Test with invalid neural pattern
        with self.assertRaises(Exception):
            invalid_neural = NeuralPattern(
                activation_pattern=np.array([]),  # Empty pattern
                temporal_sequence=[],
                strength=0.5,
                frequency=0.5,
                connections=[],
                metadata={}
            )
            self.bridge.neural_to_linguistic(invalid_neural)
            
        # Test with invalid linguistic pattern
        with self.assertRaises(Exception):
            invalid_linguistic = LinguisticPattern(
                text="",  # Empty text
                semantic_vector=np.array([]),  # Empty vector
                context={},
                associations=[],
                confidence=0.5,
                metadata={}
            )
            self.bridge.linguistic_to_neural(invalid_linguistic)
            
    def test_pattern_memory(self):
        """Test pattern memory functionality"""
        # Create and convert a pattern
        neural_pattern = NeuralPattern(
            activation_pattern=np.random.rand(64),
            temporal_sequence=[np.random.rand(64) for _ in range(3)],
            strength=0.8,
            frequency=0.5,
            connections=[(0, 1), (1, 2), (2, 0)],
            metadata={'pattern_type': 'test'}
        )
        
        linguistic_pattern = self.bridge.neural_to_linguistic(neural_pattern)
        
        # Verify pattern is stored in memory
        pattern_id = self.bridge._generate_pattern_id(neural_pattern)
        stored_neural, stored_linguistic = self.bridge.pattern_memory[pattern_id]
        
        # Verify stored patterns match
        np.testing.assert_array_equal(stored_neural.activation_pattern, neural_pattern.activation_pattern)
        np.testing.assert_array_equal(stored_linguistic.semantic_vector, linguistic_pattern.semantic_vector)

if __name__ == '__main__':
    unittest.main() 
 